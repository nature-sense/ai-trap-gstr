#include "sync_manager.h"

#include <cstdio>
#include <cstring>
#include <ctime>
#include <random>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <sys/statvfs.h>
#include <sys/stat.h>
#include <unistd.h>

// ═════════════════════════════════════════════════════════════════════════════
//  Lifecycle
// ═════════════════════════════════════════════════════════════════════════════

void SyncManager::init(sqlite3* db, const std::string& cropsDir)
{
    std::lock_guard<std::mutex> lk(m_mutex);
    m_db       = db;
    m_cropsDir = cropsDir;
    createSchema();
}

void SyncManager::createSchema()
{
    // crops table — tracks every JPEG written by CropSaver
    execSql(R"(
        CREATE TABLE IF NOT EXISTS crops (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            file            TEXT    NOT NULL UNIQUE,   -- session-relative path, e.g. "20260314_153042/insect_42.jpg"
            path            TEXT    NOT NULL,           -- full path on disk
            track_id        INTEGER NOT NULL,
            class_id        INTEGER NOT NULL DEFAULT 0,
            label           TEXT    NOT NULL DEFAULT '',
            confidence      REAL    NOT NULL DEFAULT 0,
            timestamp_us    INTEGER NOT NULL DEFAULT 0,
            bytes           INTEGER NOT NULL DEFAULT 0,
            synced          INTEGER NOT NULL DEFAULT 0, -- 0=new 1=acked 2=deleted
            capture_session TEXT    NOT NULL DEFAULT '',
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    )");

    execSql("CREATE INDEX IF NOT EXISTS idx_crops_synced "
            "ON crops(synced, created_at)");
    execSql("CREATE INDEX IF NOT EXISTS idx_crops_track "
            "ON crops(track_id)");

    // Migration: if crops table existed without some columns, add them.
    // SQLite ALTER TABLE ADD COLUMN is safe to run on existing tables.
    execSql("CREATE INDEX IF NOT EXISTS idx_crops_session "
            "ON crops(capture_session)");

    const char* migrations[] = {
        "ALTER TABLE crops ADD COLUMN class_id        INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE crops ADD COLUMN label           TEXT    NOT NULL DEFAULT ''",
        "ALTER TABLE crops ADD COLUMN confidence      REAL    NOT NULL DEFAULT 0",
        "ALTER TABLE crops ADD COLUMN timestamp_us    INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE crops ADD COLUMN bytes           INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE crops ADD COLUMN synced          INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE crops ADD COLUMN capture_session TEXT    NOT NULL DEFAULT ''",
        nullptr
    };
    for (int i = 0; migrations[i]; ++i) {
        // Ignore errors — column likely already exists
        sqlite3_exec(m_db, migrations[i], nullptr, nullptr, nullptr);
    }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Session tracking
// ═════════════════════════════════════════════════════════════════════════════

void SyncManager::setCurrentSession(const std::string& sessionId)
{
    std::lock_guard<std::mutex> lk(m_mutex);
    m_currentSessionId = sessionId;
    printf("SyncManager: current capture session = \"%s\"\n", sessionId.c_str());
}

// ═════════════════════════════════════════════════════════════════════════════
//  Register crop (called by CropSaver on each successful write)
// ═════════════════════════════════════════════════════════════════════════════

void SyncManager::registerCrop(const std::string& file,
                                int trackId, int classId,
                                const std::string& label,
                                float confidence,
                                int64_t timestampUs,
                                int64_t bytes)
{
    std::lock_guard<std::mutex> lk(m_mutex);
    if (!m_db) return;

    // Build session-relative file key: "<sessionId>/<basename>"
    // Falls back to bare filename if no session is set.
    std::string relFile = m_currentSessionId.empty()
                          ? file
                          : (m_currentSessionId + "/" + file);
    std::string path = m_cropsDir + "/" + relFile;

    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "INSERT OR IGNORE INTO crops "
        "(file, path, track_id, class_id, label, confidence, timestamp_us, bytes, synced, capture_session) "
        "VALUES (?,?,?,?,?,?,?,?,0,?)";

    if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        fprintf(stderr, "SyncManager: prepare registerCrop failed: %s\n",
                sqlite3_errmsg(m_db));
        return;
    }
    sqlite3_bind_text  (stmt, 1, relFile.c_str(),             -1, SQLITE_TRANSIENT);
    sqlite3_bind_text  (stmt, 2, path.c_str(),                -1, SQLITE_TRANSIENT);
    sqlite3_bind_int   (stmt, 3, trackId);
    sqlite3_bind_int   (stmt, 4, classId);
    sqlite3_bind_text  (stmt, 5, label.c_str(),               -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(stmt, 6, static_cast<double>(confidence));
    sqlite3_bind_int64 (stmt, 7, timestampUs);
    sqlite3_bind_int64 (stmt, 8, bytes);
    sqlite3_bind_text  (stmt, 9, m_currentSessionId.c_str(),  -1, SQLITE_TRANSIENT);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Session management
// ═════════════════════════════════════════════════════════════════════════════

SyncSession SyncManager::openSession()
{
    std::lock_guard<std::mutex> lk(m_mutex);

    SyncSession sess;
    sess.id     = makeSessionId();
    sess.crops  = queryPending();
    sess.pending = static_cast<int64_t>(sess.crops.size());

    // Prune sessions older than 1 hour (stale connections)
    int64_t now = nowSec();
    m_sessions.erase(
        std::remove_if(m_sessions.begin(), m_sessions.end(),
            [now](const SessionEntry& e) { return now - e.openedAt > 3600; }),
        m_sessions.end());

    m_sessions.push_back({sess.id, now});

    fprintf(stdout, "SyncManager: session %s opened — %lld pending crops\n",
            sess.id.c_str(), static_cast<long long>(sess.pending));
    return sess;
}

bool SyncManager::getSession(const std::string& sessionId,
                              SyncSession& out) const
{
    std::lock_guard<std::mutex> lk(m_mutex);
    if (!sessionExists(sessionId)) return false;

    out.id     = sessionId;
    out.crops  = queryPending();
    out.pending = static_cast<int64_t>(out.crops.size());
    return true;
}

bool SyncManager::sessionExists(const std::string& id) const
{
    for (const auto& s : m_sessions)
        if (s.id == id) return true;
    return false;
}

// ── Ack ───────────────────────────────────────────────────────────────────────

int SyncManager::ackFiles(const std::string& sessionId,
                           const std::vector<std::string>& files)
{
    std::lock_guard<std::mutex> lk(m_mutex);
    if (!m_db || !sessionExists(sessionId) || files.empty()) return 0;

    int updated = 0;
    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "UPDATE crops SET synced=1 WHERE file=? AND synced=0";

    if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return 0;

    execSql("BEGIN");
    for (const auto& f : files) {
        sqlite3_reset(stmt);
        sqlite3_bind_text(stmt, 1, f.c_str(), -1, SQLITE_TRANSIENT);
        if (sqlite3_step(stmt) == SQLITE_DONE)
            updated += sqlite3_changes(m_db);
    }
    execSql("COMMIT");
    sqlite3_finalize(stmt);

    fprintf(stdout, "SyncManager: acked %d / %zu files\n",
            updated, files.size());
    return updated;
}

// ── Close / delete ────────────────────────────────────────────────────────────

SyncDeleteResult SyncManager::closeSession(const std::string& sessionId)
{
    std::lock_guard<std::mutex> lk(m_mutex);

    SyncDeleteResult result;
    if (!m_db || !sessionExists(sessionId)) return result;

    // Query all acked (synced=1) crops
    sqlite3_stmt* stmt = nullptr;
    const char* sql =
        "SELECT id, file, path, bytes FROM crops WHERE synced=1";
    if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return result;

    std::vector<std::pair<int64_t, std::string>> toDelete; // id, path
    std::vector<int64_t>                          deletedIds;

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        int64_t     id    = sqlite3_column_int64(stmt, 0);
        std::string path  = reinterpret_cast<const char*>(
                                sqlite3_column_text(stmt, 2));
        int64_t     bytes = sqlite3_column_int64(stmt, 3);
        toDelete.push_back({id, path});
        result.bytesFreed += bytes;
    }
    sqlite3_finalize(stmt);

    // Delete files from disk
    for (auto& [id, path] : toDelete) {
        if (::unlink(path.c_str()) == 0) {
            result.filesDeleted++;
            deletedIds.push_back(id);
        } else {
            // File already gone — still mark as deleted in DB
            result.filesNotFound++;
            deletedIds.push_back(id);
        }
    }

    // Mark synced=2 in DB
    if (!deletedIds.empty()) {
        execSql("BEGIN");
        sqlite3_stmt* upd = nullptr;
        sqlite3_prepare_v2(m_db,
            "UPDATE crops SET synced=2 WHERE id=?", -1, &upd, nullptr);
        for (int64_t id : deletedIds) {
            sqlite3_reset(upd);
            sqlite3_bind_int64(upd, 1, id);
            sqlite3_step(upd);
        }
        sqlite3_finalize(upd);
        execSql("COMMIT");
    }

    // Remove session
    m_sessions.erase(
        std::remove_if(m_sessions.begin(), m_sessions.end(),
            [&](const SessionEntry& e) { return e.id == sessionId; }),
        m_sessions.end());

    fprintf(stdout,
            "SyncManager: session %s closed — "
            "%lld files deleted, %lld bytes freed, %lld not found\n",
            sessionId.c_str(),
            static_cast<long long>(result.filesDeleted),
            static_cast<long long>(result.bytesFreed),
            static_cast<long long>(result.filesNotFound));

    return result;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Storage guard
// ═════════════════════════════════════════════════════════════════════════════

int64_t SyncManager::enforceStorageLimit(int64_t minFreeBytes)
{
    std::lock_guard<std::mutex> lk(m_mutex);
    if (!m_db) return 0;

    int64_t freed = 0;
    int64_t free  = freeSpaceBytes(m_cropsDir);
    if (free >= minFreeBytes) return 0;

    fprintf(stderr,
            "SyncManager: storage low — %lld MB free, limit %lld MB\n",
            static_cast<long long>(free / 1024 / 1024),
            static_cast<long long>(minFreeBytes / 1024 / 1024));

    // Phase 1: delete oldest synced=1 (already transferred)
    auto purge = [&](int syncedVal, bool warn) {
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(m_db,
            "SELECT id, path, bytes FROM crops WHERE synced=? "
            "ORDER BY created_at ASC",
            -1, &stmt, nullptr);
        sqlite3_bind_int(stmt, 1, syncedVal);

        while (sqlite3_step(stmt) == SQLITE_ROW
               && freeSpaceBytes(m_cropsDir) < minFreeBytes)
        {
            int64_t     id    = sqlite3_column_int64(stmt, 0);
            std::string path  = reinterpret_cast<const char*>(
                                    sqlite3_column_text(stmt, 1));
            int64_t     bytes = sqlite3_column_int64(stmt, 2);

            if (warn)
                fprintf(stderr,
                        "SyncManager: WARNING — deleting unsynced crop: %s\n",
                        path.c_str());

            if (::unlink(path.c_str()) == 0) {
                freed += bytes;
                sqlite3_stmt* upd = nullptr;
                sqlite3_prepare_v2(m_db,
                    "UPDATE crops SET synced=2 WHERE id=?", -1, &upd, nullptr);
                sqlite3_bind_int64(upd, 1, id);
                sqlite3_step(upd);
                sqlite3_finalize(upd);
            }
        }
        sqlite3_finalize(stmt);
    };

    purge(1, false);  // acked first
    if (freeSpaceBytes(m_cropsDir) < minFreeBytes)
        purge(0, true);  // unsynced — last resort, with warning

    return freed;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Stats
// ═════════════════════════════════════════════════════════════════════════════

int64_t SyncManager::countPending() const
{
    std::lock_guard<std::mutex> lk(m_mutex);
    if (!m_db) return 0;
    sqlite3_stmt* s = nullptr;
    sqlite3_prepare_v2(m_db,
        "SELECT COUNT(*) FROM crops WHERE synced=0", -1, &s, nullptr);
    int64_t n = (sqlite3_step(s) == SQLITE_ROW) ? sqlite3_column_int64(s, 0) : 0;
    sqlite3_finalize(s);
    return n;
}

int64_t SyncManager::countAcked() const
{
    std::lock_guard<std::mutex> lk(m_mutex);
    if (!m_db) return 0;
    sqlite3_stmt* s = nullptr;
    sqlite3_prepare_v2(m_db,
        "SELECT COUNT(*) FROM crops WHERE synced=1", -1, &s, nullptr);
    int64_t n = (sqlite3_step(s) == SQLITE_ROW) ? sqlite3_column_int64(s, 0) : 0;
    sqlite3_finalize(s);
    return n;
}

int64_t SyncManager::countDeleted() const
{
    std::lock_guard<std::mutex> lk(m_mutex);
    if (!m_db) return 0;
    sqlite3_stmt* s = nullptr;
    sqlite3_prepare_v2(m_db,
        "SELECT COUNT(*) FROM crops WHERE synced=2", -1, &s, nullptr);
    int64_t n = (sqlite3_step(s) == SQLITE_ROW) ? sqlite3_column_int64(s, 0) : 0;
    sqlite3_finalize(s);
    return n;
}

int64_t SyncManager::totalBytes() const
{
    std::lock_guard<std::mutex> lk(m_mutex);
    if (!m_db) return 0;
    sqlite3_stmt* s = nullptr;
    sqlite3_prepare_v2(m_db,
        "SELECT COALESCE(SUM(bytes),0) FROM crops WHERE synced IN (0,1)",
        -1, &s, nullptr);
    int64_t n = (sqlite3_step(s) == SQLITE_ROW) ? sqlite3_column_int64(s, 0) : 0;
    sqlite3_finalize(s);
    return n;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Private helpers
// ═════════════════════════════════════════════════════════════════════════════

std::vector<CropRecord> SyncManager::queryPending() const
{
    std::vector<CropRecord> out;
    if (!m_db) return out;

    sqlite3_stmt* stmt = nullptr;
    sqlite3_prepare_v2(m_db,
        "SELECT id, file, path, track_id, class_id, label, "
        "       confidence, timestamp_us, bytes, synced, capture_session, created_at "
        "FROM crops WHERE synced=0 ORDER BY created_at ASC",
        -1, &stmt, nullptr);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        CropRecord r;
        r.id             = sqlite3_column_int64(stmt, 0);
        r.file           = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
        r.path           = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
        r.trackId        = sqlite3_column_int  (stmt, 3);
        r.classId        = sqlite3_column_int  (stmt, 4);
        r.label          = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 5));
        r.confidence     = static_cast<float>(sqlite3_column_double(stmt, 6));
        r.timestampUs    = sqlite3_column_int64(stmt, 7);
        r.bytes          = sqlite3_column_int64(stmt, 8);
        r.synced         = sqlite3_column_int  (stmt, 9);
        const auto* cs   = sqlite3_column_text (stmt, 10);
        r.captureSession = cs ? reinterpret_cast<const char*>(cs) : "";
        r.createdAt      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 11));
        out.push_back(r);
    }
    sqlite3_finalize(stmt);
    return out;
}

void SyncManager::execSql(const char* sql)
{
    char* err = nullptr;
    if (sqlite3_exec(m_db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        fprintf(stderr, "SyncManager SQL error: %s — %s\n", sql, err);
        sqlite3_free(err);
    }
}

std::string SyncManager::makeSessionId()
{
    std::random_device rd;
    std::mt19937_64 gen(rd());
    std::uniform_int_distribution<uint64_t> dist;
    std::ostringstream oss;
    oss << std::hex << std::setfill('0')
        << std::setw(16) << dist(gen)
        << std::setw(16) << dist(gen);
    return oss.str();
}

int64_t SyncManager::freeSpaceBytes(const std::string& dir)
{
    struct statvfs sv;
    if (::statvfs(dir.c_str(), &sv) != 0) return INT64_MAX;
    return static_cast<int64_t>(sv.f_bavail) *
           static_cast<int64_t>(sv.f_frsize);
}

int64_t SyncManager::nowSec()
{
    return static_cast<int64_t>(std::time(nullptr));
}
