#include "persistence.h"

#include <chrono>
#include <sstream>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>

// ═════════════════════════════════════════════════════════════════════════════
//  Construction / destruction
// ═════════════════════════════════════════════════════════════════════════════

SqliteWriter::SqliteWriter()
    : m_db        (nullptr)
    , m_insertStmt(nullptr)
    , m_open      (false)
    , m_stop      (false)
    , m_pending   (0)
{}

SqliteWriter::~SqliteWriter() {
    if (m_open.load())
        close();
}

// ═════════════════════════════════════════════════════════════════════════════
//  Lifecycle
// ═════════════════════════════════════════════════════════════════════════════

void SqliteWriter::open(const std::string& path) {
    if (m_open.load())
        throw std::runtime_error("SqliteWriter: already open");

    m_path = path;

    if (sqlite3_open(path.c_str(), &m_db) != SQLITE_OK) {
        std::string err = sqlite3_errmsg(m_db);
        sqlite3_close(m_db);
        m_db = nullptr;
        throw std::runtime_error("sqlite3_open failed: " + err);
    }

    // ── Performance pragmas ───────────────────────────────────────────────────
    // WAL mode: readers don't block writers — essential for concurrent
    // inference + background writes on Pi
    exec("PRAGMA journal_mode=WAL");

    // NORMAL: safe on Pi SD/SSD, much faster than FULL
    exec("PRAGMA synchronous=NORMAL");

    // 10 MB page cache
    exec("PRAGMA cache_size=-10000");

    // Temp tables in RAM
    exec("PRAGMA temp_store=MEMORY");

    // 4 KB pages — better sequential write throughput
    exec("PRAGMA page_size=4096");

    // Auto-checkpoint WAL at 1000 pages
    exec("PRAGMA wal_autocheckpoint=1000");

    createSchema();
    prepareStatements();

    // Start background writer thread
    m_stop = false;
    m_open = true;
    m_writerThread = std::thread([this] { writerLoop(); });

    printf("SqliteWriter: opened %s\n", path.c_str());
}

void SqliteWriter::close() {
    if (!m_open.load()) return;

    // Signal writer thread and wait for it to drain the queue
    m_stop = true;
    m_queueCv.notify_all();
    if (m_writerThread.joinable())
        m_writerThread.join();

    // Checkpoint WAL so data is fully in the main DB file
    sqlite3_wal_checkpoint_v2(
        m_db, nullptr, SQLITE_CHECKPOINT_FULL, nullptr, nullptr);

    if (m_insertStmt) {
        sqlite3_finalize(m_insertStmt);
        m_insertStmt = nullptr;
    }

    sqlite3_close(m_db);
    m_db   = nullptr;
    m_open = false;

    printf("SqliteWriter: closed %s\n", m_path.c_str());
}

// ═════════════════════════════════════════════════════════════════════════════
//  Schema
// ═════════════════════════════════════════════════════════════════════════════

void SqliteWriter::createSchema() {
    exec(R"(
        CREATE TABLE IF NOT EXISTS detections (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id     INTEGER NOT NULL,
            timestamp_us INTEGER NOT NULL,
            track_id     INTEGER NOT NULL,
            class_id     INTEGER NOT NULL,
            label        TEXT    NOT NULL,
            x1           REAL    NOT NULL,
            y1           REAL    NOT NULL,
            x2           REAL    NOT NULL,
            y2           REAL    NOT NULL,
            confidence   REAL    NOT NULL,
            frame_w      INTEGER NOT NULL,
            frame_h      INTEGER NOT NULL,
            created_at   DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    )");

    exec("CREATE INDEX IF NOT EXISTS idx_timestamp "
         "ON detections(timestamp_us)");
    exec("CREATE INDEX IF NOT EXISTS idx_track "
         "ON detections(track_id, timestamp_us)");
    exec("CREATE INDEX IF NOT EXISTS idx_class "
         "ON detections(label, timestamp_us)");
    exec("CREATE INDEX IF NOT EXISTS idx_frame "
         "ON detections(frame_id)");

    exec(R"(
        CREATE TABLE IF NOT EXISTS metadata (
            key        TEXT PRIMARY KEY,
            value      TEXT NOT NULL,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    )");
}

void SqliteWriter::prepareStatements() {
    const char* sql =
        "INSERT INTO detections "
        "(frame_id,timestamp_us,track_id,class_id,label,"
        " x1,y1,x2,y2,confidence,frame_w,frame_h) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)";

    if (sqlite3_prepare_v2(m_db, sql, -1, &m_insertStmt, nullptr) != SQLITE_OK)
        throw std::runtime_error(
            std::string("prepare failed: ") + sqlite3_errmsg(m_db));
}

// ═════════════════════════════════════════════════════════════════════════════
//  Write API
// ═════════════════════════════════════════════════════════════════════════════

void SqliteWriter::write(const DetectionRecord& record) {
    writeBatch({ record });
}

void SqliteWriter::writeBatch(const std::vector<DetectionRecord>& records) {
    if (!m_open.load() || records.empty()) return;

    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        m_queue.push(records);
        m_pending += records.size();
    }
    m_queueCv.notify_one();
}

// ═════════════════════════════════════════════════════════════════════════════
//  Background writer thread
//  Drains the entire queue in one transaction per wake-up.
//  This is the key to high throughput on Pi SD card — individual per-row
//  commits would saturate random write IOPS at 30fps.
// ═════════════════════════════════════════════════════════════════════════════

void SqliteWriter::writerLoop() {
    while (true) {
        std::unique_lock<std::mutex> lock(m_queueMutex);

        m_queueCv.wait(lock, [this] {
            return !m_queue.empty() || m_stop.load();
        });

        // Drain queue into local vector while holding the lock
        std::vector<std::vector<DetectionRecord>> batch;
        while (!m_queue.empty()) {
            batch.push_back(std::move(m_queue.front()));
            m_queue.pop();
        }
        lock.unlock();

        if (!batch.empty()) {
            exec("BEGIN TRANSACTION");
            try {
                for (const auto& records : batch)
                    execBatch(records);
                exec("COMMIT");

                size_t written = 0;
                for (const auto& b : batch) written += b.size();
                m_pending -= written;

            } catch (const std::exception& e) {
                try { exec("ROLLBACK"); } catch (...) {}
                if (m_errorCb) m_errorCb(e.what());
                else fprintf(stderr, "SqliteWriter write error: %s\n", e.what());
            }
        }

        // Exit only after queue has been fully drained
        if (m_stop.load() && m_queue.empty()) break;
    }
}

void SqliteWriter::execBatch(const std::vector<DetectionRecord>& records) {
    for (const auto& r : records)
        bindAndStep(r);
}

void SqliteWriter::bindAndStep(const DetectionRecord& r) {
    sqlite3_bind_int64 (m_insertStmt,  1, static_cast<sqlite3_int64>(r.frameId));
    sqlite3_bind_int64 (m_insertStmt,  2, r.timestampUs);
    sqlite3_bind_int   (m_insertStmt,  3, r.trackId);
    sqlite3_bind_int   (m_insertStmt,  4, r.classId);
    sqlite3_bind_text  (m_insertStmt,  5, r.label.c_str(), -1, SQLITE_TRANSIENT);
    sqlite3_bind_double(m_insertStmt,  6, static_cast<double>(r.x1));
    sqlite3_bind_double(m_insertStmt,  7, static_cast<double>(r.y1));
    sqlite3_bind_double(m_insertStmt,  8, static_cast<double>(r.x2));
    sqlite3_bind_double(m_insertStmt,  9, static_cast<double>(r.y2));
    sqlite3_bind_double(m_insertStmt, 10, static_cast<double>(r.confidence));
    sqlite3_bind_int   (m_insertStmt, 11, r.frameWidth);
    sqlite3_bind_int   (m_insertStmt, 12, r.frameHeight);

    int rc = sqlite3_step(m_insertStmt);
    sqlite3_reset(m_insertStmt);
    sqlite3_clear_bindings(m_insertStmt);

    if (rc != SQLITE_DONE)
        throw std::runtime_error(
            std::string("insert step failed: ") + sqlite3_errmsg(m_db));
}

// ═════════════════════════════════════════════════════════════════════════════
//  Helper — rowFromStmt
// ═════════════════════════════════════════════════════════════════════════════

DetectionRow SqliteWriter::rowFromStmt(sqlite3_stmt* stmt) {
    DetectionRow row{};
    row.id          = sqlite3_column_int64 (stmt,  0);
    row.frameId     = static_cast<uint64_t>(sqlite3_column_int64(stmt, 1));
    row.timestampUs = sqlite3_column_int64 (stmt,  2);
    row.trackId     = sqlite3_column_int   (stmt,  3);
    row.classId     = sqlite3_column_int   (stmt,  4);

    const unsigned char* lbl = sqlite3_column_text(stmt, 5);
    row.label       = lbl ? reinterpret_cast<const char*>(lbl) : "";

    row.x1          = static_cast<float>(sqlite3_column_double(stmt,  6));
    row.y1          = static_cast<float>(sqlite3_column_double(stmt,  7));
    row.x2          = static_cast<float>(sqlite3_column_double(stmt,  8));
    row.y2          = static_cast<float>(sqlite3_column_double(stmt,  9));
    row.confidence  = static_cast<float>(sqlite3_column_double(stmt, 10));
    row.frameWidth  = sqlite3_column_int(stmt, 11);
    row.frameHeight = sqlite3_column_int(stmt, 12);

    const unsigned char* ca = sqlite3_column_text(stmt, 13);
    row.createdAt   = ca ? reinterpret_cast<const char*>(ca) : "";

    return row;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Helper — runSelect
//  Prepares, binds, steps, finalizes a SELECT statement.
//  All read queries use this to avoid duplicating error handling.
// ═════════════════════════════════════════════════════════════════════════════

std::vector<DetectionRow> SqliteWriter::runSelect(
    const char*                        sql,
    std::function<void(sqlite3_stmt*)> bindFn) const
{
    std::lock_guard<std::mutex> lock(m_readMutex);
    std::vector<DetectionRow>   rows;

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
        fprintf(stderr, "SqliteWriter prepare error: %s\n",
                sqlite3_errmsg(m_db));
        return rows;
    }

    if (bindFn) bindFn(stmt);

    while (sqlite3_step(stmt) == SQLITE_ROW)
        rows.push_back(rowFromStmt(stmt));

    sqlite3_finalize(stmt);
    return rows;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Read API
// ═════════════════════════════════════════════════════════════════════════════

static const char* SELECT_COLS =
    "SELECT id,frame_id,timestamp_us,track_id,class_id,label,"
    "x1,y1,x2,y2,confidence,frame_w,frame_h,created_at "
    "FROM detections ";

std::vector<DetectionRow> SqliteWriter::queryByTimeRange(
    int64_t fromUs, int64_t toUs, int limit) const
{
    std::string sql = std::string(SELECT_COLS) +
        "WHERE timestamp_us BETWEEN ? AND ? "
        "ORDER BY timestamp_us ASC LIMIT ?";

    return runSelect(sql.c_str(), [&](sqlite3_stmt* s) {
        sqlite3_bind_int64(s, 1, fromUs);
        sqlite3_bind_int64(s, 2, toUs);
        sqlite3_bind_int  (s, 3, limit);
    });
}

std::vector<DetectionRow> SqliteWriter::queryByTrackId(
    int trackId, int limit) const
{
    std::string sql = std::string(SELECT_COLS) +
        "WHERE track_id = ? "
        "ORDER BY timestamp_us ASC LIMIT ?";

    return runSelect(sql.c_str(), [&](sqlite3_stmt* s) {
        sqlite3_bind_int(s, 1, trackId);
        sqlite3_bind_int(s, 2, limit);
    });
}

std::vector<DetectionRow> SqliteWriter::queryByClass(
    const std::string& label, int64_t fromUs, int limit) const
{
    std::string sql = std::string(SELECT_COLS) +
        "WHERE label = ? AND timestamp_us >= ? "
        "ORDER BY timestamp_us DESC LIMIT ?";

    return runSelect(sql.c_str(), [&](sqlite3_stmt* s) {
        sqlite3_bind_text (s, 1, label.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_int64(s, 2, fromUs);
        sqlite3_bind_int  (s, 3, limit);
    });
}

std::vector<DetectionRow> SqliteWriter::queryRecent(int n) const {
    std::string sql = std::string(SELECT_COLS) +
        "ORDER BY timestamp_us DESC LIMIT ?";

    return runSelect(sql.c_str(), [&](sqlite3_stmt* s) {
        sqlite3_bind_int(s, 1, n);
    });
}

std::vector<std::pair<std::string, int64_t>>
SqliteWriter::queryClassCounts(int64_t fromUs, int64_t toUs) const
{
    std::lock_guard<std::mutex> lock(m_readMutex);
    std::vector<std::pair<std::string, int64_t>> counts;

    const char* sql =
        "SELECT label, COUNT(*) AS cnt "
        "FROM detections "
        "WHERE timestamp_us BETWEEN ? AND ? "
        "GROUP BY label ORDER BY cnt DESC";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return counts;

    sqlite3_bind_int64(stmt, 1, fromUs);
    sqlite3_bind_int64(stmt, 2, toUs);

    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const unsigned char* lbl = sqlite3_column_text(stmt, 0);
        counts.emplace_back(
            lbl ? reinterpret_cast<const char*>(lbl) : "",
            sqlite3_column_int64(stmt, 1)
        );
    }
    sqlite3_finalize(stmt);
    return counts;
}

std::vector<DetectionRow> SqliteWriter::queryTrackHistory(int trackId) const {
    return queryByTrackId(trackId, 10000);
}

DetectionStats SqliteWriter::getStats() const {
    std::lock_guard<std::mutex> lock(m_readMutex);
    DetectionStats stats{};

    const char* sql =
        "SELECT "
        "  COUNT(*)                                        AS total_dets,  "
        "  COUNT(DISTINCT frame_id)                        AS total_frames, "
        "  COUNT(DISTINCT track_id)                        AS unique_tracks, "
        "  COUNT(DISTINCT class_id)                        AS unique_classes,"
        "  COALESCE(MIN(timestamp_us), 0)                  AS oldest_ts,    "
        "  COALESCE(MAX(timestamp_us), 0)                  AS newest_ts,    "
        "  COALESCE(AVG(confidence),   0.0)                AS avg_conf,     "
        "  COALESCE(CAST(COUNT(*) AS REAL) /                               "
        "    NULLIF(COUNT(DISTINCT frame_id), 0), 0.0)     AS avg_per_frame "
        "FROM detections";

    sqlite3_stmt* stmt = nullptr;
    if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return stats;

    if (sqlite3_step(stmt) == SQLITE_ROW) {
        stats.totalDetections       = sqlite3_column_int64 (stmt, 0);
        stats.totalFrames           = sqlite3_column_int64 (stmt, 1);
        stats.uniqueTracks          = sqlite3_column_int64 (stmt, 2);
        stats.uniqueClasses         = sqlite3_column_int64 (stmt, 3);
        stats.oldestTimestampUs     = sqlite3_column_int64 (stmt, 4);
        stats.newestTimestampUs     = sqlite3_column_int64 (stmt, 5);
        stats.avgConfidence         = sqlite3_column_double(stmt, 6);
        stats.avgDetectionsPerFrame = sqlite3_column_double(stmt, 7);
    }
    sqlite3_finalize(stmt);
    return stats;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Maintenance
// ═════════════════════════════════════════════════════════════════════════════

int64_t SqliteWriter::pruneOlderThan(int64_t ageUs) {
    flush();   // drain pending writes first

    std::lock_guard<std::mutex> lock(m_readMutex);

    int64_t cutoff = nowUs() - ageUs;

    sqlite3_stmt* stmt = nullptr;
    const char* sql = "DELETE FROM detections WHERE timestamp_us < ?";
    if (sqlite3_prepare_v2(m_db, sql, -1, &stmt, nullptr) != SQLITE_OK)
        return 0;

    sqlite3_bind_int64(stmt, 1, cutoff);
    sqlite3_step(stmt);
    sqlite3_finalize(stmt);

    return static_cast<int64_t>(sqlite3_changes(m_db));
}

int64_t SqliteWriter::pruneOlderThanDays(int days) {
    int64_t ageUs = static_cast<int64_t>(days) * 24 * 3600 * 1000000LL;
    return pruneOlderThan(ageUs);
}

int64_t SqliteWriter::fileSizeBytes() const {
    struct stat st{};
    if (::stat(m_path.c_str(), &st) == 0)
        return static_cast<int64_t>(st.st_size);
    return -1;
}

void SqliteWriter::vacuum() {
    flush();
    exec("VACUUM");
}

void SqliteWriter::flush() {
    // Spin-wait until all queued records have been written
    while (m_pending.load() > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

// ═════════════════════════════════════════════════════════════════════════════
//  Internal helpers
// ═════════════════════════════════════════════════════════════════════════════

// Execute a DDL / pragma statement.
// NOT const — sqlite3_exec takes non-const sqlite3*
void SqliteWriter::exec(const char* sql) {
    char* errMsg = nullptr;
    int   rc     = sqlite3_exec(m_db, sql, nullptr, nullptr, &errMsg);
    if (rc != SQLITE_OK) {
        std::string err = errMsg ? errMsg : "unknown";
        sqlite3_free(errMsg);
        throw std::runtime_error(
            std::string("SQL error [") + sql + "]: " + err);
    }
}

int64_t SqliteWriter::nowUs() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::system_clock::now().time_since_epoch()
    ).count();
}
