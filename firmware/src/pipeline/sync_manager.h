#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  sync_manager.h  —  Session-based crop file sync
//
//  Lifecycle per field visit:
//
//    1. POST /api/sync/session
//       → creates a session, returns { sessionId, pending }
//       → pending = count of crops with synced = 0
//
//    2. GET  /api/sync/session/{id}
//       → returns manifest: [{ file, bytes, trackId, timestampUs, conf }]
//
//    3. GET  /api/crops/{file}    (existing endpoint — unchanged)
//       → download each JPEG
//
//    4. POST /api/sync/ack
//       → body: { sessionId, files: ["insect_42.jpg", ...] }
//       → marks each file synced=1 in crops table
//       → safe to call per-file or in batches
//
//    5. DELETE /api/sync/session/{id}
//       → deletes all synced=1 files from disk
//       → marks them synced=2 in DB (audit trail preserved)
//       → returns { deleted, bytes_freed }
//
//  If the connection drops mid-session the session is abandoned (no side
//  effects). A new session on the next visit picks up all synced=0 files.
//
//  Storage guard (call periodically from main):
//    syncMgr.enforceStorageLimit(minFreeBytes)
//    → deletes oldest synced=1 crops first, then synced=0 if still tight
//
//  Requires: the crops table in the SQLite DB (created by SyncManager::init)
// ─────────────────────────────────────────────────────────────────────────────

#include <sqlite3.h>

#include <string>
#include <vector>
#include <mutex>
#include <cstdint>

// ── Crop record as stored in DB ───────────────────────────────────────────────

struct CropRecord {
    int64_t     id             = 0;
    std::string file;              // session-relative path, e.g. "20260314_153042/insect_42.jpg"
    std::string path;              // full path on disk
    int         trackId        = 0;
    int         classId        = 0;
    std::string label;
    float       confidence     = 0.f;
    int64_t     timestampUs    = 0;
    int64_t     bytes          = 0;
    int         synced         = 0; // 0=new  1=acked  2=deleted
    std::string captureSession;    // e.g. "20260314_153042"
    std::string createdAt;
};

// ── Session returned to client ────────────────────────────────────────────────

struct SyncSession {
    std::string          id;       // random hex token
    int64_t              pending;  // crops with synced=0 at session open
    std::vector<CropRecord> crops; // full manifest
};

// ── Result of DELETE /api/sync/session ───────────────────────────────────────

struct SyncDeleteResult {
    int64_t filesDeleted = 0;
    int64_t bytesFreed   = 0;
    int64_t filesNotFound = 0;  // acked but file already gone (not an error)
};

// ─────────────────────────────────────────────────────────────────────────────

class SyncManager {
public:
    SyncManager() = default;
    ~SyncManager() = default;

    SyncManager(const SyncManager&)            = delete;
    SyncManager& operator=(const SyncManager&) = delete;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    // Open DB (must be the same file as SqliteWriter).
    // Creates the crops table if absent. Migrates existing schema.
    void init(sqlite3* db, const std::string& cropsDir);

    // ── Session tracking ──────────────────────────────────────────────────────

    // Set the active capture session.  Subsequent registerCrop() calls will
    // store crops under "<sessionId>/<file>" so each session's files are
    // namespaced in the DB and on disk.  Call this when a new capture session
    // starts (and again when it ends, with an empty string).
    void setCurrentSession(const std::string& sessionId);

    // ── Called by CropSaver after each successful JPEG write ──────────────────

    // Register a new crop file (basename only, e.g. "insect_42.jpg").
    // The active session set via setCurrentSession() is prepended to form the
    // unique file key stored in the DB.  Thread-safe.
    void registerCrop(const std::string& file,
                      int trackId, int classId,
                      const std::string& label,
                      float confidence,
                      int64_t timestampUs,
                      int64_t bytes);

    // ── Sync session API (called by HttpServer) ────────────────────────────────

    // Create a new session. Returns session with full manifest.
    SyncSession openSession();

    // Return session manifest by ID (re-fetches from DB, session is stateless).
    // Returns empty optional if session ID is unknown / expired.
    bool getSession(const std::string& sessionId, SyncSession& out) const;

    // Mark files as acked (synced=1). Returns number of rows updated.
    int ackFiles(const std::string& sessionId,
                 const std::vector<std::string>& files);

    // Delete all acked files, mark synced=2. Closes session.
    SyncDeleteResult closeSession(const std::string& sessionId);

    // ── Storage guard ─────────────────────────────────────────────────────────

    // If free space on the crops filesystem < minFreeBytes:
    //   1. delete oldest synced=1 crops
    //   2. if still tight, delete oldest synced=0 crops (logs a warning)
    // Returns bytes freed.
    int64_t enforceStorageLimit(int64_t minFreeBytes);

    // ── Stats ─────────────────────────────────────────────────────────────────

    int64_t countPending()  const;  // synced=0
    int64_t countAcked()    const;  // synced=1
    int64_t countDeleted()  const;  // synced=2
    int64_t totalBytes()    const;  // sum of bytes for synced=0 and synced=1

private:
    sqlite3*    m_db      = nullptr;
    std::string m_cropsDir;
    std::string m_currentSessionId;

    mutable std::mutex m_mutex;

    // Active sessions: id → set of pending file basenames at open time
    // Sessions are lightweight — just a token + timestamp, manifest is
    // re-queried from DB on each GET.
    struct SessionEntry {
        std::string id;
        int64_t     openedAt;  // unix seconds
    };
    std::vector<SessionEntry> m_sessions;

    // ── Helpers ───────────────────────────────────────────────────────────────
    void createSchema();
    bool sessionExists(const std::string& id) const; // caller holds m_mutex

    std::vector<CropRecord> queryPending() const;
    std::vector<CropRecord> queryByFiles(
        const std::vector<std::string>& files) const;

    int64_t deleteFile(const CropRecord& rec);

    static std::string makeSessionId();
    static int64_t     freeSpaceBytes(const std::string& dir);
    static int64_t     nowSec();

    // Execute SQL with no result set. Logs errors, does not throw.
    void execSql(const char* sql);
};
