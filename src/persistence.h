#pragma once

#include <sqlite3.h>
#include <string>
#include <vector>
#include <queue>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <stdexcept>
#include <cstdint>

// ─── Detection record — one row per detection ─────────────────────────────────
struct DetectionRecord {
    uint64_t    frameId;
    int64_t     timestampUs;     // microseconds since epoch
    int         trackId;
    int         classId;
    std::string label;
    float       x1, y1, x2, y2; // original image pixels
    float       confidence;
    int         frameWidth;
    int         frameHeight;
};

// ─── Row returned from read queries ──────────────────────────────────────────
struct DetectionRow {
    int64_t     id;
    uint64_t    frameId;
    int64_t     timestampUs;
    int         trackId;
    int         classId;
    std::string label;
    float       x1, y1, x2, y2;
    float       confidence;
    int         frameWidth;
    int         frameHeight;
    std::string createdAt;
};

// ─── Summary statistics ───────────────────────────────────────────────────────
struct DetectionStats {
    int64_t totalDetections       = 0;
    int64_t totalFrames           = 0;
    int64_t uniqueTracks          = 0;
    int64_t uniqueClasses         = 0;
    int64_t oldestTimestampUs     = 0;
    int64_t newestTimestampUs     = 0;
    double  avgConfidence         = 0.0;
    double  avgDetectionsPerFrame = 0.0;
};

// ─────────────────────────────────────────────────────────────────────────────

class SqliteWriter {
public:
    // ── Construction / destruction ────────────────────────────────────────────
    SqliteWriter();
    ~SqliteWriter();

    // Non-copyable, non-movable
    SqliteWriter(const SqliteWriter&)            = delete;
    SqliteWriter& operator=(const SqliteWriter&) = delete;
    SqliteWriter(SqliteWriter&&)                 = delete;
    SqliteWriter& operator=(SqliteWriter&&)      = delete;

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    // Open database, create schema, start background writer thread.
    // Throws std::runtime_error on failure.
    void open(const std::string& path = "detections.db");

    // Flush pending writes and close the database cleanly.
    void close();

    bool isOpen() const { return m_open.load(); }

    // ── Write API  (non-blocking — queues for background thread) ─────────────

    void write(const DetectionRecord& record);
    void writeBatch(const std::vector<DetectionRecord>& records);

    // ── Synchronous read API ──────────────────────────────────────────────────

    std::vector<DetectionRow> queryByTimeRange(
        int64_t fromUs, int64_t toUs,
        int     limit = 1000) const;

    std::vector<DetectionRow> queryByTrackId(
        int trackId,
        int limit = 500) const;

    std::vector<DetectionRow> queryByClass(
        const std::string& label,
        int64_t            fromUs = 0,
        int                limit  = 1000) const;

    std::vector<DetectionRow> queryRecent(int n = 100) const;

    std::vector<std::pair<std::string, int64_t>> queryClassCounts(
        int64_t fromUs, int64_t toUs) const;

    std::vector<DetectionRow> queryTrackHistory(int trackId) const;

    DetectionStats getStats() const;

    // ── Maintenance ───────────────────────────────────────────────────────────

    // Delete rows older than ageUs microseconds. Returns rows deleted.
    int64_t pruneOlderThan(int64_t ageUs);

    // Delete rows older than N days. Returns rows deleted.
    int64_t pruneOlderThanDays(int days);

    // Database file size in bytes (-1 on error).
    int64_t fileSizeBytes() const;

    // VACUUM — reclaim space after pruning. Blocks until complete.
    void vacuum();

    // Block until the write queue is empty.
    void flush();

    // Number of records currently queued for writing.
    size_t pendingWrites() const { return m_pending.load(); }

    // Called on background write errors (optional).
    using ErrorCallback = std::function<void(const std::string& msg)>;
    void setErrorCallback(ErrorCallback cb) { m_errorCb = std::move(cb); }

private:
    // ── Database ──────────────────────────────────────────────────────────────
    sqlite3*      m_db;
    std::string   m_path;
    sqlite3_stmt* m_insertStmt;

    // ── Async write queue ─────────────────────────────────────────────────────
    std::queue<std::vector<DetectionRecord>> m_queue;
    mutable std::mutex                       m_queueMutex;
    std::condition_variable                  m_queueCv;
    std::thread                              m_writerThread;

    std::atomic<bool>   m_open;
    std::atomic<bool>   m_stop;
    std::atomic<size_t> m_pending;

    // Read queries share a separate mutex so they don't block each other
    mutable std::mutex m_readMutex;

    ErrorCallback m_errorCb;

    // ── Internal helpers ──────────────────────────────────────────────────────
    void createSchema();
    void prepareStatements();
    void writerLoop();
    void execBatch(const std::vector<DetectionRecord>& records);
    void bindAndStep(const DetectionRecord& r);

    // Execute a DDL/pragma statement. Throws on error.
    void exec(const char* sql);

    // Map a sqlite3_stmt row → DetectionRow.
    static DetectionRow rowFromStmt(sqlite3_stmt* stmt);

    // Helper: prepare, execute, and finalize a SELECT. Returns rows.
    std::vector<DetectionRow> runSelect(
        const char*                                  sql,
        std::function<void(sqlite3_stmt*)>           bindFn) const;

    // Current time as microseconds since epoch.
    static int64_t nowUs();
};
