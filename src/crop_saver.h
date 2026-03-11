#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  CropSaverConfig
// ─────────────────────────────────────────────────────────────────────────────
struct CropSaverConfig {
    // Directory to write crops into.  Created at open() if it does not exist.
    std::string outputDir   = "crops";

    // JPEG quality 1–100
    int jpegQuality         = 90;

    // Minimum confidence required before a crop is even considered.
    // Tracks below this threshold are silently ignored even if confirmed.
    float minConfidence     = 0.50f;

    // Maximum depth of the async write queue.  If the worker falls behind
    // this many jobs are held in memory; additional submits are dropped.
    int maxQueueDepth       = 16;
};

// ─────────────────────────────────────────────────────────────────────────────
//  CropSaver
//
//  Saves one JPEG crop per track ID — the highest-confidence detection seen
//  for that track across all frames.  When a new detection for an existing
//  track arrives with higher confidence than previously recorded, the file
//  is overwritten.
//
//  Output filename:  <outputDir>/<className>_<trackId>.jpg
//
//  Thread model
//  ────────────
//  submit() is called on the inference thread.  It checks the best-so-far
//  confidence map (under a small spinlock), and if the new detection is
//  better, enqueues a CropJob.  A single background worker thread dequeues
//  jobs, extracts the NV12 crop, converts to RGB, encodes JPEG, and writes.
//
//  The NV12 data is copied into the job at submit() time so the caller's
//  CaptureFrame can be freed immediately.
//
//  Usage
//  ─────
//    CropSaverConfig cfg;
//    cfg.outputDir  = "crops";
//    cfg.minConfidence = 0.55f;
//
//    CropSaver saver;
//    saver.open(cfg);
//
//    // inside FrameCallback, after tracking:
//    for (const auto& t : tracked)
//        if (t.confirmed)
//            saver.submit(t, frame, className(t.classId));
//
//    saver.flush();   // drain queue before shutdown
//    saver.close();
// ─────────────────────────────────────────────────────────────────────────────
class CropSaver {
public:
    CropSaver()  = default;
    ~CropSaver();

    CropSaver(const CropSaver&)            = delete;
    CropSaver& operator=(const CropSaver&) = delete;

    // Open output directory and start worker thread.
    // Throws std::runtime_error if the directory cannot be created.
    void open(const CropSaverConfig& cfg);

    // Submit a detection for possible saving.
    // nv12      : compact (de-strided) NV12 buffer, frameWidth * frameHeight * 3/2 bytes
    // frameW/H  : original capture dimensions
    // trackId   : unique track ID from ByteTracker
    // classId   : COCO class index
    // className : human-readable class name string
    // confidence: detection score [0, 1]
    // x1,y1,x2,y2 : bounding box in original frame coordinates
    //
    // Returns true if a crop job was enqueued (confidence was a new best).
    bool submit(const std::vector<uint8_t>& nv12,
                int frameW, int frameH,
                int trackId, int classId,
                const std::string& className,
                float confidence,
                float x1, float y1, float x2, float y2);

    // Block until the write queue is empty.
    void flush();

    // flush() then stop the worker thread.
    void close();

    // ── Stats ─────────────────────────────────────────────────────────────────
    uint64_t cropsSaved()   const { return m_cropsSaved.load();   }
    uint64_t cropsDropped() const { return m_cropsDropped.load(); }
    void     printStats()   const;

private:
    // ── Per-track best-confidence record ──────────────────────────────────────
    struct TrackRecord {
        float       bestConfidence = -1.f;
        std::string lastPath;
    };

    // ── Async job ─────────────────────────────────────────────────────────────
    struct CropJob {
        std::vector<uint8_t> nv12;   // full-frame NV12, compact
        int   frameW, frameH;
        int   trackId, classId;
        std::string className;
        float confidence;
        // Pixel-aligned crop box (even coordinates)
        int   cropX, cropY, cropW, cropH;
        std::string outPath;
    };

    CropSaverConfig m_cfg;

    // best-confidence map — accessed only under m_trackMutex
    std::mutex                           m_trackMutex;
    std::unordered_map<int, TrackRecord> m_tracks;

    // async queue
    std::queue<CropJob>     m_queue;
    std::mutex              m_queueMutex;
    std::condition_variable m_queueCv;

    std::thread       m_worker;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_shouldStop{false};
    std::atomic<int>  m_pending{0};   // jobs enqueued but not yet written

    std::atomic<uint64_t> m_cropsSaved{0};
    std::atomic<uint64_t> m_cropsDropped{0};

    void workerLoop();

    // Extract an NV12 crop and encode to JPEG file.
    // Returns true on success.
    bool writeCrop(const CropJob& job);
};
