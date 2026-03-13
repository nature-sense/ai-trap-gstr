#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  V4L2Capture
//
//  Drop-in replacement for LibcameraCapture targeting V4L2 cameras such as
//  the Radxa Camera 8M 219 (IMX219) on the Radxa Zero 3W (RK3566).
//
//  Delivers the identical CaptureFrame struct so the rest of the pipeline
//  (YoloDecoder, ByteTracker, SqliteWriter, CropSaver, MjpegStreamer) is
//  completely unchanged.
//
//  Pixel format
//  ────────────
//  Requests NV12 from the V4L2 driver.  RK3566 ISP outputs NV12 natively
//  with a single contiguous buffer (no shared-fd plane quirk).
//  On RK3566 the UV plane starts at exactly Y-plane-size bytes (stride * H),
//  unlike Pi 5 / PiSP which uses per-plane DMA offsets.
//
//  Thread model
//  ────────────
//  A capture thread calls VIDIOC_DQBUF in a tight loop, copies the NV12
//  data, re-queues the buffer immediately (VIDIOC_QBUF), and pushes a
//  RawFrame onto m_frameQueue.
//  A separate dispatch thread pops frames, runs NV12 → RGB → letterbox →
//  normalise, then calls the user FrameCallback — identical to LibcameraCapture.
//  Frames are dropped (oldest first) when the queue reaches MAX_QUEUE_DEPTH.
//
//  IMX219 sensor modes on RK3566 (Radxa Zero 3W):
//    3280 x 2464  @ 15 fps  — full resolution
//    1920 x 1080  @ 30 fps  — 1080p cropped
//    1640 x 1232  @ 30 fps  — 2x2 binned, full FOV  ← recommended
//    1280 x  720  @ 60 fps  — high framerate
//
//  Setup (once, on the board):
//    sudo rsetup → Overlays → enable imx219
//    reboot
//    v4l2-ctl --list-formats-ext   # verify NV12 is offered
//
//  Usage
//  ─────
//    V4L2Config cfg;
//    cfg.device       = "/dev/video0";
//    cfg.captureWidth = 1640;
//    cfg.captureHeight= 1232;
//    cfg.framerate    = 30;
//
//    V4L2Capture cam;
//    cam.setCallback([&](const CaptureFrame& f) {
//        auto dets    = decoder.decode(f.modelInput,
//                                      f.width, f.height,
//                                      f.scale, f.padLeft, f.padTop);
//        auto tracked = tracker.update(dets);
//    });
//    cam.setErrorCallback([](const std::string& msg) {
//        fprintf(stderr, "[cam] %s\n", msg.c_str());
//    });
//
//    cam.open(cfg);
//    cam.start();
//    while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
//    cam.stop();
// ─────────────────────────────────────────────────────────────────────────────

#include "ncnn/mat.h"

// Re-use CaptureFrame from libcamera_capture.h if both backends are present,
// otherwise define it here.  Guard prevents double-definition.
#ifndef CAPTURE_FRAME_DEFINED
#define CAPTURE_FRAME_DEFINED

#include <cstdint>
#include <vector>

struct CaptureFrame {
    uint64_t frameId     = 0;
    int64_t  timestampNs = 0;   // from v4l2_buffer.timestamp (µs → ns)

    int width  = 0;
    int height = 0;

    // Letterbox parameters — pass directly to YoloDecoder::decode()
    float scale   = 1.f;
    int   padLeft = 0;
    int   padTop  = 0;

    // Preprocessed model input: channel-planar RGB [0,1], shape [3,modelH,modelW]
    ncnn::Mat modelInput;

    // Raw compact NV12: Y plane then interleaved UV, width*height*3/2 bytes.
    std::vector<uint8_t> nv12;
};

#endif  // CAPTURE_FRAME_DEFINED

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  V4L2Config
// ─────────────────────────────────────────────────────────────────────────────
struct V4L2Config {
    // V4L2 device node
    std::string device        = "/dev/video0";

    // IMX219 on RK3566 — recommended modes:
    //   1640x1232 @ 30fps  (2x2 binned, full FOV)
    //   1280x720  @ 60fps  (high framerate)
    int captureWidth          = 1640;
    int captureHeight         = 1232;
    int framerate             = 30;

    // MMAP buffer count — 4 is a safe minimum
    int bufferCount           = 4;

    // YOLO model input size — frames are letterboxed to this
    int modelWidth            = 320;
    int modelHeight           = 320;
};

// ─────────────────────────────────────────────────────────────────────────────
//  V4L2Capture
// ─────────────────────────────────────────────────────────────────────────────
class V4L2Capture {
public:
    using FrameCallback = std::function<void(const CaptureFrame&)>;
    using ErrorCallback = std::function<void(const std::string& msg)>;

    V4L2Capture();
    ~V4L2Capture();

    V4L2Capture(const V4L2Capture&)            = delete;
    V4L2Capture& operator=(const V4L2Capture&) = delete;

    // ── Callbacks — set before open() ─────────────────────────────────────────
    void setCallback(FrameCallback cb);
    void setErrorCallback(ErrorCallback cb);

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    // Open device, negotiate format, allocate MMAP buffers.
    // Throws std::runtime_error on failure.
    void open(const V4L2Config& cfg);

    // Start streaming, launch capture + dispatch threads.
    void start();

    // Signal stop, drain queue, stop streaming, release buffers, close fd.
    void stop();

    bool isRunning() const { return m_running.load(); }

    // ── Stats ─────────────────────────────────────────────────────────────────
    uint64_t framesReceived() const { return m_framesReceived.load(); }
    uint64_t framesDropped()  const { return m_framesDropped.load();  }
    void     printStats()     const;

private:
    // ── V4L2 state ────────────────────────────────────────────────────────────
    int  m_fd     = -1;
    int  m_width  = 0;
    int  m_height = 0;
    int  m_stride = 0;   // bytes per Y row reported by driver (>= width)

    struct MmapBuffer {
        void*  start  = nullptr;
        size_t length = 0;
    };
    std::vector<MmapBuffer> m_buffers;

    // ── Config ────────────────────────────────────────────────────────────────
    V4L2Config m_cfg;

    // ── Raw frame queue (capture thread → dispatch thread) ────────────────────
    struct RawFrame {
        uint64_t             frameId;
        int64_t              timestampNs;
        std::vector<uint8_t> nv12;   // compact, de-strided
        int                  width;
        int                  height;
    };

    static constexpr int    MAX_QUEUE_DEPTH = 2;
    std::queue<RawFrame>    m_frameQueue;
    std::mutex              m_queueMutex;
    std::condition_variable m_queueCv;

    // ── Threads ───────────────────────────────────────────────────────────────
    std::thread       m_captureThread;
    std::thread       m_dispatchThread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_shouldStop{false};

    void captureLoop();
    void dispatchLoop();

    // ── Preprocessing (identical to LibcameraCapture::preprocess) ─────────────
    ncnn::Mat preprocess(const uint8_t* nv12, int width, int height,
                         float& scale, int& padLeft, int& padTop) const;

    // ── Counters ──────────────────────────────────────────────────────────────
    uint64_t              m_frameId = 0;
    std::atomic<uint64_t> m_framesReceived{0};
    std::atomic<uint64_t> m_framesDropped{0};

    // ── Callbacks ─────────────────────────────────────────────────────────────
    FrameCallback m_frameCb;
    ErrorCallback m_errorCb;

    void emitError(const std::string& msg);

    // ── Helpers ───────────────────────────────────────────────────────────────
    bool xioctl(int request, void* arg) const;
};
