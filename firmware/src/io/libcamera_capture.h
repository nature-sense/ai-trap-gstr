#pragma once

#include <libcamera/camera.h>
#include <libcamera/camera_manager.h>
#include <libcamera/controls.h>
#include <libcamera/formats.h>
#include <libcamera/framebuffer.h>
#include <libcamera/framebuffer_allocator.h>
#include <libcamera/request.h>
#include <libcamera/stream.h>

#include "ncnn/mat.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  LibcameraConfig
// ─────────────────────────────────────────────────────────────────────────────
struct LibcameraConfig {
    // Camera selection — empty = first camera found
    std::string cameraId    = "";

    // Capture resolution and framerate.
    // IMX708 sensor modes (Pi Camera Module 3, Pi 5 PiSP):
    //   4608 x 2592  @ 14 fps  — full resolution
    //   2304 x 1296  @ 56 fps  — 2x2 binned, full FOV  ← recommended
    //   1536 x  864  @120 fps  — 2x2 binned, high framerate
    int captureWidth  = 2304;
    int captureHeight = 1296;
    int framerate     = 30;

    // DMA buffer pool size — minimum 2, 4 is safe
    int bufferCount   = 4;

    // YOLO model input size — frames are letterboxed to this inside preprocess()
    int modelWidth    = 640;
    int modelHeight   = 640;

    // Sensor tuning file.
    // IMX708:      /usr/share/libcamera/ipa/rpi/pisp/imx708.json
    // IMX708 Wide: /usr/share/libcamera/ipa/rpi/pisp/imx708_wide.json
    // IMX477:      /usr/share/libcamera/ipa/rpi/pisp/imx477.json
    // Empty = use the IPA's built-in default tuning.
    std::string tuningFile = "";

    // ── Image quality controls ────────────────────────────────────────────────
    float exposureTimeUs = -1.f;   // µs; -1 = auto
    float analogGain     = -1.f;   // 1.0–16.0; -1 = auto
    float awbRed         = -1.f;   // colour gain R; -1 = auto
    float awbBlue        = -1.f;   // colour gain B; -1 = auto
    float brightness     =  0.f;   // −1.0 .. +1.0
    float contrast       =  1.f;   //  0.0 ..  4.0
    float saturation     =  1.f;   //  0.0 ..  4.0
    float sharpness      =  1.0f;  //  0.0 .. 16.0  (IMX708 IPA sharpens aggressively; 1.0 is safe)

    // ── Autofocus controls (IMX708 VCM hardware AF) ───────────────────────────
    // AfMode:
    //   0 = Manual     — lens held at lensPosition, no scanning
    //   1 = Auto       — single AF scan triggered at start(), then holds
    //   2 = Continuous — IPA continuously re-focuses (default)
    int   afMode       = 2;

    // AfRange (Auto and Continuous modes only):
    //   0 = Normal  — ~10 cm to infinity  ← recommended for insect traps
    //   1 = Macro   — optimised for < 10 cm
    //   2 = Full    — entire VCM range (slowest)
    int   afRange      = 0;

    // AfSpeed (Auto and Continuous modes only):
    //   0 = Normal  — smooth, less hunting  ← recommended
    //   1 = Fast    — aggressive, may hunt on uniform surfaces
    int   afSpeed      = 0;

    // LensPosition in dioptres — Manual mode only.
    // lensPosition = 1.0 / distance_metres
    //   0.0 = infinity
    //   1.0 = 1 m
    //   2.0 = 0.5 m  ← typical trap surface distance
    //   4.0 = 0.25 m
    // Note: if camera faces downward, gravity pulls lens toward sensor,
    // so focus lands nearer than a horizontal shot at the same distance.
    float lensPosition = 2.0f;

    // Optional AF metering window in sensor pixels [x, y, w, h].
    // All zero = full-frame metering (default).
    // Restrict to the trap surface centre to prevent focusing on edges.
    int   afWindowX    = 0;
    int   afWindowY    = 0;
    int   afWindowW    = 0;
    int   afWindowH    = 0;
};

// ─────────────────────────────────────────────────────────────────────────────
//  CaptureFrame — delivered to the FrameCallback on every captured frame
// ─────────────────────────────────────────────────────────────────────────────
struct CaptureFrame {
    uint64_t frameId     = 0;
    int64_t  timestampNs = 0;   // SensorTimestamp in nanoseconds

    // Actual captured dimensions (may differ from config after ISP adjustment)
    int width  = 0;
    int height = 0;

    // Autofocus state reported by the IPA on this frame.
    // 0=Idle, 1=Scanning, 2=Focused, 3=Failed
    // Only meaningful when afMode != Manual (0).
    int   afState      = 0;

    // Current lens position in dioptres (reported every frame by the IPA).
    float lensPosition = 0.f;

    // Letterbox parameters — pass directly to YoloDecoder::decode()
    float scale   = 1.f;
    int   padLeft = 0;
    int   padTop  = 0;

    // Preprocessed model input ready for ncnn inference.
    // Layout: channel-planar RGB, normalised [0, 1], shape [3, modelH, modelW]
    ncnn::Mat modelInput;

    // Raw NV12 frame (compact, de-strided): Y plane followed by interleaved UV.
    // Size: width * height * 3 / 2 bytes.
    // Passed to CropSaver::submit() to extract per-track JPEG crops.
    std::vector<uint8_t> nv12;
};

// ─────────────────────────────────────────────────────────────────────────────
//  LibcameraCapture
//
//  Captures frames from an IMX477 (or any libcamera-supported sensor) on
//  Raspberry Pi 5, preprocesses them for YOLO inference, and delivers them
//  via a callback.
//
//  Pixel format
//  ────────────
//  Requests NV12 from the ISP — the only format accepted directly by
//  ncnn::yuv420sp2rgb_nv12() without an extra CPU conversion step.
//  The ISP performs demosaic, AWB, AEC, and tone mapping before output.
//
//  Thread model
//  ────────────
//  libcamera fires requestComplete() on its own internal event thread.
//  That callback does the minimum: memcpy the DMA buffer, requeue the
//  request, push a RawFrame onto m_frameQueue.
//  A separate m_dispatchThread pops from the queue, runs NV12 → RGB →
//  letterbox → normalise, then calls the user FrameCallback.
//  This keeps the ISP pipeline flowing even when inference is slower than
//  the capture framerate.  Frames are dropped (oldest first) when the queue
//  reaches MAX_QUEUE_DEPTH.
//
//  Usage
//  ─────
//    LibcameraConfig cfg;
//    cfg.tuningFile = "/usr/share/libcamera/ipa/rpi/pisp/imx477.json";
//
//    LibcameraCapture cam;
//    cam.setCallback([&](const CaptureFrame& f) {
//        auto dets    = decoder.decode(f.modelInput,
//                                      f.width, f.height,
//                                      f.scale, f.padLeft, f.padTop);
//        auto tracked = tracker.update(dets);
//        db.writeBatch(...);
//    });
//    cam.setErrorCallback([](const std::string& msg) {
//        fprintf(stderr, "[cam] %s\n", msg.c_str());
//    });
//
//    cam.open(cfg);
//    cam.start();
//
//    // block until SIGINT / SIGTERM
//    while (!g_stop) std::this_thread::sleep_for(200ms);
//
//    cam.stop();
// ─────────────────────────────────────────────────────────────────────────────
class LibcameraCapture {
public:
    using FrameCallback = std::function<void(const CaptureFrame&)>;
    using ErrorCallback = std::function<void(const std::string& msg)>;

    LibcameraCapture();
    ~LibcameraCapture();

    LibcameraCapture(const LibcameraCapture&)            = delete;
    LibcameraCapture& operator=(const LibcameraCapture&) = delete;

    // ── Callbacks ─────────────────────────────────────────────────────────────
    // Must be set before open().
    void setCallback(FrameCallback cb);
    void setErrorCallback(ErrorCallback cb);

    // ── Lifecycle ─────────────────────────────────────────────────────────────

    // Configure the camera, allocate DMA buffers, build Requests.
    // Applies tuning file and image controls.
    // Throws std::runtime_error on any failure.
    void open(const LibcameraConfig& cfg);

    // Start the ISP, queue all Requests, launch dispatch thread.
    void start();

    // Signal stop, drain queue, stop ISP, release all resources.
    void stop();

    bool isRunning() const { return m_running.load(); }

    // ── Accessors ─────────────────────────────────────────────────────────────
    const LibcameraConfig& config() const { return m_cfg; }

    // ── Stats ─────────────────────────────────────────────────────────────────
    uint64_t framesReceived() const { return m_framesReceived.load(); }
    uint64_t framesDropped()  const { return m_framesDropped.load();  }
    void     printStats()     const;

private:
    // ── libcamera objects ─────────────────────────────────────────────────────
    std::unique_ptr<libcamera::CameraManager>        m_camManager;
    std::shared_ptr<libcamera::Camera>               m_camera;
    std::unique_ptr<libcamera::CameraConfiguration>  m_camConfig;
    std::unique_ptr<libcamera::FrameBufferAllocator> m_allocator;
    libcamera::Stream*                               m_stream = nullptr;
    std::vector<std::unique_ptr<libcamera::Request>> m_requests;

    // ── Config and derived state ──────────────────────────────────────────────
    LibcameraConfig m_cfg;
    int             m_stride = 0;   // Y-plane row stride in bytes (may > width)

    // ── Raw frame queue (event thread → dispatch thread) ─────────────────────
    struct RawFrame {
        uint64_t             frameId;
        int64_t              timestampNs;
        std::vector<uint8_t> nv12;     // contiguous Y + UV, de-strided
        int                  width;
        int                  height;
        int                  afState;
        float                lensPosition;
    };

    static constexpr int    MAX_QUEUE_DEPTH = 2;
    std::queue<RawFrame>    m_frameQueue;
    std::mutex              m_queueMutex;
    std::condition_variable m_queueCv;

    // ── Dispatch thread ───────────────────────────────────────────────────────
    std::thread       m_dispatchThread;
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_shouldStop{false};

    void dispatchLoop();

    // ── Internals ─────────────────────────────────────────────────────────────

    // libcamera completion signal — runs on the libcamera event thread
    void requestComplete(libcamera::Request* request);

    // NV12 → letterboxed normalised RGB ncnn::Mat
    ncnn::Mat preprocess(const uint8_t* nv12, int width, int height,
                         float& scale, int& padLeft, int& padTop) const;

    // Build the initial ControlList from m_cfg
    void applyControls(libcamera::ControlList& controls) const;

    // ── Counters (m_frameId written only on event thread) ─────────────────────
    uint64_t              m_frameId        = 0;
    std::atomic<uint64_t> m_framesReceived{0};
    std::atomic<uint64_t> m_framesDropped{0};

    // ── Callbacks ─────────────────────────────────────────────────────────────
    FrameCallback m_frameCb;
    ErrorCallback m_errorCb;

    void emitError(const std::string& msg);
};