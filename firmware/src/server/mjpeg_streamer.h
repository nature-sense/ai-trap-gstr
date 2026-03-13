#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <netinet/in.h>
#include <string>
#include <thread>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  MjpegStreamerConfig
// ─────────────────────────────────────────────────────────────────────────────
struct MjpegStreamerConfig {
    int port         = 9000;   // HTTP listen port
    int jpegQuality  = 75;     // JPEG quality 1–100 (lower = less CPU)
    int streamWidth  = 640;    // output frame width  (NV12 is scaled to this)
    int streamHeight = 480;    // output frame height
};

// ─────────────────────────────────────────────────────────────────────────────
//  MjpegStreamer
//
//  Serves a raw NV12 camera feed as an MJPEG HTTP stream.
//  Any browser or VLC can view it at:
//    http://<pi-ip>:<port>/stream
//
//  Thread model
//  ────────────
//  - One accept thread listens for new TCP connections.
//  - Each connected client gets its own sender thread that blocks on the
//    latest frame condition variable.
//  - pushFrame() is called from the LibcameraCapture dispatch thread; it
//    converts NV12 → RGB → scaled → JPEG and notifies all client threads.
//  - Clients that fall behind are skipped (latest frame wins); there is no
//    per-client queue, so slow viewers never stall the inference pipeline.
//
//  Usage
//  ─────
//    MjpegStreamer streamer;
//    streamer.open(cfg);
//
//    // inside FrameCallback:
//    streamer.pushFrame(frame.nv12, frame.width, frame.height);
//
//    streamer.close();
// ─────────────────────────────────────────────────────────────────────────────
class MjpegStreamer {
public:
    MjpegStreamer()  = default;
    ~MjpegStreamer();

    MjpegStreamer(const MjpegStreamer&)            = delete;
    MjpegStreamer& operator=(const MjpegStreamer&) = delete;

    // Start listening. Throws std::runtime_error on socket failure.
    void open(const MjpegStreamerConfig& cfg);

    // Encode a new frame and push it to all connected clients.
    // nv12    : compact de-strided NV12, frameW * frameH * 3 / 2 bytes
    // frameW/H: capture dimensions (will be scaled to streamWidth/Height)
    void pushFrame(const std::vector<uint8_t>& nv12, int frameW, int frameH);

    // Drain and stop all threads, close socket.
    void close();

    bool isRunning() const { return m_running.load(); }

    int  clientCount() const;
    void printStats()  const;

private:
    MjpegStreamerConfig m_cfg;

    // ── Server socket ─────────────────────────────────────────────────────────
    int  m_serverFd = -1;
    std::thread m_acceptThread;
    void acceptLoop();

    // ── Latest JPEG frame shared across all clients ───────────────────────────
    std::vector<uint8_t>    m_jpeg;
    uint64_t                m_frameSeq = 0;   // increments on every pushFrame
    mutable std::mutex      m_frameMutex;
    std::condition_variable m_frameCv;

    // ── Client management ─────────────────────────────────────────────────────
    struct Client {
        int      fd;
        uint64_t lastSeq = 0;
        std::thread thread;
    };

    mutable std::mutex       m_clientMutex;
    std::vector<Client*>     m_clients;

    void clientLoop(Client* c);
    void removeClient(Client* c);

    // ── State ─────────────────────────────────────────────────────────────────
    std::atomic<bool> m_running{false};
    std::atomic<bool> m_shouldStop{false};

    // ── Helpers ───────────────────────────────────────────────────────────────

    // NV12 → RGB → bilinear scale → JPEG
    std::vector<uint8_t> encodeFrame(const std::vector<uint8_t>& nv12,
                                     int srcW, int srcH) const;

    // Bilinear scale packed RGB srcW×srcH → dstW×dstH
    static std::vector<uint8_t> scaleRgb(const uint8_t* src,
                                         int srcW, int srcH,
                                         int dstW, int dstH);

    // ── Stats ─────────────────────────────────────────────────────────────────
    std::atomic<uint64_t> m_framesPushed{0};
};
