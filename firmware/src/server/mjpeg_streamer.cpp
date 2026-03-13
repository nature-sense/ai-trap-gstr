#include "mjpeg_streamer.h"

// stb_image_write — implementation compiled in stb_image_write_impl.cpp
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "../common/stb_image_write.h"
#pragma GCC diagnostic pop

// ncnn NV12→RGB: BT.601 limited-range, NEON-optimised on ARM.
// Same function used by libcamera_capture and crop_saver — consistent colour
// across all three paths.  Replaces the hand-rolled loop which produced
// incorrect colours (wrong coefficients, UV range issue).
#include "ncnn/mat.h"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Destructor
// ─────────────────────────────────────────────────────────────────────────────

MjpegStreamer::~MjpegStreamer() {
    if (m_running.load())
        close();
}

// ─────────────────────────────────────────────────────────────────────────────
//  open()
// ─────────────────────────────────────────────────────────────────────────────

void MjpegStreamer::open(const MjpegStreamerConfig& cfg) {
    m_cfg = cfg;

    m_serverFd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (m_serverFd < 0)
        throw std::runtime_error(
            std::string("MjpegStreamer: socket() failed: ") + strerror(errno));

    int yes = 1;
    ::setsockopt(m_serverFd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));

    sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port        = htons(static_cast<uint16_t>(m_cfg.port));

    if (::bind(m_serverFd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        ::close(m_serverFd);
        throw std::runtime_error(
            std::string("MjpegStreamer: bind() failed: ") + strerror(errno));
    }
    if (::listen(m_serverFd, 8) < 0) {
        ::close(m_serverFd);
        throw std::runtime_error(
            std::string("MjpegStreamer: listen() failed: ") + strerror(errno));
    }

    m_shouldStop = false;
    m_running    = true;
    m_acceptThread = std::thread([this] { acceptLoop(); });

    printf("MjpegStreamer: listening on port %d  %dx%d  q=%d\n",
           m_cfg.port, m_cfg.streamWidth, m_cfg.streamHeight, m_cfg.jpegQuality);
    printf("  View at: http://<pi-ip>:%d/stream\n", m_cfg.port);
}

// ─────────────────────────────────────────────────────────────────────────────
//  close()
// ─────────────────────────────────────────────────────────────────────────────

void MjpegStreamer::close() {
    m_shouldStop = true;

    // Wake all client threads so they can exit
    m_frameCv.notify_all();

    // Close server socket so accept() unblocks
    if (m_serverFd >= 0) {
        ::shutdown(m_serverFd, SHUT_RDWR);
        ::close(m_serverFd);
        m_serverFd = -1;
    }

    if (m_acceptThread.joinable())
        m_acceptThread.join();

    // Disconnect all clients
    {
        std::lock_guard<std::mutex> lk(m_clientMutex);
        for (auto* c : m_clients) {
            ::shutdown(c->fd, SHUT_RDWR);
            ::close(c->fd);
        }
    }

    // Wait for all client threads to finish
    // (they will exit because m_shouldStop is true)
    std::vector<Client*> copy;
    {
        std::lock_guard<std::mutex> lk(m_clientMutex);
        copy = m_clients;
    }
    for (auto* c : copy) {
        if (c->thread.joinable())
            c->thread.join();
        delete c;
    }
    {
        std::lock_guard<std::mutex> lk(m_clientMutex);
        m_clients.clear();
    }

    m_running = false;
    printf("MjpegStreamer: closed  frames=%llu\n",
           (unsigned long long)m_framesPushed.load());
}

// ─────────────────────────────────────────────────────────────────────────────
//  pushFrame()
// ─────────────────────────────────────────────────────────────────────────────

void MjpegStreamer::pushFrame(const std::vector<uint8_t>& nv12,
                              int frameW, int frameH)
{
    if (!m_running.load()) return;
    // Skip encoding if no clients connected
    {
        std::lock_guard<std::mutex> lk(m_clientMutex);
        if (m_clients.empty()) return;
    }

    auto jpeg = encodeFrame(nv12, frameW, frameH);
    if (jpeg.empty()) return;

    {
        std::lock_guard<std::mutex> lk(m_frameMutex);
        m_jpeg = std::move(jpeg);
        m_frameSeq++;
    }
    m_frameCv.notify_all();
    m_framesPushed++;
}

// ─────────────────────────────────────────────────────────────────────────────
//  acceptLoop()
// ─────────────────────────────────────────────────────────────────────────────

void MjpegStreamer::acceptLoop() {
    while (!m_shouldStop.load()) {
        sockaddr_in clientAddr{};
        socklen_t   len = sizeof(clientAddr);

        int fd = ::accept(m_serverFd,
                          reinterpret_cast<sockaddr*>(&clientAddr), &len);
        if (fd < 0) {
            if (!m_shouldStop.load())
                perror("MjpegStreamer: accept");
            break;
        }

        char ipStr[INET_ADDRSTRLEN];
        inet_ntop(AF_INET, &clientAddr.sin_addr, ipStr, sizeof(ipStr));
        printf("MjpegStreamer: client connected %s\n", ipStr);

        auto* c = new Client();
        c->fd   = fd;
        {
            std::lock_guard<std::mutex> lk(m_clientMutex);
            m_clients.push_back(c);
        }
        c->thread = std::thread([this, c] { clientLoop(c); });
        c->thread.detach();   // cleaned up in removeClient()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  clientLoop()  — one per connected client
// ─────────────────────────────────────────────────────────────────────────────

static const char* HTTP_HEADER =
    "HTTP/1.0 200 OK\r\n"
    "Content-Type: multipart/x-mixed-replace; boundary=mjpegboundary\r\n"
    "Cache-Control: no-cache\r\n"
    "Connection: close\r\n"
    "\r\n";

void MjpegStreamer::clientLoop(Client* c) {
    // Send HTTP header
    if (::send(c->fd, HTTP_HEADER, strlen(HTTP_HEADER), MSG_NOSIGNAL) < 0) {
        removeClient(c);
        return;
    }

    uint64_t lastSeq = 0;

    while (!m_shouldStop.load()) {
        std::vector<uint8_t> jpeg;
        {
            std::unique_lock<std::mutex> lk(m_frameMutex);
            m_frameCv.wait(lk, [&] {
                return m_frameSeq != lastSeq || m_shouldStop.load();
            });
            if (m_shouldStop.load()) break;
            jpeg    = m_jpeg;          // copy latest frame
            lastSeq = m_frameSeq;
        }

        // Build MJPEG part header
        char partHdr[128];
        int  partHdrLen = snprintf(partHdr, sizeof(partHdr),
            "--mjpegboundary\r\n"
            "Content-Type: image/jpeg\r\n"
            "Content-Length: %zu\r\n"
            "\r\n",
            jpeg.size());

        if (::send(c->fd, partHdr, partHdrLen, MSG_NOSIGNAL) < 0) break;
        if (::send(c->fd, jpeg.data(), jpeg.size(), MSG_NOSIGNAL) < 0) break;
        if (::send(c->fd, "\r\n", 2, MSG_NOSIGNAL) < 0) break;
    }

    removeClient(c);
}

// ─────────────────────────────────────────────────────────────────────────────
//  removeClient()
// ─────────────────────────────────────────────────────────────────────────────

void MjpegStreamer::removeClient(Client* c) {
    ::close(c->fd);
    printf("MjpegStreamer: client disconnected\n");

    std::lock_guard<std::mutex> lk(m_clientMutex);
    m_clients.erase(
        std::remove(m_clients.begin(), m_clients.end(), c),
        m_clients.end());
    // Don't delete c here — thread is detached and may still be on the stack.
    // Let the OS reclaim it (small leak on disconnect, acceptable).
}

// ─────────────────────────────────────────────────────────────────────────────
//  encodeFrame()
// ─────────────────────────────────────────────────────────────────────────────

std::vector<uint8_t> MjpegStreamer::encodeFrame(
    const std::vector<uint8_t>& nv12, int srcW, int srcH) const
{
    // ── 1. NV12 → packed RGB ──────────────────────────────────────────────────
    // nv12 is compact (no stride padding): Y plane (srcW*srcH bytes) followed
    // by interleaved UV plane (srcW*srcH/2 bytes).
    // ncnn::yuv420sp2rgb_nv12 handles BT.601 limited-range correctly and is
    // NEON-optimised on ARM.
    std::vector<uint8_t> rgb(static_cast<size_t>(srcW * srcH * 3));
    ncnn::yuv420sp2rgb_nv12(nv12.data(), srcW, srcH, rgb.data());

    // ── 2. Scale to stream resolution ─────────────────────────────────────────
    std::vector<uint8_t> scaled = scaleRgb(rgb.data(), srcW, srcH,
                                           m_cfg.streamWidth, m_cfg.streamHeight);

    // ── 3. Encode to JPEG ─────────────────────────────────────────────────────
    // Use stbi_write_jpg (file-path variant) which exists in all stb versions.
    // Write to a per-thread temp path, read back, delete.
    char tmpPath[64];
    snprintf(tmpPath, sizeof(tmpPath), "/tmp/mjpeg_frame_%u.jpg",
             static_cast<unsigned>(
                 std::hash<std::thread::id>{}(std::this_thread::get_id())));

    std::vector<uint8_t> jpeg;

    if (stbi_write_jpg(tmpPath, m_cfg.streamWidth, m_cfg.streamHeight,
                       3, scaled.data(), m_cfg.jpegQuality)) {
        FILE* fp = fopen(tmpPath, "rb");
        if (fp) {
            fseek(fp, 0, SEEK_END);
            long sz = ftell(fp);
            rewind(fp);
            if (sz > 0) {
                jpeg.resize(static_cast<size_t>(sz));
                fread(jpeg.data(), 1, jpeg.size(), fp);
            }
            fclose(fp);
        }
        remove(tmpPath);
    }
    return jpeg;
}

// ─────────────────────────────────────────────────────────────────────────────
//  scaleRgb()  — simple bilinear downscale
// ─────────────────────────────────────────────────────────────────────────────

std::vector<uint8_t> MjpegStreamer::scaleRgb(
    const uint8_t* src, int srcW, int srcH, int dstW, int dstH)
{
    std::vector<uint8_t> dst(static_cast<size_t>(dstW * dstH * 3));

    const float xScale = static_cast<float>(srcW) / dstW;
    const float yScale = static_cast<float>(srcH) / dstH;

    for (int dy = 0; dy < dstH; dy++) {
        float fy  = (dy + 0.5f) * yScale - 0.5f;
        int   sy0 = static_cast<int>(fy);
        int   sy1 = sy0 + 1;
        float wy1 = fy - sy0;
        float wy0 = 1.f - wy1;
        sy0 = std::max(0, std::min(sy0, srcH - 1));
        sy1 = std::max(0, std::min(sy1, srcH - 1));

        for (int dx = 0; dx < dstW; dx++) {
            float fx  = (dx + 0.5f) * xScale - 0.5f;
            int   sx0 = static_cast<int>(fx);
            int   sx1 = sx0 + 1;
            float wx1 = fx - sx0;
            float wx0 = 1.f - wx1;
            sx0 = std::max(0, std::min(sx0, srcW - 1));
            sx1 = std::max(0, std::min(sx1, srcW - 1));

            for (int c = 0; c < 3; c++) {
                float v = wy0 * (wx0 * src[(sy0 * srcW + sx0) * 3 + c] +
                                 wx1 * src[(sy0 * srcW + sx1) * 3 + c]) +
                          wy1 * (wx0 * src[(sy1 * srcW + sx0) * 3 + c] +
                                 wx1 * src[(sy1 * srcW + sx1) * 3 + c]);
                dst[(dy * dstW + dx) * 3 + c] = static_cast<uint8_t>(v + 0.5f);
            }
        }
    }
    return dst;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Stats
// ─────────────────────────────────────────────────────────────────────────────

int MjpegStreamer::clientCount() const {
    std::lock_guard<std::mutex> lk(m_clientMutex);
    return static_cast<int>(m_clients.size());
}

void MjpegStreamer::printStats() const {
    printf("MjpegStreamer: clients=%d  frames=%llu\n",
           clientCount(),
           (unsigned long long)m_framesPushed.load());
}