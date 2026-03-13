#include "../v4l2_capture.h"
#include "ncnn/mat.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <stdexcept>
#include <chrono>

#include <fcntl.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <linux/videodev2.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

bool V4L2Capture::xioctl(int request, void* arg) const {
    int r;
    do {
        r = ioctl(m_fd, request, arg);
    } while (r == -1 && errno == EINTR);
    return r != -1;
}

void V4L2Capture::emitError(const std::string& msg) {
    if (m_errorCb) m_errorCb(msg);
    else fprintf(stderr, "[V4L2Capture] %s\n", msg.c_str());
}

// ─────────────────────────────────────────────────────────────────────────────
//  Construction / destruction
// ─────────────────────────────────────────────────────────────────────────────

V4L2Capture::V4L2Capture()  = default;

V4L2Capture::~V4L2Capture() {
    if (m_running.load())
        stop();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Callbacks
// ─────────────────────────────────────────────────────────────────────────────

void V4L2Capture::setCallback(FrameCallback cb)      { m_frameCb = std::move(cb); }
void V4L2Capture::setErrorCallback(ErrorCallback cb) { m_errorCb = std::move(cb); }

// ─────────────────────────────────────────────────────────────────────────────
//  open()
//
//  1. Open the device node.
//  2. Verify it is a capture device.
//  3. Negotiate NV12 format at the requested resolution.
//     The driver may adjust the resolution — we accept whatever it offers.
//  4. Set framerate via VIDIOC_S_PARM.
//  5. Allocate MMAP buffers and mmap each one.
// ─────────────────────────────────────────────────────────────────────────────

void V4L2Capture::open(const V4L2Config& cfg) {
    m_cfg = cfg;

    // ── 1. Open device ────────────────────────────────────────────────────────
    m_fd = ::open(cfg.device.c_str(), O_RDWR | O_NONBLOCK);
    if (m_fd < 0)
        throw std::runtime_error("V4L2Capture: cannot open " + cfg.device +
                                 ": " + strerror(errno));

    // ── 2. Verify capture capability ──────────────────────────────────────────
    v4l2_capability cap{};
    if (!xioctl(VIDIOC_QUERYCAP, &cap))
        throw std::runtime_error("V4L2Capture: VIDIOC_QUERYCAP failed: " +
                                 std::string(strerror(errno)));

    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE))
        throw std::runtime_error("V4L2Capture: " + cfg.device +
                                 " is not a video capture device");

    if (!(cap.capabilities & V4L2_CAP_STREAMING))
        throw std::runtime_error("V4L2Capture: " + cfg.device +
                                 " does not support streaming");

    // ── 3. Negotiate NV12 format ──────────────────────────────────────────────
    v4l2_format fmt{};
    fmt.type                  = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width         = static_cast<uint32_t>(cfg.captureWidth);
    fmt.fmt.pix.height        = static_cast<uint32_t>(cfg.captureHeight);
    fmt.fmt.pix.pixelformat   = V4L2_PIX_FMT_NV12;
    fmt.fmt.pix.field         = V4L2_FIELD_NONE;

    if (!xioctl(VIDIOC_S_FMT, &fmt))
        throw std::runtime_error("V4L2Capture: VIDIOC_S_FMT failed: " +
                                 std::string(strerror(errno)));

    // Driver may have adjusted resolution — accept it
    m_width  = static_cast<int>(fmt.fmt.pix.width);
    m_height = static_cast<int>(fmt.fmt.pix.height);
    m_stride = static_cast<int>(fmt.fmt.pix.bytesperline);

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_NV12)
        throw std::runtime_error("V4L2Capture: driver did not accept NV12 format");

    printf("V4L2Capture: %s  %dx%d (requested %dx%d)  stride=%d  fmt=NV12\n",
           cfg.device.c_str(),
           m_width, m_height,
           cfg.captureWidth, cfg.captureHeight,
           m_stride);

    // ── 4. Set framerate ──────────────────────────────────────────────────────
    v4l2_streamparm parm{};
    parm.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    parm.parm.capture.timeperframe.numerator   = 1;
    parm.parm.capture.timeperframe.denominator = static_cast<uint32_t>(cfg.framerate);

    if (!xioctl(VIDIOC_S_PARM, &parm))
        fprintf(stderr, "V4L2Capture: VIDIOC_S_PARM warning: %s "
                "(framerate may not be adjustable via this ioctl)\n",
                strerror(errno));
    else {
        int actualFps = static_cast<int>(
            parm.parm.capture.timeperframe.denominator /
            std::max(1u, parm.parm.capture.timeperframe.numerator));
        printf("V4L2Capture: framerate set to %d fps\n", actualFps);
    }

    // ── 5. Request MMAP buffers ───────────────────────────────────────────────
    v4l2_requestbuffers req{};
    req.count  = static_cast<uint32_t>(cfg.bufferCount);
    req.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (!xioctl(VIDIOC_REQBUFS, &req))
        throw std::runtime_error("V4L2Capture: VIDIOC_REQBUFS failed: " +
                                 std::string(strerror(errno)));

    if (req.count < 2)
        throw std::runtime_error("V4L2Capture: insufficient MMAP buffers allocated");

    printf("V4L2Capture: allocated %u MMAP buffers\n", req.count);

    // ── 6. mmap each buffer ───────────────────────────────────────────────────
    m_buffers.resize(req.count);
    for (uint32_t i = 0; i < req.count; i++) {
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;

        if (!xioctl(VIDIOC_QUERYBUF, &buf))
            throw std::runtime_error("V4L2Capture: VIDIOC_QUERYBUF failed: " +
                                     std::string(strerror(errno)));

        m_buffers[i].length = buf.length;
        m_buffers[i].start  = mmap(nullptr, buf.length,
                                   PROT_READ | PROT_WRITE,
                                   MAP_SHARED,
                                   m_fd, buf.m.offset);

        if (m_buffers[i].start == MAP_FAILED)
            throw std::runtime_error("V4L2Capture: mmap failed for buffer " +
                                     std::to_string(i) + ": " + strerror(errno));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  start()
// ─────────────────────────────────────────────────────────────────────────────

void V4L2Capture::start() {
    if (!m_frameCb)
        throw std::runtime_error("V4L2Capture: setCallback() must be called before start()");

    // Queue all buffers
    for (uint32_t i = 0; i < m_buffers.size(); i++) {
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (!xioctl(VIDIOC_QBUF, &buf))
            throw std::runtime_error("V4L2Capture: VIDIOC_QBUF failed: " +
                                     std::string(strerror(errno)));
    }

    // Start streaming
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (!xioctl(VIDIOC_STREAMON, &type))
        throw std::runtime_error("V4L2Capture: VIDIOC_STREAMON failed: " +
                                 std::string(strerror(errno)));

    m_shouldStop = false;
    m_running    = true;

    m_captureThread  = std::thread([this] { captureLoop();  });
    m_dispatchThread = std::thread([this] { dispatchLoop(); });

    printf("V4L2Capture: streaming started\n");
}

// ─────────────────────────────────────────────────────────────────────────────
//  stop()
// ─────────────────────────────────────────────────────────────────────────────

void V4L2Capture::stop() {
    m_shouldStop = true;
    m_queueCv.notify_all();

    if (m_captureThread.joinable())  m_captureThread.join();
    if (m_dispatchThread.joinable()) m_dispatchThread.join();

    // Stop streaming
    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    xioctl(VIDIOC_STREAMOFF, &type);

    // Unmap buffers
    for (auto& buf : m_buffers)
        if (buf.start && buf.start != MAP_FAILED)
            munmap(buf.start, buf.length);
    m_buffers.clear();

    if (m_fd >= 0) {
        ::close(m_fd);
        m_fd = -1;
    }

    m_running = false;
    printf("V4L2Capture: stopped\n");
}

// ─────────────────────────────────────────────────────────────────────────────
//  captureLoop  — runs on m_captureThread
//
//  Uses select() to wait for a frame with a 1-second timeout so we can
//  check m_shouldStop without blocking forever on VIDIOC_DQBUF.
// ─────────────────────────────────────────────────────────────────────────────

void V4L2Capture::captureLoop() {
    while (!m_shouldStop.load()) {

        // Wait for a frame to be ready (1 second timeout)
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(m_fd, &fds);
        timeval tv{};
        tv.tv_sec  = 1;
        tv.tv_usec = 0;

        int r = select(m_fd + 1, &fds, nullptr, nullptr, &tv);
        if (r == -1) {
            if (errno == EINTR) continue;
            emitError("captureLoop: select() error: " + std::string(strerror(errno)));
            break;
        }
        if (r == 0) continue;  // timeout — loop and check m_shouldStop

        // Dequeue a filled buffer
        v4l2_buffer buf{};
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (!xioctl(VIDIOC_DQBUF, &buf)) {
            if (errno == EAGAIN) continue;
            emitError("captureLoop: VIDIOC_DQBUF error: " + std::string(strerror(errno)));
            break;
        }

        m_framesReceived++;
        m_frameId++;

        // Timestamp from kernel: v4l2_buffer.timestamp is a timeval (µs)
        int64_t timestampNs =
            static_cast<int64_t>(buf.timestamp.tv_sec)  * 1'000'000'000LL +
            static_cast<int64_t>(buf.timestamp.tv_usec) * 1'000LL;

        // ── De-stride NV12 into compact buffer ────────────────────────────────
        // RK3566 V4L2: single contiguous buffer, Y at offset 0, UV at stride*H.
        // Unlike Pi 5 PiSP there are no per-plane DMA fd offsets to worry about.
        const int W = m_width;
        const int H = m_height;
        const int S = m_stride;   // bytes per Y row (may be > W with padding)

        const auto* src = static_cast<const uint8_t*>(m_buffers[buf.index].start);

        std::vector<uint8_t> nv12(static_cast<size_t>(W * H * 3 / 2));

        // De-stride Y rows
        for (int row = 0; row < H; row++)
            std::memcpy(nv12.data() + row * W,
                        src          + row * S,
                        static_cast<size_t>(W));

        // De-stride UV rows (UV plane starts at S*H in the driver buffer)
        const uint8_t* uvSrc = src + S * H;
        uint8_t*       uvDst = nv12.data() + W * H;
        for (int row = 0; row < H / 2; row++)
            std::memcpy(uvDst + row * W,
                        uvSrc + row * S,
                        static_cast<size_t>(W));

        // Re-queue the buffer immediately so the ISP stays fed
        if (!xioctl(VIDIOC_QBUF, &buf))
            emitError("captureLoop: VIDIOC_QBUF error: " + std::string(strerror(errno)));

        // Push to dispatch queue — drop oldest frame if queue is full
        {
            std::lock_guard<std::mutex> lk(m_queueMutex);
            if (static_cast<int>(m_frameQueue.size()) >= MAX_QUEUE_DEPTH) {
                m_frameQueue.pop();
                m_framesDropped++;
            }
            m_frameQueue.push({
                m_frameId,
                timestampNs,
                std::move(nv12),
                W, H
            });
        }
        m_queueCv.notify_one();
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  dispatchLoop  — runs on m_dispatchThread
//
//  Identical structure to LibcameraCapture::dispatchLoop.
//  Pops RawFrames, preprocesses, fires callback.
// ─────────────────────────────────────────────────────────────────────────────

void V4L2Capture::dispatchLoop() {
    while (true) {
        RawFrame raw;
        {
            std::unique_lock<std::mutex> lk(m_queueMutex);
            m_queueCv.wait(lk, [this] {
                return !m_frameQueue.empty() || m_shouldStop.load();
            });
            if (m_frameQueue.empty()) break;
            raw = std::move(m_frameQueue.front());
            m_frameQueue.pop();
        }

        CaptureFrame frame;
        frame.frameId     = raw.frameId;
        frame.timestampNs = raw.timestampNs;
        frame.width       = raw.width;
        frame.height      = raw.height;

        frame.modelInput = preprocess(raw.nv12.data(),
                                      raw.width, raw.height,
                                      frame.scale, frame.padLeft, frame.padTop);

        frame.nv12 = std::move(raw.nv12);

        if (m_frameCb)
            m_frameCb(frame);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  preprocess  — NV12 → letterboxed normalised RGB ncnn::Mat
//
//  Identical to LibcameraCapture::preprocess.
// ─────────────────────────────────────────────────────────────────────────────

ncnn::Mat V4L2Capture::preprocess(
    const uint8_t* nv12,
    int width, int height,
    float& scale, int& padLeft, int& padTop) const
{
    const int dstW = m_cfg.modelWidth;
    const int dstH = m_cfg.modelHeight;

    // NV12 → packed RGB (NEON-optimised, handles BT.601 limited-range)
    ncnn::Mat rgb(width, height, 3);
    ncnn::yuv420sp2rgb_nv12(nv12, width, height,
                            static_cast<uint8_t*>(rgb.data));

    // Letterbox: scale to fit dstW×dstH preserving aspect ratio
    scale   = std::min(static_cast<float>(dstW) / width,
                       static_cast<float>(dstH) / height);
    int newW = static_cast<int>(width  * scale);
    int newH = static_cast<int>(height * scale);
    padLeft  = (dstW - newW) / 2;
    padTop   = (dstH - newH) / 2;

    ncnn::Mat resized;
    ncnn::resize_bilinear(rgb, resized, newW, newH);

    // Pad with grey (114)
    ncnn::Mat padded(dstW, dstH, 3);
    padded.fill(114.f);

    for (int c = 0; c < 3; c++) {
        float*       dst = padded.channel(c);
        const float* src = resized.channel(c);
        for (int y = 0; y < newH; y++) {
            std::memcpy(dst + (padTop + y) * dstW + padLeft,
                        src + y * newW,
                        static_cast<size_t>(newW) * sizeof(float));
        }
    }

    // Normalise [0,255] → [0,1]
    const float mean[3] = { 0.f,     0.f,     0.f     };
    const float norm[3] = { 1/255.f, 1/255.f, 1/255.f };
    padded.substract_mean_normalize(mean, norm);

    return padded;
}

// ─────────────────────────────────────────────────────────────────────────────
//  printStats
// ─────────────────────────────────────────────────────────────────────────────

void V4L2Capture::printStats() const {
    printf("V4L2Capture: received=%-6llu  dropped=%-6llu\n",
           (unsigned long long)m_framesReceived.load(),
           (unsigned long long)m_framesDropped.load());
}
