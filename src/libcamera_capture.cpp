#include "libcamera_capture.h"

#include "ncnn/mat.h"

// libcamera
#include <libcamera/control_ids.h>
#include <libcamera/controls.h>
#include <libcamera/pixel_format.h>
#include <libcamera/property_ids.h>

// sys
#include <sys/mman.h>
#include <cstdlib>   // setenv

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>

// ─────────────────────────────────────────────────────────────────────────────
//  Construction / destruction
// ─────────────────────────────────────────────────────────────────────────────

LibcameraCapture::LibcameraCapture() = default;

LibcameraCapture::~LibcameraCapture() {
    if (m_running.load())
        stop();
}

// ─────────────────────────────────────────────────────────────────────────────
//  Callbacks
// ─────────────────────────────────────────────────────────────────────────────

void LibcameraCapture::setCallback(FrameCallback cb)      { m_frameCb = std::move(cb); }
void LibcameraCapture::setErrorCallback(ErrorCallback cb) { m_errorCb = std::move(cb); }

// ─────────────────────────────────────────────────────────────────────────────
//  open()
//
//  Steps:
//    1. Start CameraManager and pick a camera.
//    2. Configure a single VideoRecording stream in NV12.
//    3. Allocate DMA frame buffers.
//    4. Build one libcamera::Request per buffer.
//    5. Apply image controls.
// ─────────────────────────────────────────────────────────────────────────────

void LibcameraCapture::open(const LibcameraConfig& cfg) {
    m_cfg = cfg;

    // ── 1. CameraManager ──────────────────────────────────────────────────────
    m_camManager = std::make_unique<libcamera::CameraManager>();

    // The RPi IPA reads the tuning file from this environment variable.
    // Setting it before CameraManager::start() is the supported way to
    // override tuning on all libcamera versions — no API change needed.
    // IMX477:      /usr/share/libcamera/ipa/rpi/vc4/imx477.json
    // IMX477 NOIR: /usr/share/libcamera/ipa/rpi/vc4/imx477_noir.json
    if (!m_cfg.tuningFile.empty()) {
        ::setenv("LIBCAMERA_RPI_TUNING_FILE", m_cfg.tuningFile.c_str(), 1);
        printf("LibcameraCapture: tuning file: %s\n", m_cfg.tuningFile.c_str());
    }

    if (m_camManager->start() != 0)
        throw std::runtime_error("LibcameraCapture: CameraManager::start() failed");

    auto cameras = m_camManager->cameras();
    if (cameras.empty())
        throw std::runtime_error("LibcameraCapture: no cameras found");

    // Pick camera by ID, or default to the first one
    if (m_cfg.cameraId.empty()) {
        m_camera = cameras.front();
        printf("LibcameraCapture: using camera \"%s\"\n",
               m_camera->id().c_str());
    } else {
        for (auto& c : cameras) {
            if (c->id() == m_cfg.cameraId) { m_camera = c; break; }
        }
        if (!m_camera)
            throw std::runtime_error("LibcameraCapture: camera \"" +
                                     m_cfg.cameraId + "\" not found");
    }

    if (m_camera->acquire() != 0)
        throw std::runtime_error("LibcameraCapture: camera acquire() failed — "
                                 "is another process using the camera?");

    // ── 2. Stream configuration ───────────────────────────────────────────────
    // VideoRecording role is the right choice for continuous capture:
    // it selects a sensor mode tuned for low latency over quality.
    m_camConfig = m_camera->generateConfiguration(
        { libcamera::StreamRole::VideoRecording });

    if (!m_camConfig || m_camConfig->empty())
        throw std::runtime_error("LibcameraCapture: generateConfiguration() failed");

    libcamera::StreamConfiguration& streamCfg = m_camConfig->at(0);

    // NV12: Y plane full-res, interleaved UV half-res.
    // Direct input to ncnn::yuv420sp2rgb_nv12() — no CPU colour conversion.
    streamCfg.pixelFormat = libcamera::formats::NV12;
    streamCfg.size        = { static_cast<unsigned>(m_cfg.captureWidth),
                               static_cast<unsigned>(m_cfg.captureHeight) };
    streamCfg.bufferCount = static_cast<unsigned>(m_cfg.bufferCount);

    // Framerate: set via StreamConfiguration::frameSize is not supported;
    // framerate is controlled via the FrameDuration control below.

    auto status = m_camConfig->validate();
    if (status == libcamera::CameraConfiguration::Invalid)
        throw std::runtime_error("LibcameraCapture: stream configuration is invalid\n"
                                 "  Requested: " + std::to_string(m_cfg.captureWidth) +
                                 "x" + std::to_string(m_cfg.captureHeight) +
                                 "  Check supported sensor modes with:\n"
                                 "  libcamera-hello --list-cameras");

    if (status == libcamera::CameraConfiguration::Adjusted) {
        // libcamera snapped to the nearest supported sensor mode.
        // This is normal — log the actual size so the user knows what they got.
        int adjW = static_cast<int>(streamCfg.size.width);
        int adjH = static_cast<int>(streamCfg.size.height);
        printf("LibcameraCapture: requested %dx%d adjusted to %dx%d %s\n",
               m_cfg.captureWidth, m_cfg.captureHeight,
               adjW, adjH,
               streamCfg.pixelFormat.toString().c_str());
        printf("  IMX708 supported modes: 4608x2592@14  2304x1296@56  1536x864@120\n");
        m_cfg.captureWidth  = adjW;
        m_cfg.captureHeight = adjH;
    }

    if (m_camera->configure(m_camConfig.get()) != 0)
        throw std::runtime_error("LibcameraCapture: camera configure() failed");

    m_stream = streamCfg.stream();
    m_stride = static_cast<int>(streamCfg.stride);

    printf("LibcameraCapture: stream %dx%d  fmt=%s  stride=%d  bufs=%u\n",
           m_cfg.captureWidth, m_cfg.captureHeight,
           streamCfg.pixelFormat.toString().c_str(),
           m_stride,
           streamCfg.bufferCount);

    // ── 3. Allocate frame buffers ─────────────────────────────────────────────
    m_allocator = std::make_unique<libcamera::FrameBufferAllocator>(m_camera);

    if (m_allocator->allocate(m_stream) < 0)
        throw std::runtime_error("LibcameraCapture: FrameBufferAllocator::allocate() failed");

    const auto& buffers = m_allocator->buffers(m_stream);
    if (buffers.empty())
        throw std::runtime_error("LibcameraCapture: no buffers allocated");

    // ── 4. Build one Request per buffer ──────────────────────────────────────
    m_requests.reserve(buffers.size());

    for (const auto& buffer : buffers) {
        auto request = m_camera->createRequest();
        if (!request)
            throw std::runtime_error("LibcameraCapture: createRequest() failed");

        if (request->addBuffer(m_stream, buffer.get()) != 0)
            throw std::runtime_error("LibcameraCapture: request->addBuffer() failed");

        m_requests.push_back(std::move(request));
    }

    // ── 5. Register completion callback ──────────────────────────────────────
    m_camera->requestCompleted.connect(
        this, &LibcameraCapture::requestComplete);

    printf("LibcameraCapture: ready  (%zu requests)\n", m_requests.size());
}

// ─────────────────────────────────────────────────────────────────────────────
//  start()
// ─────────────────────────────────────────────────────────────────────────────

void LibcameraCapture::start() {
    if (!m_camera)
        throw std::runtime_error("LibcameraCapture: call open() before start()");
    if (m_running.load()) return;

    // Build the initial control list: framerate + user-supplied image controls
    libcamera::ControlList controls(libcamera::controls::controls);

    // Framerate as min/max FrameDuration in microseconds
    int64_t frameDurationUs = 1000000 / m_cfg.framerate;
    controls.set(libcamera::controls::FrameDurationLimits,
                 libcamera::Span<const int64_t, 2>({ frameDurationUs,
                                                     frameDurationUs }));

    applyControls(controls);

    if (m_camera->start(&controls) != 0)
        throw std::runtime_error("LibcameraCapture: camera start() failed");

    // Queue all requests
    for (auto& req : m_requests) {
        if (m_camera->queueRequest(req.get()) != 0)
            throw std::runtime_error("LibcameraCapture: queueRequest() failed");
    }

    m_shouldStop = false;
    m_running    = true;
    m_dispatchThread = std::thread([this] { dispatchLoop(); });

    printf("LibcameraCapture: started — %d fps\n", m_cfg.framerate);
}

// ─────────────────────────────────────────────────────────────────────────────
//  stop()
// ─────────────────────────────────────────────────────────────────────────────

void LibcameraCapture::stop() {
    if (!m_running.load()) return;

    // Signal dispatch thread
    m_shouldStop = true;
    m_queueCv.notify_all();

    m_camera->stop();

    if (m_dispatchThread.joinable())
        m_dispatchThread.join();

    m_running = false;

    // Release resources
    m_requests.clear();
    m_allocator.reset();
    m_camera->release();
    m_camera.reset();
    m_camManager->stop();
    m_camManager.reset();

    printf("LibcameraCapture: stopped  "
           "(received=%llu  dropped=%llu)\n",
           (unsigned long long)m_framesReceived.load(),
           (unsigned long long)m_framesDropped.load());
}

// ─────────────────────────────────────────────────────────────────────────────
//  requestComplete  — called on libcamera's internal event thread
//
//  This must return as fast as possible to keep the ISP pipeline flowing.
//  We:
//    1. Check the request completed without error.
//    2. Map the DMA buffer and copy the NV12 data into a std::vector.
//    3. Push a RawFrame onto the dispatch queue (drop if queue full).
//    4. Requeue the request immediately.
// ─────────────────────────────────────────────────────────────────────────────

void LibcameraCapture::requestComplete(libcamera::Request* request) {
    if (request->status() == libcamera::Request::RequestCancelled)
        return;

    m_framesReceived++;
    m_frameId++;

    // ── Map buffer and copy NV12 data ─────────────────────────────────────────
    const auto& buffers = request->buffers();
    auto it = buffers.find(m_stream);
    if (it == buffers.end()) {
        emitError("requestComplete: stream buffer not found");
        request->reuse(libcamera::Request::ReuseBuffers);
        m_camera->queueRequest(request);
        return;
    }

    const libcamera::FrameBuffer* fb = it->second;
    const libcamera::FrameMetadata& meta = fb->metadata();

    // Get the SensorTimestamp control value if available.
    int64_t timestampNs = 0;
    {
        auto ts = request->metadata().get<int64_t>(
                      libcamera::controls::SensorTimestamp);
        if (ts) timestampNs = *ts;
    }

    // Extract AF state and lens position from per-frame metadata.
    int   afState      = 0;
    float lensPosition = 0.f;
    {
        auto af = request->metadata().get(libcamera::controls::AfState);
        if (af) afState = *af;
        auto lp = request->metadata().get(libcamera::controls::LensPosition);
        if (lp) lensPosition = *lp;
    }

    // Sizes in the DMA buffer (may have stride padding per row)
    const int W         = m_cfg.captureWidth;
    const int H         = m_cfg.captureHeight;
    const int S         = m_stride;              // bytes per Y row (>= W)

    // Output: compact NV12, no stride padding — width * height * 3/2 bytes
    const int compactY  = W * H;
    const int compactUV = W * H / 2;
    std::vector<uint8_t> nv12(static_cast<size_t>(compactY + compactUV));

    if (fb->planes().empty()) {
        emitError("requestComplete: frame buffer has no planes");
        request->reuse(libcamera::Request::ReuseBuffers);
        m_camera->queueRequest(request);
        return;
    }

    // ── Pi 5 PiSP NV12 buffer layout ─────────────────────────────────────────
    // PiSP reports nplanes=2 but BOTH planes share a single DMA fd.
    // plane[0].offset = 0         (Y plane start)
    // plane[1].offset = S*H       (UV plane start, may include stride padding)
    // We must map the entire buffer once using fd from plane[0] and apply
    // each plane's .offset explicitly.
    // DO NOT mmap plane[1].fd separately — it is the same fd as plane[0]
    // and mmap'ing from offset 0 gives Y data, not UV data.
    {
        int      fd0     = fb->planes()[0].fd.get();
        uint32_t yOff    = fb->planes()[0].offset;
        uint32_t uvOff   = (fb->planes().size() >= 2)
                           ? fb->planes()[1].offset
                           : static_cast<uint32_t>(S * H);
        size_t   mapSize = uvOff + static_cast<size_t>(S * H / 2);

        void* mapped = mmap(nullptr, mapSize, PROT_READ, MAP_SHARED, fd0, 0);
        if (mapped == MAP_FAILED) {
            emitError("requestComplete: mmap failed");
            request->reuse(libcamera::Request::ReuseBuffers);
            m_camera->queueRequest(request);
            return;
        }

        const auto* base = static_cast<const uint8_t*>(mapped);

        // De-stride Y plane
        const uint8_t* ySrc = base + yOff;
        for (int r = 0; r < H; r++)
            std::memcpy(nv12.data() + r * W,
                        ySrc         + r * S,
                        static_cast<size_t>(W));

        // De-stride UV plane
        const uint8_t* uvSrc = base + uvOff;
        uint8_t*       uvDst = nv12.data() + compactY;
        for (int r = 0; r < H / 2; r++)
            std::memcpy(uvDst + r * W,
                        uvSrc + r * S,
                        static_cast<size_t>(W));

        munmap(mapped, mapSize);
    }

    (void)meta;   // unused; timestamp from SensorTimestamp control

    // ── Push to dispatch queue (drop frame if queue is at capacity) ────────────
    {
        std::lock_guard<std::mutex> lock(m_queueMutex);
        if (static_cast<int>(m_frameQueue.size()) >= MAX_QUEUE_DEPTH) {
            m_framesDropped++;
            // Drop oldest frame to make room
            m_frameQueue.pop();
        }
        m_frameQueue.push({
            m_frameId,
            timestampNs,
            std::move(nv12),
            m_cfg.captureWidth,
            m_cfg.captureHeight,
            afState,
            lensPosition
        });
    }
    m_queueCv.notify_one();

    // ── Requeue the request immediately ──────────────────────────────────────
    request->reuse(libcamera::Request::ReuseBuffers);
    m_camera->queueRequest(request);
}

// ─────────────────────────────────────────────────────────────────────────────
//  dispatchLoop  — runs on m_dispatchThread
//
//  Pops RawFrames from the queue, runs the NV12 → RGB → letterbox
//  preprocessing, then calls the user FrameCallback.
// ─────────────────────────────────────────────────────────────────────────────

void LibcameraCapture::dispatchLoop() {
    while (true) {
        RawFrame raw;

        {
            std::unique_lock<std::mutex> lock(m_queueMutex);
            m_queueCv.wait(lock, [this] {
                return !m_frameQueue.empty() || m_shouldStop.load();
            });

            if (m_frameQueue.empty()) break;   // stop requested, queue drained

            raw = std::move(m_frameQueue.front());
            m_frameQueue.pop();
        }

        // ── Preprocess: NV12 → letterboxed normalised RGB ─────────────────────
        float scale; int padLeft, padTop;
        ncnn::Mat modelInput;

        try {
            modelInput = preprocess(raw.nv12.data(),
                                    raw.width, raw.height,
                                    scale, padLeft, padTop);
        } catch (const std::exception& e) {
            emitError(std::string("preprocess failed: ") + e.what());
            continue;
        }

        // ── Build CaptureFrame and call user callback ─────────────────────────
        CaptureFrame frame;
        frame.frameId     = raw.frameId;
        frame.timestampNs = raw.timestampNs;
        frame.width       = raw.width;
        frame.height      = raw.height;
        frame.scale       = scale;
        frame.padLeft     = padLeft;
        frame.padTop      = padTop;
        frame.modelInput  = std::move(modelInput);
        frame.nv12        = std::move(raw.nv12);
        frame.afState     = raw.afState;
        frame.lensPosition= raw.lensPosition;

        if (m_frameCb) {
            try {
                m_frameCb(frame);
            } catch (const std::exception& e) {
                emitError(std::string("frame callback exception: ") + e.what());
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  preprocess  — compact NV12 → letterboxed normalised RGB ncnn::Mat
//
//  nv12 must be compact (no stride padding) — de-striding is done at copy
//  time in requestComplete() using m_stride.
//
//  Steps:
//  1. yuv420sp2rgb_nv12() — NEON-optimised, no extra allocation.
//  2. Letterbox-resize to modelWidth × modelHeight preserving aspect ratio.
//  3. Pad remainder with grey (114).
//  4. Normalise [0, 255] → [0, 1].
// ─────────────────────────────────────────────────────────────────────────────

ncnn::Mat LibcameraCapture::preprocess(
    const uint8_t* nv12,
    int width, int height,
    float& scale, int& padLeft, int& padTop) const
{
    // nv12 is already compact (de-strided in requestComplete).
    // Layout: Y plane (width*height) followed by UV plane (width*height/2).
    const int dstW = m_cfg.modelWidth;
    const int dstH = m_cfg.modelHeight;

    // ── NV12 → RGB ────────────────────────────────────────────────────────────
    // yuv420sp2rgb_nv12 writes packed uint8 RGB into a raw byte buffer.
    // ncnn::Mat(w, h, 3) allocates floats (4 bytes/element) — passing that
    // directly to yuv420sp2rgb_nv12 packs RGB bytes into the first quarter of
    // the buffer and leaves the rest uninitialised, producing huge float values
    // when resize_bilinear reads the data as floats.
    // Fix: convert into a plain uint8 buffer, then use from_pixels() to load
    // into a properly typed ncnn::Mat.
    std::vector<uint8_t> rgbBuf(static_cast<size_t>(width * height * 3));
    ncnn::yuv420sp2rgb_nv12(nv12, width, height, rgbBuf.data());

    // from_pixels() allocates a float mat and scales uint8 [0,255] → float
    // [0,255] (not normalised yet — normalisation happens after padding).
    ncnn::Mat rgb = ncnn::Mat::from_pixels(rgbBuf.data(),
                                           ncnn::Mat::PIXEL_RGB,
                                           width, height);

    // ── Letterbox: resize to fit dstW×dstH preserving aspect ratio ────────────
    scale   = std::min(static_cast<float>(dstW) / width,
                       static_cast<float>(dstH) / height);
    int newW = static_cast<int>(width  * scale);
    int newH = static_cast<int>(height * scale);
    padLeft  = (dstW - newW) / 2;
    padTop   = (dstH - newH) / 2;

    ncnn::Mat resized;
    ncnn::resize_bilinear(rgb, resized, newW, newH);

    // ── Pad to model size with grey (114) ─────────────────────────────────────
    // copy_make_border is layout-aware and correctly fills border pixels.
    int padRight  = dstW - newW - padLeft;
    int padBottom = dstH - newH - padTop;
    ncnn::Mat padded;
    ncnn::copy_make_border(resized, padded,
                           padTop, padBottom, padLeft, padRight,
                           ncnn::BORDER_CONSTANT, 114.f);

    // ── Normalise [0,255] → [0,1] ─────────────────────────────────────────────
    const float mean[3] = { 0.f,     0.f,     0.f     };
    const float norm[3] = { 1/255.f, 1/255.f, 1/255.f };
    padded.substract_mean_normalize(mean, norm);

    return padded;
}

// ─────────────────────────────────────────────────────────────────────────────
//  applyControls
// ─────────────────────────────────────────────────────────────────────────────

void LibcameraCapture::applyControls(libcamera::ControlList& controls) const {
    using namespace libcamera::controls;

    // ── Image quality ─────────────────────────────────────────────────────────
    controls.set(Brightness,  m_cfg.brightness);
    controls.set(Contrast,    m_cfg.contrast);
    controls.set(Saturation,  m_cfg.saturation);
    controls.set(Sharpness,   m_cfg.sharpness);

    // ── Exposure ──────────────────────────────────────────────────────────────
    if (m_cfg.exposureTimeUs > 0.f) {
        controls.set(AeEnable,    false);
        controls.set(ExposureTime,
                     static_cast<int32_t>(m_cfg.exposureTimeUs));
    } else {
        controls.set(AeEnable,    true);
    }

    if (m_cfg.analogGain > 0.f)
        controls.set(AnalogueGain, m_cfg.analogGain);

    // ── White balance ─────────────────────────────────────────────────────────
    if (m_cfg.awbRed > 0.f && m_cfg.awbBlue > 0.f) {
        controls.set(AwbEnable,   false);
        controls.set(ColourGains,
                     libcamera::Span<const float, 2>(
                         { m_cfg.awbRed, m_cfg.awbBlue }));
    } else {
        controls.set(AwbEnable,   true);
        controls.set(AwbMode,     AwbAuto);
    }

    // ── Autofocus (IMX708 VCM hardware AF) ───────────────────────────────────
    switch (m_cfg.afMode) {

        case 0:   // Manual — hold lens at fixed dioptre position
            controls.set(AfMode,      AfModeManual);
            controls.set(LensPosition, m_cfg.lensPosition);
            printf("LibcameraCapture: AF Manual  lens=%.2f D (%.0f cm)\n",
                   m_cfg.lensPosition,
                   m_cfg.lensPosition > 0.f ? 100.f / m_cfg.lensPosition : 0.f);
            break;

        case 1:   // Auto — single scan at start(), then holds position
            controls.set(AfMode,     AfModeAuto);
            controls.set(AfRange,    m_cfg.afRange);
            controls.set(AfSpeed,    m_cfg.afSpeed);
            controls.set(AfTrigger,  AfTriggerStart);
            printf("LibcameraCapture: AF Auto  range=%d  speed=%d\n",
                   m_cfg.afRange, m_cfg.afSpeed);
            break;

        default:  // 2 = Continuous (default)
        case 2:   // IPA continuously re-evaluates focus
            controls.set(AfMode,     AfModeContinuous);
            controls.set(AfRange,    m_cfg.afRange);
            controls.set(AfSpeed,    m_cfg.afSpeed);
            printf("LibcameraCapture: AF Continuous  range=%d  speed=%d\n",
                   m_cfg.afRange, m_cfg.afSpeed);
            break;
    }

    // ── Optional AF metering window ───────────────────────────────────────────
    // Only applied in Auto or Continuous mode and when caller set a non-zero window.
    if (m_cfg.afMode != 0 && m_cfg.afWindowW > 0 && m_cfg.afWindowH > 0) {
        libcamera::Rectangle win(m_cfg.afWindowX, m_cfg.afWindowY,
                                 m_cfg.afWindowW, m_cfg.afWindowH);
        controls.set(AfMetering, AfMeteringWindows);
        controls.set(AfWindows,
                     libcamera::Span<const libcamera::Rectangle>(&win, 1));
        printf("LibcameraCapture: AF window [%d,%d %dx%d]\n",
               m_cfg.afWindowX, m_cfg.afWindowY,
               m_cfg.afWindowW, m_cfg.afWindowH);
    }

    // Note: NoiseReductionMode lives in libcamera::controls::draft and its
    // header path varies between libcamera versions.  The RPi IPA defaults
    // to fast noise reduction automatically, so we leave it unset here.
}

// ─────────────────────────────────────────────────────────────────────────────
//  Stats / helpers
// ─────────────────────────────────────────────────────────────────────────────

void LibcameraCapture::printStats() const {
    printf("LibcameraCapture: received=%llu  dropped=%llu\n",
           (unsigned long long)m_framesReceived.load(),
           (unsigned long long)m_framesDropped.load());
}

void LibcameraCapture::emitError(const std::string& msg) {
    if (m_errorCb) m_errorCb(msg);
    else fprintf(stderr, "LibcameraCapture error: %s\n", msg.c_str());
}