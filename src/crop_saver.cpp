#include "crop_saver.h"

// ncnn — for NV12→RGB conversion
#include "ncnn/mat.h"

// stb_image_write — implementation compiled in stb_image_write_impl.cpp
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#include "stb_image_write.h"
#pragma GCC diagnostic pop

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <sys/stat.h>

// ─────────────────────────────────────────────────────────────────────────────
//  Helpers
// ─────────────────────────────────────────────────────────────────────────────

static bool mkdirP(const std::string& path) {
    // Create directory (single level).  Ignore EEXIST.
    if (::mkdir(path.c_str(), 0755) != 0 && errno != EEXIST)
        return false;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  Destructor
// ─────────────────────────────────────────────────────────────────────────────

CropSaver::~CropSaver() {
    if (m_running.load())
        close();
}

// ─────────────────────────────────────────────────────────────────────────────
//  open()
// ─────────────────────────────────────────────────────────────────────────────

void CropSaver::open(const CropSaverConfig& cfg) {
    m_cfg = cfg;

    if (!mkdirP(m_cfg.outputDir))
        throw std::runtime_error(
            "CropSaver: cannot create output directory \"" +
            m_cfg.outputDir + "\": " + strerror(errno));

    m_shouldStop = false;
    m_running    = true;
    m_worker     = std::thread([this] { workerLoop(); });

    printf("CropSaver: ready  dir=%s  minConf=%.2f  jpegQ=%d\n",
           m_cfg.outputDir.c_str(), m_cfg.minConfidence, m_cfg.jpegQuality);
}

// ─────────────────────────────────────────────────────────────────────────────
//  submit()
// ─────────────────────────────────────────────────────────────────────────────

bool CropSaver::submit(const std::vector<uint8_t>& nv12,
                       int frameW, int frameH,
                       int trackId, int classId,
                       const std::string& className,
                       float confidence,
                       float x1, float y1, float x2, float y2)
{
    // ── 1. Fast-path confidence gate ──────────────────────────────────────────
    if (confidence < m_cfg.minConfidence)
        return false;

    // ── 2. Check and update best-confidence map ───────────────────────────────
    std::string outPath;
    {
        std::lock_guard<std::mutex> lk(m_trackMutex);
        auto& rec = m_tracks[trackId];
        if (confidence <= rec.bestConfidence)
            return false;   // not a new best — skip

        rec.bestConfidence = confidence;
        // Fixed filename per track — overwrites previous best on disk
        outPath = m_cfg.outputDir + "/" +
                  className + "_" + std::to_string(trackId) + ".jpg";
        rec.lastPath = outPath;
    }

    // ── 3. Clip and 2-pixel align the crop box ────────────────────────────────
    // NV12 UV plane is 2×2 subsampled — both origin and size must be even.
    int cx1 = std::max(0,      static_cast<int>(x1))  & ~1;
    int cy1 = std::max(0,      static_cast<int>(y1))  & ~1;
    int cx2 = std::min(frameW, static_cast<int>(x2 + 0.5f) + 1) & ~1;
    int cy2 = std::min(frameH, static_cast<int>(y2 + 0.5f) + 1) & ~1;
    int cw  = cx2 - cx1;
    int ch  = cy2 - cy1;

    if (cw < 4 || ch < 4)
        return false;   // degenerate box

    // ── 4. Enqueue (drop if queue full) ───────────────────────────────────────
    {
        std::lock_guard<std::mutex> lk(m_queueMutex);
        if (static_cast<int>(m_queue.size()) >= m_cfg.maxQueueDepth) {
            m_cropsDropped++;
            return false;
        }
        CropJob job;
        job.nv12       = nv12;   // copy — CaptureFrame may be freed on return
        job.frameW     = frameW;
        job.frameH     = frameH;
        job.trackId    = trackId;
        job.classId    = classId;
        job.className  = className;
        job.confidence = confidence;
        job.cropX      = cx1;
        job.cropY      = cy1;
        job.cropW      = cw;
        job.cropH      = ch;
        job.outPath    = std::move(outPath);
        m_queue.push(std::move(job));
        m_pending++;
    }
    m_queueCv.notify_one();
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  flush() / close()
// ─────────────────────────────────────────────────────────────────────────────

void CropSaver::flush() {
    // Spin-wait until worker has processed all enqueued jobs
    while (m_pending.load() > 0)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

void CropSaver::close() {
    flush();
    m_shouldStop = true;
    m_queueCv.notify_all();
    if (m_worker.joinable())
        m_worker.join();
    m_running = false;

    printf("CropSaver: closed  saved=%llu  dropped=%llu\n",
           (unsigned long long)m_cropsSaved.load(),
           (unsigned long long)m_cropsDropped.load());
}

// ─────────────────────────────────────────────────────────────────────────────
//  workerLoop()
// ─────────────────────────────────────────────────────────────────────────────

void CropSaver::workerLoop() {
    while (true) {
        CropJob job;
        {
            std::unique_lock<std::mutex> lk(m_queueMutex);
            m_queueCv.wait(lk, [this] {
                return !m_queue.empty() || m_shouldStop.load();
            });
            if (m_queue.empty()) break;
            job = std::move(m_queue.front());
            m_queue.pop();
        }

        if (writeCrop(job))
            m_cropsSaved++;
        else
            m_cropsDropped++;

        m_pending--;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  writeCrop()
//
//  Steps:
//   1. Extract the NV12 crop region into a compact sub-buffer.
//   2. Convert NV12 crop → packed RGB via ncnn::yuv420sp2rgb_nv12.
//   3. Encode RGB → JPEG via stb_image_write.
//   4. Write file (overwrites previous best for this track).
// ─────────────────────────────────────────────────────────────────────────────

bool CropSaver::writeCrop(const CropJob& job) {
    const int FW  = job.frameW;
    const int FH  = job.frameH;
    const int CX  = job.cropX;
    const int CY  = job.cropY;
    const int CW  = job.cropW;
    const int CH  = job.cropH;

    // The full-frame NV12 is compact: Y plane at [0], UV plane at [FW*FH].
    // All dimensions are even (enforced in submit()).
    const uint8_t* srcY  = job.nv12.data();
    const uint8_t* srcUV = job.nv12.data() + FW * FH;

    // ── 1. Build a compact NV12 sub-buffer for the crop region ───────────────
    // Size: CW * CH (Y) + CW * CH/2 (UV) = CW * CH * 3/2
    std::vector<uint8_t> cropNV12(static_cast<size_t>(CW * CH * 3 / 2));

    uint8_t* dstY  = cropNV12.data();
    uint8_t* dstUV = cropNV12.data() + CW * CH;

    // Y rows: one byte per pixel
    for (int row = 0; row < CH; row++)
        std::memcpy(dstY  + row * CW,
                    srcY  + (CY + row) * FW + CX,
                    static_cast<size_t>(CW));

    // UV rows: one UV pair per 2×2 block → CH/2 rows of CW bytes
    const int uvRow0 = CY / 2;
    for (int row = 0; row < CH / 2; row++)
        std::memcpy(dstUV + row * CW,
                    srcUV + (uvRow0 + row) * FW + CX,
                    static_cast<size_t>(CW));

    // ── 2. NV12 → packed RGB ─────────────────────────────────────────────────
    // ncnn::yuv420sp2rgb_nv12 expects compact NV12 and writes packed RGB.
    std::vector<uint8_t> rgb(static_cast<size_t>(CW * CH * 3));
    ncnn::yuv420sp2rgb_nv12(cropNV12.data(), CW, CH, rgb.data());

    // ── 3. Encode and write JPEG ──────────────────────────────────────────────
    // stbi_write_jpg: (path, w, h, channels, data, quality)
    int ok = stbi_write_jpg(job.outPath.c_str(),
                            CW, CH, 3,
                            rgb.data(),
                            m_cfg.jpegQuality);
    if (!ok) {
        fprintf(stderr, "CropSaver: stbi_write_jpg failed: %s\n",
                job.outPath.c_str());
        return false;
    }

    printf("CropSaver: [track %d] %s  conf=%.2f  %dx%d  %s\n",
           job.trackId, job.className.c_str(), job.confidence,
           CW, CH, job.outPath.c_str());
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  printStats()
// ─────────────────────────────────────────────────────────────────────────────

void CropSaver::printStats() const {
    printf("CropSaver: saved=%llu  dropped=%llu  pending=%d\n",
           (unsigned long long)m_cropsSaved.load(),
           (unsigned long long)m_cropsDropped.load(),
           m_pending.load());
}
