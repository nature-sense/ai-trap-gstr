#include "src/v4l2_capture.h"
#include "src/decoder.h"
#include "src/tracker.h"
#include "src/persistence.h"
#include "src/crop_saver.h"
#include "src/mjpeg_streamer.h"
#include "ncnn/net.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

// ─────────────────────────────────────────────────────────────────────────────
//  Class names
// ─────────────────────────────────────────────────────────────────────────────

static const char* CLASS_NAMES[1] = { "insect" };
static const char* className(int id) { return (id == 0) ? CLASS_NAMES[0] : "?"; }

// ─────────────────────────────────────────────────────────────────────────────
//  Signal handling
// ─────────────────────────────────────────────────────────────────────────────

static std::atomic<bool> g_stop{false};
static void onSignal(int) { g_stop = true; }

// ─────────────────────────────────────────────────────────────────────────────
//  main
//
//  Usage: ./yolo_v4l2 [param] [bin] [db] [crops-dir] [device]
//    param      ncnn .param file      (default yolo11n.param)
//    bin        ncnn .bin file        (default yolo11n.bin)
//    db         SQLite output path    (default detections.db)
//    crops-dir  directory for crops   (default crops)
//    device     V4L2 device node      (default /dev/video0)
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    const char* paramFile = argc > 1 ? argv[1] : "yolo11n.param";
    const char* binFile   = argc > 2 ? argv[2] : "yolo11n.bin";
    const char* dbPath    = argc > 3 ? argv[3] : "detections.db";
    const char* cropsDir  = argc > 4 ? argv[4] : "crops";
    const char* device    = argc > 5 ? argv[5] : "/dev/video0";

    printf("══════════════════════════════════════════════════════════\n");
    printf("  YOLO11n  Luckfox Pico Zero  IMX415 V4L2 + ncnn + ByteTracker\n");
    printf("══════════════════════════════════════════════════════════\n");
    printf("  model  : %s / %s\n  db     : %s\n"
           "  crops  : %s\n  device : %s\n\n",
           paramFile, binFile, dbPath, cropsDir, device);

    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);
    std::signal(SIGPIPE, SIG_IGN);  // prevent exit on client disconnect

    // ── Components ────────────────────────────────────────────────────────────

    SqliteWriter db;

    DecoderConfig decCfg;
    decCfg.numClasses  = 1;
    decCfg.confThresh  = 0.45f;
    decCfg.nmsThresh   = 0.45f;
    decCfg.modelWidth  = 320;
    decCfg.modelHeight = 320;
    YoloDecoder decoder(decCfg);

    ByteTracker tracker;

    // ── Database ──────────────────────────────────────────────────────────────
    try {
        db.open(dbPath);
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: cannot open database: %s\n", e.what());
        return 1;
    }

    // ── Crop saver ────────────────────────────────────────────────────────────
    CropSaver crops;
    CropSaverConfig cropCfg;
    cropCfg.outputDir     = cropsDir;
    cropCfg.jpegQuality   = 90;
    cropCfg.minConfidence = 0.50f;
    cropCfg.maxQueueDepth = 16;

    try {
        crops.open(cropCfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: cannot open crop saver: %s\n", e.what());
        db.close();
        return 1;
    }

    // ── MJPEG streamer ────────────────────────────────────────────────────────
    MjpegStreamer streamer;
    MjpegStreamerConfig streamCfg;
    streamCfg.port         = 9000;
    streamCfg.streamWidth  = 640;
    streamCfg.streamHeight = 480;
    streamCfg.jpegQuality  = 75;

    try {
        streamer.open(streamCfg);
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: cannot start streamer: %s\n", e.what());
        crops.close();
        db.close();
        return 1;
    }

    // ── ncnn model ────────────────────────────────────────────────────────────
    ncnn::Net net;
    net.opt.num_threads         = 4;
    net.opt.use_vulkan_compute  = false;
    net.opt.use_fp16_packed     = true;
    net.opt.use_fp16_storage    = true;
    net.opt.use_fp16_arithmetic = true;
    net.opt.use_packing_layout  = true;
    net.opt.lightmode           = true;

    if (net.load_param(paramFile) != 0) {
        fprintf(stderr, "Fatal: cannot load %s\n", paramFile);
        db.close();
        return 1;
    }
    if (net.load_model(binFile) != 0) {
        fprintf(stderr, "Fatal: cannot load %s\n", binFile);
        db.close();
        return 1;
    }
    printf("Model loaded: %s\n\n", paramFile);

    // Autodetect blob names from .param file
    std::string inputName;
    std::string outputName;
    {
        auto lastBlobToken = [](const std::string& s) -> std::string {
            std::string last;
            std::istringstream ss(s);
            std::string tok;
            while (ss >> tok)
                if (tok.find('=') == std::string::npos)
                    last = tok;
            return last;
        };

        FILE* f = fopen(paramFile, "r");
        if (f) {
            char line[1024];
            while (fgets(line, sizeof(line), f)) {
                std::string s(line);
                while (!s.empty() && (s.back() == '\n' || s.back() == '\r' ||
                                      s.back() == ' '  || s.back() == '\t'))
                    s.pop_back();

                if (strncmp(line, "Input",   5) == 0 && inputName.empty())
                    inputName = lastBlobToken(s);
                if (strncmp(line, "Concat",  6) == 0 ||
                    strncmp(line, "Detect",  6) == 0 ||
                    strncmp(line, "Permute", 7) == 0 ||
                    strncmp(line, "Reshape", 7) == 0)
                    outputName = lastBlobToken(s);
            }
            fclose(f);
        }
        if (inputName.empty())  inputName  = "images";
        if (outputName.empty()) outputName = "output";
        printf("Input layer:  \"%s\"\n", inputName.c_str());
        printf("Output layer: \"%s\"\n\n", outputName.c_str());
    }

    // ── V4L2Capture — Luckfox IMX415-98 IR-CUT Camera ────────────────────────
    //
    // IMX415 sensor modes on RV1103 (Luckfox Pico Zero):
    //   3864 x 2192  @  7 fps  — full 4K (ISP bandwidth limited)
    //   1920 x 1080  @ 30 fps  — 1080p  ← recommended
    //   1280 x  720  @ 60 fps  — high framerate
    //
    // Setup (once on board):
    //   v4l2-ctl -d /dev/video0 --list-formats-ext   # verify NV12 offered
    //   media-ctl --print-topology                   # confirm rkisp_mainpath
    //
    // IR-CUT: the cut filter is controlled externally (GPIO or rkaiq daemon).
    // Day/night switching is handled outside this capture path.

    V4L2Config camCfg;
    camCfg.device        = device;
    camCfg.captureWidth  = 1920;
    camCfg.captureHeight = 1080;
    camCfg.framerate     = 30;
    camCfg.bufferCount   = 4;
    camCfg.modelWidth    = 320;
    camCfg.modelHeight   = 320;

    V4L2Capture cam;

    cam.setErrorCallback([](const std::string& msg) {
        fprintf(stderr, "[camera] %s\n", msg.c_str());
    });

    // ── Per-frame callback ────────────────────────────────────────────────────
    // Identical to main_libcamera.cpp — CaptureFrame is the same struct.

    cam.setCallback([&](const CaptureFrame& frame) {

        // Inference
        ncnn::Extractor ex = net.create_extractor();
        ex.input(inputName.c_str(), frame.modelInput);

        ncnn::Mat output;
        if (ex.extract(outputName.c_str(), output) != 0) {
            fprintf(stderr, "[warn] extract() failed\n");
            return;
        }

        // Decode
        std::vector<Detection> dets = decoder.decode(
            output,
            frame.width, frame.height,
            frame.scale, frame.padLeft, frame.padTop);

        // Track
        std::vector<TrackedObject> tracked = tracker.update(dets);

        // Print confirmed detections
        for (const auto& t : tracked) {
            if (!t.confirmed) continue;
            printf("  [%4d] %-16s %3.0f%%  "
                   "(%5.0f,%5.0f)-(%5.0f,%5.0f)  age=%d\n",
                   t.trackId, className(t.classId), t.score * 100.f,
                   t.x1, t.y1, t.x2, t.y2, t.age);
        }
        if (!tracked.empty())
            printf("  frame=%-6llu  dets=%-3zu  tracks=%zu\n\n",
                   (unsigned long long)frame.frameId,
                   dets.size(), tracked.size());

        // Persist
        std::vector<DetectionRecord> records;
        for (const auto& t : tracked) {
            if (!t.confirmed) continue;
            DetectionRecord r{};
            r.frameId     = frame.frameId;
            r.timestampUs = frame.timestampNs / 1000;
            r.trackId     = t.trackId;
            r.classId     = t.classId;
            r.label       = className(t.classId);
            r.x1 = t.x1; r.y1 = t.y1;
            r.x2 = t.x2; r.y2 = t.y2;
            r.confidence  = t.score;
            r.frameWidth  = frame.width;
            r.frameHeight = frame.height;
            records.push_back(r);
        }
        if (!records.empty())
            db.writeBatch(records);

        // Stream
        streamer.pushFrame(frame.nv12, frame.width, frame.height);

        // Crops
        for (const auto& t : tracked) {
            if (!t.confirmed) continue;
            crops.submit(frame.nv12,
                         frame.width, frame.height,
                         t.trackId, t.classId,
                         className(t.classId),
                         t.score,
                         t.x1, t.y1, t.x2, t.y2);
        }
    });

    // ── Open and start ────────────────────────────────────────────────────────
    try {
        cam.open(camCfg);
        cam.start();
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: %s\n", e.what());
        db.close();
        return 1;
    }

    printf("Running — Ctrl+C to stop\n\n");

    // ── Main loop ─────────────────────────────────────────────────────────────
    auto lastStats = std::chrono::steady_clock::now();

    while (!g_stop.load() && cam.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        auto now = std::chrono::steady_clock::now();
        if (now - lastStats >= std::chrono::seconds(10)) {
            cam.printStats();
            DetectionStats ds = db.getStats();
            printf("  DB rows=%lld  tracks=%lld  size=%.1f MB\n\n",
                   (long long)ds.totalDetections,
                   (long long)ds.uniqueTracks,
                   (double)db.fileSizeBytes() / 1e6);
            crops.printStats();
            streamer.printStats();
            lastStats = now;
        }
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────
    printf("\nShutting down...\n");
    cam.stop();
    streamer.close();
    crops.flush();
    crops.close();
    db.flush();
    db.close();
    printf("Done.\n");
    return 0;
}
