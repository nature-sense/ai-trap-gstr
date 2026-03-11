#include "libcamera_capture.h"
#include "decoder.h"
#include "tracker.h"
#include "persistence.h"
#include "crop_saver.h"
#include "mjpeg_streamer.h"
#include "config_loader.h"
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

static const char* className(int id) {
    return (id == 0) ? CLASS_NAMES[0] : "?";
}

// ─────────────────────────────────────────────────────────────────────────────
//  Signal handling
// ─────────────────────────────────────────────────────────────────────────────

static std::atomic<bool> g_stop{false};
static void onSignal(int) { g_stop = true; }

// ─────────────────────────────────────────────────────────────────────────────
//  main
//
//  Usage: ./yolo_libcamera [config.toml]
//    config.toml  path to TOML config file  (default: trap_config.toml)
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    const char* configPath = argc > 1 ? argv[1] : "trap_config.toml";

    printf("═══════════════════════════════════════════════\n");
    printf("  YOLO11n  Pi 5  libcamera + ncnn + ByteTracker\n");
    printf("═══════════════════════════════════════════════\n");
    printf("  config : %s\n\n", configPath);

    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);

    // ── Load configuration ────────────────────────────────────────────────────

    TrapConfig cfg;
    if (!loadConfig(configPath, cfg))
        fprintf(stderr, "Warning: config file not found — using built-in defaults\n\n");
    printConfig(cfg);

    // ── Components ────────────────────────────────────────────────────────────

    SqliteWriter  db;
    YoloDecoder   decoder(cfg.decoder);
    ByteTracker   tracker(cfg.tracker);
    CropSaver     crops;
    MjpegStreamer streamer;

    // ── Database ──────────────────────────────────────────────────────────────

    try {
        db.open(cfg.dbPath.c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: cannot open database: %s\n", e.what());
        return 1;
    }

    // ── Crop saver ────────────────────────────────────────────────────────────

    try {
        crops.open(cfg.crops);
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: cannot open crop saver: %s\n", e.what());
        db.close();
        return 1;
    }

    // ── MJPEG streamer ────────────────────────────────────────────────────────

    try {
        streamer.open(cfg.stream);
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
    net.opt.use_fp16_packed     = false;
    net.opt.use_fp16_storage    = false;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_packing_layout  = true;
    net.opt.lightmode           = true;

    if (net.load_param(cfg.modelParam.c_str()) != 0) {
        fprintf(stderr, "Fatal: cannot load %s\n", cfg.modelParam.c_str());
        streamer.close();
        crops.close();
        db.close();
        return 1;
    }
    if (net.load_model(cfg.modelBin.c_str()) != 0) {
        fprintf(stderr, "Fatal: cannot load %s\n", cfg.modelBin.c_str());
        streamer.close();
        crops.close();
        db.close();
        return 1;
    }
    printf("Model loaded: %s\n\n", cfg.modelParam.c_str());

    // ── Autodetect blob names from .param file ────────────────────────────────
    //
    // ncnn .param line format:
    //   LayerType  layer_name  num_in  num_out  [in_blobs...]  [out_blobs...]  [key=value...]
    //
    // Blob names never contain '='.  Key=value parameters always do.
    // So the last token WITHOUT '=' on a line is the last output blob name.
    //
    // Exported names vary by ultralytics version:
    //   older → "images" / "output"
    //   newer → "in0"    / "out0"

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

        FILE* f = fopen(cfg.modelParam.c_str(), "r");
        if (f) {
            char line[1024];
            while (fgets(line, sizeof(line), f)) {
                std::string s(line);
                while (!s.empty() && (s.back() == '\n' || s.back() == '\r' ||
                                      s.back() == ' '  || s.back() == '\t'))
                    s.pop_back();

                if (strncmp(line, "Input", 5) == 0 && inputName.empty())
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

    // ── LibcameraCapture ──────────────────────────────────────────────────────

    LibcameraCapture cam;

    cam.setErrorCallback([](const std::string& msg) {
        fprintf(stderr, "[camera] %s\n", msg.c_str());
    });

    // ── Per-frame callback ────────────────────────────────────────────────────

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

        // Persist confirmed tracks to database
        std::vector<DetectionRecord> records;
        for (const auto& t : tracked) {
            if (!t.confirmed) continue;
            DetectionRecord r{};
            r.frameId     = frame.frameId;
            r.timestampUs = frame.timestampNs / 1000;  // ns → µs
            r.trackId     = t.trackId;
            r.classId     = t.classId;
            r.label       = className(t.classId);
            r.x1          = t.x1;
            r.y1          = t.y1;
            r.x2          = t.x2;
            r.y2          = t.y2;
            r.confidence  = t.score;
            r.frameWidth  = frame.width;
            r.frameHeight = frame.height;
            records.push_back(r);
        }
        if (!records.empty())
            db.writeBatch(records);

        // Stream frame to MJPEG clients
        streamer.pushFrame(frame.nv12, frame.width, frame.height);

        // Save best-confidence JPEG crop per confirmed track
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
        cam.open(cfg.camera);
        cam.start();
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: %s\n", e.what());
        streamer.close();
        crops.close();
        db.close();
        return 1;
    }

    printf("Running — Ctrl+C to stop\n\n");

    // ── Main loop — print stats every 10 s ───────────────────────────────────

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
