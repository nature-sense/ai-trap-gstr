#include "libcamera_capture.h"
#include "decoder.h"
#include "tracker.h"
#include "persistence.h"
#include "crop_saver.h"
#include "mjpeg_streamer.h"
#include "sse_server.h"
#include "http_server.h"
#include "sync_manager.h"
#include "trap_events.h"
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
static const char* className(int id) { return id == 0 ? CLASS_NAMES[0] : "?"; }

// ─────────────────────────────────────────────────────────────────────────────
//  Signal handling
// ─────────────────────────────────────────────────────────────────────────────

static std::atomic<bool> g_stop{false};
static std::atomic<int>  g_stopSignal{0};
static void onSignal(int sig) { g_stopSignal = sig; g_stop = true; }

// ─────────────────────────────────────────────────────────────────────────────
//  main
//
//  Usage: ./yolo_libcamera [config.toml]
// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    const char* configPath = argc > 1 ? argv[1] : "trap_config.toml";

    printf("═══════════════════════════════════════════════\n");
    printf("  YOLO11n  Pi 5  libcamera + ncnn + ByteTracker\n");
    printf("═══════════════════════════════════════════════\n");
    printf("  config : %s\n\n", configPath);

    std::signal(SIGINT,  onSignal);
    std::signal(SIGTERM, onSignal);
    std::signal(SIGPIPE, SIG_IGN);  // prevent exit on client disconnect

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
    SseServer     sse;
    HttpServer    http;
    SyncManager   sync;

    // ── Database ──────────────────────────────────────────────────────────────

    try {
        db.open(cfg.dbPath.c_str());
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: cannot open database: %s\n", e.what());
        return 1;
    }

    // ── Sync manager ─────────────────────────────────────────────────────────

    sync.init(db.rawDb(), cfg.crops.outputDir);

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
        fprintf(stderr, "Fatal: cannot start MJPEG streamer: %s\n", e.what());
        crops.close();
        db.close();
        return 1;
    }

    // ── SSE server ────────────────────────────────────────────────────────────

    try {
        sse.open(cfg.sse);
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: cannot start SSE server: %s\n", e.what());
        streamer.close();
        crops.close();
        db.close();
        return 1;
    }

    // ── HTTP API server ───────────────────────────────────────────────────────

    // fps is updated by the main stats loop; the HTTP server reads it on demand.
    float currentFps = 0.f;

    // Capture flag — true means inference/tracking/saving is active.
    // The MJPEG stream always runs regardless of this flag.
    std::atomic<bool> g_capturing{true};

    http.setLocationCallback([&](double lat, double lon) {
        cfg.gpsLat   = lat;
        cfg.gpsLon   = lon;
        cfg.gpsValid = true;
        // Live-update CropSaver EXIF template so new crops get GPS immediately
        auto& ep  = cfg.crops.exifTemplate;
        ep.lat    = lat;
        ep.lon    = lon;
        ep.hasGps = true;
        printf("main: GPS updated  lat=%.6f  lon=%.6f\n", lat, lon);
    });

    http.setThresholdCallback([&](float thresh) {
        DecoderConfig dc = decoder.config();
        dc.confThresh = thresh;
        decoder.setConfig(dc);
    });

    http.setCaptureCallback([&](bool active) {
        g_capturing = active;
        sse.pushEvent(TrapEvents::captureState(active));
    });

    // Populate EXIF template from TrapConfig (GPS, trapId, location)
    {
        auto& ep          = cfg.crops.exifTemplate;
        ep.trapId         = cfg.trapId;
        ep.trapLocation   = cfg.trapLocation;
        ep.hasGps         = cfg.gpsValid;
        ep.lat            = cfg.gpsLat;
        ep.lon            = cfg.gpsLon;
        ep.altM           = cfg.gpsAltM;
    }

    crops.setSavedCallback([&](int trackId, int classId,
                                  const std::string& cls,
                                  float conf, const std::string& path,
                                  int w, int h) {
        // Extract just the filename from the full path
        std::string filename = path;
        auto slash = path.rfind('/');
        if (slash != std::string::npos) filename = path.substr(slash + 1);

        sse.pushEvent(TrapEvents::cropSaved(
            trackId, cls.c_str(), conf, filename.c_str(), w, h));
    });

    http.setAfTriggerCallback([&]() {
        // Trigger a one-shot AF scan.  LibcameraCapture will pick this up
        // on the next call to applyControls() — add that method if needed.
        printf("AF trigger requested via API\n");
    });

    try {
        http.open(cfg.http, &db, &sse, &sync, &currentFps, &g_capturing);
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: cannot start HTTP server: %s\n", e.what());
        sse.close();
        streamer.close();
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
        http.close(); sse.close(); streamer.close(); crops.close(); db.close();
        return 1;
    }
    if (net.load_model(cfg.modelBin.c_str()) != 0) {
        fprintf(stderr, "Fatal: cannot load %s\n", cfg.modelBin.c_str());
        http.close(); sse.close(); streamer.close(); crops.close(); db.close();
        return 1;
    }
    printf("Model loaded: %s\n\n", cfg.modelParam.c_str());

    // ── Autodetect blob names from .param file ────────────────────────────────

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
                while (!s.empty() && (s.back()=='\n'||s.back()=='\r'||
                                      s.back()==' ' ||s.back()=='\t'))
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

    // ── LibcameraCapture ──────────────────────────────────────────────────────

    LibcameraCapture cam;

    cam.setErrorCallback([](const std::string& msg) {
        fprintf(stderr, "[camera] %s\n", msg.c_str());
    });

    // Track last AF state to include in health events
    std::atomic<int>   lastAfState{0};
    std::atomic<float> lastLensPos{0.f};

    // ── Per-frame callback ────────────────────────────────────────────────────

    cam.setCallback([&](const CaptureFrame& frame) {

        lastAfState.store(frame.afState,      std::memory_order_relaxed);
        lastLensPos.store(frame.lensPosition, std::memory_order_relaxed);

        // MJPEG stream always runs regardless of capture state
        streamer.pushFrame(frame.nv12, frame.width, frame.height);

        // Inference/tracking/saving only when capturing
        if (!g_capturing.load(std::memory_order_relaxed)) return;

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

        // Per-track actions
        for (const auto& t : tracked) {
            if (!t.confirmed) continue;

            // Log and fire SSE only on first confirmation
            if (t.age == cfg.tracker.minHits) {
                printf("  [%4d] %-16s %3.0f%%  "
                       "(%5.0f,%5.0f)-(%5.0f,%5.0f)  frame=%llu\n",
                       t.trackId, className(t.classId), t.score * 100.f,
                       t.x1, t.y1, t.x2, t.y2,
                       (unsigned long long)frame.frameId);

                sse.pushEvent(TrapEvents::detection(
                    t.trackId, className(t.classId), t.score,
                    t.x1, t.y1, t.x2, t.y2, frame.frameId));
            }
        }

        // Persist to database — write once on first confirmation only.
        // Stationary tracks do not generate a row on every frame.
        {
            std::vector<DetectionRecord> records;
            for (const auto& t : tracked) {
                if (!t.confirmed) continue;
                if (t.age != cfg.tracker.minHits) continue;  // first confirmation only
                DetectionRecord r{};
                r.frameId     = frame.frameId;
                r.timestampUs = frame.timestampNs / 1000;
                r.trackId     = t.trackId;
                r.classId     = t.classId;
                r.label       = className(t.classId);
                r.x1          = t.x1; r.y1 = t.y1;
                r.x2          = t.x2; r.y2 = t.y2;
                r.confidence  = t.score;
                r.frameWidth  = frame.width;
                r.frameHeight = frame.height;
                records.push_back(r);
            }
            if (!records.empty())
                db.writeBatch(records);
        }

        // Save best-confidence JPEG crop per confirmed track
        for (const auto& t : tracked) {
            if (!t.confirmed) continue;
            crops.submit(frame.nv12,
                         frame.width, frame.height,
                         t.trackId, t.classId,
                         className(t.classId),
                         t.score,
                         t.x1, t.y1, t.x2, t.y2,
                         static_cast<int64_t>(frame.timestampNs / 1000));
        }
    });

    // ── Open and start camera ─────────────────────────────────────────────────

    try {
        cam.open(cfg.camera);
        cam.start();
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal: %s\n", e.what());
        http.close(); sse.close(); streamer.close(); crops.close(); db.close();
        return 1;
    }

    printf("Running — Ctrl+C to stop\n");
    printf("  MJPEG stream : http://192.168.5.1:%d/stream\n", cfg.stream.port);
    printf("  SSE events   : http://192.168.5.1:%d/api/events\n", cfg.sse.port);
    printf("  REST API     : http://192.168.5.1:%d/api/status\n\n", cfg.http.port);

    // ── Main loop — stats every 30 s ─────────────────────────────────────────

    auto lastStats  = std::chrono::steady_clock::now();
    auto startTime  = lastStats;
    uint64_t lastFrameId = 0;

    while (!g_stop.load() && cam.isRunning()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        auto now = std::chrono::steady_clock::now();
        if (now - lastStats >= std::chrono::seconds(30)) {
            double elapsed = std::chrono::duration<double>(now - lastStats).count();

            cam.printStats();

            DetectionStats ds = db.getStats();
            double dbMb = static_cast<double>(db.fileSizeBytes()) / 1e6;
            long long uptimeS = std::chrono::duration_cast<std::chrono::seconds>(
                                    now - startTime).count();

            printf("  DB rows=%lld  tracks=%lld  size=%.1f MB\n\n",
                   (long long)ds.totalDetections,
                   (long long)ds.uniqueTracks,
                   dbMb);
            crops.printStats();
            streamer.printStats();
            sse.printStats();
            http.printStats();

            // Push SSE stats + health events
            sse.pushEvent(TrapEvents::stats(
                ds.totalDetections, uptimeS, currentFps,
                ds.uniqueTracks, dbMb));

            sse.pushEvent(TrapEvents::health(
                TrapEvents::readCpuTemp(),
                lastAfState.load(std::memory_order_relaxed),
                lastLensPos.load(std::memory_order_relaxed)));

            lastStats = now;
            (void)elapsed;
        }
    }

    // ── Shutdown ──────────────────────────────────────────────────────────────

    int sig = g_stopSignal.load();
    if (sig)
        printf("\nCaught signal %d (%s) — shutting down...\n", sig, strsignal(sig));
    else
        printf("\nCamera stopped unexpectedly — shutting down...\n");
    cam.stop();
    http.close();
    sse.close();
    streamer.close();
    crops.flush();
    crops.close();
    db.flush();
    db.close();
    printf("Done.\n");
    return 0;
}
