#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  config_loader.h  —  Minimal TOML config loader (header-only, no dependencies)
//
//  Supports the subset of TOML used by trap_config.toml:
//    [section]          — section headers
//    key = value        — string, integer, float, bool values
//    # comment          — line comments (also inline after values)
//
//  Usage:
//    TrapConfig cfg;
//    if (!loadConfig("trap_config.toml", cfg)) { /* handle error */ }
// ─────────────────────────────────────────────────────────────────────────────

#include "decoder.h"
#include "libcamera_capture.h"
#include "tracker.h"
#include "crop_saver.h"
#include "mjpeg_streamer.h"
#include "sse_server.h"
#include "http_server.h"

#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <string>

// ─────────────────────────────────────────────────────────────────────────────
//  TrapConfig  — all runtime-configurable parameters in one place
// ─────────────────────────────────────────────────────────────────────────────

struct TrapConfig {
    // [trap]
    std::string trapId       = "trap_001";
    std::string trapLocation = "";
    double      gpsLat       = 0.0;    // decimal degrees (negative = South)
    double      gpsLon       = 0.0;    // decimal degrees (negative = West)
    double      gpsAltM      = 0.0;    // metres above sea level
    bool        gpsValid     = false;  // true once lat/lon have been set

    // [model]
    std::string modelParam   = "yolo11n.param";
    std::string modelBin     = "yolo11n.bin";

    // [database]
    std::string dbPath       = "detections.db";

    // Component configs — populated by loadConfig()
    DecoderConfig       decoder;
    ByteTrackerConfig   tracker;
    LibcameraConfig     camera;
    CropSaverConfig     crops;
    MjpegStreamerConfig stream;
    SseConfig           sse;
    HttpServerConfig    http;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Parsing helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static std::string stripComment(const std::string& s) {
    bool inStr = false;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '"') inStr = !inStr;
        if (!inStr && s[i] == '#') return s.substr(0, i);
    }
    return s;
}

static std::string unquote(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

static float toFloat(const std::string& s) { return std::stof(s); }
static int   toInt  (const std::string& s) { return std::stoi(s); }

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
//  loadConfig
// ─────────────────────────────────────────────────────────────────────────────

static bool loadConfig(const char* path, TrapConfig& cfg) {
    FILE* f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "loadConfig: cannot open \"%s\": %s\n",
                path, strerror(errno));
        return false;
    }

    char        lineBuf[512];
    std::string section;
    int         lineNo = 0;

    while (fgets(lineBuf, sizeof(lineBuf), f)) {
        lineNo++;
        std::string line = detail::trim(detail::stripComment(lineBuf));
        if (line.empty()) continue;

        if (line.front() == '[') {
            size_t close = line.find(']');
            if (close == std::string::npos) {
                fprintf(stderr, "loadConfig:%d: malformed section\n", lineNo);
                continue;
            }
            section = detail::trim(line.substr(1, close - 1));
            continue;
        }

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key = detail::trim(line.substr(0, eq));
        std::string val = detail::trim(detail::unquote(detail::trim(line.substr(eq + 1))));

        try {
            if (section == "trap") {
                if      (key == "id")       cfg.trapId       = val;
                else if (key == "location") cfg.trapLocation = val;
                else if (key == "lat") {
                    cfg.gpsLat   = std::stod(val);
                    cfg.gpsValid = true;
                }
                else if (key == "lon") {
                    cfg.gpsLon   = std::stod(val);
                    cfg.gpsValid = true;
                }
                else if (key == "alt_m")  cfg.gpsAltM = std::stod(val);

            } else if (section == "model") {
                if      (key == "param")       cfg.modelParam          = val;
                else if (key == "bin")         cfg.modelBin            = val;
                else if (key == "width")       cfg.decoder.modelWidth  = detail::toInt(val);
                else if (key == "height")      cfg.decoder.modelHeight = detail::toInt(val);
                else if (key == "num_classes") cfg.decoder.numClasses  = detail::toInt(val);
                else if (key == "format") {
                    if      (val == "anchor_grid") cfg.decoder.format = YoloFormat::AnchorGrid;
                    else if (val == "end_to_end")  cfg.decoder.format = YoloFormat::EndToEnd;
                    else                           cfg.decoder.format = YoloFormat::Auto;
                }
                else if (key == "pre_applied_sigmoid")
                    cfg.decoder.preAppliedSigmoid = (val == "true" || val == "1");

            } else if (section == "detection") {
                if      (key == "conf_threshold") cfg.decoder.confThresh = detail::toFloat(val);
                else if (key == "nms_threshold")  cfg.decoder.nmsThresh  = detail::toFloat(val);
                else if (key == "min_box_width")    cfg.decoder.minBoxWidth    = detail::toFloat(val);
                else if (key == "min_box_height")   cfg.decoder.minBoxHeight   = detail::toFloat(val);
                else if (key == "max_aspect_ratio")  cfg.decoder.maxAspectRatio  = detail::toFloat(val);
                else if (key == "max_box_area_ratio") cfg.decoder.maxBoxAreaRatio = detail::toFloat(val);

            } else if (section == "tracker") {
                if      (key == "high_threshold") cfg.tracker.highThresh = detail::toFloat(val);
                else if (key == "low_threshold")  cfg.tracker.lowThresh  = detail::toFloat(val);
                else if (key == "iou_threshold")  cfg.tracker.iouThresh  = detail::toFloat(val);
                else if (key == "min_hits")        cfg.tracker.minHits    = detail::toInt(val);
                else if (key == "max_missed")      cfg.tracker.maxMissed  = detail::toInt(val);

            } else if (section == "camera") {
                if      (key == "camera_id")      cfg.camera.cameraId      = val;
                else if (key == "tuning_file")    cfg.camera.tuningFile    = val;
                else if (key == "capture_width")  cfg.camera.captureWidth  = detail::toInt(val);
                else if (key == "capture_height") cfg.camera.captureHeight = detail::toInt(val);
                else if (key == "framerate")      cfg.camera.framerate     = detail::toInt(val);
                else if (key == "buffer_count")   cfg.camera.bufferCount   = detail::toInt(val);
                else if (key == "brightness")     cfg.camera.brightness    = detail::toFloat(val);
                else if (key == "contrast")       cfg.camera.contrast      = detail::toFloat(val);
                else if (key == "saturation")     cfg.camera.saturation    = detail::toFloat(val);
                else if (key == "sharpness")      cfg.camera.sharpness     = detail::toFloat(val);

            } else if (section == "autofocus") {
                if      (key == "mode")           cfg.camera.afMode       = detail::toInt(val);
                else if (key == "range")          cfg.camera.afRange      = detail::toInt(val);
                else if (key == "speed")          cfg.camera.afSpeed      = detail::toInt(val);
                else if (key == "lens_position")  cfg.camera.lensPosition = detail::toFloat(val);
                else if (key == "window_x")       cfg.camera.afWindowX    = detail::toInt(val);
                else if (key == "window_y")       cfg.camera.afWindowY    = detail::toInt(val);
                else if (key == "window_w")       cfg.camera.afWindowW    = detail::toInt(val);
                else if (key == "window_h")       cfg.camera.afWindowH    = detail::toInt(val);

            } else if (section == "crops") {
                if      (key == "output_dir")      cfg.crops.outputDir      = val;
                else if (key == "jpeg_quality")    cfg.crops.jpegQuality    = detail::toInt(val);
                else if (key == "min_confidence")  cfg.crops.minConfidence  = detail::toFloat(val);
                else if (key == "min_confidence_delta") cfg.crops.minConfidenceDelta = detail::toFloat(val);
                else if (key == "max_saves_per_track")  cfg.crops.maxSavesPerTrack  = detail::toInt(val);
                else if (key == "max_queue_depth")      cfg.crops.maxQueueDepth     = detail::toInt(val);

            } else if (section == "stream") {
                if      (key == "port")         cfg.stream.port          = detail::toInt(val);
                else if (key == "width")        cfg.stream.streamWidth   = detail::toInt(val);
                else if (key == "height")       cfg.stream.streamHeight  = detail::toInt(val);
                else if (key == "jpeg_quality") cfg.stream.jpegQuality   = detail::toInt(val);

            } else if (section == "sse") {
                if      (key == "port")            cfg.sse.port           = detail::toInt(val);
                else if (key == "max_clients")     cfg.sse.maxClients     = detail::toInt(val);
                else if (key == "max_queue_depth") cfg.sse.maxQueueDepth  = detail::toInt(val);

            } else if (section == "api") {
                if (key == "port") cfg.http.port = detail::toInt(val);

            } else if (section == "database") {
                if (key == "path") cfg.dbPath = val;
            }

        } catch (const std::exception& e) {
            fprintf(stderr, "loadConfig:%d: [%s] %s = \"%s\": %s\n",
                    lineNo, section.c_str(), key.c_str(), val.c_str(), e.what());
        }
    }

    fclose(f);

    // Mirror model size into camera config (used by preprocess())
    cfg.camera.modelWidth  = cfg.decoder.modelWidth;
    cfg.camera.modelHeight = cfg.decoder.modelHeight;

    // Mirror trap identity into HttpServerConfig
    cfg.http.cropsDir     = cfg.crops.outputDir;
    cfg.http.trapId       = cfg.trapId;
    cfg.http.trapLocation = cfg.trapLocation;

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  printConfig
// ─────────────────────────────────────────────────────────────────────────────

static void printConfig(const TrapConfig& cfg) {
    printf("┌─ Configuration ───────────────────────────────────────────\n");
    printf("│  Trap          %s  (%s)\n",
           cfg.trapId.c_str(), cfg.trapLocation.c_str());
    printf("│  Model         %s\n", cfg.modelParam.c_str());
    printf("│  Format        %s\n",
           cfg.decoder.format == YoloFormat::AnchorGrid ? "anchor_grid" :
           cfg.decoder.format == YoloFormat::EndToEnd   ? "end_to_end"  : "auto");
    printf("│  Classes       %d    conf=%.2f  nms=%.2f\n",
           cfg.decoder.numClasses,
           cfg.decoder.confThresh,
           cfg.decoder.nmsThresh);
    printf("│  Camera        %dx%d @ %d fps  bufs=%d\n",
           cfg.camera.captureWidth, cfg.camera.captureHeight,
           cfg.camera.framerate, cfg.camera.bufferCount);
    printf("│  AF mode       %d  range=%d  speed=%d  lens=%.1f D\n",
           cfg.camera.afMode, cfg.camera.afRange,
           cfg.camera.afSpeed, cfg.camera.lensPosition);
    printf("│  Tracker       high=%.2f  low=%.2f  iou=%.2f  hits=%d  missed=%d\n",
           cfg.tracker.highThresh, cfg.tracker.lowThresh,
           cfg.tracker.iouThresh, cfg.tracker.minHits, cfg.tracker.maxMissed);
    printf("│  Crops         %s  q=%d  min_conf=%.2f\n",
           cfg.crops.outputDir.c_str(),
           cfg.crops.jpegQuality,
           cfg.crops.minConfidence);
    printf("│  Stream (MJPEG)port=%d  %dx%d  q=%d\n",
           cfg.stream.port,
           cfg.stream.streamWidth, cfg.stream.streamHeight,
           cfg.stream.jpegQuality);
    printf("│  SSE           port=%d  max_clients=%d\n",
           cfg.sse.port, cfg.sse.maxClients);
    printf("│  API           port=%d\n", cfg.http.port);
    printf("│  Database      %s\n", cfg.dbPath.c_str());
    printf("└───────────────────────────────────────────────────────────\n\n");
}
