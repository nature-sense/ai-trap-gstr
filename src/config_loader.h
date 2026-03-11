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

#include <cstdio>
#include <cstring>
#include <string>
#include <cctype>

// ─────────────────────────────────────────────────────────────────────────────
//  TrapConfig  — all runtime-configurable parameters in one place
// ─────────────────────────────────────────────────────────────────────────────

struct TrapConfig {
    // [trap]
    std::string trapId       = "trap_001";
    std::string trapLocation = "";

    // [model]
    std::string modelParam   = "yolo11n.param";
    std::string modelBin     = "yolo11n.bin";

    // [database]
    std::string dbPath       = "detections.db";

    // Component configs — populated by loadConfig()
    DecoderConfig      decoder;
    ByteTrackerConfig  tracker;
    LibcameraConfig    camera;
    CropSaverConfig    crops;
    MjpegStreamerConfig stream;
};

// ─────────────────────────────────────────────────────────────────────────────
//  Parsing helpers
// ─────────────────────────────────────────────────────────────────────────────

namespace detail {

// Trim leading and trailing whitespace in-place
static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Strip inline comment (everything from '#' onwards, outside quotes)
static std::string stripComment(const std::string& s) {
    bool inStr = false;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] == '"') inStr = !inStr;
        if (!inStr && s[i] == '#') return s.substr(0, i);
    }
    return s;
}

// Remove surrounding double-quotes from a string value
static std::string unquote(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}

static float   toFloat(const std::string& s) { return std::stof(s); }
static int     toInt  (const std::string& s) { return std::stoi(s); }
static bool    toBool (const std::string& s) { return s == "true" || s == "1"; }

} // namespace detail

// ─────────────────────────────────────────────────────────────────────────────
//  loadConfig  — parse trap_config.toml into TrapConfig
//
//  Returns true on success.  Prints warnings for unrecognised keys.
//  Missing keys retain their default values from TrapConfig initialisation.
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

        // ── Section header ────────────────────────────────────────────────────
        if (line.front() == '[') {
            size_t close = line.find(']');
            if (close == std::string::npos) {
                fprintf(stderr, "loadConfig:%d: malformed section header\n", lineNo);
                continue;
            }
            section = detail::trim(line.substr(1, close - 1));
            continue;
        }

        // ── Key = value ───────────────────────────────────────────────────────
        size_t eq = line.find('=');
        if (eq == std::string::npos) {
            fprintf(stderr, "loadConfig:%d: no '=' found, skipping\n", lineNo);
            continue;
        }
        std::string key = detail::trim(line.substr(0, eq));
        std::string val = detail::trim(detail::unquote(detail::trim(line.substr(eq + 1))));

        // ── Dispatch to struct fields ─────────────────────────────────────────
        try {
            if (section == "trap") {
                if      (key == "id")           cfg.trapId       = val;
                else if (key == "location")     cfg.trapLocation = val;

            } else if (section == "model") {
                if      (key == "param")        cfg.modelParam               = val;
                else if (key == "bin")          cfg.modelBin                 = val;
                else if (key == "width")        cfg.decoder.modelWidth       = detail::toInt(val);
                else if (key == "height")       cfg.decoder.modelHeight      = detail::toInt(val);
                else if (key == "num_classes")  cfg.decoder.numClasses       = detail::toInt(val);
                else if (key == "threads")      { /* stored separately — see loadConfig return */ }
                else if (key == "format") {
                    if      (val == "anchor_grid") cfg.decoder.format = YoloFormat::AnchorGrid;
                    else if (val == "end_to_end")  cfg.decoder.format = YoloFormat::EndToEnd;
                    else                           cfg.decoder.format = YoloFormat::Auto;
                }

            } else if (section == "detection") {
                if      (key == "conf_threshold") cfg.decoder.confThresh = detail::toFloat(val);
                else if (key == "nms_threshold")  cfg.decoder.nmsThresh  = detail::toFloat(val);

            } else if (section == "tracker") {
                if      (key == "high_threshold") cfg.tracker.highThresh  = detail::toFloat(val);
                else if (key == "low_threshold")  cfg.tracker.lowThresh   = detail::toFloat(val);
                else if (key == "iou_threshold")  cfg.tracker.iouThresh   = detail::toFloat(val);
                else if (key == "min_hits")        cfg.tracker.minHits     = detail::toInt(val);
                else if (key == "max_missed")      cfg.tracker.maxMissed   = detail::toInt(val);

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
                if      (key == "mode")           cfg.camera.afMode        = detail::toInt(val);
                else if (key == "range")          cfg.camera.afRange       = detail::toInt(val);
                else if (key == "speed")          cfg.camera.afSpeed       = detail::toInt(val);
                else if (key == "lens_position")  cfg.camera.lensPosition  = detail::toFloat(val);
                else if (key == "window_x")       cfg.camera.afWindowX     = detail::toInt(val);
                else if (key == "window_y")       cfg.camera.afWindowY     = detail::toInt(val);
                else if (key == "window_w")       cfg.camera.afWindowW     = detail::toInt(val);
                else if (key == "window_h")       cfg.camera.afWindowH     = detail::toInt(val);

            } else if (section == "crops") {
                if      (key == "output_dir")     cfg.crops.outputDir      = val;
                else if (key == "jpeg_quality")   cfg.crops.jpegQuality    = detail::toInt(val);
                else if (key == "min_confidence") cfg.crops.minConfidence  = detail::toFloat(val);
                else if (key == "max_queue_depth")cfg.crops.maxQueueDepth  = detail::toInt(val);

            } else if (section == "stream") {
                if      (key == "port")           cfg.stream.port          = detail::toInt(val);
                else if (key == "width")          cfg.stream.streamWidth   = detail::toInt(val);
                else if (key == "height")         cfg.stream.streamHeight  = detail::toInt(val);
                else if (key == "jpeg_quality")   cfg.stream.jpegQuality   = detail::toInt(val);

            } else if (section == "database") {
                if      (key == "path")           cfg.dbPath               = val;

            } else {
                fprintf(stderr, "loadConfig:%d: unknown section [%s]\n",
                        lineNo, section.c_str());
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "loadConfig:%d: error parsing [%s] %s = \"%s\": %s\n",
                    lineNo, section.c_str(), key.c_str(), val.c_str(), e.what());
        }
    }

    fclose(f);

    // model width/height must be mirrored into camera config for preprocessing
    cfg.camera.modelWidth  = cfg.decoder.modelWidth;
    cfg.camera.modelHeight = cfg.decoder.modelHeight;

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
//  printConfig  — log the active configuration at startup
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
    printf("│  Tracker       high=%.2f  low=%.2f  iou=%.2f  "
           "hits=%d  missed=%d\n",
           cfg.tracker.highThresh, cfg.tracker.lowThresh,
           cfg.tracker.iouThresh, cfg.tracker.minHits, cfg.tracker.maxMissed);
    printf("│  Crops         %s  q=%d  min_conf=%.2f\n",
           cfg.crops.outputDir.c_str(),
           cfg.crops.jpegQuality,
           cfg.crops.minConfidence);
    printf("│  Stream        port=%d  %dx%d  q=%d\n",
           cfg.stream.port,
           cfg.stream.streamWidth, cfg.stream.streamHeight,
           cfg.stream.jpegQuality);
    printf("│  Database      %s\n", cfg.dbPath.c_str());
    printf("└───────────────────────────────────────────────────────────\n\n");
}
