#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  exif_writer.h  —  Inject EXIF metadata into a JPEG file using libexif
//
//  Writes the following tags:
//    DateTime / DateTimeOriginal  — frame capture timestamp (UTC)
//    GPSLatitude / GPSLongitude   — trap GPS position (if set)
//    GPSAltitude                  — trap altitude (if set)
//    ImageDescription             — human-readable detection summary
//    UserComment                  — JSON machine-readable metadata
//    Make                         — "ai-trap"
//    Software                     — "ai-trap v1"
//
//  Usage:
//    ExifWriter::Params p;
//    p.timestampUs = frame.timestampNs / 1000;
//    p.trackId     = t.trackId;
//    p.className   = "insect";
//    p.confidence  = 0.74f;
//    p.trapId      = cfg.trapId;
//    p.lat         = cfg.gpsLat;
//    p.lon         = cfg.gpsLon;
//    p.hasGps      = cfg.gpsValid;
//    ExifWriter::inject("crops/insect_42.jpg", p);
//
//  Returns true on success.  On failure the JPEG is left unmodified.
//
//  No external dependencies — TIFF/EXIF block built from scratch.
// ─────────────────────────────────────────────────────────────────────────────

#include <string>
#include <cstdint>

namespace ExifWriter {

struct Params {
    // Detection metadata
    int         trackId     = 0;
    int         classId     = 0;
    std::string className;
    float       confidence  = 0.f;

    // Timing — microseconds since Unix epoch (from frame.timestampNs / 1000)
    int64_t     timestampUs = 0;

    // Trap identity
    std::string trapId;
    std::string trapLocation;

    // GPS — only written if hasGps is true
    bool        hasGps      = false;
    double      lat         = 0.0;   // decimal degrees, negative = South
    double      lon         = 0.0;   // decimal degrees, negative = West
    double      altM        = 0.0;   // metres above sea level
};

// Inject EXIF into an existing JPEG file at `path`.
// The file is read, EXIF block constructed, and the file rewritten in place.
// Returns true on success.
bool inject(const std::string& path, const Params& p);

} // namespace ExifWriter