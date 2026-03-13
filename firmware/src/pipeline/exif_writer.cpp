#include "exif_writer.h"

// ─────────────────────────────────────────────────────────────────────────────
//  exif_writer.cpp  —  Inject EXIF into a JPEG without libexif's serialiser.
//
//  Builds the EXIF / TIFF block entirely by hand per the EXIF 2.3 spec.
//  libexif is NOT used for output — only its entry/IFD data structures are
//  used to parse incoming data (none here).  This avoids the fragmented
//  public API surface across libexif versions (0.6.21 … 0.6.24).
//
//  JPEG APP1 layout:
//    FF E1  <length:2be>  "Exif\0\0"  <TIFF block>
//
//  TIFF block (little-endian, "II"):
//    Header(8) + IFD0 + IFD_Exif + IFD_GPS + value-data area
//
//  Each IFD entry = 12 bytes: tag(2) type(2) count(4) value_or_offset(4)
// ─────────────────────────────────────────────────────────────────────────────

#include <cstdio>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <string>

namespace {

using Buf = std::vector<uint8_t>;

void w16(Buf& b, uint16_t v) {
    b.push_back( v       & 0xFF);
    b.push_back((v >> 8) & 0xFF);
}
void w32(Buf& b, uint32_t v) {
    b.push_back( v        & 0xFF);
    b.push_back((v >>  8) & 0xFF);
    b.push_back((v >> 16) & 0xFF);
    b.push_back((v >> 24) & 0xFF);
}
void wStr(Buf& b, const char* s, size_t len) {
    b.insert(b.end(), s, s + len);
}

// IFD entry: tag type count value_or_offset
void ifd(Buf& b, uint16_t tag, uint16_t type,
         uint32_t count, uint32_t val) {
    w16(b, tag); w16(b, type); w32(b, count); w32(b, val);
}

// TIFF types
constexpr uint16_t T_BYTE      = 1;
constexpr uint16_t T_ASCII     = 2;
constexpr uint16_t T_RATIONAL  = 5;
constexpr uint16_t T_UNDEFINED = 7;

// Append GPS coordinate as 3 rationals (deg/1, min/1, sec*1000/1000)
void gpsRationals(Buf& b, double decimal) {
    double a   = std::fabs(decimal);
    uint32_t d = static_cast<uint32_t>(a);
    uint32_t m = static_cast<uint32_t>((a - d) * 60.0);
    uint32_t s = static_cast<uint32_t>(((a - d) * 60.0 - m) * 60000.0 + 0.5);
    w32(b, d); w32(b, 1);
    w32(b, m); w32(b, 1);
    w32(b, s); w32(b, 1000);
}

std::vector<uint8_t> readFile(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return {};
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::rewind(f);
    if (sz <= 0) { std::fclose(f); return {}; }
    std::vector<uint8_t> buf(static_cast<size_t>(sz));
    std::fread(buf.data(), 1, buf.size(), f);
    std::fclose(f);
    return buf;
}

bool writeFile(const std::string& path, const uint8_t* d, size_t n) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = (std::fwrite(d, 1, n, f) == n);
    std::fclose(f);
    return ok;
}

} // namespace

// ─────────────────────────────────────────────────────────────────────────────

bool ExifWriter::inject(const std::string& path, const Params& p)
{
    std::vector<uint8_t> jpeg = readFile(path);
    if (jpeg.size() < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8) {
        std::fprintf(stderr, "ExifWriter: not a JPEG: %s\n", path.c_str());
        return false;
    }

    // ── DateTime string ───────────────────────────────────────────────────────
    char dt[20] = "0000:00:00 00:00:00";
    if (p.timestampUs > 0) {
        time_t s = static_cast<time_t>(p.timestampUs / 1000000LL);
        if (struct tm* t = std::gmtime(&s))
            std::strftime(dt, sizeof(dt), "%Y:%m:%d %H:%M:%S", t);
    }
    const uint32_t dtLen = 20; // 19 chars + NUL

    // ── ImageDescription ──────────────────────────────────────────────────────
    char desc[256];
    std::snprintf(desc, sizeof(desc), "%s #%d  conf=%.2f  trap=%s",
                  p.className.c_str(), p.trackId,
                  p.confidence, p.trapId.c_str());
    uint32_t descLen = static_cast<uint32_t>(std::strlen(desc) + 1);

    // ── UserComment: "ASCII\0\0\0" + JSON ─────────────────────────────────────
    char json[512];
    std::snprintf(json, sizeof(json),
        "{\"trackId\":%d,\"classId\":%d,\"class\":\"%s\","
        "\"conf\":%.4f,\"trapId\":\"%s\",\"location\":\"%s\","
        "\"timestampUs\":%lld}",
        p.trackId, p.classId, p.className.c_str(),
        p.confidence, p.trapId.c_str(), p.trapLocation.c_str(),
        static_cast<long long>(p.timestampUs));
    static const char kAscii[8] = {'A','S','C','I','I','\0','\0','\0'};
    uint32_t jsonLen = static_cast<uint32_t>(std::strlen(json));
    uint32_t ucLen   = 8 + jsonLen;

    // ── GPS date string ───────────────────────────────────────────────────────
    char gpsDate[12] = "0000:00:00";
    uint32_t gpsDateLen = 11; // always fixed size
    if (p.hasGps && p.timestampUs > 0) {
        time_t s = static_cast<time_t>(p.timestampUs / 1000000LL);
        if (struct tm* t = std::gmtime(&s))
            std::strftime(gpsDate, sizeof(gpsDate), "%Y:%m:%d", t);
    }

    // ── IFD entry counts ──────────────────────────────────────────────────────
    // IFD0:      ImageDesc, Make, Software, DateTime, ExifIFD[, GpsIFD]
    // IFD_Exif:  DateTimeOriginal, DateTimeDigitized, UserComment
    // IFD_GPS:   VersionID, LatRef, Lat, LonRef, Lon, AltRef, Alt, Time, Date
    const uint16_t n0    = static_cast<uint16_t>(5 + (p.hasGps ? 1 : 0));
    const uint16_t nExif = 3;
    const uint16_t nGps  = 9;

    // ── Offset arithmetic (all offsets relative to TIFF block start) ──────────
    const uint32_t offIFD0   = 8;
    const uint32_t offExif   = offIFD0 + 2 + n0    * 12 + 4;
    const uint32_t offGps    = offExif + 2 + nExif  * 12 + 4;
    const uint32_t offGpsEnd = p.hasGps ? offGps + 2 + nGps * 12 + 4 : offGps;

    // Value-data allocation (sequential, no alignment padding needed for us)
    uint32_t cur = offGpsEnd;
    auto alloc = [&](uint32_t sz) { uint32_t o = cur; cur += sz; return o; };

    uint32_t oDesc  = alloc(descLen);
    uint32_t oMake  = alloc(8);   // "ai-trap\0"
    uint32_t oSoft  = alloc(11);  // "ai-trap v1\0"
    uint32_t oDt    = alloc(dtLen);
    uint32_t oDtO   = alloc(dtLen);
    uint32_t oDtD   = alloc(dtLen);
    uint32_t oUC    = alloc(ucLen);

    uint32_t oGpsVer = 0, oLat = 0, oLon = 0;
    uint32_t oAlt = 0, oGpsTime = 0, oGpsDate = 0;
    if (p.hasGps) {
        oGpsVer  = alloc(4);
        oLat     = alloc(24);
        oLon     = alloc(24);
        oAlt     = alloc(8);
        oGpsTime = alloc(24);
        oGpsDate = alloc(gpsDateLen);
    }

    // ── Assemble TIFF block ───────────────────────────────────────────────────
    Buf T;
    T.reserve(cur + 64);

    // TIFF header
    T.push_back('I'); T.push_back('I');
    w16(T, 42); w32(T, offIFD0);

    // IFD0
    w16(T, n0);
    ifd(T, 0x010E, T_ASCII,    descLen, oDesc);
    ifd(T, 0x010F, T_ASCII,    8,       oMake);
    ifd(T, 0x0131, T_ASCII,    11,      oSoft);
    ifd(T, 0x0132, T_ASCII,    dtLen,   oDt);
    ifd(T, 0x8769, 4/*LONG*/,  1,       offExif);
    if (p.hasGps)
        ifd(T, 0x8825, 4/*LONG*/, 1,    offGps);
    w32(T, 0); // next IFD

    // IFD_Exif
    w16(T, nExif);
    ifd(T, 0x9003, T_ASCII,    dtLen, oDtO);
    ifd(T, 0x9004, T_ASCII,    dtLen, oDtD);
    ifd(T, 0x9286, T_UNDEFINED, ucLen, oUC);
    w32(T, 0);

    // IFD_GPS
    if (p.hasGps) {
        w16(T, nGps);
        ifd(T, 0x0000, T_BYTE,      4,  oGpsVer);
        ifd(T, 0x0001, T_ASCII,     2,  static_cast<uint32_t>(p.lat >= 0 ? 'N' : 'S'));
        ifd(T, 0x0002, T_RATIONAL,  3,  oLat);
        ifd(T, 0x0003, T_ASCII,     2,  static_cast<uint32_t>(p.lon >= 0 ? 'E' : 'W'));
        ifd(T, 0x0004, T_RATIONAL,  3,  oLon);
        ifd(T, 0x0005, T_BYTE,      1,  static_cast<uint32_t>(p.altM >= 0 ? 0 : 1));
        ifd(T, 0x0006, T_RATIONAL,  1,  oAlt);
        ifd(T, 0x0007, T_RATIONAL,  3,  oGpsTime);
        ifd(T, 0x001D, T_ASCII, gpsDateLen, oGpsDate);
        w32(T, 0);
    }

    // Value data — order must match alloc() calls above
    wStr(T, desc, descLen);
    wStr(T, "ai-trap\0", 8);
    wStr(T, "ai-trap v1\0", 11);
    wStr(T, dt, dtLen);   // DateTime
    wStr(T, dt, dtLen);   // DateTimeOriginal
    wStr(T, dt, dtLen);   // DateTimeDigitized
    T.insert(T.end(), kAscii, kAscii + 8);
    wStr(T, json, jsonLen); // UserComment (no NUL)

    if (p.hasGps) {
        // GPSVersionID 2.2.0.0
        T.push_back(2); T.push_back(2); T.push_back(0); T.push_back(0);
        gpsRationals(T, p.lat);
        gpsRationals(T, p.lon);
        // Altitude rational
        uint32_t altCm = static_cast<uint32_t>(std::fabs(p.altM) * 100.0 + 0.5);
        w32(T, altCm); w32(T, 100);
        // GPS time rationals
        if (p.timestampUs > 0) {
            time_t s = static_cast<time_t>(p.timestampUs / 1000000LL);
            struct tm* t = std::gmtime(&s);
            if (t) {
                w32(T, static_cast<uint32_t>(t->tm_hour)); w32(T, 1);
                w32(T, static_cast<uint32_t>(t->tm_min));  w32(T, 1);
                w32(T, static_cast<uint32_t>(t->tm_sec));  w32(T, 1);
            } else {
                for (int i = 0; i < 3; i++) { w32(T, 0); w32(T, 1); }
            }
        } else {
            for (int i = 0; i < 3; i++) { w32(T, 0); w32(T, 1); }
        }
        wStr(T, gpsDate, gpsDateLen);
    }

    // ── Build APP1 segment ────────────────────────────────────────────────────
    // FF E1 + length(2BE) + "Exif\0\0" + TIFF
    // length covers itself + 6 + T.size()
    uint16_t segLen = static_cast<uint16_t>(2 + 6 + T.size());
    Buf app1;
    app1.reserve(2 + 2 + 6 + T.size());
    app1.push_back(0xFF); app1.push_back(0xE1);
    app1.push_back((segLen >> 8) & 0xFF);
    app1.push_back( segLen       & 0xFF);
    const uint8_t kExifId[6] = {'E','x','i','f',0,0};
    app1.insert(app1.end(), kExifId, kExifId + 6);
    app1.insert(app1.end(), T.begin(), T.end());

    // ── Splice into JPEG ──────────────────────────────────────────────────────
    size_t pos = 2;
    while (pos + 3 < jpeg.size()) {
        if (jpeg[pos] != 0xFF) break;
        uint8_t mk = jpeg[pos + 1];
        if (mk == 0xE1) {
            size_t existing = (static_cast<size_t>(jpeg[pos + 2]) << 8)
                            |  jpeg[pos + 3];
            jpeg.erase(jpeg.begin() + pos,
                       jpeg.begin() + pos + 2 + existing);
            break;
        }
        if (mk == 0xE0 || mk == 0xFE || (mk >= 0xE2 && mk <= 0xEF)) {
            size_t skip = (static_cast<size_t>(jpeg[pos + 2]) << 8)
                        |  jpeg[pos + 3];
            pos += 2 + skip;
        } else {
            break;
        }
    }
    jpeg.insert(jpeg.begin() + pos, app1.begin(), app1.end());

    // ── Write ─────────────────────────────────────────────────────────────────
    bool ok = writeFile(path, jpeg.data(), jpeg.size());
    if (ok)
        std::fprintf(stdout, "ExifWriter: injected %zu EXIF bytes into %s\n",
                     T.size(), path.c_str());
    else
        std::fprintf(stderr, "ExifWriter: write failed: %s\n", path.c_str());
    return ok;
}