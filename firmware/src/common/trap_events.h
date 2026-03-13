#pragma once

// ─────────────────────────────────────────────────────────────────────────────
//  trap_events.h  —  JSON event builders for the SSE stream
//
//  All functions return a std::string suitable for SseServer::pushEvent().
//
//  Event schema:
//
//  detection   — fired once per confirmed track per frame
//    {"type":"detection","trackId":42,"class":"insect","conf":0.87,
//     "bbox":[x1,y1,x2,y2],"frameId":1234,"ts":1741234567890}
//
//  crop_saved  — fired when CropSaver writes a better crop for a track
//    {"type":"crop_saved","trackId":42,"class":"insect","conf":0.87,
//     "file":"insect_42.jpg","w":320,"h":240,"ts":1741234567890}
//
//  stats       — periodic summary (every 30 s or on significant change)
//    {"type":"stats","today":247,"uptime_s":22440,"fps":18.3,
//     "tracks":12,"db_mb":1.4}
//
//  health      — system health (every 30 s alongside stats)
//    {"type":"health","temp_c":42.1,"af_state":2,"lens_pos":2.4}
//
//  capture     — fired when detection is started or stopped via REST
//    {"type":"capture","active":true,"ts":1741234567890}
// ─────────────────────────────────────────────────────────────────────────────

#include <chrono>
#include <cstdio>
#include <string>

namespace TrapEvents {

// Unix timestamp in milliseconds
static inline long long nowMs() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(
        system_clock::now().time_since_epoch()).count();
}

// ── detection ─────────────────────────────────────────────────────────────────

static inline std::string detection(
    int         trackId,
    const char* cls,
    float       conf,
    float       x1, float y1, float x2, float y2,
    uint64_t    frameId)
{
    char buf[256];
    snprintf(buf, sizeof(buf),
        "{\"type\":\"detection\","
        "\"trackId\":%d,"
        "\"class\":\"%s\","
        "\"conf\":%.3f,"
        "\"bbox\":[%.0f,%.0f,%.0f,%.0f],"
        "\"frameId\":%llu,"
        "\"ts\":%lld}",
        trackId, cls, conf,
        x1, y1, x2, y2,
        (unsigned long long)frameId,
        nowMs());
    return buf;
}

// ── crop_saved ────────────────────────────────────────────────────────────────

static inline std::string cropSaved(
    int         trackId,
    const char* cls,
    float       conf,
    const char* filename,
    int         w, int h)
{
    char buf[256];
    snprintf(buf, sizeof(buf),
        "{\"type\":\"crop_saved\","
        "\"trackId\":%d,"
        "\"class\":\"%s\","
        "\"conf\":%.3f,"
        "\"file\":\"%s\","
        "\"w\":%d,\"h\":%d,"
        "\"ts\":%lld}",
        trackId, cls, conf, filename, w, h,
        nowMs());
    return buf;
}

// ── stats ─────────────────────────────────────────────────────────────────────

static inline std::string stats(
    long long totalDetections,
    long long uptimeSeconds,
    float     fps,
    long long uniqueTracks,
    double    dbMb)
{
    char buf[256];
    snprintf(buf, sizeof(buf),
        "{\"type\":\"stats\","
        "\"today\":%lld,"
        "\"uptime_s\":%lld,"
        "\"fps\":%.1f,"
        "\"tracks\":%lld,"
        "\"db_mb\":%.2f,"
        "\"ts\":%lld}",
        totalDetections, uptimeSeconds, fps,
        uniqueTracks, dbMb,
        nowMs());
    return buf;
}

// ── health ────────────────────────────────────────────────────────────────────

static inline std::string health(
    float tempC,
    int   afState,
    float lensPos)
{
    char buf[128];
    snprintf(buf, sizeof(buf),
        "{\"type\":\"health\","
        "\"temp_c\":%.1f,"
        "\"af_state\":%d,"
        "\"lens_pos\":%.2f,"
        "\"ts\":%lld}",
        tempC, afState, lensPos,
        nowMs());
    return buf;
}

// ── capture ───────────────────────────────────────────────────────────────────

static inline std::string captureState(bool active)
{
    char buf[80];
    snprintf(buf, sizeof(buf),
        "{\"type\":\"capture\","
        "\"active\":%s,"
        "\"ts\":%lld}",
        active ? "true" : "false",
        nowMs());
    return buf;
}

// ── CPU temperature from /sys/class/thermal ───────────────────────────────────

static inline float readCpuTemp() {
    FILE* f = fopen("/sys/class/thermal/thermal_zone0/temp", "r");
    if (!f) return 0.f;
    int millideg = 0;
    fscanf(f, "%d", &millideg);
    fclose(f);
    return static_cast<float>(millideg) / 1000.f;
}

} // namespace TrapEvents
