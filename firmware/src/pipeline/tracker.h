#pragma once

#include "decoder.h"

#include <vector>
#include <cstdint>

// ─────────────────────────────────────────────────────────────────────────────
//  ByteTrackerConfig
// ─────────────────────────────────────────────────────────────────────────────
struct ByteTrackerConfig {
    float highThresh = 0.50f;   // detections above this go to pass 1
    float lowThresh  = 0.20f;   // detections above this go to pass 2
    float iouThresh  = 0.30f;   // IoU gate for track/det matching
    int   minHits    = 3;       // frames before a track is "confirmed"
    int   maxMissed  = 30;      // frames before a lost track is deleted
};

// ─────────────────────────────────────────────────────────────────────────────
//  TrackedObject  — output of ByteTracker::update()
// ─────────────────────────────────────────────────────────────────────────────
struct TrackedObject {
    int   trackId;      // unique, monotonically increasing
    int   classId;
    float x1, y1, x2, y2;
    float score;
    int   age;          // frames since track was first created
    int   hitStreak;    // consecutive frames with a match
    bool  confirmed;    // true once hitStreak >= minHits
};

// ─────────────────────────────────────────────────────────────────────────────
//  ByteTracker
//
//  A lightweight implementation of the ByteTrack algorithm.
//
//  Algorithm per frame (8 steps):
//    1. Predict all active tracks forward with a constant-velocity Kalman filter.
//    2. Split detections into high-confidence (≥ highThresh) and
//       low-confidence (≥ lowThresh and < highThresh) sets.
//    3. Match high-confidence detections → all active tracks  (greedy IoU).
//    4. Match low-confidence detections → unmatched active tracks.
//    5. Mark tracks unmatched in both passes as missed.
//    6. Start new tracks from unmatched HIGH-confidence detections only.
//    7. Delete tracks that have been missed for ≥ maxMissed frames.
//    8. Return all tracks whose hitStreak ≥ minHits as confirmed.
//
//  Kalman state vector (8 elements):
//    [ cx, cy, area, aspect, vcx, vcy, varea, vaspect ]
//    Constant-velocity model; area and aspect are logged to keep
//    the state near-linear across scale changes.
//
//  Usage
//  ─────
//    ByteTracker tracker;                  // default config
//    ByteTracker tracker(cfg);             // custom config
//
//    // Inside your inference callback:
//    auto tracked = tracker.update(detections);
//    for (const auto& t : tracked)
//        if (t.confirmed) { /* draw / persist */ }
// ─────────────────────────────────────────────────────────────────────────────
class ByteTracker {
public:
    explicit ByteTracker(const ByteTrackerConfig& cfg = {});

    // Feed detections for the current frame; returns all active tracks.
    // confirmed=true means the track has been seen for ≥ minHits frames.
    std::vector<TrackedObject> update(const std::vector<Detection>& dets);

    // Reset all state (e.g. on stream restart)
    void reset();

    const ByteTrackerConfig& config() const { return m_cfg; }

private:
    // ── Kalman track ─────────────────────────────────────────────────────────
    struct KalmanTrack {
        int   trackId;
        int   classId;
        float score;
        int   age        = 0;
        int   hitStreak  = 0;
        int   missedFrames = 0;
        bool  confirmed  = false;

        // Kalman state: [cx, cy, area, aspect, vcx, vcy, varea, vaspect]
        float x[8] = {};   // state estimate
        float P[8] = {};   // diagonal variance (simplified scalar Kalman)

        // Last observed box (for IoU matching after prediction)
        float x1, y1, x2, y2;

        void initFromBox(float bx1, float by1, float bx2, float by2);
        void predict();
        void update(float bx1, float by1, float bx2, float by2, float sc);
        TrackedObject toOutput() const;
    };

    // ── Matching ──────────────────────────────────────────────────────────────
    struct Match { int trackIdx; int detIdx; };

    // Greedy IoU matching: highest-IoU pairs first, up to iouThresh
    std::vector<Match> matchGreedy(
        const std::vector<KalmanTrack*>& tracks,
        const std::vector<Detection>&    dets,
        float                            iouThresh) const;

    static float boxIou(float ax1, float ay1, float ax2, float ay2,
                        float bx1, float by1, float bx2, float by2);

    // ── State ─────────────────────────────────────────────────────────────────
    ByteTrackerConfig          m_cfg;
    std::vector<KalmanTrack>   m_tracks;
    int                        m_nextId = 1;
};
