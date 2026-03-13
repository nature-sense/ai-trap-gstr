#include "tracker.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_set>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
//  KalmanTrack — helpers
// ─────────────────────────────────────────────────────────────────────────────

// Convert box → Kalman state [cx, cy, area, aspect]
static void boxToState(float x1, float y1, float x2, float y2,
                       float* cx, float* cy, float* area, float* aspect)
{
    *cx     = (x1 + x2) * 0.5f;
    *cy     = (y1 + y2) * 0.5f;
    *area   = std::max(1.f, (x2 - x1) * (y2 - y1));
    *aspect = (x2 - x1) / std::max(1.f, y2 - y1);
}

// Convert Kalman state → box corners
static void stateToBox(float cx, float cy, float area, float aspect,
                       float* x1, float* y1, float* x2, float* y2)
{
    float w = std::sqrt(std::max(1.f, area * aspect));
    float h = std::max(1.f, area / w);
    *x1 = cx - w * 0.5f;
    *y1 = cy - h * 0.5f;
    *x2 = cx + w * 0.5f;
    *y2 = cy + h * 0.5f;
}

void ByteTracker::KalmanTrack::initFromBox(
    float bx1, float by1, float bx2, float by2)
{
    float cx, cy, area, aspect;
    boxToState(bx1, by1, bx2, by2, &cx, &cy, &area, &aspect);

    // State: [cx, cy, area, aspect, vcx, vcy, varea, vaspect]
    x[0] = cx;    x[1] = cy;    x[2] = area;  x[3] = aspect;
    x[4] = 0.f;   x[5] = 0.f;   x[6] = 0.f;   x[7] = 0.f;

    // Initial uncertainty — position fairly confident, velocity unknown
    P[0] = 10.f;  P[1] = 10.f;  P[2] = 10.f;  P[3] = 10.f;
    P[4] = 1e4f;  P[5] = 1e4f;  P[6] = 1e4f;  P[7] = 1e4f;

    x1 = bx1; y1 = by1; x2 = bx2; y2 = by2;
}

// Constant-velocity predict step (simplified scalar diagonal Kalman)
void ByteTracker::KalmanTrack::predict() {
    // x_{k|k-1} = F * x_{k-1}   where F applies velocity
    x[0] += x[4];
    x[1] += x[5];
    x[2] += x[6];
    x[3] += x[7];

    // P_{k|k-1} = F*P*F' + Q  (process noise Q inflates P each step)
    static constexpr float Q_pos = 1.f;
    static constexpr float Q_vel = 10.f;
    P[0] += Q_pos; P[1] += Q_pos; P[2] += Q_pos; P[3] += Q_pos;
    P[4] += Q_vel; P[5] += Q_vel; P[6] += Q_vel; P[7] += Q_vel;

    // Update cached box from predicted state
    stateToBox(x[0], x[1], x[2], x[3], &x1, &y1, &x2, &y2);
    missedFrames++;
}

// Kalman update with a matched detection
void ByteTracker::KalmanTrack::update(
    float bx1, float by1, float bx2, float by2, float sc)
{
    float cx, cy, area, aspect;
    boxToState(bx1, by1, bx2, by2, &cx, &cy, &area, &aspect);

    // Measurement noise R (observation uncertainty)
    static constexpr float R[4] = { 1.f, 1.f, 10.f, 0.01f };

    // Update only the positional states (indices 0-3); velocities estimated
    float z[4] = { cx, cy, area, aspect };
    for (int i = 0; i < 4; i++) {
        float K = P[i] / (P[i] + R[i]);        // Kalman gain
        float innov = z[i] - x[i];             // innovation
        x[i] += K * innov;                     // update position
        x[i + 4] += K * innov * 0.5f;          // update velocity estimate
        P[i] *= (1.f - K);                     // shrink uncertainty
    }

    // Clamp velocity to prevent runaway
    for (int i = 4; i < 8; i++)
        x[i] = std::max(-200.f, std::min(x[i], 200.f));

    // Update cached box
    stateToBox(x[0], x[1], x[2], x[3], &x1, &y1, &x2, &y2);

    score        = sc;
    hitStreak++;
    missedFrames = 0;
    age++;
}

TrackedObject ByteTracker::KalmanTrack::toOutput() const {
    TrackedObject t{};
    t.trackId    = trackId;
    t.classId    = classId;
    t.x1         = x1;
    t.y1         = y1;
    t.x2         = x2;
    t.y2         = y2;
    t.score      = score;
    t.age        = age;
    t.hitStreak  = hitStreak;
    t.confirmed  = confirmed;
    return t;
}

// ─────────────────────────────────────────────────────────────────────────────
//  ByteTracker
// ─────────────────────────────────────────────────────────────────────────────

ByteTracker::ByteTracker(const ByteTrackerConfig& cfg) : m_cfg(cfg) {}

void ByteTracker::reset() {
    m_tracks.clear();
    m_nextId = 1;
}

// ─────────────────────────────────────────────────────────────────────────────
//  update  — main per-frame method
// ─────────────────────────────────────────────────────────────────────────────

std::vector<TrackedObject> ByteTracker::update(
    const std::vector<Detection>& dets)
{
    // ── Step 1: predict all tracks forward ───────────────────────────────────
    for (auto& t : m_tracks)
        t.predict();

    // ── Step 2: split detections into high / low confidence ──────────────────
    std::vector<Detection> highDets, lowDets;
    for (const auto& d : dets) {
        if      (d.confidence >= m_cfg.highThresh) highDets.push_back(d);
        else if (d.confidence >= m_cfg.lowThresh)  lowDets.push_back(d);
    }

    // Build pointer lists for active tracks
    std::vector<KalmanTrack*> activeTracks;
    for (auto& t : m_tracks) activeTracks.push_back(&t);

    // ── Step 3: match high-confidence dets → all active tracks ───────────────
    auto matches1 = matchGreedy(activeTracks, highDets, m_cfg.iouThresh);

    std::vector<bool> trackMatched(m_tracks.size(), false);
    std::vector<bool> highDetMatched(highDets.size(), false);

    for (const auto& m : matches1) {
        activeTracks[m.trackIdx]->update(
            highDets[m.detIdx].x1, highDets[m.detIdx].y1,
            highDets[m.detIdx].x2, highDets[m.detIdx].y2,
            highDets[m.detIdx].confidence);
        trackMatched[m.trackIdx]   = true;
        highDetMatched[m.detIdx]   = true;
    }

    // ── Step 4: match low-confidence dets → unmatched tracks ─────────────────
    std::vector<KalmanTrack*> unmatchedTracks;
    for (size_t i = 0; i < activeTracks.size(); i++)
        if (!trackMatched[i]) unmatchedTracks.push_back(activeTracks[i]);

    auto matches2 = matchGreedy(unmatchedTracks, lowDets,
                                m_cfg.iouThresh * 0.8f);

    // Use a pointer set — unmatchedTracks indices != activeTracks indices
    std::unordered_set<KalmanTrack*> matchedPtrs;
    for (const auto& m : matches1)
        matchedPtrs.insert(activeTracks[m.trackIdx]);
    for (const auto& m : matches2) {
        KalmanTrack* trk = unmatchedTracks[m.trackIdx];
        trk->update(
            lowDets[m.detIdx].x1, lowDets[m.detIdx].y1,
            lowDets[m.detIdx].x2, lowDets[m.detIdx].y2,
            lowDets[m.detIdx].confidence);
        matchedPtrs.insert(trk);
    }

    // ── Step 5: tracks with no match in either pass lose hit streak ───────────
    for (auto& t : m_tracks) {
        if (matchedPtrs.find(&t) == matchedPtrs.end()) {
            // Genuinely unmatched this frame
            t.hitStreak = std::max(0, t.hitStreak - 1);
        }
    }

    // ── Step 6: create new tracks from unmatched HIGH-confidence dets ─────────
    for (size_t i = 0; i < highDets.size(); i++) {
        if (highDetMatched[i]) continue;

        KalmanTrack nt{};
        nt.trackId = m_nextId++;
        nt.classId = highDets[i].classId;
        nt.score   = highDets[i].confidence;
        nt.age     = 1;
        nt.hitStreak = 1;
        nt.initFromBox(highDets[i].x1, highDets[i].y1,
                       highDets[i].x2, highDets[i].y2);
        m_tracks.push_back(nt);
    }

    // ── Step 7: remove dead tracks ────────────────────────────────────────────
    m_tracks.erase(
        std::remove_if(m_tracks.begin(), m_tracks.end(),
            [this](const KalmanTrack& t) {
                return t.missedFrames >= m_cfg.maxMissed;
            }),
        m_tracks.end());

    // ── Step 8: mark confirmed, collect output ────────────────────────────────
    std::vector<TrackedObject> out;
    out.reserve(m_tracks.size());

    for (auto& t : m_tracks) {
        // confirmed tracks hitStreak >= minHits; lost tracks become unconfirmed
        t.confirmed = (t.hitStreak >= m_cfg.minHits);
        out.push_back(t.toOutput());
    }

    // ── Debug: print tracker state every 900 frames (~30 s at 30 fps) ──────────
    {
        static int dbgFrame = 0;
        if (++dbgFrame % 900 == 0) {
            printf("[tracker] frame=%d  active_tracks=%zu\n",
                   dbgFrame, m_tracks.size());
        }
    }

    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
//  matchGreedy  — O(T·D) greedy IoU matching
//
//  Builds a full IoU matrix, then iterates in descending-IoU order picking
//  one-to-one matches above iouThresh.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<ByteTracker::Match> ByteTracker::matchGreedy(
    const std::vector<KalmanTrack*>& tracks,
    const std::vector<Detection>&    dets,
    float                            iouThresh) const
{
    std::vector<Match> matches;
    if (tracks.empty() || dets.empty()) return matches;

    const int T = (int)tracks.size();
    const int D = (int)dets.size();

    // Build IoU matrix as (trackIdx, detIdx, iou) triples
    struct Cell { int ti, di; float iou; };
    std::vector<Cell> cells;
    cells.reserve(static_cast<size_t>(T * D));

    for (int ti = 0; ti < T; ti++) {
        for (int di = 0; di < D; di++) {
            float score = boxIou(tracks[ti]->x1, tracks[ti]->y1,
                                 tracks[ti]->x2, tracks[ti]->y2,
                                 dets[di].x1,    dets[di].y1,
                                 dets[di].x2,    dets[di].y2);
            if (score > iouThresh)
                cells.push_back({ ti, di, score });
        }
    }

    // Sort descending by IoU
    std::sort(cells.begin(), cells.end(),
        [](const Cell& a, const Cell& b) { return a.iou > b.iou; });

    std::vector<bool> trackUsed(static_cast<size_t>(T), false);
    std::vector<bool> detUsed  (static_cast<size_t>(D), false);

    for (const auto& c : cells) {
        if (trackUsed[c.ti] || detUsed[c.di]) continue;
        matches.push_back({ c.ti, c.di });
        trackUsed[c.ti] = true;
        detUsed  [c.di] = true;
    }

    return matches;
}

// ─────────────────────────────────────────────────────────────────────────────
//  boxIou
// ─────────────────────────────────────────────────────────────────────────────

float ByteTracker::boxIou(float ax1, float ay1, float ax2, float ay2,
                           float bx1, float by1, float bx2, float by2)
{
    float ix1   = std::max(ax1, bx1);
    float iy1   = std::max(ay1, by1);
    float ix2   = std::min(ax2, bx2);
    float iy2   = std::min(ay2, by2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float aA    = std::max(0.f, ax2 - ax1) * std::max(0.f, ay2 - ay1);
    float bA    = std::max(0.f, bx2 - bx1) * std::max(0.f, by2 - by1);
    float uni   = aA + bA - inter + 1e-6f;
    return inter / uni;
}