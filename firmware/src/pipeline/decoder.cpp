#include "decoder.h"
#include "ncnn/mat.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

// ─────────────────────────────────────────────────────────────────────────────
//  Static helpers
// ─────────────────────────────────────────────────────────────────────────────

static inline float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

float YoloDecoder::clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Inverse letterbox: model pixel → original image pixel
float YoloDecoder::unpad(float coord, int pad, float scale) {
    return (coord - static_cast<float>(pad)) / scale;
}

float YoloDecoder::iou(const Detection& a, const Detection& b) {
    float ix1   = std::max(a.x1, b.x1);
    float iy1   = std::max(a.y1, b.y1);
    float ix2   = std::min(a.x2, b.x2);
    float iy2   = std::min(a.y2, b.y2);
    float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
    float aA    = std::max(0.f, a.x2 - a.x1) * std::max(0.f, a.y2 - a.y1);
    float bA    = std::max(0.f, b.x2 - b.x1) * std::max(0.f, b.y2 - b.y1);
    return inter / (aA + bA - inter + 1e-6f);
}

// ─────────────────────────────────────────────────────────────────────────────
//  Construction
// ─────────────────────────────────────────────────────────────────────────────

YoloDecoder::YoloDecoder(const DecoderConfig& cfg) : m_cfg(cfg) {}

// ─────────────────────────────────────────────────────────────────────────────
//  decode  — public entry point
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Detection> YoloDecoder::decode(
    const ncnn::Mat& out,
    int   srcW,    int srcH,
    float scale,
    int   padLeft, int padTop) const
{
    if (out.dims == 3) {
        // Squeeze batch dim: [1, rows, cols] → [rows, cols]
        ncnn::Mat sq = out.reshape(out.w, out.h);
        return dispatch(sq, srcW, srcH, scale, padLeft, padTop);
    }
    if (out.dims != 2) {
        fprintf(stderr, "YoloDecoder: unexpected dims=%d (expected 2 or 3)\n",
                out.dims);
        return {};
    }
    return dispatch(out, srcW, srcH, scale, padLeft, padTop);
}

// ─────────────────────────────────────────────────────────────────────────────
//  dispatch  — route to the correct decoder path
//
//  Format forced via DecoderConfig::format when set to AnchorGrid or EndToEnd.
//
//  Auto-detection:
//    Multi-class (numClasses > 1): shapes are unambiguous.
//      Format A: h = 4 + numClasses, w = ~anchors
//      Format B: minor dim = 4 + numClasses + 1 (classId col)
//
//    Single-class (numClasses == 1): both formats give h=5 — ambiguous.
//      Disambiguate by sampling score from row[4]:
//        Format A: raw logit  → typically outside [0, 1]
//        Format B: post-sigmoid → always in [0, 1]
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Detection> YoloDecoder::dispatch(
    const ncnn::Mat& out,
    int srcW, int srcH, float scale, int padLeft, int padTop) const
{
    // ── Forced format ─────────────────────────────────────────────────────────
    if (m_cfg.format == YoloFormat::AnchorGrid)
        return decodeAnchorGrid(out, srcW, srcH, scale, padLeft, padTop);
    if (m_cfg.format == YoloFormat::EndToEnd)
        return decodeEndToEnd(out, srcW, srcH, scale, padLeft, padTop);

    // ── Auto-detection ────────────────────────────────────────────────────────
    // Format B minor dimension: 5 for single-class, 6+ for multi-class.
    const int fmtB = 4 + m_cfg.numClasses + (m_cfg.numClasses > 1 ? 1 : 0);

    const bool couldBeB = (out.w == fmtB || out.h == fmtB);
    const bool couldBeA = (out.h == 4 + m_cfg.numClasses);

    if (!couldBeB && !couldBeA) {
        fprintf(stderr,
            "YoloDecoder: unrecognised tensor  dims=%d  w=%d  h=%d\n"
            "  Expected Format A: h=%d (4+%d classes), w=~anchors\n"
            "  Expected Format B: minor dim=%d\n"
            "  Set DecoderConfig::format to force a path.\n",
            out.dims, out.w, out.h,
            4 + m_cfg.numClasses, m_cfg.numClasses, fmtB);
        return {};
    }

    if (couldBeB && !couldBeA) {
        static bool printed = false;
        if (!printed) { printf("YoloDecoder: Format B (auto)\n"); printed = true; }
        return decodeEndToEnd(out, srcW, srcH, scale, padLeft, padTop);
    }

    if (couldBeA && !couldBeB) {
        static bool printed = false;
        if (!printed) { printf("YoloDecoder: Format A (auto)\n"); printed = true; }
        return decodeAnchorGrid(out, srcW, srcH, scale, padLeft, padTop);
    }

    // ── Ambiguous (single-class): sample a score to decide ───────────────────
    // For Format A (h=5, w=anchors): row 4 holds raw logit scores.
    // Sample row[4][0] — the score of the first anchor.
    float sampleScore = out.row(4)[0];

    if (sampleScore < 0.f || sampleScore > 1.f) {
        // Raw logit outside [0,1] → Format A. Only print once.
        static bool printed = false;
        if (!printed) {
            printf("YoloDecoder: Format A (auto — sample score=%.3f "
                   "outside [0,1], raw logit)\n", sampleScore);
            printed = true;
        }
        return decodeAnchorGrid(out, srcW, srcH, scale, padLeft, padTop);
    }

    // Post-sigmoid score in [0,1] → Format B.
    static bool printed = false;
    if (!printed) {
        printf("YoloDecoder: Format B (auto — sample score=%.3f "
               "in [0,1], post-sigmoid)\n", sampleScore);
        printed = true;
    }
    return decodeEndToEnd(out, srcW, srcH, scale, padLeft, padTop);
}

// ─────────────────────────────────────────────────────────────────────────────
//  decodeAnchorGrid  — Format A
//
//  Tensor layout (dims=2, h=4+numClasses, w=numAnchors):
//    row 0   : cx   (model pixel space)
//    row 1   : cy
//    row 2   : bw
//    row 3   : bh
//    row 4+c : raw logit score for class c  ← sigmoid applied here
//
//  NMS applied after decoding.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Detection> YoloDecoder::decodeAnchorGrid(
    const ncnn::Mat& out,
    int srcW, int srcH, float scale, int padLeft, int padTop) const
{
    const int numAnchors = out.w;
    const int numClasses = m_cfg.numClasses;

    std::vector<Detection> dets;
    dets.reserve(256);

    for (int i = 0; i < numAnchors; i++) {
        // Apply sigmoid to raw logit; compare against confThresh.
        float bestScore = m_cfg.confThresh;
        int   bestCls   = -1;
        for (int c = 0; c < numClasses; c++) {
            float s = m_cfg.preAppliedSigmoid
                      ? out.row(4 + c)[i]
                      : sigmoid(out.row(4 + c)[i]);
            if (s > bestScore) { bestScore = s; bestCls = c; }
        }
        if (bestCls < 0) continue;

        float cx = out.row(0)[i];
        float cy = out.row(1)[i];
        float bw = out.row(2)[i];
        float bh = out.row(3)[i];

        float x1 = clampf(unpad(cx - bw * 0.5f, padLeft, scale), 0.f, (float)srcW);
        float y1 = clampf(unpad(cy - bh * 0.5f, padTop,  scale), 0.f, (float)srcH);
        float x2 = clampf(unpad(cx + bw * 0.5f, padLeft, scale), 0.f, (float)srcW);
        float y2 = clampf(unpad(cy + bh * 0.5f, padTop,  scale), 0.f, (float)srcH);

        if (x2 - x1 < 1.f || y2 - y1 < 1.f) continue;

        // ── Box sanity filters ────────────────────────────────────────────────
        {
            float w = x2 - x1, h = y2 - y1;
            if (w < m_cfg.minBoxWidth)  continue;
            if (h < m_cfg.minBoxHeight) continue;
            float ar = (w > h) ? (w / h) : (h / w);
            if (ar > m_cfg.maxAspectRatio) continue;
            float frameArea = static_cast<float>(srcW) * static_cast<float>(srcH);
            if ((w * h) / frameArea > m_cfg.maxBoxAreaRatio) continue;
        }

        dets.push_back({ x1, y1, x2, y2, bestScore, bestCls });
    }

    return nms(std::move(dets));
}

// ─────────────────────────────────────────────────────────────────────────────
//  decodeEndToEnd  — Format B
//
//  NMS baked into model; scores are post-sigmoid [0,1].
//
//  Single-class (numClasses==1):  minor dim = 5  (x1,y1,x2,y2,score)
//  Multi-class  (numClasses >1):  minor dim = 6  (x1,y1,x2,y2,score,classId)
//
//  Handles both normal (h=numDets) and transposed (w=numDets) layouts.
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Detection> YoloDecoder::decodeEndToEnd(
    const ncnn::Mat& out,
    int srcW, int srcH, float scale, int padLeft, int padTop) const
{
    const int  fmtB       = 4 + m_cfg.numClasses + (m_cfg.numClasses > 1 ? 1 : 0);
    const bool transposed = (out.h == fmtB && out.w != fmtB);
    const bool hasClsCol  = (m_cfg.numClasses > 1);
    const int  numDets    = transposed ? out.w : out.h;

    std::vector<Detection> result;
    result.reserve(numDets);

    for (int i = 0; i < numDets; i++) {
        float x1, y1, x2, y2, conf;
        int   cls = 0;   // default class 0 for single-class models

        if (!transposed) {
            const float* row = out.row(i);
            x1   = row[0];
            y1   = row[1];
            x2   = row[2];
            y2   = row[3];
            conf = row[4];
            if (hasClsCol) cls = static_cast<int>(row[5]);
        } else {
            x1   = out.row(0)[i];
            y1   = out.row(1)[i];
            x2   = out.row(2)[i];
            y2   = out.row(3)[i];
            conf = out.row(4)[i];
            if (hasClsCol) cls = static_cast<int>(out.row(5)[i]);
        }

        if (conf < m_cfg.confThresh)            continue;
        if (cls < 0 || cls >= m_cfg.numClasses) continue;

        float ox1 = clampf(unpad(x1, padLeft, scale), 0.f, (float)srcW);
        float oy1 = clampf(unpad(y1, padTop,  scale), 0.f, (float)srcH);
        float ox2 = clampf(unpad(x2, padLeft, scale), 0.f, (float)srcW);
        float oy2 = clampf(unpad(y2, padTop,  scale), 0.f, (float)srcH);

        if (ox2 - ox1 < 1.f || oy2 - oy1 < 1.f) continue;

        // ── Box sanity filters ────────────────────────────────────────────────
        {
            float w = ox2 - ox1, h = oy2 - oy1;
            if (w < m_cfg.minBoxWidth)  continue;
            if (h < m_cfg.minBoxHeight) continue;
            float ar = (w > h) ? (w / h) : (h / w);
            if (ar > m_cfg.maxAspectRatio) continue;
            float frameArea = static_cast<float>(srcW) * static_cast<float>(srcH);
            if ((w * h) / frameArea > m_cfg.maxBoxAreaRatio) continue;
        }

        result.push_back({ ox1, oy1, ox2, oy2, conf, cls });
    }

    return result;   // NMS already applied by the model
}

// ─────────────────────────────────────────────────────────────────────────────
//  nms  — per-class non-maximum suppression (Format A only)
// ─────────────────────────────────────────────────────────────────────────────

std::vector<Detection> YoloDecoder::nms(std::vector<Detection> dets) const {
    if (dets.empty()) return dets;

    std::sort(dets.begin(), dets.end(),
        [](const Detection& a, const Detection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<bool>      suppressed(dets.size(), false);
    std::vector<Detection> result;
    result.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j])                       continue;
            if (dets[j].classId != dets[i].classId) continue;
            if (iou(dets[i], dets[j]) > m_cfg.nmsThresh)
                suppressed[j] = true;
        }
    }
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
//  debugTensor
// ─────────────────────────────────────────────────────────────────────────────

void YoloDecoder::debugTensor(const ncnn::Mat& out, int maxRows) const {
    printf("YoloDecoder::debugTensor  dims=%d  w=%d  h=%d  c=%d  elemsize=%zu\n",
           out.dims, out.w, out.h, out.c, out.elemsize);

    const int  fmtB     = 4 + m_cfg.numClasses + (m_cfg.numClasses > 1 ? 1 : 0);
    const bool couldBeB = (out.w == fmtB || out.h == fmtB);
    const bool couldBeA = (out.h == 4 + m_cfg.numClasses);

    printf("  numClasses=%d  fmtB_minor=%d  couldBeA=%d  couldBeB=%d\n",
           m_cfg.numClasses, fmtB, couldBeA, couldBeB);

    // Print raw rows
    printf("  Raw rows (first %d):\n", maxRows);
    for (int r = 0; r < std::min(out.h, maxRows); r++) {
        const float* row = out.row(r);
        int cols = std::min(out.w, 8);
        printf("    row[%d]:", r);
        for (int c = 0; c < cols; c++)
            printf("  %.3f", row[c]);
        if (out.w > 8) printf("  ...");
        printf("\n");
    }

    // If Format A: show top-5 anchors by sigmoid score
    if (couldBeA) {
        printf("  Format A — top %d anchors by sigmoid(score):\n", maxRows);
        std::vector<std::pair<float,int>> scored;
        scored.reserve(out.w);
        for (int i = 0; i < out.w; i++)
            scored.push_back({ sigmoid(out.row(4)[i]), i });
        std::sort(scored.begin(), scored.end(),
                  [](auto& a, auto& b){ return a.first > b.first; });
        int n = std::min((int)scored.size(), maxRows);
        for (int k = 0; k < n; k++) {
            int i = scored[k].second;
            printf("    anchor[%4d]: cx=%6.1f cy=%6.1f bw=%6.1f bh=%6.1f "
                   "score=%.4f\n",
                   i,
                   out.row(0)[i], out.row(1)[i],
                   out.row(2)[i], out.row(3)[i],
                   scored[k].first);
        }
    }
}
