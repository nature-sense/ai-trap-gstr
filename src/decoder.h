#pragma once

#include <cmath>
#include <vector>
#include <cstdint>

namespace ncnn { class Mat; }

// ─────────────────────────────────────────────────────────────────────────────
//  Detection  — one bounding box from one inference pass
// ─────────────────────────────────────────────────────────────────────────────
struct Detection {
    float x1, y1, x2, y2;   // corners in original image pixels
    float confidence;        // class score (sigmoid-activated)
    int   classId;           // index into your class name table
};

// ─────────────────────────────────────────────────────────────────────────────
//  YoloFormat  — controls which decoder path is used
//
//  Auto        — heuristic based on tensor shape + score sampling.
//                Reliable for multi-class models.  For single-class models
//                both formats produce h=5, so score sampling is used:
//                scores > 1.0 → Format A (raw logits); ≤ 1.0 → Format B.
//  AnchorGrid  — force Format A (YOLO11 default ncnn export)
//  EndToEnd    — force Format B (NMS baked into model)
// ─────────────────────────────────────────────────────────────────────────────
enum class YoloFormat { Auto, AnchorGrid, EndToEnd };

// ─────────────────────────────────────────────────────────────────────────────
//  DecoderConfig
// ─────────────────────────────────────────────────────────────────────────────
struct DecoderConfig {
    float      confThresh  = 0.45f;            // minimum score to keep
    float      nmsThresh   = 0.45f;            // IoU threshold for NMS (Format A only)
    int        numClasses  = 80;               // set to match your model
    int        modelWidth  = 640;              // must match the ncnn export imgsz
    int        modelHeight = 640;
    YoloFormat format      = YoloFormat::Auto; // override auto-detection if needed
};

// ─────────────────────────────────────────────────────────────────────────────
//  YoloDecoder
//
//  Decodes raw ncnn output tensors from YOLO11n into detections.
//
//  FORMAT A  anchor-grid  (YOLO11n default ncnn export)
//    dims=2,  h = 4 + numClasses,  w = numAnchors (~2100 for 320 input)
//    Column i:  rows 0-3 = cx,cy,bw,bh (model pixels)
//               rows 4+  = raw logit class scores (sigmoid applied in decoder)
//    NMS applied after decoding.
//
//  FORMAT B  end-to-end  (NMS baked into model at export time)
//    dims=2,  minor dimension = 4 + numClasses + (numClasses>1 ? 1 : 0)
//    Normal:     h=numDets, w=minor  → row i: [x1,y1,x2,y2,score(,cls)]
//    Transposed: h=minor, w=numDets
//    Scores are post-sigmoid [0,1].  NMS NOT applied in decoder.
//
//  A dims=3 batch tensor is automatically squeezed before dispatch.
// ─────────────────────────────────────────────────────────────────────────────
class YoloDecoder {
public:
    explicit YoloDecoder(const DecoderConfig& cfg = {});

    // Decode a raw ncnn output tensor.
    // srcW/srcH   — original camera frame dimensions in pixels
    // scale       — letterbox scale factor from preprocessing
    // padLeft/Top — letterbox padding in model pixels
    // Returns detections in original image pixel coordinates.
    std::vector<Detection> decode(
        const ncnn::Mat& out,
        int   srcW,    int srcH,
        float scale,
        int   padLeft, int padTop) const;

    const DecoderConfig& config() const { return m_cfg; }
    void setConfig(const DecoderConfig& cfg) { m_cfg = cfg; }

    // Print tensor shape and sample rows — useful when verifying a new export.
    void debugTensor(const ncnn::Mat& out, int maxRows = 5) const;

private:
    DecoderConfig m_cfg;

    std::vector<Detection> dispatch(
        const ncnn::Mat& out,
        int srcW, int srcH, float scale, int padLeft, int padTop) const;

    std::vector<Detection> decodeAnchorGrid(
        const ncnn::Mat& out,
        int srcW, int srcH, float scale, int padLeft, int padTop) const;

    std::vector<Detection> decodeEndToEnd(
        const ncnn::Mat& out,
        int srcW, int srcH, float scale, int padLeft, int padTop) const;

    std::vector<Detection> nms(std::vector<Detection> dets) const;

    static float iou(const Detection& a, const Detection& b);
    static float clampf(float v, float lo, float hi);
    static float unpad(float coord, int pad, float scale);
};
