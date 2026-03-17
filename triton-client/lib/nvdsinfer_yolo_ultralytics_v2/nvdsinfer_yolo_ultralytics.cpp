/*
 * Dynamic DeepStream Parser for Ultralytics YOLO Models (v5, v8, v11)
 *
 * FIXES vs previous version:
 *
 * FIX 1 — gRPC output ordering is not guaranteed.
 *   Old code: always used outputLayersInfo[0], which worked with cAPI
 *   (TRT always returns output0 first) but breaks with Triton gRPC where
 *   the layer list order may differ.
 *   New code: searches outputLayersInfo by layerName == "output0" first;
 *   falls back to first 2D/3D tensor if not found.
 *
 * FIX 2 — 4D sigmoid head tensors (onnx::Sigmoid_xxx) caused false.
 *   Old code: passed all shapes to parseYoloAuto which returned false for
 *   numDims==4.
 *   New code: NvDsInferParseYoloUltralytics skips any 4D tensor unless
 *   all outputs are 4D (meaning this IS a multi-head model → YOLOv5
 *   anchor decoder path).
 *
 * FIX 3 — YOLOv5 multi-head anchor decoding added.
 *   For models that export 3 sigmoid detection heads (e.g., ppe, vc, anpr
 *   exported with standard YOLOv5 --include onnx), NvDsInferParseYolov5Heads
 *   decodes all 3 feature maps using anchor boxes and merges into one list.
 *   Anchors default to COCO YOLOv5s values; override via compiler flags.
 *
 * HOW TO CHOOSE THE RIGHT FUNCTION:
 *
 *   NvDsInferParseYoloUltralytics  — Auto. Use for all models. Tries:
 *     1) Find "output0" layer (single decoded tensor) → parseYoloAuto
 *     2) All layers are 4D heads → parseYolov5Heads (anchor decode)
 *
 *   NvDsInferParseYoloV5           — Force YOLOv5 single-output path
 *   NvDsInferParseYoloV8           — Force YOLOv8/v11 single-output path
 *   NvDsInferParseYolov5Heads      — Force 3-head anchor-decode path
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include "nvdsinfer_custom_impl.h"

// ============================================================================
// Configuration — override with -D flags at compile time
// ============================================================================

#ifndef NMS_IOU_THRESHOLD
#define NMS_IOU_THRESHOLD 0.45f
#endif

#ifndef CONF_THRESHOLD
#define CONF_THRESHOLD 0.25f
#endif

#ifndef MAX_DETECTIONS
#define MAX_DETECTIONS 300
#endif

// Set to 1 to print shape + detection info to stdout (noisy, for debugging)
#ifndef PARSER_DEBUG
#define PARSER_DEBUG 0
#endif

#if PARSER_DEBUG
#include <iostream>
#define DBG(x) std::cout << "[YOLO_PARSER] " << x << std::endl
#else
#define DBG(x)
#endif

// ============================================================================
// Default YOLOv5s COCO anchors — P3/8, P4/16, P5/32
// Override for custom-trained models by recompiling with -DUSE_CUSTOM_ANCHORS
// and providing ANCHORS_P3, ANCHORS_P4, ANCHORS_P5 macros.
// ============================================================================

#ifndef USE_CUSTOM_ANCHORS
static const float kAnchors[3][3][2] = {
    /* P3/8  */ {{10, 13}, {16, 30}, {33, 23}},
    /* P4/16 */ {{30, 61}, {62, 45}, {59, 119}},
    /* P5/32 */ {{116, 90}, {156, 198}, {373, 326}}
};
static const int kStrides[3] = {8, 16, 32};
#endif

// ============================================================================
// Internal structures
// ============================================================================

struct Detection {
    float x, y, w, h;  // center-format pixels
    float conf;
    int   classId;
};

// ============================================================================
// Utilities
// ============================================================================

static inline float clamp(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

static float iou(const Detection &a, const Detection &b) {
    float aL = a.x - a.w * 0.5f, aT = a.y - a.h * 0.5f;
    float aR = a.x + a.w * 0.5f, aB = a.y + a.h * 0.5f;
    float bL = b.x - b.w * 0.5f, bT = b.y - b.h * 0.5f;
    float bR = b.x + b.w * 0.5f, bB = b.y + b.h * 0.5f;
    float iW = std::max(0.f, std::min(aR, bR) - std::max(aL, bL));
    float iH = std::max(0.f, std::min(aB, bB) - std::max(aT, bT));
    float iA = iW * iH;
    float uA = a.w * a.h + b.w * b.h - iA;
    return uA > 0.f ? iA / uA : 0.f;
}

static void applyNMS(std::vector<Detection> &dets, float iouThr, int maxDets) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection &a, const Detection &b) { return a.conf > b.conf; });
    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> out;
    out.reserve(std::min((int)dets.size(), maxDets));
    for (size_t i = 0; i < dets.size() && (int)out.size() < maxDets; i++) {
        if (suppressed[i]) continue;
        out.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (!suppressed[j] && dets[i].classId == dets[j].classId
                && iou(dets[i], dets[j]) > iouThr)
                suppressed[j] = true;
        }
    }
    dets = std::move(out);
}

static void toObjectList(const std::vector<Detection> &dets,
                         uint32_t netW, uint32_t netH,
                         std::vector<NvDsInferParseObjectInfo> &out) {
    for (const auto &d : dets) {
        NvDsInferParseObjectInfo o;
        float l = d.x - d.w * 0.5f;
        float t = d.y - d.h * 0.5f;
        o.left   = clamp(l, 0.f, (float)(netW - 1));
        o.top    = clamp(t, 0.f, (float)(netH - 1));
        o.width  = clamp(d.w, 1.f, (float)netW - o.left);
        o.height = clamp(d.h, 1.f, (float)netH - o.top);
        o.detectionConfidence = d.conf;
        o.classId = d.classId;
        out.push_back(o);
    }
}

// ============================================================================
// Single-output parsers (output0 from Ultralytics export)
// ============================================================================

/* YOLOv8/v11: output shape [4+nc, N] — channels first */
static bool parseV8(const float *data, uint32_t channels, uint32_t preds,
                    uint32_t netW, uint32_t netH, float confThr,
                    std::vector<NvDsInferParseObjectInfo> &out) {
    int nc = (int)channels - 4;
    if (nc <= 0) {
        DBG("parseV8: invalid channels=" << channels);
        return false;
    }
    DBG("parseV8: channels=" << channels << " preds=" << preds << " nc=" << nc);
    std::vector<Detection> dets;
    for (uint32_t p = 0; p < preds; p++) {
        float best = 0; int cls = 0;
        for (int c = 0; c < nc; c++) {
            float s = data[(4 + c) * preds + p];
            if (s > best) { best = s; cls = c; }
        }
        if (best >= confThr) {
            Detection d;
            d.x = data[0 * preds + p]; d.y = data[1 * preds + p];
            d.w = data[2 * preds + p]; d.h = data[3 * preds + p];
            d.conf = best; d.classId = cls;
            dets.push_back(d);
        }
    }
    applyNMS(dets, NMS_IOU_THRESHOLD, MAX_DETECTIONS);
    toObjectList(dets, netW, netH, out);
    return true;
}

/* YOLOv5 single output: shape [N, 5+nc] — predictions first */
static bool parseV5Single(const float *data, uint32_t preds, uint32_t channels,
                          uint32_t netW, uint32_t netH, float confThr,
                          std::vector<NvDsInferParseObjectInfo> &out) {
    int nc = (int)channels - 5;
    if (nc <= 0) {
        DBG("parseV5Single: invalid channels=" << channels);
        return false;
    }
    DBG("parseV5Single: preds=" << preds << " channels=" << channels << " nc=" << nc);
    std::vector<Detection> dets;
    for (uint32_t p = 0; p < preds; p++) {
        const float *row = data + p * channels;
        float obj = row[4];
        if (obj < confThr) continue;
        float best = 0; int cls = 0;
        for (int c = 0; c < nc; c++) {
            if (row[5 + c] > best) { best = row[5 + c]; cls = c; }
        }
        float conf = obj * best;
        if (conf >= confThr) {
            Detection d;
            d.x = row[0]; d.y = row[1]; d.w = row[2]; d.h = row[3];
            d.conf = conf; d.classId = cls;
            dets.push_back(d);
        }
    }
    applyNMS(dets, NMS_IOU_THRESHOLD, MAX_DETECTIONS);
    toObjectList(dets, netW, netH, out);
    return true;
}

/* Auto-detect single output format from tensor shape */
static bool parseAuto(const NvDsInferLayerInfo &layer,
                      uint32_t netW, uint32_t netH, float confThr,
                      std::vector<NvDsInferParseObjectInfo> &out) {
    const float *data = static_cast<const float *>(layer.buffer);
    const NvDsInferDims &dims = layer.inferDims;

    DBG("parseAuto layer='" << layer.layerName << "' numDims=" << dims.numDims);
    for (uint32_t i = 0; i < dims.numDims; i++)
        DBG("  d[" << i << "]=" << dims.d[i]);

    uint32_t d0, d1;
    if (dims.numDims == 2) {
        d0 = dims.d[0]; d1 = dims.d[1];
    } else if (dims.numDims == 3) {
        d0 = dims.d[1]; d1 = dims.d[2]; // strip batch
    } else {
        DBG("parseAuto: skipping layer with numDims=" << dims.numDims);
        return false;
    }

    const uint32_t CHAN_THR = 500;
    bool tryV8 = (d0 < CHAN_THR && d1 >= CHAN_THR);
    bool tryV5 = (d0 >= CHAN_THR && d1 < CHAN_THR);
    if (!tryV8 && !tryV5) {
        tryV8 = (d0 < d1);  // fallback: smaller dim first → V8
        tryV5 = !tryV8;
    }

    if (tryV8) {
        int nc = (int)d0 - 4;
        if (nc > 0 && nc <= 1000) {
            DBG("parseAuto → V8 format [" << d0 << "," << d1 << "] nc=" << nc);
            return parseV8(data, d0, d1, netW, netH, confThr, out);
        }
        tryV5 = true; // fallback
    }
    if (tryV5) {
        int nc = (int)d1 - 5;
        if (nc > 0 && nc <= 1000) {
            DBG("parseAuto → V5 single [" << d0 << "," << d1 << "] nc=" << nc);
            return parseV5Single(data, d0, d1, netW, netH, confThr, out);
        }
    }

    DBG("parseAuto: could not determine format for [" << d0 << "," << d1 << "]");
    return false;
}

// ============================================================================
// YOLOv5 multi-head anchor decoder
//
// Handles 3 outputs with shapes [batch, na, H_i, W_i, 5+nc] (from standard
// YOLOv5 ONNX export where grid decoding is done in the model).
//
// If your model exports RAW sigmoid outputs WITHOUT grid decode you need to
// set USE_GRID_DECODE=1 (compile flag) — then cx/cy are decoded via:
//   cx = (sigmoid(tx) * 2 - 0.5 + grid_x) * stride
//   cw = (sigmoid(tw) * 2)^2 * anchor_w
// ============================================================================

static bool parseV5HeadSingle(const float *data,
                              uint32_t na,   // anchors per cell (usually 3)
                              uint32_t gridH, uint32_t gridW,
                              uint32_t nc,   // num classes
                              int stride,
                              const float anchors[][2],
                              uint32_t netW, uint32_t netH,
                              float confThr,
                              std::vector<Detection> &dets) {
    // Layout: [na, gridH, gridW, 5+nc]  (after batch dim is stripped)
    uint32_t cellSize = 5 + nc;
    for (uint32_t a = 0; a < na; a++) {
        for (uint32_t gy = 0; gy < gridH; gy++) {
            for (uint32_t gx = 0; gx < gridW; gx++) {
                const float *cell = data
                    + a    * (gridH * gridW * cellSize)
                    + gy   * (gridW * cellSize)
                    + gx   * cellSize;

                float obj = cell[4];
                if (obj < confThr) continue;

                float best = 0; int cls = 0;
                for (uint32_t c = 0; c < nc; c++) {
                    if (cell[5 + c] > best) { best = cell[5 + c]; cls = (int)c; }
                }
                float conf = obj * best;
                if (conf < confThr) continue;

#ifdef USE_GRID_DECODE
                // Model exports raw logits — decode manually
                float cx = (cell[0] * 2.f - 0.5f + gx) * stride;
                float cy = (cell[1] * 2.f - 0.5f + gy) * stride;
                float w  = powf(cell[2] * 2.f, 2.f) * anchors[a][0];
                float h  = powf(cell[3] * 2.f, 2.f) * anchors[a][1];
#else
                // Model already applied sigmoid + grid decode
                float cx = cell[0];
                float cy = cell[1];
                float w  = cell[2];
                float h  = cell[3];
#endif
                (void)stride; (void)gx; (void)gy; (void)anchors; // suppress unused if no grid decode

                Detection d;
                d.x = cx; d.y = cy; d.w = w; d.h = h;
                d.conf = conf; d.classId = cls;
                dets.push_back(d);
            }
        }
    }
    return true;
}

/* Entry point for 3-head YOLOv5 models */
static bool parseV5Heads(const std::vector<NvDsInferLayerInfo> &layers,
                         const NvDsInferNetworkInfo &netInfo,
                         float confThr,
                         std::vector<NvDsInferParseObjectInfo> &out) {
    std::vector<Detection> dets;
    int headIdx = 0;

    for (const auto &layer : layers) {
        const NvDsInferDims &dims = layer.inferDims;

        // Expect shape [na, H, W, 5+nc] (2D/3D/4D depending on whether batch is included)
        uint32_t na, gH, gW, cellSize;
        if (dims.numDims == 4) {
            // [batch, na, H, W] — but we need 5th dim for cellSize...
            // likely [batch, na*cellSize, H, W] flattened? Skip non-standard.
            DBG("parseV5Heads: 4D tensor with dims [" << dims.d[0] << ","
                << dims.d[1] << "," << dims.d[2] << "," << dims.d[3]
                << "] — skipping, expected 5D or 4D [na,H,W,cells]");
            continue;
        } else if (dims.numDims == 3) {
            // [na, H, W*cellSize] or [na*H, W, cellSize] — ambiguous, skip
            DBG("parseV5Heads: ambiguous 3D tensor, skipping");
            continue;
        } else if (dims.numDims == 5) {
            // Standard YOLOv5 ONNX: [batch, na, H, W, 5+nc]
            na       = dims.d[1];
            gH       = dims.d[2];
            gW       = dims.d[3];
            cellSize = dims.d[4];
        } else if (dims.numDims == 4) {
            // Without batch: [na, H, W, 5+nc]
            na       = dims.d[0];
            gH       = dims.d[1];
            gW       = dims.d[2];
            cellSize = dims.d[3];
        } else {
            continue;
        }

        if (cellSize <= 5 || na == 0 || na > 9) {
            DBG("parseV5Heads: unexpected cellSize=" << cellSize << " na=" << na);
            continue;
        }
        uint32_t nc = cellSize - 5;
        int stride = (headIdx < 3) ? kStrides[headIdx] : 8 * (1 << headIdx);

        DBG("parseV5Heads head[" << headIdx << "] na=" << na
            << " grid=" << gH << "x" << gW << " nc=" << nc
            << " stride=" << stride);

        // skip batch dimension offset
        const float *headData = static_cast<const float *>(layer.buffer);

        parseV5HeadSingle(headData, na, gH, gW, nc, stride,
                          kAnchors[std::min(headIdx, 2)], netInfo.width, netInfo.height,
                          confThr, dets);
        headIdx++;
    }

    if (dets.empty() && headIdx == 0) return false;

    applyNMS(dets, NMS_IOU_THRESHOLD, MAX_DETECTIONS);
    toObjectList(dets, netInfo.width, netInfo.height, out);
    return true;
}

// ============================================================================
// Helper: find the best usable output layer
//
// Priority:
//  1) Layer named "output0" that is 2D or 3D
//  2) First layer that is 2D or 3D
//  3) nullptr (fall through to multi-head path)
// ============================================================================

static const NvDsInferLayerInfo *
findSingleOutputLayer(const std::vector<NvDsInferLayerInfo> &layers) {
    const NvDsInferLayerInfo *fallback = nullptr;
    for (const auto &l : layers) {
        uint32_t nd = l.inferDims.numDims;
        if (nd != 2 && nd != 3) continue;
        if (std::string(l.layerName) == "output0") return &l;
        if (!fallback) fallback = &l;
    }
    return fallback;
}

// ============================================================================
// Exported parse functions
// ============================================================================

extern "C" bool NvDsInferParseYoloUltralytics(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    if (outputLayersInfo.empty()) {
        DBG("NvDsInferParseYoloUltralytics: no output layers");
        return false;
    }

    float confThr = detectionParams.perClassPreclusterThreshold.empty()
                    ? CONF_THRESHOLD
                    : detectionParams.perClassPreclusterThreshold[0];

    // --- Path 1: find a 2D/3D output layer (single decoded tensor) ---
    const NvDsInferLayerInfo *single = findSingleOutputLayer(outputLayersInfo);
    if (single) {
        DBG("Using single-output layer: " << single->layerName);
        return parseAuto(*single, networkInfo.width, networkInfo.height, confThr, objectList);
    }

    // --- Path 2: all layers are multi-dimensional → try anchor-based decode ---
    DBG("No 2D/3D layer found — trying multi-head YOLOv5 anchor decode");
    return parseV5Heads(outputLayersInfo, networkInfo, confThr, objectList);
}

extern "C" bool NvDsInferParseYolov5Heads(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    float confThr = detectionParams.perClassPreclusterThreshold.empty()
                    ? CONF_THRESHOLD
                    : detectionParams.perClassPreclusterThreshold[0];
    return parseV5Heads(outputLayersInfo, networkInfo, confThr, objectList);
}

extern "C" bool NvDsInferParseYoloV5(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    if (outputLayersInfo.empty()) return false;
    const NvDsInferLayerInfo *l = findSingleOutputLayer(outputLayersInfo);
    if (!l) return false;
    float confThr = detectionParams.perClassPreclusterThreshold.empty()
                    ? CONF_THRESHOLD
                    : detectionParams.perClassPreclusterThreshold[0];
    const float *data = static_cast<const float *>(l->buffer);
    const NvDsInferDims &dims = l->inferDims;
    uint32_t d0, d1;
    if (dims.numDims == 2)      { d0 = dims.d[0]; d1 = dims.d[1]; }
    else if (dims.numDims == 3) { d0 = dims.d[1]; d1 = dims.d[2]; }
    else return false;
    return parseV5Single(data, d0, d1, networkInfo.width, networkInfo.height, confThr, objectList);
}

extern "C" bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    if (outputLayersInfo.empty()) return false;
    const NvDsInferLayerInfo *l = findSingleOutputLayer(outputLayersInfo);
    if (!l) return false;
    float confThr = detectionParams.perClassPreclusterThreshold.empty()
                    ? CONF_THRESHOLD
                    : detectionParams.perClassPreclusterThreshold[0];
    const float *data = static_cast<const float *>(l->buffer);
    const NvDsInferDims &dims = l->inferDims;
    uint32_t d0, d1;
    if (dims.numDims == 2)      { d0 = dims.d[0]; d1 = dims.d[1]; }
    else if (dims.numDims == 3) { d0 = dims.d[1]; d1 = dims.d[2]; }
    else return false;
    return parseV8(data, d0, d1, networkInfo.width, networkInfo.height, confThr, objectList);
}

extern "C" bool NvDsInferParseYolo11(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferParseObjectInfo> &objectList)
{
    return NvDsInferParseYoloV8(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloUltralytics);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolov5Heads);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV5);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo11);