/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: MIT
 *
 * Dynamic DeepStream Parser for Ultralytics YOLO Models (v5, v8, v11)
 * 
 * KEY FEATURE: Automatically detects number of classes from output tensor shape.
 * Works with ANY number of classes - no need to match config exactly.
 * 
 * Supports:
 * - YOLOv5: Output [batch, num_preds, 5+num_classes] where 5 = 4(xywh) + 1(obj)
 * - YOLOv8/v11: Output [batch, 4+num_classes, num_preds]
 * 
 * Auto-detection logic:
 * - Small dim first (< 1000) + Large dim second (> 1000) = YOLOv8/v11 format
 * - Large dim first (> 1000) + Small dim second (< 1000) = YOLOv5 format
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include "nvdsinfer_custom_impl.h"

// ============================================================================
// Configuration - Override via compiler flags if needed
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

// Debug output (set to 1 for debugging)
#ifndef PARSER_DEBUG
#define PARSER_DEBUG 0
#endif

#if PARSER_DEBUG
#include <iostream>
#define DBG_PRINT(x) std::cout << "[YOLO_PARSER] " << x << std::endl
#else
#define DBG_PRINT(x)
#endif

// ============================================================================
// Internal Detection Structure
// ============================================================================

struct Detection {
    float x;      // center x (pixels)
    float y;      // center y (pixels)
    float w;      // width (pixels)
    float h;      // height (pixels)
    float conf;   // confidence score
    int classId;  // class index
};

// ============================================================================
// Utility Functions
// ============================================================================

static inline float clamp(float val, float minVal, float maxVal) {
    return std::max(minVal, std::min(val, maxVal));
}

static float iouCalc(const Detection& a, const Detection& b) {
    // Convert center format to corner format
    float aLeft = a.x - a.w / 2.0f;
    float aTop = a.y - a.h / 2.0f;
    float aRight = a.x + a.w / 2.0f;
    float aBottom = a.y + a.h / 2.0f;
    
    float bLeft = b.x - b.w / 2.0f;
    float bTop = b.y - b.h / 2.0f;
    float bRight = b.x + b.w / 2.0f;
    float bBottom = b.y + b.h / 2.0f;
    
    // Intersection
    float interLeft = std::max(aLeft, bLeft);
    float interTop = std::max(aTop, bTop);
    float interRight = std::min(aRight, bRight);
    float interBottom = std::min(aBottom, bBottom);
    
    float interWidth = std::max(0.0f, interRight - interLeft);
    float interHeight = std::max(0.0f, interBottom - interTop);
    float interArea = interWidth * interHeight;
    
    // Union
    float aArea = a.w * a.h;
    float bArea = b.w * b.h;
    float unionArea = aArea + bArea - interArea;
    
    return (unionArea > 0.0f) ? (interArea / unionArea) : 0.0f;
}

static void nmsSort(std::vector<Detection>& detections, float iouThreshold, int maxDets) {
    // Sort by confidence (descending)
    std::sort(detections.begin(), detections.end(),
              [](const Detection& a, const Detection& b) {
                  return a.conf > b.conf;
              });
    
    std::vector<Detection> result;
    result.reserve(std::min((int)detections.size(), maxDets));
    
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size() && (int)result.size() < maxDets; ++i) {
        if (suppressed[i]) continue;
        
        result.push_back(detections[i]);
        
        // Suppress overlapping detections of the same class
        for (size_t j = i + 1; j < detections.size(); ++j) {
            if (suppressed[j]) continue;
            if (detections[i].classId != detections[j].classId) continue;
            
            if (iouCalc(detections[i], detections[j]) > iouThreshold) {
                suppressed[j] = true;
            }
        }
    }
    
    detections = std::move(result);
}

// ============================================================================
// YOLOv8/v11 Parser (Format: [batch, 4+num_classes, num_predictions])
// Dynamically infers num_classes from tensor shape
// ============================================================================

static bool parseYoloV8Format(
    const float* output,
    const uint32_t numChannels,     // 4 + num_classes (e.g., 5 for 1 class, 84 for 80 classes)
    const uint32_t numPredictions,  // e.g., 8400 for 640x640 input
    const uint32_t networkWidth,
    const uint32_t networkHeight,
    const float confThreshold,
    const float nmsThreshold,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // Dynamically calculate number of classes from tensor shape
    // YOLOv8/v11: channels = 4 (xywh) + num_classes
    const int numClasses = numChannels - 4;
    
    if (numClasses <= 0) {
        DBG_PRINT("ERROR: Invalid numClasses calculated: " << numClasses 
                  << " (channels=" << numChannels << ")");
        return false;
    }
    
    DBG_PRINT("YOLOv8/v11 format: channels=" << numChannels 
              << " predictions=" << numPredictions 
              << " classes=" << numClasses << " (auto-detected)");
    
    std::vector<Detection> detections;
    detections.reserve(1000);
    
    // Output layout: [4+num_classes, num_predictions]
    // Row 0: x coordinates for all predictions
    // Row 1: y coordinates for all predictions  
    // Row 2: w coordinates for all predictions
    // Row 3: h coordinates for all predictions
    // Rows 4+: class scores for all predictions
    
    for (uint32_t p = 0; p < numPredictions; ++p) {
        // Extract coordinates
        float cx = output[0 * numPredictions + p];
        float cy = output[1 * numPredictions + p];
        float w  = output[2 * numPredictions + p];
        float h  = output[3 * numPredictions + p];
        
        // Find best class
        float maxClassScore = 0.0f;
        int bestClassId = 0;
        
        for (int c = 0; c < numClasses; ++c) {
            float score = output[(4 + c) * numPredictions + p];
            if (score > maxClassScore) {
                maxClassScore = score;
                bestClassId = c;
            }
        }
        
        // Confidence is the class score directly (no objectness in v8/v11)
        float confidence = maxClassScore;
        
        if (confidence >= confThreshold) {
            Detection det;
            det.x = cx;
            det.y = cy;
            det.w = w;
            det.h = h;
            det.conf = confidence;
            det.classId = bestClassId;
            detections.push_back(det);
        }
    }
    
    DBG_PRINT("Pre-NMS detections: " << detections.size());
    
    // Apply NMS
    nmsSort(detections, nmsThreshold, MAX_DETECTIONS);
    
    DBG_PRINT("Post-NMS detections: " << detections.size());
    
    // Convert to DeepStream format
    for (const auto& det : detections) {
        NvDsInferParseObjectInfo obj;
        
        float left = det.x - det.w / 2.0f;
        float top = det.y - det.h / 2.0f;
        
        obj.left = clamp(left, 0.0f, (float)networkWidth - 1.0f);
        obj.top = clamp(top, 0.0f, (float)networkHeight - 1.0f);
        obj.width = clamp(det.w, 1.0f, (float)networkWidth - obj.left);
        obj.height = clamp(det.h, 1.0f, (float)networkHeight - obj.top);
        obj.detectionConfidence = det.conf;
        obj.classId = det.classId;
        
        objectList.push_back(obj);
    }
    
    return true;
}

// ============================================================================
// YOLOv5 Parser (Format: [batch, num_predictions, 5+num_classes])
// Dynamically infers num_classes from tensor shape
// ============================================================================

static bool parseYoloV5Format(
    const float* output,
    const uint32_t numPredictions,  // e.g., 25200 for 640x640 input
    const uint32_t numChannels,     // 5 + num_classes (e.g., 6 for 1 class, 85 for 80 classes)
    const uint32_t networkWidth,
    const uint32_t networkHeight,
    const float confThreshold,
    const float nmsThreshold,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    // Dynamically calculate number of classes from tensor shape
    // YOLOv5: channels = 4 (xywh) + 1 (objectness) + num_classes
    const int numClasses = numChannels - 5;
    
    if (numClasses <= 0) {
        DBG_PRINT("ERROR: Invalid numClasses calculated: " << numClasses 
                  << " (channels=" << numChannels << ")");
        return false;
    }
    
    DBG_PRINT("YOLOv5 format: predictions=" << numPredictions 
              << " channels=" << numChannels 
              << " classes=" << numClasses << " (auto-detected)");
    
    std::vector<Detection> detections;
    detections.reserve(1000);
    
    // Output layout: [num_predictions, 5+num_classes]
    // Each row: [cx, cy, w, h, obj_conf, class_0, class_1, ..., class_N]
    
    const int stride = numChannels;
    
    for (uint32_t p = 0; p < numPredictions; ++p) {
        const float* row = output + p * stride;
        
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        float objectness = row[4];
        
        // Early rejection based on objectness
        if (objectness < confThreshold) continue;
        
        // Find best class
        float maxClassScore = 0.0f;
        int bestClassId = 0;
        
        for (int c = 0; c < numClasses; ++c) {
            float score = row[5 + c];
            if (score > maxClassScore) {
                maxClassScore = score;
                bestClassId = c;
            }
        }
        
        // Final confidence = objectness * class_score
        float confidence = objectness * maxClassScore;
        
        if (confidence >= confThreshold) {
            Detection det;
            det.x = cx;
            det.y = cy;
            det.w = w;
            det.h = h;
            det.conf = confidence;
            det.classId = bestClassId;
            detections.push_back(det);
        }
    }
    
    DBG_PRINT("Pre-NMS detections: " << detections.size());
    
    // Apply NMS
    nmsSort(detections, nmsThreshold, MAX_DETECTIONS);
    
    DBG_PRINT("Post-NMS detections: " << detections.size());
    
    // Convert to DeepStream format
    for (const auto& det : detections) {
        NvDsInferParseObjectInfo obj;
        
        float left = det.x - det.w / 2.0f;
        float top = det.y - det.h / 2.0f;
        
        obj.left = clamp(left, 0.0f, (float)networkWidth - 1.0f);
        obj.top = clamp(top, 0.0f, (float)networkHeight - 1.0f);
        obj.width = clamp(det.w, 1.0f, (float)networkWidth - obj.left);
        obj.height = clamp(det.h, 1.0f, (float)networkHeight - obj.top);
        obj.detectionConfidence = det.conf;
        obj.classId = det.classId;
        
        objectList.push_back(obj);
    }
    
    return true;
}

// ============================================================================
// Auto-detect Format and Parse
// Uses dimension ratios to determine format, not config values
// ============================================================================

static bool parseYoloAuto(
    const NvDsInferLayerInfo& outputLayer,
    const NvDsInferNetworkInfo& networkInfo,
    const NvDsInferParseDetectionParams& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    const float* output = static_cast<const float*>(outputLayer.buffer);
    const NvDsInferDims& dims = outputLayer.inferDims;
    
    DBG_PRINT("Output layer: " << outputLayer.layerName);
    DBG_PRINT("Dimensions: numDims=" << dims.numDims);
    for (uint32_t i = 0; i < dims.numDims; ++i) {
        DBG_PRINT("  dim[" << i << "] = " << dims.d[i]);
    }
    
    // Get the two relevant dimensions (skip batch if present)
    uint32_t dim0, dim1;
    
    if (dims.numDims == 2) {
        dim0 = dims.d[0];
        dim1 = dims.d[1];
    } else if (dims.numDims == 3) {
        // Skip batch dimension
        dim0 = dims.d[1];
        dim1 = dims.d[2];
    } else {
        DBG_PRINT("ERROR: Unexpected number of dimensions: " << dims.numDims);
        return false;
    }
    
    // Get confidence threshold from config, or use default
    const float confThreshold = detectionParams.perClassPreclusterThreshold.size() > 0 
                                ? detectionParams.perClassPreclusterThreshold[0] 
                                : CONF_THRESHOLD;
    
    DBG_PRINT("Tensor shape: [" << dim0 << ", " << dim1 << "]");
    DBG_PRINT("Confidence threshold: " << confThreshold);
    
    /*
     * Auto-detection logic:
     * 
     * YOLOv8/v11 format: [channels, predictions] where channels << predictions
     *   - channels = 4 + num_classes (small number, e.g., 5 to ~100)
     *   - predictions = 8400 for 640x640, scales with input size (large number)
     *   - Example: [84, 8400] for 80 classes, [5, 8400] for 1 class
     * 
     * YOLOv5 format: [predictions, channels] where predictions >> channels
     *   - predictions = 25200 for 640x640 (large number)
     *   - channels = 5 + num_classes (small number, e.g., 6 to ~100)
     *   - Example: [25200, 85] for 80 classes, [25200, 6] for 1 class
     * 
     * Detection strategy:
     *   - If dim0 < dim1 and dim0 < 1000: YOLOv8/v11 format
     *   - If dim0 > dim1 and dim1 < 1000: YOLOv5 format
     *   - Otherwise: Use heuristics based on typical YOLO dimensions
     */
    
    // Threshold to distinguish between "channels" and "predictions"
    // Channels are typically < 200 (even with 100+ classes)
    // Predictions are typically > 1000 (8400, 25200, etc.)
    const uint32_t CHANNEL_THRESHOLD = 500;
    
    bool isV8Format = false;
    bool isV5Format = false;
    
    if (dim0 < CHANNEL_THRESHOLD && dim1 >= CHANNEL_THRESHOLD) {
        // dim0 is small (channels), dim1 is large (predictions) -> YOLOv8/v11
        isV8Format = true;
    } else if (dim0 >= CHANNEL_THRESHOLD && dim1 < CHANNEL_THRESHOLD) {
        // dim0 is large (predictions), dim1 is small (channels) -> YOLOv5
        isV5Format = true;
    } else if (dim0 < dim1) {
        // Both dimensions are similar size, but dim0 < dim1
        // More likely YOLOv8/v11 format
        isV8Format = true;
        DBG_PRINT("Warning: Ambiguous dimensions, assuming YOLOv8/v11 format");
    } else {
        // dim0 >= dim1
        // More likely YOLOv5 format
        isV5Format = true;
        DBG_PRINT("Warning: Ambiguous dimensions, assuming YOLOv5 format");
    }
    
    // Additional validation: check if dimensions make sense
    if (isV8Format) {
        int inferredClasses = dim0 - 4;
        if (inferredClasses <= 0 || inferredClasses > 1000) {
            DBG_PRINT("Warning: YOLOv8 format would give " << inferredClasses 
                      << " classes, trying YOLOv5 format instead");
            isV8Format = false;
            isV5Format = true;
        }
    }
    
    if (isV5Format) {
        int inferredClasses = dim1 - 5;
        if (inferredClasses <= 0 || inferredClasses > 1000) {
            DBG_PRINT("Warning: YOLOv5 format would give " << inferredClasses 
                      << " classes, trying YOLOv8 format instead");
            isV5Format = false;
            isV8Format = true;
        }
    }
    
    // Parse based on detected format
    if (isV8Format) {
        DBG_PRINT("Detected: YOLOv8/v11 format [channels=" << dim0 << ", predictions=" << dim1 << "]");
        DBG_PRINT("Inferred classes: " << (dim0 - 4));
        return parseYoloV8Format(
            output, dim0, dim1,
            networkInfo.width, networkInfo.height,
            confThreshold, NMS_IOU_THRESHOLD,
            objectList);
    } else if (isV5Format) {
        DBG_PRINT("Detected: YOLOv5 format [predictions=" << dim0 << ", channels=" << dim1 << "]");
        DBG_PRINT("Inferred classes: " << (dim1 - 5));
        return parseYoloV5Format(
            output, dim0, dim1,
            networkInfo.width, networkInfo.height,
            confThreshold, NMS_IOU_THRESHOLD,
            objectList);
    }
    
    DBG_PRINT("ERROR: Could not determine output format");
    return false;
}

// ============================================================================
// Exported Parse Functions (C linkage for DeepStream)
// ============================================================================

extern "C" bool NvDsInferParseYoloUltralytics(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) {
        DBG_PRINT("ERROR: No output layers");
        return false;
    }
    
    return parseYoloAuto(outputLayersInfo[0], networkInfo, detectionParams, objectList);
}

extern "C" bool NvDsInferParseYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const float* output = static_cast<const float*>(layer.buffer);
    const NvDsInferDims& dims = layer.inferDims;
    
    uint32_t numPredictions, numChannels;
    if (dims.numDims == 2) {
        numPredictions = dims.d[0];
        numChannels = dims.d[1];
    } else if (dims.numDims == 3) {
        numPredictions = dims.d[1];
        numChannels = dims.d[2];
    } else {
        return false;
    }
    
    const float confThreshold = detectionParams.perClassPreclusterThreshold.size() > 0 
                                ? detectionParams.perClassPreclusterThreshold[0] 
                                : CONF_THRESHOLD;
    
    return parseYoloV5Format(
        output, numPredictions, numChannels,
        networkInfo.width, networkInfo.height,
        confThreshold, NMS_IOU_THRESHOLD,
        objectList);
}

extern "C" bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    if (outputLayersInfo.empty()) return false;
    
    const NvDsInferLayerInfo& layer = outputLayersInfo[0];
    const float* output = static_cast<const float*>(layer.buffer);
    const NvDsInferDims& dims = layer.inferDims;
    
    uint32_t numChannels, numPredictions;
    if (dims.numDims == 2) {
        numChannels = dims.d[0];
        numPredictions = dims.d[1];
    } else if (dims.numDims == 3) {
        numChannels = dims.d[1];
        numPredictions = dims.d[2];
    } else {
        return false;
    }
    
    const float confThreshold = detectionParams.perClassPreclusterThreshold.size() > 0 
                                ? detectionParams.perClassPreclusterThreshold[0] 
                                : CONF_THRESHOLD;
    
    return parseYoloV8Format(
        output, numChannels, numPredictions,
        networkInfo.width, networkInfo.height,
        confThreshold, NMS_IOU_THRESHOLD,
        objectList);
}

// Alias for YOLO11 (same format as YOLOv8)
extern "C" bool NvDsInferParseYolo11(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    return NvDsInferParseYoloV8(outputLayersInfo, networkInfo, detectionParams, objectList);
}

// Check if parser symbols are present
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloUltralytics);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV5);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo11);
