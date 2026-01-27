/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Custom DeepStream Parser for Ultralytics YOLO Models (v5, v8, v11)
 * 
 * Supports:
 * - YOLOv5: Output [batch, num_preds, 85] where 85 = 4(xywh) + 1(obj) + 80(cls)
 * - YOLOv8/v11: Output [batch, 84, num_preds] where 84 = 4(xywh) + 80(cls)
 * 
 * The parser auto-detects format based on tensor dimensions.
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
// YOLOv8/v11 Parser (Transposed format: [batch, 4+num_classes, num_predictions])
// ============================================================================

static bool parseYoloV8Format(
    const float* output,
    const uint32_t numChannels,     // 84 for COCO (4 + 80)
    const uint32_t numPredictions,  // 8400 for 640x640 input
    const uint32_t networkWidth,
    const uint32_t networkHeight,
    const float confThreshold,
    const float nmsThreshold,
    const int numClasses,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    DBG_PRINT("YOLOv8/v11 format: channels=" << numChannels 
              << " predictions=" << numPredictions 
              << " classes=" << numClasses);
    
    std::vector<Detection> detections;
    detections.reserve(1000);  // Pre-allocate for efficiency
    
    // Output layout: [4+num_classes, num_predictions]
    // Row 0: x coordinates for all predictions
    // Row 1: y coordinates for all predictions  
    // Row 2: w coordinates for all predictions
    // Row 3: h coordinates for all predictions
    // Rows 4+: class scores for all predictions
    
    for (uint32_t p = 0; p < numPredictions; ++p) {
        // Extract coordinates (already in pixel space for standard Ultralytics export)
        float cx = output[0 * numPredictions + p];  // center x
        float cy = output[1 * numPredictions + p];  // center y
        float w  = output[2 * numPredictions + p];  // width
        float h  = output[3 * numPredictions + p];  // height
        
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
        
        // Convert center format to top-left corner format
        float left = det.x - det.w / 2.0f;
        float top = det.y - det.h / 2.0f;
        
        // Clamp to image bounds
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
// YOLOv5 Parser (Standard format: [batch, num_predictions, 4+1+num_classes])
// ============================================================================

static bool parseYoloV5Format(
    const float* output,
    const uint32_t numPredictions,  // 25200 for 640x640 input
    const uint32_t numChannels,     // 85 for COCO (4 + 1 + 80)
    const uint32_t networkWidth,
    const uint32_t networkHeight,
    const float confThreshold,
    const float nmsThreshold,
    const int numClasses,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    DBG_PRINT("YOLOv5 format: predictions=" << numPredictions 
              << " channels=" << numChannels 
              << " classes=" << numClasses);
    
    std::vector<Detection> detections;
    detections.reserve(1000);
    
    // Output layout: [num_predictions, 85]
    // Each row: [cx, cy, w, h, obj_conf, class_0, class_1, ..., class_79]
    
    const int stride = numChannels;
    
    for (uint32_t p = 0; p < numPredictions; ++p) {
        const float* row = output + p * stride;
        
        float cx = row[0];
        float cy = row[1];
        float w = row[2];
        float h = row[3];
        float objectness = row[4];  // YOLOv5 has objectness score
        
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
// Auto-detect and Parse Function
// ============================================================================

static bool parseYoloAuto(
    const NvDsInferLayerInfo& outputLayer,
    const NvDsInferNetworkInfo& networkInfo,
    const NvDsInferParseDetectionParams& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList)
{
    const float* output = static_cast<const float*>(outputLayer.buffer);
    
    // Get dimensions
    const NvDsInferDims& dims = outputLayer.inferDims;
    
    DBG_PRINT("Output layer: " << outputLayer.layerName);
    DBG_PRINT("Dimensions: numDims=" << dims.numDims);
    for (uint32_t i = 0; i < dims.numDims; ++i) {
        DBG_PRINT("  dim[" << i << "] = " << dims.d[i]);
    }
    
    // Determine format based on dimensions
    // YOLOv8/v11: [84, 8400] or [batch, 84, 8400] -> dim0 < dim1 typically
    // YOLOv5: [25200, 85] or [batch, 25200, 85] -> dim0 > dim1 typically
    
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
    
    const int numClasses = detectionParams.numClassesConfigured;
    const float confThreshold = detectionParams.perClassPreclusterThreshold.size() > 0 
                                ? detectionParams.perClassPreclusterThreshold[0] 
                                : CONF_THRESHOLD;
    
    DBG_PRINT("Config: numClasses=" << numClasses << " confThreshold=" << confThreshold);
    
    // Detect format:
    // YOLOv8/v11: dim0 = 4 + num_classes (e.g., 84 for 80 classes)
    // YOLOv5: dim1 = 4 + 1 + num_classes (e.g., 85 for 80 classes)
    
    if (dim0 == (uint32_t)(4 + numClasses)) {
        // YOLOv8/v11 format: [4+classes, num_predictions]
        DBG_PRINT("Detected YOLOv8/v11 format");
        return parseYoloV8Format(
            output, dim0, dim1,
            networkInfo.width, networkInfo.height,
            confThreshold, NMS_IOU_THRESHOLD, numClasses,
            objectList);
    }
    else if (dim1 == (uint32_t)(4 + 1 + numClasses)) {
        // YOLOv5 format: [num_predictions, 4+1+classes]
        DBG_PRINT("Detected YOLOv5 format");
        return parseYoloV5Format(
            output, dim0, dim1,
            networkInfo.width, networkInfo.height,
            confThreshold, NMS_IOU_THRESHOLD, numClasses,
            objectList);
    }
    else if (dim1 == (uint32_t)(4 + numClasses)) {
        // Transposed YOLOv5-style but without objectness (custom model)
        DBG_PRINT("Detected transposed format without objectness");
        // Treat as YOLOv8 but transposed
        // This requires special handling - transpose the data
        std::vector<float> transposed(dim0 * dim1);
        for (uint32_t i = 0; i < dim0; ++i) {
            for (uint32_t j = 0; j < dim1; ++j) {
                transposed[j * dim0 + i] = output[i * dim1 + j];
            }
        }
        return parseYoloV8Format(
            transposed.data(), dim1, dim0,
            networkInfo.width, networkInfo.height,
            confThreshold, NMS_IOU_THRESHOLD, numClasses,
            objectList);
    }
    else {
        DBG_PRINT("ERROR: Unknown output format. dim0=" << dim0 << " dim1=" << dim1 
                  << " expected 4+numClasses=" << (4 + numClasses)
                  << " or 5+numClasses=" << (5 + numClasses));
        return false;
    }
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
    
    // Use first output layer (standard YOLO has single output)
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
    
    const int numClasses = detectionParams.numClassesConfigured;
    const float confThreshold = detectionParams.perClassPreclusterThreshold.size() > 0 
                                ? detectionParams.perClassPreclusterThreshold[0] 
                                : CONF_THRESHOLD;
    
    return parseYoloV5Format(
        output, numPredictions, numChannels,
        networkInfo.width, networkInfo.height,
        confThreshold, NMS_IOU_THRESHOLD, numClasses,
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
    
    const int numClasses = detectionParams.numClassesConfigured;
    const float confThreshold = detectionParams.perClassPreclusterThreshold.size() > 0 
                                ? detectionParams.perClassPreclusterThreshold[0] 
                                : CONF_THRESHOLD;
    
    return parseYoloV8Format(
        output, numChannels, numPredictions,
        networkInfo.width, networkInfo.height,
        confThreshold, NMS_IOU_THRESHOLD, numClasses,
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

// Check if parser symbols are present (for debugging)
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloUltralytics);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV5);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYoloV8);
CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo11);
