/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Custom DeepStream Parser for Ultralytics YOLO Models (v5, v8, v11)
 * Handles standard Ultralytics TensorRT/ONNX export format
 * 
 * Output tensor format: [batch, 4+num_classes, num_predictions]
 * - YOLOv5: [batch, 25200, 85] or [batch, 85, 25200] depending on export
 * - YOLOv8/v11: [batch, 84, 8400] for 80 classes
 * 
 * This parser is optimized for nvinferserver (Triton) and nvinfer plugins.
 */

#ifndef NVDSINFER_YOLO_ULTRALYTICS_H
#define NVDSINFER_YOLO_ULTRALYTICS_H

#include <vector>
#include <cstdint>

// NMS parameters - can be overridden via config
#ifndef NMS_IOU_THRESHOLD
#define NMS_IOU_THRESHOLD 0.45f
#endif

#ifndef CONF_THRESHOLD
#define CONF_THRESHOLD 0.25f
#endif

#ifndef MAX_DETECTIONS
#define MAX_DETECTIONS 300
#endif

// Detection structure for internal use
struct Detection {
    float x;      // center x
    float y;      // center y
    float w;      // width
    float h;      // height
    float conf;   // confidence (max class probability)
    int classId;  // class index
};

// Utility functions
float iouCalc(const Detection& a, const Detection& b);

void nmsSort(std::vector<Detection>& detections, float iouThreshold, int maxDets);

// Main parsing functions - exported for DeepStream
#ifdef __cplusplus
extern "C" {
#endif

/**
 * Parse function for nvinfer plugin (standard DeepStream)
 * For use with config: parse-bbox-func-name=NvDsInferParseYoloUltralytics
 */
bool NvDsInferParseYoloUltralytics(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

/**
 * Parse function for YOLOv5 format (transposed output)
 * Output shape: [batch, num_predictions, 85]
 */
bool NvDsInferParseYoloV5(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

/**
 * Parse function for YOLOv8/v11 format
 * Output shape: [batch, 84, num_predictions]
 */
bool NvDsInferParseYoloV8(
    std::vector<NvDsInferLayerInfo> const& outputLayersInfo,
    NvDsInferNetworkInfo const& networkInfo,
    NvDsInferParseDetectionParams const& detectionParams,
    std::vector<NvDsInferParseObjectInfo>& objectList);

// Alias for YOLO11 (same format as v8)
#define NvDsInferParseYolo11 NvDsInferParseYoloV8

#ifdef __cplusplus
}
#endif

#endif // NVDSINFER_YOLO_ULTRALYTICS_H
