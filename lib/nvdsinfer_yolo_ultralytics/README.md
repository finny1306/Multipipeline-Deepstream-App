# Ultralytics YOLO Parser for DeepStream

A lightweight, optimized custom parser for DeepStream that supports standard **Ultralytics exports** of YOLOv5, YOLOv8, and YOLO11 models.

## Overview

This parser handles the native Ultralytics export format without requiring any special export scripts or model modifications.

### Supported Output Formats

| Model | Export Command | Output Shape | Parser |
|-------|---------------|--------------|--------|
| YOLOv5 | `model.export(format="onnx")` | `[batch, 25200, 85]` | `NvDsInferParseYoloV5` |
| YOLOv8 | `model.export(format="onnx")` | `[batch, 84, 8400]` | `NvDsInferParseYoloV8` |
| YOLO11 | `model.export(format="onnx")` | `[batch, 84, 8400]` | `NvDsInferParseYolo11` |
| Auto-detect | Any | Any | `NvDsInferParseYoloUltralytics` |

### Key Features

- ✅ **Auto-detection**: Automatically detects YOLOv5 vs YOLOv8/v11 format
- ✅ **Custom models**: Works with both pretrained and custom-trained models
- ✅ **Optimized NMS**: Efficient CPU-based Non-Maximum Suppression
- ✅ **Configurable thresholds**: Confidence and IoU thresholds via config
- ✅ **Multiple classes**: Supports any number of classes
- ✅ **nvinfer & nvinferserver**: Works with both DeepStream plugins

## Quick Start

### 1. Build the Parser

```bash
cd nvdsinfer_yolo_ultralytics

# Set CUDA version for your DeepStream
export CUDA_VER=12.8  # DeepStream 8.0

# Build
make

# Optional: Build with debug output
make debug
```

### 2. Export Your Model

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo11s.pt")

# Export to ONNX
model.export(format="onnx")

# Or export directly to TensorRT engine
model.export(format="engine", half=True)
```

### 3. Configure DeepStream

**For nvinfer plugin:**
```ini
[property]
onnx-file=/path/to/yolo11s.onnx
custom-lib-path=/path/to/libnvdsinfer_yolo_ultralytics.so
parse-bbox-func-name=NvDsInferParseYoloUltralytics
num-detected-classes=80
cluster-mode=2
```

**For nvinferserver (Triton):**
```protobuf
postprocess {
  detection {
    num_detected_classes: 80
    custom_parse_bbox_func: "NvDsInferParseYoloUltralytics"
  }
}
custom_lib {
  path: "/path/to/libnvdsinfer_yolo_ultralytics.so"
}
```

## Configuration Options

### Compile-time Options

```bash
# Custom confidence threshold (default: 0.25)
make CONF_THRESH=0.5

# Custom NMS IoU threshold (default: 0.45)
make NMS_THRESH=0.6

# Custom max detections (default: 300)
make MAX_DETS=100

# Enable debug output
make DEBUG=1
```

### Runtime Configuration

For **nvinfer**, use `[class-attrs-all]`:
```ini
[class-attrs-all]
pre-cluster-threshold=0.25
nms-iou-threshold=0.45
topk=300
```

For **nvinferserver**, use `nms` block:
```protobuf
nms {
  confidence_threshold: 0.25
  iou_threshold: 0.45
  topk: 300
}
```

## Output Tensor Formats Explained

### YOLOv8 / YOLO11 Format

Output shape: `[batch, 4 + num_classes, num_predictions]`

For 80 classes at 640x640 input: `[1, 84, 8400]`

```
Channel layout:
  [0]: center_x for all 8400 predictions
  [1]: center_y for all 8400 predictions
  [2]: width for all 8400 predictions
  [3]: height for all 8400 predictions
  [4-83]: class scores (80 classes) for all predictions
```

**Note**: YOLOv8/v11 does NOT have an objectness score. The confidence is simply the max class score.

### YOLOv5 Format

Output shape: `[batch, num_predictions, 4 + 1 + num_classes]`

For 80 classes at 640x640 input: `[1, 25200, 85]`

```
Row layout for each prediction:
  [0-3]: center_x, center_y, width, height
  [4]: objectness score
  [5-84]: class scores (80 classes)
```

**Note**: YOLOv5 has an objectness score. Final confidence = objectness × class_score.

## Comparison with DeepStream-Yolo

| Feature | This Parser | DeepStream-Yolo |
|---------|-------------|-----------------|
| Standard Ultralytics export | ✅ Yes | ❌ Requires custom export |
| Export script needed | ❌ No | ✅ Yes (export_yolo11.py) |
| DeepStreamOutput layer | ❌ Not needed | ✅ Required |
| GPU postprocessing | ❌ CPU (fast enough) | ✅ CUDA kernel |
| nvinferserver support | ✅ Yes | ✅ Yes |
| Code complexity | Low (~400 lines) | High (~2000+ lines) |

**When to use this parser:**
- You have an existing Ultralytics model and don't want to re-export
- You want a simple, maintainable codebase
- CPU NMS performance is acceptable for your use case

**When to use DeepStream-Yolo:**
- You need maximum performance (GPU postprocessing)
- You're starting fresh and can use their export script
- You need advanced features like segmentation masks

## Troubleshooting

### Debug Output

Build with debug enabled:
```bash
make debug
```

This will print parsing information:
```
[YOLO_PARSER] YOLOv8/v11 format: channels=84 predictions=8400 classes=80
[YOLO_PARSER] Pre-NMS detections: 1523
[YOLO_PARSER] Post-NMS detections: 12
```

### Common Issues

**Issue: No detections**
- Check confidence threshold is not too high
- Verify label file has correct number of classes
- Enable debug output to see pre-NMS detection count

**Issue: Wrong detections (garbage labels, negative coords)**
- Your model was exported with a different format
- Try specifying parser explicitly: `NvDsInferParseYoloV8` or `NvDsInferParseYoloV5`
- Verify `num-detected-classes` matches your model

**Issue: Library not found**
- Ensure library is built: `ls libnvdsinfer_yolo_ultralytics.so`
- Check path in config is absolute
- Verify `LD_LIBRARY_PATH` includes the library directory

### Verify Parser Symbols

```bash
nm -D libnvdsinfer_yolo_ultralytics.so | grep -i parse
```

Expected output:
```
T NvDsInferParseYolo11
T NvDsInferParseYoloUltralytics
T NvDsInferParseYoloV5
T NvDsInferParseYoloV8
```

## File Structure

```
nvdsinfer_yolo_ultralytics/
├── Makefile
├── README.md
├── nvdsinfer_yolo_ultralytics.h      # Header file
├── nvdsinfer_yolo_ultralytics.cpp    # Parser implementation
├── config_infer_yolo_ultralytics.txt # nvinfer config example
└── nvinferserver_yolo_ultralytics.txt # nvinferserver config example
```

## License

MIT License - Free for commercial and personal use.

## Credits

Based on output format specifications from:
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [NVIDIA DeepStream SDK](https://developer.nvidia.com/deepstream-sdk)
- [DeepStream-Yolo by marcoslucianops](https://github.com/marcoslucianops/DeepStream-Yolo)
