# Dynamic Ultralytics YOLO Parser for DeepStream

A fully dynamic, lightweight parser for DeepStream that **automatically detects the number of classes** from the model's output tensor shape. Works with **any** Ultralytics YOLO model - pretrained or custom trained.

## Key Features

- ✅ **Dynamic Class Detection**: Automatically infers `num_classes` from tensor shape
- ✅ **Zero Config Required**: No need to match `num-detected-classes` exactly
- ✅ **Universal Support**: Works with YOLOv5, YOLOv8, YOLO11
- ✅ **Custom Models**: Works with 1 class, 80 classes, or any number
- ✅ **Auto Format Detection**: Detects v5 vs v8/v11 format automatically

## How It Works

### Auto-Detection Logic

The parser uses the output tensor dimensions to determine both the format and number of classes:

| Format | Tensor Shape | Classes Calculation |
|--------|--------------|---------------------|
| YOLOv8/v11 | `[C, N]` where C < N | `num_classes = C - 4` |
| YOLOv5 | `[N, C]` where N > C | `num_classes = C - 5` |

**Examples:**

| Model | Output Shape | Detected Format | Inferred Classes |
|-------|-------------|-----------------|------------------|
| YOLOv8 (COCO) | `[84, 8400]` | YOLOv8 | 84 - 4 = 80 |
| YOLOv8 (1 class) | `[5, 8400]` | YOLOv8 | 5 - 4 = 1 |
| YOLOv8 (10 classes) | `[14, 8400]` | YOLOv8 | 14 - 4 = 10 |
| YOLOv5 (COCO) | `[25200, 85]` | YOLOv5 | 85 - 5 = 80 |
| YOLOv5 (1 class) | `[25200, 6]` | YOLOv5 | 6 - 5 = 1 |
| YOLOv5 (5 classes) | `[25200, 10]` | YOLOv5 | 10 - 5 = 5 |

## Quick Start

### 1. Build the Parser

```bash
cd nvdsinfer_yolo_ultralytics_v2

# For DeepStream 8.0
export CUDA_VER=12.8
make

# Build with debug output (recommended for testing)
make debug
```

### 2. Export Your Custom Model

```python
from ultralytics import YOLO

# Load your custom model
model = YOLO("my_custom_model.pt")  # Could be 1 class, 5 classes, etc.

# Export to ONNX or TensorRT
model.export(format="onnx")
# or
model.export(format="engine", half=True)
```

### 3. Configure DeepStream

**For nvinfer (recommended for simplicity):**

```ini
[property]
gpu-id=0
onnx-file=/path/to/my_custom_model.onnx
labelfile-path=/path/to/my_labels.txt

# IMPORTANT: Set this to match your model's classes
# But the parser will work even if it doesn't match exactly!
num-detected-classes=1

# Custom parser
custom-lib-path=/workspace/lib/libnvdsinfer_yolo_ultralytics.so
parse-bbox-func-name=NvDsInferParseYoloUltralytics

# Required settings
network-type=0
cluster-mode=2
maintain-aspect-ratio=1
net-scale-factor=0.0039215697906911373

[class-attrs-all]
pre-cluster-threshold=0.25
```

**For nvinferserver (Triton):**

```protobuf
infer_config {
  unique_id: 1
  gpu_ids: [0]
  max_batch_size: 50

  backend {
    triton {
      model_name: "my_custom_model"
      version: -1
      model_repo {
        root: "/workspace/model_repository"
        strict_model_config: true
      }
    }
    inputs: [{ name: "images" }]
    outputs: [{ name: "output0" }]
  }

  preprocess {
    network_format: IMAGE_FORMAT_RGB
    tensor_order: TENSOR_ORDER_LINEAR
    tensor_name: "images"
    maintain_aspect_ratio: 1
    symmetric_padding: 1
    frame_scaling_hw: FRAME_SCALING_HW_GPU
    normalize {
      scale_factor: 0.0039215697906911373
      channel_offsets: [0, 0, 0]
    }
  }

  postprocess {
    labelfile_path: "/path/to/my_labels.txt"
    detection {
      # Set to your model's class count (parser validates but doesn't require exact match)
      num_detected_classes: 1
      custom_parse_bbox_func: "NvDsInferParseYoloUltralytics"
      nms {
        confidence_threshold: 0.25
        iou_threshold: 0.45
        topk: 300
      }
    }
  }

  custom_lib {
    path: "/workspace/lib/libnvdsinfer_yolo_ultralytics.so"
  }
}
```

### 4. Create Labels File

Create a labels file with one label per line:

```text
# my_labels.txt for a 1-class model
person
```

```text
# my_labels.txt for a 5-class model
car
truck
bus
motorcycle
bicycle
```

## Debugging

### Enable Debug Output

Build with debug enabled to see what the parser detects:

```bash
make debug
```

Example debug output for a 1-class YOLOv5 model:

```
[YOLO_PARSER] Output layer: output0
[YOLO_PARSER] Dimensions: numDims=2
[YOLO_PARSER]   dim[0] = 25200
[YOLO_PARSER]   dim[1] = 6
[YOLO_PARSER] Tensor shape: [25200, 6]
[YOLO_PARSER] Detected: YOLOv5 format [predictions=25200, channels=6]
[YOLO_PARSER] Inferred classes: 1
[YOLO_PARSER] Pre-NMS detections: 523
[YOLO_PARSER] Post-NMS detections: 15
```

### Common Issues

**Issue: No detections**
- Lower the confidence threshold in config
- Build with `make debug` to see pre-NMS detection count
- Verify model is exporting correctly with Ultralytics inference

**Issue: Parser returns error**
- Check tensor dimensions with debug output
- Ensure output tensor name is correct (usually "output0")
- Verify model export completed successfully

## Technical Details

### Detection Logic

```
if (dim0 < 500 && dim1 >= 500):
    # Small first dim, large second dim
    # -> YOLOv8/v11 format: [channels, predictions]
    num_classes = dim0 - 4
    
elif (dim0 >= 500 && dim1 < 500):
    # Large first dim, small second dim  
    # -> YOLOv5 format: [predictions, channels]
    num_classes = dim1 - 5
```

### Why 500 as Threshold?

- **Channels** (4 + num_classes): Typically 5-200 for most models
- **Predictions**: Typically 1000+ (8400 for 640x640, 25200 for YOLOv5)

This threshold safely separates the two dimensions.

## Comparison with Previous Version

| Feature | v1 (Previous) | v2 (This Version) |
|---------|---------------|-------------------|
| Class detection | From config | From tensor shape |
| Custom models | Required exact config match | Works automatically |
| 1-class models | ❌ Often failed | ✅ Works |
| Error handling | Limited | Comprehensive |
| Debug output | Basic | Detailed |

## File Structure

```
nvdsinfer_yolo_ultralytics_v2/
├── nvdsinfer_yolo_ultralytics.cpp  # Parser implementation
├── Makefile                         # Build system
└── README.md                        # This file
```

## License

MIT License - Free for commercial and personal use.
