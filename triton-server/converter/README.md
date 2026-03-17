# Universal YOLO to ONNX Converter

A unified, scalable converter that supports YOLOv5, YOLOv8, and YOLOv11 models with consistent interface and automatic version detection.

## Features

- 🎯 **Auto-detection**: Automatically detects YOLO model version
- 🔄 **Unified Interface**: Same arguments work across all YOLO versions
- 📦 **Multiple Versions**: Supports YOLOv5, YOLOv8, and YOLOv11
- ⚡ **Flexible Options**: Dynamic axes, FP16 precision, custom image sizes
- 🛠️ **Easy to Use**: Simple CLI with sensible defaults

## Installation

### Prerequisites

```bash
# For YOLOv8/v11 support
pip install ultralytics

# For YOLOv5 support (clone official repo)
git clone https://github.com/ultralytics/yolov5 /workspace/lib/yolov5
cd /workspace/lib/yolov5
pip install -r requirements.txt
```

### Additional Requirements

```bash
# ONNX tools
pip install onnx onnxslim onnxruntime

# For GPU support
pip install onnxruntime-gpu
```

## Usage

### Basic Usage

```bash
# Auto-detect version and export with defaults
python unified_yolo_converter.py yolov5s.pt

# Works with any YOLO version
python unified_yolo_converter.py yolov8n.pt
python unified_yolo_converter.py yolov11m.pt
```

### Custom Output Path

```bash
python unified_yolo_converter.py model.pt -o exported_model.onnx
python unified_yolo_converter.py model.pt --output /path/to/output.onnx
```

### Custom Image Size

```bash
# Square input (640x640)
python unified_yolo_converter.py model.pt --img-size 640 640

# Rectangular input (1280x640)
python unified_yolo_converter.py model.pt --img-size 1280 640
```

### Dynamic Axes

Enable dynamic batch size and image dimensions:

```bash
python unified_yolo_converter.py model.pt --dynamic
```

⚠️ **Note**: Dynamic axes are typically used for inference engines that support variable input sizes. Not recommended for TensorRT.

### FP16 Half Precision

Export with FP16 precision (requires GPU):

```bash
python unified_yolo_converter.py model.pt --half --device 0
```

### GPU Export

```bash
# Use GPU 0
python unified_yolo_converter.py model.pt --device 0

# Use GPU 1
python unified_yolo_converter.py model.pt --device 1

# Use CPU
python unified_yolo_converter.py model.pt --device cpu
```

### Force Specific Version

Override auto-detection:

```bash
# Force YOLOv5 export method
python unified_yolo_converter.py model.pt --version yolov5

# Force Ultralytics method (for YOLOv8/v11)
python unified_yolo_converter.py model.pt --version ultralytics
```

### Advanced Options

```bash
# Custom batch size
python unified_yolo_converter.py model.pt --batch-size 4

# Different ONNX opset
python unified_yolo_converter.py model.pt --opset 12

# Disable simplification
python unified_yolo_converter.py model.pt --no-simplify
```

### Complete Example

```bash
python unified_yolo_converter.py \
    yolov8x.pt \
    --output optimized_model.onnx \
    --img-size 1280 1280 \
    --opset 17 \
    --simplify \
    --device 0 \
    --batch-size 1
```

## Command Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `model_path` | str | *required* | Path to YOLO .pt model file |
| `-o, --output` | str | auto | Custom output path for ONNX model |
| `--img-size` | int int | 640 640 | Input image size (height width) |
| `--batch-size` | int | 1 | Batch size for export |
| `--opset` | int | 17 | ONNX opset version |
| `--simplify` | flag | enabled | Simplify ONNX model |
| `--no-simplify` | flag | - | Disable simplification |
| `--dynamic` | flag | disabled | Enable dynamic axes |
| `--half` | flag | disabled | Use FP16 precision (GPU only) |
| `--device` | str | cpu | Device: 'cpu', '0', '1', etc. |
| `--version` | str | auto | Force version: auto/yolov5/yolov8/yolov11 |

## Supported Export Configurations

### YOLOv5

Uses official YOLOv5 `export.py`:
- ✅ ONNX with simplification
- ✅ Dynamic axes
- ✅ FP16 on GPU
- ✅ Custom opset versions

### YOLOv8/v11

Uses Ultralytics export:
- ✅ ONNX with simplification
- ✅ Dynamic axes
- ✅ FP16 on GPU
- ✅ Custom opset versions

## Tips & Best Practices

### 1. TensorRT Deployment

For TensorRT, use these settings:

```bash
python unified_yolo_converter.py model.pt \
    --opset 17 \
    --simplify \
    --device 0
    # Do NOT use --dynamic or --half
```

### 2. ONNX Runtime

For ONNX Runtime inference:

```bash
python unified_yolo_converter.py model.pt \
    --simplify \
    --device cpu
```

### 3. Mobile Deployment

For mobile (smaller file size):

```bash
python unified_yolo_converter.py model.pt \
    --img-size 320 320 \
    --simplify \
    --device cpu
```

### 4. Maximum Compatibility

For maximum compatibility across platforms:

```bash
python unified_yolo_converter.py model.pt \
    --opset 12 \
    --simplify \
    --device cpu
```

## Troubleshooting

### Issue: "Could not import YOLOv5 export module"

**Solution**: Ensure YOLOv5 is cloned to `/workspace/lib/yolov5`:

```bash
git clone https://github.com/ultralytics/yolov5 /workspace/lib/yolov5
```

Or edit `YOLOV5_PATH` in the script to match your installation.

### Issue: "Ultralytics package not found"

**Solution**: Install ultralytics:

```bash
pip install ultralytics
```

### Issue: FP16 export fails

**Solution**: FP16 requires GPU. Either:
1. Use `--device 0` (or another GPU index)
2. Remove `--half` flag for CPU export

### Issue: Dynamic export fails

**Solution**: Some models/configurations don't support dynamic axes. Try without `--dynamic` flag.

### Issue: Wrong version detected

**Solution**: Manually specify version:

```bash
python unified_yolo_converter.py model.pt --version yolov5
```

## Version Detection Logic

The script auto-detects YOLO version by examining:

1. Model checkpoint structure
2. Presence of `yaml` attribute (YOLOv5)
3. Version metadata in checkpoint
4. Epoch information (typically YOLOv5)

If detection fails, it defaults to 'ultralytics' format.

## Output

The converter provides detailed output including:

- Detected/specified YOLO version
- Export configuration
- Progress messages
- Final file location and size

Example output:

```
============================================================
🎯 Universal YOLO to ONNX Converter
============================================================
📄 Model Path:     yolov8n.pt
💾 Output Path:    Auto (same directory)
📐 Image Size:     640x640
📦 Batch Size:     1
🔧 ONNX Opset:     17
🎨 Simplify:       True
🔄 Dynamic Axes:   False
⚡ Half Precision: False
🖥️  Device:         cpu
🏷️  Version:        auto
============================================================

🔍 Detected YOLO version: ULTRALYTICS

============================================================
🚀 Exporting Ultralytics YOLO model to ONNX
   Using ultralytics v8.x.x
============================================================

... export progress ...

============================================================
✅ Export completed successfully!
============================================================
📁 Output file: yolov8n.onnx
📊 File size:   6.24 MB
============================================================
```

## Integration with Other Tools

### Using with Python

```python
from unified_yolo_converter import export_yolo_to_onnx

# Export model programmatically
output_path = export_yolo_to_onnx(
    model_path="yolov8n.pt",
    output_path="model.onnx",
    img_size=(640, 640),
    opset=17,
    simplify=True,
    dynamic=False,
    half=False,
    device='cpu',
    batch_size=1,
    version='auto'
)

print(f"Model exported to: {output_path}")
```

### Batch Processing

```bash
#!/bin/bash
# Convert multiple models

for model in models/*.pt; do
    echo "Converting $model"
    python unified_yolo_converter.py "$model" \
        --img-size 640 640 \
        --simplify \
        --device 0
done
```

## License

This converter uses:
- YOLOv5: AGPL-3.0 License
- Ultralytics: AGPL-3.0 License

Ensure compliance with respective licenses when using exported models.

## Contributing

Contributions are welcome! Please ensure:
- Backward compatibility with existing arguments
- Support for new YOLO versions
- Clear documentation of changes

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review YOLOv5 and Ultralytics documentation
3. Open an issue with:
   - Model version
   - Export command used
   - Error message
   - System information