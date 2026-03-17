#!/usr/bin/env python3
"""
Universal YOLO to ONNX Converter
Supports YOLOv5, YOLOv8, and YOLOv11 models with consistent interface
"""

import argparse
import sys
from pathlib import Path
import torch


# Add YOLOv5 to path if it exists
YOLOV5_PATH = Path("/root/multipipeline-deepstream/lib/yolov5")
if YOLOV5_PATH.exists():
    sys.path.insert(0, str(YOLOV5_PATH))


def detect_yolo_version(model_path: str) -> str:
    """
    Detect YOLO version from model file.
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Version string: 'yolov5', 'yolov8', 'yolov11', or 'ultralytics'
    """
    try:
        # Try loading with torch to check model structure
        ckpt = torch.load(model_path, map_location='cpu')
        
        # Check for version indicators in the checkpoint
        if 'model' in ckpt:
            # YOLOv5 typically has 'model' key with yaml attribute
            model_info = ckpt.get('model', None)
            if hasattr(model_info, 'yaml') or 'yaml' in ckpt:
                return 'yolov5'
        
        # Check for explicit version field
        if 'version' in ckpt:
            version = str(ckpt['version'])
            if '5' in version or 'v5' in version.lower():
                return 'yolov5'
            elif '8' in version or 'v8' in version.lower():
                return 'yolov8'
            elif '11' in version or 'v11' in version.lower():
                return 'yolov11'
        
        # Check model name or epoch info
        if 'epoch' in ckpt:
            # Likely YOLOv5 format
            return 'yolov5'
        
        # Default to ultralytics (works for v8/v11)
        return 'ultralytics'
        
    except Exception as e:
        print(f"⚠️  Warning: Could not auto-detect version: {e}")
        print("   Defaulting to 'ultralytics' (YOLOv8/v11 format)")
        return 'ultralytics'


def export_yolov5_to_onnx(
    model_path: str,
    output_path: str = None,
    img_size: tuple = (640, 640),
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
    half: bool = False,
    device: str = 'cpu',
    batch_size: int = 1
):
    """
    Export YOLOv5 model to ONNX using official export.py
    
    Args:
        model_path: Path to YOLOv5 .pt model
        output_path: Custom output path (optional)
        img_size: Input image size (height, width)
        opset: ONNX opset version
        simplify: Simplify ONNX model
        dynamic: Enable dynamic axes
        half: Use FP16 precision
        device: Device to use (cpu/cuda)
        batch_size: Batch size for export
        
    Returns:
        Path to exported ONNX model
    """
    try:
        from export import run as yolov5_export
    except ImportError:
        raise ImportError(
            "Could not import YOLOv5 export module. "
            f"Please ensure YOLOv5 is available at {YOLOV5_PATH}"
        )
    
    print(f"\n{'='*60}")
    print(f"🚀 Exporting YOLOv5 model to ONNX")
    print(f"{'='*60}\n")
    
    # Prepare arguments for YOLOv5 export
    export_args = {
        'weights': model_path,
        'imgsz': img_size,
        'device': device,
        'include': ['onnx'],
        'opset': opset,
        'simplify': simplify,
        'dynamic': dynamic,
        'half': half,
        'batch_size': batch_size,
    }
    
    # Run YOLOv5 export
    exported_files = yolov5_export(**export_args)
    
    if not exported_files:
        raise RuntimeError("YOLOv5 export failed - no output files generated")
    
    result_path = exported_files[0]
    
    # Handle custom output path
    if output_path and result_path:
        import shutil
        source = Path(result_path)
        dest = Path(output_path)
        if source != dest and source.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
            print(f"📦 Moved model to: {dest}")
            return str(dest)
    
    return result_path


def export_ultralytics_to_onnx(
    model_path: str,
    output_path: str = None,
    img_size: tuple = (640, 640),
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
    half: bool = False,
    device: str = 'cpu',
    batch_size: int = 1
):
    """
    Export YOLOv8/YOLOv11 model to ONNX using ultralytics library
    
    Args:
        model_path: Path to YOLO .pt model
        output_path: Custom output path (optional)
        img_size: Input image size (height, width)
        opset: ONNX opset version
        simplify: Simplify ONNX model
        dynamic: Enable dynamic axes
        half: Use FP16 precision
        device: Device to use
        batch_size: Batch size for export
        
    Returns:
        Path to exported ONNX model
    """
    try:
        import ultralytics
    except ImportError:
        raise ImportError(
            "Ultralytics package not found. "
            "Install with: pip install ultralytics"
        )
    
    print(f"\n{'='*60}")
    print(f"🚀 Exporting Ultralytics YOLO model to ONNX")
    print(f"   Using ultralytics v{ultralytics.__version__}")
    print(f"{'='*60}\n")
    
    # Load model
    model = ultralytics.YOLO(model_path)
    
    # Convert device to ultralytics format
    if isinstance(device, str) and device.lower() == 'cpu':
        device_id = 'cpu'
    elif isinstance(device, str) and device.isdigit():
        device_id = int(device)
    else:
        device_id = device
    
    # Export model
    export_path = model.export(
        format="onnx",
        imgsz=img_size,
        opset=opset,
        device=device_id,
        dynamic=dynamic,
        simplify=simplify,
        half=half,
        batch=batch_size
    )
    
    if not export_path:
        raise RuntimeError("Ultralytics export failed - no output file generated")
    
    # Handle custom output path
    if output_path and export_path:
        import shutil
        source = Path(export_path)
        dest = Path(output_path)
        if source != dest and source.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(dest))
            print(f"📦 Moved model to: {dest}")
            return str(dest)
    
    return export_path


def export_yolo_to_onnx(
    model_path: str,
    output_path: str = None,
    img_size: tuple = (640, 640),
    opset: int = 17,
    simplify: bool = True,
    dynamic: bool = False,
    half: bool = False,
    device: str = 'cpu',
    batch_size: int = 1,
    version: str = 'auto'
):
    """
    Universal YOLO to ONNX converter supporting YOLOv5, YOLOv8, and YOLOv11
    
    Args:
        model_path: Path to YOLO .pt model
        output_path: Custom output path for ONNX model
        img_size: Input image size as (height, width)
        opset: ONNX opset version
        simplify: Simplify ONNX model
        dynamic: Enable dynamic axes for batch/height/width
        half: Use FP16 half precision
        device: Device ('cpu' or GPU index like '0', '1')
        batch_size: Batch size for export
        version: YOLO version ('auto', 'yolov5', 'yolov8', 'yolov11', 'ultralytics')
    
    Returns:
        Path to exported ONNX model
    """
    model_path = str(model_path)
    
    # Validate model file exists
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Auto-detect version if needed
    if version == 'auto':
        version = detect_yolo_version(model_path)
        print(f"🔍 Detected YOLO version: {version.upper()}")
    else:
        print(f"📌 Using specified version: {version.upper()}")
    
    # Normalize device parameter for both exporters
    device_str = str(device).lower()
    
    # Route to appropriate export function
    if version == 'yolov5':
        return export_yolov5_to_onnx(
            model_path=model_path,
            output_path=output_path,
            img_size=img_size,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            half=half,
            device=device_str,
            batch_size=batch_size
        )
    else:  # yolov8, yolov11, or ultralytics
        return export_ultralytics_to_onnx(
            model_path=model_path,
            output_path=output_path,
            img_size=img_size,
            opset=opset,
            simplify=simplify,
            dynamic=dynamic,
            half=half,
            device=device_str,
            batch_size=batch_size
        )


def main():
    parser = argparse.ArgumentParser(
        description="Universal YOLO to ONNX Converter - Supports YOLOv5, YOLOv8, and YOLOv11",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect version and export with defaults
  python unified_yolo_converter.py yolov5s.pt
  
  # Export YOLOv8 with custom output path
  python unified_yolo_converter.py yolov8n.pt -o model.onnx
  
  # Export with dynamic axes and FP16 on GPU
  python unified_yolo_converter.py yolov11m.pt --dynamic --half --device 0
  
  # Export with custom image size
  python unified_yolo_converter.py model.pt --img-size 1280 1280
  
  # Force specific YOLO version
  python unified_yolo_converter.py model.pt --version yolov5
        """
    )
    
    # Required arguments
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the YOLO .pt model file"
    )
    
    # Optional arguments
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Custom output path for ONNX model (default: same directory as input)"
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        nargs=2,
        default=[640, 640],
        metavar=("HEIGHT", "WIDTH"),
        help="Input image size as height width (default: 640 640)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for export (default: 1)"
    )
    
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )
    
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="Simplify ONNX model (default: enabled)"
    )
    
    parser.add_argument(
        "--no-simplify",
        action="store_false",
        dest="simplify",
        help="Disable ONNX model simplification"
    )
    
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic axes for variable batch/image sizes"
    )
    
    parser.add_argument(
        "--half",
        action="store_true",
        help="Use FP16 half precision (requires GPU)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use: 'cpu', '0', '1', etc. (default: cpu)"
    )
    
    parser.add_argument(
        "--version",
        type=str,
        choices=["auto", "yolov5", "yolov8", "yolov11", "ultralytics"],
        default="auto",
        help="YOLO version (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "="*60)
    print("🎯 Universal YOLO to ONNX Converter")
    print("="*60)
    print(f"📄 Model Path:     {args.model_path}")
    print(f"💾 Output Path:    {args.output or 'Auto (same directory)'}")
    print(f"📐 Image Size:     {args.img_size[0]}x{args.img_size[1]}")
    print(f"📦 Batch Size:     {args.batch_size}")
    print(f"🔧 ONNX Opset:     {args.opset}")
    print(f"🎨 Simplify:       {args.simplify}")
    print(f"🔄 Dynamic Axes:   {args.dynamic}")
    print(f"⚡ Half Precision: {args.half}")
    print(f"🖥️  Device:         {args.device}")
    print(f"🏷️  Version:        {args.version}")
    print("="*60 + "\n")
    
    try:
        # Export model
        output_file = export_yolo_to_onnx(
            model_path=args.model_path,
            output_path=args.output,
            img_size=tuple(args.img_size),
            opset=args.opset,
            simplify=args.simplify,
            dynamic=args.dynamic,
            half=args.half,
            device=args.device,
            batch_size=args.batch_size,
            version=args.version
        )
        
        # Success message
        output_path = Path(output_file)
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        
        print(f"\n{'='*60}")
        print(f"✅ Export completed successfully!")
        print(f"{'='*60}")
        print(f"📁 Output file: {output_file}")
        print(f"📊 File size:   {file_size_mb:.2f} MB")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ Export failed!")
        print(f"{'='*60}")
        print(f"Error: {str(e)}")
        print(f"{'='*60}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()