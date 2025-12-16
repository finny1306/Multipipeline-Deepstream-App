# script to export general pt model to onnx format

import argparse
import ultralytics

def export_yolo_to_onnx(model_path: str):
    # Load the YOLO model
    model = ultralytics.YOLO(model_path)

    model.export(
        format="onnx",
        imgsz=(640, 640),
        opset=17,
        device=0,
        dynamic=True,    # ❗ disable dynamic
        simplify=True,
        half=True,       # ❗ ONNX must be FP32
        nms=False,        # ❗ TensorRT can't parse Ultralytics NMS
        batch=128
    )

    print(f"Model export done!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format")
    parser.add_argument("model_path", type=str, help="Path to the YOLO .pt model file")

    args = parser.parse_args()

    export_yolo_to_onnx(args.model_path)