#!/usr/bin/env python3
"""
DeepStream REST API Client
Manage video streams dynamically across multiple pipelines
"""

import requests
import json
import argparse
from datetime import datetime, UTC

from typing import Dict, Any


class DeepStreamRESTClient:
    def __init__(self, host: str = "localhost", port: int = 9002):
        self.base_url = f"http://{host}:{port}"
        self.api_version = "v1"

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict[str, Any]:
        """Make HTTP request to DeepStream REST API"""
        url = f"{self.base_url}/api/{self.api_version}{endpoint}"

        try:
            if method.upper() == "GET":
                response = requests.get(url)
            elif method.upper() == "POST":
                response = requests.post(url, json=data, headers={"Content-Type": "application/json"})
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error communicating with {self.base_url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None

    def check_health(self) -> Dict:
        """Check DeepStream pipeline health"""
        return self._make_request("GET", "/health/get-dsready-state")

    def get_streams(self) -> Dict:
        """Get all active streams"""
        return self._make_request("GET", "/stream/get-stream-info")

    # Stream management
    def add_stream(self, camera_id: str, camera_name: str, rtsp_url: str,
                   resolution: str = "1920x1080", codec: str = "h264",
                   framerate: int = 10, protocol: str = "tcp") -> Dict:
        """Add a new video stream"""
        payload = {
            "key": "sensor",
            "value": {
                "camera_id": camera_id,
                "camera_name": camera_name,
                "camera_url": rtsp_url,
                "change": "camera_add",
                "metadata": {
                    "resolution": resolution,
                    "codec": codec,
                    "framerate": framerate,
                }
            },
            "headers": {
                "source": "python_client",
                "created_at": datetime.now(UTC).isoformat() + "Z"
            }
        }

        return self._make_request("POST", "/stream/add", payload)

    def remove_stream(self, camera_id: str, rtsp_url: str) -> Dict:
        """Remove a video stream"""
        payload = {
            "key": "sensor",
            "value": {
                "camera_id": camera_id,
                "camera_url": rtsp_url,
                "change": "camera_remove"
            },
            "headers": {
                "source": "python_client",
                "created_at": datetime.now(UTC).isoformat() + "Z"
            }
        }

        return self._make_request("POST", "/stream/remove", payload)

    # INFERENCE SETTINGS
    def set_inference_interval(self, stream_id: str, interval: int) -> Dict:
        """Set inference interval (frame skip)"""
        payload = {
            "stream": {
                "stream_id": stream_id,
                "interval": interval
            }
        }
        try:
            return self._make_request("POST", "/infer/set-interval", payload)
        
        except Exception as e:
            print(f"Plugin type 'infer' not found, trying 'inferserver' endpoint: {e}")
            return self._make_request("POST", "/inferserver/set-interval", payload)
        
    # DECODER API
    def drop_frame_interval(self, stream_id: str, drop_interval: int) -> Dict:
        """Set frame drop interval for decoder"""
        payload = {
            "stream": {
                "stream_id": stream_id,
                "drop_interval": drop_interval
            }
        }
        return self._make_request("POST", "/dec/drop-frame-interval", payload)
    
    def skip_frames(self, stream_id: str, skip_frames: int) -> Dict:
        """Configuration values for "skip_frames" field of the schema:
            (0): - Decoder will decode all frames of the encoded bitstream
            (1): - Decoder will decode only non-reference frames of the encoded bitstream
            (2): - Decoder will decode only key frames of the encoded bitstream
        """
        if skip_frames not in [0, 1, 2]:
            raise ValueError("Invalid skip_frames value. Must be 0 (all), 1 (non-ref), or 2 (key).")
        payload = {
            "stream" : {
                "stream_id": stream_id,
                "skip_frames": skip_frames
            }
        }
        return self._make_request("POST", "/dec/skip-frames", payload)
    
    # ENCODER API
    def force_idr(self, stream_id: str, force_idr: int) -> Dict:
        """Force IDR frame generation on encoder"""
        if force_idr not in [0, 1]:
            raise ValueError("Invalid force_idr value. Must be 0 (disable) or 1 (enable).")
        
        payload = {
            "stream": {
                "stream_id": stream_id,
                "force_idr": force_idr
            }
        }
        return self._make_request("POST", "/enc/force-idr", payload)
    
    def force_intra(self, stream_id: str, force_intra: int) -> Dict:
        """Force Intra frame"""
        if force_intra not in [0, 1]:
            raise ValueError("Invalid force_intra value. Must be 0 (disable) or 1 (enable).")
        
        payload = {
            "stream": {
                "stream_id": stream_id,
                "force_intra": force_intra
            }
        }
        return self._make_request("POST", "/enc/force-intra", payload)

    def set_encoder_bitrate(self, stream_id: str, bitrate: int) -> Dict:
        """Set encoder bitrate"""
        payload = {
            "stream": {
                "stream_id": stream_id,
                "bitrate": bitrate
            }
        }
        return self._make_request("POST", "/enc/bitrate", payload)
    
    def iframe_interval(self, stream_id: str, iframe_interval: int) -> Dict:
        """Set I-frame interval for encoder"""
        payload = {
            "stream": {
                "stream_id": stream_id,
                "iframeinterval": iframe_interval
            }
        }
        return self._make_request("POST", "/enc/iframe-interval", payload)
    
    # ROI API
    def update_roi(self, stream_id: str, rois: list) -> Dict:
        """Update Region of Interest for preprocessing"""
        payload = {
            "stream": {
                "stream_id": stream_id,
                "roi_count": len(rois),
                "roi": rois
            }
        }
        return self._make_request("POST", "/roi/update", payload)
    
    # STREAMMUX API
    def set_batched_push_timeout(self, time_in_ms: int) -> Dict:
        """Change batched push timeout"""
        payload = {
            "stream": {
                "batches_push_timeout": time_in_ms
            }
        }
        return self._make_request("POST", "/mux/batched-push-timeout", payload)
    
    # OSD API
    def change_process_mode(self, stream_id: str, mode: int) -> Dict:
        """Change OSD process mode. 
            0 and 1, 0=CPU mode, 1=GPU mode"""
            
        if mode != 0 and mode != 1:
            raise ValueError("Invalid mode value. Must be 0 (CPU) or 1 (GPU).")
        
        payload = {
            "stream": {
                "stream_id": stream_id,
                "process_mode": mode
            }
        }
        return self._make_request("POST", "/osd/process-mode", payload)
    

def main():
    parser = argparse.ArgumentParser(description="DeepStream REST API Client")
    parser.add_argument("--host", default="localhost", help="Host address (default: localhost)")
    parser.add_argument("--port", type=int, default=9002,
                        help="Port of the pipeline to target (default: 9002)")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Health check
    subparsers.add_parser("health", help="Check pipeline health")

    # List streams
    subparsers.add_parser("list", help="List all streams")

    # Add stream
    add_parser = subparsers.add_parser("add", help="Add a stream")
    add_parser.add_argument("--id", required=True, help="Camera ID")
    add_parser.add_argument("--name", required=True, help="Camera name")
    add_parser.add_argument("--url", required=True, dest="stream_url", help="RTSP URL")
    add_parser.add_argument("--protocol", default="tcp", choices=["tcp", "udp"],
                            help="RTSP protocol (default: tcp)")

    # Remove stream
    remove_parser = subparsers.add_parser("remove", help="Remove a stream")
    remove_parser.add_argument("--id", required=True, help="Camera ID")
    remove_parser.add_argument("--url", required=True, dest="stream_url",
                               help="RTSP URL (original URL used when adding)")

    # Set inference interval
    interval_parser = subparsers.add_parser("interval", help="Set inference interval")
    interval_parser.add_argument("--stream", default="0", help="Stream ID")
    interval_parser.add_argument("--value", type=int, required=True, help="Interval value")
    
    # Change frame drop interval for decoder
    drop_interval_parser = subparsers.add_parser("drop-interval", help="Set decoder frame drop interval")
    drop_interval_parser.add_argument("--stream", default="0", help="Stream ID")
    drop_interval_parser.add_argument("--value", type=int, required=True, help="Drop interval value")
    
    # Change skip frames setting for decoder
    skip_frames_parser = subparsers.add_parser("skip-frames", help="Set decoder skip frames setting")
    skip_frames_parser.add_argument("--stream", default="0", help="Stream ID")
    skip_frames_parser.add_argument("--value", type=int, choices=[0, 1, 2], required=True, help="Skip frames value (0=all, 1=non-ref, 2=key)")
    
    # Force IDR frame generation on encoder
    idr_parser = subparsers.add_parser("force-idr", help="Force IDR frame generation on encoder")
    idr_parser.add_argument("--stream", default="0", help="Stream ID")
    idr_parser.add_argument("--value", type=int, choices=[0, 1], required=True, help="Force IDR value (0=disable, 1=enable)")
    
    # Force Intra frame on encoder
    intra_parser = subparsers.add_parser("force-intra", help="Force Intra frame on encoder")
    intra_parser.add_argument("--stream", default="0", help="Stream ID")
    intra_parser.add_argument("--value", type=int, choices=[0, 1], required=True, help="Force Intra value (0=disable, 1=enable)")
    
    # Change I-frame interval for encoder
    iframe_parser = subparsers.add_parser("iframe-interval", help="Set encoder I-frame interval")
    iframe_parser.add_argument("--stream", default="0", help="Stream ID")
    iframe_parser.add_argument("--value", type=int, required=True, help="I-frame interval value")
    
    # Change encoder bitrate
    bitrate_parser = subparsers.add_parser("bitrate", help="Set encoder bitrate")
    bitrate_parser.add_argument("--stream", default="0", help="Stream ID")
    bitrate_parser.add_argument("--value", type=int, required=True, help="Bitrate in kbps")
    
    # Update ROI
    roi_parser = subparsers.add_parser("roi", help="Update Region of Interest")
    roi_parser.add_argument("--stream", default="0", help="Stream ID")
    roi_parser.add_argument("--roi", type=list, required=True,
                            help="""ROI list in format: [{
                                    "roi_id": "0", 
                                    "left": x1, 
                                    "top": y1, 
                                    "width": w1, 
                                    "height": h1
                                    }, 
                                {
                                    "roi_id": "1", 
                                    "left": x2, 
                                    "top": y2, 
                                    "width": w2, 
                                    "height": h2
                                    },
                                ...]""")
    
    # Change batched push timeout for streammux
    mux_parser = subparsers.add_parser("mux-timeout", help="Set streammux batched push timeout")
    mux_parser.add_argument("--value", type=int, required=True, help="Timeout in microseconds")
    
    # Change OSD process mode
    osd_parser = subparsers.add_parser("osd-mode", help="Change OSD process mode")
    osd_parser.add_argument("--stream", default="0", help="Stream ID")
    osd_parser.add_argument("--mode", type=int, choices=[0, 1], required=True, help="Process mode (0=CPU, 1=GPU)")
    
    parser.add_argument("-h", "--help", action="help", help="Show this help message and exit")

    args = parser.parse_args()

    client = DeepStreamRESTClient(host=args.host, port=args.port)

    if args.command == "health":
        result = client.check_health()
        print(json.dumps(result, indent=2))

    elif args.command == "list":
        result = client.get_streams()
        print(json.dumps(result, indent=2))

    elif args.command == "add":
        result = client.add_stream(args.id, args.name, args.stream_url)
        print(json.dumps(result, indent=2))

    elif args.command == "remove":
        result = client.remove_stream(args.id, args.stream_url)
        print(json.dumps(result, indent=2))

    elif args.command == "interval":
        result = client.set_inference_interval(args.stream, args.value)
        print(json.dumps(result, indent=2))
        
    elif args.command == "drop-interval":
        result = client.drop_frame_interval(args.stream, args.value)
        print(json.dumps(result, indent=2))
        
    elif args.command == "skip-frames":
        result = client.skip_frames(args.stream, args.value)
        print(json.dumps(result, indent=2))
        
    elif args.command == "force-idr":
        result = client.force_idr(args.stream, args.value)
        print(json.dumps(result, indent=2))
        
    elif args.command == "force-intra":
        result = client.force_intra(args.stream, args.value)
        print(json.dumps(result, indent=2))
        
    elif args.command == "iframe-interval":
        result = client.iframe_interval(args.stream, args.value)
        print(json.dumps(result, indent=2))
        
    elif args.command == "bitrate":
        result = client.set_encoder_bitrate(args.stream, args.value)
        print(json.dumps(result, indent=2))

    elif args.command == "mux-timeout":
        result = client.set_mux_timeout(args.value)
        print(json.dumps(result, indent=2))

    elif args.command == "osd-mode":
        result = client.set_osd_mode(args.stream, args.mode)
        print(json.dumps(result, indent=2))
        
    elif args.command == "roi":
        result = client.update_roi(args.stream, args.roi)
        print(json.dumps(result, indent=2))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
