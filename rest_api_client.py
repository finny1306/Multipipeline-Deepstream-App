#!/usr/bin/env python3
"""
DeepStream REST API Client
Manage video streams dynamically
"""

import requests
import json
import argparse
from datetime import datetime, UTC  # Updated import to include UTC

from typing import Dict, Any

class DeepStreamRESTClient:
    def __init__(self, base_url: str = "http://localhost:9002"):
        self.base_url = base_url
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
            print(f"Error: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response: {e.response.text}")
            return None
    
    def check_health(self) -> Dict:
        """Check DeepStream pipeline health"""
        return self._make_request("GET", "/health/get-dsready-state")
    
    def get_streams(self) -> Dict:
        """Get all active streams"""
        return self._make_request("GET", "/stream/get-stream-info")
    
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
                    "protocols": protocols
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
    
    def set_inference_interval(self, stream_id: str, interval: int) -> Dict:
        """Set inference interval (frame skip)"""
        payload = {
            "stream": {
                "stream_id": stream_id,
                "interval": interval
            }
        }
        return self._make_request("POST", "/infer/set-interval", payload)
    
    def set_encoder_bitrate(self, stream_id: str, bitrate: int) -> Dict:
        """Set encoder bitrate"""
        payload = {
            "stream": {
                "stream_id": stream_id,
                "bitrate": bitrate
            }
        }
        return self._make_request("POST", "/enc/bitrate", payload)
    
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

def main():
    parser = argparse.ArgumentParser(description="DeepStream REST API Client")
    parser.add_argument("--url", default="http://localhost:9002", help="Base URL")
    
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
    remove_parser.add_argument("--url", required=True, dest="stream_url", help="RTSP URL (original URL used when adding)")
    
    # Set inference interval
    interval_parser = subparsers.add_parser("interval", help="Set inference interval")
    interval_parser.add_argument("--stream", default="0", help="Stream ID")
    interval_parser.add_argument("--value", type=int, required=True, help="Interval value")
    
    args = parser.parse_args()
    
    client = DeepStreamRESTClient(args.url)
    
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
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()