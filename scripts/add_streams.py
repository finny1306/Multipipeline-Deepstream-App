import subprocess
import time
import sys
import argparse

# Configuration

def run_command(cmd_list):
    """Executes a shell command provided as a list."""
    try:
        # Check if rest_api_client.py exists to avoid confusing errors
        result = subprocess.run(cmd_list, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Success: {result.stdout.strip()}")
        else:
            print(f"Error: {result.stderr.strip()}")
    except FileNotFoundError:
        print("Error: 'rest_api_client.py' not found. Make sure it is in this folder.")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Add multiple streams using rest_api_client.py")
    parser.add_argument("--max_streams", type=int, default=10, help="Maximum number of streams to add")
    parser.add_argument("--sleep_time", type=int, default=1, help="Sleep time between adding streams")
    parser.add_argument("--mode", choices=['rtsp-h264', 'rtsp-h265', 'srt-h264', 'srt-h265'],
                        default='srt-h264',
                        help="Protocol + codec (default: srt-h264)")
    parser.add_argument("--start_from", type=int, default=1, help="Starting stream number")
    args = parser.parse_args()

    print("--- Starting Stream Addition Sequence ---")

    # 1. Add the First Stream (File Source)
    print("Adding Initial File Stream (cam001)...")
    base_cmd = ["python3", "rest_api_client.py", "add"]
    
    if args.start_from == 1:
        # only run if starting from 1
        initial_args = [
            "--id", "cam000",
            "--name", "Front Door File",
            "--url", "file:///workspace/test-media/sample_1080p_h264_15fps.mp4"
        ]
        run_command(base_cmd + initial_args)
        
        time.sleep(args.sleep_time)

    # 2. Add RTSP Streams Loop
    if 'rtsp' in args.mode:
        print("Adding RTSP Streams...") 
        for i in range(args.start_from, args.max_streams + 1):
            # Format ID: cam0001, cam0002...
            cam_id = f"cam{i:04d}"
            stream_url = f"rtsp://mediamtx:8554/stream{i}"
            
            print(f"Adding RTSP Stream {i}/{args.max_streams} ({cam_id})...")
            
            loop_args = [
                "--id", cam_id,
                "--name", f"Front Door {i}",
                "--url", stream_url
            ]
            
            run_command(base_cmd + loop_args)
            time.sleep(args.sleep_time)

    elif 'srt' in args.mode:
        print("Adding SRT Streams...") 
        for i in range(args.start_from, args.max_streams + 1):
            # Format ID: cam0001, cam0002...
            cam_id = f"cam{i:04d}"
            stream_url = f"srt://mediamtx:8890?streamid=read:stream{i}"
            
            print(f"Adding SRT Stream {i}/{args.max_streams} ({cam_id})...")
            
            loop_args = [
                "--id", cam_id,
                "--name", f"Front Door {i}",
                "--url", stream_url
            ]
            
            run_command(base_cmd + loop_args)
            time.sleep(args.sleep_time)

    print("--- All streams added ---")

if __name__ == "__main__":
    main()