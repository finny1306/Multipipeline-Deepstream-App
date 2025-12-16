#!/usr/bin/env python3
import subprocess
import argparse
import time
import sys

processes = []

def publish(video_file, url_template, n, loop, mode):
    for i in range(1, n+1):
        url = url_template.replace("{}", str(i))

        if mode == "rtsp-h264":
            cmd = [
                'ffmpeg', '-re',
                '-stream_loop', '-1' if loop else '0',
                '-i', video_file,
                '-c:v', 'copy', '-c:a', 'copy',
                '-f', 'rtsp',
                '-rtsp_transport', 'tcp',
                url
            ]

        elif mode == "rtsp-h265":
            cmd = [
                'ffmpeg', '-re',
                '-stream_loop', '-1' if loop else '0',
                '-i', video_file,
                '-c:v', 'hevc_nvenc',
                '-preset', 'p4',
                '-b:v', '2M',
                '-bf', '0',
                '-g', '60',
                '-c:a', 'copy',
                '-f', 'rtsp',
                '-rtsp_transport', 'tcp',
                url
            ]

        elif "srt" in mode:
            cmd = [
                'ffmpeg', '-re',
                '-stream_loop', '-1' if loop else '0',
                '-i', video_file,
                '-c:v', 'copy' if mode == 'srt-h264' else 'hevc_nvenc',
            ]
            
            # Add encoding params only if transcoding
            if mode == 'srt-h265':
                cmd.extend(['-preset', 'p4', '-b:v', '2M', '-bf', '0', '-g', '60'])
            
            cmd.extend([
                '-c:a', 'aac', '-b:a', '64k',
                '-f', 'mpegts',
                url
            ])

        else:
            raise ValueError("Invalid mode")

        print(f"[{i:02d}] → {url}  ({mode})")
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processes.append(p)
        time.sleep(0.25)

    print(f"\nAll {n} streams started in mode '{mode}'. Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all streams...")
        for p in processes:
            p.terminate()
        for p in processes:
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Publish N streams via RTSP or SRT")
    parser.add_argument("video", help="Path to source video file")
    parser.add_argument("-n", "--num", type=int, default=8, help="Number of parallel streams")
    parser.add_argument("--url", default="srt://localhost:8890?streamid=publish:stream{}",
                        help="URL template")
    parser.add_argument("--no-loop", action="store_true", help="Do not loop video")
    parser.add_argument("--mode", choices=['rtsp-h264', 'rtsp-h265', 'srt-h264', 'srt-h265'],
                        default='srt-h264',
                        help="Protocol + codec (default: srt-h264)")

    args = parser.parse_args()

    # Auto-adjust default URL template for SRT
    if "rtsp" in args.mode:
        args.url = "rtsp://localhost:8554/stream{}"

    elif "srt" in args.mode:
        args.url = "srt://localhost:8890?streamid=publish:stream{}"
        if "mode=listener" in args.url:
            print("Error: You cannot use mode=listener for multiple streams on one port.")
            sys.exit(1)

    print(f"Starting {args.num} streams...")
    
    # Recommendation: Use 'srt-h264' (which maps to copy) to save CPU/GPU
    publish(args.video, args.url, args.num, loop=not args.no_loop, mode=args.mode)

