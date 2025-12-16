# DeepStream REST API - Quick Start Guide

## Overview
This DeepStream setup now supports: 
 - Dynamic stream add/remove via REST API
 - Dynamic pipeline creation/deletion that is independent from one another
 - This app was tested with CUDA_VER=12.8 in L20 GPU

## Key Features
🔄 Dynamic Multi-Pipeline
 - Independent Instances: Each pipeline runs in its own process
 - Isolated REST APIs: Each on unique port (9000, 9001, ...)
 - No Interference: Failure in one pipeline doesn't affect others

📡 Dynamic Stream Management
 - Runtime Add/Remove: Streams can be added/removed without restart
 - High Capacity: ~120 concurrent, 15 FPS streams total across all pipelines
 - Multiple Protocols: RTSP, SRT, File sources

⚡ Performance Optimizations
 - Queue-Based Decoupling: 10-buffer queues prevent blocking
 - Batch Processing: All streams processed in single GPU batch
 - GPU-Accelerated: CUDA-optimized converters and inference
   
## Quick Start

### 1. Initial Setup
```bash

# Restart docker compose
docker compose down && docker compose up -d

# create a venv for some python scipts
```bash
python3 -m venv testenv
```
**Install additional libs and compile the app**
```bash
docker compose exec deepstream bash
./install.sh
./user_additional_install.sh
./update_rtpmanager.sh
```
**Compile the App**
```bash
# Run the new script
/workspace/scripts/dynamic_deepstream_server_with_sgie.sh

# ensure the message "Compilation finished" is seen. If not, do:
cd /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-server
export CUDA_VER=12.8
make clean && make
```

**Get the YOLO Custom Parser**
```bash
# Can get it by following the steps in https://github.com/NVIDIA-AI-IOT/deepstream_tools.git
```

### 2. Starting 
**Terminal 1: Run Stream**
```bash
source multipipeline-deepstream/testenv/bin/activate

# with 20 fps
python /root/multipipeline-deepstream/scripts/stream_publisher.py /root/multipipeline-deepstream/test-media/sample_1080p_h264_20fps.mp4 -n 64 --mode 'rtsp-h264'

python /root/multipipeline-deepstream/scripts/stream_publisher.py /root/multipipeline-deepstream/test-media/sample_1080p_h264_20fps.mp4 -n 64 --mode 'srt-h264'

#with 15 fps
python /root/multipipeline-deepstream/scripts/stream_publisher.py /root/multipipeline-deepstream/test-media/sample_1080p_h264_15fps.mp4 -n 50 --mode 'rtsp-h264'

#with 10 fps
python /root/multipipeline-deepstream/scripts/stream_publisher.py /root/multipipeline-deepstream/test-media/sample_1080p_h264_10fps.mp4 -n 130 --mode 'rtsp-h264'
```

**Terminal 2: DeepStream with REST API**
```bash
cd /root/multipipeline-deepstream
docker compose exec deepstream bash
```

**Run the python manager**
```bash
# use the venv 
source workspace/testenv/bin/activate
cd /workspace
python manager.py
```

### 3. Add New Pipelines
**To Add A New one**
```bash
# In a new terminal, inside the docker container, run
curl -X POST http://localhost:5000/pipelines/spawn \
-H "Content-Type: application/json" \
-d '{ 
		"config_path": "/workspace/configs/config_person.yml", # place config here
		"port": 9000   #change accordingly
		}'
```
**To Delete One**
```bash
curl -X DELETE "http://localhost:5000/pipelines/pipeline_id"
```

**To Kill All** \
Simply ctrl + C in the python manager terminal.


### 4. Test REST API
**Option A: Use Python Client**
```bash
# Call the pipeline via port first
python3 rest_api_client.py --url http://localhost:<port>

# Check health
python3 rest_api_client.py health

# List streams
python3 rest_api_client.py list

# file stream:
python3 rest_api_client.py add \
  --id cam001 \
  --name "Front Door" \
  --url file:///workspace/test-media/sample_1080p_h264_20fps.mp4

# RTSP stream:
python3 rest_api_client.py add \
  --id cam0001 \
  --name "Front Door" \
  --url rtsp://mediamtx:8554/stream1

# Remove stream
python3 rest_api_client.py remove --id cam001 --url rtsp://mediamtx:8554/stream1

# Set inference interval (process every 2nd frame)
python3 rest_api_client.py interval --stream 0 --value 2
```

**Option B: Use curl**
```bash
# Check status
curl http://localhost:9000/api/v1/health/get-dsready-state

# Add stream
curl -X POST http://localhost:9000/api/v1/stream/add \
  -H "Content-Type: application/json" \
  -d '{
    "key": "sensor",
    "value": {
      "camera_id": "test_001",
      "camera_name": "Test Stream",
      "camera_url": "rtsp://mediamtx:8554/stream1",
      "change": "camera_add"
    }
  }'

# Get streams
curl http://localhost:9000/api/v1/stream/get-stream-info

# Remove stream
curl -X POST http://localhost:9000/api/v1/stream/remove \
  -H "Content-Type: application/json" \
  -d '{
    "key": "sensor",
    "value": {
      "camera_id": "test_001",
      "change": "camera_remove"
    }
  }'
```

## View Output

**RTSP Stream:**
```bash
# VLC or ffplay
ffplay rtsp://localhost:8555/ds-out

# Or VLC
vlc rtsp://localhost:8555/ds-out
```

## REST API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health/get-dsready-state` | GET | Check pipeline status |
| `/api/v1/stream/get-stream-info` | GET | List active streams |
| `/api/v1/stream/add` | POST | Add new stream |
| `/api/v1/stream/remove` | POST | Remove stream |
| `/api/v1/infer/set-interval` | POST | Set inference interval |
| `/api/v1/enc/bitrate` | POST | Update encoder bitrate |
| `/api/v1/roi/update` | POST | Update ROI |


## Troubleshooting

**Port 9000 not accessible:**
```bash
# Check if port is exposed
docker ps | grep deepstream

# Check if DeepStream is running
curl http://localhost:9000/api/v1/health/get-dsready-state
```

**Stream not adding:**
```bash
# Verify RTSP URL is accessible
docker compose exec deepstream bash
gst-launch-1.0 rtspsrc location=rtsp://mediamtx:8554/stream1 ! fakesink
```

**Check logs:**
```bash
docker compose logs -f deepstream
```

## Important Notes

1. Each `camera_id` must be unique
2. Maximum 120 concurrent streams for best results for 1 pipeline (configurable in config)
3. Make sure all ports in config and when spawning a pipeline has the same value
4. Max-batch size == number of max streams that can be added
5. Changes take effect immediately without restart

## Performance Tuning

**Reduce GPU load:**
```bash
# Process every 2nd frame
python3 rest_api_client.py interval --stream 0 --value 2
```

**Adjust encoder bitrate:**
```bash
curl -X POST http://localhost:9000/api/v1/enc/bitrate \
  -d '{"stream":{"stream_id":"0","bitrate":3000000}}'
