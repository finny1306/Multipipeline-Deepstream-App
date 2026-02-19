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
/workspace/scripts/deepstream_v7.sh
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
python3 /root/multipipeline-deepstream/scripts/stream_publisher.py /root/multipipeline-deepstream/test-media/sample_1080p_h264_15fps.mp4 -n 1 --mode 'rtsp-h264'

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
source /workspace/testenv/bin/activate
python /workspace/manager_v2.py
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
# Pipeline on port 9000, and localhost (change according to pipeline port)

# Check health
python rest_client.py --port 9000 health

# List streams
python rest_client.py --port 9000 list

# Remote host
python rest_client.py --host 192.168.1.50 --port 9001 list

# file stream:
python3 rest_api_client.py --port 9000 add \
  --id cam001 \
  --name "Front Door" \
  --url file:///workspace/test-media/sample_1080p_h264_15fps.mp4

# RTSP stream:
python3 rest_api_client.py --port 9000 add \
  --id cam0001 \
  --name "Front Door" \
  --url rtsp://mediamtx:8554/stream1

# Remove stream
python3 rest_api_client.py --port 9000 remove --id cam001 --url rtsp://mediamtx:8554/stream1

# Set inference interval (process every 2nd frame)
python3 rest_api_client.py --port 9000 interval --stream 0 --value 2
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

### More Details:
Note: Currently REST API version "v1" is supported.

Features supported with this application are:

1. Stream add/remove
  a. Stream add

  Endpoint: /api/v1/stream/add
  Curl command to add stream:

  curl -XPOST 'http://localhost:9000/api/v1/stream/add' -d '{
    "key": "sensor",
    "value": {
        "camera_id": "uniqueSensorID1",
        "camera_name": "front_door",
        "camera_url": "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4",
        "change": "camera_add",
        "metadata": {
            "resolution": "1920 x1080",
            "codec": "h264",
            "framerate": 30
        }
    },
    "headers": {
        "source": "vst",
        "created_at": "2021-06-01T14:34:13.417Z"
    }
  }'


  Expected output: The uri specified should be added to the display.
  Note: The camera_id should be unique for each newly added streams.

  b. Stream remove

  Endpoint: /api/v1/stream/remove
  Curl command to remove stream:

  curl -XPOST 'http://localhost:9000/api/v1/stream/remove' -d '{
    "key": "sensor",
    "value": {
        "camera_id": "uniqueSensorID1",
        "camera_name": "front_door",
        "camera_url": "file:///opt/nvidia/deepstream/deepstream/samples/streams/sample_1080p_h264.mp4",
        "change": "camera_remove",
        "metadata": {
            "resolution": "1920 x1080",
            "codec": "h264",
            "framerate": 30
        }
    },
    "headers": {
        "source": "vst",
        "created_at": "2021-06-01T14:34:13.417Z"
    }
  }'

  Expected output: The uri specified should be removed from the display.
  Note: The camera_id used to remove stream should be same as being used while adding stream using REST API.

2. ROI

  Endpoint: /api/v1/roi/update
  Curl command to update ROI:

  curl -XPOST 'http://localhost:9000/api/v1/roi/update' -d '{
    "stream": {
        "stream_id": "0",
        "roi_count": 2,
        "roi": [{
                "roi_id": "0",
                "left": 100,
                "top": 300,
                "width": 400,
                "height": 400
            },
            {
                "roi_id": "1",
                "left": 550,
                "top": 300,
                "width": 500,
                "height": 500
            }
        ]
    }
  }'

  Expected output: The updated roi dimension should be observed at display.

3. Decoder
  a. Drop frame interval

  Endpoint: /api/v1/dec/drop-frame-interval
  Configuration values for "drop_frame_interval" field of the schema: Range [0 - 30]
  Curl command to configure decoder drop-frame-interval property:

  curl -XPOST 'http://localhost:9000/api/v1/dec/drop-frame-interval' -d '{
  "stream":
    {
        "stream_id":"0",
        "drop_frame_interval":2
    }
  }'

  Expected output: The drop-frame-interval value will be set on the decoder.
  Decoder drop frame interval should reflect with every interval <value> frame
  given by decoder, rest all dropped for selected stream.

  b. Skip frame

  Endpoint: /api/v1/dec/skip-frames
  Configuration values for "skip_frames" field of the schema:
    (0): - Decode all frames
    (1): - Decode non-ref frames
    (2): - Decode key frames
  Curl command to configure decoder skip-frames property:

  curl -XPOST 'http://localhost:9000/api/v1/dec/skip-frames' -d '{
  "stream":
    {
        "stream_id":"0",
        "skip_frames":2
    }
  }'

  Expected output: The skip-frames property value will be set on the decoder.
  (0): - Decoder will decode all frames of the encoded bitstream
  (1): - Decoder will decode only non-reference frames of the encoded bitstream
  (2): - Decoder will decode only key frames of the encoded bitstream

4. Nvinfer

  Endpoint: /api/v1/infer/set-interval

  Curl command to configure nvinfer interval property:

  curl -XPOST 'http://localhost:9000/api/v1/infer/set-interval' -d '{
  "stream":
    {
        "stream_id":"0",
        "interval":2
    }
  }'

  Expected output: The interval value will be set on the nvinfer.
  Interval value specify consecutive batches will be skipped for inference for
  the video stream.

  Note: Disable/comment "input-tensor-meta" property in dsserver_pgie_config.yml
  to see "interval" property functionality of nvinfer/nvinferserver.
  Currently stream_id (specified in the schema) do not have any impact on specified
  stream_id, rather configuration is getting applied to all active streams.

5. Nvinferserver

  Endpoint: /api/v1/inferserver/set-interval
  Curl command to configure nvinferserver interval property:

  curl -XPOST 'http://localhost:9000/api/v1/inferserver/set-interval' -d '{
  "stream":
    {
        "stream_id":"0",
        "interval":2
    }
  }'

  Expected output: The interval value will be set on nvinferserver.
  Interval value specify consecutive batches will be skipped for inference for
  the video stream.

  Note: Currently stream_id (specified in the schema) do not have any impact on specified
  stream_id, rather configuration is getting applied to all active streams.

6. Encoder

  Note: By default encoder is disabled. To enable, set enable: 1 in the "encoder" group
  of dsserver_config.yml. Currently stream_id (specified in the schema) do not have
  any impact on specified stream_id, rather configuration is gettng applied on
  muxed encoded bitstream.

  a. Force-idr

  Endpoint: /api/v1/enc/force-idr
  Configuration value for "force_idr" field of the schema:
    (1): - Force IDR frame
  Curl command to configure encoder force idr frame property:

  curl -XPOST 'http://localhost:9000/api/v1/enc/force-idr' -d '{
  "stream":
    {
        "stream_id":"0",
        "force_idr":1
    }
  }'

  Expected output: The force-idr property value will be set on the encoder.
  Encoder force-idr property should reflect with insertion of the IDR frame with the
  encoded bitstream by the encoder.

  Note: By default encoder is disabled. To enable, set enable: 1 in the "encoder" group of dsserver_config.yml
  
  b. Force-intra

  Endpoint: /api/v1/enc/force-intra
  Configuration value for "force_intra" field of the schema:
    (1): - Force Intra frame
  Curl command to configure encoder force intra frame property:

  curl -XPOST 'http://localhost:9000/api/v1/enc/force-intra' -d '{
  "stream":
    {
        "stream_id":"0",
        "force_intra":1
    }
  }'

  Expected output: The force-intra property value will be set on the encoder.
  Encoder force-intra property should reflect with insertion of the intra frame with the
  encoded bitstream by the encoder.

  c. Bitrate

  Endpoint: /api/v1/enc/bitrate

  Curl command to configure encoder bitrate property:

  curl -XPOST 'http://localhost:9000/api/v1/enc/bitrate' -d '{
  "stream":
    {
        "stream_id":"0",
        "bitrate":2000000
    }
  }'

  Convert generated .h264 elementary bitstream to mp4 file using below commands:
  $ ffmpeg -i out.h264 -vcodec copy out.mp4
  $ mediainfo out.mp4

  Expected output: Encoder should be reconfigured to use updated bitrate <value>
  and provide corresponding encoded bitstream. Mediainfo should show Encoder bitrate
  corresponding to updated value.

  d. iframeinterval

  Endpoint: /api/v1/enc/iframe-interval

  Curl command to configure encoder iframeinterval property:

  curl -XPOST 'http://localhost:9000/api/v1/enc/iframe-interval' -d '{
  "stream":
    {
        "stream_id":"0",
        "iframeinterval":50
    }
  }'

  Expected output: The iframeinterval property value will be set on the encoder.
  Encoder iframeinterval property should reflect with insertion of the I-frame at
  specified interval with the encoded bitstream by the encoder.

7. Streammux

  Note: Applicable for old nvstreammux

  Endpoint: /api/v1/mux/batched-push-timeout
  Configuration value for "batched_push_timeout" field of the schema:
    (microseconds): - Timeout value
  Curl command to configure streammux batched pushed timeout property:

  curl -XPOST 'http://localhost:9000/api/v1/mux/batched-push-timeout' -d '{
  "stream":
    {
        "batched_push_timeout":100000
    }
  }'

  Expected output: The batched push timeout property value will be set on the nvstreammux.
  nvstreammux property should reflect with the timeout in microseconds to wait after the
  first buffer is available to push the batch even if the complete batch is not formed.

8. Nvdsosd

  Endpoint: /api/v1/osd/process-mode
  Configuration value for "process_mode" field of the schema:
    0 and 1, 0=CPU mode, 1=GPU mode
  Curl command to configure nvdsosd process_mode property:

  curl -XPOST 'http://localhost:9000/api/v1/osd/process-mode' -d '{
  "stream":
    {
        "stream_id":"0",
        "process_mode":0
    }
  }'

  Expected output: There would not be any visual change, but applied
  process-mode should be used for drawing bounding boxes.

9. Application Instance

  Application quit

  Endpoint: /api/v1/app/quit
  Configuration value for "app_quit" field of the schema:
  (1): - Application quit (boolean)
  Curl command to quit the sample application:

  curl -XPOST 'http://localhost:9000/api/v1/app/quit' -d '{
  "stream":
    {
        "app_quit":1
    }
  }'

  Expected output: The application should quit.

10. GET requests

   GET stream info

   Endpoint: /api/v1/stream/get-stream-info
   Curl command to get the stream info:

   curl -XGET 'http://localhost:9000/api/v1/stream/get-stream-info'
   OR
   curl -XGET 'http://localhost:9000/api/v1/stream/get-stream-info' -d '{}'

   Expected output: The sample stream-info response returned to the client:

   {
        "reason" : "GET_LIVE_STREAM_INFO_SUCCESS",
        "status" : "HTTP/1.1 200 OK",
        "stream-info" :
        {
                "stream-count" : 1,
                "stream-info" :
                [
				{
				"camera_id" : "UniqueSensorId1",
				"camera_name" : "UniqueSensorName1",
				"source_id" : 0
				}
				]
        }
  }

  Note: If source_id : -1 is observed in the above response, it signifies
  source_id is not available.

  GET DeepStream readiness info

   Endpoint: /api/v1/health/get-dsready-state
   Curl command to get the DS readiness info:

   curl -XGET 'http://localhost:9000/api/v1/health/get-dsready-state'
   OR
   curl -XGET 'http://localhost:9000/api/v1/health/get-dsready-state' -d '{}'

   Expected output: The sample pipepine state response returned to the client:

   {
        "health-info" :
        {
                "ds-ready" : "YES"
        },
        "reason" : "GET_DS_READINESS_INFO_SUCCESS",
        "status" : "HTTP/1.1 200 OK"
   }

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

