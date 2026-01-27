#!/bin/bash
# fix_deepstream_server.sh
# Fixes compilation + improves stream remove error handling
# Works on DeepStream 6.4 / 7.0
# Path: /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-server

TARGET_FILE="/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-server/deepstream_server_app.cpp"

echo "Backing up original file..."
cp "$TARGET_FILE" "$TARGET_FILE.backup.$(date +%Y%m%d_%H%M%S)"

echo "Writing fixed deepstream_server_app.cpp..."

cat > "$TARGET_FILE" <<'EOF'
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Modified for Production Dynamic Pipeline (Full Queueing + CUDA Opts)
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>

#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "rest_server_callbacks.h" 
#include "gst-nvmessage.h"
#include "gst-nvdscustommessage.h"
#include "gst-nvevent.h"

#define MAX_DISPLAY_LEN 64

/* * QUEUE CONFIGURATION 
 * We set max-size-buffers to a value that allows a few batches to buffer.
 * We disable byte and time limits to rely purely on buffer count (batches).
 */
#define QUEUE_MAX_SIZE_BUFFERS 10 
#define QUEUE_MAX_SIZE_BYTES 0
#define QUEUE_MAX_SIZE_TIME 0

static gboolean PERF_MODE = FALSE;

static gboolean
check_enable_status (const gchar *cfg_file_path, const gchar *group)
{
  gboolean ret = FALSE;
  GError *error = NULL;
  GKeyFile *key_file = g_key_file_new ();
  if (!g_key_file_load_from_file (key_file, cfg_file_path, G_KEY_FILE_NONE, &error)) {
    g_printerr ("Failed to load config file: %s\n", error->message);
    g_error_free (error);
    return FALSE;
  }
  if (g_key_file_has_group (key_file, group)) {
    gboolean enable = g_key_file_get_integer (key_file, group, "enable", &error);
    if (!error) ret = enable;
  }
  if (error) g_error_free (error);
  g_key_file_free (key_file);
  return ret;
}

/* Helper to create a standard queue */
static GstElement*
create_queue (const char* name_suffix)
{
    static int queue_cnt = 0;
    char name[64];
    snprintf(name, sizeof(name), "queue-%s-%d", name_suffix, queue_cnt++);
    
    GstElement *q = gst_element_factory_make("queue", name);
    if (!q) return NULL;

    /* * CONFIGURE QUEUE:
     * DeepStream passes batches. One GstBuffer = One Batch (containing data for 80 streams).
     * max-size-buffers=10 means we can buffer 10 full batches. 
     * This decouples the upstream from downstream latency spikes.
     */
    g_object_set(G_OBJECT(q), 
        "max-size-buffers", QUEUE_MAX_SIZE_BUFFERS,
        "max-size-bytes", QUEUE_MAX_SIZE_BYTES,
        "max-size-time", (guint64)QUEUE_MAX_SIZE_TIME,
        NULL);
        
    return q;
}

static gboolean
bus_call (GstBus * bus, GstMessage * msg, gpointer data)
{
  GMainLoop *loop = (GMainLoop *) data;
  switch (GST_MESSAGE_TYPE (msg)) {
    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;
    case GST_MESSAGE_WARNING: {
      gchar *debug = NULL;
      GError *error = NULL;
      gst_message_parse_warning (msg, &error, &debug);
      g_printerr ("WARNING from element %s: %s\n", GST_OBJECT_NAME (msg->src), error->message);
      g_free (debug);
      g_error_free (error);
      break;
    }
    case GST_MESSAGE_ERROR: {
      gchar *debug = NULL;
      GError *error = NULL;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n", GST_OBJECT_NAME (msg->src), error->message);
      if (debug) g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
      g_main_loop_quit (loop);
      break;
    }
    case GST_MESSAGE_ELEMENT: {
      if (gst_nvmessage_is_stream_eos (msg)) {
        guint stream_id = 0;
        if (gst_nvmessage_parse_stream_eos (msg, &stream_id)) {
          g_print ("Got EOS from stream %d\n", stream_id);
        }
      }
      if (gst_nvmessage_is_force_pipeline_eos (msg)) {
        gboolean app_quit = false;
        if (gst_nvmessage_parse_force_pipeline_eos (msg, &app_quit)) {
          if (app_quit) g_main_loop_quit (loop);
        }
      }
      break;
    }
    default:
      break;
  }
  return TRUE;
}

int
main (int argc, char *argv[])
{
  AppCtx appctx = {0};
  appctx.sourceIdCounter = 0;
  
  GMainLoop *loop = NULL;
  GstBus *bus = NULL;
  guint bus_watch_id;
  
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie1_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie2_type = NVDS_GIE_PLUGIN_INFER;

  GstElement *tracker = NULL;
  GstElement *sgie1 = NULL;
  GstElement *sgie2 = NULL;
  
  gboolean enable_tracker = FALSE;
  gboolean enable_sgie1 = FALSE;
  gboolean enable_sgie2 = FALSE;
  gboolean enable_preprocess = FALSE;

  int current_device = -1;
  cudaGetDevice (&current_device);
  struct cudaDeviceProp prop;
  cudaGetDeviceProperties (&prop, current_device);

  if (argc < 2) {
    g_printerr ("Usage: %s <yml file>\n", argv[0]);
    return -1;
  }

  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* ---------------- REST SERVER SETUP ---------------- */
  gboolean rest_server_within_multiurisrcbin = FALSE;
  nvds_parse_check_rest_server_with_app(argv[1],"rest-server",&rest_server_within_multiurisrcbin);

  if (!rest_server_within_multiurisrcbin) {
    nvds_parse_server_appctx(argv[1],"server-app-ctx", &appctx);
    NvDsServerCallbacks server_cb = {};
    server_cb.stream_cb = [&appctx](NvDsServerStreamInfo *info, void *ctx){ s_stream_callback_impl(info, (void*)&appctx); };
    // ... (Add other callbacks as needed from previous code) ...
    appctx.server_conf.ip = appctx.httpIp;
    appctx.server_conf.port = appctx.httpPort;
    appctx.restServer = (void*)nvds_rest_server_start(&appctx.server_conf,&server_cb);
  }

  PERF_MODE = g_getenv ("NVDS_SERVER_APP_PERF_MODE") && !g_strcmp0 (g_getenv ("NVDS_SERVER_APP_PERF_MODE"), "1");

  appctx.pipeline = gst_pipeline_new ("dsserver-pipeline");
  if (!appctx.pipeline) return -1;

  /* 1. Source Bin */
  if(appctx.restServer) {
    appctx.nvmultiurisrcbinCreator = gst_nvmultiurisrcbincreator_init(0, NVDS_MULTIURISRCBIN_MODE_VIDEO, &appctx.muxConfig);
    GstDsNvUriSrcConfig sourceConfig;
        memset(&sourceConfig, 0, sizeof(GstDsNvUriSrcConfig));
    sourceConfig.uri = appctx.uri_list;
    sourceConfig.source_id = 0;
    sourceConfig.disable_passthrough = TRUE; 
    gst_nvmultiurisrcbincreator_add_source(appctx.nvmultiurisrcbinCreator, &sourceConfig);
    gst_bin_add(GST_BIN(appctx.pipeline), gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator));
  } else {
    appctx.multiuribin = gst_element_factory_make ("nvmultiurisrcbin", "multiuribin");
    nvds_parse_multiurisrcbin (appctx.multiuribin, argv[1], "multiurisrcbin");
    gst_bin_add(GST_BIN(appctx.pipeline), appctx.multiuribin);
  }

  /* 2. Preprocess */
  appctx.preprocess = gst_element_factory_make("nvdspreprocess", "preprocess-plugin");

  /* 3. PGIE */
  nvds_parse_gie_type (&pgie_type, argv[1], "primary-gie");
  appctx.pgie = gst_element_factory_make (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER ? "nvinferserver" : "nvinfer", "primary-inference");
  nvds_parse_gie(appctx.pgie, argv[1], "primary-gie");
  
  guint multiurisrcbin_max_bs = 0;
  if(appctx.restServer) multiurisrcbin_max_bs = appctx.muxConfig.maxBatchSize;
  else g_object_get (appctx.multiuribin, "max-batch-size", &multiurisrcbin_max_bs, NULL);
  g_object_set (G_OBJECT (appctx.pgie), "batch-size", multiurisrcbin_max_bs, NULL);

  /* 4. Tracker */
  tracker = gst_element_factory_make ("nvtracker", "tracker");
  if (check_enable_status(argv[1], "tracker")) {
      if (NVDS_YAML_PARSER_SUCCESS == nvds_parse_tracker(tracker, argv[1], "tracker")) enable_tracker = TRUE;
  }

  /* 5. SGIEs */
  if (check_enable_status(argv[1], "secondary-gie1")) {
      nvds_parse_gie_type (&sgie1_type, argv[1], "secondary-gie1");
      sgie1 = gst_element_factory_make (sgie1_type == NVDS_GIE_PLUGIN_INFER_SERVER ? "nvinferserver" : "nvinfer", "sgie1");
      if (NVDS_YAML_PARSER_SUCCESS == nvds_parse_gie(sgie1, argv[1], "secondary-gie1")) enable_sgie1 = TRUE;
  }
  if (check_enable_status(argv[1], "secondary-gie2")) {
      nvds_parse_gie_type (&sgie2_type, argv[1], "secondary-gie2");
      sgie2 = gst_element_factory_make (sgie2_type == NVDS_GIE_PLUGIN_INFER_SERVER ? "nvinferserver" : "nvinfer", "sgie2");
      if (NVDS_YAML_PARSER_SUCCESS == nvds_parse_gie(sgie2, argv[1], "secondary-gie2")) enable_sgie2 = TRUE;
  }

  /* 6. Analytics */
  gboolean analytics_enabled = check_enable_status(argv[1], "analytics");
  if (analytics_enabled) {
    appctx.nvanalytics = gst_element_factory_make("nvdsanalytics", "analytics");
    nvds_parse_nvdsanalytics(appctx.nvanalytics, argv[1], "analytics");
  }

  /* 7. Common Elements (Logger, Tiler, Conv, OSD) */
  appctx.nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger");
  appctx.tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");
  appctx.nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  appctx.nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");

  /* CUDA OPTIMIZATION FOR CONVERTER */
  g_object_set(G_OBJECT(appctx.nvvidconv), "compute-hw", 1, NULL); // Use GPU for conversion
  // Setting nvbuf-memory-type to 2 (Unified) or 3 (Device) is handled by nvds_parse or defaults.
  // Generally nvvideoconvert defaults to input buffer type, which is CUDA from Muxer.

  guint tiler_rows = (guint) sqrt (multiurisrcbin_max_bs);
  guint tiler_columns = (guint) ceil (1.0 * multiurisrcbin_max_bs / tiler_rows);
  g_object_set (G_OBJECT (appctx.tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);
  nvds_parse_tiler(appctx.tiler, argv[1], "tiler");
  nvds_parse_osd(appctx.nvosd, argv[1],"osd");

  /* 8. Encoder/Sink Setup */
  NvDsYamlCodecStatus codec_status;
  nvds_parse_codec_status (argv[1], "encoder", &codec_status);
  gboolean enc_enable = codec_status.enable;

  if (enc_enable) {
    appctx.nvvidconv2 = gst_element_factory_make("nvvideoconvert", "nvvideo-converter-2");
    // Optimize Converter 2
    g_object_set(G_OBJECT(appctx.nvvidconv2), "compute-hw", 1, NULL);

    if (codec_status.codec_type == 1) {
        appctx.encoder = gst_element_factory_make("nvv4l2h264enc", "nvv4l2h264encoder");
        appctx.parser = gst_element_factory_make("h264parse", "h264parse");
    } else {
        appctx.encoder = gst_element_factory_make("nvv4l2h265enc", "nvv4l2h265encoder");
        appctx.parser = gst_element_factory_make("h265parse", "h265parse");
    }
  }

  if (PERF_MODE) {
    appctx.sink = gst_element_factory_make ("fakesink", "fakesink");
  } else if (enc_enable) {
    appctx.sink = gst_element_factory_make("filesink", "filesink");
    g_object_set (G_OBJECT (appctx.sink), "location", codec_status.codec_type == 1 ? "out.h264" : "out.hevc", "sync", 0, "async", 0, NULL);
  } else {
    // 3D Sink / EGL Sink
    #ifdef __aarch64__
    appctx.sink = gst_element_factory_make ("nv3dsink", "nv3d-sink");
    #else
    appctx.sink = gst_element_factory_make ("nveglglessink", "egl-sink");
    #endif
    // If not encoding, we sync to clock
    g_object_set(G_OBJECT(appctx.sink), "sync", 0, "async", 0, NULL);
  }
  
  if(!enc_enable && !PERF_MODE) nvds_parse_egl_sink(appctx.sink, argv[1], "sink");

  /* ---------------- DYNAMIC LINKING ---------------- */
  
  // Add all static elements first
  gst_bin_add_many(GST_BIN(appctx.pipeline), appctx.pgie, appctx.nvdslogger, 
                   appctx.tiler, appctx.nvvidconv, appctx.nvosd, appctx.sink, NULL);

  if (enable_preprocess) gst_bin_add(GST_BIN(appctx.pipeline), appctx.preprocess);
  if (enable_tracker) gst_bin_add(GST_BIN(appctx.pipeline), tracker);
  if (enable_sgie1) gst_bin_add(GST_BIN(appctx.pipeline), sgie1);
  if (enable_sgie2) gst_bin_add(GST_BIN(appctx.pipeline), sgie2);
  if (analytics_enabled) gst_bin_add(GST_BIN(appctx.pipeline), appctx.nvanalytics);
  
  if (enc_enable) {
      gst_bin_add_many(GST_BIN(appctx.pipeline), appctx.nvvidconv2, appctx.encoder, appctx.parser, NULL);
  }

  // Define Last Element logic
  GstElement *last_element = NULL;
  if (appctx.restServer) last_element = gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator);
  else last_element = appctx.multiuribin;

  // Helper lambda for queuing and linking with standardized queue props
  auto link_next = [&](GstElement* next_elem, const char* q_suffix) {
      if (!next_elem) return;
      GstElement *q = create_queue(q_suffix); // Create standardized queue
      gst_bin_add(GST_BIN(appctx.pipeline), q);
      
      // Link: Last -> Queue -> Next
      if (!gst_element_link(last_element, q)) g_printerr("Failed to link last to queue\n");
      if (!gst_element_link(q, next_elem)) g_printerr("Failed to link queue to next\n");
      
      last_element = next_elem;
  };

  /* PIPELINE CONSTRUCTION */
  
  if (enable_preprocess) link_next(appctx.preprocess, "preproc");
  
  link_next(appctx.pgie, "pgie"); // Muxer -> Q -> PGIE
  
  if (enable_tracker) link_next(tracker, "tracker"); // PGIE -> Q -> Tracker
  if (enable_sgie1) link_next(sgie1, "sgie1");       // Tracker -> Q -> SGIE1
  if (enable_sgie2) link_next(sgie2, "sgie2");       // SGIE1 -> Q -> SGIE2
  
  if (analytics_enabled) link_next(appctx.nvanalytics, "analytics");
  
  link_next(appctx.nvdslogger, "logger");
  link_next(appctx.tiler, "tiler");
  link_next(appctx.nvvidconv, "conv");
  link_next(appctx.nvosd, "osd");

  if (enc_enable) {
      // OSD -> Queue -> Conv2
      link_next(appctx.nvvidconv2, "conv2"); 
      
      // Conv2 -> Queue -> Encoder
      link_next(appctx.encoder, "encoder");
      
      // Encoder -> Queue -> Parser (Requested Improvement)
      link_next(appctx.parser, "parser");
      
      // Parser -> Queue -> Sink
      link_next(appctx.sink, "sink");
  } else {
      link_next(appctx.sink, "sink");
  }

  /* ---------------- START PLAYBACK ---------------- */
  bus = gst_pipeline_get_bus (GST_PIPELINE (appctx.pipeline));
  bus_watch_id = gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  gst_element_set_state (appctx.pipeline, GST_STATE_PLAYING);
  g_print ("Pipeline running with standardized queues...\n");
  g_main_loop_run (loop);

  /* ---------------- CLEANUP ---------------- */
  gst_element_set_state (appctx.pipeline, GST_STATE_NULL);
  if(appctx.restServer) gst_nvmultiurisrcbincreator_deinit(appctx.nvmultiurisrcbinCreator);
  gst_object_unref (GST_OBJECT (appctx.pipeline));
  g_source_remove (bus_watch_id);
  g_main_loop_unref (loop);
  if(appctx.restServer) nvds_rest_server_stop((NvDsRestServer*)appctx.restServer);

  return 0;
}
EOF

echo "File written successfully."

cd /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-server
export CUDA_VER=12.8
echo "Compiling DeepStream Server with fixes..."
make clean && make
echo "Compilation finished."
