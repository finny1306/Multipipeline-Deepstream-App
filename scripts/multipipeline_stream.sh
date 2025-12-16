#!/bin/bash
# fix_deepstream_server_custom.sh
# Fixes:
# 1. COMPILATION: Uses correct struct members (pipeline_width, pipeline_height)
# 2. FEATURE: Adds support for 'mux-config-file' in YAML to load external config
# 3. BUG: Fixes 0% GPU usage by correctly populating Mux Config

TARGET_DIR="/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-server"
TARGET_FILE="$TARGET_DIR/deepstream_server_app.cpp"

echo "Backing up original file..."
cp "$TARGET_FILE" "$TARGET_FILE.backup.$(date +%Y%m%d_%H%M%S)"

echo "Writing fixed deepstream_server_app.cpp..."

cat > "$TARGET_FILE" <<'EOF'
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Modified for Multi-Process Architecture (Custom Mux Config Support)
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
#include <string>  
#include <getopt.h> 

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

/* Struct Renamed to avoid collision */
typedef struct {
  gdouble fps[MAX_SOURCE_BINS];
  gdouble fps_avg[MAX_SOURCE_BINS];
  guint64 num_instances[MAX_SOURCE_BINS];
  guint64 diff_cur_prev_instances[MAX_SOURCE_BINS];
  GstClockTime last_sample_time;
} ServerPerfStruct;

static ServerPerfStruct perf_struct = {0};

/* PERF CALLBACK */
static gboolean
perf_print_callback (gpointer data)
{
  AppCtx *appctx = (AppCtx *) data;
  guint source_count = appctx->sourceIdCounter;
  g_print ("\n**PERF: Pipeline Running. Active Sources Added Total: %d\n", source_count);
  return TRUE;
}

/* Helper: Create Standardized Queue */
static GstElement*
create_queue (const char* name_suffix)
{
    static int queue_cnt = 0;
    char name[64];
    snprintf(name, sizeof(name), "queue-%s-%d", name_suffix, queue_cnt++);
    GstElement *q = gst_element_factory_make("queue", name);
    if (!q) return NULL;
    g_object_set(G_OBJECT(q), "max-size-buffers", QUEUE_MAX_SIZE_BUFFERS, "max-size-bytes", QUEUE_MAX_SIZE_BYTES, "max-size-time", (guint64)QUEUE_MAX_SIZE_TIME, NULL);
    return q;
}

/* * Helper: Parse Custom Mux Config File
 * This function reads your custom_streammux_config.txt and fills the struct 
 */
static void
parse_mux_config (const char* main_cfg_path, GstDsNvStreammuxConfig *muxConfig)
{
  GError *error = NULL;
  GKeyFile *key_file = g_key_file_new ();
  
  // 1. Set Safe Defaults
  muxConfig->pipeline_width = 1920;
  muxConfig->pipeline_height = 1080;
  muxConfig->maxBatchSize = 30;
  muxConfig->batched_push_timeout = 40000;
  muxConfig->gpu_id = 0;
  muxConfig->nvbuf_memory_type = 0; // 0=Device, 2=Unified
  muxConfig->compute_hw = 1; // GPU
  muxConfig->buffer_pool_size = 4;
  muxConfig->enable_padding = 0;
  muxConfig->attach_sys_ts_as_ntp = 1;

  // 2. Load Main YAML to find the path to the custom config
  if (!g_key_file_load_from_file (key_file, main_cfg_path, G_KEY_FILE_NONE, &error)) {
    g_printerr ("Failed to load config file: %s\n", error->message);
    g_error_free (error);
    return;
  }

  std::string custom_cfg_path = "";
  
  if (g_key_file_has_key(key_file, "server-app-ctx", "mux-config-file", NULL)) {
      char *path = g_key_file_get_string(key_file, "server-app-ctx", "mux-config-file", NULL);
      if (path) custom_cfg_path = path;
      g_free(path);
  }

  // If no custom file, try to read standard YAML keys as fallback
  if (custom_cfg_path.empty()) {
      if(g_key_file_has_key(key_file, "server-app-ctx", "pipeline_width", NULL))
        muxConfig->pipeline_width = g_key_file_get_integer(key_file, "server-app-ctx", "pipeline_width", NULL);
      if(g_key_file_has_key(key_file, "server-app-ctx", "pipeline_height", NULL))
        muxConfig->pipeline_height = g_key_file_get_integer(key_file, "server-app-ctx", "pipeline_height", NULL);
      if(g_key_file_has_key(key_file, "server-app-ctx", "maxBatchSize", NULL))
        muxConfig->maxBatchSize = g_key_file_get_integer(key_file, "server-app-ctx", "maxBatchSize", NULL);
  } else {
      // 3. Load Custom Mux Config (INI Format)
      GKeyFile *mux_key_file = g_key_file_new();
      GError *mux_err = NULL;
      
      g_print(">> Loading Custom Mux Config: %s\n", custom_cfg_path.c_str());
      
      if (!g_key_file_load_from_file (mux_key_file, custom_cfg_path.c_str(), G_KEY_FILE_NONE, &mux_err)) {
          g_printerr ("Error loading custom mux config: %s\n", mux_err->message);
          g_error_free(mux_err);
      } else {
          // Parse [property] group
          if (g_key_file_has_group(mux_key_file, "property")) {
              if (g_key_file_has_key(mux_key_file, "property", "width", NULL))
                  muxConfig->pipeline_width = g_key_file_get_integer(mux_key_file, "property", "width", NULL);
                  
              if (g_key_file_has_key(mux_key_file, "property", "height", NULL))
                  muxConfig->pipeline_height = g_key_file_get_integer(mux_key_file, "property", "height", NULL);
                  
              if (g_key_file_has_key(mux_key_file, "property", "batch-size", NULL))
                  muxConfig->maxBatchSize = g_key_file_get_integer(mux_key_file, "property", "batch-size", NULL);
                  
              if (g_key_file_has_key(mux_key_file, "property", "batched-push-timeout", NULL))
                  muxConfig->batched_push_timeout = g_key_file_get_integer(mux_key_file, "property", "batched-push-timeout", NULL);
                  
              if (g_key_file_has_key(mux_key_file, "property", "gpu-id", NULL))
                  muxConfig->gpu_id = g_key_file_get_integer(mux_key_file, "property", "gpu-id", NULL);
                  
              if (g_key_file_has_key(mux_key_file, "property", "nvbuf-memory-type", NULL))
                  muxConfig->nvbuf_memory_type = g_key_file_get_integer(mux_key_file, "property", "nvbuf-memory-type", NULL);
                  
              if (g_key_file_has_key(mux_key_file, "property", "enable-padding", NULL))
                  muxConfig->enable_padding = g_key_file_get_integer(mux_key_file, "property", "enable-padding", NULL);
                  
              if (g_key_file_has_key(mux_key_file, "property", "buffer-pool-size", NULL))
                  muxConfig->buffer_pool_size = g_key_file_get_integer(mux_key_file, "property", "buffer-pool-size", NULL);
                  
              if (g_key_file_has_key(mux_key_file, "property", "attach-sys-ts", NULL))
                  muxConfig->attach_sys_ts_as_ntp = g_key_file_get_integer(mux_key_file, "property", "attach-sys-ts", NULL);
          }
      }
      g_key_file_free(mux_key_file);
  }

  g_key_file_free (key_file);
  
  // Verify Config
  g_print("DEBUG: Mux Config Applied -> W:%d H:%d Batch:%d GPU:%d\n", 
      muxConfig->pipeline_width, muxConfig->pipeline_height, muxConfig->maxBatchSize, muxConfig->gpu_id);
      
  if (muxConfig->pipeline_width == 0 || muxConfig->pipeline_height == 0) {
      g_print("WARNING: Mux width/height is 0. Resetting to 1920x1080 to prevent crash.\n");
      muxConfig->pipeline_width = 1920;
      muxConfig->pipeline_height = 1080;
  }
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
    case GST_MESSAGE_ERROR: {
      gchar *debug = NULL;
      GError *error = NULL;
      gst_message_parse_error (msg, &error, &debug);
      g_printerr ("ERROR from element %s: %s\n", GST_OBJECT_NAME (msg->src), error->message);
      if (debug) g_printerr ("Error details: %s\n", debug);
      g_free (debug);
      g_error_free (error);
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
  
  g_mutex_init(&appctx.bincreator_lock);

  GMainLoop *loop = NULL;
  GstBus *bus = NULL;
  
  /* GIE Types */
  NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie1_type = NVDS_GIE_PLUGIN_INFER;
  NvDsGieType sgie2_type = NVDS_GIE_PLUGIN_INFER;

  /* Elements */
  GstElement *tracker = NULL;
  GstElement *sgie1 = NULL, *sgie2 = NULL;
  
  /* Enable Flags */
  gboolean enable_tracker = FALSE;
  gboolean enable_sgie1 = FALSE, enable_sgie2 = FALSE;
  gboolean enable_preprocess = FALSE;
  gboolean enable_tiler = FALSE;
  gboolean enable_osd = FALSE;
  gboolean enable_analytics = FALSE;

  /* --- 1. ARGUMENT PARSING --- */
  guint override_port = 0;
  int c;
  while ((c = getopt (argc, argv, "p:")) != -1) {
    switch (c) {
      case 'p':
        override_port = atoi(optarg);
        break;
      default:
        break;
    }
  }

  if (optind >= argc) {
    g_printerr ("Usage: %s [-p port] <yml file>\n", argv[0]);
    return -1;
  }
  const char* config_file = argv[optind];

  gst_init (&argc, &argv);
  loop = g_main_loop_new (NULL, FALSE);

  /* --- 2. REST SERVER SETUP --- */
  nvds_parse_server_appctx((char*)config_file, "server-app-ctx", &appctx);
  
  if (override_port != 0) {
      g_print(">> OVERRIDING CONFIG PORT with: %d\n", override_port);
      std::string port_str = std::to_string(override_port);
      appctx.server_conf.port = port_str; 
      appctx.httpPort = g_strdup(port_str.c_str()); 
  }

  NvDsServerCallbacks server_cb = {};
  server_cb.stream_cb = [&appctx](NvDsServerStreamInfo *info, void *ctx){ s_stream_callback_impl(info, (void*)&appctx); };
  server_cb.appinstance_cb = [&appctx](NvDsServerAppInstanceInfo *info, void *ctx){ s_appinstance_callback_impl(info, (void*)&appctx); };
  
  appctx.server_conf.ip = appctx.httpIp;
  if (appctx.server_conf.port.empty()) appctx.server_conf.port = appctx.httpPort;
  
  appctx.restServer = (void*)nvds_rest_server_start(&appctx.server_conf,&server_cb);
  if (!appctx.restServer) return -1;

  PERF_MODE = g_getenv ("NVDS_SERVER_APP_PERF_MODE") && !g_strcmp0 (g_getenv ("NVDS_SERVER_APP_PERF_MODE"), "1");

  appctx.pipeline = gst_pipeline_new ("dsserver-pipeline");
  if (!appctx.pipeline) return -1;

  /* --- 3. SOURCE BIN --- */
  
  // *** CRITICAL FIX: MANUALLY POPULATE MUX CONFIG ***
  parse_mux_config(config_file, &appctx.muxConfig);

  appctx.nvmultiurisrcbinCreator = gst_nvmultiurisrcbincreator_init(0, NVDS_MULTIURISRCBIN_MODE_VIDEO, &appctx.muxConfig);
  gst_bin_add(GST_BIN(appctx.pipeline), gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator));

  /* --- 4. ELEMENT CREATION (Robust YAML Logic) --- */
  
  auto is_yaml_enabled = [&](const char* group) -> bool {
      std::string cmd = "grep -A 2 '" + std::string(group) + ":' " + std::string(config_file) + " | grep 'enable: 1'";
      int ret = system(cmd.c_str());
      return (ret == 0);
  };

  enable_preprocess = is_yaml_enabled("preprocess");
  if(enable_preprocess) appctx.preprocess = gst_element_factory_make("nvdspreprocess", "preprocess-plugin");

  nvds_parse_gie_type (&pgie_type, (char*)config_file, "primary-gie");
  appctx.pgie = gst_element_factory_make (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER ? "nvinferserver" : "nvinfer", "primary-inference");
  nvds_parse_gie(appctx.pgie, (char*)config_file, "primary-gie");
  g_object_set (G_OBJECT (appctx.pgie), "batch-size", appctx.muxConfig.maxBatchSize, NULL);

  enable_tracker = is_yaml_enabled("tracker");
  if (enable_tracker) {
      tracker = gst_element_factory_make ("nvtracker", "tracker");
      nvds_parse_tracker(tracker, (char*)config_file, "tracker");
  }

  enable_sgie1 = is_yaml_enabled("secondary-gie1");
  if (enable_sgie1) {
      nvds_parse_gie_type (&sgie1_type, (char*)config_file, "secondary-gie1");
      sgie1 = gst_element_factory_make (sgie1_type == NVDS_GIE_PLUGIN_INFER_SERVER ? "nvinferserver" : "nvinfer", "sgie1");
      nvds_parse_gie(sgie1, (char*)config_file, "secondary-gie1");
  }

  enable_sgie2 = is_yaml_enabled("secondary-gie2");
  if (enable_sgie2) {
      nvds_parse_gie_type (&sgie2_type, (char*)config_file, "secondary-gie2");
      sgie2 = gst_element_factory_make (sgie2_type == NVDS_GIE_PLUGIN_INFER_SERVER ? "nvinferserver" : "nvinfer", "sgie2");
      nvds_parse_gie(sgie2, (char*)config_file, "secondary-gie2");
  }

  enable_analytics = is_yaml_enabled("analytics");
  if (enable_analytics) {
    appctx.nvanalytics = gst_element_factory_make("nvdsanalytics", "analytics");
    nvds_parse_nvdsanalytics(appctx.nvanalytics, (char*)config_file, "analytics");
  }

  appctx.nvdslogger = gst_element_factory_make ("nvdslogger", "nvdslogger"); 
  appctx.tiler = gst_element_factory_make ("nvmultistreamtiler", "nvtiler");
  
  guint tiler_rows = (guint) sqrt (appctx.muxConfig.maxBatchSize);
  guint tiler_columns = (guint) ceil (1.0 * appctx.muxConfig.maxBatchSize / tiler_rows);
  g_object_set (G_OBJECT (appctx.tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);
  nvds_parse_tiler(appctx.tiler, (char*)config_file, "tiler");

  appctx.nvvidconv = gst_element_factory_make ("nvvideoconvert", "nvvideo-converter");
  g_object_set(G_OBJECT(appctx.nvvidconv), "compute-hw", 1, NULL); 

  appctx.nvosd = gst_element_factory_make ("nvdsosd", "nv-onscreendisplay");
  if(is_yaml_enabled("osd")) nvds_parse_osd(appctx.nvosd, (char*)config_file,"osd");

  // Encoder/Sink
  NvDsYamlCodecStatus codec_status = {0};
  nvds_parse_codec_status ((char*)config_file, "encoder", &codec_status);
  gboolean enc_enable = codec_status.enable; 

  if (enc_enable) {
    appctx.nvvidconv2 = gst_element_factory_make("nvvideoconvert", "nvvideo-converter-2");
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
    #ifdef __aarch64__
    appctx.sink = gst_element_factory_make ("nv3dsink", "nv3d-sink");
    #else
    appctx.sink = gst_element_factory_make ("nveglglessink", "egl-sink");
    #endif
    g_object_set(G_OBJECT(appctx.sink), "sync", 0, "async", 0, NULL);
  }
  
  if(!enc_enable && !PERF_MODE) nvds_parse_egl_sink(appctx.sink, (char*)config_file, "sink");

  /* --- 5. PIPELINE ADDING --- */
  
  gst_bin_add_many(GST_BIN(appctx.pipeline), appctx.pgie, appctx.nvdslogger, appctx.nvvidconv, appctx.sink, NULL);
  
  if (enable_preprocess) gst_bin_add(GST_BIN(appctx.pipeline), appctx.preprocess);
  if (enable_tracker) gst_bin_add(GST_BIN(appctx.pipeline), tracker);
  if (enable_sgie1) gst_bin_add(GST_BIN(appctx.pipeline), sgie1);
  if (enable_sgie2) gst_bin_add(GST_BIN(appctx.pipeline), sgie2);
  if (enable_analytics) gst_bin_add(GST_BIN(appctx.pipeline), appctx.nvanalytics);
  gst_bin_add(GST_BIN(appctx.pipeline), appctx.tiler); 
  gst_bin_add(GST_BIN(appctx.pipeline), appctx.nvosd);
  
  if (enc_enable) {
      gst_bin_add_many(GST_BIN(appctx.pipeline), appctx.nvvidconv2, appctx.encoder, appctx.parser, NULL);
  }

  /* --- 6. LINKING WITH QUEUES --- */

  GstElement *last_element = gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator);

  auto link_next = [&](GstElement* next_elem, const char* q_suffix) {
      if (!next_elem) return;
      GstElement *q = create_queue(q_suffix); 
      gst_bin_add(GST_BIN(appctx.pipeline), q);
      gst_element_link(last_element, q);
      gst_element_link(q, next_elem);
      last_element = next_elem;
  };

  if (enable_preprocess) link_next(appctx.preprocess, "preproc");
  link_next(appctx.pgie, "pgie"); 

  if (enable_tracker) link_next(tracker, "tracker");
  if (enable_sgie1) link_next(sgie1, "sgie1");
  if (enable_sgie2) link_next(sgie2, "sgie2");

  if (enable_analytics) link_next(appctx.nvanalytics, "analytics");
  
  link_next(appctx.nvdslogger, "logger");
  link_next(appctx.tiler, "tiler");
  link_next(appctx.nvvidconv, "conv");
  link_next(appctx.nvosd, "osd");

  if (enc_enable) {
      link_next(appctx.nvvidconv2, "conv2"); 
      link_next(appctx.encoder, "encoder");
      link_next(appctx.parser, "parser");
      link_next(appctx.sink, "sink");
  } else {
      link_next(appctx.sink, "sink");
  }

  /* --- 7. RUN --- */
  bus = gst_pipeline_get_bus (GST_PIPELINE (appctx.pipeline));
  gst_bus_add_watch (bus, bus_call, loop);
  gst_object_unref (bus);

  g_timeout_add (5000, perf_print_callback, &appctx);

  gst_element_set_state (appctx.pipeline, GST_STATE_PLAYING);
  g_print ("Pipeline running... (Port: %s)\n", appctx.server_conf.port.c_str());
  g_main_loop_run (loop);

  /* --- 8. CLEANUP --- */
  gst_element_set_state (appctx.pipeline, GST_STATE_NULL);
  gst_nvmultiurisrcbincreator_deinit(appctx.nvmultiurisrcbinCreator);
  gst_object_unref (GST_OBJECT (appctx.pipeline));
  g_main_loop_unref (loop);
  nvds_rest_server_stop((NvDsRestServer*)appctx.restServer);

  return 0;
}
EOF

echo "Compiling deepstream-server-app..."
cd "$TARGET_DIR"
export CUDA_VER=12.8
make clean 2>/dev/null
make

if [ -f "deepstream-server-app" ]; then
    echo "SUCCESS: deepstream-server-app compiled successfully."
else
    echo "ERROR: Compilation failed."
    exit 1
fi