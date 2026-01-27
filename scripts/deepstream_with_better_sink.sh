#!/bin/bash
# fix_build_v4.sh
# 
# FIXES:
# 1. Rewrites the Makefile completely to fix the "recipe commences before first target" error.
# 2. Updates deepstream_server_app.cpp with the OSD debug probes and fixed properties.
# 3. Compiles the app.

TARGET_DIR="/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-server"
CPP_FILE="$TARGET_DIR/deepstream_server_app.cpp"
MAKEFILE="$TARGET_DIR/Makefile"

# Ensure we are in the right directory or fail if it doesn't exist
if [ ! -d "$TARGET_DIR" ]; then
    echo "ERROR: Target directory $TARGET_DIR does not exist."
    exit 1
fi

echo "=========================================="
echo "1. Writing correct Makefile..."
echo "=========================================="

# We write the FULL Makefile to ensure no formatting errors occur with sed
cat > "$MAKEFILE" <<'EOF'
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
################################################################################

CUDA_VER?=
ifeq ($(CUDA_VER),)
  $(error "CUDA_VER is not set")
endif

APP:= deepstream-server-app

CC:= g++

TARGET_DEVICE = $(shell g++ -dumpmachine | cut -f1 -d -)

NVDS_VERSION:=8.0

LIB_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/lib/
APP_INSTALL_DIR?=/opt/nvidia/deepstream/deepstream-$(NVDS_VERSION)/bin/

SRCS:= $(wildcard *.cpp)

INCS:= $(wildcard *.h)

PKGS:= yaml-cpp gstreamer-rtsp-server-1.0 gstreamer-1.0

OBJS:= $(SRCS:.cpp=.o)

CFLAGS+= -I../../../includes \
         -I../../apps-common/includes \
         -I /usr/local/cuda-$(CUDA_VER)/include \
         -I/usr/include/jsoncpp

CFLAGS+= $(shell pkg-config --cflags $(PKGS))

# Include yaml-cpp in LIBS cleanly
LIBS:= $(shell pkg-config --libs $(PKGS)) -lyaml-cpp

LIBS+= -L/usr/local/cuda-$(CUDA_VER)/lib64/ -lcudart -lnvdsgst_helper -lnvdsgst_customhelper -lnvds_rest_server -lm \
       -L$(LIB_INSTALL_DIR) -lnvdsgst_meta -lnvds_meta -lnvds_yml_parser \
       -lcuda -Wl,-rpath,$(LIB_INSTALL_DIR) -lnvds_rest_server

ifeq ($(TARGET_DEVICE),aarch64)
   LIBS+= -L/usr/lib/aarch64-linux-gnu/tegra/ -lgstnvcustomhelper -ljsoncpp
else
   LIBS+= -L$(LIB_INSTALL_DIR) -lgstnvcustomhelper -ljsoncpp
endif

all: $(APP)

%.o: %.cpp $(INCS) Makefile
	$(CC) -c -o $@ $(CFLAGS) $<

$(APP): $(OBJS) Makefile
	$(CC) -o $(APP) $(OBJS) $(LIBS)

install: $(APP)
	cp -rv $(APP) $(APP_INSTALL_DIR)

clean:
	rm -rf $(OBJS) $(APP)
EOF

echo "=========================================="
echo "2. Writing updated C++ App (v3)..."
echo "=========================================="

cat > "$CPP_FILE" <<'EOF'
/*
 * DeepStream Server App - Fixed Version 3
 * * FIXES:
 * - Removed invalid encoder property 'preset-level'
 * - Fixed bus_call state change handler
 * - Added OSD probe for detection debugging
 * - Proper metadata handling verification
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
#include <signal.h> 
#include <unistd.h> 
#include <gst/rtsp-server/rtsp-server.h> 
#include <yaml-cpp/yaml.h> 

#include "gstnvdsmeta.h"
#include "nvds_yml_parser.h"
#include "rest_server_callbacks.h" 
#include "gst-nvmessage.h"
#include "gst-nvdscustommessage.h"
#include "gst-nvevent.h"

// Queue settings - reasonable defaults
#define QUEUE_MAX_SIZE_BUFFERS 20
#define QUEUE_MAX_SIZE_BYTES (10 * 1024 * 1024)
#define QUEUE_MAX_SIZE_TIME (5 * GST_SECOND)

GMainLoop *global_loop = NULL;
GstElement *global_pipeline = NULL;  // For state change tracking

static GstElement* create_queue(const char* name_suffix) {
    static int queue_cnt = 0;
    char name[64];
    snprintf(name, sizeof(name), "queue-%s-%d", name_suffix, queue_cnt++);
    GstElement *q = gst_element_factory_make("queue", name);
    if (!q) {
        g_printerr("ERROR: Failed to create queue %s\n", name);
        return NULL;
    }
    g_object_set(G_OBJECT(q), 
        "max-size-buffers", (guint)QUEUE_MAX_SIZE_BUFFERS, 
        "max-size-bytes", (guint)QUEUE_MAX_SIZE_BYTES, 
        "max-size-time", (guint64)QUEUE_MAX_SIZE_TIME,
        "leaky", 0,
        NULL);
    return q;
}

static GstElement* create_leaky_queue(const char* name_suffix) {
    static int leaky_queue_cnt = 0;
    char name[64];
    snprintf(name, sizeof(name), "leaky-queue-%s-%d", name_suffix, leaky_queue_cnt++);
    GstElement *q = gst_element_factory_make("queue", name);
    if (!q) return NULL;
    g_object_set(G_OBJECT(q), 
        "max-size-buffers", (guint)5,
        "max-size-bytes", (guint)0, 
        "max-size-time", (guint64)0,
        "leaky", 2,
        NULL);
    return q;
}

/* OSD Sink Pad Buffer Probe - Debug detection metadata */
static GstPadProbeReturn osd_sink_pad_buffer_probe(GstPad *pad, GstPadProbeInfo *info, gpointer u_data) {
    GstBuffer *buf = (GstBuffer *)info->data;
    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    
    static guint frame_count = 0;
    static guint last_print_frame = 0;
    frame_count++;
    
    if (batch_meta) {
        guint total_objects = 0;
        
        for (NvDsMetaList *l_frame = batch_meta->frame_meta_list; l_frame != NULL; l_frame = l_frame->next) {
            NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)(l_frame->data);
            total_objects += frame_meta->num_obj_meta;
            
            // Print details every 150 frames (~10 seconds at 15fps)
            if (frame_count - last_print_frame >= 150) {
                g_print("[DEBUG] Frame %u: source_id=%u, num_objects=%u\n", 
                        frame_count, frame_meta->source_id, frame_meta->num_obj_meta);
                
                // Print first few objects
                int obj_count = 0;
                for (NvDsMetaList *l_obj = frame_meta->obj_meta_list; l_obj != NULL && obj_count < 5; l_obj = l_obj->next) {
                    NvDsObjectMeta *obj_meta = (NvDsObjectMeta *)(l_obj->data);
                    g_print("  [OBJ %d] class_id=%d, confidence=%.2f, label='%s', bbox=[%.0f,%.0f,%.0f,%.0f]\n",
                            obj_count,
                            obj_meta->class_id,
                            obj_meta->confidence,
                            obj_meta->obj_label ? obj_meta->obj_label : "NULL",
                            obj_meta->rect_params.left,
                            obj_meta->rect_params.top,
                            obj_meta->rect_params.width,
                            obj_meta->rect_params.height);
                    obj_count++;
                }
                last_print_frame = frame_count;
            }
        }
        
        // Print summary every 150 frames
        if (frame_count - last_print_frame == 0 && total_objects > 0) {
            g_print("[DETECTION SUMMARY] Frame %u: Total objects detected = %u\n", frame_count, total_objects);
        }
    } else {
        if (frame_count % 150 == 0) {
            g_print("[WARNING] Frame %u: No batch_meta found!\n", frame_count);
        }
    }
    
    return GST_PAD_PROBE_OK;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *)data;
    
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
            
        case GST_MESSAGE_WARNING: {
            gchar *debug = NULL;
            GError *error = NULL;
            gst_message_parse_warning(msg, &error, &debug);
            g_printerr("WARNING from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
            if (debug) g_printerr("Warning details: %s\n", debug);
            g_free(debug);
            g_error_free(error);
            break;
        }
        
        case GST_MESSAGE_ERROR: {
            gchar *debug = NULL;
            GError *error = NULL;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
            if (debug) g_printerr("Error details: %s\n", debug);
            g_free(debug);
            g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        
        case GST_MESSAGE_STATE_CHANGED: {
            // FIXED: Only track pipeline state changes, use global_pipeline
            if (global_pipeline && GST_MESSAGE_SRC(msg) == GST_OBJECT(global_pipeline)) {
                GstState old_state, new_state, pending_state;
                gst_message_parse_state_changed(msg, &old_state, &new_state, &pending_state);
                g_print("Pipeline state: %s -> %s\n",
                        gst_element_state_get_name(old_state),
                        gst_element_state_get_name(new_state));
            }
            break;
        }
        
        default:
            break;
    }
    return TRUE;
}

void sigint_handler(int sig) {
    if (global_loop && g_main_loop_is_running(global_loop)) {
        g_print("\nCaught SIGINT. Quitting loop...\n");
        g_main_loop_quit(global_loop);
    }
}

int main(int argc, char *argv[]) {
    AppCtx appctx = {0};
    appctx.sourceIdCounter = 0;
    
    GMainLoop *loop = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id = 0;
    
    GstElement *tracker = NULL, *sgie1 = NULL, *sgie2 = NULL, *sgie3 = NULL, *sgie4 = NULL, *sgie5 = NULL;
    GstElement *payloader = NULL, *qtmux = NULL, *caps_filter = NULL;
    GstElement *caps_filter_osd = NULL;
    GstRTSPServer *rtsp_server = NULL;
    
    gboolean enable_tracker = FALSE, enable_sgie1 = FALSE, enable_sgie2 = FALSE;
    gboolean enable_sgie3 = FALSE, enable_sgie4 = FALSE, enable_sgie5 = FALSE;
    gboolean enable_preprocess = FALSE;

    int current_device = -1;
    cudaGetDevice(&current_device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, current_device);
    g_print("Using GPU: %s\n", prop.name);

    if (argc < 2) {
        g_printerr("Usage: %s <yml file>\n", argv[0]);
        return -1;
    }

    // --- Parse YAML Config ---
    int sink_type = 3; 
    std::string output_file = "output.mkv";
    std::string sink_framerate = "15/1";
    int rtsp_port_num = 8554;
    int udp_port_num = 5400;
    bool custom_sink_enable = false;
    int tiler_rows_yml = 1;
    int tiler_cols_yml = 1;

    bool tracker_enable_yml = false;
    bool sgie1_enable_yml = false;
    bool sgie2_enable_yml = false;
    bool sgie3_enable_yml = false;
    bool sgie4_enable_yml = false;
    bool sgie5_enable_yml = false;
    bool analytics_enable_yml = false;
    int custom_sync = 1;
    int custom_qos = 0;

    try {
        YAML::Node config = YAML::LoadFile(argv[1]);
        
        if (config["sink"]) {
            if (config["sink"]["enable"] && config["sink"]["enable"].as<int>() == 1) custom_sink_enable = true;
            if (config["sink"]["type"]) sink_type = config["sink"]["type"].as<int>();
            if (config["sink"]["output-file"]) output_file = config["sink"]["output-file"].as<std::string>();
            if (config["sink"]["rtsp-port"]) rtsp_port_num = config["sink"]["rtsp-port"].as<int>();
            if (config["sink"]["udp-port"]) udp_port_num = config["sink"]["udp-port"].as<int>();
            if (config["sink"]["framerate"]) sink_framerate = config["sink"]["framerate"].as<std::string>();
            if (config["sink"]["sync"]) custom_sync = config["sink"]["sync"].as<int>();
            if (config["sink"]["qos"]) custom_qos = config["sink"]["qos"].as<int>();
        }

        if (config["tiler"]) {
            if (config["tiler"]["rows"]) tiler_rows_yml = config["tiler"]["rows"].as<int>();
            if (config["tiler"]["columns"]) tiler_cols_yml = config["tiler"]["columns"].as<int>();
        }

        if (config["tracker"] && config["tracker"]["enable"] && config["tracker"]["enable"].as<int>() == 1) 
            tracker_enable_yml = true;
        if (config["secondary-gie1"] && config["secondary-gie1"]["enable"] && config["secondary-gie1"]["enable"].as<int>() == 1) 
            sgie1_enable_yml = true;
        if (config["secondary-gie2"] && config["secondary-gie2"]["enable"] && config["secondary-gie2"]["enable"].as<int>() == 1) 
            sgie2_enable_yml = true;
        if (config["secondary-gie3"] && config["secondary-gie3"]["enable"] && config["secondary-gie3"]["enable"].as<int>() == 1) 
            sgie3_enable_yml = true;
        if (config["secondary-gie4"] && config["secondary-gie4"]["enable"] && config["secondary-gie4"]["enable"].as<int>() == 1) 
            sgie4_enable_yml = true;
        if (config["secondary-gie5"] && config["secondary-gie5"]["enable"] && config["secondary-gie5"]["enable"].as<int>() == 1) 
            sgie5_enable_yml = true;
        if (config["analytics"] && config["analytics"]["enable"] && config["analytics"]["enable"].as<int>() == 1) 
            analytics_enable_yml = true;

    } catch (const YAML::Exception& e) {
        g_printerr("Error parsing YAML: %s\n", e.what());
        return -1;
    }

    g_print("\n=== SINK CONFIGURATION ===\n");
    g_print("Sink Type: %d (1=File, 2=RTSP, 3=Display)\n", sink_type);
    g_print("Output File: %s\n", output_file.c_str());
    g_print("Framerate: %s\n", sink_framerate.c_str());
    g_print("Sync: %d\n", custom_sync);
    g_print("==========================\n\n");

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);
    global_loop = loop;
    signal(SIGINT, sigint_handler);

    // --- REST SERVER ---
    gboolean rest_server_within_multiurisrcbin = FALSE;
    nvds_parse_check_rest_server_with_app(argv[1], "rest-server", &rest_server_within_multiurisrcbin);

    if (!rest_server_within_multiurisrcbin) {
        nvds_parse_server_appctx(argv[1], "server-app-ctx", &appctx);
        NvDsServerCallbacks server_cb = {};
        server_cb.stream_cb = [&appctx](NvDsServerStreamInfo *info, void *ctx){ s_stream_callback_impl(info, (void*)&appctx); };
        server_cb.appinstance_cb = [&appctx](NvDsServerAppInstanceInfo *info, void *ctx){ s_appinstance_callback_impl(info, (void*)&appctx); };
        server_cb.osd_cb = [&appctx](NvDsServerOsdInfo *info, void *ctx){ s_osd_callback_impl(info, (void*)&appctx); };
        server_cb.mux_cb = [&appctx](NvDsServerMuxInfo *info, void *ctx){ s_mux_callback_impl(info, (void*)&appctx); };
        server_cb.enc_cb = [&appctx](NvDsServerEncInfo *info, void *ctx){ s_enc_callback_impl(info, (void*)&appctx); };
        server_cb.conv_cb = [&appctx](NvDsServerConvInfo *info, void *ctx){ s_conv_callback_impl(info, (void*)&appctx); };
        server_cb.inferserver_cb = [&appctx](NvDsServerInferServerInfo *info, void *ctx){ s_inferserver_callback_impl(info, (void*)&appctx); };
        server_cb.infer_cb = [&appctx](NvDsServerInferInfo *info, void *ctx){ s_infer_callback_impl(info, (void*)&appctx); };
        server_cb.dec_cb = [&appctx](NvDsServerDecInfo *info, void *ctx){ s_dec_callback_impl(info, (void*)&appctx); };
        server_cb.roi_cb = [&appctx](NvDsServerRoiInfo *info, void *ctx){ s_roi_callback_impl(info, (void*)&appctx); };

        appctx.server_conf.ip = appctx.httpIp;
        appctx.server_conf.port = appctx.httpPort;
        appctx.restServer = (void*)nvds_rest_server_start(&appctx.server_conf, &server_cb);
    }

    appctx.pipeline = gst_pipeline_new("dsserver-pipeline");
    global_pipeline = appctx.pipeline;  // For state change tracking
    if (!appctx.pipeline) {
        g_printerr("ERROR: Failed to create pipeline\n");
        return -1;
    }

    // 1. Source Bin
    if (appctx.restServer) {
        appctx.nvmultiurisrcbinCreator = gst_nvmultiurisrcbincreator_init(0, NVDS_MULTIURISRCBIN_MODE_VIDEO, &appctx.muxConfig);
        GstDsNvUriSrcConfig sourceConfig;
        memset(&sourceConfig, 0, sizeof(GstDsNvUriSrcConfig));
        sourceConfig.uri = appctx.uri_list;
        sourceConfig.source_id = 0;
        sourceConfig.disable_passthrough = TRUE; 
        gst_nvmultiurisrcbincreator_add_source(appctx.nvmultiurisrcbinCreator, &sourceConfig);
        gst_bin_add(GST_BIN(appctx.pipeline), gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator));
    } else {
        appctx.multiuribin = gst_element_factory_make("nvmultiurisrcbin", "multiuribin");
        nvds_parse_multiurisrcbin(appctx.multiuribin, argv[1], "multiurisrcbin");
        gst_bin_add(GST_BIN(appctx.pipeline), appctx.multiuribin);
    }

    // 2. Preprocess
    appctx.preprocess = gst_element_factory_make("nvdspreprocess", "preprocess-plugin");

    // 3. PGIE
    NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;
    nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie");
    appctx.pgie = gst_element_factory_make(
        pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER ? "nvinferserver" : "nvinfer", 
        "primary-inference");
    nvds_parse_gie(appctx.pgie, argv[1], "primary-gie");
    
    guint multiurisrcbin_max_bs = 50; 
    if (appctx.restServer) 
        multiurisrcbin_max_bs = appctx.muxConfig.maxBatchSize;
    else 
        g_object_get(appctx.multiuribin, "max-batch-size", &multiurisrcbin_max_bs, NULL);
    g_object_set(G_OBJECT(appctx.pgie), "batch-size", multiurisrcbin_max_bs, NULL);

    // 4. Tracker
    tracker = gst_element_factory_make("nvtracker", "tracker");
    if (tracker_enable_yml) {
        if (NVDS_YAML_PARSER_SUCCESS == nvds_parse_tracker(tracker, argv[1], "tracker")) 
            enable_tracker = TRUE;
    }

    // 5. SGIEs
    NvDsGieType sgie_type;
    if (sgie1_enable_yml) {
        nvds_parse_gie_type(&sgie_type, argv[1], "secondary-gie1");
        sgie1 = gst_element_factory_make(sgie_type == NVDS_GIE_PLUGIN_INFER_SERVER ? "nvinferserver" : "nvinfer", "sgie1");
        if (NVDS_YAML_PARSER_SUCCESS == nvds_parse_gie(sgie1, argv[1], "secondary-gie1")) enable_sgie1 = TRUE;
    }
    if (sgie2_enable_yml) {
        nvds_parse_gie_type(&sgie_type, argv[1], "secondary-gie2");
        sgie2 = gst_element_factory_make(sgie_type == NVDS_GIE_PLUGIN_INFER_SERVER ? "nvinferserver" : "nvinfer", "sgie2");
        if (NVDS_YAML_PARSER_SUCCESS == nvds_parse_gie(sgie2, argv[1], "secondary-gie2")) enable_sgie2 = TRUE;
    }
    if (sgie3_enable_yml) {
        nvds_parse_gie_type(&sgie_type, argv[1], "secondary-gie3");
        sgie3 = gst_element_factory_make(sgie_type == NVDS_GIE_PLUGIN_INFER_SERVER ? "nvinferserver" : "nvinfer", "sgie3");
        if (NVDS_YAML_PARSER_SUCCESS == nvds_parse_gie(sgie3, argv[1], "secondary-gie3")) enable_sgie3 = TRUE;
    }
    if (sgie4_enable_yml) {
        nvds_parse_gie_type(&sgie_type, argv[1], "secondary-gie4");
        sgie4 = gst_element_factory_make(sgie_type == NVDS_GIE_PLUGIN_INFER_SERVER ? "nvinferserver" : "nvinfer", "sgie4");
        if (NVDS_YAML_PARSER_SUCCESS == nvds_parse_gie(sgie4, argv[1], "secondary-gie4")) enable_sgie4 = TRUE;
    }
    if (sgie5_enable_yml) {
        nvds_parse_gie_type(&sgie_type, argv[1], "secondary-gie5");
        sgie5 = gst_element_factory_make(sgie_type == NVDS_GIE_PLUGIN_INFER_SERVER ? "nvinferserver" : "nvinfer", "sgie5");
        if (NVDS_YAML_PARSER_SUCCESS == nvds_parse_gie(sgie5, argv[1], "secondary-gie5")) enable_sgie5 = TRUE;
    }

    // 6. Analytics
    if (analytics_enable_yml) {
        appctx.nvanalytics = gst_element_factory_make("nvdsanalytics", "analytics");
        nvds_parse_nvdsanalytics(appctx.nvanalytics, argv[1], "analytics");
    }

    // 7. Common Elements
    appctx.nvdslogger = gst_element_factory_make("nvdslogger", "nvdslogger");
    appctx.tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");
    appctx.nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    appctx.nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");

    if (!appctx.nvosd) {
        g_printerr("ERROR: Failed to create nvdsosd\n");
        return -1;
    }

    g_object_set(G_OBJECT(appctx.nvvidconv), "compute-hw", 1, NULL);

    // TILER
    guint tiler_rows, tiler_columns;
    if (tiler_rows_yml > 0 && tiler_cols_yml > 0) {
        tiler_rows = (guint)tiler_rows_yml;
        tiler_columns = (guint)tiler_cols_yml;
        g_print("Using Custom Tiler Layout: %d x %d\n", tiler_rows, tiler_columns);
    } else {
        tiler_rows = (guint)sqrt(multiurisrcbin_max_bs);
        tiler_columns = (guint)ceil(1.0 * multiurisrcbin_max_bs / tiler_rows);
        g_print("Using Auto Tiler Layout: %d x %d\n", tiler_rows, tiler_columns);
    }

    g_object_set(G_OBJECT(appctx.tiler), "rows", tiler_rows, "columns", tiler_columns, NULL);
    nvds_parse_tiler(appctx.tiler, argv[1], "tiler");
    nvds_parse_osd(appctx.nvosd, argv[1], "osd");

    // Caps filter before OSD - force RGBA for proper OSD rendering
    caps_filter_osd = gst_element_factory_make("capsfilter", "osd_caps");
    if (caps_filter_osd) {
        GstCaps *caps_osd = gst_caps_from_string("video/x-raw(memory:NVMM), format=RGBA");
        g_object_set(G_OBJECT(caps_filter_osd), "caps", caps_osd, NULL);
        gst_caps_unref(caps_osd);
        g_print("[CONFIG] OSD input format: RGBA\n");
    }

    /* --- SINK LOGIC --- */
    NvDsYamlCodecStatus codec_status;
    nvds_parse_codec_status(argv[1], "encoder", &codec_status);
    gboolean enc_enable = (sink_type == 1 || sink_type == 2); 

    if (enc_enable) {
        appctx.nvvidconv2 = gst_element_factory_make("nvvideoconvert", "nvvideo-converter-2");
        g_object_set(G_OBJECT(appctx.nvvidconv2), 
                     "compute-hw", 1,
                     "nvbuf-memory-type", 0,
                     NULL);

        // Caps filter after nvvidconv2 - NV12 for encoder
        caps_filter = gst_element_factory_make("capsfilter", "encoder_caps");
        if (caps_filter) {
            gchar *caps_str = g_strdup_printf(
                "video/x-raw(memory:NVMM), format=NV12, width=1920, height=1080, framerate=(fraction)%s",
                sink_framerate.c_str());
            g_print("[CONFIG] Encoder input caps: %s\n", caps_str);
            GstCaps *caps = gst_caps_from_string(caps_str);
            g_object_set(G_OBJECT(caps_filter), "caps", caps, NULL);
            gst_caps_unref(caps);
            g_free(caps_str);
        }

        if (codec_status.codec_type == 1) {
            appctx.encoder = gst_element_factory_make("nvv4l2h264enc", "nvv4l2h264encoder");
            appctx.parser = gst_element_factory_make("h264parse", "h264parse");
            g_print("[CONFIG] Using H264 Encoder\n");
        } else {
            appctx.encoder = gst_element_factory_make("nvv4l2h265enc", "nvv4l2h265encoder");
            appctx.parser = gst_element_factory_make("h265parse", "h265parse");
            g_print("[CONFIG] Using H265 Encoder\n");
        }
        
        // FIXED: Only use valid encoder properties
        g_object_set(G_OBJECT(appctx.encoder), 
                     "insert-sps-pps", TRUE, 
                     "idrinterval", 30,
                     "bitrate", 4000000,
                     "maxperf-enable", TRUE,
                     NULL);
        
        g_print("[CONFIG] Encoder: IDR=30, bitrate=4Mbps\n");
    }

    if (sink_type == 1) {
        g_print("[CONFIG] Selecting FILE Sink (MKV)\n");
        qtmux = gst_element_factory_make("matroskamux", "muxer");
        if (qtmux) {
            g_object_set(G_OBJECT(qtmux), "streamable", FALSE, NULL);
        }
        
        appctx.sink = gst_element_factory_make("filesink", "filesink");
        g_object_set(G_OBJECT(appctx.sink), 
                     "location", output_file.c_str(), 
                     "sync", custom_sync,
                     "async", FALSE,
                     NULL);
        
        g_print("[CONFIG] File output: %s (sync=%d)\n", output_file.c_str(), custom_sync);

    } else if (sink_type == 2) {
        g_print("[CONFIG] Selecting RTSP Sink\n");
        
        if (codec_status.codec_type == 1)
            payloader = gst_element_factory_make("rtph264pay", "rtp-payloader");
        else
            payloader = gst_element_factory_make("rtph265pay", "rtp-payloader");
        
        if (payloader) {
            g_object_set(G_OBJECT(payloader), "config-interval", 1, "pt", 96, NULL);
        }

        appctx.sink = gst_element_factory_make("udpsink", "udp-sink");
        g_object_set(G_OBJECT(appctx.sink), 
                     "host", "127.0.0.1", 
                     "port", udp_port_num, 
                     "async", FALSE, 
                     "sync", FALSE,
                     "qos", FALSE,
                     NULL);

        rtsp_server = gst_rtsp_server_new();
        gchar port_str[16];
        snprintf(port_str, sizeof(port_str), "%d", rtsp_port_num);
        g_object_set(rtsp_server, "service", port_str, NULL);

        GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(rtsp_server);
        GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();
        
        gchar *launch_str;
        if (codec_status.codec_type == 1) {
            launch_str = g_strdup_printf(
                "( udpsrc name=pay0 port=%d buffer-size=524288 "
                "caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)H264, payload=96\" )", 
                udp_port_num);
        } else {
            launch_str = g_strdup_printf(
                "( udpsrc name=pay0 port=%d buffer-size=524288 "
                "caps=\"application/x-rtp, media=video, clock-rate=90000, encoding-name=(string)H265, payload=96\" )", 
                udp_port_num);
        }
        
        gst_rtsp_media_factory_set_launch(factory, launch_str);
        gst_rtsp_media_factory_set_shared(factory, TRUE);
        gst_rtsp_media_factory_set_latency(factory, 100);
        
        gst_rtsp_mount_points_add_factory(mounts, "/ds-test", factory);
        g_object_unref(mounts);
        gst_rtsp_server_attach(rtsp_server, NULL);
        g_print("\n*** RTSP Streaming at rtsp://localhost:%d/ds-test ***\n\n", rtsp_port_num);
        g_free(launch_str);

    } else if (sink_type == 3) {
        g_print("[CONFIG] Selecting Display Sink\n");
        #ifdef __aarch64__
        appctx.sink = gst_element_factory_make("nv3dsink", "nv3d-sink");
        #else
        appctx.sink = gst_element_factory_make("nveglglessink", "egl-sink");
        #endif
        g_object_set(G_OBJECT(appctx.sink), "sync", 1, "async", 0, "qos", 1, NULL);
    } else {
        appctx.sink = gst_element_factory_make("fakesink", "fakesink");
    }

    /* ---------------- ADD ELEMENTS TO BIN ---------------- */
    gst_bin_add_many(GST_BIN(appctx.pipeline), 
                     appctx.pgie, appctx.nvdslogger, 
                     appctx.tiler, appctx.nvvidconv, 
                     caps_filter_osd, appctx.nvosd, 
                     appctx.sink, NULL);

    if (enable_preprocess) gst_bin_add(GST_BIN(appctx.pipeline), appctx.preprocess);
    if (enable_tracker) gst_bin_add(GST_BIN(appctx.pipeline), tracker);
    if (enable_sgie1) gst_bin_add(GST_BIN(appctx.pipeline), sgie1);
    if (enable_sgie2) gst_bin_add(GST_BIN(appctx.pipeline), sgie2);
    if (enable_sgie3) gst_bin_add(GST_BIN(appctx.pipeline), sgie3);
    if (enable_sgie4) gst_bin_add(GST_BIN(appctx.pipeline), sgie4);
    if (enable_sgie5) gst_bin_add(GST_BIN(appctx.pipeline), sgie5);
    if (analytics_enable_yml) gst_bin_add(GST_BIN(appctx.pipeline), appctx.nvanalytics);
    
    if (enc_enable) {
        gst_bin_add_many(GST_BIN(appctx.pipeline), appctx.nvvidconv2, caps_filter, appctx.encoder, appctx.parser, NULL);
        if (qtmux) gst_bin_add(GST_BIN(appctx.pipeline), qtmux);
        if (payloader) gst_bin_add(GST_BIN(appctx.pipeline), payloader);
    }

    /* ---------------- LINKING ---------------- */
    GstElement *last_element = NULL;
    if (appctx.restServer) 
        last_element = gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator);
    else 
        last_element = appctx.multiuribin;

    auto link_next = [&](GstElement* next_elem, const char* q_suffix, bool use_leaky = false) {
        if (!next_elem) {
            g_printerr("WARNING: Skipping NULL element after %s\n", q_suffix);
            return;
        }
        GstElement *q = use_leaky ? create_leaky_queue(q_suffix) : create_queue(q_suffix);
        if (!q) {
            g_printerr("ERROR: Failed to create queue for %s\n", q_suffix);
            return;
        }
        gst_bin_add(GST_BIN(appctx.pipeline), q);
        if (!gst_element_link(last_element, q)) {
            g_printerr("LINK ERROR: %s -> queue-%s\n", GST_ELEMENT_NAME(last_element), q_suffix);
        }
        if (!gst_element_link(q, next_elem)) {
            g_printerr("LINK ERROR: queue-%s -> %s\n", q_suffix, GST_ELEMENT_NAME(next_elem));
        }
        last_element = next_elem;
    };

    if (enable_preprocess) link_next(appctx.preprocess, "preproc");
    link_next(appctx.pgie, "pgie"); 
    if (enable_tracker) link_next(tracker, "tracker"); 
    if (enable_sgie1) link_next(sgie1, "sgie1");       
    if (enable_sgie2) link_next(sgie2, "sgie2");   
    if (enable_sgie3) link_next(sgie3, "sgie3"); 
    if (enable_sgie4) link_next(sgie4, "sgie4"); 
    if (enable_sgie5) link_next(sgie5, "sgie5");     
    if (analytics_enable_yml) link_next(appctx.nvanalytics, "analytics");
    
    link_next(appctx.nvdslogger, "logger");
    link_next(appctx.tiler, "tiler");
    link_next(appctx.nvvidconv, "conv");
    link_next(caps_filter_osd, "osd_caps");
    link_next(appctx.nvosd, "osd");

    /* Add probe to OSD sink pad to debug detections */
    GstPad *osd_sink_pad = gst_element_get_static_pad(appctx.nvosd, "sink");
    if (osd_sink_pad) {
        gst_pad_add_probe(osd_sink_pad, GST_PAD_PROBE_TYPE_BUFFER, osd_sink_pad_buffer_probe, NULL, NULL);
        gst_object_unref(osd_sink_pad);
        g_print("[DEBUG] OSD sink pad probe added for detection monitoring\n");
    }

    if (enc_enable) {
        link_next(appctx.nvvidconv2, "conv2"); 
        link_next(caps_filter, "caps"); 
        link_next(appctx.encoder, "encoder");
        link_next(appctx.parser, "parser");
        
        if (sink_type == 1 && qtmux) {
            link_next(qtmux, "muxer");
            link_next(appctx.sink, "sink");
        } 
        else if (sink_type == 2 && payloader) {
            link_next(payloader, "payloader", true);
            link_next(appctx.sink, "sink", true);
        } else {
            link_next(appctx.sink, "sink");
        }
    } else {
        link_next(appctx.sink, "sink");
    }

    bus = gst_pipeline_get_bus(GST_PIPELINE(appctx.pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    g_print("\n=== STARTING PIPELINE ===\n");
    GstStateChangeReturn ret = gst_element_set_state(appctx.pipeline, GST_STATE_PLAYING);
    if (ret == GST_STATE_CHANGE_FAILURE) {
        g_printerr("ERROR: Failed to start pipeline!\n");
        gst_object_unref(appctx.pipeline);
        return -1;
    }
    
    g_print("Pipeline running. Waiting for detections...\n");
    g_main_loop_run(loop);

    g_print("\nMain loop finished. Cleaning up...\n");

    // Proper shutdown for file writing
    if (sink_type == 1) {
        g_print("Sending EOS to finalize file...\n");
        gst_element_send_event(appctx.pipeline, gst_event_new_eos());
        GstBus *shutdown_bus = gst_element_get_bus(appctx.pipeline);
        gst_bus_timed_pop_filtered(shutdown_bus, 5 * GST_SECOND, GST_MESSAGE_EOS);
        gst_object_unref(shutdown_bus);
    }

    gst_element_set_state(appctx.pipeline, GST_STATE_NULL);
    if (appctx.restServer) gst_nvmultiurisrcbincreator_deinit(appctx.nvmultiurisrcbinCreator);
    gst_object_unref(GST_OBJECT(appctx.pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    if (appctx.restServer) nvds_rest_server_stop((NvDsRestServer*)appctx.restServer);

    g_print("Cleanup complete.\n");
    return 0;
}
EOF

echo "=========================================="
echo "3. Compiling..."
echo "=========================================="
cd "$TARGET_DIR"
export CUDA_VER=12.8
make clean && make

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "BUILD SUCCESSFUL!"
    echo "=========================================="
    echo ""
else
    echo ""
    echo "BUILD FAILED! Check errors above."
fi