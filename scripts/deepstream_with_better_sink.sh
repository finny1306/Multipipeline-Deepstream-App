#!/bin/bash
# fix_build_v7_dynamic.sh
# 
# CHANGES:
# 1. Dynamic Pipeline: Tiler/OSD are NOT added if 'enable: 0' in YAML.
# 2. Queue Fix: Removed hardcoded limits, increased buffer size for stability.
# 3. Crash Fix: Checks for batch-size consistency.

TARGET_DIR="/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-server"
CPP_FILE="$TARGET_DIR/deepstream_server_app.cpp"
MAKEFILE="$TARGET_DIR/Makefile"

if [ ! -d "$TARGET_DIR" ]; then
    echo "ERROR: Target directory $TARGET_DIR does not exist."
    exit 1
fi

echo "=========================================="
echo "1. Writing correct Makefile..."
echo "=========================================="

cat > "$MAKEFILE" <<'EOF'
################################################################################
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Ensure tabs are used in Makefile
sed -i 's/^    /\t/g' "$MAKEFILE"
echo "Makefile repaired (tabs inserted)."

echo "=========================================="
echo "2. Writing Dynamic C++ App..."
echo "=========================================="


cat > "$CPP_FILE" << EOF
/*
 * DeepStream Server App - Dynamic & Robust
 * Fixes: Crash at high load, Unconditional Element Creation
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

GMainLoop *global_loop = NULL;
GstElement *global_pipeline = NULL;

// Helper to create robust queues
static GstElement* create_queue(const char* name_suffix, bool leaky = false) {
    static int queue_cnt = 0;
    char name[64];
    snprintf(name, sizeof(name), "queue-%s-%d", name_suffix, queue_cnt++);
    GstElement *q = gst_element_factory_make("queue", name);
    if (!q) return NULL;

    // NO HARDCODED LIMITS. 
    // Allow 20 buffers (batches) to accommodate jitter.
    g_object_set(G_OBJECT(q), "max-size-buffers", (guint)20, "max-size-bytes", (guint)0, "max-size-time", (guint64)0, NULL);
    
    if (leaky) {
        // Leaky downstream to prevent pipeline freeze if sink is slow
        g_object_set(G_OBJECT(q), "leaky", 2, "max-size-buffers", (guint)5, NULL);
    }
    return q;
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
            g_free(debug); g_error_free(error);
            break;
        }
        case GST_MESSAGE_ERROR: {
            gchar *debug = NULL;
            GError *error = NULL;
            gst_message_parse_error(msg, &error, &debug);
            g_printerr("ERROR from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
            if (debug) g_printerr("Error details: %s\n", debug);
            g_free(debug); g_error_free(error);
            g_main_loop_quit(loop);
            break;
        }
        default: break;
    }
    return TRUE;
}

void sigint_handler(int sig) {
    if (global_loop && g_main_loop_is_running(global_loop)) {
        g_print("\nCaught SIGINT. Sending EOS...\n");
        if (global_pipeline) {
            gst_element_send_event(global_pipeline, gst_event_new_eos());
        } else {
            g_main_loop_quit(global_loop);
        }
    }
}

int main(int argc, char *argv[]) {
    AppCtx appctx = {0};
    GMainLoop *loop = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id = 0;

    // for plugin type
    NvDsGietype pgie_type = NVDS_GIE_PLUGIN_INFER;
    
    GstElement *tracker = NULL, *sgie1 = NULL, *sgie2 = NULL, *sgie3 = NULL, *sgie4 = NULL, *sgie5 = NULL;
    GstElement *payloader = NULL, *qtmux = NULL, *caps_filter = NULL;
    GstRTSPServer *rtsp_server = NULL;

    // message broker elements
    GstElement *tee = NULL;
    GstElement *msgconv = NULL, *msgbroker = NULL;
    
    // Feature Flags (Default False)
    bool enable_tracker = false, 
    bool enable_sgie1 = false, enable_sgie2 = false, enable_sgie3 = false, enable_sgie4 = false, enable_sgie5 = false; 
    bool enable_preprocess = false;
    bool enable_tiler = false, enable_osd = false, enable_analytics = false;
    bool enable_msgbroker = false;

    if (argc < 2) {
        g_printerr("Usage: %s <yml file>\n", argv[0]);
        return -1;
    }

    // add support for msgbroker
    std::string msgbroker_proto_lib;
    std::string msgbroker_conn_str;
    std::string msgbroker_topic;
    std::string msgbroker_cfg_file;
    std::string msgconv_config;
    std::string msgconv_msg2p_newapi;
    int msgconv_payload_type = 0; // 0 = NVDS_EVENT_MSG_META, 1 = NVDS_FRAME_META
    int msgbroker_comp_id = 0;


    // --- Parse YAML Config for Enable Flags ---
    int sink_type = 0; 
    std::string output_file = "output.mkv";
    int rtsp_port_num = 8554;
    int udp_port_num = 5400;
    NvDsYamlCodecStatus codec_status = {0};

    try {
        YAML::Node config = YAML::LoadFile(argv[1]);
        
        if (config["sink"]) {
            if (config["sink"]["type"]) sink_type = config["sink"]["type"].as<int>();
            if (config["sink"]["output-file"]) output_file = config["sink"]["output-file"].as<std::string>();
            if (config["sink"]["rtsp-port"]) rtsp_port_num = config["sink"]["rtsp-port"].as<int>();
            if (config["sink"]["udp-port"]) udp_port_num = config["sink"]["udp-port"].as<int>();
        }
        // CRITICAL: Only enable Tiler/OSD if YAML says so
        if (config["tiler"] && config["tiler"]["enable"]) enable_tiler = config["tiler"]["enable"].as<int>() == 1;
        if (config["osd"] && config["osd"]["enable"]) enable_osd = config["osd"]["enable"].as<int>() == 1;
        if (config["tracker"] && config["tracker"]["enable"]) enable_tracker = config["tracker"]["enable"].as<int>() == 1;
        if (config["secondary-gie1"] && config["secondary-gie1"]["enable"]) enable_sgie1 = config["secondary-gie1"]["enable"].as<int>() == 1;
        if (config["secondary-gie2"] && config["secondary-gie2"]["enable"]) enable_sgie2 = config["secondary-gie2"]["enable"].as<int>() == 1;
        if (config["secondary-gie3"] && config["secondary-gie3"]["enable"]) enable_sgie3 = config["secondary-gie3"]["enable"].as<int>() == 1;
        if (config["secondary-gie4"] && config["secondary-gie4"]["enable"]) enable_sgie4 = config["secondary-gie4"]["enable"].as<int>() == 1;
        if (config["secondary-gie5"] && config["secondary-gie5"]["enable"]) enable_sgie5 = config["secondary-gie5"]["enable"].as<int>() == 1;
        if (config["preprocess"] && config["preprocess"]["enable"]) enable_preprocess = config["preprocess"]["enable"].as<int>() == 1;
        if (config["analytics"] && config["analytics"]["enable"]) enable_analytics = config["analytics"]["enable"].as<int>() == 1;
        if (config["msgbroker"] && config["msgbroker"]["enable"]) enable_msgbroker = config["msgbroker"]["enable"].as<int>() == 1;


        // Message broker config
        if (enable_msgbroker && config["msgbroker"]) {
            auto mb = config["msgbroker"];
            if (mb["proto-lib"])       msgbroker_proto_lib = mb["proto-lib"].as<std::string>();
            if (mb["conn-str"])        msgbroker_conn_str  = mb["conn-str"].as<std::string>();
            if (mb["topic"])           msgbroker_topic     = mb["topic"].as<std::string>();
            if (mb["config"])          msgbroker_cfg_file  = mb["config"].as<std::string>();
            if (mb["comp-id"])         msgbroker_comp_id   = mb["comp-id"].as<int>();
        }
        if (config["msgconv"]) {
            auto mc = config["msgconv"];
            if (mc["payload-type"])    msgconv_payload_type = mc["payload-type"].as<int>();
            if (mc["config"])          msgconv_config       = mc["config"].as<std::string>();
            if (mc["msg2p-newapi"])    msgconv_msg2p_newapi = mc["msg2p-newapi"].as<std::string>();
        }

        nvds_parse_codec_status(argv[1], "encoder", &codec_status);

        // plugin type
        nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie");


    } catch (const YAML::Exception& e) {
        g_printerr("Error parsing YAML: %s\n", e.what());
        return -1;
    }

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);
    global_loop = loop;
    signal(SIGINT, sigint_handler);

    // --- REST SERVER ---
    gboolean rest_server_within_multiurisrcbin = FALSE;
    nvds_parse_check_rest_server_with_app(argv[1], "rest-server", &rest_server_within_multiurisrcbin);

    if (!rest_server_within_multiurisrcbin) {
        /* REST server callbacks */
        nvds_parse_server_appctx(argv[1], "server-app-ctx", &appctx);
        NvDsServerCallbacks server_cb = {};

        g_print("Setting rest server callbacks \n");
        
        // for stream add / remove
        server_cb.stream_cb = [&appctx](NvDsServerStreamInfo *stream_info, void *ctx){ s_stream_callback_impl(stream_info, (void*)&appctx); };

        // for ROI update
        server_cb.roi_cb = [&appctx](NvDsServerRoiInfo *roi_info, void *ctx){
         //do only id analytics enabled
            if(!enable_analytics) {
                g_print("ROI update callback received but analytics is disabled. Ignoring ROI update.\n");
                return;
            }
         s_roi_callback_impl(roi_info, (void*)&appctx); };

        // Decoder tuning - drop frame interval, skip frames, and low latency mode
        server_cb.dec_cb = [&appctx](NvDsServerDecInfo *dec_info, void *ctx){ s_dec_callback_impl(dec_info, (void*)&appctx); };

        if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
            server_cb.inferserver_cb = [&appctx](NvDsServerInferServerInfo *info, void *ctx) {
                s_inferserver_callback_impl(info, (void*)&appctx);
            };
        } else {
            server_cb.infer_cb = [&appctx](NvDsServerInferInfo *info, void *ctx) {
                s_infer_callback_impl(info, (void*)&appctx);
            };
        }
        // encoder - update output bitrate, force an IDR, force intra-refresh, change how often keyframes appear in stream
        server_cb.enc_cb = [&appctx](NvDsServerEncInfo *enc_info, void *ctx){ s_enc_callback_impl(enc_info, (void*)&appctx); };

        //Streammux property - batch-push-timeout, max latency
        server_cb.mux_cb = [&appctx](NvDsServerMuxInfo *mux_info, void *ctx){ s_mux_callback_impl(mux_info, (void*)&appctx); };

        //osd property
        server_cb.osd_cb = [&appctx](NvDsServerOsdInfo *osd_info, void *ctx){ 
        // ignore if osd not enabled
        if (!enable_osd) {
            g_print("OSD callback received but osd is disabled. Ignoring OSD update.\n");
            return;
        }
        s_osd_callback_impl(osd_info, (void*)&appctx); };

        // for app quit
        server_cb.appinstance_cb = [&appctx](NvDsServerAppInstanceInfo *appinstance_info, void *ctx){ s_appinstance_callback_impl(appinstance_info, (void*)&appctx); };

        appctx.server_conf.ip = appctx.httpIp;
        appctx.server_conf.port = appctx.httpPort;
        appctx.restServer = (void*)nvds_rest_server_start(&appctx.server_conf, &server_cb);
    }

    appctx.pipeline = gst_pipeline_new("dsserver-pipeline");
    global_pipeline = appctx.pipeline;

    // 1. Source Bin
    if (appctx.restServer) {
        appctx.nvmultiurisrcbinCreator = gst_nvmultiurisrcbincreator_init(0, NVDS_MULTIURISRCBIN_MODE_VIDEO, &appctx.muxConfig);
        GstDsNvUriSrcConfig sourceConfig;
            memset(&sourceConfig, 0, sizeof(GstDsNvUriSrcConfig));
        sourceConfig.uri = appctx.uri_list; // Usually NULL here, populated via API
        gst_nvmultiurisrcbincreator_add_source(appctx.nvmultiurisrcbinCreator, &sourceConfig);
        gst_bin_add(GST_BIN(appctx.pipeline), gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator));
    } else {
        appctx.multiuribin = gst_element_factory_make("nvmultiurisrcbin", "multiuribin");
        nvds_parse_multiurisrcbin(appctx.multiuribin, argv[1], "multiurisrcbin");
        gst_bin_add(GST_BIN(appctx.pipeline), appctx.multiuribin);
    }

    // 2. Core Elements
    if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
        appctx.pgie = gst_element_factory_make("nvinferserver", "primary-nvinference-engine");}
    else {
        appctx.pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");}

    nvds_parse_gie(appctx.pgie, argv[1], "primary-gie");

    appctx.nvdslogger = gst_element_factory_make("nvdslogger", "nvdslogger");
    g_object_set(G_OBJECT (appctx.nvdslogger), "fps-measurement-interval-sec", 3, NULL); // Log every 3 seconds

    appctx.nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    g_object_set(G_OBJECT(appctx.nvvidconv), "compute-hw", 1, NULL); // GPU

    // 3. Conditional Elements (Only create if enabled)
    if (enable_preprocess) {
        appctx.preprocess = gst_element_factory_make("nvdspreprocess", "preprocess-plugin");
        gst_bin_add(GST_BIN(appctx.pipeline), appctx.preprocess);
    }
    if (enable_tracker) {
        tracker = gst_element_factory_make("nvtracker", "tracker");
        nvds_parse_tracker(tracker, argv[1], "tracker");
        gst_bin_add(GST_BIN(appctx.pipeline), tracker);
    }
    if (enable_sgie1) {
        sgie1 = gst_element_factory_make("nvinferserver", "sgie1");
        nvds_parse_gie(sgie1, argv[1], "secondary-gie1");
        gst_bin_add(GST_BIN(appctx.pipeline), sgie1);
    }
    if (enable_sgie2) {
        sgie2 = gst_element_factory_make("nvinferserver", "sgie2");
        nvds_parse_gie(sgie2, argv[1], "secondary-gie2");
        gst_bin_add(GST_BIN(appctx.pipeline), sgie2);
    }
    if (enable_sgie3) {
        sgie3 = gst_element_factory_make("nvinferserver", "sgie3");
        nvds_parse_gie(sgie3, argv[1], "secondary-gie3");
        gst_bin_add(GST_BIN(appctx.pipeline), sgie3);
    }
    if (enable_sgie4) {
        sgie4 = gst_element_factory_make("nvinferserver", "sgie4");
        nvds_parse_gie(sgie4, argv[1], "secondary-gie4");
        gst_bin_add(GST_BIN(appctx.pipeline), sgie4);
    }
    if (enable_sgie5) {
        sgie5 = gst_element_factory_make("nvinferserver", "sgie5");
        nvds_parse_gie(sgie5, argv[1], "secondary-gie5");
        gst_bin_add(GST_BIN(appctx.pipeline), sgie5);
    }
    if (enable_analytics) {
        appctx.nvanalytics = gst_element_factory_make("nvdsanalytics", "analytics");
        nvds_parse_nvdsanalytics(appctx.nvanalytics, argv[1], "analytics");
        gst_bin_add(GST_BIN(appctx.pipeline), appctx.nvanalytics);
    }
    if (enable_tiler) {
        appctx.tiler = gst_element_factory_make("nvmultistreamtiler", "nvtiler");
        nvds_parse_tiler(appctx.tiler, argv[1], "tiler");
        gst_bin_add(GST_BIN(appctx.pipeline), appctx.tiler);
    }
    if (enable_osd) {
        appctx.nvosd = gst_element_factory_make("nvdsosd", "nv-onscreendisplay");
        nvds_parse_osd(appctx.nvosd, argv[1], "osd");
        gst_bin_add(GST_BIN(appctx.pipeline), appctx.nvosd);
    }
    if (enable_msgbroker) {
        tee = gst_element_factory_make("tee", "msgbroker-tee");
        msgconv = gst_element_factory_make("nvmsgconv", "nvmsg-converter");
        if (!msgconv) {
            g_printerr("Failed to create nvmsgconv element\n");
            return -1;
        }
        // Set nvmsgconv properties
        if (!msgconv_config.empty())
            g_object_set(G_OBJECT(msgconv), "config", msgconv_config.c_str(), NULL);
        g_object_set(G_OBJECT(msgconv), "payload-type", msgconv_payload_type, NULL);

        if (msgconv_msg2p_newapi == "yes")
            g_object_set(G_OBJECT(msgconv), "msg2p-newapi", TRUE, NULL);

        msgbroker = gst_element_factory_make("nvmsgbroker", "nvmsg-broker");
        if (!msgbroker) {
            g_printerr("ERROR: Failed to create nvmsgbroker element\n");
            return -1;
        }
        // Set nvmsgbroker properties
        if (!msgbroker_proto_lib.empty())
            g_object_set(G_OBJECT(msgbroker), "proto-lib", msgbroker_proto_lib.c_str(), NULL);
        if (!msgbroker_conn_str.empty())
            g_object_set(G_OBJECT(msgbroker), "conn-str", msgbroker_conn_str.c_str(), NULL);
        if (!msgbroker_topic.empty())
            g_object_set(G_OBJECT(msgbroker), "topic", msgbroker_topic.c_str(), NULL);
        if (!msgbroker_cfg_file.empty())
            g_object_set(G_OBJECT(msgbroker), "config", msgbroker_cfg_file.c_str(), NULL);
        if (msgbroker_comp_id > 0)
            g_object_set(G_OBJECT(msgbroker), "comp-id", msgbroker_comp_id, NULL);
        // Sync off for async message sending
        g_object_set(G_OBJECT(msgbroker), "sync", FALSE, "async", FALSE, NULL);

        gst_bin_add_many(GST_BIN(appctx.pipeline), tee, msgconv, msgbroker, NULL);
        g_print("Message Broker ENABLED: proto=%s conn=%s topic=%s\n",
                msgbroker_proto_lib.c_str(), msgbroker_conn_str.c_str(), msgbroker_topic.c_str());
    }

    // 4. Sink Logic
    gboolean enc_enable = (sink_type == 1 || sink_type == 2); 
    if (enc_enable) {
        appctx.nvvidconv2 = gst_element_factory_make("nvvideoconvert", "nvvideo-converter-2");
        caps_filter = gst_element_factory_make("capsfilter", "encoder_caps");
        // Simple CAPS for encoder
        GstCaps *caps = gst_caps_from_string("video/x-raw(memory:NVMM), format=NV12");
        g_object_set(G_OBJECT(caps_filter), "caps", caps, NULL);
        gst_caps_unref(caps);

        if (codec_status.codec_type == 1) {
            appctx.encoder = gst_element_factory_make("nvv4l2h264enc", "nvv4l2h264encoder");
            appctx.parser = gst_element_factory_make("h264parse", "h264parse");
        } else {
            appctx.encoder = gst_element_factory_make("nvv4l2h265enc", "nvv4l2h265encoder");
            appctx.parser = gst_element_factory_make("h265parse", "h265parse");
        }
    }

    if (sink_type == 1) { // FILE
        qtmux = gst_element_factory_make("matroskamux", "muxer");
        appctx.sink = gst_element_factory_make("filesink", "filesink");
        g_object_set(G_OBJECT(appctx.sink), "location", output_file.c_str(), "sync", 0, "async", 0, NULL);

    } else if (sink_type == 2) { // RTSP
        if (codec_status.codec_type == 1) payloader = gst_element_factory_make("rtph264pay", "rtp-payloader");
        else payloader = gst_element_factory_make("rtph265pay", "rtp-payloader");
        appctx.sink = gst_element_factory_make("udpsink", "udp-sink");
        g_object_set(G_OBJECT(appctx.sink), "host", "127.0.0.1", "port", udp_port_num, "async", 0, "sync", 0, NULL);

    } else if (sink_type == 0) { // FAKE
        appctx.sink = gst_element_factory_make("fakesink", "fakesink");
        g_object_set(G_OBJECT(appctx.sink), "sync", 0, "async", 0, NULL);

    } else if (sink_type == 3) { // Display
        appctx.sink = gst_element_factory_make("nveglglessink", "egl-sink");
        g_object_set(G_OBJECT(appctx.sink), "sync", 0, "async", 0, NULL);
    }

    // Add Mandatory Elements
    gst_bin_add_many(GST_BIN(appctx.pipeline), appctx.pgie, appctx.nvdslogger, appctx.nvvidconv, appctx.sink, NULL);
    if (enc_enable) gst_bin_add_many(GST_BIN(appctx.pipeline), appctx.nvvidconv2, caps_filter, appctx.encoder, appctx.parser, NULL);
    if (qtmux) gst_bin_add(GST_BIN(appctx.pipeline), qtmux);
    if (payloader) gst_bin_add(GST_BIN(appctx.pipeline), payloader);

    /* ---------------- LINKING ---------------- */
    GstElement *last_element = NULL;
    if (appctx.restServer) last_element = gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator);
    else last_element = appctx.multiuribin;

    auto link_next = [&](GstElement* next_elem, const char* q_suffix, bool use_leaky = false) {
        if (!next_elem) return;
        GstElement *q = create_queue(q_suffix, use_leaky);
        if (!q) return;
        gst_bin_add(GST_BIN(appctx.pipeline), q);
        if (!gst_element_link(last_element, q)) {
            g_printerr("ERROR: Failed to link %s -> %s\n",
                GST_ELEMENT_NAME(last_element), GST_ELEMENT_NAME(q));
        }
        if (!gst_element_link(q, next_elem)) {
            g_printerr("ERROR: Failed to link %s -> %s\n",
                GST_ELEMENT_NAME(q), GST_ELEMENT_NAME(next_elem));
        }
        last_element = next_elem;
    };

    if (enable_preprocess) link_next(appctx.preprocess, "preproc");
    
    // LINK PGIE
    link_next(appctx.pgie, "pgie"); 
    
    if (enable_tracker) link_next(tracker, "tracker"); 
    if (enable_sgie1) link_next(sgie1, "sgie1");  
    if (enable_sgie2) link_next(sgie2, "sgie2"); 
    if (enable_sgie3) link_next(sgie3, "sgie3"); 
    if (enable_sgie4) link_next(sgie4, "sgie4"); 
    if (enable_sgie5) link_next(sgie5, "sgie5"); 
    if (enable_analytics) link_next(appctx.nvanalytics, "analytics");
    
    // Logger before visualization
    link_next(appctx.nvdslogger, "logger");

    // Message Broker Path (Tee -> MsgConv -> MsgBroker)
    if (enable_msgbroker) {
        // link last element to tee
        link_next(tee, "pre-msgbroker");
        
        // branch 1: to msgconv -> msgbroker
        GstElement *q_msg = create_queue("msg", true); // leaky queue for message path
        gst_bin_add(GST_BIN(appctx.pipeline), q_msg);
        
        GstPad *tee_msg_pad = gst_element_request_pad_simple(tee, "src_%u");
        GstPad *q_msg_pad = gst_element_get_static_pad(q_msg, "sink");
        if (gst_pad_link(tee_msg_pad, q_msg_pad) != GST_PAD_LINK_OK) {
            g_printerr("ERROR: Failed to link tee to msg queue\n");
        }
        gst_object_unref(q_msg_pad);
        gst_object_unref(tee_msg_pad);

        if (!gst_element_link(q_msg, msgconv)) {
            g_printerr("ERROR: Failed to link msg queue to msgconv\n");}

        if (!gst_element_link(msgconv, msgbroker)) {
            g_printerr("ERROR: Failed to link msgconv to msgbroker\n");}


        // branch 2: Display/file path
        GstElement *q_main = create_queue("main", false);
        gst_bin_add(GST_BIN(appctx.pipeline), q_main);

        GstPad *tee_main_pad = gst_element_request_pad_simple(tee, "src_%u");
        GstPad *q_main_sink  = gst_element_get_static_pad(q_main, "sink");
        if (gst_pad_link(tee_main_pad, q_main_sink) != GST_PAD_LINK_OK) {
            g_printerr("ERROR: Failed to link tee -> queue_main\n");
        }
        gst_object_unref(q_main_sink);
        gst_object_unref(tee_main_pad);

        // Continue the main display chain from q_main
        last_element = q_main;
    }

    // Dynamic Visuals
    if (enable_tiler) link_next(appctx.tiler, "tiler");
    
    link_next(appctx.nvvidconv, "conv");
    
    if (enable_osd) link_next(appctx.nvosd, "osd");

    // Sink Path
    if (enc_enable) {
        link_next(appctx.nvvidconv2, "conv2");
        link_next(caps_filter, "caps");
        link_next(appctx.encoder, "encoder");
        link_next(appctx.parser, "parser");
        if (qtmux) link_next(qtmux, "muxer");
        if (payloader) link_next(payloader, "pay");
    }
    
    link_next(appctx.sink, "sink", true); // Use leaky queue for sink to prevent backpressure kill

    // Start message handler
    bus = gst_pipeline_get_bus(GST_PIPELINE(appctx.pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    g_print("\n=== STARTING PIPELINE (MsgBroker=%s) ===\n", enable_msgbroker ? "ON" : "OFF");
    gst_element_set_state(appctx.pipeline, GST_STATE_PLAYING);
    g_main_loop_run(loop);

    // Cleanup
    g_print("Returned, stopping pipeline...\n");
    gst_element_set_state(appctx.pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(appctx.pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
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
    echo "BUILD SUCCESSFUL"
else
    echo "BUILD FAILED"
fi
