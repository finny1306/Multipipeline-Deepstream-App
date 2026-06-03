/**
 * deepstream_server_app.cpp — v12 (tee-based parallel inference)
 */

#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <set>
#include <map>
#include <mutex>
#include <vector>
#include <sstream>
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

/* ═══════════════════════════════════════════════════════════════════════════
 * DEMUX PAD CONFIGURATION
 * Max 100 concurrent streams, but demux needs 101 pads (0..100) because
 * nvmultiurisrcbin may assign source_id starting from 0.
 * Pads are consumed permanently — once used + deleted, that slot is gone.
 * ═══════════════════════════════════════════════════════════════════════════ */

static int max_demux_pads = 101;

/* ═══════════════════════════════════════════════════════════════════════════
 * DemuxPadTracker — thread-safe tracker for demux slot consumption
 * ═══════════════════════════════════════════════════════════════════════════ */
struct DemuxPadTracker {
    std::mutex mtx;
    int max_pads = max_demux_pads;
    int pads_consumed = 0;       // Total pads ever linked (never decreases)
    int active_streams = 0;      // Currently active streams


    void on_stream_added(int source_id) {
        std::lock_guard<std::mutex> lock(mtx);
        pads_consumed++;
        active_streams++;
        int remaining = max_pads - pads_consumed;

        /*g_print("[DemuxTracker] Stream ADDED: source_id=%d | "
                "Pad %d/%d consumed | %d slots remaining | %d active streams\n",
                source_id, pads_consumed, max_pads, remaining, active_streams); */
        if (remaining <= 10) {
            g_print("[DemuxTracker] WARNING: Only %d demux slots remaining!\n", remaining);
        }
    }

    void on_stream_removed(int source_id) {
        std::lock_guard<std::mutex> lock(mtx);
        active_streams = std::max(0, active_streams - 1);
        // srcid_to_branch.erase(source_id);  // No longer needed
        int remaining = max_pads - pads_consumed;
        g_print("[DemuxTracker] Stream REMOVED: source_id=%d | "
                "Pad %d/%d consumed | %d slots remaining | %d active streams\n",
                source_id, pads_consumed, max_pads, remaining, active_streams);
    }


    void print_status() {
        std::lock_guard<std::mutex> lock(mtx);
        int remaining = max_pads - pads_consumed;
        g_print("[DemuxTracker] STATUS: %d active | %d/%d pads consumed | %d remaining | assignments: {",
                active_streams, pads_consumed, max_pads, remaining);
    }
};

static DemuxPadTracker g_demux_tracker;

/* ═══════════════════════════════════════════════════════════════════════════
 * BranchFpsTracker — accurate per-source FPS measured at branch outputs
 *
 * nvdslogger sits after nvdsmetamux which mangles source_ids when branches
 * carry DIFFERENT frames (our architecture). This tracker counts frames
 * inside branch_fps_probe (before metamux) where source_ids are
 * correctly restored, then reports via the diagnostic timer.
 * ═══════════════════════════════════════════════════════════════════════════ */
struct BranchFpsTracker {
    std::mutex mtx;
    std::map<int, uint64_t> frame_counts;   // source_id → frames since last report
    struct timespec last_report_time;
    bool initialized = false;

    void tick(int source_id) {
        std::lock_guard<std::mutex> lock(mtx);
        if (!initialized) {
            clock_gettime(CLOCK_MONOTONIC, &last_report_time);
            initialized = true;
        }
        frame_counts[source_id]++;
    }

    void report_and_reset() {
        std::lock_guard<std::mutex> lock(mtx);
        if (!initialized || frame_counts.empty()) {
            return;
        }

        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - last_report_time.tv_sec)
                       + (now.tv_nsec - last_report_time.tv_nsec) / 1e9;
        if (elapsed < 0.1) return;  /* avoid division by near-zero */

        /* Print in FPS_N (x.xx) format matching nvdslogger style */
        double total_fps = 0.0;
        g_print("**BRANCH_PERF (%.1fs):", elapsed);
        for (auto &kv : frame_counts) {
            double fps = kv.second / elapsed;
            total_fps += fps;
            g_print("\tFPS_%d (%.2f)", kv.first, fps);
        }
        g_print("\n");
        g_print("[BranchFPS] %d sources | aggregate=%.1f FPS | interval=%.1fs\n",
                (int)frame_counts.size(), total_fps, elapsed);

        /* Reset for next interval */
        frame_counts.clear();
        last_report_time = now;
    }
};

static BranchFpsTracker g_branch_fps;

/* ═══════════════════════════════════════════════════════════════════════════
 * Source Transform Injection — injects nvvideoconvert + capsfilter +
 * videorate + capsfilter between each nvurisrcbin output and the internal
 * nvstreammux inside nvmultiurisrcbin.
 *
 * Per NVIDIA guidance (forums.developer.nvidia.com/t/307167/3):
 *   nvmultiurisrcbin is a GstBin containing nvurisrcbin + nvstreammux.
 *   Use gst_bin_iterate_elements to find each src bin, then attach elements.
 *   Ref: set_nvuribin_conv_prop in gst-nvmultiurisrcbincreator.cpp
 *
 * Injected chain (all optional — skipped if field is 0/empty):
 *   nvurisrcbin.src → [nvvideoconvert → capsfilter(format,W,H)]
 *                    → [videorate → capsfilter(framerate)]
 *                    → internal_mux.sink_N
 *
 * For dynamic streams added via REST API, "deep-element-added" signal
 * detects new nvurisrcbin elements and triggers a deferred re-scan.
 * ═══════════════════════════════════════════════════════════════════════════ */
struct SourceTransformConfig {
    bool enable = false;

    /* Resolution — 0 means "pass through, don't force" */
    int width = 0;
    int height = 0;

    /* Image format — empty means "pass through"
     * Supported: RGBA, NV12, I420, BGRx, GRAY8 */
    std::string format;

    /* nvvideoconvert hw accel: 0=default, 1=GPU, 2=VIC (Jetson) */
    int compute_hw = 1;
    int gpu_id = 0;

    /* FPS control — 0 means "don't inject videorate" */
    int target_fps = 0;
    bool drop_only = true;   /* Only drop frames, never duplicate */
    int max_rate = 0;        /* 0 = same as target_fps */

    /* Derived convenience flags */
    bool needs_vidconv() const {
        return (width > 0 && height > 0) || !format.empty();
    }
    bool needs_videorate() const {
        return target_fps > 0;
    }
};

static SourceTransformConfig g_src_transform;

struct SourceTransformCtx {
    std::mutex mtx;
    std::set<std::string> injected_pads;   /* mux sink pad names already injected */
    GstElement *source_bin = nullptr;      /* the nvmultiurisrcbin GstBin */
    GstElement *internal_mux = nullptr;    /* the internal nvstreammux (cached) */
    int injection_counter = 0;             /* unique names for injected elements */
};

static SourceTransformCtx g_st_ctx;

/* ── find_internal_nvstreammux ──
 * Recursively iterate a GstBin to find the first element whose factory
 * name is "nvstreammux". Returns ref'd element (caller must unref). */
static GstElement* find_internal_nvstreammux(GstBin *bin) {
    GstIterator *it = gst_bin_iterate_recurse(bin);
    GValue item = G_VALUE_INIT;
    GstElement *found = nullptr;

    while (gst_iterator_next(it, &item) == GST_ITERATOR_OK) {
        GstElement *elem = GST_ELEMENT(g_value_get_object(&item));
        GstElementFactory *factory = gst_element_get_factory(elem);
        if (factory) {
            const gchar *fname = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
            if (fname && g_strcmp0(fname, "nvstreammux") == 0) {
                found = (GstElement *)gst_object_ref(elem);
                g_value_unset(&item);
                break;
            }
        }
        g_value_unset(&item);
    }
    gst_iterator_free(it);
    return found;
}

/* ── BlockProbeInjectCtx ──
 * Passed to the blocking probe that does the actual unlink + inject + relink.
 * The probe fires on the nvurisrcbin's ghost src pad, blocking data flow
 * so we can safely splice the transform chain into the path. */
struct BlockProbeInjectCtx {
    GstPad *upstream_src;    /* nvurisrcbin ghost src (blocked pad) */
    GstPad *mux_sink;        /* internal nvstreammux sink_N */
    GstElement *parent_bin;  /* the nvmultiurisrcbin — we add elements here */
    int inject_id;           /* unique ID for naming elements */
};

static GstPadProbeReturn
source_transform_block_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    BlockProbeInjectCtx *ctx = (BlockProbeInjectCtx *)user_data;
    const SourceTransformConfig &cfg = g_src_transform;

    g_print("[SourceTransform] Blocked pad '%s', injecting transform chain (id=%d)...\n",
            GST_PAD_NAME(pad), ctx->inject_id);

    /* 1. Unlink: nvurisrcbin.src -X-> mux.sink_N */
    gst_pad_unlink(ctx->upstream_src, ctx->mux_sink);

    /* ── Collect all elements to inject ──
     * Chain order: nvvideoconvert → format_caps → videorate → fps_caps
     * Each segment is optional. */
    std::vector<GstElement*> chain_elements;
    char name_buf[64];

    /* ── Segment A: nvvideoconvert + format/resolution capsfilter ── */
    GstElement *vidconv = nullptr;
    GstElement *fmt_caps = nullptr;

    if (cfg.needs_vidconv()) {
        snprintf(name_buf, sizeof(name_buf), "src_nvvidconv_%d", ctx->inject_id);
        vidconv = gst_element_factory_make("nvvideoconvert", name_buf);
        if (!vidconv) {
            g_printerr("[SourceTransform] ERROR: Failed to create nvvideoconvert (id=%d)\n",
                        ctx->inject_id);
            gst_pad_link(ctx->upstream_src, ctx->mux_sink);
            delete ctx;
            return GST_PAD_PROBE_REMOVE;
        }
        g_object_set(G_OBJECT(vidconv),
            "compute-hw", cfg.compute_hw,
            "gpu-id",     cfg.gpu_id,
            NULL);
        chain_elements.push_back(vidconv);

        /* Build caps string for format + resolution */
        snprintf(name_buf, sizeof(name_buf), "src_fmt_caps_%d", ctx->inject_id);
        fmt_caps = gst_element_factory_make("capsfilter", name_buf);
        if (!fmt_caps) {
            g_printerr("[SourceTransform] ERROR: Failed to create format capsfilter (id=%d)\n",
                        ctx->inject_id);
            gst_pad_link(ctx->upstream_src, ctx->mux_sink);
            delete ctx;
            return GST_PAD_PROBE_REMOVE;
        }

        std::string caps_str = "video/x-raw(memory:NVMM)";
        if (!cfg.format.empty()) {
            caps_str += ", format=(string)" + cfg.format;
        }
        if (cfg.width > 0 && cfg.height > 0) {
            caps_str += ", width=(int)" + std::to_string(cfg.width)
                      + ", height=(int)" + std::to_string(cfg.height);
        }

        GstCaps *caps = gst_caps_from_string(caps_str.c_str());
        g_object_set(G_OBJECT(fmt_caps), "caps", caps, NULL);
        gst_caps_unref(caps);
        chain_elements.push_back(fmt_caps);

        g_print("[SourceTransform]   nvvideoconvert → capsfilter(%s) (id=%d)\n",
                caps_str.c_str(), ctx->inject_id);
    }

    /* ── Segment B: videorate + FPS capsfilter ── */
    GstElement *videorate = nullptr;
    GstElement *fps_caps = nullptr;

    if (cfg.needs_videorate()) {
        snprintf(name_buf, sizeof(name_buf), "src_videorate_%d", ctx->inject_id);
        videorate = gst_element_factory_make("videorate", name_buf);
        if (!videorate) {
            g_printerr("[SourceTransform] ERROR: Failed to create videorate (id=%d)\n",
                        ctx->inject_id);
            gst_pad_link(ctx->upstream_src, ctx->mux_sink);
            delete ctx;
            return GST_PAD_PROBE_REMOVE;
        }
        g_object_set(G_OBJECT(videorate),
            "drop-only", (gboolean)cfg.drop_only,
            "max-rate",  (gint)(cfg.max_rate > 0 ? cfg.max_rate : cfg.target_fps),
            NULL);
        chain_elements.push_back(videorate);

        snprintf(name_buf, sizeof(name_buf), "src_fps_caps_%d", ctx->inject_id);
        fps_caps = gst_element_factory_make("capsfilter", name_buf);
        if (!fps_caps) {
            g_printerr("[SourceTransform] ERROR: Failed to create fps capsfilter (id=%d)\n",
                        ctx->inject_id);
            gst_pad_link(ctx->upstream_src, ctx->mux_sink);
            delete ctx;
            return GST_PAD_PROBE_REMOVE;
        }

        char fps_caps_str[128];
        snprintf(fps_caps_str, sizeof(fps_caps_str),
                 "video/x-raw(memory:NVMM), framerate=%d/1", cfg.target_fps);
        GstCaps *fcaps = gst_caps_from_string(fps_caps_str);
        g_object_set(G_OBJECT(fps_caps), "caps", fcaps, NULL);
        gst_caps_unref(fcaps);
        chain_elements.push_back(fps_caps);

        g_print("[SourceTransform]   videorate(drop-only=%d) → capsfilter(%s) (id=%d)\n",
                cfg.drop_only, fps_caps_str, ctx->inject_id);
    }

    if (chain_elements.empty()) {
        g_print("[SourceTransform] Nothing to inject (id=%d), re-linking original\n",
                ctx->inject_id);
        gst_pad_link(ctx->upstream_src, ctx->mux_sink);
        delete ctx;
        return GST_PAD_PROBE_REMOVE;
    }

    /* 2. Add all elements to the parent bin and sync state */
    for (auto *elem : chain_elements) {
        gst_bin_add(GST_BIN(ctx->parent_bin), elem);
        gst_element_sync_state_with_parent(elem);
    }

    /* 3. Link the chain: elem[0] → elem[1] → ... → elem[N-1] */
    for (size_t i = 0; i + 1 < chain_elements.size(); i++) {
        if (!gst_element_link(chain_elements[i], chain_elements[i + 1])) {
            g_printerr("[SourceTransform] ERROR: chain link failed at index %d (id=%d)\n",
                        (int)i, ctx->inject_id);
            gst_pad_link(ctx->upstream_src, ctx->mux_sink);
            delete ctx;
            return GST_PAD_PROBE_REMOVE;
        }
    }

    /* 4. Link: upstream_src → chain[0].sink */
    GstPad *first_sink = gst_element_get_static_pad(chain_elements.front(), "sink");
    GstPadLinkReturn lr1 = gst_pad_link(ctx->upstream_src, first_sink);
    gst_object_unref(first_sink);

    /* 5. Link: chain[N-1].src → mux_sink */
    GstPad *last_src = gst_element_get_static_pad(chain_elements.back(), "src");
    GstPadLinkReturn lr2 = gst_pad_link(last_src, ctx->mux_sink);
    gst_object_unref(last_src);

    if (lr1 == GST_PAD_LINK_OK && lr2 == GST_PAD_LINK_OK) {
        g_print("[SourceTransform] SUCCESS: injected %d element(s) on mux pad '%s' (id=%d)\n",
                (int)chain_elements.size(), GST_PAD_NAME(ctx->mux_sink), ctx->inject_id);
    } else {
        g_printerr("[SourceTransform] ERROR: re-link failed lr1=%d lr2=%d (id=%d)\n",
                    lr1, lr2, ctx->inject_id);
    }

    delete ctx;
    return GST_PAD_PROBE_REMOVE;  /* Unblock — data flows through new path */
}

/* ── source_transform_scan_and_inject ──
 * Scans the internal nvstreammux inside the source bin.
 * For each linked sink pad that hasn't been injected yet,
 * installs a blocking probe that does the splicing.
 * Called from main loop thread (g_idle_add / g_timeout_add). */
static gboolean source_transform_scan_and_inject(gpointer user_data) {
    if (!g_src_transform.enable) return G_SOURCE_REMOVE;

    std::lock_guard<std::mutex> lock(g_st_ctx.mtx);

    /* Find the internal nvstreammux if not cached */
    if (!g_st_ctx.internal_mux && g_st_ctx.source_bin) {
        g_st_ctx.internal_mux = find_internal_nvstreammux(
            GST_BIN(g_st_ctx.source_bin));
        if (g_st_ctx.internal_mux) {
            gchar *mux_name = gst_element_get_name(g_st_ctx.internal_mux);
            g_print("[SourceTransform] Found internal nvstreammux: '%s'\n", mux_name);
            g_free(mux_name);
        } else {
            g_print("[SourceTransform] Internal nvstreammux not found yet, will retry...\n");
            return G_SOURCE_REMOVE;
        }
    }

    if (!g_st_ctx.internal_mux) return G_SOURCE_REMOVE;

    /* Iterate all sink pads on the internal mux */
    GstIterator *pad_it = gst_element_iterate_sink_pads(g_st_ctx.internal_mux);
    GValue pad_val = G_VALUE_INIT;
    int injected_count = 0;

    while (gst_iterator_next(pad_it, &pad_val) == GST_ITERATOR_OK) {
        GstPad *mux_sink = GST_PAD(g_value_get_object(&pad_val));
        const gchar *pad_name = GST_PAD_NAME(mux_sink);

        /* Skip already-injected pads */
        if (g_st_ctx.injected_pads.count(pad_name)) {
            g_value_unset(&pad_val);
            continue;
        }

        /* Check if this pad is linked */
        GstPad *peer = gst_pad_get_peer(mux_sink);
        if (!peer) {
            g_value_unset(&pad_val);
            continue;
        }

        /* Check: is there already an injected element upstream?
         * (Avoid double-injection if scan is called multiple times) */
        GstElement *peer_parent = gst_pad_get_parent_element(peer);
        if (peer_parent) {
            GstElementFactory *pf = gst_element_get_factory(peer_parent);
            if (pf) {
                const gchar *pfn = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(pf));
                if (pfn && (g_strcmp0(pfn, "capsfilter") == 0 ||
                            g_strcmp0(pfn, "nvvideoconvert") == 0)) {
                    g_print("[SourceTransform] Pad '%s' appears already injected, skipping\n",
                            pad_name);
                    g_st_ctx.injected_pads.insert(pad_name);
                    gst_object_unref(peer_parent);
                    gst_object_unref(peer);
                    g_value_unset(&pad_val);
                    continue;
                }
            }
            gst_object_unref(peer_parent);
        }

        /* Determine the parent bin to add elements into.
         * We walk up from the internal mux to find the nvmultiurisrcbin. */
        GstElement *parent_bin = GST_ELEMENT(gst_element_get_parent(g_st_ctx.internal_mux));
        if (!parent_bin) {
            g_printerr("[SourceTransform] ERROR: Cannot find parent bin of internal mux\n");
            gst_object_unref(peer);
            g_value_unset(&pad_val);
            continue;
        }

        /* Install blocking probe on the upstream src pad */
        BlockProbeInjectCtx *bctx = new BlockProbeInjectCtx();
        bctx->upstream_src = peer;         /* takes ownership of ref */
        bctx->mux_sink = (GstPad *)gst_object_ref(mux_sink);
        bctx->parent_bin = parent_bin;     /* takes ownership of ref */
        bctx->inject_id = g_st_ctx.injection_counter++;

        gst_pad_add_probe(peer, GST_PAD_PROBE_TYPE_BLOCK_DOWNSTREAM,
                          source_transform_block_probe_cb, bctx, NULL);

        g_st_ctx.injected_pads.insert(pad_name);
        injected_count++;

        g_print("[SourceTransform] Scheduled injection on mux pad '%s' (id=%d)\n",
                pad_name, bctx->inject_id);

        g_value_unset(&pad_val);
    }
    gst_iterator_free(pad_it);

    if (injected_count > 0) {
        g_print("[SourceTransform] Scheduled %d injection(s) this scan\n", injected_count);
    }

    return G_SOURCE_REMOVE;
}

/* ── source_transform_deep_element_added_cb ──
 * Connected on the source bin. When nvmultiurisrcbin internally creates
 * a new nvurisrcbin (dynamic stream via REST API), this fires.
 * We schedule a deferred re-scan to inject the transform chain on the
 * new connection. */
static void
source_transform_deep_element_added_cb(GstBin *bin, GstBin *sub_bin,
                                       GstElement *element, gpointer user_data)
{
    if (!g_src_transform.enable) return;

    GstElementFactory *factory = gst_element_get_factory(element);
    if (!factory) return;

    const gchar *fname = gst_plugin_feature_get_name(GST_PLUGIN_FEATURE(factory));
    if (!fname) return;

    /* Detect nvurisrcbin additions → new stream added */
    if (g_str_has_prefix(fname, "nvurisrcbin")) {
        gchar *elem_name = gst_element_get_name(element);
        g_print("[SourceTransform] Detected new nvurisrcbin: '%s', "
                "scheduling deferred transform injection...\n", elem_name);
        g_free(elem_name);

        /* Defer by 500ms to allow internal linking to complete.
         * nvmultiurisrcbin links nvurisrcbin→nvstreammux asynchronously. */
        g_timeout_add(500, source_transform_scan_and_inject, NULL);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ParallelBranch — one inference branch with its own demux + streammux
 * v12: Each branch has its own nvstreamdemux (tee copies to all branches)
 * ═══════════════════════════════════════════════════════════════════════════ */
struct ParallelBranch {
    int index;
    std::string pgie_key, tracker_key, sgie_key, analytics_key;
    bool enable_tracker = false, enable_sgie = false, enable_analytics = false;
    GstElement *pgie = nullptr;
    GstElement *tracker = nullptr;
    GstElement *sgie = nullptr;
    GstElement *analytics = nullptr;

    /* Per-branch nvstreammux (re-batches for this branch's model) */
    GstElement *branch_mux = nullptr;
};

/* ═══════════════════════════════════════════════════════════════════════════
 * DemuxProbeCtx — passed to the tee sink pad probe
 *
 * v12: Installed on TEE's sink pad. When a new source_id is detected,
 * schedules deferred linking on ALL branches' demuxes.
 * ═══════════════════════════════════════════════════════════════════════════ */
struct DemuxProbeCtx {
    GstElement *pipeline;
    GstElement *stream_demux;
    std::vector<ParallelBranch> *branches;
    std::mutex mtx;
    std::set<int> linked_source_ids;  // source_ids already linked
    std::map<int, int> frame_counters; // per-source heartbeat counter

    // Per source tees: source_id -> GstElement tee 
    std::map<int, GstElement*> source_tees;
};

GMainLoop *global_loop = NULL;
GstElement *global_pipeline = NULL;

static GstPadProbeReturn
branch_fps_probe(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        if (!frame_meta) continue;

        /* source_id is already correct (preserved by sink pad naming) */
        g_branch_fps.tick(frame_meta->source_id);
    }

    return GST_PAD_PROBE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════════════════ */
static GstElement* create_queue(const char* name_suffix, bool leaky = false) {
    static int queue_cnt = 0;
    char name[64];
    snprintf(name, sizeof(name), "queue-%s-%d", name_suffix, queue_cnt++);
    GstElement *q = gst_element_factory_make("queue", name);
    if (!q) return NULL;
    g_object_set(G_OBJECT(q),
        "max-size-buffers", (guint)10,
        "max-size-bytes",   (guint)0,
        "max-size-time",    (guint64)0, NULL);
    if (leaky) {
        g_object_set(G_OBJECT(q), "leaky", 2, "max-size-buffers", (guint)10, NULL);
    }
    return q;
}

static gboolean link_to_metamux(GstElement *metamux, GstElement *src_elem, gint index) {
    gchar pad_name[16];
    g_snprintf(pad_name, 16, "sink_%u", index);
    GstPad *mux_pad = gst_element_request_pad_simple(metamux, pad_name);
    GstPad *src_pad = gst_element_get_static_pad(src_elem, "src");
    if (!mux_pad || !src_pad) {
        g_printerr("link_to_metamux: pad error at sink_%u\n", index);
        if (mux_pad) gst_object_unref(mux_pad);
        if (src_pad) gst_object_unref(src_pad);
        return FALSE;
    }
    gboolean ret = (gst_pad_link(src_pad, mux_pad) == GST_PAD_LINK_OK);
    if (!ret) g_printerr("link_to_metamux: link failed at sink_%u\n", index);
    gst_object_unref(mux_pad);
    gst_object_unref(src_pad);
    return ret;
}

static gboolean link_from_tee(GstElement *tee, GstElement *sink_elem) {
    GstPad *tee_pad = gst_element_request_pad_simple(tee, "src_%u");
    GstPad *sink_pad = gst_element_get_static_pad(sink_elem, "sink");
    if (!tee_pad || !sink_pad) {
        g_printerr("link_from_tee: pad error\n");
        if (tee_pad) gst_object_unref(tee_pad);
        if (sink_pad) gst_object_unref(sink_pad);
        return FALSE;
    }
    gboolean ret = (gst_pad_link(tee_pad, sink_pad) == GST_PAD_LINK_OK);
    if (!ret) g_printerr("link_from_tee: link failed\n");
    gst_object_unref(tee_pad);
    gst_object_unref(sink_pad);
    return ret;
}

static gboolean bus_call(GstBus *bus, GstMessage *msg, gpointer data) {
    GMainLoop *loop = (GMainLoop *)data;
    switch (GST_MESSAGE_TYPE(msg)) {
        case GST_MESSAGE_EOS:
            g_print("End of stream\n");
            g_main_loop_quit(loop);
            break;
        case GST_MESSAGE_WARNING: {
            gchar *debug = NULL; GError *error = NULL;
            gst_message_parse_warning(msg, &error, &debug);
            g_printerr("WARNING from element %s: %s\n", GST_OBJECT_NAME(msg->src), error->message);
            if (debug) g_printerr("Warning details: %s\n", debug);
            g_free(debug); g_error_free(error);
            break;
        }
        case GST_MESSAGE_ERROR: {
            gchar *debug = NULL; GError *error = NULL;
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

static gboolean diagnostic_timer_cb(gpointer user_data) {
    g_demux_tracker.print_status();
    return TRUE;
}

static gboolean fps_report_timer_cb(gpointer user_data) {
    g_branch_fps.report_and_reset();
    return TRUE;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * do_deferred_link — v12-fixed (NVIDIA reference pattern)
 *
 * For a new source_id N, creates:
 *   demux.src_N → source_tee[N] → queue → branch0_mux.sink_N
 *                                → queue → branch1_mux.sink_N
 *                                → ... (for all branches)
 *
 * Using sink_<source_id> naming on branch_mux preserves source_ids.
 * Deferred to main loop thread to avoid STATE_LOCK deadlock (from v11).
 * ═══════════════════════════════════════════════════════════════════════════ */
struct DeferredLinkInfo {
    DemuxProbeCtx *ctx;
    int source_id;
};

static gboolean do_deferred_link(gpointer user_data) {
    DeferredLinkInfo *info = (DeferredLinkInfo *)user_data;
    DemuxProbeCtx *ctx = info->ctx;
    int source_id = info->source_id;
    int num_branches = (int)ctx->branches->size();

    g_print("[DeferredLink] Linking source_id=%d: demux → source_tee → %d branch muxes (main loop)\n",
            source_id, num_branches);

    /* --- Step 1: Get the pre-created demux src pad --- */
    char pad_name[32];
    snprintf(pad_name, sizeof(pad_name), "src_%d", source_id);
    GstPad *demux_src = gst_element_get_static_pad(ctx->stream_demux, pad_name);
    if (!demux_src) {
        demux_src = gst_element_request_pad_simple(ctx->stream_demux, pad_name);
    }
    if (!demux_src) {
        g_printerr("[DeferredLink] ERROR: Cannot get demux pad %s\n", pad_name);
        std::lock_guard<std::mutex> lock(ctx->mtx);
        ctx->linked_source_ids.erase(source_id);
        delete info;
        return G_SOURCE_REMOVE;
    }
    if (gst_pad_is_linked(demux_src)) {
        g_print("[DeferredLink] demux.%s already linked, skipping\n", pad_name);
        gst_object_unref(demux_src);
        delete info;
        return G_SOURCE_REMOVE;
    }

    /* --- Step 2: Create per-source tee (following NVIDIA reference) --- */
    char tee_name[64];
    snprintf(tee_name, sizeof(tee_name), "source_tee_%d", source_id);
    GstElement *source_tee = gst_element_factory_make("tee", tee_name);
    if (!source_tee) {
        g_printerr("[DeferredLink] ERROR: Failed to create source_tee for source_id=%d\n", source_id);
        gst_object_unref(demux_src);
        std::lock_guard<std::mutex> lock(ctx->mtx);
        ctx->linked_source_ids.erase(source_id);
        delete info;
        return G_SOURCE_REMOVE;
    }
    gst_bin_add(GST_BIN(ctx->pipeline), source_tee);
    gst_element_sync_state_with_parent(source_tee);

    /* --- Step 3: Link demux.src_N → source_tee --- */
    GstPad *tee_sink = gst_element_get_static_pad(source_tee, "sink");
    GstPadLinkReturn link_ret = gst_pad_link(demux_src, tee_sink);
    gst_object_unref(tee_sink);
    gst_object_unref(demux_src);

    if (link_ret != GST_PAD_LINK_OK) {
        g_printerr("[DeferredLink] ERROR: demux.%s → source_tee link failed (ret=%d)\n",
                    pad_name, link_ret);
        std::lock_guard<std::mutex> lock(ctx->mtx);
        ctx->linked_source_ids.erase(source_id);
        delete info;
        return G_SOURCE_REMOVE;
    }

    /* Store the source_tee for potential future use (stream removal etc.) */
    {
        std::lock_guard<std::mutex> lock(ctx->mtx);
        ctx->source_tees[source_id] = source_tee;
    }

    /* --- Step 4: Fan out source_tee → queue → branch_mux[i].sink_N for ALL branches --- */
    int branches_linked = 0;
    for (int i = 0; i < num_branches; i++) {
        ParallelBranch &br = (*ctx->branches)[i];

        /* Create queue: source_tee → queue → branch_mux.sink_N */
        char q_name[64];
        snprintf(q_name, sizeof(q_name), "src%d-to-br%d", source_id, i);
        GstElement *q = create_queue(q_name, true);
        g_object_set(G_OBJECT(q), "max-size-buffers", (guint)5, NULL);
        if (!q) {
            g_printerr("[DeferredLink] ERROR: Failed to create queue src%d→br%d\n", source_id, i);
            continue;
        }
        gst_bin_add(GST_BIN(ctx->pipeline), q);
        gst_element_sync_state_with_parent(q);

        /* Link: source_tee → queue */
        if (!link_from_tee(source_tee, q)) {
            g_printerr("[DeferredLink] ERROR: source_tee_%d → queue link failed for branch %d\n",
                        source_id, i);
            continue;
        }

        /* Link: queue → branch_mux.sink_<source_id>
         * Using source_id as sink pad name preserves the ID through nvstreammux */
        char sink_name[32];
        snprintf(sink_name, sizeof(sink_name), "sink_%d", source_id);
        GstPad *mux_sink = gst_element_request_pad_simple(br.branch_mux, sink_name);
        GstPad *q_src = gst_element_get_static_pad(q, "src");

        if (!mux_sink || gst_pad_link(q_src, mux_sink) != GST_PAD_LINK_OK) {
            g_printerr("[DeferredLink] ERROR: queue → branch%d_mux.%s link failed\n", i, sink_name);
            if (mux_sink) gst_object_unref(mux_sink);
            gst_object_unref(q_src);
            continue;
        }

        g_print("[DeferredLink] SUCCESS: demux.%s → source_tee_%d → queue → branch%d_mux.%s\n",
                pad_name, source_id, i, sink_name);

        gst_object_unref(mux_sink);
        gst_object_unref(q_src);
        branches_linked++;
    }

    if (branches_linked == num_branches) {
        g_demux_tracker.on_stream_added(source_id);
        g_print("[DeferredLink] source_id=%d linked to all %d branches\n", source_id, num_branches);
    } else if (branches_linked > 0) {
        g_demux_tracker.on_stream_added(source_id);
        g_printerr("[DeferredLink] WARNING: source_id=%d only linked to %d/%d branches\n",
                    source_id, branches_linked, num_branches);
    } else {
        /* Total failure — unmark so probe can retry */
        std::lock_guard<std::mutex> lock(ctx->mtx);
        ctx->linked_source_ids.erase(source_id);
        g_printerr("[DeferredLink] ERROR: source_id=%d failed to link to any branch\n", source_id);
    }
    if (branches_linked > 0) {
    char dot_name[128];
    snprintf(dot_name, sizeof(dot_name),
             "deepstream-after-source-%d", source_id);
    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(ctx->pipeline),
        GST_DEBUG_GRAPH_SHOW_ALL, dot_name);
    g_print("[DOT] Dumped pipeline graph: ./tmp/%s.dot\n", dot_name);
}

    delete info;
    return G_SOURCE_REMOVE;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * demux_sink_probe_cb — installed on the SINGLE nvstreamdemux's sink pad
 *
 * Detects new source_ids, schedules deferred linking via g_idle_add.
 * v12-fixed: No branch routing — every source goes to ALL branches.
 * ═══════════════════════════════════════════════════════════════════════════ */
static GstPadProbeReturn
demux_sink_probe_cb(GstPad *pad, GstPadProbeInfo *info, gpointer user_data)
{
    DemuxProbeCtx *ctx = (DemuxProbeCtx *)user_data;

    GstBuffer *buf = GST_PAD_PROBE_INFO_BUFFER(info);
    if (!buf) return GST_PAD_PROBE_OK;

    NvDsBatchMeta *batch_meta = gst_buffer_get_nvds_batch_meta(buf);
    if (!batch_meta) return GST_PAD_PROBE_OK;

    for (NvDsMetaList *l_frame = batch_meta->frame_meta_list;
         l_frame != NULL; l_frame = l_frame->next)
    {
        NvDsFrameMeta *frame_meta = (NvDsFrameMeta *)l_frame->data;
        if (!frame_meta) continue;

        int source_id = frame_meta->source_id;

        {
            std::lock_guard<std::mutex> lock(ctx->mtx);

            if (ctx->linked_source_ids.count(source_id)) {
                ctx->frame_counters[source_id]++;
                if (ctx->frame_counters[source_id] % 900 == 0) {
                    g_print("[DemuxProbe] heartbeat source_id=%d frame=%d\n",
                            source_id, frame_meta->frame_num);
                }
                continue;
            }
            ctx->linked_source_ids.insert(source_id);
        }

        g_print("[DemuxProbe] New source_id=%d detected, scheduling link on all branches...\n",
                source_id);

        DeferredLinkInfo *link_info = new DeferredLinkInfo();
        link_info->ctx = ctx;
        link_info->source_id = source_id;
        g_idle_add(do_deferred_link, link_info);
    }

    return GST_PAD_PROBE_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char *argv[]) {
    AppCtx appctx = {0};
    GMainLoop *loop = NULL;
    GstBus *bus = NULL;
    guint bus_watch_id = 0;

    NvDsGieType pgie_type = NVDS_GIE_PLUGIN_INFER;

    GstElement *pgie = NULL;
    GstElement *tracker = NULL, *sgie1 = NULL, *sgie2 = NULL, *sgie3 = NULL, *sgie4 = NULL, *sgie5 = NULL;
    GstElement *payloader = NULL, *qtmux = NULL, *caps_filter = NULL;
    GstRTSPServer *rtsp_server = NULL;

    /* Parallel PGIE support */
    bool parallel_mode = false;
    std::vector<ParallelBranch> parallel_branches;
    GstElement *metamux = nullptr;
    GstElement *pipeline_tee = nullptr;  /* v12: tee copies batch to all branches */
    DemuxProbeCtx *probe_ctx = nullptr;
    GstElement *stream_demux = nullptr;  /* v12: one demux per branch removed, single demux added */

    /* Branch streammux config */
    int branch_mux_width = 1920;
    int branch_mux_height = 1080;
    int branch_mux_batch_timeout = 40000;
    int branch_mux_batch_size = 8;  /* v10: realistic default, overridden by YAML */

    /* Message broker elements */
    GstElement *tee = NULL;
    GstElement *msgconv = NULL, *msgbroker = NULL;

    /* Feature Flags (Default False) */
    bool enable_pgie = false;
    bool enable_tracker = false;
    bool enable_sgie1 = false, enable_sgie2 = false, enable_sgie3 = false, enable_sgie4 = false, enable_sgie5 = false;
    bool enable_preprocess = false;
    bool enable_tiler = false, enable_osd = false, enable_analytics = false;
    bool enable_msgbroker = false;

    if (argc < 2) {
        g_printerr("Usage: %s <yml file>\n", argv[0]);
        return -1;
    }

    std::string msgbroker_proto_lib;
    std::string msgbroker_conn_str;
    std::string msgbroker_topic;
    std::string msgbroker_cfg_file;
    std::string msgconv_config;
    std::string msgconv_msg2p_newapi;
    int msgconv_payload_type = 0;
    int msgbroker_comp_id = 0;

    int sink_type = 0;
    int sync = 0;
    int row = 1, col = 1;
    std::string output_file = "output.mkv";
    int rtsp_port_num = 8554;
    int udp_port_num = 5400;
    NvDsYamlCodecStatus codec_status = {0};

    /* ─── Parse YAML ─── */
    try {
        YAML::Node config = YAML::LoadFile(argv[1]);

        if (config["demux"]) {
            if (config["demux"]["max-pads"]) max_demux_pads = config["demux"]["max-pads"].as<int>();
        }

        if (config["sink"]) {
            if (config["sink"]["type"]) sink_type = config["sink"]["type"].as<int>();
            if (config["sink"]["sync"]) sync = config["sink"]["sync"].as<int>();
            if (config["sink"]["output-file"]) output_file = config["sink"]["output-file"].as<std::string>();
            if (config["sink"]["rtsp-port"]) rtsp_port_num = config["sink"]["rtsp-port"].as<int>();
            if (config["sink"]["udp-port"]) udp_port_num = config["sink"]["udp-port"].as<int>();
        }
        if (config["tiler"] && config["tiler"]["enable"]) {
            enable_tiler = config["tiler"]["enable"].as<int>() == 1;
            if (config["tiler"]["rows"]) row = config["tiler"]["rows"].as<int>();
            if (config["tiler"]["columns"]) col = config["tiler"]["columns"].as<int>();
        }
        if (config["primary-gie"] && config["primary-gie"]["enable"]) enable_pgie = config["primary-gie"]["enable"].as<int>() == 1;
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

        /* Message broker config */
        if (enable_msgbroker && config["msgbroker"]) {
            auto mb = config["msgbroker"];
            if (mb["proto-lib"])  msgbroker_proto_lib = mb["proto-lib"].as<std::string>();
            if (mb["conn-str"])   msgbroker_conn_str  = mb["conn-str"].as<std::string>();
            if (mb["topic"])      msgbroker_topic     = mb["topic"].as<std::string>();
            if (mb["config"])     msgbroker_cfg_file  = mb["config"].as<std::string>();
            if (mb["comp-id"])    msgbroker_comp_id   = mb["comp-id"].as<int>();
        }
        if (config["msgconv"]) {
            auto mc = config["msgconv"];
            if (mc["payload-type"]) msgconv_payload_type = mc["payload-type"].as<int>();
            if (mc["config"])       msgconv_config       = mc["config"].as<std::string>();
            if (mc["msg2p-newapi"]) msgconv_msg2p_newapi = mc["msg2p-newapi"].as<std::string>();
        }

        /* Branch streammux config */
        if (config["branch-streammux"]) {
            auto bm = config["branch-streammux"];
            if (bm["width"])                branch_mux_width         = bm["width"].as<int>();
            if (bm["height"])               branch_mux_height        = bm["height"].as<int>();
            if (bm["batched-push-timeout"]) branch_mux_batch_timeout = bm["batched-push-timeout"].as<int>();
            if (bm["batch-size"])           branch_mux_batch_size    = bm["batch-size"].as<int>();
        }

        /* Source Transform config — per-source resolution, format, FPS */
        if (config["source-transform"]) {
            auto st = config["source-transform"];
            if (st["enable"])     g_src_transform.enable     = st["enable"].as<int>() == 1;
            if (st["width"])      g_src_transform.width      = st["width"].as<int>();
            if (st["height"])     g_src_transform.height     = st["height"].as<int>();
            if (st["format"])     g_src_transform.format     = st["format"].as<std::string>();
            if (st["compute-hw"]) g_src_transform.compute_hw = st["compute-hw"].as<int>();
            if (st["gpu-id"])     g_src_transform.gpu_id     = st["gpu-id"].as<int>();
            if (st["target-fps"]) g_src_transform.target_fps = st["target-fps"].as<int>();
            if (st["drop-only"])  g_src_transform.drop_only  = st["drop-only"].as<int>() == 1;
            if (st["max-rate"])   g_src_transform.max_rate   = st["max-rate"].as<int>();
        }

        nvds_parse_codec_status(argv[1], "encoder", &codec_status);

        /* ===== AUTO-DETECT PARALLEL PGIES ===== */
        for (int i = 0; i < 10; i++) {
            std::string key = "primary-gie" + std::to_string(i);
            if (!config[key]) break;
            if (config[key]["enable"] && config[key]["enable"].as<int>() == 0)
                continue;

            ParallelBranch br;
            br.index = i;
            br.pgie_key = key;

            /* Per-branch tracker */
            std::string tk = "tracker" + std::to_string(i);
            if (config[tk] && config[tk]["enable"] && config[tk]["enable"].as<int>() == 1) {
                br.enable_tracker = true;
                br.tracker_key = tk;
            }

            /* Per-branch sgie */
            std::string sk = "secondary-gie" + std::to_string(i) + "-1";
            if (config[sk] && config[sk]["enable"] && config[sk]["enable"].as<int>() == 1) {
                br.enable_sgie = true;
                br.sgie_key = sk;
            }

            /* Per-branch analytics */
            std::string an = "analytics" + std::to_string(i);
            if (config[an] && config[an]["enable"] && config[an]["enable"].as<int>() == 1) {
                br.enable_analytics = true;
                br.analytics_key = an;
            }

            /* v12: Static source assignment removed — tee sends all streams to all branches */

            parallel_branches.push_back(br);
        }

        if (parallel_branches.size() >= 2) {
            parallel_mode = true;
            g_demux_tracker.max_pads = max_demux_pads;
            g_print("=== PARALLEL TEE MODE (v12): %d branches, demux max=%d per branch ===\n",
                    (int)parallel_branches.size(), max_demux_pads);
        }

        if (parallel_mode) {
            nvds_parse_gie_type(&pgie_type, argv[1], parallel_branches[0].pgie_key.c_str());
        } else {
            nvds_parse_gie_type(&pgie_type, argv[1], "primary-gie");
        }

    } catch (const YAML::Exception& e) {
        g_printerr("Error parsing YAML: %s\n", e.what());
        return -1;
    }

    /* ─── Logs ─── */
    g_print("=== SINK CONFIGURATION ===\n");
    g_print("Sink Type: %d (0=fakesink, 1=File, 2=RTSP, 3=Display)\n", sink_type);
    g_print("Output File: %s\n", output_file.c_str());
    g_print("Sync: %d (0 or 1)\n", sync);
    g_print("==========================\n\n");
    g_print("=== TILER CONFIGURATION ===\n");
    g_print("Tiler Config: %d x %d\n", row, col);
    g_print("==========================\n\n");
    if (parallel_mode) {
        g_print("=== BRANCH STREAMMUX CONFIG ===\n");
        g_print("Width: %d, Height: %d, BatchSize: %d, Timeout: %d\n",
                branch_mux_width, branch_mux_height, branch_mux_batch_size, branch_mux_batch_timeout);
        g_print("================================\n\n");
    }
    if (g_src_transform.enable) {
        g_print("=== SOURCE TRANSFORM CONFIGURATION ===\n");
        if (g_src_transform.needs_vidconv()) {
            g_print("  Resolution: %dx%d | Format: %s | GPU: %d | Compute-HW: %d\n",
                    g_src_transform.width, g_src_transform.height,
                    g_src_transform.format.empty() ? "(passthrough)" : g_src_transform.format.c_str(),
                    g_src_transform.gpu_id, g_src_transform.compute_hw);
        }
        if (g_src_transform.needs_videorate()) {
            g_print("  Target FPS: %d | Drop-Only: %d | Max-Rate: %d\n",
                    g_src_transform.target_fps, g_src_transform.drop_only,
                    g_src_transform.max_rate > 0 ? g_src_transform.max_rate : g_src_transform.target_fps);
        }
        g_print("======================================\n\n");
    }

    setenv("GST_DEBUG_DUMP_DOT_DIR", "./tmp", 1);

    gst_init(&argc, &argv);
    loop = g_main_loop_new(NULL, FALSE);
    global_loop = loop;
    signal(SIGINT, sigint_handler);

    /* ═══════════════════════════════════════════════════════════════════
     * REST SERVER
     * ═══════════════════════════════════════════════════════════════════ */
    gboolean rest_server_within_multiurisrcbin = FALSE;
    nvds_parse_check_rest_server_with_app(argv[1], "rest-server", &rest_server_within_multiurisrcbin);

    if (!rest_server_within_multiurisrcbin) {
        nvds_parse_server_appctx(argv[1], "server-app-ctx", &appctx);
        NvDsServerCallbacks server_cb = {};
        g_print("Setting rest server callbacks\n");

        server_cb.stream_cb = [&appctx](NvDsServerStreamInfo *stream_info, void *ctx) {
            if (stream_info && !stream_info->value_camera_id.empty()) {
                g_print("[StreamAPI] Camera '%s' → all branches (v12-fixed tee pattern)\n",
                        stream_info->value_camera_id.c_str());
            }
            s_stream_callback_impl(stream_info, (void*)&appctx);
        };

        server_cb.roi_cb = [&appctx, enable_analytics](NvDsServerRoiInfo *roi_info, void *ctx) {
            if (!enable_analytics) return;
            s_roi_callback_impl(roi_info, (void*)&appctx);
        };

        server_cb.dec_cb = [&appctx](NvDsServerDecInfo *dec_info, void *ctx) {
            s_dec_callback_impl(dec_info, (void*)&appctx);
        };

        if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER) {
            server_cb.inferserver_cb = [&appctx, parallel_mode](NvDsServerInferServerInfo *info, void *ctx) {
                if (parallel_mode) return;
                s_inferserver_callback_impl(info, (void*)&appctx);
            };
        } else {
            server_cb.infer_cb = [&appctx, parallel_mode](NvDsServerInferInfo *info, void *ctx) {
                if (parallel_mode) return;
                s_infer_callback_impl(info, (void*)&appctx);
            };
        }

        server_cb.enc_cb = [&appctx](NvDsServerEncInfo *enc_info, void *ctx) {
            s_enc_callback_impl(enc_info, (void*)&appctx);
        };

        server_cb.mux_cb = [&appctx](NvDsServerMuxInfo *mux_info, void *ctx) {
            s_mux_callback_impl(mux_info, (void*)&appctx);
        };

        server_cb.osd_cb = [&appctx, enable_osd](NvDsServerOsdInfo *osd_info, void *ctx) {
            if (!enable_osd) return;
            s_osd_callback_impl(osd_info, (void*)&appctx);
        };

        server_cb.appinstance_cb = [&appctx](NvDsServerAppInstanceInfo *appinstance_info, void *ctx) {
            s_appinstance_callback_impl(appinstance_info, (void*)&appctx);
        };

        appctx.server_conf.ip = appctx.httpIp;
        appctx.server_conf.port = appctx.httpPort;
        appctx.restServer = (void*)nvds_rest_server_start(&appctx.server_conf, &server_cb);
    }

    /* ═══════════════════════════════════════════════════════════════════
     * PIPELINE CONSTRUCTION
     * ═══════════════════════════════════════════════════════════════════ */
    appctx.pipeline = gst_pipeline_new("dsserver-pipeline");
    global_pipeline = appctx.pipeline;

    /* 1. Source Bin */
    if (appctx.restServer) {
        appctx.nvmultiurisrcbinCreator = gst_nvmultiurisrcbincreator_init(0, NVDS_MULTIURISRCBIN_MODE_VIDEO, &appctx.muxConfig);
        GstDsNvUriSrcConfig sourceConfig;
        memset(&sourceConfig, 0, sizeof(GstDsNvUriSrcConfig));
        sourceConfig.uri = appctx.uri_list;
        gst_nvmultiurisrcbincreator_add_source(appctx.nvmultiurisrcbinCreator, &sourceConfig);
        gst_bin_add(GST_BIN(appctx.pipeline), gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator));
    } else {
        appctx.multiuribin = gst_element_factory_make("nvmultiurisrcbin", "multiuribin");
        nvds_parse_multiurisrcbin(appctx.multiuribin, argv[1], "multiurisrcbin");
        gst_bin_add(GST_BIN(appctx.pipeline), appctx.multiuribin);
    }

    /* 1b. Source Transform injection setup — watch for internal nvurisrcbin additions */
    if (g_src_transform.enable) {
        GstElement *source_bin = appctx.restServer
            ? gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator)
            : appctx.multiuribin;

        g_st_ctx.source_bin = source_bin;
        g_signal_connect(GST_BIN(source_bin), "deep-element-added",
                         G_CALLBACK(source_transform_deep_element_added_cb), NULL);
        g_print("[SourceTransform] Watching source bin '%s' for dynamic stream additions\n",
                GST_ELEMENT_NAME(source_bin));
    }

    /* 2. Core Elements */
    if (!parallel_mode) {
        if (enable_pgie) {
        if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER)
            appctx.pgie = gst_element_factory_make("nvinferserver", "primary-nvinference-engine");
        else
            appctx.pgie = gst_element_factory_make("nvinfer", "primary-nvinference-engine");
        nvds_parse_gie(appctx.pgie, argv[1], "primary-gie");
    }}

    appctx.nvdslogger = gst_element_factory_make("nvdslogger", "nvdslogger");
    g_object_set(G_OBJECT(appctx.nvdslogger), "fps-measurement-interval-sec", 3, NULL);
    appctx.nvvidconv = gst_element_factory_make("nvvideoconvert", "nvvideo-converter");
    g_object_set(G_OBJECT(appctx.nvvidconv), "compute-hw", 1, NULL);

    /* 3. Conditional Elements */
    if (enable_preprocess) {
        appctx.preprocess = gst_element_factory_make("nvdspreprocess", "preprocess-plugin");
        gst_bin_add(GST_BIN(appctx.pipeline), appctx.preprocess);
    }
    if (!parallel_mode) {
        if (enable_tracker) {
            tracker = gst_element_factory_make("nvtracker", "tracker");
            nvds_parse_tracker(tracker, argv[1], "tracker");
            gst_bin_add(GST_BIN(appctx.pipeline), tracker);
        }
        if (enable_sgie1) { sgie1 = gst_element_factory_make("nvinferserver", "sgie1"); nvds_parse_gie(sgie1, argv[1], "secondary-gie1"); gst_bin_add(GST_BIN(appctx.pipeline), sgie1); }
        if (enable_sgie2) { sgie2 = gst_element_factory_make("nvinferserver", "sgie2"); nvds_parse_gie(sgie2, argv[1], "secondary-gie2"); gst_bin_add(GST_BIN(appctx.pipeline), sgie2); }
        if (enable_sgie3) { sgie3 = gst_element_factory_make("nvinferserver", "sgie3"); nvds_parse_gie(sgie3, argv[1], "secondary-gie3"); gst_bin_add(GST_BIN(appctx.pipeline), sgie3); }
        if (enable_sgie4) { sgie4 = gst_element_factory_make("nvinferserver", "sgie4"); nvds_parse_gie(sgie4, argv[1], "secondary-gie4"); gst_bin_add(GST_BIN(appctx.pipeline), sgie4); }
        if (enable_sgie5) { sgie5 = gst_element_factory_make("nvinferserver", "sgie5"); nvds_parse_gie(sgie5, argv[1], "secondary-gie5"); gst_bin_add(GST_BIN(appctx.pipeline), sgie5); }
        if (enable_analytics) {
            appctx.nvanalytics = gst_element_factory_make("nvdsanalytics", "analytics");
            nvds_parse_nvdsanalytics(appctx.nvanalytics, argv[1], "analytics");
            gst_bin_add(GST_BIN(appctx.pipeline), appctx.nvanalytics);
        }
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
        if (!msgconv) { g_printerr("Failed to create nvmsgconv\n"); return -1; }
        if (!msgconv_config.empty()) g_object_set(G_OBJECT(msgconv), "config", msgconv_config.c_str(), NULL);
        g_object_set(G_OBJECT(msgconv), "payload-type", msgconv_payload_type, NULL);
        if (msgconv_msg2p_newapi == "yes") g_object_set(G_OBJECT(msgconv), "msg2p-newapi", TRUE, NULL);

        msgbroker = gst_element_factory_make("nvmsgbroker", "nvmsg-broker");
        if (!msgbroker) { g_printerr("Failed to create nvmsgbroker\n"); return -1; }
        if (!msgbroker_proto_lib.empty()) g_object_set(G_OBJECT(msgbroker), "proto-lib", msgbroker_proto_lib.c_str(), NULL);
        if (!msgbroker_conn_str.empty()) g_object_set(G_OBJECT(msgbroker), "conn-str", msgbroker_conn_str.c_str(), NULL);
        if (!msgbroker_topic.empty()) g_object_set(G_OBJECT(msgbroker), "topic", msgbroker_topic.c_str(), NULL);
        if (!msgbroker_cfg_file.empty()) g_object_set(G_OBJECT(msgbroker), "config", msgbroker_cfg_file.c_str(), NULL);
        if (msgbroker_comp_id > 0) g_object_set(G_OBJECT(msgbroker), "comp-id", msgbroker_comp_id, NULL);
        g_object_set(G_OBJECT(msgbroker), "sync", FALSE, "async", FALSE, NULL);
        gst_bin_add_many(GST_BIN(appctx.pipeline), tee, msgconv, msgbroker, NULL);
        g_print("Message Broker ENABLED: proto=%s conn=%s topic=%s\n",
                msgbroker_proto_lib.c_str(), msgbroker_conn_str.c_str(), msgbroker_topic.c_str());
    }

    /* 4. Sink Logic */
    gboolean enc_enable = (sink_type == 1 || sink_type == 2);
    if (enc_enable) {
        appctx.nvvidconv2 = gst_element_factory_make("nvvideoconvert", "nvvideo-converter-2");
        caps_filter = gst_element_factory_make("capsfilter", "encoder_caps");
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

    if (sink_type == 1) {
        qtmux = gst_element_factory_make("matroskamux", "muxer");
        appctx.sink = gst_element_factory_make("filesink", "filesink");
        g_object_set(G_OBJECT(appctx.sink), "location", output_file.c_str(), "sync", 0, "async", 0, NULL);
    } else if (sink_type == 2) {
        if (codec_status.codec_type == 1) payloader = gst_element_factory_make("rtph264pay", "rtp-payloader");
        else payloader = gst_element_factory_make("rtph265pay", "rtp-payloader");
        g_object_set(G_OBJECT(payloader), "pt", 96, NULL);
        appctx.sink = gst_element_factory_make("udpsink", "udp-sink");
        g_object_set(G_OBJECT(appctx.sink), "host", "127.0.0.1", "port", udp_port_num, "async", 0, "sync", 0, "qos", 0, NULL);
    } else if (sink_type == 0) {
        appctx.sink = gst_element_factory_make("fakesink", "fakesink");
        g_object_set(G_OBJECT(appctx.sink), "sync", 0, "async", 0, NULL);
    } else if (sink_type == 3) {
        appctx.sink = gst_element_factory_make("nveglglessink", "egl-sink");
        g_object_set(G_OBJECT(appctx.sink), "sync", 0, "async", 0, NULL);
    }

    /* Add mandatory elements */
    if (!parallel_mode && appctx.pgie) gst_bin_add(GST_BIN(appctx.pipeline), appctx.pgie);
    gst_bin_add_many(GST_BIN(appctx.pipeline), appctx.nvdslogger, appctx.nvvidconv, appctx.sink, NULL);
    if (enc_enable) gst_bin_add_many(GST_BIN(appctx.pipeline), appctx.nvvidconv2, caps_filter, appctx.encoder, appctx.parser, NULL);
    if (qtmux) gst_bin_add(GST_BIN(appctx.pipeline), qtmux);
    if (payloader) gst_bin_add(GST_BIN(appctx.pipeline), payloader);

    /* ═══════════════════════════════════════════════════════════════════
     * LINKING
     * ═══════════════════════════════════════════════════════════════════ */
    GstElement *last_element = NULL;
    if (appctx.restServer) last_element = gst_nvmultiurisrcbincreator_get_bin(appctx.nvmultiurisrcbinCreator);
    else last_element = appctx.multiuribin;

    auto link_next = [&](GstElement* next_elem, const char* q_suffix, bool use_leaky = false) {
        if (!next_elem) return;
        GstElement *q = create_queue(q_suffix, use_leaky);
        if (!q) return;
        gst_bin_add(GST_BIN(appctx.pipeline), q);
        if (!gst_element_link(last_element, q))
            g_printerr("ERROR: Failed to link %s -> %s\n", GST_ELEMENT_NAME(last_element), GST_ELEMENT_NAME(q));
        if (!gst_element_link(q, next_elem))
            g_printerr("ERROR: Failed to link %s -> %s\n", GST_ELEMENT_NAME(q), GST_ELEMENT_NAME(next_elem));
        last_element = next_elem;
    };

    if (parallel_mode) {
        /* ═══════════════════════════════════════════════════════════════
         * v12-fixed PARALLEL MODE (NVIDIA reference architecture)
         *
         * nvmultiurisrcbin → tee → queue → metamux.sink_0   (passthrough)
         *                       → queue → nvstreamdemux (SINGLE)
         *                                    ├ src_N → source_tee[N] → queue → branch0_mux.sink_N
         *                                    │                        → queue → branch1_mux.sink_N
         *
         * branch0_mux → pgie0 → ... → metamux.sink_1
         * branch1_mux → pgie1 → ... → metamux.sink_2
         * metamux → nvdslogger → ...
         * ═══════════════════════════════════════════════════════════════ */

        /* --- Create main tee --- */
        pipeline_tee = gst_element_factory_make("tee", "parallel_tee");
        if (!pipeline_tee) { g_printerr("ERROR: Failed to create tee!\n"); goto done; }
        gst_bin_add(GST_BIN(appctx.pipeline), pipeline_tee);
        link_next(pipeline_tee, "pre_tee");

        /* --- Create nvdsmetamux --- */
        metamux = gst_element_factory_make("nvdsmetamux", "metamux");
        if (!metamux) { g_printerr("ERROR: Failed to create metamux!\n"); goto done; }
        gst_bin_add(GST_BIN(appctx.pipeline), metamux);

        /* --- PASSTHROUGH PATH: tee → queue → metamux.sink_0 ---
         * This sends the original unmodified buffer as the "base" for metamux.
         * Metamux uses this as the output buffer and merges metadata from
         * all inference branches onto it. This is how the NVIDIA reference works. */
        {
            GstElement *passthrough_queue = create_queue("passthrough", true);   //use leaky instead 
            g_object_set(G_OBJECT(passthrough_queue), "max-size-buffers", (guint)2, "max-size-time", (guint64)0, "max-size-bytes", 0, NULL);

            gst_bin_add(GST_BIN(appctx.pipeline), passthrough_queue);
            link_from_tee(pipeline_tee, passthrough_queue);
            link_to_metamux(metamux, passthrough_queue, 0);
            g_print("[v12-fixed] Passthrough path: tee → queue → metamux.sink_0\n");
        }

        /* --- Create SINGLE nvstreamdemux --- */
        stream_demux = gst_element_factory_make("nvstreamdemux", "stream_demux");
        if (!stream_demux) { g_printerr("ERROR: Failed to create nvstreamdemux!\n"); goto done; }
        gst_bin_add(GST_BIN(appctx.pipeline), stream_demux);

        /* Link: tee → queue → demux */
        {
            GstElement *demux_queue = create_queue("tee-to-demux", true);
            gst_bin_add(GST_BIN(appctx.pipeline), demux_queue);
            link_from_tee(pipeline_tee, demux_queue);
            gst_element_link(demux_queue, stream_demux);
            g_print("[v12-fixed] Demux path: tee → queue → nvstreamdemux\n");
        }

        /* --- Pre-create all src pads on the SINGLE demux --- */
        g_print("[v12-fixed] Pre-creating %d src pads on nvstreamdemux...\n", max_demux_pads);
        for (int p = 0; p < max_demux_pads; p++) {
            char pad_name[32];
            snprintf(pad_name, sizeof(pad_name), "src_%d", p);
            GstPad *src_pad = gst_element_request_pad_simple(stream_demux, pad_name);
            if (src_pad) gst_object_unref(src_pad);
        }
        g_print("[v12-fixed] Pre-created %d demux src pads.\n", max_demux_pads);

        /* --- Build each parallel branch --- */
        for (int i = 0; i < (int)parallel_branches.size(); i++) {
            ParallelBranch &br = parallel_branches[i];
            char name[64];

            /* Per-branch nvstreammux (batch_size = maxBatchSize since all sources go to all branches) */
            snprintf(name, sizeof(name), "branch%d_mux", br.index);
            br.branch_mux = gst_element_factory_make("nvstreammux", name);
            if (!br.branch_mux) { g_printerr("ERROR: Failed to create branch%d nvstreammux!\n", br.index); goto done; }
            g_object_set(G_OBJECT(br.branch_mux),
                "width",                (guint)branch_mux_width,
                "height",               (guint)branch_mux_height,
                "batch-size",           (guint)branch_mux_batch_size,
                "batched-push-timeout", (guint)branch_mux_batch_timeout,
                "live-source",          TRUE,
                NULL);
            gst_bin_add(GST_BIN(appctx.pipeline), br.branch_mux);

            /* NOTE: demux → source_tee → branch_mux linking is DYNAMIC via do_deferred_link */

            GstElement *branch_last = br.branch_mux;

            /* PGIE */
            snprintf(name, sizeof(name), "parallel-pgie-%d", br.index);
            if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER)
                br.pgie = gst_element_factory_make("nvinferserver", name);
            else
                br.pgie = gst_element_factory_make("nvinfer", name);
            if (!br.pgie) { g_printerr("ERROR: Failed to create PGIE for branch %d\n", br.index); goto done; }
            nvds_parse_gie(br.pgie, argv[1], br.pgie_key.c_str());
            gst_bin_add(GST_BIN(appctx.pipeline), br.pgie);
            gst_element_link(branch_last, br.pgie);
            branch_last = br.pgie;

            /* Tracker */
            if (br.enable_tracker) {
                snprintf(name, sizeof(name), "parallel-tracker-%d", br.index);
                br.tracker = gst_element_factory_make("nvtracker", name);
                nvds_parse_tracker(br.tracker, argv[1], br.tracker_key.c_str());
                gst_bin_add(GST_BIN(appctx.pipeline), br.tracker);
                gst_element_link(branch_last, br.tracker);
                branch_last = br.tracker;
            }

            /* SGIE */
            if (br.enable_sgie) {
                snprintf(name, sizeof(name), "parallel-sgie-%d", br.index);
                if (pgie_type == NVDS_GIE_PLUGIN_INFER_SERVER)
                    br.sgie = gst_element_factory_make("nvinferserver", name);
                else
                    br.sgie = gst_element_factory_make("nvinfer", name);
                if (!br.sgie) { g_printerr("ERROR: Failed to create SGIE for branch%d\n", br.index); goto done; }
                nvds_parse_gie(br.sgie, argv[1], br.sgie_key.c_str());
                gst_bin_add(GST_BIN(appctx.pipeline), br.sgie);
                gst_element_link(branch_last, br.sgie);
                branch_last = br.sgie;
            }

            /* Analytics */
            if (br.enable_analytics) {
                snprintf(name, sizeof(name), "parallel-analytics-%d", br.index);
                br.analytics = gst_element_factory_make("nvdsanalytics", name);
                if (!br.analytics) { g_printerr("ERROR: Failed to create analytics for branch %d\n", br.index); goto done; }
                nvds_parse_nvdsanalytics(br.analytics, argv[1], br.analytics_key.c_str());
                gst_bin_add(GST_BIN(appctx.pipeline), br.analytics);
                gst_element_link(branch_last, br.analytics);
                branch_last = br.analytics;
            }

            /* Connect branch output → metamux.sink_(i+1) */
            link_to_metamux(metamux, branch_last, i + 1);

            /* FPS probe on branch output */
            {
                GstPad *src_pad = gst_element_get_static_pad(branch_last, "src");
                if (src_pad) {
                    gst_pad_add_probe(src_pad, GST_PAD_PROBE_TYPE_BUFFER,
                                      branch_fps_probe, NULL, NULL);
                    gst_object_unref(src_pad);
                    g_print("[v12-fixed] Installed FPS probe on branch %d output\n", i);
                }
            }

            g_print("  Branch %d: %s | mux (batch=%d) | tracker=%s sgie=%s analytics=%s → metamux.sink_%d\n",
                    br.index, br.pgie_key.c_str(), branch_mux_batch_size,
                    br.enable_tracker ? "ON" : "OFF",
                    br.enable_sgie ? "ON" : "OFF",
                    br.enable_analytics ? "ON" : "OFF",
                    i + 1);
        }

        /* ── Install pad probe on the SINGLE demux's SINK pad ──
         * Detects new source_ids, schedules deferred link to create:
         *   demux.src_N → source_tee[N] → queue → branch_mux[i].sink_N (for all i)
         */
        probe_ctx = new DemuxProbeCtx();
        probe_ctx->pipeline = appctx.pipeline;
        probe_ctx->stream_demux = stream_demux;
        probe_ctx->branches = &parallel_branches;

        {
            GstPad *demux_sinkpad = gst_element_get_static_pad(stream_demux, "sink");
            if (demux_sinkpad) {
                gst_pad_add_probe(demux_sinkpad, GST_PAD_PROBE_TYPE_BUFFER,
                                  demux_sink_probe_cb, probe_ctx, NULL);
                gst_object_unref(demux_sinkpad);
                g_print("[v12-fixed] Installed probe on nvstreamdemux sink pad\n");
            } else {
                g_printerr("[v12-fixed] ERROR: Could not get demux sink pad!\n");
                goto done;
            }
        }

        /* Continue pipeline after metamux */
        last_element = metamux;
        link_next(appctx.nvdslogger, "logger");

    } else {
        /* ═══════════════════════════════════════════════════════════════
         * LINEAR MODE
         * ═══════════════════════════════════════════════════════════════ */
        if (enable_preprocess) link_next(appctx.preprocess, "preproc");
        if (enable_pgie) link_next(appctx.pgie, "pgie");
        if (enable_tracker) link_next(tracker, "tracker");
        if (enable_sgie1) link_next(sgie1, "sgie1");
        if (enable_sgie2) link_next(sgie2, "sgie2");
        if (enable_sgie3) link_next(sgie3, "sgie3");
        if (enable_sgie4) link_next(sgie4, "sgie4");
        if (enable_sgie5) link_next(sgie5, "sgie5");
        if (enable_analytics) link_next(appctx.nvanalytics, "analytics");
        link_next(appctx.nvdslogger, "logger");
    }

    /* Message Broker Path */
    if (enable_msgbroker) {
        link_next(tee, "pre-msgbroker");
        GstElement *q_msg = create_queue("msg", true);
        gst_bin_add(GST_BIN(appctx.pipeline), q_msg);
        GstPad *tee_msg_pad = gst_element_request_pad_simple(tee, "src_%u");
        GstPad *q_msg_pad = gst_element_get_static_pad(q_msg, "sink");
        if (gst_pad_link(tee_msg_pad, q_msg_pad) != GST_PAD_LINK_OK) g_printerr("ERROR: tee→msg queue\n");
        gst_object_unref(q_msg_pad); gst_object_unref(tee_msg_pad);
        gst_element_link(q_msg, msgconv); gst_element_link(msgconv, msgbroker);

        GstElement *q_main = create_queue("main", false);
        gst_bin_add(GST_BIN(appctx.pipeline), q_main);
        GstPad *tee_main_pad = gst_element_request_pad_simple(tee, "src_%u");
        GstPad *q_main_sink = gst_element_get_static_pad(q_main, "sink");
        if (gst_pad_link(tee_main_pad, q_main_sink) != GST_PAD_LINK_OK) g_printerr("ERROR: tee→main queue\n");
        gst_object_unref(q_main_sink); gst_object_unref(tee_main_pad);
        last_element = q_main;
    }

    /* Visualization chain */
    if (enable_tiler) link_next(appctx.tiler, "tiler");
    link_next(appctx.nvvidconv, "conv");
    if (enable_osd) link_next(appctx.nvosd, "osd");

    /* Sink path */
    if (enc_enable) {
        link_next(appctx.nvvidconv2, "conv2");
        link_next(caps_filter, "caps");
        link_next(appctx.encoder, "encoder");
        link_next(appctx.parser, "parser");
        if (qtmux) link_next(qtmux, "muxer");
        if (payloader) link_next(payloader, "pay");
    }
    link_next(appctx.sink, "sink", true);

    /* ═══════════════════════════════════════════════════════════════════
     * START
     * ═══════════════════════════════════════════════════════════════════ */
    bus = gst_pipeline_get_bus(GST_PIPELINE(appctx.pipeline));
    bus_watch_id = gst_bus_add_watch(bus, bus_call, loop);
    gst_object_unref(bus);

    if (parallel_mode) {
        g_timeout_add_seconds(30, diagnostic_timer_cb, NULL);
        g_timeout_add_seconds(5, fps_report_timer_cb, NULL);
        g_print("[v12-fixed] Diagnostic timer (30s) + FPS report timer (5s) installed\n");
    }

    g_print("\n=== STARTING PIPELINE v12-fixed (Parallel=%s, SingleDemux+SourceTees, DemuxPads=%d) ===\n",
            parallel_mode ? "ON" : "OFF",
            parallel_mode ? max_demux_pads : 0);
    gst_element_set_state(appctx.pipeline, GST_STATE_PLAYING);

    /* Source Transform: schedule initial scan after pipeline is PLAYING.
     * Delay 2s to let nvmultiurisrcbin create internal nvurisrcbin
     * elements and link them to its internal nvstreammux. */
    if (g_src_transform.enable) {
        g_timeout_add(2000, source_transform_scan_and_inject, NULL);
        g_print("[SourceTransform] Initial injection scan scheduled in 2s\n");
    }

    GST_DEBUG_BIN_TO_DOT_FILE(GST_BIN(appctx.pipeline), GST_DEBUG_GRAPH_SHOW_ALL, "deepstream-pipeline");
    g_print("Pipeline DOT file written to ./tmp/deepstream-pipeline.dot\n");

    /* RTSP Server */
    if (sink_type == 2) {
        rtsp_server = gst_rtsp_server_new();
        g_object_set(rtsp_server, "service", std::to_string(rtsp_port_num).c_str(), NULL);
        GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(rtsp_server);
        GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();
        char launch_str[256];
        if (codec_status.codec_type == 1) {
            snprintf(launch_str, sizeof(launch_str),
                "( udpsrc address=127.0.0.1 port=%d caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\" "
                "! rtph264depay ! rtph264pay name=pay0 pt=96 )", udp_port_num);
        } else {
            snprintf(launch_str, sizeof(launch_str),
                "( udpsrc address=127.0.0.1 port=%d caps=\"application/x-rtp,media=video,encoding-name=H265,payload=96\" "
                "! rtph265depay ! rtph265pay name=pay0 pt=96 )", udp_port_num);
        }
        gst_rtsp_media_factory_set_launch(factory, launch_str);
        gst_rtsp_media_factory_set_shared(factory, TRUE);
        gst_rtsp_mount_points_add_factory(mounts, "/output", factory);
        g_object_unref(mounts);
        guint server_id = gst_rtsp_server_attach(rtsp_server, NULL);
        if (server_id == 0) {
            g_printerr("ERROR: Failed to attach RTSP server on port %d\n", rtsp_port_num);
        } else {
            g_print("\n========================================\n");
            g_print(" RTSP OUTPUT: rtsp://127.0.0.1:%d/output\n", rtsp_port_num);
            g_print("========================================\n\n");
        }
    }

    g_main_loop_run(loop);

    /* ═══════════════════════════════════════════════════════════════════
     * CLEANUP
     * ═══════════════════════════════════════════════════════════════════ */
done:
    g_print("Returned, stopping pipeline...\n");
    if (probe_ctx) { delete probe_ctx; probe_ctx = nullptr; }
    if (g_st_ctx.internal_mux) { gst_object_unref(g_st_ctx.internal_mux); g_st_ctx.internal_mux = nullptr; }
    if (rtsp_server) { g_object_unref(rtsp_server); rtsp_server = NULL; }
    gst_element_set_state(appctx.pipeline, GST_STATE_NULL);
    gst_object_unref(GST_OBJECT(appctx.pipeline));
    g_source_remove(bus_watch_id);
    g_main_loop_unref(loop);
    return 0;
}