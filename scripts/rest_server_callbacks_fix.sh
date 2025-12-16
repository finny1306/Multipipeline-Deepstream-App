#!/bin/bash
# fix_deepstream_server.sh
# Fixes compilation + improves stream remove error handling
# Works on DeepStream 6.4 / 7.0
# Path: /opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-server

TARGET_FILE="/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-server/rest_server_callbacks.cpp"

echo "Backing up original file..."
cp "$TARGET_FILE" "$TARGET_FILE.backup.$(date +%Y%m%d_%H%M%S)"

echo "Writing fixed rest_server_callbacks.cpp..."
cat > "$TARGET_FILE" <<'EOF'
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "rest_server_callbacks.h"

void s_appinstance_callback_impl (NvDsServerAppInstanceInfo * appinstance_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  if (appinstance_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!s_force_eos_handle (serverappctx->nvmultiurisrcbinCreator,
            appinstance_info)) {
      if (appinstance_info->appinstance_flag == QUIT_APP) {
        appinstance_info->app_log =
          "QUIT_FAIL, Unable to handle force-pipeline-eos nvmultiurisrcbin";
        appinstance_info->status = QUIT_FAIL;
        appinstance_info->err_info.code = StatusInternalServerError;
      }
    } else {
      if (appinstance_info->appinstance_flag == QUIT_APP) {
        appinstance_info->status = QUIT_SUCCESS;
        appinstance_info->err_info.code = StatusOk;
        appinstance_info->app_log = "QUIT_SUCCESS";
        g_print ("appinstance force quit success\n");
      }
    }
  } else {
    g_print("Unsupported REST API version\n");
  }
}

void s_osd_callback_impl (NvDsServerOsdInfo * osd_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  guint sourceId = std::stoi (osd_info->stream_id);

  if (osd_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (serverappctx->nvmultiurisrcbinCreator, sourceId)) {
      if (osd_info->osd_flag == PROCESS_MODE) {
        osd_info->status = PROCESS_MODE_UPDATE_FAIL;
        osd_info->err_info.code = StatusInternalServerError;
        osd_info->osd_log = "PROCESS_MODE_UPDATE_FAIL, Unable to find stream id for osd property updation";
      }
    } else {
      GstEvent *nvevent =
          gst_nvevent_osd_process_mode_update ((char *) osd_info->stream_id.
          c_str (), osd_info->process_mode);
      if (!nvevent) {
        osd_info->status = PROCESS_MODE_UPDATE_FAIL;
        osd_info->osd_log = "PROCESS_MODE_UPDATE_FAIL, nv-osd-process-mode-update event creation failed";
        osd_info->err_info.code = StatusInternalServerError;
      }

      if (!gst_pad_push_event ((GstPad
                  *) (gst_nvmultiurisrcbincreator_get_source_pad (serverappctx->
                      nvmultiurisrcbinCreator)), nvevent)) {
        g_print
            ("[WARN] nv-osd-process-mode-update event not pushed downstream.. !! \n");
        osd_info->status = PROCESS_MODE_UPDATE_FAIL;
        osd_info->osd_log = "PROCESS_MODE_UPDATE_FAIL, nv-osd-process-mode-update event not pushed";
        osd_info->err_info.code = StatusInternalServerError;
      } else {
        osd_info->status = PROCESS_MODE_UPDATE_SUCCESS;
        osd_info->err_info.code = StatusOk;
        osd_info->osd_log = "PROCESS_MODE_UPDATE_SUCCESS";
      }

    }
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void s_mux_callback_impl (NvDsServerMuxInfo * mux_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  if (mux_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!set_nvuribin_mux_prop (serverappctx->nvmultiurisrcbinCreator,mux_info)) {
      switch (mux_info->mux_flag) {
        case BATCHED_PUSH_TIMEOUT:
          g_print ("[WARN] batched-push-timeout update failed .. !! \n");
          mux_info->mux_log = "BATCHED_PUSH_TIMEOUT_UPDATE_FAIL, batched-push-timeout value not updated";
          mux_info->status = BATCHED_PUSH_TIMEOUT_UPDATE_FAIL;
          mux_info->err_info.code = StatusInternalServerError;
          break;
        case MAX_LATENCY:
          g_print ("[WARN] max-latency update failed .. !! \n");
          mux_info->mux_log = "MAX_LATENCY_UPDATE_FAIL, MAX_LATENCY_UPDATE_FAIL, max-latency value not updated";
          mux_info->status = MAX_LATENCY_UPDATE_FAIL;
          mux_info->err_info.code = StatusInternalServerError;
          break;
        default:
          break;
      }
    } else {
      switch (mux_info->mux_flag) {
        case BATCHED_PUSH_TIMEOUT:
          mux_info->status =
              mux_info->status !=
              BATCHED_PUSH_TIMEOUT_UPDATE_FAIL ?
              BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS :
              BATCHED_PUSH_TIMEOUT_UPDATE_FAIL;
          if ( mux_info->status == BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS ){
            mux_info->err_info.code = StatusOk;
            mux_info->mux_log = "BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS";
          } else{
            mux_info->err_info.code = StatusInternalServerError;
            mux_info->mux_log = "BATCHED_PUSH_TIMEOUT_UPDATE_SUCCESS, Error while setting batched-push-timeout property";
          }
          break;
        case MAX_LATENCY:
          mux_info->status =
              mux_info->status !=
              MAX_LATENCY_UPDATE_FAIL ? MAX_LATENCY_UPDATE_SUCCESS :
              MAX_LATENCY_UPDATE_FAIL;
          if ( mux_info->status == MAX_LATENCY_UPDATE_SUCCESS ){
            mux_info->err_info.code = StatusOk;
            mux_info->mux_log = "MAX_LATENCY_UPDATE_SUCCESS";
          } else{
            mux_info->err_info.code = StatusInternalServerError;
            mux_info->mux_log = "MAX_LATENCY_UPDATE_FAIL, Error while setting max-latency property";
          }
          break;
        default:
          break;
      }
    }

    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
      nvmultiurisrcbinCreator);
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void s_enc_callback_impl (NvDsServerEncInfo * enc_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;
  guint sourceId = std::stoi (enc_info->stream_id);
  GstEvent *nvevent = NULL;

  if (enc_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (serverappctx->nvmultiurisrcbinCreator, sourceId)) {
      /* source not found - set error for all encoder commands */
      switch (enc_info->enc_flag) {
        case BITRATE:        enc_info->status = BITRATE_UPDATE_FAIL; break;
        case FORCE_IDR:      enc_info->status = FORCE_IDR_UPDATE_FAIL; break;
        case FORCE_INTRA:    enc_info->status = FORCE_INTRA_UPDATE_FAIL; break;
        case IFRAME_INTERVAL:enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL; break;
        default: break;
      }
      enc_info->err_info.code = StatusInternalServerError;
      return;
    }

    switch (enc_info->enc_flag) {
      case BITRATE:
        nvevent = gst_nvevent_enc_bitrate_update ((gchar*)enc_info->stream_id.c_str (), enc_info->bitrate);
        if (!nvevent || !gst_pad_push_event ((GstPad*)gst_nvmultiurisrcbincreator_get_source_pad (serverappctx->nvmultiurisrcbinCreator), nvevent)) {
          enc_info->status = BITRATE_UPDATE_FAIL;
          enc_info->err_info.code = StatusInternalServerError;
        } else {
          enc_info->status = BITRATE_UPDATE_SUCCESS;
          enc_info->err_info.code = StatusOk;
        }
        break;

      case FORCE_IDR:
        nvevent = gst_nvevent_enc_force_idr ((gchar*)enc_info->stream_id.c_str (), enc_info->force_idr);
        if (!nvevent || !gst_pad_push_event ((GstPad*)gst_nvmultiurisrcbincreator_get_source_pad (serverappctx->nvmultiurisrcbinCreator), nvevent)) {
          enc_info->status = FORCE_IDR_UPDATE_FAIL;
          enc_info->err_info.code = StatusInternalServerError;
        } else {
          enc_info->status = FORCE_IDR_UPDATE_SUCCESS;
          enc_info->err_info.code = StatusOk;
        }
        break;

      case FORCE_INTRA:
        nvevent = gst_nvevent_enc_force_intra ((gchar*)enc_info->stream_id.c_str (), enc_info->force_intra);
        if (!nvevent || !gst_pad_push_event ((GstPad*)gst_nvmultiurisrcbincreator_get_source_pad (serverappctx->nvmultiurisrcbinCreator), nvevent)) {
          enc_info->status = FORCE_INTRA_UPDATE_FAIL;
          enc_info->err_info.code = StatusInternalServerError;
        } else {
          enc_info->status = FORCE_INTRA_UPDATE_SUCCESS;
          enc_info->err_info.code = StatusOk;
        }
        break;

      case IFRAME_INTERVAL:
        nvevent = gst_nvevent_enc_iframeinterval_update ((gchar*)enc_info->stream_id.c_str (), enc_info->iframeinterval);
        if (!nvevent || !gst_pad_push_event ((GstPad*)gst_nvmultiurisrcbincreator_get_source_pad (serverappctx->nvmultiurisrcbinCreator), nvevent)) {
          enc_info->status = IFRAME_INTERVAL_UPDATE_FAIL;
          enc_info->err_info.code = StatusInternalServerError;
        } else {
          enc_info->status = IFRAME_INTERVAL_UPDATE_SUCCESS;
          enc_info->err_info.code = StatusOk;
        }
        break;

      default:
        break;
    }
    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->nvmultiurisrcbinCreator);
  } else {
    g_print("Unsupported REST API version\n");
  }
}

void s_conv_callback_impl (NvDsServerConvInfo * conv_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  guint sourceId = std::stoi (conv_info->stream_id);

  if (conv_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!set_nvuribin_conv_prop (serverappctx->nvmultiurisrcbinCreator, sourceId,
            conv_info)) {
      switch (conv_info->conv_flag) {
        case SRC_CROP:
          g_print ("[WARN] source-crop update failed .. !! \n");
          conv_info->conv_log = "SRC_CROP_UPDATE_FAIL, source-crop update failed";
          conv_info->status = SRC_CROP_UPDATE_FAIL;
          conv_info->err_info.code = StatusInternalServerError;
          break;
        case DEST_CROP:
          g_print ("[WARN] source-crop update failed .. !! \n");
          conv_info->conv_log = "DEST_CROP_UPDATE_FAIL, dest-crop update failed";
          conv_info->status = DEST_CROP_UPDATE_FAIL;
          conv_info->err_info.code = StatusInternalServerError;
          break;
        case FLIP_METHOD:
          g_print ("[WARN] flip-method update failed .. !! \n");
          conv_info->conv_log = "FLIP_METHOD_UPDATE_FAIL, flip-method update failed";
          conv_info->status = FLIP_METHOD_UPDATE_FAIL;
          conv_info->err_info.code = StatusInternalServerError;
          break;
        case INTERPOLATION_METHOD:
          g_print ("[WARN] interpolation-method update failed .. !! \n");
          conv_info->conv_log = "INTERPOLATION_METHOD_UPDATE_FAIL, interpolation-method update failed";
          conv_info->status = INTERPOLATION_METHOD_UPDATE_FAIL;
          conv_info->err_info.code = StatusInternalServerError;
          break;
        default:
          break;
      }
    } else {
      switch (conv_info->conv_flag) {
        case SRC_CROP:
          conv_info->status =
              conv_info->status !=
              SRC_CROP_UPDATE_FAIL ? SRC_CROP_UPDATE_SUCCESS :
              SRC_CROP_UPDATE_FAIL;
              if ( conv_info->status == SRC_CROP_UPDATE_SUCCESS ){
                conv_info->err_info.code = StatusOk;
                conv_info->conv_log = "SRC_CROP_UPDATE_SUCCESS";
              } else{
                conv_info->err_info.code = StatusInternalServerError;
                conv_info->conv_log = "SRC_CROP_UPDATE_FAIL, Error while setting src-crop property";
              }
          break;
        case DEST_CROP:
          conv_info->status =
              conv_info->status !=
              DEST_CROP_UPDATE_FAIL ? DEST_CROP_UPDATE_SUCCESS :
              DEST_CROP_UPDATE_FAIL;
          if ( conv_info->status == DEST_CROP_UPDATE_SUCCESS ){
            conv_info->err_info.code = StatusOk;
            conv_info->conv_log = "DEST_CROP_UPDATE_SUCCESS";
          } else{
            conv_info->err_info.code = StatusInternalServerError;
            conv_info->conv_log = "DEST_CROP_UPDATE_FAIL, Error while setting dest-crop property";
          }
          break;
        case FLIP_METHOD:
          conv_info->status =
              conv_info->status !=
              FLIP_METHOD_UPDATE_FAIL ? FLIP_METHOD_UPDATE_SUCCESS :
              FLIP_METHOD_UPDATE_FAIL;
          if ( conv_info->status == FLIP_METHOD_UPDATE_SUCCESS ){
            conv_info->err_info.code = StatusOk;
            conv_info->conv_log = "FLIP_METHOD_UPDATE_SUCCESS";
          } else{
            conv_info->err_info.code = StatusInternalServerError;
            conv_info->conv_log = "FLIP_METHOD_UPDATE_FAIL, Error while setting flip-method property";
          }
          break;
        case INTERPOLATION_METHOD:
          conv_info->status =
              conv_info->status !=
              INTERPOLATION_METHOD_UPDATE_FAIL ?
              INTERPOLATION_METHOD_UPDATE_SUCCESS :
              INTERPOLATION_METHOD_UPDATE_FAIL;
          if ( conv_info->status == INTERPOLATION_METHOD_UPDATE_SUCCESS ){
            conv_info->err_info.code = StatusOk;
            conv_info->conv_log = "INTERPOLATION_METHOD_UPDATE_SUCCESS";
          } else{
            conv_info->err_info.code = StatusInternalServerError;
            conv_info->conv_log = "INTERPOLATION_METHOD_UPDATE_FAIL, Error while setting interpolation-method property";
          }
          break;
        default:
          break;
      }
    }
    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
      nvmultiurisrcbinCreator);
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void s_inferserver_callback_impl (NvDsServerInferServerInfo * inferserver_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  guint sourceId = std::stoi (inferserver_info->stream_id);

  if (inferserver_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (serverappctx->nvmultiurisrcbinCreator, sourceId)) {
      if (inferserver_info->inferserver_flag == INFERSERVER_INTERVAL) {
        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
        inferserver_info->err_info.code = StatusInternalServerError;
        inferserver_info->inferserver_log = "INFERSERVER_INTERVAL_UPDATE_FAIL, Unable to find stream id for infer (inferserver) property updation";
      }
    } else {
      GstEvent *nvevent =
          gst_nvevent_infer_interval_update ((char *) inferserver_info->stream_id.
          c_str (), inferserver_info->interval);
      if (!nvevent) {
        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
        inferserver_info->inferserver_log =
            "INFERSERVER_INTERVAL_UPDATE_FAIL, nv-infer-interval-update event (inferserver) creation failed";
        inferserver_info->err_info.code = StatusInternalServerError;
      }

      /* send nv-infer-interval-update event */
      if (!gst_pad_push_event ((GstPad
                  *) (gst_nvmultiurisrcbincreator_get_source_pad (serverappctx->
                      nvmultiurisrcbinCreator)), nvevent)) {
        g_print
            ("[WARN] nv-infer-interval-update (inferserver) event not pushed downstream.. !! \n");
        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_FAIL;
        inferserver_info->inferserver_log =
            "INFERSERVER_INTERVAL_UPDATE_FAIL, nv-infer-interval-update event not pushed";
        inferserver_info->err_info.code = StatusInternalServerError;
      } else {
        inferserver_info->status = INFERSERVER_INTERVAL_UPDATE_SUCCESS;
        inferserver_info->err_info.code = StatusOk;
        inferserver_info->inferserver_log = "INFERSERVER_INTERVAL_UPDATE_SUCCESS";
      }
      gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
          nvmultiurisrcbinCreator);
    }
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void s_infer_callback_impl (NvDsServerInferInfo * infer_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  guint sourceId = std::stoi (infer_info->stream_id);

  if (infer_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (serverappctx->nvmultiurisrcbinCreator, sourceId)) {
      infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
      infer_info->err_info.code = StatusInternalServerError;
      infer_info->infer_log = "INFER_INTERVAL_UPDATE_FAIL, Unable to find stream id for infer property updation";
    } else {
      GstEvent *nvevent =
          gst_nvevent_infer_interval_update ((char *) infer_info->stream_id.
          c_str (), infer_info->interval);
      if (!nvevent) {
        infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
        infer_info->infer_log = "INFER_INTERVAL_UPDATE_FAIL, nv-infer-interval-update event creation failed";
        infer_info->err_info.code = StatusInternalServerError;
      }
      /* send nv-infer-interval-update event */
      if (!gst_pad_push_event ((GstPad
                  *) (gst_nvmultiurisrcbincreator_get_source_pad (serverappctx->
                      nvmultiurisrcbinCreator)), nvevent)) {
        g_print
            ("[WARN] nv-infer-interval-update event not pushed downstream.. !! \n");
        infer_info->status = INFER_INTERVAL_UPDATE_FAIL;
        infer_info->infer_log = "INFER_INTERVAL_UPDATE_FAIL, nv-infer-interval-update event not pushed";
        infer_info->err_info.code = StatusInternalServerError;
      } else {
        infer_info->status = INFER_INTERVAL_UPDATE_SUCCESS;
        infer_info->infer_log = "INFER_INTERVAL_UPDATE_SUCCESS";
        infer_info->err_info.code = StatusOk;
      }
    }
    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
      nvmultiurisrcbinCreator);
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void s_dec_callback_impl (NvDsServerDecInfo * dec_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  guint sourceId = std::stoi (dec_info->stream_id);

  if (dec_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!set_nvuribin_dec_prop (serverappctx->nvmultiurisrcbinCreator, sourceId,
            dec_info)) {
      switch (dec_info->dec_flag) {
        case DROP_FRAME_INTERVAL:
          g_print ("[WARN] drop-frame-interval not set on decoder .. !! \n");
          dec_info->status = DROP_FRAME_INTERVAL_UPDATE_FAIL;
          dec_info->dec_log = "DROP_FRAME_INTERVAL_UPDATE_FAIL, drop-frame-interval not set on decoder";
          dec_info->err_info.code = StatusInternalServerError;
          break;
        case SKIP_FRAMES:
          g_print ("[WARN] skip-frame not set on decoder .. !! \n");
          dec_info->status = SKIP_FRAMES_UPDATE_FAIL;
          dec_info->dec_log = "SKIP_FRAMES_UPDATE_FAIL, skip-frame not set on decoder";
          dec_info->err_info.code = StatusInternalServerError;
          break;
        case LOW_LATENCY_MODE:
          g_print ("[WARN] low-latency-mode not set on decoder .. !! \n");
          dec_info->status = LOW_LATENCY_MODE_UPDATE_FAIL;
          dec_info->dec_log = "LOW_LATENCY_MODE_UPDATE_FAIL, low-latency-mode not set on decoder";
          dec_info->err_info.code = StatusInternalServerError;
          break;
        default:
          break;
      }
    } else {
      switch (dec_info->dec_flag) {
        case DROP_FRAME_INTERVAL:
          dec_info->status =
              dec_info->status !=
              DROP_FRAME_INTERVAL_UPDATE_FAIL ? DROP_FRAME_INTERVAL_UPDATE_SUCCESS
              : DROP_FRAME_INTERVAL_UPDATE_FAIL;
          if ( dec_info->status == DROP_FRAME_INTERVAL_UPDATE_SUCCESS ){
            dec_info->err_info.code = StatusOk;
            dec_info->dec_log = "DROP_FRAME_INTERVAL_UPDATE_SUCCESS";
          } else{
            dec_info->err_info.code = StatusInternalServerError;
            dec_info->dec_log = "DROP_FRAME_INTERVAL_UPDATE_FAIL, Error while setting drop-frame-interval property";
          }
          break;
        case SKIP_FRAMES:
          dec_info->status =
              dec_info->status !=
              SKIP_FRAMES_UPDATE_FAIL ? SKIP_FRAMES_UPDATE_SUCCESS :
              SKIP_FRAMES_UPDATE_FAIL;
          if ( dec_info->status == SKIP_FRAMES_UPDATE_SUCCESS ){
            dec_info->err_info.code = StatusOk;
            dec_info->dec_log = "SKIP_FRAMES_UPDATE_SUCCESS";
          } else{
            dec_info->err_info.code = StatusInternalServerError;
            dec_info->dec_log = "SKIP_FRAMES_UPDATE_FAIL, Error while setting skip-frame property";
          }
          break;
        case LOW_LATENCY_MODE:
          dec_info->status =
              dec_info->status !=
              LOW_LATENCY_MODE_UPDATE_FAIL ? LOW_LATENCY_MODE_UPDATE_SUCCESS :
              LOW_LATENCY_MODE_UPDATE_FAIL;
          if ( dec_info->status == LOW_LATENCY_MODE_UPDATE_SUCCESS ){
            dec_info->err_info.code = StatusOk;
            dec_info->dec_log = "LOW_LATENCY_MODE_UPDATE_SUCCESS";
          } else{
            dec_info->err_info.code = StatusInternalServerError;
            dec_info->dec_log = "LOW_LATENCY_MODE_UPDATE_FAIL, Error while setting skip-frame property";
          }
          break;
        default:
          break;
      }
    }

    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
      nvmultiurisrcbinCreator);
  } else {
    g_print("Unsupported REST API version\n");
  }

  return;
}

void s_roi_callback_impl (NvDsServerRoiInfo * roi_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;
  (void) serverappctx;

  guint sourceId = std::stoi (roi_info->stream_id);

  if (roi_info->uri.find ("/api/v1/") != std::string::npos) {
    if (!find_source (serverappctx->nvmultiurisrcbinCreator, sourceId)) {
      roi_info->status = ROI_UPDATE_FAIL;
      roi_info->err_info.code = StatusInternalServerError;
      roi_info->roi_log = "ROI_UPDATE_FAIL, Unable to find stream id for ROI updation";
    } else {
      RoiDimension roi_dim[roi_info->roi_count];

      for (int i = 0; i < (int) roi_info->roi_count; i++) {
        g_strlcpy (roi_dim[i].roi_id, roi_info->vect[i].roi_id,
            sizeof (roi_dim[i].roi_id));
        roi_dim[i].left = roi_info->vect[i].left;
        roi_dim[i].top = roi_info->vect[i].top;
        roi_dim[i].width = roi_info->vect[i].width;
        roi_dim[i].height = roi_info->vect[i].height;
      }

      GstEvent *nvevent =
          gst_nvevent_new_roi_update ((char *) roi_info->stream_id.c_str (),
          roi_info->roi_count, roi_dim);

      if (!nvevent) {
        roi_info->roi_log = "ROI_UPDATE_FAIL, nv-roi-update event creation failed";
        roi_info->status = ROI_UPDATE_FAIL;
        roi_info->err_info.code = StatusInternalServerError;
      }
      /* send nv-new_roi_update event */
      if (!gst_pad_push_event ((GstPad
                  *) (gst_nvmultiurisrcbincreator_get_source_pad (serverappctx->
                      nvmultiurisrcbinCreator)), nvevent)) {
        switch (roi_info->roi_flag) {
          case ROI_UPDATE:
            g_print ("[WARN] ROI UPDATE event not pushed downstream.. !! \n");
            roi_info->roi_log = "ROI_UPDATE_FAIL, nv-roi-update event not pushed";
            roi_info->status = ROI_UPDATE_FAIL;
            roi_info->err_info.code = StatusInternalServerError;
            break;
          default:
            break;
        }
      } else {
        switch (roi_info->roi_flag) {
          case ROI_UPDATE:
            roi_info->status = ROI_UPDATE_SUCCESS;
            roi_info->err_info.code = StatusOk;
            roi_info->roi_log = "ROI_UPDATE_SUCCESS";
            break;
          default:
            break;
        }
      }
    }
    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->
      nvmultiurisrcbinCreator);
  } else {
    g_print ("Unsupported REST API version\n");
  }

  return;
}

void s_stream_callback_impl (NvDsServerStreamInfo * stream_info, void *ctx)
{
  AppCtx *serverappctx = (AppCtx *) ctx;

  g_print ("Inside s_stream_callback_impl +++ \n");
  g_mutex_lock (&serverappctx->bincreator_lock);

  if (stream_info->uri.find ("/api/v1/") == std::string::npos) {
    g_print("Unsupported REST API version\n");
    g_mutex_unlock (&serverappctx->bincreator_lock);
    return;
  }

  if (g_strrstr (stream_info->value_change.c_str (), "add")) {
    /* ---- ADD STREAM ---- */
    GstDsNvUriSrcConfig **sources = NULL;
    guint num = 0;

    if (gst_nvmultiurisrcbincreator_get_active_sources_list (serverappctx->nvmultiurisrcbinCreator, &num, &sources)) {
      gst_nvmultiurisrcbincreator_src_config_list_free (serverappctx->nvmultiurisrcbinCreator, num, sources);
      if (num >= serverappctx->muxConfig.maxBatchSize) {
        stream_info->status = STREAM_ADD_FAIL;
        stream_info->stream_log = "STREAM_ADD_FAIL, max-batch-size reached";
        stream_info->err_info.code = StatusInternalServerError;
        g_mutex_unlock (&serverappctx->bincreator_lock);
        return;
      }
    }

    if (gst_nvmultiurisrcbincreator_get_source_config_by_sensorid (serverappctx->nvmultiurisrcbinCreator,
            stream_info->value_camera_id.c_str ())) {
      stream_info->status = STREAM_ADD_FAIL;
      stream_info->stream_log = "STREAM_ADD_FAIL, Duplicate camera_id";
      stream_info->err_info.code = StatusInternalServerError;
      g_mutex_unlock (&serverappctx->bincreator_lock);
      return;
    }

    serverappctx->config.uri       = (gchar *) stream_info->value_camera_url.c_str ();
    serverappctx->config.sensorId  = (gchar *) stream_info->value_camera_id.c_str ();
    serverappctx->config.sensorName= (gchar *) stream_info->value_camera_name.c_str ();
    serverappctx->config.source_id = ++serverappctx->sourceIdCounter;

    if (!gst_nvmultiurisrcbincreator_add_source (serverappctx->nvmultiurisrcbinCreator, &serverappctx->config)) {
      stream_info->status = STREAM_ADD_FAIL;
      stream_info->stream_log = "STREAM_ADD_FAIL, add_source failed";
      stream_info->err_info.code = StatusInternalServerError;
    } else {
      stream_info->status = STREAM_ADD_SUCCESS;
      stream_info->err_info.code = StatusOk;
      stream_info->stream_log = "STREAM_ADD_SUCCESS";
    }

    serverappctx->config.uri = serverappctx->config.sensorId = NULL;
    gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->nvmultiurisrcbinCreator);

  } else if (g_strrstr (stream_info->value_change.c_str (), "remove")) {
    /* ---- REMOVE STREAM ---- */
    GstDsNvUriSrcConfig *cfg = gst_nvmultiurisrcbincreator_get_source_config (
        serverappctx->nvmultiurisrcbinCreator,
        stream_info->value_camera_url.c_str (),
        stream_info->value_camera_id.c_str ());

    if (!cfg) {
      g_print ("REMOVE: No source found for id=%s url=%s\n",
          stream_info->value_camera_id.c_str(), stream_info->value_camera_url.c_str());
      stream_info->status = STREAM_REMOVE_FAIL;
      stream_info->stream_log = "STREAM_REMOVE_FAIL, source not found";
      stream_info->err_info.code = StatusBadRequest;
      g_mutex_unlock (&serverappctx->bincreator_lock);
      return;
    }

    gboolean ret = gst_nvmultiurisrcbincreator_remove_source (serverappctx->nvmultiurisrcbinCreator, cfg->source_id);
    gst_nvmultiurisrcbincreator_src_config_free (cfg);

    if (!ret) {
      stream_info->status = STREAM_REMOVE_FAIL;
      stream_info->stream_log = "STREAM_REMOVE_FAIL, remove_source returned FALSE";
      stream_info->err_info.code = StatusInternalServerError;
    } else {
      stream_info->status = STREAM_REMOVE_SUCCESS;
      stream_info->err_info.code = StatusOk;
      stream_info->stream_log = "STREAM_REMOVE_SUCCESS";
      gst_nvmultiurisrcbincreator_sync_children_states (serverappctx->nvmultiurisrcbinCreator);
    }
  } else {
    stream_info->stream_log = "Unknown value_change";
    stream_info->err_info.code = StatusBadRequest;
  }

  g_mutex_unlock (&serverappctx->bincreator_lock);
  g_print ("Exiting s_stream_callback_impl\n");
}
EOF

echo "File written successfully."
