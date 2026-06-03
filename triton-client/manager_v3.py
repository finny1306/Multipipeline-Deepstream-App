import subprocess
import os
import re
import signal
import sys
import yaml
import time
import uuid
import threading
import atexit
from flask import Flask, request, jsonify

# ================= CONFIGURATION =================
BINARY_PATH = "/workspace/triton-client/apps/deepstream-server-app"
TEMP_CONFIG_DIR = "/workspace/temp_configs"
LOG_DIR = "./logs"
ORCHESTRATOR_PORT = 5000

# Health check interval in seconds
HEALTH_CHECK_INTERVAL = 5
# =================================================

app = Flask(__name__)

# Active pipelines dictionary
active_pipelines = {}

# Lock for thread-safe access to active_pipelines
pipelines_lock = threading.Lock()

# Flag to stop the health check thread
shutdown_flag = threading.Event()


def ensure_dirs():   
    """Ensure necessary directories exist."""
    if not os.path.exists(TEMP_CONFIG_DIR):
        os.makedirs(TEMP_CONFIG_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)


# ===========================================================================
#  NVINFERSERVER CONFIG PATCHING
#  Rewrites unique_id, operate_on_gie_id, operate_on_class_ids in the
#  protobuf-text nvinferserver config so there are no ID conflicts between
#  branches and you don't have to manually edit .txt files per-pipeline.
# ===========================================================================

def _replace_field_in_block(block_text, field_name, new_value):
    """
    Within a protobuf-text block string, replace an existing field's value
    or append the field if it doesn't exist.
    
    Returns the modified block text.
    """
    field_re = re.compile(
        rf'^(\s*){field_name}\s*:.*$', re.MULTILINE
    )
    
    m = field_re.search(block_text)
    if m:
        indent = m.group(1)
        new_line = f"{indent}{field_name}: {new_value}"
        return block_text[:m.start()] + new_line + block_text[m.end():]
    else:
        # Append before closing brace - find the last non-whitespace line
        return block_text.rstrip() + f"\n  {field_name}: {new_value}\n"


def patch_nvinferserver_config(original_path, overrides, pipeline_id):
    """
    Read an nvinferserver protobuf-text config, patch fields from the
    overrides dict, and write to a temp file.
    
    Supported overrides (YAML key -> nvinferserver field):
        unique-id            -> infer_config { unique_id }
        operate-on-gie-id    -> input_control { operate_on_gie_id }
        operate-on-class-ids -> input_control { operate_on_class_ids }
    
    Returns:
        Path to the patched temp config file, or original_path if nothing changed.
    """
    if not overrides:
        return original_path
    
    if not os.path.exists(original_path):
        print(f"[ConfigPatch] WARNING: nvinferserver config not found: {original_path}")
        return original_path
    
    with open(original_path, 'r') as f:
        content = f.read()
    
    modified = False
    
    # --- Patch unique_id inside infer_config { ... } ---
    if 'unique-id' in overrides:
        new_uid = overrides['unique-id']
        # Find the infer_config block and replace unique_id within it
        infer_re = re.compile(
            r'(infer_config\s*\{)(.*?)(^\})', re.DOTALL | re.MULTILINE
        )
        m = infer_re.search(content)
        if m:
            block_body = m.group(2)
            new_body = _replace_field_in_block(block_body, 'unique_id', new_uid)
            content = content[:m.start(2)] + new_body + content[m.end(2):]
            modified = True
            print(f"[ConfigPatch]   unique_id -> {new_uid}")
    
    # --- Patch fields inside input_control { ... } ---
    input_control_fields = {}
    
    if 'operate-on-gie-id' in overrides:
        input_control_fields['operate_on_gie_id'] = overrides['operate-on-gie-id']
    
    if 'operate-on-class-ids' in overrides:
        raw = overrides['operate-on-class-ids']
        if isinstance(raw, list):
            input_control_fields['operate_on_class_ids'] = ' '.join(str(x) for x in raw)
        else:
            input_control_fields['operate_on_class_ids'] = str(raw)
    
    if input_control_fields:
        # Find the input_control block
        ic_re = re.compile(
            r'(input_control\s*\{)(.*?)(^\})', re.DOTALL | re.MULTILINE
        )
        m = ic_re.search(content)
        
        if m:
            block_body = m.group(2)
            for field_name, field_val in input_control_fields.items():
                block_body = _replace_field_in_block(block_body, field_name, field_val)
                print(f"[ConfigPatch]   {field_name} -> {field_val}")
            content = content[:m.start(2)] + block_body + content[m.end(2):]
        else:
            # No input_control block exists — create one
            lines = []
            for field_name, field_val in input_control_fields.items():
                lines.append(f"  {field_name}: {field_val}")
                print(f"[ConfigPatch]   {field_name} -> {field_val} (new block)")
            block = "\ninput_control {\n" + "\n".join(lines) + "\n}\n"
            content += block
        
        modified = True
    
    if not modified:
        return original_path
    
    # Write patched file
    base_name = os.path.splitext(os.path.basename(original_path))[0]
    patched_name = f"{base_name}_{pipeline_id}.txt"
    patched_path = os.path.join(TEMP_CONFIG_DIR, patched_name)
    
    with open(patched_path, 'w') as f:
        f.write(content)
    
    print(f"[ConfigPatch] Wrote: {patched_path}")
    return patched_path


def patch_gie_configs_in_pipeline(config, pipeline_id):
    """
    Scan all GIE sections in the YAML pipeline config. For each enabled GIE
    that has operate-on-gie-id, operate-on-class-ids, or a unique-id that
    differs from what's in the .txt file, create a patched copy of the
    nvinferserver .txt config and update the YAML to reference it.
    
    Returns:
        List of patched temp file paths (for cleanup tracking).
    """
    patched_files = []
    
    gie_pattern = re.compile(r'^(primary-gie|secondary-gie)', re.IGNORECASE)
    
    for section_name, section_data in list(config.items()):
        if not gie_pattern.match(section_name):
            continue
        if not isinstance(section_data, dict):
            continue
        if not section_data.get('enable', 0):
            continue
        
        config_file_path = section_data.get('config-file-path')
        if not config_file_path:
            continue
        
        # Collect overrides from the YAML
        overrides = {}
        
        if 'unique-id' in section_data:
            overrides['unique-id'] = section_data['unique-id']
        
        if 'operate-on-gie-id' in section_data:
            overrides['operate-on-gie-id'] = section_data['operate-on-gie-id']
        
        if 'operate-on-class-ids' in section_data:
            overrides['operate-on-class-ids'] = section_data['operate-on-class-ids']
        
        if overrides:
            print(f"[ConfigPatch] [{section_name}] patching {os.path.basename(config_file_path)}")
            patched_path = patch_nvinferserver_config(
                config_file_path, overrides, pipeline_id
            )
            
            if patched_path != config_file_path:
                section_data['config-file-path'] = patched_path
                patched_files.append(patched_path)
    
    return patched_files


def generate_config(base_config_path, port):
    """
    Generate a temporary pipeline config with the specified port.
    Also patches any nvinferserver .txt configs that need ID overrides.
    """
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Config file not found: {base_config_path}")

    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'server-app-ctx' not in config:
        config['server-app-ctx'] = {}
    
    # === FORCE NETWORK SETTINGS ===
    config['server-app-ctx']['httpPort'] = str(port)
    config['server-app-ctx']['httpIp'] = "0.0.0.0"
    config['server-app-ctx']['enable'] = 1
    # ==============================

    if 'rest-server' in config:
        config['rest-server']['enable'] = 1
    
    pipeline_id = str(uuid.uuid4())[:8]
    
    # === PATCH NVINFERSERVER CONFIGS ===
    patched_files = patch_gie_configs_in_pipeline(config, pipeline_id)
    if patched_files:
        print(f"[ConfigPatch] {len(patched_files)} nvinferserver config(s) patched for pipeline {pipeline_id}")
    # ==================================
    
    new_config_name = f"pipeline_{pipeline_id}_port_{port}.yml"
    new_config_path = os.path.join(TEMP_CONFIG_DIR, new_config_name)

    with open(new_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return pipeline_id, new_config_path, patched_files


def get_pipeline_env():
    """
    Creates the environment dictionary with the specific exports 
    required for DeepStream and the custom library paths.
    """
    env = os.environ.copy()
    
    env["NVDS_MULTIURI_ALLOW_MIXED_PROTOCOL"] = "1"
    env["NVDS_MULTIURI_ALLOW_EMPTY"] = "1"
    env["GIO_MODULE_DIR"] = "/nonexistent"
    env["NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT"] = "1"
    env["NVDS_ENABLE_DEBUG"] = "1"
    # env["GST_DEBUG"] = "2"
    
    ds_plugin_path = "/opt/nvidia/deepstream/deepstream-8.0/lib/gst-plugins"
    current_gst_path = env.get("GST_PLUGIN_PATH", "")
    if ds_plugin_path not in current_gst_path:
        env["GST_PLUGIN_PATH"] = f"{ds_plugin_path}:{current_gst_path}"

    extra_libs = "/workspace/lib:/opt/nvidia/deepstream/deepstream-8.0/lib"
    current_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{extra_libs}:{current_ld_path}"

    return env


def cleanup_dead_pipeline(pipeline_id, info, reason="died"):
    """
    Clean up a single dead pipeline.
    Closes log file handle, removes temp config and patched nvinferserver configs.
    """
    print(f"[Orchestrator] Pipeline {pipeline_id} {reason}. Cleaning up...")
    
    if 'log_file' in info and info['log_file']:
        try:
            info['log_file'].close()
        except Exception as e:
            print(f"[Orchestrator] Warning: Could not close log file for {pipeline_id}: {e}")
    
    if 'temp_config_path' in info and os.path.exists(info['temp_config_path']):
        try:
            os.remove(info['temp_config_path'])
            print(f"[Orchestrator] Removed temp config: {info['temp_config_path']}")
        except Exception as e:
            print(f"[Orchestrator] Warning: Could not remove temp config for {pipeline_id}: {e}")
    
    # Remove patched nvinferserver config files
    for patched_file in info.get('patched_nvinfer_configs', []):
        if os.path.exists(patched_file):
            try:
                os.remove(patched_file)
                print(f"[Orchestrator] Removed patched nvinfer config: {patched_file}")
            except Exception as e:
                print(f"[Orchestrator] Warning: Could not remove {patched_file}: {e}")
    
    print(f"[Orchestrator] Pipeline {pipeline_id} cleaned up. Port {info.get('port', 'unknown')} is now available.")


def check_pipeline_health():
    """Check all active pipelines and clean up dead ones."""
    dead_ids = []
    
    with pipelines_lock:
        for pid, info in list(active_pipelines.items()):
            ret_code = info['process'].poll()
            if ret_code is not None:
                dead_ids.append(pid)
                cleanup_dead_pipeline(pid, info, reason=f"exited with code {ret_code}")
        
        for pid in dead_ids:
            del active_pipelines[pid]
    
    return dead_ids


def health_check_worker():
    """Background worker that periodically checks pipeline health."""
    print(f"[Orchestrator] Health check worker started (interval: {HEALTH_CHECK_INTERVAL}s)")
    
    while not shutdown_flag.is_set():
        shutdown_flag.wait(timeout=HEALTH_CHECK_INTERVAL)
        if shutdown_flag.is_set():
            break
        dead_ids = check_pipeline_health()
        if dead_ids:
            print(f"[Orchestrator] Health check: Cleaned up {len(dead_ids)} dead pipeline(s): {dead_ids}")


def start_health_check_thread():
    """Start the background health check thread."""
    health_thread = threading.Thread(target=health_check_worker, daemon=True)
    health_thread.start()
    return health_thread


@app.route('/pipelines', methods=['GET'])
def list_pipelines():
    """List all active pipelines."""
    check_pipeline_health()
    
    status_list = []
    with pipelines_lock:
        for pid, info in active_pipelines.items():
            status_list.append({
                "id": pid,
                "port": info['port'],
                "config": info['base_config_path'],
                "pid": info['process'].pid,
                "running": info['process'].poll() is None
            })
    
    return jsonify({
        "active_pipelines": status_list, 
        "count": len(status_list),
        "health_check_interval": HEALTH_CHECK_INTERVAL
    })


@app.route('/pipelines/health', methods=['GET'])
def health_check_endpoint():
    """Endpoint to manually trigger a health check."""
    dead_ids = check_pipeline_health()
    
    with pipelines_lock:
        active_count = len(active_pipelines)
    
    return jsonify({
        "cleaned_up": dead_ids,
        "cleaned_count": len(dead_ids),
        "active_count": active_count
    })


@app.route('/pipelines/spawn', methods=['POST'])
def spawn_pipeline():
    """Spawn a new pipeline."""
    data = request.json
    base_config = data.get('config_path')
    requested_port = data.get('port')

    if not base_config:
        return jsonify({"error": "config_path is required"}), 400

    check_pipeline_health()

    with pipelines_lock:
        if not requested_port:
            used_ports = [p['port'] for p in active_pipelines.values()]
            requested_port = 9000
            while requested_port in used_ports:
                requested_port += 1
        else:
            for info in active_pipelines.values():
                if info['port'] == requested_port:
                    return jsonify({"error": f"Port {requested_port} is already in use"}), 409

        try:
            pipeline_id, temp_config_path, patched_files = generate_config(base_config, requested_port)
            
            log_file_path = os.path.join(LOG_DIR, f"{pipeline_id}.log")
            log_file = open(log_file_path, "w", buffering=1)

            cmd = [BINARY_PATH, temp_config_path]
            print(f"[Orchestrator] Spawning: {' '.join(cmd)}")
            print(f"[Orchestrator] Logging to: {log_file_path}")
            
            proc = subprocess.Popen(
                cmd, 
                env=get_pipeline_env(), 
                stdout=log_file, 
                stderr=subprocess.STDOUT
            )
            
            active_pipelines[pipeline_id] = {
                "process": proc,
                "port": requested_port,
                "base_config_path": base_config,
                "temp_config_path": temp_config_path,
                "patched_nvinfer_configs": patched_files,
                "log_file": log_file,
                "log_file_path": log_file_path,
                "start_time": time.time()
            }

            return jsonify({
                "message": "Pipeline spawned successfully",
                "pipeline_id": pipeline_id,
                "port": requested_port,
                "pipeline_api_url": f"http://localhost:{requested_port}/api/v1/",
                "log_file": log_file_path,
                "patched_configs": len(patched_files)
            }), 201

        except Exception as e:
            return jsonify({"error": str(e)}), 500


@app.route('/pipelines/<pipeline_id>', methods=['DELETE'])
def kill_pipeline(pipeline_id):
    """Kill a specific pipeline."""
    with pipelines_lock:
        if pipeline_id not in active_pipelines:
            return jsonify({"error": "Pipeline ID not found"}), 404

        info = active_pipelines[pipeline_id]
        proc = info['process']
        
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        
        cleanup_dead_pipeline(pipeline_id, info, reason="terminated by user")
        del active_pipelines[pipeline_id]
    
    return jsonify({"message": f"Pipeline {pipeline_id} terminated"}), 200


@app.route('/pipelines/kill-all', methods=['POST'])
def kill_all_pipelines():
    """Kill all active pipelines."""
    killed = []
    
    with pipelines_lock:
        for pid, info in list(active_pipelines.items()):
            proc = info['process']
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            cleanup_dead_pipeline(pid, info, reason="killed by kill-all")
            killed.append(pid)
        active_pipelines.clear()
    
    return jsonify({
        "message": f"Killed {len(killed)} pipeline(s)",
        "killed": killed
    })


def cleanup_all():
    """Clean up all resources on shutdown."""
    print("\n[Orchestrator] Shutting down...")
    shutdown_flag.set()
    
    with pipelines_lock:
        for pid, info in active_pipelines.items():
            print(f"[Orchestrator] Cleaning up pipeline {pid}...")
            
            if info['process'].poll() is None:
                print(f"[Orchestrator] Terminating pipeline {pid}...")
                info['process'].terminate()
                try:
                    info['process'].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"[Orchestrator] Force killing pipeline {pid}...")
                    info['process'].kill()
                    info['process'].wait()
            
            if 'log_file' in info and info['log_file']:
                try:
                    info['log_file'].close()
                except:
                    pass
            
            if 'temp_config_path' in info and os.path.exists(info['temp_config_path']):
                try:
                    os.remove(info['temp_config_path'])
                except Exception as e:
                    print(f"[Orchestrator] Warning: Could not remove {info['temp_config_path']}: {e}")
            
            for patched_file in info.get('patched_nvinfer_configs', []):
                if os.path.exists(patched_file):
                    try:
                        os.remove(patched_file)
                    except Exception as e:
                        print(f"[Orchestrator] Warning: Could not remove {patched_file}: {e}")
        
        active_pipelines.clear()
    
    # Clean up any orphaned temp files
    if os.path.exists(TEMP_CONFIG_DIR):
        for filename in os.listdir(TEMP_CONFIG_DIR):
            if filename.endswith(('.yml', '.yaml', '.txt')):
                filepath = os.path.join(TEMP_CONFIG_DIR, filename)
                try:
                    os.remove(filepath)
                    print(f"[Orchestrator] Removed orphaned config: {filepath}")
                except Exception as e:
                    print(f"[Orchestrator] Warning: Could not remove {filepath}: {e}")
    
    print("[Orchestrator] Cleanup complete. Logs preserved in:", LOG_DIR)


def signal_handler(signum, frame):
    """Handle shutdown signals."""
    print(f"\n[Orchestrator] Received signal {signum}")
    cleanup_all()
    sys.exit(0)


if __name__ == '__main__':
    ensure_dirs()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_all)
    
    health_thread = start_health_check_thread()
    
    print(f"==================================================")
    print(f" DEEPSTREAM ORCHESTRATOR v3 ON PORT {ORCHESTRATOR_PORT}")
    print(f"==================================================")
    print(f" Health Check Interval: {HEALTH_CHECK_INTERVAL}s")
    print(f" Temp Configs: {os.path.abspath(TEMP_CONFIG_DIR)}")
    print(f" Logs: {os.path.abspath(LOG_DIR)}")
    print(f"==================================================")
    print(f"")
    print(f" Features:")
    print(f"   - Auto-patches nvinferserver unique_id from YAML")
    print(f"   - Auto-patches operate_on_gie_id from YAML")
    print(f"   - Auto-patches operate_on_class_ids from YAML")
    print(f"")
    print(f" Endpoints:")
    print(f"   GET  /pipelines         - List active pipelines")
    print(f"   GET  /pipelines/health  - Manual health check")
    print(f"   POST /pipelines/spawn   - Spawn new pipeline")
    print(f"   DELETE /pipelines/<id>  - Kill specific pipeline")
    print(f"   POST /pipelines/kill-all - Kill all pipelines")
    print(f"")
    print(f"==================================================")
    
    app.run(host='0.0.0.0', port=ORCHESTRATOR_PORT, debug=False, threaded=True)