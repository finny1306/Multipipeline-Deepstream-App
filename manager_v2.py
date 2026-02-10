import subprocess
import os
import signal
import sys
import yaml
import time
import uuid
import threading
import atexit
from flask import Flask, request, jsonify

# ================= CONFIGURATION =================
BINARY_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-server/deepstream-server-app"
TEMP_CONFIG_DIR = "./temp_configs"
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


def generate_config(base_config_path, port):
    """Generate a temporary config file with the specified port."""
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
    new_config_name = f"pipeline_{pipeline_id}_port_{port}.yml"
    new_config_path = os.path.join(TEMP_CONFIG_DIR, new_config_name)

    with open(new_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    return pipeline_id, new_config_path


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
    #env["USE_NEW_NVSTREAMMUX"] = "yes"
    #env["GST_DEBUG"] = "3" # comment if needed
    
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
    Closes log file handle and removes temp config.
    """
    print(f"[Orchestrator] Pipeline {pipeline_id} {reason}. Cleaning up...")
    
    # Close log file handle
    if 'log_file' in info and info['log_file']:
        try:
            info['log_file'].close()
        except Exception as e:
            print(f"[Orchestrator] Warning: Could not close log file for {pipeline_id}: {e}")
    
    # Remove temp config file
    if 'temp_config_path' in info and os.path.exists(info['temp_config_path']):
        try:
            os.remove(info['temp_config_path'])
            print(f"[Orchestrator] Removed temp config: {info['temp_config_path']}")
        except Exception as e:
            print(f"[Orchestrator] Warning: Could not remove temp config for {pipeline_id}: {e}")
    
    # Log the cleanup
    print(f"[Orchestrator] Pipeline {pipeline_id} cleaned up. Port {info.get('port', 'unknown')} is now available.")


def check_pipeline_health():
    """
    Check all active pipelines and clean up dead ones.
    Returns list of cleaned up pipeline IDs.
    """
    dead_ids = []
    
    with pipelines_lock:
        for pid, info in list(active_pipelines.items()):
            ret_code = info['process'].poll()
            
            if ret_code is not None:
                # Process has exited
                dead_ids.append(pid)
                cleanup_dead_pipeline(pid, info, reason=f"exited with code {ret_code}")
        
        # Remove dead pipelines from active list
        for pid in dead_ids:
            del active_pipelines[pid]
    
    return dead_ids


def health_check_worker():
    """
    Background worker that periodically checks pipeline health.
    """
    print(f"[Orchestrator] Health check worker started (interval: {HEALTH_CHECK_INTERVAL}s)")
    
    while not shutdown_flag.is_set():
        # Wait for the interval or until shutdown
        shutdown_flag.wait(timeout=HEALTH_CHECK_INTERVAL)
        
        if shutdown_flag.is_set():
            break
        
        # Check pipeline health
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
    # First, run a health check to clean up any dead pipelines
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
    """
    Endpoint to manually trigger a health check.
    Returns list of cleaned up pipelines.
    """
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

    # Run health check first to free up any dead pipeline ports
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
            pipeline_id, temp_config_path = generate_config(base_config, requested_port)
            
            # Prepare Log File
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
                "log_file": log_file,
                "log_file_path": log_file_path,
                "start_time": time.time()
            }

            return jsonify({
                "message": "Pipeline spawned successfully",
                "pipeline_id": pipeline_id,
                "port": requested_port,
                "pipeline_api_url": f"http://localhost:{requested_port}/api/v1/",
                "log_file": log_file_path
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
        
        # Terminate the process if still running
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        
        # Clean up resources
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
    """
    Clean up all resources on shutdown.
    - Terminates all running pipelines
    - Closes all log file handles
    - Removes all temp config files
    - Keeps log files for debugging
    """
    print("\n[Orchestrator] Shutting down...")
    
    # Signal the health check thread to stop
    shutdown_flag.set()
    
    with pipelines_lock:
        for pid, info in active_pipelines.items():
            print(f"[Orchestrator] Cleaning up pipeline {pid}...")
            
            # Terminate process if running
            if info['process'].poll() is None:
                print(f"[Orchestrator] Terminating pipeline {pid}...")
                info['process'].terminate()
                try:
                    info['process'].wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"[Orchestrator] Force killing pipeline {pid}...")
                    info['process'].kill()
                    info['process'].wait()
            
            # Close log file handle
            if 'log_file' in info and info['log_file']:
                try:
                    info['log_file'].close()
                except:
                    pass
            
            # Remove temp config file
            if 'temp_config_path' in info and os.path.exists(info['temp_config_path']):
                try:
                    os.remove(info['temp_config_path'])
                    print(f"[Orchestrator] Removed temp config: {info['temp_config_path']}")
                except Exception as e:
                    print(f"[Orchestrator] Warning: Could not remove {info['temp_config_path']}: {e}")
        
        active_pipelines.clear()
    
    # Clean up any orphaned temp config files
    if os.path.exists(TEMP_CONFIG_DIR):
        for filename in os.listdir(TEMP_CONFIG_DIR):
            if filename.endswith('.yml') or filename.endswith('.yaml'):
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
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup on exit
    atexit.register(cleanup_all)
    
    # Start the health check background thread
    health_thread = start_health_check_thread()
    
    print(f"==================================================")
    print(f" DEEPSTREAM ORCHESTRATOR RUNNING ON PORT {ORCHESTRATOR_PORT}")
    print(f"==================================================")
    print(f" Health Check Interval: {HEALTH_CHECK_INTERVAL}s")
    print(f" Temp Configs: {os.path.abspath(TEMP_CONFIG_DIR)}")
    print(f" Logs: {os.path.abspath(LOG_DIR)}")
    print(f"==================================================")
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
