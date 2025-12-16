import subprocess
import os
import signal
import sys
import yaml
import time
import uuid
from flask import Flask, request, jsonify

# ================= CONFIGURATION =================
BINARY_PATH = "/opt/nvidia/deepstream/deepstream/sources/apps/sample_apps/deepstream-server/deepstream-server-app"
TEMP_CONFIG_DIR = "./temp_configs"
LOG_DIR = "./logs"
ORCHESTRATOR_PORT = 5000
# =================================================

app = Flask(__name__)

# Active pipelines dictionary
active_pipelines = {}

def ensure_dirs():
    if not os.path.exists(TEMP_CONFIG_DIR):
        os.makedirs(TEMP_CONFIG_DIR)
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)

def generate_config(base_config_path, port):
    if not os.path.exists(base_config_path):
        raise FileNotFoundError(f"Config file not found: {base_config_path}")

    with open(base_config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'server-app-ctx' not in config:
        config['server-app-ctx'] = {}
    
    # === FORCE NETWORK SETTINGS ===
    # 1. Force the correct port
    config['server-app-ctx']['httpPort'] = str(port)
    # 2. Force binding to 0.0.0.0 so localhost/127.0.0.1 works
    config['server-app-ctx']['httpIp'] = "0.0.0.0"
    # 3. Explicitly enable the server (in case the base config has enable: 0)
    config['server-app-ctx']['enable'] = 1
    # ==============================

    if 'rest-server' in config:
        # config['rest-server']['within_multiurisrcbin'] = 0
        # config['rest-server']['port'] = port
        # Ensure the rest-server group itself is enabled so the parsing logic works
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
    
    # 1. Set specific flags
    env["NVDS_MULTIURI_ALLOW_MIXED_PROTOCOL"] = "1"
    env["NVDS_MULTIURI_ALLOW_EMPTY"] = "1"
    env["GIO_MODULE_DIR"] = "/nonexistent"
    env["NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT"] = "1"
    env["NVDS_ENABLE_DEBUG"] = "1"
    # env["GST_DEBUG"] = "3" # comment if needed
    
    # 2. Update GST_PLUGIN_PATH
    ds_plugin_path = "/opt/nvidia/deepstream/deepstream-8.0/lib/gst-plugins"
    current_gst_path = env.get("GST_PLUGIN_PATH", "")
    if ds_plugin_path not in current_gst_path:
        env["GST_PLUGIN_PATH"] = f"{ds_plugin_path}:{current_gst_path}"

    # 3. Update LD_LIBRARY_PATH
    # Note: Order matters. Prepending custom workspace libs.
    extra_libs = "/workspace/lib:/opt/nvidia/deepstream/deepstream-8.0/lib"
    current_ld_path = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{extra_libs}:{current_ld_path}"

    return env

@app.route('/pipelines', methods=['GET'])
def list_pipelines():
    status_list = []
    dead_ids = []
    
    for pid, info in active_pipelines.items():
        # Check if process is still running
        ret_code = info['process'].poll()
        
        if ret_code is not None:
            # Process died
            dead_ids.append(pid)
            print(f"[Orchestrator] Pipeline {pid} DIED with code {ret_code}. Check logs.")
        else:
            status_list.append({
                "id": pid,
                "port": info['port'],
                "config": info['base_config_path'],
                "pid": info['process'].pid
            })
    
    # Cleanup dead processes
    for died in dead_ids:
        del active_pipelines[died]

    return jsonify({"active_pipelines": status_list, "count": len(status_list)})

@app.route('/pipelines/spawn', methods=['POST'])
def spawn_pipeline():
    data = request.json
    base_config = data.get('config_path')
    requested_port = data.get('port')

    if not base_config:
        return jsonify({"error": "config_path is required"}), 400

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
        log_file = open(log_file_path, "w", buffering=1)  # Line buffered

        cmd = [BINARY_PATH, temp_config_path]
        print(f"[Orchestrator] Spawning: {' '.join(cmd)}")
        print(f"[Orchestrator] Logging to: {log_file_path}")
        
        # Spawn with custom ENV and Log Redirection
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
            "log_file": log_file # Keep handle to close later if needed
        }

        return jsonify({
            "message": "Pipeline spawned successfully",
            "pipeline_id": pipeline_id,
            "pipeline_api_url": f"http://localhost:{requested_port}/api/v1/",
            "log_file": log_file_path
        }), 201

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/pipelines/<pipeline_id>', methods=['DELETE'])
def kill_pipeline(pipeline_id):
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
    
    # Close log file handle
    if 'log_file' in info:
        info['log_file'].close()

    # Cleanup temp file
    if os.path.exists(info['temp_config_path']):
        os.remove(info['temp_config_path'])
        
    del active_pipelines[pipeline_id]
    return jsonify({"message": f"Pipeline {pipeline_id} terminated"}), 200

def cleanup_all():
    print("\n[Orchestrator] Shutting down all pipelines...")
    for pid, info in active_pipelines.items():
        if info['process'].poll() is None:
            info['process'].terminate()
            info['process'].wait()
        if 'log_file' in info:
            info['log_file'].close()

        # Cleanup temp file
    if os.path.exists(info['temp_config_path']):
        os.remove(info['temp_config_path'])

if __name__ == '__main__':
    ensure_dirs()
    signal.signal(signal.SIGINT, lambda s, f: (cleanup_all(), sys.exit(0)))
    
    print(f"==================================================")
    print(f" DEEPSTREAM ORCHESTRATOR RUNNING ON PORT {ORCHESTRATOR_PORT}")
    print(f"==================================================")
    app.run(host='0.0.0.0', port=ORCHESTRATOR_PORT, debug=False)