import psutil
import pynvml
from datetime import datetime

# Configuration
GPU_INDEX = 0    

# Initialize NVML
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(GPU_INDEX)


def performance_metrics():
    """Return a dict with current metrics (memory values in GB)."""
    ts = datetime.utcnow().isoformat()
    # GPU basic utilization and memory
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    power_mw = None
    temp_c = None
    try:
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # milliwatts
    except pynvml.NVMLError:
        power_mw = None
    try:
        temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except pynvml.NVMLError:
        temp_c = None

    # GPU memory utilization percent (computed)
    gpu_mem_used = mem_info.used
    gpu_mem_total = mem_info.total
    gpu_mem_util_pct = (gpu_mem_used / gpu_mem_total * 100.0) if gpu_mem_total else None

    # Convert bytes to GB
    def _bytes_to_gb(b):
        return round(b / (1024 ** 3), 3) if b is not None else None

    gpu_mem_used_gb = _bytes_to_gb(gpu_mem_used)
    gpu_mem_total_gb = _bytes_to_gb(gpu_mem_total)

    # CPU and system memory
    cpu_percent = psutil.cpu_percent(interval=None)
    cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
    virtual_mem = psutil.virtual_memory()

    system_mem_total_gb = _bytes_to_gb(virtual_mem.total)
    system_mem_used_gb = _bytes_to_gb(virtual_mem.used)

    # Optional: per-process GPU usage (requires NVML process query)
    processes = []
    try:
        procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        for p in procs:
            processes.append({
                "pid": p.pid,
                "usedGpuMemory": getattr(p, "usedGpuMemory", None)
            })
    except pynvml.NVMLError:
        processes = []

    return {
        "timestamp_utc": ts,
        "gpu_index": GPU_INDEX,
        "gpu_util_sm_pct": util.gpu,            # SM/GPU utilization percent
        "gpu_mem_util_pct": gpu_mem_util_pct,
        "gpu_mem_used_gb": gpu_mem_used_gb,
        "gpu_mem_total_gb": gpu_mem_total_gb,
        "gpu_power_mw": power_mw,
        "gpu_temp_c": temp_c,
        "cpu_total_pct": cpu_percent,
        "cpu_per_core_pct": ";".join(f"{v:.1f}" for v in cpu_per_core),
        "system_mem_total_gb": system_mem_total_gb,
        "system_mem_used_gb": system_mem_used_gb,
        "system_mem_percent": virtual_mem.percent,
    }

if __name__ == "__main__":
    import json
    metrics = performance_metrics()
    print(json.dumps(metrics, indent=2))
