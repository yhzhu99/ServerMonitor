# backend.py
import asyncio
import os
import re
import socket
import subprocess
import sys # For sys.platform for macOS CPU info (though focus is Linux)
from typing import List, Dict, Any, Optional

import psutil
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

app = FastAPI(
    title="ServerMonitor API",
    description="API for monitoring Linux server status.",
    version="0.2.1"
)

# --- Helper Functions ---
def run_command(command: str) -> str:
    """Executes a shell command and returns its stdout."""
    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8', # Explicitly set encoding
            errors='replace' # Handle potential encoding errors gracefully
        )
        stdout, stderr = process.communicate(timeout=15) # Set a timeout for the command
        if process.returncode != 0:
            # Log stderr for debugging but don't necessarily raise an exception for all non-zero exits
            # Some commands might return non-zero on warnings or if specific devices aren't found.
            # Specific parsers should handle empty or partial output if critical.
            # print(f"Command '{command}' exited with code {process.returncode}. Stderr: {stderr.strip()}")
            return "" # Return empty string, let parsers decide how to handle
        return stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"Command '{command}' timed out after 15 seconds.")
        if 'process' in locals() and hasattr(process, 'kill'): # Ensure process was initialized
            try:
                process.kill()
                process.communicate() # Clean up
            except Exception as e_kill:
                print(f"Error trying to kill timed out process: {e_kill}")
        return ""
    except FileNotFoundError:
        # This occurs if the command itself (e.g., nvidia-smi) is not found.
        print(f"Command not found (FileNotFoundError for executable): {command.split()[0] if command else 'N/A'}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred while executing command '{command}': {e}")
        return ""

def get_gpu_info_list() -> List[Dict[str, Any]]:
    """
    Parses nvidia-smi output to get GPU details.
    Returns a list of dicts, each representing a GPU.
    """
    gpus_data = []
    # Query for essential GPU static and dynamic info.
    # Ensure no spaces around commas in --query-gpu arguments for reliability.
    smi_output_basic = run_command(
        "nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits"
    )

    if not smi_output_basic: # Handles command failure or no GPUs
        return gpus_data

    for line in smi_output_basic.strip().split("\n"):
        if not line.strip(): # Skip empty lines
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            # Ensure all expected parts are present before trying to access them
            if len(parts) < 6: # Min parts for id, uuid, name, util, mem.total, mem.used
                print(f"Skipping incomplete GPU line: '{line}'. Parts: {parts}")
                continue

            gpus_data.append({
                "id": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "utilization_gpu": float(parts[3]) if parts[3] != "[Not Supported]" else 0.0,
                "memory_total_mb": float(parts[4]) if parts[4] != "[Not Supported]" else 0.0,
                "memory_used_mb": float(parts[5]) if parts[5] != "[Not Supported]" else 0.0,
                "temperature_c": float(parts[6]) if len(parts) > 6 and parts[6] != "[Not Supported]" else None,
                "processes": [] # Placeholder, will be populated by get_gpu_processes if requested
            })
        except (IndexError, ValueError) as e:
            print(f"Error parsing GPU line '{line}': {e}. Parts: {parts}")
            continue
    return gpus_data

def get_gpu_processes(gpu_uuid: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Gets top N processes running on a specific GPU, sorted by memory."""
    # Query for processes on all GPUs, then filter by UUID.
    # This is often more efficient than querying per GPU if multiple GPUs are checked.
    smi_processes_output = run_command(
        f"nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits"
    )

    gpu_processes_details = []
    if not smi_processes_output:
        return gpu_processes_details

    for line in smi_processes_output.strip().split("\n"):
        if not line.strip():
            continue
        try:
            proc_gpu_uuid, pid_str, proc_name_full, mem_used_str = [p.strip() for p in line.split(",")]

            if proc_gpu_uuid == gpu_uuid: # Filter for the target GPU
                pid = int(pid_str)
                # Handle "[Not Supported]" for memory usage
                mem_used = float(mem_used_str) if mem_used_str != "[Not Supported]" else 0.0

                user = "N/A"
                try:
                    # Check if process exists before trying to get username to avoid stale PIDs
                    if psutil.pid_exists(pid):
                        process = psutil.Process(pid)
                        user = process.username()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    # Process might have ended, or insufficient permissions to get username
                    pass
                except Exception as e_psutil:
                    # Catch other potential psutil errors for this specific process
                    print(f"Error getting psutil info for PID {pid}: {e_psutil}")


                gpu_processes_details.append({
                    "pid": pid,
                    "name": os.path.basename(proc_name_full), # Get just the process name
                    "user": user,
                    "gpu_memory_mb": mem_used
                })
        except (IndexError, ValueError) as e:
            print(f"Error parsing GPU process line '{line}': {e}")
            continue

    # Sort processes by GPU memory used in descending order and take top_n
    gpu_processes_details.sort(key=lambda x: x["gpu_memory_mb"], reverse=True)
    return gpu_processes_details[:top_n]


def get_cpu_config_info() -> Dict[str, Any]:
    """Gets static CPU configuration information."""
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False) # None if undetermined

    model_name = "N/A"
    if sys.platform == "linux":
        try:
            # Reading /proc/cpuinfo is a common way to get CPU model on Linux
            with open("/proc/cpuinfo", "r", encoding='utf-8') as f:
                for line in f:
                    if "model name" in line:
                        model_name = line.split(":", 1)[1].strip()
                        break
        except Exception as e:
            print(f"Could not read /proc/cpuinfo for CPU model: {e}")
    elif sys.platform == "darwin": # macOS specific command
        model_name_raw = run_command("sysctl -n machdep.cpu.brand_string")
        if model_name_raw: model_name = model_name_raw
    # For other OS, psutil doesn't directly provide CPU model name easily.

    return {
        "model_name": model_name if model_name else "N/A", # Ensure not None
        "physical_cores": cpu_count_physical if cpu_count_physical else 0, # Default to 0 if None
        "logical_cores": cpu_count_logical if cpu_count_logical else 0, # Default to 0 if None
    }

def get_top_cpu_processes(top_n: int = 3) -> List[Dict[str, Any]]:
    """Gets top N CPU consuming processes."""
    processes_data = []

    # Attributes to fetch. cpu_percent needs to be called.
    # The initial psutil.cpu_percent(interval=X) in the endpoint primes this.
    # Here, p.cpu_percent(interval=None) gets usage since the last call.

    procs_info_list = []
    for p in psutil.process_iter():
        try:
            with p.oneshot(): # Efficiently gather multiple attributes for a process
                # Call cpu_percent with interval=None; it relies on a previous system-wide call.
                # If no prior system-wide call with an interval, this might return 0 or inaccurate on first call.
                # The endpoint /api/cpu/top_processes ensures a system-wide call is made first.
                cpu_p = p.cpu_percent(interval=None)

                # Filter out processes with negligible CPU usage early if desired
                if cpu_p <= 0.01 and p.pid != 0: # Keep idle process (PID 0 on some systems) if needed, or filter
                    continue

                p_info = {
                    "pid": p.pid,
                    "name": p.name(),
                    "username": p.username(),
                    "cpu_percent": cpu_p,
                    "memory_percent": p.memory_percent()
                }
                procs_info_list.append(p_info)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass # Process ended or permissions issue
        except Exception as e:
            # print(f"Skipping process {getattr(p, 'pid', 'UNKNOWN PID')} due to error: {e}")
            pass

    # Sort by CPU percentage in descending order
    procs_info_list.sort(key=lambda x: x['cpu_percent'], reverse=True)

    # Format and return top N
    for p_info_sorted in procs_info_list[:top_n]:
        processes_data.append({
            "pid": p_info_sorted['pid'],
            "name": p_info_sorted['name'],
            "user": p_info_sorted['username'],
            "cpu_percent": round(p_info_sorted['cpu_percent'], 2),
            "memory_percent": round(p_info_sorted['memory_percent'], 2)
        })
    return processes_data

# --- Pydantic Models for API Responses ---
class GPUProcessInfoModel(BaseModel):
    pid: int
    name: str
    user: str
    gpu_memory_mb: float

class BaseGPUInfoModel(BaseModel): # Static config part of GPU
    id: int
    uuid: str
    name: str
    memory_total_mb: float

class GPUStatusInfoModel(BaseGPUInfoModel): # Dynamic status part of GPU
    utilization_gpu: float
    memory_used_mb: float
    temperature_c: Optional[float] = None
    processes: List[GPUProcessInfoModel] = []

class CPUConfigInfoModel(BaseModel):
    model_name: str
    physical_cores: int
    logical_cores: int

class ServerConfig(BaseModel):
    server_ip: str
    server_name: str
    gpus: List[BaseGPUInfoModel] # Use base model for config
    cpus: CPUConfigInfoModel
    memory_total_gb: float

class CPUProcessDetailModel(BaseModel):
    pid: int
    name: str
    user: str
    cpu_percent: float
    memory_percent: float

class ResourceStatus(BaseModel):
    cpu_utilization_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_utilization_percent: float
    gpus: List[GPUStatusInfoModel] # Use status model for real-time status


# --- API Endpoints ---
@app.get("/api/server/config", response_model=ServerConfig)
async def get_server_configuration():
    # Attempt to get a non-loopback IP address
    server_ip = "127.0.0.1" # Default
    try:
        # This trick often works to get an IP used for outbound connections
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1) # Non-blocking
        s.connect(("8.8.8.8", 80)) # Doesn't actually send data
        server_ip = s.getsockname()[0]
        s.close()
    except Exception:
        # Fallback if the above fails (e.g., no internet, firewall)
        try:
            # Get hostname, then resolve it. Might still be loopback if not configured.
            hostname = socket.gethostname()
            server_ip = socket.gethostbyname(hostname)
        except socket.gaierror:
            # If hostname resolution fails, stick to default or try other methods
            pass # server_ip remains "127.0.0.1" or a previously found value

    server_name = socket.gethostname()

    # For config, we only need static GPU info
    gpu_static_info_list_raw = get_gpu_info_list() # This fetches current status too, but we only use parts
    gpu_config_models = []
    for gpu_data in gpu_static_info_list_raw:
        gpu_config_models.append(BaseGPUInfoModel(
            id=gpu_data['id'],
            uuid=gpu_data['uuid'],
            name=gpu_data['name'],
            memory_total_mb=gpu_data['memory_total_mb']
        ))

    cpu_config = get_cpu_config_info()
    mem_info = psutil.virtual_memory()

    return ServerConfig(
        server_ip=server_ip,
        server_name=server_name,
        gpus=gpu_config_models,
        cpus=CPUConfigInfoModel(**cpu_config),
        memory_total_gb=round(mem_info.total / (1024**3), 2)
    )

@app.get("/api/server/status", response_model=ResourceStatus)
async def get_realtime_status(top_n_gpu_processes: int = Query(3, ge=0, le=10, description="Number of top processes per GPU")):
    # System-wide CPU utilization. interval=0.1 makes it slightly blocking but more accurate than non-blocking.
    # This call also helps prime per-process cpu_percent calls if they use interval=None.
    cpu_util = psutil.cpu_percent(interval=0.1)

    ram_info = psutil.virtual_memory()

    # Get current GPU status including utilization, memory, temp
    gpus_current_status_raw = get_gpu_info_list()
    detailed_gpus_status_models = []

    for gpu_stat_raw in gpus_current_status_raw:
        gpu_procs_pydantic = []
        if top_n_gpu_processes > 0:
             # Fetch processes specifically for this GPU's UUID
             gpu_procs_raw_data = get_gpu_processes(gpu_stat_raw["uuid"], top_n=top_n_gpu_processes)
             gpu_procs_pydantic = [GPUProcessInfoModel(**proc) for proc in gpu_procs_raw_data]

        detailed_gpus_status_models.append(GPUStatusInfoModel(
            id=gpu_stat_raw['id'],
            uuid=gpu_stat_raw['uuid'],
            name=gpu_stat_raw['name'],
            memory_total_mb=gpu_stat_raw['memory_total_mb'], # From BaseGPUInfoModel
            utilization_gpu=gpu_stat_raw['utilization_gpu'],
            memory_used_mb=gpu_stat_raw['memory_used_mb'],
            temperature_c=gpu_stat_raw.get('temperature_c'), # Optional
            processes=gpu_procs_pydantic
        ))

    return ResourceStatus(
        cpu_utilization_percent=cpu_util,
        ram_used_gb=round(ram_info.used / (1024**3), 2),
        ram_total_gb=round(ram_info.total / (1024**3), 2),
        ram_utilization_percent=ram_info.percent,
        gpus=detailed_gpus_status_models
    )

@app.get("/api/cpu/top_processes", response_model=List[CPUProcessDetailModel])
async def get_cpu_top_processes_endpoint(n: int = Query(3, ge=1, le=20, description="Number of top CPU processes")):
    # Crucial: Call system-wide psutil.cpu_percent() with an interval BEFORE
    # calling per-process cpu_percent(interval=None).
    # This establishes the time window for per-process calculations.
    psutil.cpu_percent(interval=0.1) # Blocking call for 0.1s.
    # A very short sleep might sometimes help ensure the interval is distinct if calls are extremely rapid,
    # but psutil's interval handling should generally cover this.
    # await asyncio.sleep(0.01) # Optional, usually not needed if interval above is sufficient.

    top_processes_raw = get_top_cpu_processes(top_n=n)
    # Convert dicts to Pydantic models for response validation and schema generation
    return [CPUProcessDetailModel(**proc) for proc in top_processes_raw]


if __name__ == "__main__":
    import uvicorn
    print("Starting ServerMonitor API backend...")
    print("Ensure 'nvidia-smi' command is available and in PATH if NVIDIA GPUs are to be monitored.")
    print("Ensure 'psutil' can access necessary system information (permissions).")
    print("Backend will be available at http://0.0.0.0:8801 (or the port you configure).")

    # To run: uvicorn backend:app --reload --host 0.0.0.0 --port 8801
    # The host 0.0.0.0 makes it accessible from other machines on the network.
    # The port can be configured as needed.
    uvicorn.run(app, host="0.0.0.0", port=8801)