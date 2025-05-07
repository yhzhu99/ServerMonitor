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
    version="0.2.0" # Updated version
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
            encoding='utf-8', # Be explicit about encoding
            errors='replace' # Handles potential decoding errors
        )
        stdout, stderr = process.communicate(timeout=20) # Slightly increased timeout
        if process.returncode != 0:
            print(f"Command '{command}' exited with code {process.returncode}. Stderr: {stderr.strip()}")
            return "" # Return empty string on error, specific handling below if needed
        return stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"Command '{command}' timed out.")
        # Ensure process is killed if it exists and timed out
        if 'process' in locals() and hasattr(process, 'kill'):
            try:
                process.kill()
                process.communicate() # Clean up zombie
            except Exception as e_kill:
                print(f"Error trying to kill timed out process: {e_kill}")
        return ""
    except FileNotFoundError:
        print(f"Command not found (FileNotFoundError): {command.split()[0] if command else 'N/A'}")
        return ""
    except Exception as e:
        print(f"Error executing command '{command}': {e}")
        return ""

def get_gpu_info_list() -> List[Dict[str, Any]]:
    """
    Parses nvidia-smi output to get GPU details.
    Returns a list of dicts, each representing a GPU.
    """
    gpus_data = []
    # Query for basic GPU info
    smi_output_basic = run_command(
        "nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits"
    )

    if not smi_output_basic: # Handles command error or no GPUs
        return gpus_data

    for line in smi_output_basic.strip().split("\n"):
        if not line.strip(): # Skip empty lines
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            gpus_data.append({
                "id": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "utilization_gpu": float(parts[3]),
                "memory_total_mb": float(parts[4]),
                "memory_used_mb": float(parts[5]),
                "temperature_c": float(parts[6]) if len(parts) > 6 and parts[6] != "[Not Supported]" else None,
                "processes": [] # Will be populated by get_gpu_processes if requested
            })
        except (IndexError, ValueError) as e:
            print(f"Error parsing GPU line '{line}': {e}. Parts: {parts}")
            continue
    return gpus_data

def get_gpu_processes(gpu_uuid: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Gets top N processes running on a specific GPU, sorted by memory."""
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
            uuid, pid_str, proc_name_full, mem_used_str = [p.strip() for p in line.split(",")]
            if uuid == gpu_uuid:
                pid = int(pid_str)
                mem_used = float(mem_used_str) # nvidia-smi reports in MiB for used_gpu_memory

                user = "N/A"
                try:
                    process = psutil.Process(pid)
                    user = process.username()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    # Process might have ended, or permission issues
                    pass

                gpu_processes_details.append({
                    "pid": pid,
                    "name": os.path.basename(proc_name_full), # Get basename of process
                    "user": user,
                    "gpu_memory_mb": mem_used
                })
        except (IndexError, ValueError) as e:
            print(f"Error parsing GPU process line '{line}': {e}")
            continue

    # Sort by GPU memory used and take top N
    gpu_processes_details.sort(key=lambda x: x["gpu_memory_mb"], reverse=True)
    return gpu_processes_details[:top_n]


def get_cpu_config_info() -> Dict[str, Any]:
    """Gets static CPU configuration information."""
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)

    model_name = "N/A"
    if sys.platform == "linux":
        try:
            with open("/proc/cpuinfo", "r", encoding='utf-8') as f:
                for line in f:
                    if "model name" in line:
                        model_name = line.split(":", 1)[1].strip()
                        break
        except Exception as e:
            print(f"Could not read /proc/cpuinfo for CPU model: {e}")
    elif sys.platform == "darwin": # macOS fallback
        model_name = run_command("sysctl -n machdep.cpu.brand_string")
    # Add other platform checks if necessary, e.g., Windows using wmic

    return {
        "model_name": model_name if model_name else "N/A",
        "physical_cores": cpu_count_physical if cpu_count_physical else 0,
        "logical_cores": cpu_count_logical if cpu_count_logical else 0,
    }

def get_top_cpu_processes(top_n: int = 3) -> List[Dict[str, Any]]:
    """
    Gets top N CPU consuming processes.
    Note: psutil.cpu_percent() for a process should be called after a system-wide
    psutil.cpu_percent(interval=...) or after a prior call to the process's
    cpu_percent(interval=None) to get meaningful non-zero results.
    The priming call is expected to be in the endpoint.
    """
    processes_data = []
    # Attributes to fetch. 'cpu_percent' and 'memory_percent' are methods, handled below.
    attrs = ['pid', 'name', 'username', 'memory_percent']
    for proc in psutil.process_iter(attrs):
        try:
            pinfo = proc.info # This contains pid, name, username, memory_percent
            # Call cpu_percent() explicitly. interval=None uses system-wide time since last call.
            # This assumes the endpoint has primed psutil.cpu_percent()
            current_cpu_percent = proc.cpu_percent(interval=None)

            if current_cpu_percent > 0.0: # Consider processes with some CPU usage
                processes_data.append({
                    "pid": pinfo['pid'],
                    "name": pinfo['name'],
                    "user": pinfo['username'],
                    "cpu_percent": round(current_cpu_percent / psutil.cpu_count(logical=True), 2), # Normalize by logical core count
                    "memory_percent": round(pinfo['memory_percent'], 2)
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass # Process might have ended or restricted
        except Exception as e:
            # Catch any other psutil errors for a specific process
            # print(f"Skipping process {getattr(proc, 'pid', 'UNKNOWN')} due to error: {e}")
            pass

    processes_data.sort(key=lambda x: x["cpu_percent"], reverse=True)
    return processes_data[:top_n]

# --- Pydantic Models for API Responses ---
class GPUProcessInfo(BaseModel):
    pid: int
    name: str
    user: str
    gpu_memory_mb: float

class GPUInfo(BaseModel):
    id: int
    uuid: str
    name: str
    utilization_gpu: float
    memory_total_mb: float
    memory_used_mb: float
    temperature_c: Optional[float] = None
    processes: List[GPUProcessInfo] = [] # Updated to use specific model

class CPUConfigInfo(BaseModel): # Renamed from CPUInfo for clarity
    model_name: str
    physical_cores: int
    logical_cores: int

class ServerConfig(BaseModel):
    server_ip: str
    server_name: str
    gpus: List[GPUInfo] # Static part: name, total_mem, id, uuid. Other fields are placeholders.
    cpus: CPUConfigInfo
    memory_total_gb: float

class ResourceStatus(BaseModel):
    cpu_utilization_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_utilization_percent: float
    gpus: List[GPUInfo] # Dynamic part: utilization, mem_used, temp, processes

class CPUProcessInfo(BaseModel): # For top CPU processes
    pid: int
    name: str
    user: str
    cpu_percent: float
    memory_percent: float


# --- API Endpoints ---
@app.get("/api/server/config", response_model=ServerConfig)
async def get_server_configuration():
    """Provides static configuration of the server."""
    try:
        # Try to get a non-loopback IP, might not be the primary one in complex setups
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1) # Avoid blocking
        s.connect(("8.8.8.8", 80)) # Connect to a known external server (doesn't send data)
        server_ip = s.getsockname()[0]
        s.close()
    except Exception:
        # Fallback: gethostname might resolve to loopback or an internal IP
        try:
            server_ip = socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            server_ip = "127.0.0.1" # Ultimate fallback

    server_name = socket.gethostname()

    gpu_static_info_list = []
    raw_gpus_config = get_gpu_info_list() # Gets all info, we extract static parts
    for gpu_data in raw_gpus_config:
        gpu_static_info_list.append(GPUInfo(
            id=gpu_data['id'],
            uuid=gpu_data['uuid'],
            name=gpu_data['name'],
            utilization_gpu=0.0, # Placeholder for config
            memory_total_mb=gpu_data['memory_total_mb'],
            memory_used_mb=0.0, # Placeholder for config
            temperature_c=None, # Placeholder for config
            processes=[]
        ))

    cpu_config = get_cpu_config_info()
    mem_info = psutil.virtual_memory()

    return ServerConfig(
        server_ip=server_ip,
        server_name=server_name,
        gpus=gpu_static_info_list,
        cpus=cpu_config,
        memory_total_gb=round(mem_info.total / (1024**3), 2)
    )

@app.get("/api/server/status", response_model=ResourceStatus)
async def get_realtime_status(top_n_gpu_processes: int = Query(3, ge=0, le=10, description="Number of top processes per GPU")):
    """Provides real-time resource utilization."""
    # Prime psutil.cpu_percent for subsequent calls if interval is None
    cpu_util = psutil.cpu_percent(interval=0.1) # Non-blocking, captures usage over 0.1s
    ram_info = psutil.virtual_memory()

    gpus_current_status = get_gpu_info_list() # Gets current utilization, memory, temp

    detailed_gpus_status = []
    for gpu_stat in gpus_current_status:
        gpu_procs = []
        if top_n_gpu_processes > 0:
             gpu_procs_raw = get_gpu_processes(gpu_stat["uuid"], top_n=top_n_gpu_processes)
             # Convert to Pydantic model
             gpu_procs = [GPUProcessInfo(**proc) for proc in gpu_procs_raw]


        detailed_gpus_status.append(GPUInfo(
            id=gpu_stat['id'],
            uuid=gpu_stat['uuid'],
            name=gpu_stat['name'],
            utilization_gpu=gpu_stat['utilization_gpu'],
            memory_total_mb=gpu_stat['memory_total_mb'],
            memory_used_mb=gpu_stat['memory_used_mb'],
            temperature_c=gpu_stat.get('temperature_c'), # Already optional from get_gpu_info_list
            processes=gpu_procs
        ))

    return ResourceStatus(
        cpu_utilization_percent=cpu_util,
        ram_used_gb=round(ram_info.used / (1024**3), 2),
        ram_total_gb=round(ram_info.total / (1024**3), 2),
        ram_utilization_percent=ram_info.percent,
        gpus=detailed_gpus_status
    )

@app.get("/api/cpu/top_processes", response_model=List[CPUProcessInfo])
async def get_cpu_top_processes_endpoint(n: int = Query(3, ge=1, le=20, description="Number of top CPU processes")):
    """Gets top N CPU consuming processes."""
    # Prime psutil.cpu_percent() for the process specific calls in get_top_cpu_processes
    # This sets the system-wide "last call time".
    psutil.cpu_percent(interval=0.1) # Small blocking call to establish baseline

    top_processes_raw = get_top_cpu_processes(top_n=n)
    # Convert to Pydantic model list
    return [CPUProcessInfo(**proc) for proc in top_processes_raw]


if __name__ == "__main__":
    import uvicorn
    # For development, run with: uvicorn backend:app --reload --host 0.0.0.0 --port 8801
    # The host 0.0.0.0 makes it accessible from other machines on the network.
    uvicorn.run(app, host="0.0.0.0", port=8801)