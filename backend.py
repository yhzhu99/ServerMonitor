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
    version="0.2.1" # Updated version
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
            encoding='utf-8',
            errors='replace'
        )
        stdout, stderr = process.communicate(timeout=15) # Adjusted timeout
        if process.returncode != 0:
            # Log stderr for debugging but don't necessarily raise an exception here
            # Some commands might return non-zero on warnings or if specific devices aren't found
            # print(f"Command '{command}' exited with code {process.returncode}. Stderr: {stderr.strip()}")
            return "" # Let specific parsers handle empty output
        return stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"Command '{command}' timed out.")
        if 'process' in locals() and hasattr(process, 'kill'):
            try:
                process.kill()
                process.communicate()
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
    smi_output_basic = run_command(
        "nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits"
    )

    if not smi_output_basic:
        return gpus_data

    for line in smi_output_basic.strip().split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            gpus_data.append({
                "id": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "utilization_gpu": float(parts[3]) if parts[3] != "[Not Supported]" else 0.0,
                "memory_total_mb": float(parts[4]) if parts[4] != "[Not Supported]" else 0.0,
                "memory_used_mb": float(parts[5]) if parts[5] != "[Not Supported]" else 0.0,
                "temperature_c": float(parts[6]) if len(parts) > 6 and parts[6] != "[Not Supported]" else None,
                "processes": [] # Placeholder, populated if requested
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
                mem_used = float(mem_used_str) if mem_used_str != "[Not Supported]" else 0.0

                user = "N/A"
                try:
                    # Check if process exists before trying to get username
                    if psutil.pid_exists(pid):
                        process = psutil.Process(pid)
                        user = process.username()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass # Process might have ended, or permission issues

                gpu_processes_details.append({
                    "pid": pid,
                    "name": os.path.basename(proc_name_full),
                    "user": user,
                    "gpu_memory_mb": mem_used
                })
        except (IndexError, ValueError) as e:
            print(f"Error parsing GPU process line '{line}': {e}")
            continue

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
    elif sys.platform == "darwin":
        model_name = run_command("sysctl -n machdep.cpu.brand_string")

    return {
        "model_name": model_name if model_name else "N/A",
        "physical_cores": cpu_count_physical if cpu_count_physical else 0,
        "logical_cores": cpu_count_logical if cpu_count_logical else 0,
    }

def get_top_cpu_processes(top_n: int = 3) -> List[Dict[str, Any]]:
    """Gets top N CPU consuming processes."""
    processes_data = []
    attrs = ['pid', 'name', 'username', 'cpu_percent', 'memory_percent'] # cpu_percent will be method call

    # Iterate over processes, getting necessary info
    # We need to call cpu_percent() twice for it to be accurate for a specific process
    # The first call (system wide) is done in the endpoint.
    # The second call (per process) here gets the utilization since the last call.
    procs = []
    for p in psutil.process_iter():
        try:
            with p.oneshot(): # Efficiently gather multiple attributes
                pinfo = {
                    "pid": p.pid,
                    "name": p.name(),
                    "username": p.username(),
                    "cpu_percent": p.cpu_percent(interval=None), # Use existing system-wide interval
                    "memory_percent": p.memory_percent()
                }
                # Filter out idle or very low usage processes early if desired
                if pinfo['cpu_percent'] > 0.01: # Consider processes with some CPU usage
                     procs.append(pinfo)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
        except Exception as e:
            # print(f"Skipping process {getattr(p, 'pid', 'UNKNOWN')} due to error: {e}")
            pass

    # Sort by CPU percentage
    procs.sort(key=lambda x: x['cpu_percent'], reverse=True)

    # Format and return top N
    for pinfo in procs[:top_n]:
        processes_data.append({
            "pid": pinfo['pid'],
            "name": pinfo['name'],
            "user": pinfo['username'],
            # Normalize by logical core count for overall system percentage if desired,
            # or keep as is (percentage of one core). psutil already provides system-wide %.
            # Here, process.cpu_percent() is % of total CPU time for that process.
            "cpu_percent": round(pinfo['cpu_percent'], 2),
            "memory_percent": round(pinfo['memory_percent'], 2)
        })
    return processes_data

# --- Pydantic Models for API Responses ---
class GPUProcessInfoModel(BaseModel): # Renamed to avoid conflict with internal dict key
    pid: int
    name: str
    user: str
    gpu_memory_mb: float

class GPUInfoModel(BaseModel): # Renamed
    id: int
    uuid: str
    name: str
    utilization_gpu: float
    memory_total_mb: float
    memory_used_mb: float
    temperature_c: Optional[float] = None
    processes: List[GPUProcessInfoModel] = []

class CPUConfigInfoModel(BaseModel): # Renamed
    model_name: str
    physical_cores: int
    logical_cores: int

class ServerConfig(BaseModel):
    server_ip: str
    server_name: str
    gpus: List[GPUInfoModel]
    cpus: CPUConfigInfoModel
    memory_total_gb: float

class CPUProcessDetailModel(BaseModel): # Renamed
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
    gpus: List[GPUInfoModel]


# --- API Endpoints ---
@app.get("/api/server/config", response_model=ServerConfig)
async def get_server_configuration():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        server_ip = s.getsockname()[0]
        s.close()
    except Exception:
        try:
            server_ip = socket.gethostbyname(socket.gethostname())
        except socket.gaierror:
            server_ip = "127.0.0.1"

    server_name = socket.gethostname()
    gpu_static_info_list = []
    raw_gpus_config = get_gpu_info_list()
    for gpu_data in raw_gpus_config:
        gpu_static_info_list.append(GPUInfoModel(
            id=gpu_data['id'],
            uuid=gpu_data['uuid'],
            name=gpu_data['name'],
            utilization_gpu=0.0,
            memory_total_mb=gpu_data['memory_total_mb'],
            memory_used_mb=0.0,
            temperature_c=None,
            processes=[]
        ))

    cpu_config = get_cpu_config_info()
    mem_info = psutil.virtual_memory()

    return ServerConfig(
        server_ip=server_ip,
        server_name=server_name,
        gpus=gpu_static_info_list,
        cpus=CPUConfigInfoModel(**cpu_config),
        memory_total_gb=round(mem_info.total / (1024**3), 2)
    )

@app.get("/api/server/status", response_model=ResourceStatus)
async def get_realtime_status(top_n_gpu_processes: int = Query(3, ge=0, le=10, description="Number of top processes per GPU")):
    cpu_util = psutil.cpu_percent(interval=0.1) # Non-blocking, captures usage over 0.1s
    ram_info = psutil.virtual_memory()
    gpus_current_status = get_gpu_info_list() # Gets current utilization, memory, temp

    detailed_gpus_status = []
    for gpu_stat in gpus_current_status:
        gpu_procs_pydantic = []
        if top_n_gpu_processes > 0:
             gpu_procs_raw = get_gpu_processes(gpu_stat["uuid"], top_n=top_n_gpu_processes)
             gpu_procs_pydantic = [GPUProcessInfoModel(**proc) for proc in gpu_procs_raw]

        detailed_gpus_status.append(GPUInfoModel(
            id=gpu_stat['id'],
            uuid=gpu_stat['uuid'],
            name=gpu_stat['name'],
            utilization_gpu=gpu_stat['utilization_gpu'],
            memory_total_mb=gpu_stat['memory_total_mb'],
            memory_used_mb=gpu_stat['memory_used_mb'],
            temperature_c=gpu_stat.get('temperature_c'),
            processes=gpu_procs_pydantic
        ))

    return ResourceStatus(
        cpu_utilization_percent=cpu_util,
        ram_used_gb=round(ram_info.used / (1024**3), 2),
        ram_total_gb=round(ram_info.total / (1024**3), 2),
        ram_utilization_percent=ram_info.percent,
        gpus=detailed_gpus_status
    )

@app.get("/api/cpu/top_processes", response_model=List[CPUProcessDetailModel])
async def get_cpu_top_processes_endpoint(n: int = Query(3, ge=1, le=20, description="Number of top CPU processes")):
    # Prime psutil.cpu_percent() for the process specific calls
    psutil.cpu_percent(interval=0.1) # Small blocking call to establish baseline for subsequent per-process calls
    await asyncio.sleep(0.01) # Short sleep to ensure the interval is meaningful for per-process calls

    top_processes_raw = get_top_cpu_processes(top_n=n)
    return [CPUProcessDetailModel(**proc) for proc in top_processes_raw]


if __name__ == "__main__":
    import uvicorn
    print("Starting ServerMonitor API backend...")
    print("Ensure 'nvidia-smi' is installed and in PATH if you have NVIDIA GPUs.")
    print("Backend will be available at http://0.0.0.0:8801")
    # For development, run with: uvicorn backend:app --reload --host 0.0.0.0 --port 8801
    # The host 0.0.0.0 makes it accessible from other machines on the network.
    uvicorn.run(app, host="0.0.0.0", port=8801)