# backend.py
import asyncio
import datetime
import json
import os
import re
import socket
import subprocess
from typing import List, Dict, Any, Optional

import psutil
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --- Configuration ---
LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

app = FastAPI(
    title="ServerMonitor API",
    description="API for monitoring Linux server status.",
    version="0.1.0"
)

# --- Helper Functions ---
def run_command(command: str) -> str:
    """Executes a shell command and returns its stdout."""
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=15) # Added timeout
        if process.returncode != 0:
            # Log stderr but don't raise immediately for commands like nvidia-smi if no GPUs
            print(f"Command '{command}' failed with error: {stderr.strip()}")
            return "" # Return empty string on error
        return stdout.strip()
    except subprocess.TimeoutExpired:
        print(f"Command '{command}' timed out.")
        process.kill()
        return ""
    except Exception as e:
        print(f"Error executing command '{command}': {e}")
        return ""

def get_gpu_info():
    """
    Parses nvidia-smi output to get GPU details.
    Returns a list of dicts, each representing a GPU.
    """
    gpus = []
    # Query for basic GPU info
    smi_output_basic = run_command(
        "nvidia-smi --query-gpu=index,uuid,name,utilization.gpu,memory.total,memory.used,temperature.gpu --format=csv,noheader,nounits"
    )
    if not smi_output_basic:
        return gpus # No GPUs or nvidia-smi not found

    for line in smi_output_basic.strip().split("\n"):
        if not line:
            continue
        parts = line.split(", ")
        try:
            gpus.append({
                "id": int(parts[0]),
                "uuid": parts[1],
                "name": parts[2],
                "utilization_gpu": float(parts[3]),
                "memory_total_mb": float(parts[4]),
                "memory_used_mb": float(parts[5]),
                "temperature_c": float(parts[6]) if len(parts) > 6 and parts[6] != "[Not Supported]" else None,
                "processes": [] # Will be populated later if requested
            })
        except (IndexError, ValueError) as e:
            print(f"Error parsing GPU line '{line}': {e}")
            continue
    return gpus

def get_gpu_processes(gpu_uuid: str, top_n: int = 3) -> List[Dict[str, Any]]:
    """Gets top N processes running on a specific GPU."""
    # nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits
    # This command lists processes and the GPU they are on.
    smi_processes_output = run_command(
        f"nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits"
    )

    gpu_processes = []
    if not smi_processes_output:
        return gpu_processes

    for line in smi_processes_output.strip().split("\n"):
        if not line:
            continue
        try:
            uuid, pid_str, proc_name, mem_used_str = line.split(", ")
            if uuid == gpu_uuid:
                pid = int(pid_str)
                mem_used = float(mem_used_str)

                # Get user for the process
                user = "N/A"
                try:
                    process = psutil.Process(pid)
                    user = process.username()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass # Process might have ended or access issues

                gpu_processes.append({
                    "pid": pid,
                    "name": proc_name.split('/')[-1], # Get basename of process
                    "user": user,
                    "gpu_memory_mb": mem_used
                })
        except (IndexError, ValueError) as e:
            print(f"Error parsing GPU process line '{line}': {e}")
            continue

    # Sort by GPU memory used and take top N
    gpu_processes.sort(key=lambda x: x["gpu_memory_mb"], reverse=True)
    return gpu_processes[:top_n]


def get_cpu_info():
    """Gets CPU information."""
    cpu_count_logical = psutil.cpu_count(logical=True)
    cpu_count_physical = psutil.cpu_count(logical=False)

    # Try to get CPU model name (platform dependent)
    model_name = "N/A"
    try:
        if os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if "model name" in line:
                        model_name = line.split(":")[1].strip()
                        break
        elif sys.platform == "darwin": # macOS
            model_name = run_command("sysctl -n machdep.cpu.brand_string")
    except Exception:
        pass # Could not get model name

    return {
        "model_name": model_name,
        "physical_cores": cpu_count_physical,
        "logical_cores": cpu_count_logical,
    }

def get_top_cpu_processes(top_n: int = 3) -> List[Dict[str, Any]]:
    """Gets top N CPU consuming processes."""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cpu_percent', 'memory_percent']):
        try:
            pinfo = proc.info
            # cpu_percent can be > 100% for multi-core, sum over all cores
            pinfo['cpu_percent'] = proc.cpu_percent(interval=None) / psutil.cpu_count()
            if pinfo['cpu_percent'] > 0.0: # Only consider processes with some CPU usage
                 processes.append({
                    "pid": pinfo['pid'],
                    "name": pinfo['name'],
                    "user": pinfo['username'],
                    "cpu_percent": round(pinfo['cpu_percent'], 2),
                    "memory_percent": round(pinfo['memory_percent'], 2)
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass

    processes.sort(key=lambda x: x["cpu_percent"], reverse=True)
    return processes[:top_n]

# --- Pydantic Models for API Responses ---
class GPUInfo(BaseModel):
    id: int
    uuid: str
    name: str
    utilization_gpu: float
    memory_total_mb: float
    memory_used_mb: float
    temperature_c: Optional[float] = None
    processes: List[Dict[str, Any]] = []

class CPUInfo(BaseModel):
    model_name: str
    physical_cores: int
    logical_cores: int

class ServerConfig(BaseModel):
    server_ip: str
    server_name: str
    gpus: List[GPUInfo] # Only static part: name, total_mem, id, uuid
    cpus: CPUInfo
    memory_total_gb: float

class ResourceStatus(BaseModel):
    cpu_utilization_percent: float
    ram_used_gb: float
    ram_total_gb: float
    ram_utilization_percent: float
    gpus: List[GPUInfo] # Dynamic part: utilization, mem_used, temp, processes

class ProcessInfo(BaseModel):
    pid: int
    name: str
    user: str
    cpu_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None
    memory_percent: Optional[float] = None


# --- API Endpoints ---
@app.get("/api/server/config", response_model=ServerConfig)
async def get_server_configuration():
    """Provides static configuration of the server."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80)) # Connect to a known external server
        server_ip = s.getsockname()[0]
        s.close()
    except Exception:
        server_ip = "127.0.0.1" # Fallback

    server_name = socket.gethostname()

    gpu_static_info = []
    raw_gpus = get_gpu_info() # This gets dynamic info, we'll extract static parts
    for gpu_data in raw_gpus:
        gpu_static_info.append(GPUInfo(
            id=gpu_data['id'],
            uuid=gpu_data['uuid'],
            name=gpu_data['name'],
            utilization_gpu=0.0, # Placeholder, not relevant for static config
            memory_total_mb=gpu_data['memory_total_mb'],
            memory_used_mb=0.0, # Placeholder
            temperature_c=None, # Placeholder
            processes=[]
        ))

    cpu_info = get_cpu_info()
    mem_info = psutil.virtual_memory()

    return ServerConfig(
        server_ip=server_ip,
        server_name=server_name,
        gpus=gpu_static_info,
        cpus=cpu_info,
        memory_total_gb=round(mem_info.total / (1024**3), 2)
    )

@app.get("/api/server/status", response_model=ResourceStatus)
async def get_realtime_status(top_n_gpu_processes: int = Query(3, ge=0, le=10)):
    """Provides real-time resource utilization."""
    cpu_util = psutil.cpu_percent(interval=0.1) # Non-blocking, use last measured
    ram_info = psutil.virtual_memory()

    gpus_data = get_gpu_info() # Gets current utilization, memory, temp

    detailed_gpus = []
    for gpu_stat in gpus_data:
        # Fetch top processes for this specific GPU
        # Make sure get_gpu_processes is efficient or consider caching if too slow
        gpu_procs = get_gpu_processes(gpu_stat["uuid"], top_n=top_n_gpu_processes)
        detailed_gpus.append(GPUInfo(
            id=gpu_stat['id'],
            uuid=gpu_stat['uuid'],
            name=gpu_stat['name'],
            utilization_gpu=gpu_stat['utilization_gpu'],
            memory_total_mb=gpu_stat['memory_total_mb'],
            memory_used_mb=gpu_stat['memory_used_mb'],
            temperature_c=gpu_stat.get('temperature_c'),
            processes=gpu_procs
        ))

    return ResourceStatus(
        cpu_utilization_percent=cpu_util,
        ram_used_gb=round(ram_info.used / (1024**3), 2),
        ram_total_gb=round(ram_info.total / (1024**3), 2),
        ram_utilization_percent=ram_info.percent,
        gpus=detailed_gpus
    )

@app.get("/api/cpu/top_processes", response_model=List[ProcessInfo])
async def get_cpu_top_processes_endpoint(n: int = Query(3, ge=1, le=20)):
    """Gets top N CPU consuming processes."""
    return get_top_cpu_processes(top_n=n)

# --- Reporting ---
def generate_report_content(period_name: str, data: Dict[str, Any]) -> str:
    """Generates markdown report content."""
    now = datetime.datetime.now()
    report_date = now.strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"# Server Resource Report ({period_name.capitalize()})\n"
    md_content += f"Generated on: {report_date}\n\n"

    md_content += f"## Server: {data['config']['server_name']} ({data['config']['server_ip']})\n\n"

    md_content += "## System Summary\n"
    md_content += f"- **CPU Usage (avg):** {data['status']['cpu_utilization_percent']:.2f}%\n"
    md_content += f"- **RAM Usage (avg):** {data['status']['ram_utilization_percent']:.2f}% ({data['status']['ram_used_gb']:.2f} GB / {data['status']['ram_total_gb']:.2f} GB)\n\n"

    # User-centric aggregation (simplified for this example)
    # A more robust report would aggregate data over the period (daily/weekly)
    # This is a snapshot report.
    user_cpu_usage = {}
    user_gpu_mem_usage = {} # Per GPU, per user

    # CPU Processes
    md_content += "### Top CPU Consuming Processes (Snapshot)\n"
    top_cpu_procs = get_top_cpu_processes(10) # Get more for reporting
    if top_cpu_procs:
        md_content += "| User | Process Name | PID | CPU % |\n"
        md_content += "|------|--------------|-----|-------|\n"
        for p in top_cpu_procs:
            md_content += f"| {p['user']} | {p['name']} | {p['pid']} | {p['cpu_percent']:.2f} |\n"
            user_cpu_usage[p['user']] = user_cpu_usage.get(p['user'], 0) + p['cpu_percent']
    else:
        md_content += "No significant CPU processes found.\n"
    md_content += "\n"

    # GPU Processes
    md_content += "### GPU Usage (Snapshot)\n"
    if data['status']['gpus']:
        for gpu in data['status']['gpus']:
            md_content += f"#### GPU {gpu['id']}: {gpu['name']} (UUID: {gpu['uuid']})\n"
            md_content += f"- Utilization: {gpu['utilization_gpu']:.2f}%\n"
            md_content += f"- Memory: {gpu['memory_used_mb']:.2f} MB / {gpu['memory_total_mb']:.2f} MB\n"

            gpu_procs_for_report = get_gpu_processes(gpu['uuid'], top_n=5) # Get more for reporting
            if gpu_procs_for_report:
                md_content += "**Top Processes on this GPU:**\n"
                md_content += "| User | Process Name | PID | GPU Memory (MB) |\n"
                md_content += "|------|--------------|-----|-----------------|\n"
                for p in gpu_procs_for_report:
                    md_content += f"| {p['user']} | {p['name']} | {p['pid']} | {p['gpu_memory_mb']:.2f} |\n"

                    # Aggregate GPU memory by user for this GPU
                    gpu_user_key = f"GPU{gpu['id']}_{p['user']}"
                    user_gpu_mem_usage[gpu_user_key] = user_gpu_mem_usage.get(gpu_user_key, 0) + p['gpu_memory_mb']
            else:
                md_content += "No active processes found on this GPU.\n"
            md_content += "\n"
    else:
        md_content += "No GPUs detected or nvidia-smi not available.\n"
    md_content += "\n"

    md_content += "## User Resource Summary (Snapshot)\n"
    md_content += "### CPU Usage by User (Sum of top processes %)\n"
    if user_cpu_usage:
        for user, cpu_p in sorted(user_cpu_usage.items(), key=lambda item: item[1], reverse=True):
            md_content += f"- **{user}:** {cpu_p:.2f}%\n"
    else:
        md_content += "No user CPU usage data collected.\n"
    md_content += "\n"

    md_content += "### GPU Memory Usage by User (Sum of MBs on GPUs)\n"
    if user_gpu_mem_usage:
        # For simplicity, sum all GPU memory per user across all GPUs.
        # A more detailed report might list per GPU per user.
        agg_user_gpu_mem = {}
        for gpu_user_key, mem in user_gpu_mem_usage.items():
            user = gpu_user_key.split('_',1)[1] # Extract user from "GPU0_username"
            agg_user_gpu_mem[user] = agg_user_gpu_mem.get(user,0) + mem

        for user, mem_mb in sorted(agg_user_gpu_mem.items(), key=lambda item: item[1], reverse=True):
            md_content += f"- **{user}:** {mem_mb:.2f} MB\n"
    else:
        md_content += "No user GPU usage data collected.\n"

    return md_content

@app.post("/api/reports/generate")
async def generate_report_endpoint(period: str = Query("daily", enum=["daily", "weekly"])):
    """Generates a resource usage report."""
    # For this example, "daily" and "weekly" reports will contain a snapshot.
    # A real implementation would involve data collection over time.

    # Fetch current data
    config_data = await get_server_configuration()
    # Use a small top_n for status endpoint to avoid making it too slow for live view,
    # but report generation can query more if needed (as done inside generate_report_content)
    status_data = await get_realtime_status(top_n_gpu_processes=0) # Report function will get its own processes

    report_data = {
        "config": config_data.model_dump(),
        "status": status_data.model_dump()
    }

    md_content = generate_report_content(period, report_data)

    now = datetime.datetime.now()
    if period == "daily":
        filename = f"report_daily_{now.strftime('%Y-%m-%d')}.md"
    elif period == "weekly":
        # For weekly, use start of the week (Monday) and end of week (Sunday)
        start_of_week = now - datetime.timedelta(days=now.weekday())
        end_of_week = start_of_week + datetime.timedelta(days=6)
        filename = f"report_weekly_{start_of_week.strftime('%Y-%m-%d')}_to_{end_of_week.strftime('%Y-%m-%d')}.md"
    else:
        raise HTTPException(status_code=400, detail="Invalid report period.")

    filepath = os.path.join(LOGS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)

    return {"message": f"Report '{filename}' generated successfully.", "filename": filename}

@app.get("/api/reports/list")
async def list_reports_endpoint():
    """Lists available reports."""
    try:
        files = [f for f in os.listdir(LOGS_DIR) if f.endswith(".md")]
        # Sort by name, which should roughly sort by date due to naming convention
        files.sort(reverse=True)
        return {"reports": files}
    except FileNotFoundError:
        return {"reports": []}

@app.get("/api/reports/view/{filename}")
async def view_report_endpoint(filename: str):
    """Gets the content of a specific report."""
    filepath = os.path.join(LOGS_DIR, filename)
    if not os.path.exists(filepath) or not filename.endswith(".md"):
        raise HTTPException(status_code=404, detail="Report not found.")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return {"filename": filename, "content": content}

@app.get("/api/reports/download/{filename}")
async def download_report_endpoint(filename: str):
    """Downloads a specific report file."""
    filepath = os.path.join(LOGS_DIR, filename)
    if not os.path.exists(filepath) or not filename.endswith(".md"):
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(filepath, media_type='text/markdown', filename=filename)


if __name__ == "__main__":
    import uvicorn
    # For development, run with: uvicorn backend:app --reload --host 0.0.0.0 --port 8801
    # The host 0.0.0.0 makes it accessible from other machines on the network.
    # For production, consider a more robust setup (e.g., Gunicorn with Uvicorn workers).
    uvicorn.run(app, host="0.0.0.0", port=8801)