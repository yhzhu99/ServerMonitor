# frontend.py
import streamlit as st
import requests
import pandas as pd
import time

# --- Page Configuration ---
st.set_page_config(
    page_title="ServerMonitor",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🖥️ ServerMonitor Dashboard")

# --- Global State Management with st.cache_resource ---
# This dictionary will be shared across all user sessions for this Streamlit app instance.
@st.cache_resource # Persists across reruns and sessions for the lifetime of the app process
def get_global_app_state():
    """Initializes and returns the global state dictionary."""
    return {
        "monitored_servers": [],  # List of {'name': str, 'url': str, 'config_cache': None}
        "top_n_processes": 3, # Default Top-N processes
    }

GLOBAL_APP_STATE = get_global_app_state()

# --- Helper Function to Call Backend API ---
def fetch_from_api_server(base_url: str, endpoint: str, params: dict = None):
    """Generic function to fetch data from a specific server's backend."""
    try:
        url = f"{base_url.rstrip('/')}{endpoint}"
        response = requests.get(url, params=params, timeout=4) # Slightly reduced timeout for responsiveness
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": f"Timeout connecting to {url}"}
    except requests.exceptions.ConnectionError:
        return {"error": f"Connection error to {url}. Is backend server running?"}
    except requests.exceptions.HTTPError as e:
        return {"error": f"HTTP error for {url}: {e.response.status_code} {e.response.reason}"}
    except requests.exceptions.RequestException as e: # Catch other request-related errors
        return {"error": f"Request error for {url}: {e}"}
    except requests.exceptions.JSONDecodeError: # If response is not valid JSON
        return {"error": f"Failed to decode JSON from {url}. Response: {response.text[:200]}"}


# --- Cached Data Fetching Functions ---

# Cache for server configuration (long TTL as it's mostly static)
@st.cache_data(ttl=3600) # Cache config for 1 hour
def get_server_config_cached(server_url: str):
    """Fetches and caches server configuration."""
    return fetch_from_api_server(server_url, "/api/server/config")

# For dynamic data
@st.cache_data(ttl=1.0)
def fetch_status_cached(base_url: str, top_n_gpu_processes: int):
    """Fetches and caches real-time server status."""
    return fetch_from_api_server(base_url, "/api/server/status", params={"top_n_gpu_processes": top_n_gpu_processes})

@st.cache_data(ttl=1.0)
def fetch_cpu_processes_cached(base_url: str, n: int):
    """Fetches and caches top CPU processes."""
    return fetch_from_api_server(base_url, "/api/cpu/top_processes", params={"n": n})


# --- Sidebar for Global Controls & Server Management ---
st.sidebar.header("⚙️ Global Controls")

# Top N Processes
current_top_n = GLOBAL_APP_STATE["top_n_processes"]
# Use a number input instead of dropdown to allow custom values
top_n = st.sidebar.number_input(
    "Number of Top Processes to Display",
    min_value=1,
    max_value=20,
    value=current_top_n,
    step=1,
    help="Number of top CPU/GPU processes to display for each server."
)

if top_n != current_top_n:
    GLOBAL_APP_STATE["top_n_processes"] = top_n
    # Clear caches that depend on top_n
    fetch_status_cached.clear()
    fetch_cpu_processes_cached.clear()
    st.rerun()

# Manual refresh button
if st.sidebar.button("🔄 Refresh Data Now"):
    # Clear all dynamic data caches to force fresh data fetching
    fetch_status_cached.clear()
    fetch_cpu_processes_cached.clear()
    st.rerun()


st.sidebar.divider()
st.sidebar.header("🔗 Manage Servers")
st.sidebar.caption("(Shared list for all users of this app instance)")


with st.sidebar.form("add_server_form", clear_on_submit=True):
    st.subheader("Add New Server")
    new_server_name = st.text_input("Server Name (e.g., Prod-Server-1)", placeholder="My Web Server")
    new_server_url = st.text_input("Server API URL (e.g., http://192.168.1.100:8801)", placeholder="http://localhost:8801")
    add_server_submitted = st.form_submit_button("➕ Add Server")

    if add_server_submitted:
        if new_server_name and new_server_url:
            cleaned_url = new_server_url.strip().rstrip('/')
            if not (cleaned_url.startswith("http://") or cleaned_url.startswith("https://")):
                st.sidebar.error("URL must start with http:// or https://")
            elif any(s['url'] == cleaned_url for s in GLOBAL_APP_STATE["monitored_servers"]) or \
                 any(s['name'].lower() == new_server_name.lower() for s in GLOBAL_APP_STATE["monitored_servers"]): # Case-insensitive name check
                st.sidebar.warning("Server with this name or URL already exists.")
            else:
                # Test connection by fetching config before adding
                st.sidebar.info(f"Trying to connect to {cleaned_url}...")
                test_config = fetch_from_api_server(cleaned_url, "/api/server/config")

                if test_config and "error" not in test_config:
                    GLOBAL_APP_STATE["monitored_servers"].append(
                        {'name': new_server_name, 'url': cleaned_url}
                    )
                    st.sidebar.success(f"Added server: {new_server_name}")
                    st.rerun() # Rerun to update server list and tabs
                else:
                    error_msg = test_config.get('error', 'Unknown error') if isinstance(test_config, dict) else 'Unknown error during connection test.'
                    st.sidebar.error(f"Failed to add {new_server_name}. Error: {error_msg}")
        else:
            st.sidebar.error("Both server name and URL are required.")

if GLOBAL_APP_STATE["monitored_servers"]:
    st.sidebar.subheader("Monitored Servers")
    servers_to_remove_indices = []
    for i, server in enumerate(GLOBAL_APP_STATE["monitored_servers"]):
        col1, col2 = st.sidebar.columns([0.85, 0.15]) # Adjust column ratio for button
        col1.text(f"{server['name']} ({server['url']})")
        if col2.button("🗑️", key=f"remove_server_{i}", help=f"Remove {server['name']}"):
            servers_to_remove_indices.append(i)

    if servers_to_remove_indices:
        for i in sorted(servers_to_remove_indices, reverse=True): # Remove from end to keep indices valid
            removed_server = GLOBAL_APP_STATE["monitored_servers"].pop(i)
            st.sidebar.toast(f"Removed server: {removed_server['name']}")
        st.rerun() # Rerun to reflect removed server


# --- Main Display Area for Servers ---
def display_server_data(server_info: dict, top_n_global: int):
    """Displays monitoring data for a single server."""
    base_url = server_info['url']
    server_name = server_info['name']

    # --- Server Configuration Section (fetched once or from cache) ---
    st.subheader(f"ℹ️ {server_name} - Configuration", anchor=False)
    config_data = get_server_config_cached(base_url)

    if config_data and "error" not in config_data:
        col_cfg1, col_cfg2 = st.columns([1,1])
        with col_cfg1:
            st.markdown(f"**Backend Name:** `{config_data.get('server_name', 'N/A')}`")
            st.markdown(f"**Backend IP:** `{config_data.get('server_ip', 'N/A')}`")
        with col_cfg2:
            st.markdown(f"**Total RAM:** {config_data.get('memory_total_gb', 0):.2f} GB")

        cpus_cfg = config_data.get('cpus', {})
        with st.expander("🔩 CPU Configuration", expanded=False):
            st.markdown(f"- **Model:** {cpus_cfg.get('model_name', 'N/A')}")
            st.markdown(f"- **Physical Cores:** {cpus_cfg.get('physical_cores', 'N/A')}")
            st.markdown(f"- **Logical Cores:** {cpus_cfg.get('logical_cores', 'N/A')}")

        gpus_cfg = config_data.get('gpus', [])
        if gpus_cfg:
            with st.expander("🎮 GPU Configuration", expanded=False):
                for i, gpu in enumerate(gpus_cfg):
                    st.markdown(f"--- \n **GPU {gpu.get('id','N/A')}: {gpu.get('name','N/A')}**")
                    st.markdown(f"  - UUID: `{gpu.get('uuid','N/A')}`")
                    st.markdown(f"  - Total Memory: {gpu.get('memory_total_mb', 0) / 1024:.2f} GB")
        # else:
            # st.caption("No GPU configuration data reported by this server.") # Less prominent than info box
    elif config_data and "error" in config_data:
        err_msg = config_data.get('error', 'Failed to load configuration')
        st.warning(f"Could not load server configuration for {server_name}: {err_msg}", icon="⚠️")
        # Potentially return here if config is critical for further display
    else: # Handle case where config_data might be None (e.g. initial state before first fetch attempt)
        st.info(f"Fetching configuration for {server_name}...", icon="⏳")


    st.divider()

    # --- Real-time Status Section ---
    st.subheader(f"📊 {server_name} - Real-time Usage", anchor=False)
    # Pass top_n_global to fetch_status_cached, as the API endpoint uses it.
    status_data = fetch_status_cached(base_url, top_n_global)

    if status_data and "error" not in status_data:
        col_rt_summary1, col_rt_summary2 = st.columns(2)
        with col_rt_summary1: # CPU Usage
            cpu_util = status_data.get('cpu_utilization_percent', 0.0)
            st.progress(int(cpu_util), text=f"CPU Usage: {cpu_util:.1f}%")
        with col_rt_summary2: # RAM Usage
            ram_util = status_data.get('ram_utilization_percent', 0.0)
            ram_used_gb = status_data.get('ram_used_gb', 0)
            ram_total_gb = status_data.get('ram_total_gb', 0)
            ram_text = f"RAM: {ram_util:.1f}% ({ram_used_gb:.2f} GB / {ram_total_gb:.2f} GB)"
            st.progress(int(ram_util), text=ram_text)

        st.markdown("---") # Visual separator

        # Columns for CPU processes and GPU details
        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.markdown("##### <p style='text-align: center;'>🚀 CPU Top Processes</p>", unsafe_allow_html=True)
            cpu_processes_data = fetch_cpu_processes_cached(base_url, top_n_global)
            if cpu_processes_data and "error" not in cpu_processes_data:
                if cpu_processes_data: # Check if list is not empty
                    df_cpu_procs = pd.DataFrame(cpu_processes_data)
                    # Format the percentage columns for better readability
                    if 'cpu_percent' in df_cpu_procs.columns:
                        df_cpu_procs['cpu_percent'] = df_cpu_procs['cpu_percent'].apply(lambda x: f"{x:.2f}%")
                    if 'memory_percent' in df_cpu_procs.columns:
                        df_cpu_procs['memory_percent'] = df_cpu_procs['memory_percent'].apply(lambda x: f"{x:.2f}%")

                    cols_to_show = [col for col in ['pid', 'name', 'user', 'cpu_percent', 'memory_percent'] if col in df_cpu_procs.columns]
                    st.dataframe(df_cpu_procs[cols_to_show], use_container_width=True, hide_index=True, height=len(df_cpu_procs)*35 + 40 if len(df_cpu_procs) > 0 else 75) # Dynamic height
                else:
                    st.caption("No significant CPU processes reported or matching criteria.")
            elif cpu_processes_data and "error" in cpu_processes_data:
                st.warning(f"CPU top processes data unavailable: {cpu_processes_data.get('error','N/A')}", icon="⚠️")
            else:
                st.caption("Fetching CPU process data...")


        with detail_cols[1]:
            st.markdown("##### <p style='text-align: center;'>🎮 GPU(s) Detailed Status</p>", unsafe_allow_html=True)
            gpus_status_list = status_data.get('gpus', [])
            if gpus_status_list:
                for i, gpu in enumerate(gpus_status_list):
                    exp_title = (f"GPU {gpu.get('id','N/A')}: {gpu.get('name','N/A')} "
                                 f"(Util: {gpu.get('utilization_gpu',0):.1f}%)")
                    # Keep GPUs expanded by default or make it configurable
                    with st.expander(exp_title, expanded=True):
                        mem_total_mb = gpu.get('memory_total_mb', 1) # Avoid division by zero if data is malformed
                        mem_used_mb = gpu.get('memory_used_mb', 0)
                        mem_util_percent = (mem_used_mb / mem_total_mb) * 100 if mem_total_mb > 0 else 0
                        mem_total_gb = mem_total_mb / 1024
                        mem_used_gb = mem_used_mb / 1024

                        # Using st.metric for GPU stats
                        gpu_metric_cols = st.columns(2)
                        with gpu_metric_cols[0]:
                            st.metric(label=f"GPU {gpu.get('id','N/A')} Util.", value=f"{gpu.get('utilization_gpu',0):.1f}%")
                            if gpu.get('temperature_c') is not None:
                                st.metric(label=f"GPU {gpu.get('id','N/A')} Temp.", value=f"{gpu.get('temperature_c',0):.1f}°C")
                        with gpu_metric_cols[1]:
                            st.metric(label=f"VRAM Usage", value=f"{mem_util_percent:.1f}%",
                                      help=f"{mem_used_gb:.2f}GB / {mem_total_gb:.2f}GB")

                        st.progress(int(mem_util_percent), text=f"VRAM: {mem_used_gb:.2f}GB / {mem_total_gb:.2f}GB ({mem_util_percent:.1f}%)")

                        gpu_procs = gpu.get('processes', [])
                        if gpu_procs:
                            st.markdown("**Top Processes on this GPU:**")
                            df_gpu_procs = pd.DataFrame(gpu_procs)
                            # Format memory usage for better readability
                            if 'gpu_memory_mb' in df_gpu_procs.columns:
                                df_gpu_procs['gpu_memory'] = df_gpu_procs['gpu_memory_mb'].apply(lambda x: f"{x/1024:.2f} GB")

                            # Ensure columns exist before trying to select them
                            cols_to_show_gpu = ['pid', 'name', 'user', 'gpu_memory']
                            if 'gpu_memory' in df_gpu_procs.columns:
                                df_gpu_procs = df_gpu_procs[['pid', 'name', 'user', 'gpu_memory']]
                                st.dataframe(df_gpu_procs, use_container_width=True, hide_index=True, height=len(df_gpu_procs)*35 + 40 if len(df_gpu_procs) > 0 else 75)
                            else:
                                cols_to_show_gpu = [col for col in ['pid', 'name', 'user', 'gpu_memory_mb'] if col in df_gpu_procs.columns]
                                st.dataframe(df_gpu_procs[cols_to_show_gpu], use_container_width=True, hide_index=True, height=len(df_gpu_procs)*35 + 40 if len(df_gpu_procs) > 0 else 75)
                        elif top_n_global > 0 : # Only show "no processes" if we asked for them
                            st.caption(f"No top processes reported on GPU {gpu.get('id','N/A')}.")
            elif 'gpus' in status_data and not status_data['gpus']: # API reported gpus key, but it's empty
                 st.info("No GPUs detected or reported by this server for real-time status.", icon="ℹ️")
            # else: No specific message if 'gpus' key is missing, implies data not yet loaded or error already shown

    elif status_data and "error" in status_data:
        err_msg = status_data.get('error', 'Failed to load real-time status')
        st.error(f"Could not load real-time status for {server_name}: {err_msg}", icon="🚨")
    else: # Handle case where status_data might be None (e.g. initial state)
        st.info(f"Fetching real-time status for {server_name}...", icon="⏳")


# --- Main Application Logic ---
if not GLOBAL_APP_STATE["monitored_servers"]:
    st.info(
        "👋 Welcome to ServerMonitor! \n\n"
        "Please add servers using the 'Manage Servers' section in the sidebar to begin monitoring. "
        "All users viewing this application instance will share the same list of monitored servers and global settings."
    )
else:
    tab_titles = [s['name'] for s in GLOBAL_APP_STATE["monitored_servers"]]

    # Streamlit's st.tabs remembers the last selected tab *per session* by default.
    # If a tab (server) is removed (list changes), Streamlit handles this gracefully, usually selecting the first available tab.
    created_tabs = st.tabs(tab_titles)

    for i, tab_widget in enumerate(created_tabs):
        with tab_widget:
            # Defensive check: ensure server index is still valid, especially if GLOBAL_APP_STATE
            # could be modified by another session between creating tabs and rendering content.
            if i < len(GLOBAL_APP_STATE["monitored_servers"]):
                current_server_info = GLOBAL_APP_STATE["monitored_servers"][i]
                display_server_data(current_server_info, GLOBAL_APP_STATE["top_n_processes"])
            else:
                # This state should be rare due to st.rerun() on server list changes.
                st.warning("This server tab is no longer valid (server may have been removed by another user or session). Please refresh or select another tab.", icon="⚠️")