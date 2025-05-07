# frontend.py
import streamlit as st
import requests
import pandas as pd
import time
from streamlit_autorefresh import st_autorefresh

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
@st.cache_resource
def get_global_app_state():
    """Initializes and returns the global state dictionary."""
    return {
        "monitored_servers": [],  # List of {'name': str, 'url': str, 'config_cache': None}
        "refresh_interval_label": "10s",
        "top_n_processes": 3,
        # 'active_tab_name': None # Active tab is best kept per-session (Streamlit's default)
    }

GLOBAL_APP_STATE = get_global_app_state()

# Map refresh labels to seconds
REFRESH_INTERVALS_MAP = {"Off": 0, "1s": 1, "5s": 5, "10s": 10, "30s": 30, "60s": 60}

# --- Helper Function to Call Backend API ---
def fetch_from_api_server(base_url: str, endpoint: str, params: dict = None):
    """Generic function to fetch data from a specific server's backend."""
    try:
        url = f"{base_url.rstrip('/')}{endpoint}"
        response = requests.get(url, params=params, timeout=5) # Reduced timeout for quicker feedback
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        # st.error(f"Timeout connecting to {base_url}{endpoint}", icon="🚨") # Errors shown in context
        return {"error": f"Timeout connecting to {url}"}
    except requests.exceptions.ConnectionError:
        # st.error(f"Connection error to {base_url}{endpoint}.", icon="🚨")
        return {"error": f"Connection error to {url}. Is the backend server running?"}
    except requests.exceptions.RequestException as e:
        # st.error(f"Error fetching data from {base_url}{endpoint}: {e}", icon="🚨")
        return {"error": f"Request error for {url}: {e}"}
    except requests.exceptions.JSONDecodeError:
        # st.error(f"Failed to decode JSON from {base_url}{endpoint}", icon="🚨")
        return {"error": f"Failed to decode JSON from {url}"}

# --- Cached Data Fetching Functions (shared across sessions) ---

# Cache for server configuration (long TTL as it's mostly static)
@st.cache_data(ttl=3600) # Cache config for 1 hour
def get_server_config_cached(server_url: str):
    # print(f"Fetching CONFIG from {server_url} at {time.time()}") # Debug
    data = fetch_from_api_server(server_url, "/api/server/config")
    if data and "error" not in data:
        # Store this fetched config in our global state for quick access if needed,
        # though st.cache_data handles the primary caching.
        for srv in GLOBAL_APP_STATE["monitored_servers"]:
            if srv["url"] == server_url:
                srv["config_cache"] = data # Optional: update a direct cache in GLOBAL_APP_STATE
                break
    return data


def get_dynamic_cache_ttl():
    """Calculates TTL for dynamic data based on global refresh interval."""
    interval_seconds = REFRESH_INTERVALS_MAP.get(GLOBAL_APP_STATE["refresh_interval_label"], 0)
    if interval_seconds > 0:
        return max(0.5, interval_seconds * 0.9) # TTL is 90% of interval, min 0.5s
    return 60 # Default TTL (e.g. 60s) if refresh is "Off"


@st.cache_data(ttl=get_dynamic_cache_ttl)
def fetch_status_cached(base_url: str, top_n_gpu_processes: int):
    # print(f"Fetching STATUS from {base_url} (top_n_gpu={top_n_gpu_processes}) at {time.time()}") # Debug
    return fetch_from_api_server(base_url, "/api/server/status", params={"top_n_gpu_processes": top_n_gpu_processes})

@st.cache_data(ttl=get_dynamic_cache_ttl)
def fetch_cpu_processes_cached(base_url: str, n: int):
    # print(f"Fetching CPU_PROCS from {base_url} (n={n}) at {time.time()}") # Debug
    return fetch_from_api_server(base_url, "/api/cpu/top_processes", params={"n": n})


# --- Sidebar for Global Controls & Server Management ---
st.sidebar.header("Global Controls")

# Refresh Interval
current_refresh_label = GLOBAL_APP_STATE["refresh_interval_label"]
selected_interval_label = st.sidebar.selectbox(
    "Refresh Interval",
    options=list(REFRESH_INTERVALS_MAP.keys()),
    index=list(REFRESH_INTERVALS_MAP.keys()).index(current_refresh_label),
    help="Global: How often to refresh data for all servers. Affects all users."
)
if selected_interval_label != current_refresh_label:
    GLOBAL_APP_STATE["refresh_interval_label"] = selected_interval_label
    st.experimental_rerun() # Rerun to apply new interval and clear caches if TTL changes

refresh_interval_seconds = REFRESH_INTERVALS_MAP[GLOBAL_APP_STATE["refresh_interval_label"]]

if refresh_interval_seconds > 0:
    st_autorefresh(interval=refresh_interval_seconds * 1000, key="global_data_refresher")

# Top N Processes
current_top_n = GLOBAL_APP_STATE["top_n_processes"]
top_n_options = [1, 2, 3, 4, 5, 10]
selected_top_n = st.sidebar.selectbox(
    "Top N Processes",
    options=top_n_options,
    index=top_n_options.index(current_top_n),
    help="Global: Number of top CPU/GPU processes to display. Affects all users."
)
if selected_top_n != current_top_n:
    GLOBAL_APP_STATE["top_n_processes"] = selected_top_n
    st.experimental_rerun()


st.sidebar.divider()
st.sidebar.header("Manage Servers (Shared List)")

with st.sidebar.form("add_server_form", clear_on_submit=True):
    st.subheader("Add New Server")
    new_server_name = st.text_input("Server Name (e.g., Prod-Server-1)", placeholder="My Web Server")
    new_server_url = st.text_input("Server API URL (e.g., http://192.168.1.100:8801)", placeholder="http://localhost:8801")
    add_server_submitted = st.form_submit_button("Add Server")

    if add_server_submitted:
        if new_server_name and new_server_url:
            cleaned_url = new_server_url.strip().rstrip('/')
            if not (cleaned_url.startswith("http://") or cleaned_url.startswith("https://")):
                st.sidebar.error("URL must start with http:// or https://")
            elif any(s['url'] == cleaned_url for s in GLOBAL_APP_STATE["monitored_servers"]) or \
                 any(s['name'] == new_server_name for s in GLOBAL_APP_STATE["monitored_servers"]):
                st.sidebar.warning("Server with this name or URL already exists.")
            else:
                # Test connection before adding
                test_config = fetch_from_api_server(cleaned_url, "/api/server/config")
                if test_config and "error" not in test_config:
                    GLOBAL_APP_STATE["monitored_servers"].append(
                        {'name': new_server_name, 'url': cleaned_url, 'config_cache': test_config}
                    )
                    st.sidebar.success(f"Added server: {new_server_name}")
                    st.experimental_rerun()
                else:
                    error_msg = test_config.get('error', 'Unknown error') if isinstance(test_config, dict) else 'Unknown error'
                    st.sidebar.error(f"Failed to connect or get config from {cleaned_url}. Error: {error_msg}")
        else:
            st.sidebar.error("Both server name and URL are required.")

if GLOBAL_APP_STATE["monitored_servers"]:
    st.sidebar.subheader("Monitored Servers")
    servers_to_remove_indices = []
    for i, server in enumerate(GLOBAL_APP_STATE["monitored_servers"]):
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        col1.text(f"{server['name']} ({server['url']})")
        if col2.button("🗑️", key=f"remove_server_{i}", help=f"Remove {server['name']}"):
            servers_to_remove_indices.append(i)

    if servers_to_remove_indices:
        for i in sorted(servers_to_remove_indices, reverse=True):
            removed_server = GLOBAL_APP_STATE["monitored_servers"].pop(i)
            st.sidebar.toast(f"Removed server: {removed_server['name']}")
        st.experimental_rerun()


# --- Main Display Area for Servers ---
def display_server_data(server_info: dict, top_n_global: int):
    """Displays monitoring data for a single server."""
    base_url = server_info['url']
    server_name = server_info['name']

    # --- Server Configuration Section ---
    st.subheader(f"⚙️ {server_name} - Configuration", anchor=False)
    config_data = get_server_config_cached(base_url) # Uses st.cache_data

    if config_data and "error" not in config_data:
        col_cfg1, col_cfg2 = st.columns([1,1]) # Use 2 columns for config
        with col_cfg1:
            st.markdown(f"**Backend Name:** `{config_data.get('server_name', 'N/A')}`")
            st.markdown(f"**Backend IP:** `{config_data.get('server_ip', 'N/A')}`")
        with col_cfg2:
            st.markdown(f"**Total RAM:** {config_data.get('memory_total_gb', 0):.2f} GB")

        exp_cpu_cfg = st.expander("CPU Configuration", expanded=False)
        cpus_cfg = config_data.get('cpus', {})
        exp_cpu_cfg.markdown(f"- **Model:** {cpus_cfg.get('model_name', 'N/A')}")
        exp_cpu_cfg.markdown(f"- **Physical Cores:** {cpus_cfg.get('physical_cores', 'N/A')}")
        exp_cpu_cfg.markdown(f"- **Logical Cores:** {cpus_cfg.get('logical_cores', 'N/A')}")

        gpus_cfg = config_data.get('gpus', [])
        if gpus_cfg:
            exp_gpu_cfg = st.expander("GPU Configuration", expanded=False)
            for i, gpu in enumerate(gpus_cfg):
                exp_gpu_cfg.markdown(f"--- \n **GPU {gpu.get('id','N/A')}: {gpu.get('name','N/A')}**")
                exp_gpu_cfg.markdown(f"  - UUID: `{gpu.get('uuid','N/A')}`")
                exp_gpu_cfg.markdown(f"  - Total Memory: {gpu.get('memory_total_mb', 0):.0f} MB")
        else:
            st.info("No GPUs detected in server configuration.", icon="ℹ️")
    else:
        err_msg = config_data.get('error', 'Failed to load configuration') if isinstance(config_data, dict) else "Failed to load configuration"
        st.warning(f"Could not load server configuration for {server_name}: {err_msg}", icon="⚠️")
        return

    st.divider()

    # --- Real-time Status Section ---
    st.subheader(f"📊 {server_name} - Real-time Usage", anchor=False)
    status_data = fetch_status_cached(base_url, top_n_global) # Uses st.cache_data

    if status_data and "error" not in status_data:
        col_rt_summary1, col_rt_summary2 = st.columns(2)
        with col_rt_summary1: # CPU Usage
            cpu_util = status_data.get('cpu_utilization_percent', 0.0)
            st.progress(int(cpu_util), text=f"CPU Utilization: {cpu_util:.1f}%")
        with col_rt_summary2: # RAM Usage
            ram_util = status_data.get('ram_utilization_percent', 0.0)
            ram_text = f"RAM: {ram_util:.1f}% ({status_data.get('ram_used_gb', 0):.2f}GB / {status_data.get('ram_total_gb', 0):.2f}GB)"
            st.progress(int(ram_util), text=ram_text)

        st.markdown("---")

        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.markdown("##### <p style='text-align: center;'>🚀 CPU Top Processes</p>", unsafe_allow_html=True)
            cpu_processes_data = fetch_cpu_processes_cached(base_url, top_n_global) # Uses st.cache_data
            if cpu_processes_data and "error" not in cpu_processes_data:
                if cpu_processes_data:
                    df_cpu_procs = pd.DataFrame(cpu_processes_data)
                    # Ensure columns exist before selecting
                    cols_to_show = [col for col in ['pid', 'name', 'user', 'cpu_percent', 'memory_percent'] if col in df_cpu_procs.columns]
                    st.dataframe(df_cpu_procs[cols_to_show], use_container_width=True, hide_index=True, height=180) # Adjust height
                else:
                    st.info("No significant CPU processes found.", icon="ℹ️")
            else:
                err_msg = cpu_processes_data.get('error',"N/A") if isinstance(cpu_processes_data, dict) else "N/A"
                st.warning(f"CPU top processes data unavailable: {err_msg}", icon="⚠️")

        with detail_cols[1]:
            st.markdown("##### <p style='text-align: center;'>🎮 GPU(s) Detailed Status</p>", unsafe_allow_html=True)
            gpus_status_list = status_data.get('gpus', [])
            if gpus_status_list:
                for i, gpu in enumerate(gpus_status_list):
                    exp_title = (f"GPU {gpu.get('id','N/A')}: {gpu.get('name','N/A')} "
                                 f"(Util: {gpu.get('utilization_gpu',0):.1f}%)")
                    with st.expander(exp_title, expanded=True): # Keep GPUs expanded by default
                        mem_total = gpu.get('memory_total_mb', 1) # Avoid division by zero
                        mem_used = gpu.get('memory_used_mb', 0)
                        mem_util = (mem_used / mem_total) * 100 if mem_total > 0 else 0

                        gpu_metric_cols = st.columns(2)
                        with gpu_metric_cols[0]:
                            st.metric(label=f"GPU {gpu.get('id','N/A')} Utilization", value=f"{gpu.get('utilization_gpu',0):.1f}%")
                            if gpu.get('temperature_c') is not None:
                                st.metric(label=f"GPU {gpu.get('id','N/A')} Temp.", value=f"{gpu.get('temperature_c',0):.1f}°C")
                        with gpu_metric_cols[1]:
                            st.metric(label=f"VRAM Usage", value=f"{mem_util:.1f}%",
                                      help=f"{mem_used:.0f}MB / {mem_total:.0f}MB")
                        st.progress(int(mem_util), text=f"VRAM: {mem_used:.0f}MB / {mem_total:.0f}MB")

                        gpu_procs = gpu.get('processes', [])
                        if gpu_procs:
                            st.markdown("**Top Processes on this GPU:**")
                            df_gpu_procs = pd.DataFrame(gpu_procs)
                            cols_to_show_gpu = [col for col in ['pid', 'name', 'user', 'gpu_memory_mb'] if col in df_gpu_procs.columns]
                            st.dataframe(df_gpu_procs[cols_to_show_gpu], use_container_width=True, hide_index=True, height=150) # Adjust height
                        elif top_n_global > 0:
                            st.caption(f"No top processes reported on GPU {gpu.get('id','N/A')}.")
            elif 'gpus' in status_data and not status_data['gpus']:
                 st.info("No GPUs detected or reported by the server's API for real-time status.", icon="ℹ️")
            else:
                st.info("GPU status not available or no GPUs detected on this server.", icon="ℹ️")
    else:
        err_msg = status_data.get('error', 'Failed to load real-time status') if isinstance(status_data, dict) else "Failed to load real-time status"
        st.error(f"Could not load real-time status for {server_name}: {err_msg}", icon="🚨")


# --- Main Application Logic ---
if not GLOBAL_APP_STATE["monitored_servers"]:
    st.info("👋 Welcome to ServerMonitor! Please add servers using the 'Manage Servers' section in the sidebar to begin monitoring. All users will see the same list of servers and global settings.")
else:
    tab_titles = [s['name'] for s in GLOBAL_APP_STATE["monitored_servers"]]

    # Streamlit's st.tabs remembers the last selected tab *per session* by default.
    # If a tab (server) is removed, Streamlit handles this by selecting the first tab.
    created_tabs = st.tabs(tab_titles)

    for i, tab_widget in enumerate(created_tabs):
        with tab_widget:
            # Check if server index is still valid (e.g., if a server was removed by another user session)
            if i < len(GLOBAL_APP_STATE["monitored_servers"]):
                current_server_info = GLOBAL_APP_STATE["monitored_servers"][i]
                display_server_data(current_server_info, GLOBAL_APP_STATE["top_n_processes"])
            else:
                st.warning("This server tab is no longer valid, possibly removed. Please refresh or select another tab.", icon="⚠️")
                # This state should ideally be rare due to reruns on server list changes.