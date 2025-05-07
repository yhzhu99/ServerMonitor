# frontend.py
import streamlit as st
import requests
import pandas as pd
import time
from streamlit_autorefresh import st_autorefresh # Ensure this is installed: uv pip install streamlit-autorefresh

# --- Page Configuration ---
st.set_page_config(
    page_title="ServerMonitor",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🖥️ ServerMonitor Dashboard")

# --- Helper Function to Call Backend API ---
def fetch_from_api_server(base_url: str, endpoint: str, params: dict = None):
    """Generic function to fetch data from a specific server's backend."""
    try:
        # Ensure base_url doesn't have trailing slash and endpoint has leading slash
        url = f"{base_url.rstrip('/')}{endpoint}"
        response = requests.get(url, params=params, timeout=10) # 10s timeout
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.Timeout:
        st.error(f"Timeout connecting to {base_url}{endpoint}", icon="🚨")
        return None
    except requests.exceptions.ConnectionError:
        st.error(f"Connection error to {base_url}{endpoint}. Is the backend server running and accessible?", icon="🚨")
        return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data from {base_url}{endpoint}: {e}", icon="🚨")
        return None

# --- Session State Initialization ---
if 'monitored_servers' not in st.session_state:
    # List of dictionaries: {'name': str, 'url': str}
    st.session_state.monitored_servers = []
if 'active_server_tab' not in st.session_state:
     st.session_state.active_server_tab = None


# --- Sidebar for Controls & Server Management ---
st.sidebar.header("Global Controls")

refresh_intervals = {"Off": 0, "1s": 1, "5s": 5, "10s": 10, "30s": 30, "60s": 60}
selected_interval_label = st.sidebar.selectbox(
    "Refresh Interval",
    options=list(refresh_intervals.keys()),
    index=3,  # Default to 10 seconds
    help="Set how often the data for the active tab(s) should refresh."
)
refresh_interval_seconds = refresh_intervals[selected_interval_label]

if refresh_interval_seconds > 0:
    st_autorefresh(interval=refresh_interval_seconds * 1000, key="global_data_refresher")

top_n_options = [1, 2, 3, 4, 5, 10]
top_n_processes = st.sidebar.selectbox(
    "Top N Processes",
    options=top_n_options,
    index=2,  # Default to Top 3
    help="Number of top CPU/GPU processes to display for each server."
)

st.sidebar.divider()
st.sidebar.header("Manage Servers")

with st.sidebar.form("add_server_form", clear_on_submit=True):
    st.subheader("Add New Server")
    new_server_name = st.text_input("Server Name (e.g., Prod-Server-1)", placeholder="My Web Server")
    new_server_url = st.text_input("Server API URL (e.g., http://192.168.1.100:8801)", placeholder="http://localhost:8801")
    add_server_submitted = st.form_submit_button("Add Server")

    if add_server_submitted:
        if new_server_name and new_server_url:
            if not (new_server_url.startswith("http://") or new_server_url.startswith("https://")):
                st.sidebar.error("URL must start with http:// or https://")
            else:
                # Check for duplicates
                if any(s['url'] == new_server_url.rstrip('/') for s in st.session_state.monitored_servers) or \
                   any(s['name'] == new_server_name for s in st.session_state.monitored_servers):
                    st.sidebar.warning("Server with this name or URL already exists.")
                else:
                    st.session_state.monitored_servers.append(
                        {'name': new_server_name, 'url': new_server_url.rstrip('/')}
                    )
                    st.sidebar.success(f"Added server: {new_server_name}")
                    # st.experimental_rerun() # Rerun to update tabs; form submission already does this.
        else:
            st.sidebar.error("Both server name and URL are required.")

if st.session_state.monitored_servers:
    st.sidebar.subheader("Monitored Servers")
    servers_to_remove = []
    for i, server in enumerate(st.session_state.monitored_servers):
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        col1.text(f"{server['name']}\n({server['url']})")
        if col2.button("🗑️", key=f"remove_server_{i}", help=f"Remove {server['name']}"):
            servers_to_remove.append(i)

    if servers_to_remove:
        # Remove in reverse order to avoid index issues
        for i in sorted(servers_to_remove, reverse=True):
            removed_server = st.session_state.monitored_servers.pop(i)
            st.sidebar.toast(f"Removed server: {removed_server['name']}")
        if not st.session_state.monitored_servers: # If last server removed
            st.session_state.active_server_tab = None
        elif st.session_state.active_server_tab not in [s['name'] for s in st.session_state.monitored_servers]:
            # If active tab was removed, reset (or set to first available)
             st.session_state.active_server_tab = st.session_state.monitored_servers[0]['name'] if st.session_state.monitored_servers else None
        st.experimental_rerun()


# --- Cached Data Fetching Functions ---
@st.cache_data(ttl=300) # Cache for 5 minutes per server_url
def get_server_config_cached(server_url: str):
    return fetch_from_api_server(server_url, "/api/server/config")

# --- Main Display Area for Servers ---
def display_server_data(server_info: dict, top_n: int):
    """Displays monitoring data for a single server."""
    base_url = server_info['url']
    server_name = server_info['name']

    # --- Server Configuration Section ---
    st.subheader(f"⚙️ Configuration Details", anchor=False)
    config_data = get_server_config_cached(base_url) # Use cached version

    if config_data:
        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            st.markdown(f"**Server Name:** `{config_data['server_name']}`")
            st.markdown(f"**Server IP:** `{config_data['server_ip']}`")
            st.markdown(f"**Total RAM:** {config_data['memory_total_gb']:.2f} GB")

        with col_cfg2:
            with st.expander("CPU Configuration", expanded=False):
                st.markdown(f"- **Model:** {config_data['cpus']['model_name']}")
                st.markdown(f"- **Physical Cores:** {config_data['cpus']['physical_cores']}")
                st.markdown(f"- **Logical Cores:** {config_data['cpus']['logical_cores']}")

        if config_data['gpus']:
            with st.expander("GPU Configuration", expanded=False):
                for i, gpu in enumerate(config_data['gpus']):
                    st.markdown(f"--- \n **GPU {gpu['id']}: {gpu['name']}**")
                    st.markdown(f"  - UUID: `{gpu['uuid']}`")
                    st.markdown(f"  - Total Memory: {gpu['memory_total_mb']:.0f} MB")
        else:
            st.info("No GPUs detected in configuration or `nvidia-smi` might not be available on the server.")
    else:
        st.warning(f"Could not load server configuration for {server_name}. The server might be offline or the API endpoint is incorrect.", icon="⚠️")
        return # Stop further display for this server if config fails

    st.divider()

    # --- Real-time Status Section ---
    st.subheader("📊 Real-time Resource Usage", anchor=False)
    status_data = fetch_from_api_server(base_url, "/api/server/status", params={"top_n_gpu_processes": top_n})

    if status_data:
        layout_cols = st.columns([1, 1]) # For CPU/RAM summary
        with layout_cols[0]: # CPU Usage
            cpu_util = status_data['cpu_utilization_percent']
            st.progress(int(cpu_util), text=f"CPU Utilization: {cpu_util:.1f}%")
        with layout_cols[1]: # RAM Usage
            ram_util = status_data['ram_utilization_percent']
            ram_text = f"RAM: {ram_util:.1f}% ({status_data['ram_used_gb']:.2f} GB / {status_data['ram_total_gb']:.2f} GB)"
            st.progress(int(ram_util), text=ram_text)

        st.markdown("---") # Visual separator

        # Detailed CPU and GPU Usage
        detail_cols = st.columns(2)
        with detail_cols[0]:
            st.subheader("🚀 CPU Top Processes", anchor=False)
            cpu_processes_data = fetch_from_api_server(base_url, "/api/cpu/top_processes", params={"n": top_n})
            if cpu_processes_data is not None:
                if cpu_processes_data: # If list is not empty
                    df_cpu_procs = pd.DataFrame(cpu_processes_data)
                    df_cpu_procs = df_cpu_procs[['pid', 'name', 'user', 'cpu_percent', 'memory_percent']]
                    st.dataframe(df_cpu_procs, use_container_width=True, hide_index=True)
                else:
                    st.info("No significant CPU processes found or reported.")
            else:
                st.warning("Could not load CPU top processes data.", icon="⚠️")

        with detail_cols[1]:
            st.subheader("🎮 GPU(s) Detailed Status", anchor=False)
            if status_data.get('gpus'):
                for i, gpu in enumerate(status_data['gpus']):
                    exp_title = (f"GPU {gpu['id']}: {gpu['name']} "
                                 f"(Util: {gpu['utilization_gpu']:.1f}%)")
                    with st.expander(exp_title, expanded=True):
                        mem_util = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100 if gpu['memory_total_mb'] > 0 else 0

                        gpu_cols = st.columns(2)
                        with gpu_cols[0]:
                            st.metric(label=f"GPU {gpu['id']} Utilization", value=f"{gpu['utilization_gpu']:.1f}%")
                            if gpu.get('temperature_c') is not None:
                                st.metric(label=f"GPU {gpu['id']} Temperature", value=f"{gpu['temperature_c']:.1f}°C")
                        with gpu_cols[1]:
                            st.metric(label=f"GPU {gpu['id']} VRAM Usage", value=f"{mem_util:.1f}%",
                                      help=f"{gpu['memory_used_mb']:.0f} MB / {gpu['memory_total_mb']:.0f} MB")

                        st.progress(int(mem_util), text=f"VRAM: {gpu['memory_used_mb']:.0f}MB / {gpu['memory_total_mb']:.0f}MB")

                        if gpu.get('processes'):
                            st.markdown("**Top Processes on this GPU:**")
                            df_gpu_procs = pd.DataFrame([p for p in gpu['processes']]) # Ensure it's a list of dicts
                            df_gpu_procs = df_gpu_procs[['pid', 'name', 'user', 'gpu_memory_mb']]
                            st.dataframe(df_gpu_procs, use_container_width=True, hide_index=True)
                        elif top_n > 0 : # Only show if user expects processes
                            st.info(f"No active processes found on GPU {gpu['id']} meeting criteria.")
            elif 'gpus' in status_data and not status_data['gpus']: # API returned gpus: []
                 st.info("No GPUs detected or reported by the server's API.")
            else: # Fallback if 'gpus' key missing or other issue with status_data for gpus
                st.info("GPU status not available or no GPUs detected on this server.")
    else:
        st.error(f"Could not load real-time status for {server_name}. The server might be offline or experiencing issues.", icon="🚨")


if not st.session_state.monitored_servers:
    st.info("👋 Welcome to ServerMonitor! Please add servers using the 'Manage Servers' section in the sidebar to begin monitoring.")
else:
    tab_titles = [s['name'] for s in st.session_state.monitored_servers]

    # Handle tab selection persistence if needed, though Streamlit tabs reset selection on most actions
    # For simplicity, default to first tab or last known if still valid
    if st.session_state.active_server_tab not in tab_titles and tab_titles:
        st.session_state.active_server_tab = tab_titles[0]

    # Create tabs. The `st.tabs` function itself doesn't have a default selected index parameter directly.
    # It activates the first tab by default.
    created_tabs = st.tabs(tab_titles)

    for i, tab_widget in enumerate(created_tabs):
        with tab_widget:
            current_server_info = st.session_state.monitored_servers[i]
            # Update active tab name based on which tab is currently being rendered by Streamlit
            # This is more for logical tracking if we needed it, st.tabs handles display.
            # st.session_state.active_server_tab = current_server_info['name']

            # Display data for the server associated with this tab
            display_server_data(current_server_info, top_n_processes)