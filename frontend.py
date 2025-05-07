# frontend.py
import streamlit as st
import requests
import pandas as pd
import time
from streamlit_autorefresh import st_autorefresh # Ensure this is installed

# --- Configuration ---
# This should point to your FastAPI backend URL
# If running backend.py on the same machine:
BACKEND_URL = "http://localhost:8801/api"
# If backend.py is on a remote server 'myserver.com':
# BACKEND_URL = "http://myserver.com:8801/api"

# --- Helper Functions to Call Backend API ---
def fetch_from_backend(endpoint: str, params: dict = None):
    """Generic function to fetch data from the backend."""
    try:
        response = requests.get(f"{BACKEND_URL}{endpoint}", params=params, timeout=10)
        response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend ({endpoint}): {e}")
        return None

def post_to_backend(endpoint: str, params: dict = None, data: dict = None):
    """Generic function to post data to the backend."""
    try:
        response = requests.post(f"{BACKEND_URL}{endpoint}", params=params, json=data, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to backend ({endpoint}): {e}")
        return None

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="ServerMonitor",
    page_icon="🖥️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🖥️ ServerMonitor Dashboard")

# --- Sidebar for Controls ---
st.sidebar.header("Controls")

# Auto-refresh interval
refresh_intervals = {"Off": 0, "1 Second": 1, "5 Seconds": 5, "10 Seconds": 10, "30 Seconds": 30, "60 Seconds": 60}
selected_interval_label = st.sidebar.selectbox(
    "Refresh Interval",
    options=list(refresh_intervals.keys()),
    index=3 # Default to 10 seconds
)
refresh_interval_seconds = refresh_intervals[selected_interval_label]

if refresh_interval_seconds > 0:
    st_autorefresh(interval=refresh_interval_seconds * 1000, key="data_refresher")

# Top N processes
top_n_options = [1, 2, 3, 4, 5, 10]
top_n_processes = st.sidebar.selectbox(
    "Top N Processes to Display",
    options=top_n_options,
    index=2 # Default to Top 3
)

# --- Main Page Display ---

# Column layout for better organization
col_config, col_status = st.columns(2)

# --- Server Configuration Section ---
with col_config:
    st.subheader("Server Configuration")
    config_data = fetch_from_backend("/server/config")
    if config_data:
        st.markdown(f"**Server Name:** `{config_data['server_name']}`")
        st.markdown(f"**Server IP:** `{config_data['server_ip']}`")

        with st.expander("CPU Configuration", expanded=False):
            st.markdown(f"- **Model:** {config_data['cpus']['model_name']}")
            st.markdown(f"- **Physical Cores:** {config_data['cpus']['physical_cores']}")
            st.markdown(f"- **Logical Cores:** {config_data['cpus']['logical_cores']}")

        st.markdown(f"**Total RAM:** {config_data['memory_total_gb']:.2f} GB")

        if config_data['gpus']:
            with st.expander("GPU Configuration", expanded=False):
                for i, gpu in enumerate(config_data['gpus']):
                    st.markdown(f"--- \n **GPU {gpu['id']}: {gpu['name']}**")
                    st.markdown(f"  - UUID: `{gpu['uuid']}`")
                    st.markdown(f"  - Total Memory: {gpu['memory_total_mb']:.0f} MB")
        else:
            st.info("No GPUs detected or `nvidia-smi` not available on the server.")
    else:
        st.warning("Could not load server configuration.")

# --- Real-time Status Section ---
with col_status:
    st.subheader("Real-time Resource Usage")
    status_data = fetch_from_backend("/server/status", params={"top_n_gpu_processes": top_n_processes})
    if status_data:
        # CPU Usage
        cpu_util = status_data['cpu_utilization_percent']
        st.progress(int(cpu_util), text=f"CPU Utilization: {cpu_util:.1f}%")

        # RAM Usage
        ram_util = status_data['ram_utilization_percent']
        ram_text = f"RAM Usage: {ram_util:.1f}% ({status_data['ram_used_gb']:.2f} GB / {status_data['ram_total_gb']:.2f} GB)"
        st.progress(int(ram_util), text=ram_text)
    else:
        st.warning("Could not load real-time status.")


# --- Detailed CPU and GPU Usage ---
st.divider()
col_cpu_details, col_gpu_details = st.columns(2)

with col_cpu_details:
    st.subheader("CPU Top Processes")
    if status_data: # Re-use status_data if available, or fetch specifically
        cpu_processes_data = fetch_from_backend("/cpu/top_processes", params={"n": top_n_processes})
        if cpu_processes_data:
            if cpu_processes_data:
                df_cpu_procs = pd.DataFrame(cpu_processes_data)
                df_cpu_procs = df_cpu_procs[['pid', 'name', 'user', 'cpu_percent', 'memory_percent']]
                st.dataframe(df_cpu_procs, use_container_width=True, hide_index=True)
            else:
                st.info("No active CPU processes found meeting criteria.")
        else:
            st.warning("Could not load CPU top processes.")

with col_gpu_details:
    st.subheader("GPU(s) Detailed Status")
    if status_data and status_data.get('gpus'):
        for i, gpu in enumerate(status_data['gpus']):
            with st.expander(f"GPU {gpu['id']}: {gpu['name']} (Util: {gpu['utilization_gpu']:.1f}%)", expanded=True):
                mem_util = (gpu['memory_used_mb'] / gpu['memory_total_mb']) * 100 if gpu['memory_total_mb'] > 0 else 0

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="GPU Utilization", value=f"{gpu['utilization_gpu']:.1f}%")
                    if gpu.get('temperature_c') is not None:
                        st.metric(label="Temperature", value=f"{gpu['temperature_c']:.1f}°C")
                with col2:
                    st.metric(label="Memory Usage", value=f"{mem_util:.1f}%",
                              help=f"{gpu['memory_used_mb']:.0f} MB / {gpu['memory_total_mb']:.0f} MB")

                st.progress(int(mem_util), text=f"VRAM: {gpu['memory_used_mb']:.0f}MB / {gpu['memory_total_mb']:.0f}MB")

                if gpu['processes']:
                    st.markdown("**Top Processes on this GPU:**")
                    df_gpu_procs = pd.DataFrame(gpu['processes'])
                    # Select and reorder columns for display
                    df_gpu_procs = df_gpu_procs[['pid', 'name', 'user', 'gpu_memory_mb']]
                    st.dataframe(df_gpu_procs, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No active processes found on GPU {gpu['id']} meeting criteria.")
    elif status_data: # status_data exists but no GPUs
        st.info("No GPUs detected or `nvidia-smi` not available on the server.")
    else:
        st.warning("Could not load GPU status.")


# --- Reporting Section ---
st.divider()
st.header("📊 Reports")

col_report_action, col_report_view = st.columns([1, 2])

with col_report_action:
    st.subheader("Generate Report")
    report_period = st.selectbox("Report Period", ["daily", "weekly"])
    if st.button(f"Generate {report_period.capitalize()} Report"):
        with st.spinner(f"Generating {report_period} report..."):
            response = post_to_backend("/reports/generate", params={"period": report_period})
            if response:
                st.success(response.get("message", "Report generation initiated."))
                # Force a refresh of the report list by clearing cache or re-fetching
                # For simplicity, we just inform the user. A more robust way would be to
                # update st.session_state or use st.experimental_rerun() after a short delay.
                st.toast("Report list will update on next refresh.")
            else:
                st.error("Failed to generate report.")

with col_report_view:
    st.subheader("View & Download Reports")
    reports_list_data = fetch_from_backend("/reports/list")
    if reports_list_data and reports_list_data.get("reports"):
        available_reports = reports_list_data["reports"]
        selected_report = st.selectbox("Select a report to view/download:", available_reports)

        if selected_report:
            col_view, col_download = st.columns(2)
            with col_view:
                if st.button("View Report Content", key=f"view_{selected_report}"):
                    st.session_state.report_to_display = selected_report # Store for display below
            with col_download:
                # Create download link using backend endpoint
                report_download_url = f"{BACKEND_URL}/reports/download/{selected_report}"
                st.markdown(f'<a href="{report_download_url}" download="{selected_report}" class="stButton"><button>Download {selected_report}</button></a>', unsafe_allow_html=True)

            if 'report_to_display' in st.session_state and st.session_state.report_to_display == selected_report:
                report_content_data = fetch_from_backend(f"/reports/view/{selected_report}")
                if report_content_data:
                    with st.expander("Report Content", expanded=True):
                        st.markdown(report_content_data["content"], unsafe_allow_html=True)
                else:
                    st.warning(f"Could not load content for {selected_report}")
    else:
        st.info("No reports available yet. Generate one first!")