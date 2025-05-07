# ServerMonitor

Monitor resources of server(s), e.g., GPUs, RAM, and CPU utilization.

This application consists of a FastAPI backend (`backend.py`) to be run on each server you want to monitor, and a Streamlit frontend (`frontend.py`) to view the status of all configured servers.

## Project Structure
```bash
.
├── .venv/                     # Virtual environment (created by uv)
├── pyproject.toml             # Project configuration and dependencies
├── backend.py                 # FastAPI backend server code
├── frontend.py                # Streamlit frontend application code
└── README.md                  # This file
```

## Prerequisites

-   Python 3.12+
-   `uv` (Python package installer and virtual environment manager)
-   `nvidia-smi` command-line utility installed and in PATH on servers with NVIDIA GPUs that you wish to monitor.

## Setup Instructions

1.  **Clone/Download Project:**
    Get the project files (`pyproject.toml`, `backend.py`, `frontend.py`, `README.md`) into a local directory.

2.  **Initialize Project & Create Virtual Environment (using `uv`):**
    Open your terminal in the project's root directory.
    ```bash
    uv sync
    ```
## Running the Application

1.  **Run the Backend Server (`backend.py`):**
    -   Copy `backend.py` to each Linux server you want to monitor.
    -   On each of those servers, navigate to the directory containing `backend.py`.
    -   Ensure the packages are installed in the Python environment you use to run the script.
    -   Run the backend: `uvicorn backend:app --host 0.0.0.0 --port {PORT_ID}`
    -   The backend will typically listen on `http://0.0.0.0:{PORT_ID}`. Note the IP address and port for each server.
2.  **Run the Frontend Application (`frontend.py`):**
    -   On a machine where you want to view the dashboard (this can be your local machine or another server).
    -   Ensure the virtual environment (with `streamlit` and other frontend dependencies) is active.
    -   Navigate to the project root directory (where `frontend.py` is).
    -   Run the Streamlit app: `streamlit run frontend.py`.
    -   Streamlit will typically open the application in your web browser (e.g., at `http://localhost:{PORT_ID}`).
3.  **Using ServerMonitor:**
    -   In the Streamlit application (web interface), use the sidebar to "Add New Server".
    -   Enter a unique name for the server (e.g., "GPU Server 1") and its API URL (e.g., `http://<IP_OF_GPU_SERVER_1>:8801`).
    -   The frontend will attempt to connect and fetch configuration. If successful, the server will be added to the shared list.
    -   Added servers will appear as tabs in the main area.

**Notes:**
-   Ensure firewalls are configured to allow connections from the machine running `frontend.py` to the `backend.py` API ports on your monitored servers.
-   The `backend.py` relies on system commands like `nvidia-smi`. If these are not available or permissions are insufficient, some data may not be displayed.
-   The shared state in `frontend.py` (server list) is maintained for the lifetime of the Streamlit application process. If the Streamlit app restarts, this state will be reset unless persisted externally (which is beyond the current scope).
