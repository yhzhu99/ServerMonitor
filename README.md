# ServerMonitor

Monitor resources of server(s), e.g., GPUs, RAM, and CPU utilization.

This application consists of a FastAPI backend (`backend.py`) to be run on each server you want to monitor, and a Streamlit frontend (`frontend.py`) to view the status of all configured servers.

**Features:**
-   View configuration (IP, Name, CPU, GPU, Memory) for multiple Linux servers.
-   Real-time monitoring of CPU utilization, RAM usage, and GPU (NVIDIA) utilization and memory.
-   Display top N CPU-consuming processes (with user and name).
-   Display top N GPU-memory-consuming processes (with user and name) for each GPU.
-   Configurable data refresh interval (global setting for all users).
-   Configurable number of "Top N" processes to display (global setting for all users).
-   Shared server list: All users connected to the same Streamlit frontend instance will see and manage the same list of monitored servers.
-   Tabbed interface to switch between monitored servers.
-   Optimized data fetching using shared caches to reduce redundant calls to backend servers when multiple users are viewing.

**Architecture:**
-   **Backend**: FastAPI application (`backend.py`) runs on each monitored Linux server. It exposes API endpoints to query system configuration and real-time resource usage by executing local commands (e.g., `nvidia-smi`, `psutil`).
-   **Frontend**: Streamlit application (`frontend.py`) provides the user interface. Users add the API URLs of the backend servers. The frontend then polls these backends to display the monitoring data. All settings (server list, refresh rate, Top-N) are shared across all connected users of the same Streamlit instance.

**Project Structure:**
```bash
.
├── .venv/                     # Virtual environment (created by uv)
├── pyproject.toml             # Project configuration and dependencies
├── backend.py                 # FastAPI backend server code
├── frontend.py                # Streamlit frontend application code
└── README.md                  # This file
```

**Prerequisites:**
-   Python 3.12+
-   `uv` (Python package installer and virtual environment manager)
-   `nvidia-smi` command-line utility installed and in PATH on servers with NVIDIA GPUs that you wish to monitor.

**Setup Instructions:**

1.  **Clone/Download Project:**
    Get the project files (`pyproject.toml`, `backend.py`, `frontend.py`, `README.md`) into a local directory.

2.  **Initialize Project & Create Virtual Environment (using `uv`):**
    Open your terminal in the project's root directory.
    ```bash
    # Initialize uv project (if not already done, primarily for uv-specific project configs)
    uv init --no-prompt
    # (Note: `uv init` is more for setting up a publishable package. For simple apps, it's optional if you manually create pyproject.toml)

    # Create a virtual environment named .venv using Python 3.12
    uv venv .venv --python=3.12

    # Activate the virtual environment
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows (PowerShell):
    # .\.venv\Scripts\Activate.ps1
    # On Windows (CMD):
    # .\.venv\Scripts\activate.bat
    ```

3.  **Install Dependencies (using `uv`):**
    Ensure your virtual environment is activated. `uv` will read `pyproject.toml`.
    ```bash
    uv pip sync
    ```
    This command installs all dependencies listed in `pyproject.toml`.

**Running the Application:**

1.  **Run the Backend Server (`backend.py`):**
    -   Copy `backend.py` to each Linux server you want to monitor.
    -   On each of those servers, navigate to the directory containing `backend.py`.
    -   Ensure the virtual environment (with `fastapi`, `uvicorn`, `psutil`) is active on each server, or that these packages are installed in the Python environment you use to run the script. You might need to replicate the `uv venv` and `uv pip sync` steps on each server or ensure the necessary packages are globally available/managed by another means.
    -   Run the backend:
        ```bash
        python backend.py
        ```
        Or using uvicorn for more control (especially in production):
        ```bash
        uvicorn backend:app --host 0.0.0.0 --port 8801
        ```
        The backend will typically listen on `http://0.0.0.0:8801`. Note the IP address and port for each server.

2.  **Run the Frontend Application (`frontend.py`):**
    -   On a machine where you want to view the dashboard (this can be your local machine or another server).
    -   Ensure the virtual environment (with `streamlit` and other frontend dependencies) is active.
    -   Navigate to the project root directory (where `frontend.py` is).
    -   Run the Streamlit app:
        ```bash
        streamlit run frontend.py
        ```
    -   Streamlit will typically open the application in your web browser (e.g., at `http://localhost:8501`).

3.  **Using ServerMonitor:**
    -   In the Streamlit application (web interface), use the sidebar to "Add New Server".
    -   Enter a unique name for the server (e.g., "GPU Server 1") and its API URL (e.g., `http://<IP_OF_GPU_SERVER_1>:8801`).
    -   The frontend will attempt to connect and fetch configuration. If successful, the server will be added to the shared list.
    -   Added servers will appear as tabs in the main area.
    -   Use the "Global Controls" in the sidebar to set the refresh interval and the number of top processes to display. These settings apply to all users viewing this Streamlit instance.

**Notes:**
-   Ensure firewalls are configured to allow connections from the machine running `frontend.py` to the `backend.py` API ports on your monitored servers.
-   The `backend.py` relies on system commands like `nvidia-smi`. If these are not available or permissions are insufficient, some data may not be displayed.
-   The shared state in `frontend.py` (server list, global settings) is maintained for the lifetime of the Streamlit application process. If the Streamlit app restarts, this state will be reset unless persisted externally (which is beyond the current scope).
