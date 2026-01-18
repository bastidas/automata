# Acinonyx 🐆

A Python application for simulating and visualizing mechanical linkage systems - fast and agile like a cheetah.

This project provides tools for creating, simulating, and visualizing mechanical linkage systems with exceptional performance. It includes:

- **Mechanical Link Simulation**: Define and simulate multi-link mechanical systems using pylinkage
- **Visualization Tools**: Generate static plots and animations of linkage motion using a React frontend
- **REST API**: FastAPI backend for graph operations and trajectory computation

## Requirements

- Python >= 3.11
- Node.js >= 18 (for frontend)
- Conda (recommended) or pip
- See `pyproject.toml` for complete dependency list

## Configuration

The application uses `configs/appconfig.py` for centralized configuration:

- **USER_DIR**: Location for saved graphs and user data (`user/` directory)
- **BACKEND_PORT**: `8021` - FastAPI backend server port
- **FRONTEND_PORT**: `5173` - Vite development server port
- **API URLs**: Automatically constructed from port configuration

All port settings are centralized in `appconfig.py` - modify this file if you need to change ports due to conflicts.

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd automata
```

### 2. Set Up Python Environment

#### Option A: Using Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate automata
```

#### Option B: Using pip

```bash
# Create virtual environment
python -m venv venv

# Activate environment
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate

# Install dependencies
pip install -e .
```

### 3. Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### 4. Verify Installation

Run the test suite to ensure everything is working:

```bash
# Using pytest
pytest tests/ -v

# Or run a specific test
pytest tests/test_pylink_trajectory.py -v
```

All tests should pass without errors.

## Usage

### Quick Start

Use the provided run script to start both backend and frontend:

```bash
# On Unix/MacOS:
chmod +x run.sh
./run.sh
```

The script will:
1. Start the FastAPI backend server on `http://localhost:8021` (configured in `configs/appconfig.py`)
2. Start the Vite development server on `http://localhost:5173`
3. Open the web GUI in your browser

### Manual Start

If you prefer to run the servers manually:

#### Terminal 1 - Backend Server

```bash
# Activate your environment first
conda activate automata  # or: source venv/bin/activate

# Start backend
cd backend
python run_server.py
```

The API will be available at `http://localhost:8021` (port configured in `configs/appconfig.py`)

#### Terminal 2 - Frontend Server

```bash
cd frontend
npm run dev
```

The web interface will be available at `http://localhost:5173`

### API Documentation

Once the backend is running, visit:
- Swagger UI: `http://localhost:8021/docs`
- ReDoc: `http://localhost:8021/redoc`
- Status Check: `http://localhost:8021/status`

**Note**: The backend port (8021) is configured in `configs/appconfig.py` and can be changed if needed.

## Project Structure

```
automata/
├── backend/
│   ├── query_api.py          # FastAPI endpoints
│   └── run_server.py         # Backend server entry point
├── configs/
│   ├── appconfig.py          # Centralized config (ports, paths)
│   ├── link_models.py        # Pydantic models for links/nodes
│   └── matplotlib_config.py  # Matplotlib backend config
├── frontend/
│   ├── src/
│   │   ├── components/       # React components
│   │   ├── App.tsx           # Main application
│   │   └── main.tsx          # Entry point
│   ├── package.json
│   └── vite.config.ts
├── link/                     # Core linkage utilities
├── tests/
│   ├── test_pylink_trajectory.py  # Trajectory computation tests
│   ├── test_matplotlib_backend.py # Backend configuration tests
│   └── 4bar_test.json            # Test data
├── pyproject.toml            # Python project configuration
└── README.md
```

## Testing

### Run All Tests

```bash
conda activate automata
pytest tests/ -v
```

### Run Specific Test Files

```bash
# Test trajectory computation
pytest tests/test_pylink_trajectory.py -v

# Test with output visible
pytest tests/test_pylink_trajectory.py -s -v
```

### Run Tests with Coverage

```bash
pytest tests/ --cov=backend --cov=link --cov-report=html
```

Coverage report will be generated in `htmlcov/index.html`

## Available Endpoints

The backend provides the following REST API endpoints:

### Pylink Graph Management
- `POST /save-pylink-graph` - Save a pylink graph
- `GET /load-pylink-graph` - Load a saved graph
- `GET /list-pylink-graphs` - List all saved graphs
- `POST /compute-pylink-trajectory` - Compute joint trajectories

### Force Graph
- `GET /load-last-force-graph` - Load most recent force graph visualization

### Demo & Utilities
- `POST /demo-4bar-pylinkage` - Generate demo 4-bar linkage
- `POST /convert-to-pylinkage` - Convert graph to pylinkage format
- `POST /simulate-pylinkage` - Run simulation using pylinkage
- `POST /compare-solvers` - Compare different solver implementations

## Development

### Code Style

The project uses standard Python and TypeScript formatting:

```bash
# Format Python code
black backend/ configs/ link/ tests/

# Format TypeScript/React code
cd frontend
npm run format
```

### Adding New Tests

Create test files in the `tests/` directory following the naming convention `test_*.py`:

```python
def test_your_feature():
    """Test description"""
    # Your test code
    assert True
```

## Troubleshooting

### Common Issues

**1. Module not found errors**
- Ensure your conda environment is activated: `conda activate automata`
- Verify installation: `pip list | grep automata`

**2. Frontend not loading**
- Check Node.js version: `node --version` (should be >= 18)
- Reinstall dependencies: `cd frontend && npm install`

**3. Backend connection refused**
- Verify backend is running: `curl http://localhost:8021/status`
- Check for port conflicts (default port is 8021, see `configs/appconfig.py`)
- Ensure CORS is properly configured

**4. Tests failing**
- Ensure pytest is installed: `pip install pytest pytest-cov`
- Check Python version: `python --version` (should be >= 3.11)

**5. Port conflicts**
- If ports 8021 or 5173 are already in use, edit `configs/appconfig.py` to change ports
- Restart both backend and frontend servers after changing ports

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests to ensure they pass
5. Submit a pull request

## License

TBD







