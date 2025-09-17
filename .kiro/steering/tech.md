# Technology Stack & Build System

## Core Technologies

- **Python 3.9+** (3.10+ recommended, 3.12 optimal)
- **MCP (Model Context Protocol)** - Anthropic's protocol for AI tool communication
- **Pydantic 2.0+** - Data validation and settings management
- **asyncio** - Asynchronous programming for concurrent agent operations

## AI Provider SDKs

- **google-genai** - Gemini model integration
- **openai** - OpenAI model integration  
- **httpx** - HTTP client for API communications
- **python-dotenv** - Environment variable management

## Development Tools

- **pytest** - Testing framework with asyncio support
- **black** - Code formatting (120 char line length)
- **isort** - Import sorting
- **ruff** - Fast Python linter
- **pre-commit** - Git hooks for code quality

## Build System

Uses **setuptools** with **pyproject.toml** configuration:
- Entry point: `zen-mcp-server = "server:run"`
- Package discovery includes: `tools*`, `providers*`, `systemprompts*`, `utils*`, `conf*`
- Data files: `conf/custom_models.json`

## Common Commands

### Development Setup
```bash
# Clone and setup
git clone <repo-url>
cd zen-mcp-server
python -m venv .zen_venv
source .zen_venv/bin/activate  # or .zen_venv\Scripts\activate on Windows
pip install -r requirements-dev.txt

# Copy environment template
cp .env.example .env
# Edit .env with your API keys
```

### Running the Server
```bash
# Direct execution
python server.py

# Via setuptools entry point
zen-mcp-server

# Via uvx (for distribution)
uvx --from git+https://github.com/BeehiveInnovations/zen-mcp-server.git zen-mcp-server
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test categories
pytest tests/test_*.py
pytest simulator_tests/test_*.py

# Integration tests
./run_integration_tests.sh  # Unix
./run_integration_tests.ps1  # Windows
```

### Code Quality
```bash
# Format code
black .
isort .

# Lint code
ruff check .

# Pre-commit hooks
pre-commit install
pre-commit run --all-files
```

### Docker Deployment
```bash
# Build and run
docker-compose up --build

# Health check
python docker/scripts/healthcheck.py

# Deploy scripts
./docker/scripts/deploy.sh  # Unix
./docker/scripts/deploy.ps1  # Windows
```

## Configuration Management

- **Environment Variables**: Primary configuration via `.env` file
- **Model Restrictions**: `OPENAI_ALLOWED_MODELS`, `GOOGLE_ALLOWED_MODELS`, etc.
- **Tool Configuration**: `DISABLED_TOOLS` for selective tool enabling
- **Logging**: Configurable via `LOG_LEVEL` (DEBUG/INFO/WARNING/ERROR)
- **Custom Models**: `conf/custom_models.json` for model definitions

## Architecture Patterns

- **Provider Pattern**: Abstracted AI model providers with unified interface
- **Tool Pattern**: Modular tools inheriting from `BaseTool`
- **Registry Pattern**: `ModelProviderRegistry` for provider management
- **Factory Pattern**: Dynamic provider instantiation based on API keys
- **Observer Pattern**: Conversation memory and context tracking