# Project Structure & Organization

## Root Directory Layout

```
zen-mcp-server/
├── server.py              # Main MCP server entry point
├── config.py              # Central configuration and constants
├── pyproject.toml          # Build configuration and dependencies
├── requirements.txt        # Runtime dependencies
├── requirements-dev.txt    # Development dependencies
├── .env.example           # Environment variable template
├── docker-compose.yml     # Docker deployment configuration
└── README.md              # Project documentation
```

## Core Modules

### `/tools/` - AI Tool Implementations
- **Purpose**: Individual AI-powered tools that provide specialized functionality
- **Base Class**: All tools inherit from `tools/shared/base_tool.py:BaseTool`
- **Key Files**:
  - `__init__.py` - Tool registry exports
  - `shared/base_tool.py` - Abstract base class with common functionality
  - `shared/base_models.py` - Shared data models
  - `workflow/` - Workflow-based tools (multi-step processes)
  - `simple/` - Simple single-step tools

**Tool Categories**:
- **Collaboration**: `chat.py`, `thinkdeep.py`, `planner.py`, `consensus.py`
- **Code Quality**: `codereview.py`, `precommit.py`, `debug.py`
- **Analysis**: `analyze.py`, `refactor.py`, `testgen.py`, `secaudit.py`
- **Utilities**: `challenge.py`, `listmodels.py`, `version.py`

### `/providers/` - AI Model Provider Abstractions
- **Purpose**: Unified interface for different AI model providers
- **Base Class**: `base.py:ModelProvider`
- **Registry**: `registry.py:ModelProviderRegistry` manages provider instances
- **Providers**:
  - `gemini.py` - Google Gemini models
  - `openai_provider.py` - OpenAI models (GPT, O3)
  - `xai.py` - X.AI GROK models
  - `dial.py` - DIAL platform models
  - `openrouter.py` - OpenRouter unified access
  - `custom.py` - Local/custom API endpoints

### `/systemprompts/` - AI System Prompts
- **Purpose**: Tool-specific system prompts that define AI behavior
- **Pattern**: Each tool has corresponding `{tool_name}_prompt.py`
- **Content**: Role definitions, instructions, and behavioral guidelines

### `/utils/` - Shared Utilities
- **Purpose**: Common functionality used across tools and providers
- **Key Modules**:
  - `conversation_memory.py` - Thread-based conversation tracking
  - `file_utils.py` - File reading and processing utilities
  - `model_context.py` - Token management and context handling
  - `agent_communication.py` - Inter-agent communication protocols
  - `security_config.py` - Security and validation utilities

### `/conf/` - Configuration Data
- **Purpose**: Static configuration files and model definitions
- **Files**:
  - `custom_models.json` - Custom model configurations
  - `__init__.py` - Configuration module exports

## Testing Structure

### `/tests/` - Unit and Integration Tests
- **Pattern**: `test_{module_name}.py` for each module
- **Categories**:
  - Provider tests: `test_*_provider.py`
  - Tool tests: `test_{tool_name}.py`
  - Integration tests: `test_*_integration.py`
  - Configuration tests: `test_config.py`

### `/simulator_tests/` - AI Behavior Simulation Tests
- **Purpose**: Test AI tool behavior with simulated responses
- **Pattern**: `test_{tool_name}_validation.py`
- **Base Classes**: `base_test.py`, `conversation_base_test.py`

## Documentation Structure

### `/docs/` - Comprehensive Documentation
- **Getting Started**: `getting-started.md`, `configuration.md`
- **Architecture**: `agent-architecture.md`, `enhanced-parallel-thinking.md`
- **Tool Docs**: `tools/{tool_name}.md` for each tool
- **Deployment**: `docker-deployment.md`, `wsl-setup.md`
- **Development**: `contributions.md`, `testing.md`

## Deployment Structure

### `/docker/` - Container Deployment
- **Scripts**: `scripts/build.sh`, `scripts/deploy.sh`, `scripts/healthcheck.py`
- **Documentation**: `README.md` with deployment instructions

### `/scripts/` - Utility Scripts
- **Version Management**: `sync_version.py`
- **Cross-platform**: Both `.sh` and `.ps1` versions for compatibility

## Logging and Monitoring

### `/logs/` - Runtime Logs
- **Server Logs**: `mcp_server.log` (rotated, 20MB max)
- **Activity Logs**: `mcp_activity.log` (MCP protocol activity)
- **Rotation**: Automatic log rotation with configurable size limits

## Naming Conventions

### Files and Directories
- **Snake Case**: `file_name.py`, `directory_name/`
- **Tool Files**: Match tool name exactly (e.g., `codereview.py` for `codereview` tool)
- **Test Files**: Prefix with `test_` (e.g., `test_codereview.py`)

### Python Code
- **Classes**: PascalCase (e.g., `CodeReviewTool`, `ModelProvider`)
- **Functions/Methods**: snake_case (e.g., `get_model_provider`, `validate_request`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `DEFAULT_MODEL`, `MCP_PROMPT_SIZE_LIMIT`)
- **Private Methods**: Leading underscore (e.g., `_validate_token_limit`)

### Configuration
- **Environment Variables**: UPPER_SNAKE_CASE (e.g., `GEMINI_API_KEY`, `DEFAULT_MODEL`)
- **Tool Names**: lowercase (e.g., `codereview`, `thinkdeep`)
- **Model Names**: Provider-specific conventions preserved

## Import Organization

### Import Order (enforced by isort)
1. Standard library imports
2. Third-party imports (mcp, pydantic, etc.)
3. Local imports (config, tools, providers, utils)

### Conditional Imports
- **TYPE_CHECKING**: Use for type hints to avoid circular imports
- **Try/Except**: For optional dependencies (e.g., dotenv)

## Security Considerations

### File Path Validation
- **Absolute Paths Only**: All file paths must be absolute to prevent traversal attacks
- **Validation**: `BaseTool.validate_file_paths()` enforces this requirement

### API Key Management
- **Environment Variables**: Never hardcode API keys
- **Validation**: Check for placeholder values (e.g., `your_api_key_here`)
- **Logging**: Never log API keys or sensitive data