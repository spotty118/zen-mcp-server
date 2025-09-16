# Nexus MCP: Multi-Agent AI Communication Hub

[zen_web.webm](https://github.com/user-attachments/assets/851e3911-7f06-47c0-a4ab-a2601236697c)

<div align="center">
  <b>🤖 <a href="https://www.anthropic.com/claude-code">Claude Code</a> OR <a href="https://github.com/google-gemini/gemini-cli">Gemini CLI</a> OR <a href="https://github.com/openai/codex">Codex CLI</a> + [Gemini / OpenAI / Grok / OpenRouter / DIAL / Ollama / Anthropic / Any Model] = Your Ultimate AI Agent Team</b>
</div>

<br/>

**Multi-agent AI orchestration for Claude Code** - A Model Context Protocol server where CPU cores act as autonomous AI agents that communicate, collaborate, and coordinate to solve complex problems. Each agent has its own personality, expertise area, and thought processes while working together as a unified team. Nexus
works with Claude Code, Gemini CLI, Codex CLI as well as others.

**True AI agent collaboration with inter-agent communication** - Claude coordinates a team of specialized AI agents that maintain individual context and thoughts while sharing insights and discoveries. Each CPU core becomes an autonomous agent with specific roles like Security Analyst, Performance Optimizer, or Architecture Reviewer, enabling dynamic team formation and sophisticated multi-agent workflows.

> **You're in control.** Claude orchestrates the agent team, but you decide the agent composition and workflows. Craft powerful prompts that leverage specialized agents (Security Analysts, Performance Optimizers, Debuggers) exactly when needed for complex multi-agent problem solving.

<details>
<summary><b>Reasons to Use Nexus MCP</b></summary>

1. **Multi-Agent Orchestration** - CPU cores act as specialized AI agents (Security Analyst, Performance Optimizer, Architecture Reviewer) that communicate and collaborate on complex tasks

2. **Agent-to-Agent Communication** - Agents maintain individual thoughts and context while sharing insights, discoveries, and coordinating their analysis in real-time

3. **Dynamic Team Formation** - Agents automatically form teams based on task requirements, with role-specific expertise and communication patterns

4. **Persistent Agent Memory** - Each agent maintains its own context, personality, and decision-making history across interactions

5. **Intelligent Agent Coordination** - Agents can discover each other, share workload, and provide cross-domain insights (security agent informing performance agent about timing attacks)

6. **Model-Specific Strengths with Agent Personalities** - Combine model capabilities with agent roles: Security-focused agents use analytical models, Creative agents use models optimized for brainstorming

7. **Professional Multi-Agent Code Reviews** - Security, performance, and architecture agents work together, each contributing their specialized perspective

8. **Collaborative Debugging** - Debug specialists work with domain experts to provide systematic root cause analysis with multi-agent hypothesis testing

9. **Agent Status Monitoring** - Track what each agent is thinking, their confidence levels, and how they're communicating with other team members

10. **Cross-Agent Learning** - Agents learn from each other's successful strategies and adapt their approaches based on team collaboration patterns

11. **Local Model Support** - Run Llama, Mistral, or other models locally for complete privacy and zero API costs

12. **Bypass MCP Token Limits** - Automatically works around MCP's 25K limit for large prompts and responses

**The Killer Feature:** When Claude's context resets, just ask to "continue with O3" - the other model's response magically revives Claude's understanding without re-ingesting documents!

#### Example: Multi-Model Code Review Workflow

1. `Perform a codereview using gemini pro and o3 and use planner to generate a detailed plan, implement the fixes and do a final precommit check by continuing from the previous codereview`
2. This triggers a [`codereview`](docs/tools/codereview.md) workflow where Claude walks the code, looking for all kinds of issues
3. After multiple passes, collects relevant code and makes note of issues along the way
4. Maintains a `confidence` level between `exploring`, `low`, `medium`, `high` and `certain` to track how confidently it's been able to find and identify issues
5. Generates a detailed list of critical -> low issues
6. Shares the relevant files, findings, etc with **Gemini Pro** to perform a deep dive for a second [`codereview`](docs/tools/codereview.md)
7. Comes back with a response and next does the same with o3, adding to the prompt if a new discovery comes to light
8. When done, Claude takes in all the feedback and combines a single list of all critical -> low issues, including good patterns in your code. The final list includes new findings or revisions in case Claude misunderstood or missed something crucial and one of the other models pointed this out
9. It then uses the [`planner`](docs/tools/planner.md) workflow to break the work down into simpler steps if a major refactor is required
10. Claude then performs the actual work of fixing highlighted issues
11. When done, Claude returns to Gemini Pro for a [`precommit`](docs/tools/precommit.md) review

All within a single conversation thread! Gemini Pro in step 11 _knows_ what was recommended by O3 in step 7! Taking that context
and review into consideration to aid with its final pre-commit review.

**Think of it as Claude Code commanding an agent team.** This MCP isn't magic. It's **intelligent coordination**.

> **Remember:** Claude stays in full control — but **YOU** define the agent team composition.
> Nexus is designed to have Claude deploy specialized agents only when needed — and to orchestrate meaningful agent collaboration.
> **You're** the one who crafts the powerful prompt that makes Claude deploy Security Analysts, Performance Optimizers, Architecture Reviewers — or work with a single generalist agent.
> You're the orchestrator. The prompter. The team commander.
> #### You are the AI - **Actually Intelligent**.

#### Recommended AI Stack

For best results, use Claude Code with:
- **Opus 4.1** - All agentic orchestration and team coordination
- **Gemini 2.5 Pro** - Deep thinking agents, architecture review, debugging specialists
- **Specialized Agent Roles** - Security analysts, performance optimizers, code quality inspectors

</details>

## Quick Start (5 minutes)

**Prerequisites:** Python 3.10+, Git, [uv installed](https://docs.astral.sh/uv/getting-started/installation/)

**1. Get API Keys** (choose one or more):
- **[OpenRouter](https://openrouter.ai/)** - Access multiple models with one API
- **[Gemini](https://makersuite.google.com/app/apikey)** - Google's latest models
- **[OpenAI](https://platform.openai.com/api-keys)** - O3, GPT-5 series
- **[X.AI](https://console.x.ai/)** - Grok models
- **[DIAL](https://dialx.ai/)** - Vendor-agnostic model access
- **[Ollama](https://ollama.ai/)** - Local models (free)

**2. Install** (choose one):

**Option A: Clone and Automatic Setup** (recommended)
```bash
git clone https://github.com/BeehiveInnovations/zen-mcp-server.git
cd zen-mcp-server

# Handles everything: setup, config, API keys from system environment. 
# Auto-configures Claude Desktop, Claude Code, Gemini CLI, Codex CLI
# Enable / disable additional settings in .env
./run-server.sh  
```

**Option B: Instant Setup with [uvx](https://docs.astral.sh/uv/getting-started/installation/)**
```json
// Add to ~/.claude/settings.json or .mcp.json
// Don't forget to add your API keys under env
{
  "mcpServers": {
    "zen": {
      "command": "bash",
      "args": ["-c", "for p in $(which uvx 2>/dev/null) $HOME/.local/bin/uvx /opt/homebrew/bin/uvx /usr/local/bin/uvx uvx; do [ -x \"$p\" ] && exec \"$p\" --from git+https://github.com/BeehiveInnovations/zen-mcp-server.git zen-mcp-server; done; echo 'uvx not found' >&2; exit 1"],
      "env": {
        "PATH": "/usr/local/bin:/usr/bin:/bin:/opt/homebrew/bin:~/.local/bin",
        "GEMINI_API_KEY": "your-key-here",
        "DISABLED_TOOLS": "analyze,refactor,testgen,secaudit,docgen,tracer",
        "DEFAULT_MODEL": "auto"
      }
    }
  }
}
```

**3. Start Using!**
```
"Use zen to analyze this code for security issues with gemini pro"
"Debug this error with o3 and then get flash to suggest optimizations"
"Plan the migration strategy with zen, get consensus from multiple models"
```

👉 **[Complete Setup Guide](docs/getting-started.md)** with detailed installation, configuration for Gemini / Codex, and troubleshooting

## Core Tools

> **Note:** Each tool comes with its own multi-step workflow, parameters, and descriptions that consume valuable context window space even when not in use. To optimize performance, some tools are disabled by default. See [Tool Configuration](#tool-configuration) below to enable them.

**Collaboration & Planning** *(Enabled by default)*
- **[`chat`](docs/tools/chat.md)** - Brainstorm ideas, get second opinions, validate approaches
- **[`thinkdeep`](docs/tools/thinkdeep.md)** - Extended reasoning, edge case analysis, alternative perspectives
- **[`planner`](docs/tools/planner.md)** - Break down complex projects into structured, actionable plans
- **[`consensus`](docs/tools/consensus.md)** - Get expert opinions from multiple AI models with stance steering

**Code Analysis & Quality**
- **[`debug`](docs/tools/debug.md)** - Systematic investigation and root cause analysis
- **[`precommit`](docs/tools/precommit.md)** - Validate changes before committing, prevent regressions
- **[`codereview`](docs/tools/codereview.md)** - Professional reviews with severity levels and actionable feedback
- **[`analyze`](docs/tools/analyze.md)** *(disabled by default - [enable](#tool-configuration))* - Understand architecture, patterns, dependencies across entire codebases

**Development Tools** *(Disabled by default - [enable](#tool-configuration))*
- **[`refactor`](docs/tools/refactor.md)** - Intelligent code refactoring with decomposition focus
- **[`testgen`](docs/tools/testgen.md)** - Comprehensive test generation with edge cases
- **[`secaudit`](docs/tools/secaudit.md)** - Security audits with OWASP Top 10 analysis
- **[`docgen`](docs/tools/docgen.md)** - Generate documentation with complexity analysis

**Utilities**
- **[`challenge`](docs/tools/challenge.md)** - Prevent "You're absolutely right!" responses with critical analysis
- **[`tracer`](docs/tools/tracer.md)** *(disabled by default - [enable](#tool-configuration))* - Static analysis prompts for call-flow mapping

<details>
<summary><b id="tool-configuration">👉 Tool Configuration</b></summary>

### Default Configuration

To optimize context window usage, only essential tools are enabled by default:

**Enabled by default:**
- `chat`, `thinkdeep`, `planner`, `consensus` - Core collaboration tools
- `codereview`, `precommit`, `debug` - Essential code quality tools
- `challenge` - Critical thinking utility

**Disabled by default:**
- `analyze`, `refactor`, `testgen`, `secaudit`, `docgen`, `tracer`

### Enabling Additional Tools

To enable additional tools, remove them from the `DISABLED_TOOLS` list:

**Option 1: Edit your .env file**
```bash
# Default configuration (from .env.example)
DISABLED_TOOLS=analyze,refactor,testgen,secaudit,docgen,tracer

# To enable specific tools, remove them from the list
# Example: Enable analyze tool
DISABLED_TOOLS=refactor,testgen,secaudit,docgen,tracer

# To enable ALL tools
DISABLED_TOOLS=
```

**Option 2: Configure in MCP settings**
```json
// In ~/.claude/settings.json or .mcp.json
{
  "mcpServers": {
    "zen": {
      "env": {
        // Tool configuration
        "DISABLED_TOOLS": "refactor,testgen,secaudit,docgen,tracer",
        "DEFAULT_MODEL": "pro",
        "DEFAULT_THINKING_MODE_THINKDEEP": "high",
        
        // API configuration
        "GEMINI_API_KEY": "your-gemini-key",
        "OPENAI_API_KEY": "your-openai-key",
        "OPENROUTER_API_KEY": "your-openrouter-key",
        
        // Logging and performance
        "LOG_LEVEL": "INFO",
        "CONVERSATION_TIMEOUT_HOURS": "6",
        "MAX_CONVERSATION_TURNS": "50"
      }
    }
  }
}
```

**Option 3: Enable all tools**
```json
// Remove or empty the DISABLED_TOOLS to enable everything
{
  "mcpServers": {
    "zen": {
      "env": {
        "DISABLED_TOOLS": ""
      }
    }
  }
}
```

**Note:** 
- Essential tools (`version`, `listmodels`) cannot be disabled
- After changing tool configuration, restart your Claude session for changes to take effect
- Each tool adds to context window usage, so only enable what you need

</details>

## Key Features

**AI Orchestration**
- **Auto model selection** - Claude picks the right AI for each task
- **Multi-model workflows** - Chain different models in single conversations
- **Conversation continuity** - Context preserved across tools and models
- **[Context revival](docs/context-revival.md)** - Continue conversations even after context resets

**Model Support**
- **Multiple providers** - Gemini, OpenAI, X.AI, OpenRouter, DIAL, Ollama
- **Provider selection** - Force specific providers with `openrouter:model` syntax
- **Latest models** - GPT-5, Gemini 2.5 Pro, O3, Grok-4, local Llama
- **[Thinking modes](docs/advanced-usage.md#thinking-modes)** - Control reasoning depth vs cost
- **Vision support** - Analyze images, diagrams, screenshots

**Advanced Parallel Processing**
- **[Intelligent CPU utilization](docs/enhanced-parallel-thinking.md)** - Optimized multi-core processing with context sharing
- **Per-core context isolation** - Each CPU core maintains its own context while sharing insights
- **Cross-platform optimization** - Intel, AMD, Apple Silicon architecture awareness
- **Smart execution strategies** - Adaptive selection between async, threaded, and hybrid approaches

**Developer Experience**
- **Guided workflows** - Systematic investigation prevents rushed analysis
- **Smart file handling** - Auto-expand directories, manage token limits
- **Web search integration** - Access current documentation and best practices
- **[Large prompt support](docs/advanced-usage.md#working-with-large-prompts)** - Bypass MCP's 25K token limit

## Example Workflows

**Multi-model Code Review:**
```
"Perform a codereview using gemini pro and o3, then use planner to create a fix strategy"
```
→ Claude reviews code systematically → Consults Gemini Pro → Gets O3's perspective → Creates unified action plan

**Collaborative Debugging:**
```
"Debug this race condition with max thinking mode, then validate the fix with precommit"
```
→ Deep investigation → Expert analysis → Solution implementation → Pre-commit validation

**Architecture Planning:**
```
"Plan our microservices migration, get consensus from pro and o3 on the approach"
```
→ Structured planning → Multiple expert opinions → Consensus building → Implementation roadmap

👉 **[Advanced Usage Guide](docs/advanced-usage.md)** for complex workflows, model configuration, and power-user features

## CPU Utilization & Agent Architecture

### Intelligent Multi-Agent Processing

Nexus MCP uses a **shared-instance architecture with per-agent context isolation** - CPU cores become autonomous agents with specialized roles:

**Why Agent-Based Architecture?**

✅ **Agent-Based Multi-Core (Current Approach)**
- **Specialized expertise** - Each agent has specific domain knowledge (security, performance, etc.)
- **Inter-agent communication** - Agents share insights and coordinate analysis
- **Memory efficient** - Single process with agent context isolation
- **Dynamic teams** - Agents form collaborative teams for complex tasks
- **Individual thoughts** - Each agent maintains its own reasoning and decision history

❌ **Simple Parallel Processing**
- **No specialization** - All cores perform identical generic analysis
- **No communication** - Cores cannot coordinate or share insights
- **No memory** - No persistent agent context or learning
- **No collaboration** - Missed cross-domain insights and synergies

### Smart CPU Optimization with Agent Deployment

**Architecture-Aware Agent Deployment:**
- **Apple Silicon (M1/M2/M3+)**: Performance cores run analytical agents, efficiency cores handle communication
- **AMD Ryzen X3D**: Leverages 3D V-Cache for agent context storage
- **Intel 12th gen+**: Balances performance agents on P-cores, support agents on E-cores
- **Cross-platform**: Graceful fallback with round-robin agent assignment

**Agent Execution Strategies:**
- **`adaptive`** (default): System assigns agents to optimal cores automatically
- **`threads`**: CPU-intensive agents for deep analysis tasks
- **`asyncio`**: Communication-focused agents for coordination
- **`hybrid`**: Intelligent combination of both

**Context Sharing Benefits:**
```
Core 0: Analyzing security → shares "SQL injection found" 
Core 1: Analyzing performance → uses security insight to check query optimization
Core 2: Analyzing architecture → incorporates both security and performance findings
Result: Comprehensive analysis with cross-domain insights
```

### When CPU Optimization Matters

**Benefit High:** Complex reasoning, code analysis, multi-model workflows
**Benefit Medium:** Planning, consensus building, debugging  
**Benefit Low:** Simple queries, single-path thinking

**Configuration:**
```json
// High-performance setup
{
  "thinking_paths": 6,
  "execution_strategy": "adaptive", 
  "enable_cpu_affinity": true,
  "share_insights_between_cores": true
}

// Memory-optimized setup
{
  "thinking_paths": 3,
  "execution_strategy": "hybrid",
  "enable_cpu_affinity": false,
  "batch_size": 1
}
```

👉 **[Enhanced Parallel Thinking Guide](docs/enhanced-parallel-thinking.md)** for detailed CPU optimization and configuration

## Quick Links

**📖 Documentation**
- [Getting Started](docs/getting-started.md) - Complete setup guide
- [Tools Reference](docs/tools/) - All tools with examples
- [Advanced Usage](docs/advanced-usage.md) - Power user features
- [Configuration](docs/configuration.md) - Environment variables, restrictions
- [CPU Optimization Guide](docs/cpu-optimization-guide.md) - When and how to optimize CPU usage

**🔧 Setup & Support**
- [WSL Setup](docs/wsl-setup.md) - Windows users
- [Troubleshooting](docs/troubleshooting.md) - Common issues
- [Contributing](docs/contributions.md) - Code standards, PR process

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Built with the power of **Multi-Model AI** collaboration 🤝
- **A**ctual **I**ntelligence by real Humans
- [MCP (Model Context Protocol)](https://modelcontextprotocol.com) by Anthropic
- [Claude Code](https://claude.ai/code) - Your AI coding orchestrator
- [Gemini 2.5 Pro & Flash](https://ai.google.dev/) - Extended thinking & fast analysis
- [OpenAI O3 & GPT-5](https://openai.com/) - Strong reasoning & latest capabilities

### Star History

[![Star History Chart](https://api.star-history.com/svg?repos=BeehiveInnovations/zen-mcp-server&type=Date)](https://www.star-history.com/#BeehiveInnovations/zen-mcp-server&Date)
