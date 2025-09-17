# Product Overview

**Nexus MCP** is a multi-agent AI communication hub that implements the Model Context Protocol (MCP) to orchestrate specialized AI agents for code analysis, review, and development assistance.

## Core Concept

The system transforms CPU cores into autonomous AI agents with specialized roles (Security Analyst, Performance Optimizer, Architecture Reviewer, etc.) that communicate and collaborate to solve complex development problems. Each agent maintains individual context and expertise while sharing insights through inter-agent communication.

## Key Features

- **Multi-Agent Orchestration**: CPU cores act as specialized AI agents that communicate and collaborate
- **Agent-to-Agent Communication**: Agents maintain individual thoughts while sharing insights and discoveries
- **Dynamic Team Formation**: Agents automatically form teams based on task requirements
- **Persistent Agent Memory**: Each agent maintains context, personality, and decision history
- **Model Provider Abstraction**: Supports multiple AI providers (Gemini, OpenAI, X.AI, OpenRouter, DIAL, Ollama)
- **Conversation Continuity**: Context preserved across tools and models with thread resumption
- **Context Revival**: Continue conversations even after context resets

## Target Users

- Developers using Claude Code, Gemini CLI, or Codex CLI
- Teams needing comprehensive code review and analysis
- Organizations requiring multi-perspective AI assistance for complex development tasks
- Users wanting to leverage multiple AI models in coordinated workflows

## Architecture Philosophy

The system bridges stateless MCP protocol with stateful multi-turn AI conversations, enabling seamless handoffs between different tools while preserving full conversation context and file references.