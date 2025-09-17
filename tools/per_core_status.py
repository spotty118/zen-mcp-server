"""
Per-Core Agent Status Tool

This tool provides status information about the per-core agent coordination system,
including agent health, configuration, and system availability.
"""

from typing import Any, Dict, Optional

from pydantic import Field

from tools.shared.base_models import ToolRequest
from tools.simple.base import SimpleTool


class PerCoreStatusRequest(ToolRequest):
    """Request model for Per-Core Agent Status tool"""
    
    detailed: Optional[bool] = Field(
        default=False,
        description="Include detailed agent information and health metrics"
    )


class PerCoreStatusTool(SimpleTool):
    """
    Tool for checking the status of the per-core agent coordination system.
    
    This tool provides information about:
    - System availability and configuration
    - Agent health and status
    - OpenRouter connectivity
    - Performance metrics
    """
    
    def get_name(self) -> str:
        return "per_core_status"
    
    def get_description(self) -> str:
        return (
            "Check the status and health of the per-core agent coordination system. "
            "Shows agent availability, roles, health metrics, and system configuration. "
            "Use this to troubleshoot multi-agent coordination issues or verify system status."
        )
    
    def get_system_prompt(self) -> str:
        return """You are a system status reporter for the per-core agent coordination system.

Provide clear, structured information about:
- System availability and configuration
- Agent health and status
- Any issues or recommendations
- Performance metrics when available

Format your response in a clear, readable structure with appropriate sections and bullet points.
If the system is not available, explain why and suggest next steps."""
    
    def get_request_model(self):
        return PerCoreStatusRequest
    
    def get_tool_fields(self) -> Dict[str, Dict[str, Any]]:
        return {
            "detailed": {
                "type": "boolean",
                "description": "Include detailed agent information and health metrics",
                "default": False
            }
        }
    
    def get_required_fields(self) -> list[str]:
        return []  # No required fields
    
    def requires_model(self) -> bool:
        """This tool doesn't require AI model access - it's pure data reporting"""
        return False
    
    async def prepare_prompt(self, request) -> str:
        """This tool doesn't use AI models, so no prompt preparation needed"""
        return ""
    
    async def execute(self, arguments: Dict[str, Any]) -> list:
        """Execute the per-core status check"""
        import json
        from mcp.types import TextContent
        from tools.models import ToolOutput
        from utils.per_core_tool_integration import get_per_core_integrator
        from config import (
            ENABLE_PER_CORE_AGENTS, 
            PER_CORE_MAX_AGENTS, 
            PER_CORE_OPENROUTER_REQUIRED,
            PER_CORE_FALLBACK_MODE,
            PER_CORE_MULTI_AGENT_TOOLS
        )
        import os
        
        try:
            # Validate request
            request = self.get_request_model()(**arguments)
            detailed = request.detailed
            
            # Get per-core integrator
            integrator = get_per_core_integrator()
            
            # Build status report
            status_report = {
                "system_status": "unknown",
                "configuration": {},
                "agents": {},
                "health": {},
                "recommendations": []
            }
            
            # Configuration information
            status_report["configuration"] = {
                "enabled": ENABLE_PER_CORE_AGENTS,
                "max_agents": PER_CORE_MAX_AGENTS,
                "openrouter_required": PER_CORE_OPENROUTER_REQUIRED,
                "fallback_mode": PER_CORE_FALLBACK_MODE,
                "multi_agent_tools": list(PER_CORE_MULTI_AGENT_TOOLS),
                "openrouter_configured": bool(
                    os.getenv("OPENROUTER_API_KEY") and 
                    os.getenv("OPENROUTER_API_KEY") != "your_openrouter_api_key_here"
                )
            }
            
            # Check system availability
            if not ENABLE_PER_CORE_AGENTS:
                status_report["system_status"] = "disabled"
                status_report["recommendations"].append(
                    "Set ENABLE_PER_CORE_AGENTS=true to enable per-core agent coordination"
                )
            elif integrator.is_available():
                status_report["system_status"] = "available"
                
                # Get system health
                health_info = integrator.get_system_health()
                status_report["health"] = health_info
                
                # Get agent information
                if integrator.per_core_manager:
                    agents_info = {}
                    
                    for core_id, agent in integrator.per_core_manager.agents.items():
                        agent_info = {
                            "agent_id": agent.agent_id,
                            "role": agent.role.value,
                            "status": agent.status.value,
                            "core_id": core_id
                        }
                        
                        # Add detailed information if requested
                        if detailed and agent.agent_id in integrator.per_core_manager.agent_statuses:
                            agent_status = integrator.per_core_manager.agent_statuses[agent.agent_id]
                            agent_info.update({
                                "openrouter_connected": agent_status.openrouter_connected,
                                "active_thinking_sessions": agent_status.active_thinking_sessions,
                                "total_api_calls": agent_status.total_api_calls,
                                "success_rate": agent_status.success_rate,
                                "memory_usage_mb": agent_status.memory_usage_mb,
                                "is_healthy": agent_status.is_healthy(),
                                "needs_attention": agent_status.needs_attention()
                            })
                        
                        agents_info[f"core_{core_id}"] = agent_info
                    
                    status_report["agents"] = agents_info
                    
                    # Add recommendations based on agent health
                    if detailed:
                        unhealthy_agents = [
                            agent_id for agent_id, status in integrator.per_core_manager.agent_statuses.items()
                            if not status.is_healthy()
                        ]
                        
                        if unhealthy_agents:
                            status_report["recommendations"].append(
                                f"Check health of agents: {', '.join(unhealthy_agents)}"
                            )
                        
                        attention_needed = [
                            agent_id for agent_id, status in integrator.per_core_manager.agent_statuses.items()
                            if status.needs_attention()
                        ]
                        
                        if attention_needed:
                            status_report["recommendations"].append(
                                f"Agents needing attention: {', '.join(attention_needed)}"
                            )
            else:
                status_report["system_status"] = "unavailable"
                
                # Add specific recommendations based on configuration
                if not status_report["configuration"]["openrouter_configured"] and PER_CORE_OPENROUTER_REQUIRED:
                    status_report["recommendations"].append(
                        "Configure OPENROUTER_API_KEY environment variable to enable per-core agents"
                    )
                
                status_report["recommendations"].append(
                    "Check logs for per-core agent initialization errors"
                )
            
            # Format the response
            response_text = self._format_status_report(status_report, detailed)
            
            # Create successful response
            tool_output = ToolOutput(
                status="success",
                content=response_text,
                content_type="text",
                metadata={
                    "system_available": status_report["system_status"] == "available",
                    "agents_count": len(status_report["agents"]),
                    "configuration": status_report["configuration"]
                }
            )
            
            return [TextContent(type="text", text=tool_output.model_dump_json())]
            
        except Exception as e:
            # Handle errors gracefully
            error_output = ToolOutput(
                status="error",
                content=f"Error checking per-core agent status: {str(e)}",
                content_type="text"
            )
            return [TextContent(type="text", text=error_output.model_dump_json())]
    
    def _format_status_report(self, status_report: Dict[str, Any], detailed: bool) -> str:
        """Format the status report into a readable text response"""
        
        lines = ["# Per-Core Agent System Status\n"]
        
        # System Status
        status = status_report["system_status"]
        status_emoji = {
            "available": "✅",
            "unavailable": "❌", 
            "disabled": "⚪",
            "unknown": "❓"
        }
        
        lines.append(f"**System Status:** {status_emoji.get(status, '❓')} {status.upper()}\n")
        
        # Configuration
        config = status_report["configuration"]
        lines.append("## Configuration")
        lines.append(f"- **Enabled:** {'✅' if config['enabled'] else '❌'} {config['enabled']}")
        lines.append(f"- **Max Agents:** {config['max_agents'] or 'All available cores'}")
        lines.append(f"- **OpenRouter Required:** {'✅' if config['openrouter_required'] else '❌'} {config['openrouter_required']}")
        lines.append(f"- **OpenRouter Configured:** {'✅' if config['openrouter_configured'] else '❌'} {config['openrouter_configured']}")
        lines.append(f"- **Fallback Mode:** {config['fallback_mode']}")
        lines.append(f"- **Multi-Agent Tools:** {', '.join(config['multi_agent_tools'])}\n")
        
        # Agents
        if status_report["agents"]:
            lines.append("## Agents")
            for core_name, agent_info in status_report["agents"].items():
                status_emoji = "✅" if agent_info["status"] == "ACTIVE" else "❌"
                lines.append(f"- **{core_name}:** {status_emoji} {agent_info['role']} ({agent_info['status']})")
                
                if detailed and "is_healthy" in agent_info:
                    health_emoji = "✅" if agent_info["is_healthy"] else "❌"
                    lines.append(f"  - Health: {health_emoji} {'Healthy' if agent_info['is_healthy'] else 'Unhealthy'}")
                    lines.append(f"  - OpenRouter: {'✅' if agent_info['openrouter_connected'] else '❌'}")
                    lines.append(f"  - API Calls: {agent_info['total_api_calls']}")
                    lines.append(f"  - Success Rate: {agent_info['success_rate']:.1%}")
            lines.append("")
        
        # Health Information
        if status_report["health"] and detailed:
            lines.append("## System Health")
            health = status_report["health"]
            if "available" in health and health["available"]:
                lines.append("- **Overall Health:** ✅ System operational")
                if "total_agents" in health:
                    lines.append(f"- **Total Agents:** {health['total_agents']}")
                if "healthy_agents" in health:
                    lines.append(f"- **Healthy Agents:** {health['healthy_agents']}")
            else:
                lines.append("- **Overall Health:** ❌ System issues detected")
                if "reason" in health:
                    lines.append(f"- **Issue:** {health['reason']}")
            lines.append("")
        
        # Recommendations
        if status_report["recommendations"]:
            lines.append("## Recommendations")
            for rec in status_report["recommendations"]:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Usage Information
        if status == "available":
            lines.append("## Usage")
            lines.append("The per-core agent system is active and will automatically enhance tool execution.")
            lines.append("Tools that benefit from multi-agent coordination:")
            for tool in status_report["configuration"]["multi_agent_tools"]:
                lines.append(f"- `{tool}`")
        elif status == "disabled":
            lines.append("## Usage")
            lines.append("To enable per-core agent coordination, set `ENABLE_PER_CORE_AGENTS=true` in your environment.")
        elif status == "unavailable":
            lines.append("## Usage")
            lines.append("Per-core agent system is enabled but not available. Check the recommendations above.")
        
        return "\n".join(lines)