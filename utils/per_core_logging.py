"""
Per-Core Agent Structured Logging

This module implements structured logging for agent activities, API calls,
and system events with proper formatting, filtering, and log rotation.

Key Features:
- Structured logging with JSON formatting for machine readability
- Agent-specific log filtering and routing
- OpenRouter API call logging with rate limiting awareness
- System event logging with correlation IDs
- Log rotation and retention management
- Performance metrics logging
"""

import json
import logging
import logging.handlers
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from utils.agent_core import Agent, AgentRole, AgentStatus


class LogLevel(Enum):
    """Custom log levels for per-core agent system"""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    AGENT_ACTIVITY = 25
    API_CALL = 22
    SYSTEM_EVENT = 28


class LogCategory(Enum):
    """Categories for structured logging"""
    AGENT_LIFECYCLE = "agent_lifecycle"
    AGENT_COMMUNICATION = "agent_communication"
    AGENT_THINKING = "agent_thinking"
    OPENROUTER_API = "openrouter_api"
    SYSTEM_HEALTH = "system_health"
    ERROR_RECOVERY = "error_recovery"
    PERFORMANCE = "performance"
    CONFIGURATION = "configuration"
    SECURITY = "security"


@dataclass
class LogEntry:
    """Structured log entry for per-core agent system"""
    timestamp: float = field(default_factory=time.time)
    level: str = "INFO"
    category: LogCategory = LogCategory.SYSTEM_HEALTH
    message: str = ""
    agent_id: Optional[str] = None
    core_id: Optional[int] = None
    correlation_id: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "level": self.level,
            "category": self.category.value,
            "message": self.message,
            "agent_id": self.agent_id,
            "core_id": self.core_id,
            "correlation_id": self.correlation_id,
            "session_id": self.session_id,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Convert log entry to JSON string"""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class AgentActivityLog:
    """Specific log entry for agent activities"""
    agent_id: str
    activity_type: str
    activity_details: Dict[str, Any]
    duration_ms: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_log_entry(self) -> LogEntry:
        """Convert to structured log entry"""
        level = "INFO" if self.success else "ERROR"
        category = LogCategory.AGENT_LIFECYCLE
        
        # Determine category based on activity type
        if "thinking" in self.activity_type.lower():
            category = LogCategory.AGENT_THINKING
        elif "communication" in self.activity_type.lower():
            category = LogCategory.AGENT_COMMUNICATION
        
        message = f"Agent {self.agent_id} {self.activity_type}"
        if self.duration_ms:
            message += f" (took {self.duration_ms:.2f}ms)"
        if not self.success and self.error_message:
            message += f" - Error: {self.error_message}"
        
        metadata = {
            "activity_type": self.activity_type,
            "activity_details": self.activity_details,
            "duration_ms": self.duration_ms,
            "success": self.success
        }
        
        if self.error_message:
            metadata["error_message"] = self.error_message
        
        return LogEntry(
            timestamp=self.timestamp,
            level=level,
            category=category,
            message=message,
            agent_id=self.agent_id,
            metadata=metadata
        )


@dataclass
class OpenRouterAPILog:
    """Specific log entry for OpenRouter API calls"""
    agent_id: str
    api_call_id: str
    model_used: str
    request_type: str
    response_time_ms: float
    tokens_used: Optional[int] = None
    cost_estimate: Optional[float] = None
    success: bool = True
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    rate_limited: bool = False
    timestamp: float = field(default_factory=time.time)
    
    def to_log_entry(self) -> LogEntry:
        """Convert to structured log entry"""
        level = "INFO" if self.success else "ERROR"
        if self.rate_limited:
            level = "WARNING"
        
        message = f"OpenRouter API call {self.api_call_id} for agent {self.agent_id}"
        message += f" using {self.model_used} ({self.response_time_ms:.2f}ms)"
        
        if self.rate_limited:
            message += " - RATE LIMITED"
        elif not self.success:
            message += f" - ERROR: {self.error_message}"
        elif self.tokens_used:
            message += f" - {self.tokens_used} tokens"
        
        metadata = {
            "api_call_id": self.api_call_id,
            "model_used": self.model_used,
            "request_type": self.request_type,
            "response_time_ms": self.response_time_ms,
            "tokens_used": self.tokens_used,
            "cost_estimate": self.cost_estimate,
            "success": self.success,
            "rate_limited": self.rate_limited
        }
        
        if self.error_code:
            metadata["error_code"] = self.error_code
        if self.error_message:
            metadata["error_message"] = self.error_message
        
        return LogEntry(
            timestamp=self.timestamp,
            level=level,
            category=LogCategory.OPENROUTER_API,
            message=message,
            agent_id=self.agent_id,
            metadata=metadata
        )


@dataclass
class SystemEventLog:
    """Specific log entry for system events"""
    event_type: str
    event_details: Dict[str, Any]
    severity: str = "INFO"
    affected_agents: Optional[List[str]] = None
    correlation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_log_entry(self) -> LogEntry:
        """Convert to structured log entry"""
        message = f"System event: {self.event_type}"
        
        if self.affected_agents:
            message += f" (affects {len(self.affected_agents)} agents)"
        
        metadata = {
            "event_type": self.event_type,
            "event_details": self.event_details,
            "affected_agents": self.affected_agents or []
        }
        
        return LogEntry(
            timestamp=self.timestamp,
            level=self.severity,
            category=LogCategory.SYSTEM_HEALTH,
            message=message,
            correlation_id=self.correlation_id,
            metadata=metadata
        )


class PerCoreLogFormatter(logging.Formatter):
    """Custom formatter for per-core agent logs"""
    
    def __init__(self, use_json: bool = True):
        super().__init__()
        self.use_json = use_json
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data"""
        # Check if record has structured log entry
        if hasattr(record, 'log_entry') and isinstance(record.log_entry, LogEntry):
            if self.use_json:
                return record.log_entry.to_json()
            else:
                entry = record.log_entry
                return (f"{datetime.fromtimestamp(entry.timestamp).isoformat()} "
                       f"[{entry.level}] [{entry.category.value}] "
                       f"{entry.message}")
        
        # Check if record has error context from error handler
        if hasattr(record, 'error_context'):
            if self.use_json:
                return json.dumps(record.error_context, default=str)
            else:
                ctx = record.error_context
                return (f"{datetime.fromtimestamp(ctx.get('timestamp', time.time())).isoformat()} "
                       f"[{ctx.get('severity', 'ERROR')}] [{ctx.get('category', 'unknown')}] "
                       f"{ctx.get('error_message', record.getMessage())}")
        
        # Fallback to standard formatting
        if self.use_json:
            log_data = {
                "timestamp": time.time(),
                "datetime": datetime.now().isoformat(),
                "level": record.levelname,
                "category": "general",
                "message": record.getMessage(),
                "logger": record.name,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add exception info if present
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)
            
            return json.dumps(log_data, default=str)
        else:
            return super().format(record)


class PerCoreLogFilter(logging.Filter):
    """Filter for per-core agent logs"""
    
    def __init__(self, agent_id: Optional[str] = None, category: Optional[LogCategory] = None,
                 min_level: int = logging.INFO):
        super().__init__()
        self.agent_id = agent_id
        self.category = category
        self.min_level = min_level
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter log records based on criteria"""
        # Check log level
        if record.levelno < self.min_level:
            return False
        
        # Check for structured log entry
        if hasattr(record, 'log_entry') and isinstance(record.log_entry, LogEntry):
            entry = record.log_entry
            
            # Filter by agent ID
            if self.agent_id and entry.agent_id != self.agent_id:
                return False
            
            # Filter by category
            if self.category and entry.category != self.category:
                return False
        
        return True


class PerCoreLogger:
    """
    Structured logger for per-core agent system
    """
    
    def __init__(self, log_dir: Optional[str] = None, use_json: bool = True,
                 max_file_size: int = 50 * 1024 * 1024, backup_count: int = 5):
        """
        Initialize per-core logger
        
        Args:
            log_dir: Directory for log files (default: ~/.zen_mcp/logs)
            use_json: Whether to use JSON formatting
            max_file_size: Maximum size per log file in bytes
            backup_count: Number of backup files to keep
        """
        self.log_dir = log_dir or os.path.join(os.path.expanduser("~"), ".zen_mcp", "logs")
        self.use_json = use_json
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Initialize loggers
        self.loggers: Dict[str, logging.Logger] = {}
        self.handlers: Dict[str, logging.Handler] = {}
        
        # Thread safety
        self._logger_lock = threading.RLock()
        
        # Initialize main loggers
        self._initialize_loggers()
        
        # Correlation ID tracking
        self._correlation_ids: Dict[str, str] = {}
        self._correlation_lock = threading.Lock()
    
    def _initialize_loggers(self) -> None:
        """Initialize all required loggers"""
        logger_configs = [
            ("per_core_agents", "per_core_agents.log", logging.INFO),
            ("openrouter_api", "openrouter_api.log", logging.INFO),
            ("system_events", "system_events.log", logging.INFO),
            ("error_recovery", "error_recovery.log", logging.WARNING),
            ("performance", "performance.log", logging.INFO),
            ("agent_communication", "agent_communication.log", logging.DEBUG)
        ]
        
        for logger_name, log_file, level in logger_configs:
            self._create_logger(logger_name, log_file, level)
    
    def _create_logger(self, name: str, filename: str, level: int) -> logging.Logger:
        """Create a logger with rotating file handler"""
        with self._logger_lock:
            if name in self.loggers:
                return self.loggers[name]
            
            # Create logger
            logger = logging.getLogger(f"per_core.{name}")
            logger.setLevel(level)
            logger.propagate = False
            
            # Create rotating file handler
            log_file_path = os.path.join(self.log_dir, filename)
            handler = logging.handlers.RotatingFileHandler(
                log_file_path,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count,
                encoding='utf-8'
            )
            
            # Set formatter
            formatter = PerCoreLogFormatter(use_json=self.use_json)
            handler.setFormatter(formatter)
            
            # Add handler to logger
            logger.addHandler(handler)
            
            # Store references
            self.loggers[name] = logger
            self.handlers[name] = handler
            
            return logger
    
    def log_agent_activity(self, activity_log: AgentActivityLog) -> None:
        """Log agent activity"""
        logger = self.loggers.get("per_core_agents")
        if logger:
            log_entry = activity_log.to_log_entry()
            record = logging.LogRecord(
                name=logger.name,
                level=getattr(logging, log_entry.level),
                pathname="",
                lineno=0,
                msg=log_entry.message,
                args=(),
                exc_info=None
            )
            record.log_entry = log_entry
            logger.handle(record)
    
    def log_openrouter_api_call(self, api_log: OpenRouterAPILog) -> None:
        """Log OpenRouter API call"""
        logger = self.loggers.get("openrouter_api")
        if logger:
            log_entry = api_log.to_log_entry()
            record = logging.LogRecord(
                name=logger.name,
                level=getattr(logging, log_entry.level),
                pathname="",
                lineno=0,
                msg=log_entry.message,
                args=(),
                exc_info=None
            )
            record.log_entry = log_entry
            logger.handle(record)
    
    def log_system_event(self, event_log: SystemEventLog) -> None:
        """Log system event"""
        logger = self.loggers.get("system_events")
        if logger:
            log_entry = event_log.to_log_entry()
            record = logging.LogRecord(
                name=logger.name,
                level=getattr(logging, log_entry.level),
                pathname="",
                lineno=0,
                msg=log_entry.message,
                args=(),
                exc_info=None
            )
            record.log_entry = log_entry
            logger.handle(record)
    
    def log_error_recovery(self, message: str, error_context: Dict[str, Any],
                          level: str = "ERROR") -> None:
        """Log error recovery attempt"""
        logger = self.loggers.get("error_recovery")
        if logger:
            log_entry = LogEntry(
                level=level,
                category=LogCategory.ERROR_RECOVERY,
                message=message,
                agent_id=error_context.get("agent_id"),
                core_id=error_context.get("core_id"),
                metadata=error_context
            )
            
            record = logging.LogRecord(
                name=logger.name,
                level=getattr(logging, level),
                pathname="",
                lineno=0,
                msg=message,
                args=(),
                exc_info=None
            )
            record.log_entry = log_entry
            logger.handle(record)
    
    def log_performance_metric(self, metric_name: str, value: Union[int, float],
                              agent_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Log performance metric"""
        logger = self.loggers.get("performance")
        if logger:
            log_entry = LogEntry(
                level="INFO",
                category=LogCategory.PERFORMANCE,
                message=f"Performance metric: {metric_name} = {value}",
                agent_id=agent_id,
                metadata={
                    "metric_name": metric_name,
                    "value": value,
                    **(metadata or {})
                }
            )
            
            record = logging.LogRecord(
                name=logger.name,
                level=logging.INFO,
                pathname="",
                lineno=0,
                msg=log_entry.message,
                args=(),
                exc_info=None
            )
            record.log_entry = log_entry
            logger.handle(record)
    
    def create_correlation_id(self, operation: str) -> str:
        """Create a correlation ID for tracking related operations"""
        import uuid
        correlation_id = f"{operation}_{int(time.time())}_{str(uuid.uuid4())[:8]}"
        
        with self._correlation_lock:
            self._correlation_ids[correlation_id] = operation
        
        return correlation_id
    
    def get_agent_logger(self, agent_id: str) -> logging.Logger:
        """Get a logger filtered for a specific agent"""
        logger_name = f"agent_{agent_id}"
        
        with self._logger_lock:
            if logger_name not in self.loggers:
                # Create agent-specific logger
                logger = logging.getLogger(f"per_core.agent.{agent_id}")
                logger.setLevel(logging.DEBUG)
                logger.propagate = False
                
                # Create agent-specific log file
                log_file_path = os.path.join(self.log_dir, f"agent_{agent_id}.log")
                handler = logging.handlers.RotatingFileHandler(
                    log_file_path,
                    maxBytes=self.max_file_size // 4,  # Smaller files for individual agents
                    backupCount=3,
                    encoding='utf-8'
                )
                
                # Set formatter and filter
                formatter = PerCoreLogFormatter(use_json=self.use_json)
                handler.setFormatter(formatter)
                
                # Add filter for this agent
                agent_filter = PerCoreLogFilter(agent_id=agent_id)
                handler.addFilter(agent_filter)
                
                logger.addHandler(handler)
                
                self.loggers[logger_name] = logger
                self.handlers[logger_name] = handler
            
            return self.loggers[logger_name]
    
    def flush_all_logs(self) -> None:
        """Flush all log handlers"""
        with self._logger_lock:
            for handler in self.handlers.values():
                if hasattr(handler, 'flush'):
                    handler.flush()
    
    def close_all_logs(self) -> None:
        """Close all log handlers"""
        with self._logger_lock:
            for handler in self.handlers.values():
                if hasattr(handler, 'close'):
                    handler.close()
            
            self.handlers.clear()
            self.loggers.clear()
    
    def get_log_statistics(self) -> Dict[str, Any]:
        """Get logging statistics"""
        stats = {
            "log_directory": self.log_dir,
            "active_loggers": len(self.loggers),
            "log_files": [],
            "total_log_size_bytes": 0
        }
        
        # Collect log file information
        try:
            for filename in os.listdir(self.log_dir):
                if filename.endswith('.log'):
                    file_path = os.path.join(self.log_dir, filename)
                    file_size = os.path.getsize(file_path)
                    stats["log_files"].append({
                        "filename": filename,
                        "size_bytes": file_size,
                        "size_mb": file_size / (1024 * 1024)
                    })
                    stats["total_log_size_bytes"] += file_size
        except Exception as e:
            stats["error"] = f"Could not collect log file statistics: {e}"
        
        stats["total_log_size_mb"] = stats["total_log_size_bytes"] / (1024 * 1024)
        
        return stats


# Global logger instance
_per_core_logger: Optional[PerCoreLogger] = None
_logger_lock = threading.Lock()


def get_per_core_logger() -> PerCoreLogger:
    """Get the global per-core logger instance"""
    global _per_core_logger
    
    if _per_core_logger is None:
        with _logger_lock:
            if _per_core_logger is None:
                _per_core_logger = PerCoreLogger()
    
    return _per_core_logger


def shutdown_per_core_logger() -> None:
    """Shutdown the global per-core logger"""
    global _per_core_logger
    
    if _per_core_logger:
        _per_core_logger.flush_all_logs()
        _per_core_logger.close_all_logs()
        _per_core_logger = None


# Convenience functions for common logging operations
def log_agent_activity(agent_id: str, activity_type: str, activity_details: Dict[str, Any],
                      duration_ms: Optional[float] = None, success: bool = True,
                      error_message: Optional[str] = None) -> None:
    """Convenience function to log agent activity"""
    logger = get_per_core_logger()
    activity_log = AgentActivityLog(
        agent_id=agent_id,
        activity_type=activity_type,
        activity_details=activity_details,
        duration_ms=duration_ms,
        success=success,
        error_message=error_message
    )
    logger.log_agent_activity(activity_log)


def log_openrouter_api_call(agent_id: str, api_call_id: str, model_used: str,
                           request_type: str, response_time_ms: float,
                           tokens_used: Optional[int] = None, cost_estimate: Optional[float] = None,
                           success: bool = True, error_code: Optional[str] = None,
                           error_message: Optional[str] = None, rate_limited: bool = False) -> None:
    """Convenience function to log OpenRouter API call"""
    logger = get_per_core_logger()
    api_log = OpenRouterAPILog(
        agent_id=agent_id,
        api_call_id=api_call_id,
        model_used=model_used,
        request_type=request_type,
        response_time_ms=response_time_ms,
        tokens_used=tokens_used,
        cost_estimate=cost_estimate,
        success=success,
        error_code=error_code,
        error_message=error_message,
        rate_limited=rate_limited
    )
    logger.log_openrouter_api_call(api_log)


def log_system_event(event_type: str, event_details: Dict[str, Any],
                    severity: str = "INFO", affected_agents: Optional[List[str]] = None,
                    correlation_id: Optional[str] = None) -> None:
    """Convenience function to log system event"""
    logger = get_per_core_logger()
    event_log = SystemEventLog(
        event_type=event_type,
        event_details=event_details,
        severity=severity,
        affected_agents=affected_agents,
        correlation_id=correlation_id
    )
    logger.log_system_event(event_log)