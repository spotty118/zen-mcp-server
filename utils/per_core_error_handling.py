"""
Per-Core Agent Error Handling and Recovery

This module implements comprehensive error handling for OpenRouter API failures,
agent communication errors, and system-wide error recovery strategies with
exponential backoff and retry logic.

Key Features:
- Categorized error handling for different failure types
- Exponential backoff and retry strategies
- Circuit breaker patterns for API failures
- Error recovery coordination between agents
- Structured error logging and metrics
"""

import asyncio
import logging
import random
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from utils.agent_core import Agent, AgentStatus


class ErrorCategory(Enum):
    """Categories of errors for appropriate handling strategies"""
    OPENROUTER_API_ERROR = "openrouter_api_error"
    OPENROUTER_RATE_LIMIT = "openrouter_rate_limit"
    OPENROUTER_AUTH_ERROR = "openrouter_auth_error"
    OPENROUTER_TIMEOUT = "openrouter_timeout"
    OPENROUTER_SERVER_ERROR = "openrouter_server_error"
    AGENT_COMMUNICATION_ERROR = "agent_communication_error"
    AGENT_INITIALIZATION_ERROR = "agent_initialization_error"
    AGENT_THINKING_ERROR = "agent_thinking_error"
    SYSTEM_RESOURCE_ERROR = "system_resource_error"
    CONFIGURATION_ERROR = "configuration_error"
    NETWORK_ERROR = "network_error"
    UNKNOWN_ERROR = "unknown_error"


class ErrorSeverity(Enum):
    """Error severity levels for prioritized handling"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for error handling and recovery"""
    error_id: str
    category: ErrorCategory
    severity: ErrorSeverity
    agent_id: Optional[str] = None
    core_id: Optional[int] = None
    timestamp: float = field(default_factory=time.time)
    error_message: str = ""
    exception_type: str = ""
    stack_trace: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    backoff_factor: float = 2.0
    recovery_attempted: bool = False
    recovery_successful: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def should_retry(self) -> bool:
        """Check if error should be retried based on category and retry count"""
        if self.retry_count >= self.max_retries:
            return False
        
        # Don't retry authentication errors
        if self.category == ErrorCategory.OPENROUTER_AUTH_ERROR:
            return False
        
        # Don't retry configuration errors
        if self.category == ErrorCategory.CONFIGURATION_ERROR:
            return False
        
        return True
    
    def get_backoff_delay(self) -> float:
        """Calculate exponential backoff delay with jitter"""
        base_delay = self.backoff_factor ** self.retry_count
        jitter = random.uniform(0.1, 0.3) * base_delay
        return base_delay + jitter
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error context to dictionary for logging"""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "severity": self.severity.value,
            "agent_id": self.agent_id,
            "core_id": self.core_id,
            "timestamp": self.timestamp,
            "error_message": self.error_message,
            "exception_type": self.exception_type,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "recovery_attempted": self.recovery_attempted,
            "recovery_successful": self.recovery_successful,
            "metadata": self.metadata
        }


class ErrorRecoveryStrategy:
    """Base class for error recovery strategies"""
    
    def __init__(self, name: str, applicable_categories: Set[ErrorCategory]):
        self.name = name
        self.applicable_categories = applicable_categories
        self.success_count = 0
        self.failure_count = 0
        self.last_used = 0.0
    
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this strategy can handle the given error"""
        return error_context.category in self.applicable_categories
    
    async def recover(self, error_context: ErrorContext, **kwargs) -> bool:
        """
        Attempt to recover from the error
        
        Args:
            error_context: The error context to recover from
            **kwargs: Additional recovery parameters
            
        Returns:
            True if recovery was successful, False otherwise
        """
        self.last_used = time.time()
        try:
            success = await self._execute_recovery(error_context, **kwargs)
            if success:
                self.success_count += 1
            else:
                self.failure_count += 1
            return success
        except Exception as e:
            self.failure_count += 1
            logger.error(f"Error recovery strategy '{self.name}' failed: {e}")
            return False
    
    async def _execute_recovery(self, error_context: ErrorContext, **kwargs) -> bool:
        """Override this method to implement specific recovery logic"""
        raise NotImplementedError
    
    def get_success_rate(self) -> float:
        """Get the success rate of this recovery strategy"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0


class OpenRouterRetryStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for OpenRouter API errors with exponential backoff"""
    
    def __init__(self):
        super().__init__(
            "openrouter_retry",
            {
                ErrorCategory.OPENROUTER_API_ERROR,
                ErrorCategory.OPENROUTER_TIMEOUT,
                ErrorCategory.OPENROUTER_SERVER_ERROR,
                ErrorCategory.NETWORK_ERROR
            }
        )
    
    async def _execute_recovery(self, error_context: ErrorContext, **kwargs) -> bool:
        """Execute OpenRouter retry with exponential backoff"""
        if not error_context.should_retry():
            return False
        
        # Calculate backoff delay
        delay = error_context.get_backoff_delay()
        
        logger.info(f"OpenRouter retry strategy: waiting {delay:.2f}s before retry "
                   f"{error_context.retry_count + 1}/{error_context.max_retries} "
                   f"for agent {error_context.agent_id}")
        
        # Wait with exponential backoff
        await asyncio.sleep(delay)
        
        # Increment retry count
        error_context.retry_count += 1
        
        # Get the API client and retry the operation
        api_client = kwargs.get('api_client')
        if api_client:
            try:
                # Reset circuit breaker if it's in half-open state
                if hasattr(api_client, 'openrouter_circuit_half_open'):
                    api_client.openrouter_circuit_half_open = True
                
                return True  # Indicate that retry setup was successful
            except Exception as e:
                logger.error(f"OpenRouter retry setup failed: {e}")
                return False
        
        return False


class AgentRestartStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for agent failures requiring restart"""
    
    def __init__(self):
        super().__init__(
            "agent_restart",
            {
                ErrorCategory.AGENT_INITIALIZATION_ERROR,
                ErrorCategory.AGENT_COMMUNICATION_ERROR,
                ErrorCategory.AGENT_THINKING_ERROR
            }
        )
    
    async def _execute_recovery(self, error_context: ErrorContext, **kwargs) -> bool:
        """Execute agent restart recovery"""
        agent_manager = kwargs.get('agent_manager')
        if not agent_manager or not error_context.agent_id:
            return False
        
        logger.info(f"Agent restart strategy: restarting agent {error_context.agent_id} "
                   f"on core {error_context.core_id}")
        
        try:
            # Get the failed agent
            agent = agent_manager.get_agent_by_id(error_context.agent_id)
            if not agent:
                logger.error(f"Agent {error_context.agent_id} not found for restart")
                return False
            
            # Attempt to restart the agent
            success = await agent_manager.restart_agent(error_context.agent_id)
            
            if success:
                logger.info(f"Successfully restarted agent {error_context.agent_id}")
                return True
            else:
                logger.error(f"Failed to restart agent {error_context.agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Agent restart strategy failed for {error_context.agent_id}: {e}")
            return False


class WorkloadRedistributionStrategy(ErrorRecoveryStrategy):
    """Recovery strategy for redistributing workload when agents fail"""
    
    def __init__(self):
        super().__init__(
            "workload_redistribution",
            {
                ErrorCategory.AGENT_COMMUNICATION_ERROR,
                ErrorCategory.AGENT_THINKING_ERROR,
                ErrorCategory.SYSTEM_RESOURCE_ERROR
            }
        )
    
    async def _execute_recovery(self, error_context: ErrorContext, **kwargs) -> bool:
        """Execute workload redistribution recovery"""
        agent_manager = kwargs.get('agent_manager')
        if not agent_manager or not error_context.agent_id:
            return False
        
        logger.info(f"Workload redistribution strategy: redistributing workload from "
                   f"agent {error_context.agent_id}")
        
        try:
            # Redistribute workload from failed agent
            success = agent_manager.redistribute_workload(error_context.agent_id)
            
            if success:
                logger.info(f"Successfully redistributed workload from agent {error_context.agent_id}")
                return True
            else:
                logger.error(f"Failed to redistribute workload from agent {error_context.agent_id}")
                return False
                
        except Exception as e:
            logger.error(f"Workload redistribution strategy failed for {error_context.agent_id}: {e}")
            return False


class PerCoreErrorHandler:
    """
    Comprehensive error handler for per-core agent coordination system
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.PerCoreErrorHandler")
        
        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.error_counts: Dict[ErrorCategory, int] = {}
        self.recovery_strategies: List[ErrorRecoveryStrategy] = []
        
        # Thread safety
        self._error_lock = threading.RLock()
        
        # Circuit breaker state
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
        
        # Error processing queue
        self.error_queue: asyncio.Queue = asyncio.Queue()
        self._error_processor_task: Optional[asyncio.Task] = None
        self._shutdown = False
        
        self.logger.info("PerCoreErrorHandler initialized with comprehensive error recovery")
    
    def _initialize_recovery_strategies(self) -> None:
        """Initialize all available recovery strategies"""
        self.recovery_strategies = [
            OpenRouterRetryStrategy(),
            AgentRestartStrategy(),
            WorkloadRedistributionStrategy()
        ]
        
        self.logger.info(f"Initialized {len(self.recovery_strategies)} error recovery strategies")
    
    def categorize_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> ErrorCategory:
        """
        Categorize an error based on its type and message
        
        Args:
            error: The exception to categorize
            context: Additional context for categorization
            
        Returns:
            ErrorCategory for the error
        """
        error_str = str(error).lower()
        error_type = type(error).__name__.lower()
        
        # OpenRouter API errors
        if any(term in error_str for term in ["openrouter", "api key", "unauthorized"]):
            if any(term in error_str for term in ["401", "403", "unauthorized", "api key"]):
                return ErrorCategory.OPENROUTER_AUTH_ERROR
            elif any(term in error_str for term in ["429", "rate limit", "too many requests"]):
                return ErrorCategory.OPENROUTER_RATE_LIMIT
            elif any(term in error_str for term in ["timeout", "timed out"]):
                return ErrorCategory.OPENROUTER_TIMEOUT
            elif any(term in error_str for term in ["500", "502", "503", "504", "server error"]):
                return ErrorCategory.OPENROUTER_SERVER_ERROR
            else:
                return ErrorCategory.OPENROUTER_API_ERROR
        
        # Agent communication errors
        if any(term in error_str for term in ["agent", "communication", "message"]):
            return ErrorCategory.AGENT_COMMUNICATION_ERROR
        
        # Agent thinking errors
        if any(term in error_str for term in ["thinking", "session", "thought"]):
            return ErrorCategory.AGENT_THINKING_ERROR
        
        # Network errors
        if any(term in error_str for term in ["connection", "network", "dns", "resolve"]):
            return ErrorCategory.NETWORK_ERROR
        
        # System resource errors
        if any(term in error_str for term in ["memory", "cpu", "resource", "disk"]):
            return ErrorCategory.SYSTEM_RESOURCE_ERROR
        
        # Configuration errors
        if any(term in error_str for term in ["config", "setting", "parameter"]):
            return ErrorCategory.CONFIGURATION_ERROR
        
        # Agent initialization errors
        if any(term in error_str for term in ["initialization", "startup", "init"]):
            return ErrorCategory.AGENT_INITIALIZATION_ERROR
        
        return ErrorCategory.UNKNOWN_ERROR
    
    def determine_severity(self, error: Exception, category: ErrorCategory, 
                          context: Optional[Dict[str, Any]] = None) -> ErrorSeverity:
        """
        Determine the severity of an error
        
        Args:
            error: The exception
            category: The error category
            context: Additional context
            
        Returns:
            ErrorSeverity for the error
        """
        # Critical errors that affect system stability
        if category in [
            ErrorCategory.SYSTEM_RESOURCE_ERROR,
            ErrorCategory.CONFIGURATION_ERROR
        ]:
            return ErrorSeverity.CRITICAL
        
        # High severity errors that affect agent functionality
        if category in [
            ErrorCategory.AGENT_INITIALIZATION_ERROR,
            ErrorCategory.OPENROUTER_AUTH_ERROR
        ]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors that can be recovered from
        if category in [
            ErrorCategory.AGENT_COMMUNICATION_ERROR,
            ErrorCategory.AGENT_THINKING_ERROR,
            ErrorCategory.OPENROUTER_API_ERROR,
            ErrorCategory.OPENROUTER_SERVER_ERROR,
            ErrorCategory.NETWORK_ERROR
        ]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors that are temporary
        if category in [
            ErrorCategory.OPENROUTER_RATE_LIMIT,
            ErrorCategory.OPENROUTER_TIMEOUT
        ]:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    async def handle_error(self, error: Exception, agent_id: Optional[str] = None,
                          core_id: Optional[int] = None, context: Optional[Dict[str, Any]] = None,
                          **recovery_kwargs) -> bool:
        """
        Handle an error with appropriate recovery strategy
        
        Args:
            error: The exception to handle
            agent_id: ID of the agent that encountered the error
            core_id: Core ID where the error occurred
            context: Additional context for error handling
            **recovery_kwargs: Additional parameters for recovery strategies
            
        Returns:
            True if error was successfully handled/recovered, False otherwise
        """
        # Create error context
        error_context = self._create_error_context(error, agent_id, core_id, context)
        
        # Log the error
        self._log_error(error_context)
        
        # Track error statistics
        with self._error_lock:
            self.error_history.append(error_context)
            self.error_counts[error_context.category] = self.error_counts.get(error_context.category, 0) + 1
            
            # Keep error history manageable
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-500:]
        
        # Check circuit breaker
        if self._should_circuit_break(error_context):
            self.logger.warning(f"Circuit breaker activated for {error_context.category.value}")
            return False
        
        # Attempt recovery
        recovery_successful = await self._attempt_recovery(error_context, **recovery_kwargs)
        
        # Update circuit breaker state
        self._update_circuit_breaker(error_context, recovery_successful)
        
        return recovery_successful
    
    def _create_error_context(self, error: Exception, agent_id: Optional[str],
                             core_id: Optional[int], context: Optional[Dict[str, Any]]) -> ErrorContext:
        """Create error context from exception and metadata"""
        import traceback
        import uuid
        
        category = self.categorize_error(error, context)
        severity = self.determine_severity(error, category, context)
        
        return ErrorContext(
            error_id=str(uuid.uuid4()),
            category=category,
            severity=severity,
            agent_id=agent_id,
            core_id=core_id,
            error_message=str(error),
            exception_type=type(error).__name__,
            stack_trace=traceback.format_exc(),
            metadata=context or {}
        )
    
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with appropriate level based on severity"""
        log_data = error_context.to_dict()
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ERROR [{error_context.category.value}]: "
                               f"{error_context.error_message}", extra={"error_context": log_data})
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(f"HIGH SEVERITY ERROR [{error_context.category.value}]: "
                            f"{error_context.error_message}", extra={"error_context": log_data})
        elif error_context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"MEDIUM SEVERITY ERROR [{error_context.category.value}]: "
                              f"{error_context.error_message}", extra={"error_context": log_data})
        else:
            self.logger.info(f"LOW SEVERITY ERROR [{error_context.category.value}]: "
                           f"{error_context.error_message}", extra={"error_context": log_data})
    
    async def _attempt_recovery(self, error_context: ErrorContext, **recovery_kwargs) -> bool:
        """Attempt recovery using appropriate strategies"""
        error_context.recovery_attempted = True
        
        # Find applicable recovery strategies
        applicable_strategies = [
            strategy for strategy in self.recovery_strategies
            if strategy.can_handle(error_context)
        ]
        
        if not applicable_strategies:
            self.logger.warning(f"No recovery strategies available for error category {error_context.category.value}")
            return False
        
        # Sort strategies by success rate (best first)
        applicable_strategies.sort(key=lambda s: s.get_success_rate(), reverse=True)
        
        # Try each strategy until one succeeds
        for strategy in applicable_strategies:
            try:
                self.logger.info(f"Attempting recovery with strategy '{strategy.name}' "
                               f"for error {error_context.error_id}")
                
                success = await strategy.recover(error_context, **recovery_kwargs)
                
                if success:
                    error_context.recovery_successful = True
                    self.logger.info(f"Recovery successful with strategy '{strategy.name}' "
                                   f"for error {error_context.error_id}")
                    return True
                else:
                    self.logger.warning(f"Recovery failed with strategy '{strategy.name}' "
                                      f"for error {error_context.error_id}")
                    
            except Exception as e:
                self.logger.error(f"Recovery strategy '{strategy.name}' raised exception: {e}")
                continue
        
        self.logger.error(f"All recovery strategies failed for error {error_context.error_id}")
        return False
    
    def _should_circuit_break(self, error_context: ErrorContext) -> bool:
        """Check if circuit breaker should activate for this error category"""
        category_key = error_context.category.value
        
        if category_key not in self.circuit_breakers:
            self.circuit_breakers[category_key] = {
                "failure_count": 0,
                "last_failure": 0,
                "circuit_open": False,
                "circuit_open_time": 0,
                "threshold": 5,  # Open circuit after 5 failures
                "timeout": 60    # Keep circuit open for 60 seconds
            }
        
        breaker = self.circuit_breakers[category_key]
        current_time = time.time()
        
        # Check if circuit is open and should be closed
        if breaker["circuit_open"]:
            if current_time - breaker["circuit_open_time"] > breaker["timeout"]:
                breaker["circuit_open"] = False
                breaker["failure_count"] = 0
                self.logger.info(f"Circuit breaker closed for {category_key}")
                return False
            else:
                return True
        
        return False
    
    def _update_circuit_breaker(self, error_context: ErrorContext, recovery_successful: bool) -> None:
        """Update circuit breaker state based on recovery result"""
        category_key = error_context.category.value
        
        if category_key not in self.circuit_breakers:
            return
        
        breaker = self.circuit_breakers[category_key]
        current_time = time.time()
        
        if recovery_successful:
            # Reset failure count on successful recovery
            breaker["failure_count"] = 0
        else:
            # Increment failure count
            breaker["failure_count"] += 1
            breaker["last_failure"] = current_time
            
            # Check if we should open the circuit
            if breaker["failure_count"] >= breaker["threshold"]:
                breaker["circuit_open"] = True
                breaker["circuit_open_time"] = current_time
                self.logger.warning(f"Circuit breaker opened for {category_key} "
                                  f"after {breaker['failure_count']} failures")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        with self._error_lock:
            total_errors = len(self.error_history)
            
            # Calculate error rates by category
            category_stats = {}
            for category, count in self.error_counts.items():
                category_stats[category.value] = {
                    "count": count,
                    "percentage": (count / total_errors * 100) if total_errors > 0 else 0
                }
            
            # Calculate recovery success rates
            recovery_stats = {}
            for strategy in self.recovery_strategies:
                recovery_stats[strategy.name] = {
                    "success_count": strategy.success_count,
                    "failure_count": strategy.failure_count,
                    "success_rate": strategy.get_success_rate(),
                    "last_used": strategy.last_used
                }
            
            # Circuit breaker status
            circuit_status = {}
            for category, breaker in self.circuit_breakers.items():
                circuit_status[category] = {
                    "circuit_open": breaker["circuit_open"],
                    "failure_count": breaker["failure_count"],
                    "last_failure": breaker["last_failure"]
                }
            
            return {
                "total_errors": total_errors,
                "error_categories": category_stats,
                "recovery_strategies": recovery_stats,
                "circuit_breakers": circuit_status,
                "recent_errors": [ctx.to_dict() for ctx in self.error_history[-10:]]
            }
    
    def shutdown(self) -> None:
        """Shutdown the error handler"""
        self._shutdown = True
        if self._error_processor_task:
            self._error_processor_task.cancel()
        
        self.logger.info("PerCoreErrorHandler shutdown completed")


# Global error handler instance
_error_handler: Optional[PerCoreErrorHandler] = None
_error_handler_lock = threading.Lock()

logger = logging.getLogger(__name__)


def get_per_core_error_handler() -> PerCoreErrorHandler:
    """Get the global per-core error handler instance"""
    global _error_handler
    
    if _error_handler is None:
        with _error_handler_lock:
            if _error_handler is None:
                _error_handler = PerCoreErrorHandler()
    
    return _error_handler


def shutdown_error_handler() -> None:
    """Shutdown the global error handler"""
    global _error_handler
    
    if _error_handler:
        _error_handler.shutdown()
        _error_handler = None