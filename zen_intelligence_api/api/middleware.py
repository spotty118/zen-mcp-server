"""
Custom middleware for the Zen Intelligence API

This module provides custom middleware for logging, security, and monitoring.
"""

import time
import logging
import uuid
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


# Configure logging
logger = logging.getLogger("zen_intelligence_api")


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive request/response logging."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with logging."""
        
        # Generate request ID
        request_id = str(uuid.uuid4())
        
        # Start timing
        start_time = time.time()
        
        # Log request
        logger.info(
            f"Request {request_id}: {request.method} {request.url.path} "
            f"from {request.client.host if request.client else 'unknown'}"
        )
        
        # Add request ID to request state
        request.state.request_id = request_id
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Log response
            logger.info(
                f"Response {request_id}: {response.status_code} "
                f"in {duration:.3f}s"
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Processing-Time"] = f"{duration:.3f}s"
            
            return response
            
        except Exception as e:
            # Calculate duration
            duration = time.time() - start_time
            
            # Log error
            logger.error(
                f"Error {request_id}: {str(e)} "
                f"after {duration:.3f}s"
            )
            
            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "request_id": request_id,
                    "message": str(e)
                },
                headers={
                    "X-Request-ID": request_id,
                    "X-Processing-Time": f"{duration:.3f}s"
                }
            )


class SecurityMiddleware(BaseHTTPMiddleware):
    """Middleware for security headers and basic protection."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with security measures."""
        
        # Check rate limiting (simplified)
        client_ip = request.client.host if request.client else "unknown"
        
        # TODO: Implement proper rate limiting with Redis or in-memory store
        # For now, we'll just log the IP
        logger.debug(f"Request from IP: {client_ip}")
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response


class MonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for performance monitoring and metrics collection."""
    
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.total_duration = 0.0
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with monitoring."""
        
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Update metrics
            self.request_count += 1
            duration = time.time() - start_time
            self.total_duration += duration
            
            # Log performance metrics periodically
            if self.request_count % 100 == 0:
                avg_duration = self.total_duration / self.request_count
                error_rate = self.error_count / self.request_count
                
                logger.info(
                    f"Performance metrics: {self.request_count} requests, "
                    f"avg duration: {avg_duration:.3f}s, "
                    f"error rate: {error_rate:.3%}"
                )
            
            return response
            
        except Exception as e:
            self.error_count += 1
            raise


def add_logging_middleware(app: FastAPI) -> None:
    """Add logging middleware to the FastAPI app."""
    app.add_middleware(LoggingMiddleware)


def add_security_middleware(app: FastAPI) -> None:
    """Add security middleware to the FastAPI app."""
    app.add_middleware(SecurityMiddleware)


def add_monitoring_middleware(app: FastAPI) -> None:
    """Add monitoring middleware to the FastAPI app."""
    app.add_middleware(MonitoringMiddleware)