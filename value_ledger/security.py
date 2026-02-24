# value_ledger/security.py
"""
Security Integration Module for Value Ledger.

Integrates with:
- Boundary-SIEM: Security event reporting and monitoring
- Boundary Daemon: Connection protection and policy enforcement

This module provides:
- Custom exception hierarchy for structured error handling
- SIEM event reporting (CEF format, JSON HTTP)
- Connection protection via Boundary Daemon policy checks
- Security event logging with hash chains
- Decorators for protected operations
"""

from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import socket
import threading
import time
import urllib.request
import urllib.error
import uuid
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TypeVar
from contextlib import contextmanager

# Configure module logger
logger = logging.getLogger("value_ledger.security")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)


_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)


def get_correlation_id() -> str:
    """Get or create a correlation ID for the current operation."""
    cid = _correlation_id.get()
    if cid is None:
        cid = uuid.uuid4().hex[:12]
        _correlation_id.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Explicitly set correlation ID (e.g., from external request)."""
    _correlation_id.set(cid)


@contextmanager
def correlation_scope(cid: Optional[str] = None):
    """Context manager that sets a correlation ID for the duration."""
    token = _correlation_id.set(cid or uuid.uuid4().hex[:12])
    try:
        yield _correlation_id.get()
    finally:
        _correlation_id.reset(token)


class _JSONFormatter(logging.Formatter):
    """JSON log formatter for structured log aggregation."""

    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        cid = _correlation_id.get()
        if cid is not None:
            log_entry["correlation_id"] = cid
        return json.dumps(log_entry)


def configure_logging(
    json_format: Optional[bool] = None,
    level: Optional[str] = None,
) -> None:
    """
    Configure logging for all value_ledger modules.

    Args:
        json_format: Use JSON log format. Default: checks VALUE_LEDGER_LOG_FORMAT env var.
        level: Log level. Default: checks VALUE_LEDGER_LOG_LEVEL env var, then INFO.
    """
    if json_format is None:
        json_format = os.environ.get("VALUE_LEDGER_LOG_FORMAT", "").lower() == "json"
    if level is None:
        level = os.environ.get("VALUE_LEDGER_LOG_LEVEL", "INFO")

    root_logger = logging.getLogger("value_ledger")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    root_logger.handlers.clear()

    handler = logging.StreamHandler()
    if json_format:
        handler.setFormatter(_JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
    root_logger.addHandler(handler)


# =============================================================================
# Custom Exception Hierarchy
# =============================================================================


class ValueLedgerError(Exception):
    """Base exception for all Value Ledger errors."""

    def __init__(self, message: str, code: str = "VL_ERROR", details: Optional[Dict] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


class ValidationError(ValueLedgerError):
    """Raised when data validation fails."""

    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, code="VL_VALIDATION_ERROR", **kwargs)
        self.field = field
        if field:
            self.details["field"] = field


class StorageError(ValueLedgerError):
    """Raised when storage operations fail."""

    def __init__(self, message: str, path: Optional[str] = None, **kwargs):
        super().__init__(message, code="VL_STORAGE_ERROR", **kwargs)
        if path:
            self.details["path"] = path


class CryptographyError(ValueLedgerError):
    """Raised when cryptographic operations fail."""

    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(message, code="VL_CRYPTO_ERROR", **kwargs)
        if operation:
            self.details["operation"] = operation


class IntegrationError(ValueLedgerError):
    """Raised when external integration fails."""

    def __init__(self, message: str, service: Optional[str] = None, **kwargs):
        super().__init__(message, code="VL_INTEGRATION_ERROR", **kwargs)
        if service:
            self.details["service"] = service


class SecurityError(ValueLedgerError):
    """Raised when security violations are detected."""

    def __init__(self, message: str, violation_type: Optional[str] = None, **kwargs):
        super().__init__(message, code="VL_SECURITY_ERROR", **kwargs)
        if violation_type:
            self.details["violation_type"] = violation_type


class ConnectionProtectionError(ValueLedgerError):
    """Raised when connection protection denies access."""

    def __init__(self, message: str, policy: Optional[str] = None, **kwargs):
        super().__init__(message, code="VL_CONNECTION_DENIED", **kwargs)
        if policy:
            self.details["policy"] = policy


class RateLimitError(ValueLedgerError):
    """Raised when rate limits are exceeded."""

    def __init__(self, message: str, limit: Optional[int] = None, **kwargs):
        super().__init__(message, code="VL_RATE_LIMIT", **kwargs)
        if limit:
            self.details["limit"] = limit


# =============================================================================
# Security Event Types
# =============================================================================


class SecurityEventSeverity(Enum):
    """Severity levels for security events (CEF compatible)."""

    DEBUG = 0
    INFO = 1
    LOW = 3
    MEDIUM = 5
    HIGH = 7
    CRITICAL = 9


class SecurityEventType(Enum):
    """Types of security events to report to SIEM."""

    # Authentication & Access
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    AUTHENTICATION_SUCCESS = "auth_success"
    AUTHENTICATION_FAILURE = "auth_failure"

    # Data Operations
    LEDGER_READ = "ledger_read"
    LEDGER_WRITE = "ledger_write"
    ENTRY_CREATED = "entry_created"
    ENTRY_REVOKED = "entry_revoked"
    ENTRY_AGGREGATED = "entry_aggregated"

    # Security Violations
    PATH_TRAVERSAL_ATTEMPT = "path_traversal_attempt"
    SSRF_ATTEMPT = "ssrf_attempt"
    INJECTION_ATTEMPT = "injection_attempt"
    TAMPERING_DETECTED = "tampering_detected"
    CLOCK_DRIFT_DETECTED = "clock_drift_detected"

    # Connection & Network
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_BLOCKED = "connection_blocked"
    POLICY_VIOLATION = "policy_violation"

    # System Events
    SERVICE_START = "service_start"
    SERVICE_STOP = "service_stop"
    CONFIGURATION_CHANGE = "config_change"
    ERROR_OCCURRED = "error_occurred"


# =============================================================================
# Security Event
# =============================================================================


@dataclass
class SecurityEvent:
    """
    Represents a security event for SIEM reporting.

    Compatible with CEF (Common Event Format) and JSON HTTP ingestion.
    """

    event_type: SecurityEventType
    severity: SecurityEventSeverity
    source: str
    message: str
    timestamp: float = field(default_factory=time.time)
    outcome: str = "success"  # success, failure, unknown
    actor: Optional[str] = None
    target: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    event_id: Optional[str] = None

    def __post_init__(self):
        if not self.event_id:
            self.event_id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate deterministic event ID."""
        data = f"{self.timestamp}:{self.source}:{self.event_type.value}:{self.message}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_json(self) -> Dict[str, Any]:
        """Convert to JSON format for HTTP ingestion."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "source": self.source,
            "action": self.event_type.value,
            "outcome": self.outcome,
            "severity": self.severity.value,
            "severity_name": self.severity.name,
            "message": self.message,
            "actor": self.actor,
            "target": self.target,
            "details": self.details,
        }

    def to_cef(self) -> str:
        """
        Convert to CEF (Common Event Format) string.

        Format: CEF:0|Vendor|Product|Version|EventID|EventName|Severity|Extensions
        """
        extensions = []
        if self.actor:
            extensions.append(f"suser={self.actor}")
        if self.target:
            extensions.append(f"dhost={self.target}")
        extensions.append(f"msg={self.message}")
        extensions.append(f"outcome={self.outcome}")
        extensions.append(f"rt={int(self.timestamp * 1000)}")

        for key, value in self.details.items():
            safe_key = str(key).replace("=", "_").replace(" ", "_")[:23]
            safe_value = str(value).replace("\\", "\\\\").replace("=", "\\=")[:1024]
            extensions.append(f"cs1Label={safe_key} cs1={safe_value}")

        ext_str = " ".join(extensions)

        return (
            f"CEF:0|ValueLedger|Ledger|1.0|{self.event_type.value}|"
            f"{self.event_type.name}|{self.severity.value}|{ext_str}"
        )


# =============================================================================
# Boundary SIEM Client
# =============================================================================


@dataclass
class SIEMConfig:
    """Configuration for Boundary SIEM connection."""

    enabled: bool = True
    http_endpoint: str = "http://localhost:8080/api/v1/events"
    cef_host: str = "localhost"
    cef_port: int = 1514
    use_tcp: bool = True
    timeout: float = 5.0
    retry_count: int = 3
    retry_delay: float = 1.0
    batch_size: int = 100
    source_identifier: str = "value-ledger"
    dry_run: bool = False
    dry_run_path: str = "siem_events.jsonl"


class BoundarySIEMClient:
    """
    Client for reporting security events to Boundary SIEM.

    Supports both JSON HTTP and CEF protocol ingestion.
    """

    def __init__(self, config: Optional[SIEMConfig] = None):
        self.config = config or SIEMConfig()
        self._event_buffer: List[SecurityEvent] = []
        self._last_flush = time.time()
        self._hash_chain: Optional[str] = None
        self._connected = False

        if self.config.enabled:
            self._test_connection()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self._connected = False
        return False

    def _test_connection(self) -> bool:
        """Test SIEM connectivity."""
        try:
            # Try HTTP endpoint
            req = urllib.request.Request(
                self.config.http_endpoint.replace("/events", "/health"),
                method="GET",
            )
            req.add_header("User-Agent", "ValueLedger/1.0")
            with urllib.request.urlopen(req, timeout=2) as response:
                self._connected = response.status == 200
        except (urllib.error.URLError, socket.error, OSError) as e:
            logger.debug(f"SIEM HTTP health check failed: {e}")
            self._connected = False

        return self._connected

    def _update_hash_chain(self, event: SecurityEvent) -> str:
        """Update hash chain with new event for tamper detection."""
        event_hash = hashlib.sha256(json.dumps(event.to_json()).encode()).hexdigest()
        if self._hash_chain:
            combined = f"{self._hash_chain}:{event_hash}"
            self._hash_chain = hashlib.sha256(combined.encode()).hexdigest()
        else:
            self._hash_chain = event_hash
        return self._hash_chain

    def report(self, event: SecurityEvent) -> bool:
        """
        Report a security event to SIEM.

        Args:
            event: The security event to report

        Returns:
            True if event was successfully reported or buffered
        """
        if not self.config.enabled:
            logger.debug(f"SIEM disabled, skipping event: {event.event_type.value}")
            return True

        if self.config.dry_run:
            logger.info(f"[DRY RUN] SIEM event: {event.event_type.value}")
            with open(self.config.dry_run_path, "a") as f:
                json.dump(event.to_json(), f)
                f.write("\n")
            return True

        # Update hash chain
        chain_hash = self._update_hash_chain(event)
        event.details["_chain_hash"] = chain_hash

        # Buffer the event
        self._event_buffer.append(event)

        # Log locally
        logger.info(
            f"Security event: {event.event_type.value} | "
            f"severity={event.severity.name} | outcome={event.outcome}"
        )

        # Flush if buffer is full or event is critical
        if (
            len(self._event_buffer) >= self.config.batch_size
            or event.severity.value >= SecurityEventSeverity.HIGH.value
        ):
            return self.flush()

        return True

    def flush(self) -> bool:
        """Flush buffered events to SIEM."""
        if not self._event_buffer:
            return True

        if not self.config.enabled:
            self._event_buffer.clear()
            return True

        success = True
        events_to_send = self._event_buffer.copy()
        self._event_buffer.clear()

        # Try HTTP first
        if self._send_http(events_to_send):
            return True

        # Fall back to CEF/syslog
        for event in events_to_send:
            if not self._send_cef(event):
                success = False
                # Re-buffer failed events
                self._event_buffer.append(event)

        return success

    def _send_http(self, events: List[SecurityEvent]) -> bool:
        """Send events via HTTP POST."""
        try:
            payload = json.dumps({"events": [e.to_json() for e in events]}).encode()
            req = urllib.request.Request(
                self.config.http_endpoint,
                data=payload,
                method="POST",
            )
            req.add_header("Content-Type", "application/json")
            req.add_header("User-Agent", "ValueLedger/1.0")
            req.add_header("X-Source", self.config.source_identifier)

            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                return response.status in (200, 201, 202)

        except urllib.error.URLError as e:
            logger.warning(f"SIEM HTTP send failed: {e}")
            return False
        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"SIEM HTTP unexpected error: {e}")
            return False

    def _send_cef(self, event: SecurityEvent) -> bool:
        """Send event via CEF protocol."""
        try:
            cef_message = event.to_cef() + "\n"
            if self.config.use_tcp:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                    sock.settimeout(self.config.timeout)
                    sock.connect((self.config.cef_host, self.config.cef_port))
                    sock.sendall(cef_message.encode())
            else:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.sendto(
                        cef_message.encode(),
                        (self.config.cef_host, self.config.cef_port),
                    )
            return True

        except socket.error as e:
            logger.warning(f"SIEM CEF send failed: {e}")
            return False
        except OSError as e:
            logger.error(f"SIEM CEF unexpected error: {e}")
            return False

    def report_error(self, error: Exception, context: Optional[Dict] = None) -> bool:
        """Convenience method to report an error as a security event."""
        severity = SecurityEventSeverity.MEDIUM
        if isinstance(error, SecurityError):
            severity = SecurityEventSeverity.HIGH
        elif isinstance(error, ConnectionProtectionError):
            severity = SecurityEventSeverity.CRITICAL

        event = SecurityEvent(
            event_type=SecurityEventType.ERROR_OCCURRED,
            severity=severity,
            source=self.config.source_identifier,
            message=str(error),
            outcome="failure",
            details={
                "error_type": type(error).__name__,
                "error_code": getattr(error, "code", "UNKNOWN"),
                **(context or {}),
            },
        )
        return self.report(event)


# =============================================================================
# Boundary Daemon Client
# =============================================================================


class BoundaryMode(Enum):
    """Boundary Daemon protection modes."""

    OPEN = "open"  # Minimal restrictions
    RESTRICTED = "restricted"  # Standard protection
    TRUSTED = "trusted"  # Verified connections only
    AIRGAP = "airgap"  # No network access
    COLDROOM = "coldroom"  # Maximum isolation
    LOCKDOWN = "lockdown"  # Emergency lockdown


@dataclass
class PolicyDecision:
    """Result of a policy check from Boundary Daemon."""

    allowed: bool
    mode: BoundaryMode
    reason: str
    constraints: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class BoundaryDaemonConfig:
    """Configuration for Boundary Daemon connection."""

    enabled: bool = True
    socket_path: str = "/var/run/boundary-daemon/api.sock"
    http_fallback: str = "http://localhost:9090/api/v1"
    timeout: float = 2.0
    default_deny: bool = True  # Fail-closed
    cache_ttl: float = 60.0
    dry_run: bool = False


class BoundaryDaemonClient:
    """
    Client for Boundary Daemon connection protection.

    Provides policy checks and connection protection.
    """

    def __init__(self, config: Optional[BoundaryDaemonConfig] = None):
        self.config = config or BoundaryDaemonConfig()
        self._policy_cache: Dict[str, PolicyDecision] = {}
        self._current_mode: Optional[BoundaryMode] = None
        self._connected = False

        if self.config.enabled:
            self._check_daemon()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._connected = False
        self._policy_cache.clear()
        return False

    def _check_daemon(self) -> bool:
        """Check if Boundary Daemon is available."""
        if self.config.dry_run:
            logger.info("[DRY RUN] Boundary Daemon check skipped, all operations allowed")
            self._connected = True
            self._current_mode = BoundaryMode.OPEN
            return True

        # Try Unix socket first
        if Path(self.config.socket_path).exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                    sock.settimeout(1.0)
                    sock.connect(self.config.socket_path)
                    self._connected = True
                    return True
            except socket.error:
                pass

        # Try HTTP fallback
        try:
            req = urllib.request.Request(
                f"{self.config.http_fallback}/status",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=1.0) as response:
                if response.status == 200:
                    data = json.loads(response.read())
                    self._current_mode = BoundaryMode(data.get("mode", "restricted"))
                    self._connected = True
                    return True
        except (urllib.error.URLError, socket.error, json.JSONDecodeError, OSError) as e:
            logger.debug(f"Boundary Daemon HTTP fallback unavailable: {e}")

        self._connected = False
        return False

    def check_policy(
        self,
        action: str,
        resource: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> PolicyDecision:
        """
        Check if an action is allowed by current policy.

        Args:
            action: The action to check (e.g., "network_connect", "file_write")
            resource: Optional resource identifier
            context: Additional context for policy decision

        Returns:
            PolicyDecision with allow/deny and constraints
        """
        cache_key = f"{action}:{resource or ''}"

        # Check cache
        if cache_key in self._policy_cache:
            cached = self._policy_cache[cache_key]
            if time.time() - cached.timestamp < self.config.cache_ttl:
                return cached

        if not self.config.enabled:
            return PolicyDecision(
                allowed=True,
                mode=BoundaryMode.OPEN,
                reason="Protection disabled",
            )

        if not self._connected:
            # Fail-closed or fail-open based on config
            return PolicyDecision(
                allowed=not self.config.default_deny,
                mode=BoundaryMode.LOCKDOWN if self.config.default_deny else BoundaryMode.OPEN,
                reason="Daemon unavailable - default policy applied",
            )

        # Query daemon
        decision = self._query_policy(action, resource, context)
        self._policy_cache[cache_key] = decision
        return decision

    def _query_policy(
        self,
        action: str,
        resource: Optional[str],
        context: Optional[Dict],
    ) -> PolicyDecision:
        """Query Boundary Daemon for policy decision."""
        request_data = {
            "action": action,
            "resource": resource,
            "context": context or {},
            "timestamp": time.time(),
        }

        # Try Unix socket
        if Path(self.config.socket_path).exists():
            try:
                return self._query_socket(request_data)
            except (socket.error, json.JSONDecodeError, OSError) as e:
                logger.debug(f"Socket query failed: {e}")

        # Try HTTP
        try:
            return self._query_http(request_data)
        except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:
            logger.warning(f"HTTP policy query failed: {e}")

        # Default deny
        return PolicyDecision(
            allowed=not self.config.default_deny,
            mode=BoundaryMode.RESTRICTED,
            reason="Policy query failed - default applied",
        )

    def _query_socket(self, request_data: Dict) -> PolicyDecision:
        """Query policy via Unix socket."""
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(self.config.timeout)
            sock.connect(self.config.socket_path)
            sock.sendall(json.dumps(request_data).encode() + b"\n")

            response = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
                if b"\n" in chunk:
                    break

            data = json.loads(response.decode())
            return PolicyDecision(
                allowed=data.get("allowed", False),
                mode=BoundaryMode(data.get("mode", "restricted")),
                reason=data.get("reason", ""),
                constraints=data.get("constraints", {}),
            )

    def _query_http(self, request_data: Dict) -> PolicyDecision:
        """Query policy via HTTP API."""
        payload = json.dumps(request_data).encode()
        req = urllib.request.Request(
            f"{self.config.http_fallback}/policy/check",
            data=payload,
            method="POST",
        )
        req.add_header("Content-Type", "application/json")

        with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
            data = json.loads(response.read())
            return PolicyDecision(
                allowed=data.get("allowed", False),
                mode=BoundaryMode(data.get("mode", "restricted")),
                reason=data.get("reason", ""),
                constraints=data.get("constraints", {}),
            )

    def protect_connection(
        self,
        host: str,
        port: int,
        protocol: str = "tcp",
    ) -> PolicyDecision:
        """
        Check if a network connection is allowed.

        Args:
            host: Target hostname or IP
            port: Target port
            protocol: Protocol (tcp/udp)

        Returns:
            PolicyDecision
        """
        return self.check_policy(
            action="network_connect",
            resource=f"{protocol}://{host}:{port}",
            context={"host": host, "port": port, "protocol": protocol},
        )

    def protect_file_access(
        self,
        path: str,
        mode: str = "read",
    ) -> PolicyDecision:
        """
        Check if file access is allowed.

        Args:
            path: File path
            mode: Access mode (read/write/execute)

        Returns:
            PolicyDecision
        """
        return self.check_policy(
            action=f"file_{mode}",
            resource=path,
            context={"path": path, "mode": mode},
        )

    def report_violation(
        self,
        violation_type: str,
        details: Dict[str, Any],
    ) -> bool:
        """Report a security violation to Boundary Daemon."""
        if not self.config.enabled or not self._connected:
            return False

        event_data = {
            "type": "violation",
            "violation_type": violation_type,
            "details": details,
            "timestamp": time.time(),
        }

        try:
            payload = json.dumps(event_data).encode()
            req = urllib.request.Request(
                f"{self.config.http_fallback}/events",
                data=payload,
                method="POST",
            )
            req.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                return response.status in (200, 201, 202)
        except (urllib.error.URLError, socket.error, OSError) as e:
            logger.error(f"Failed to report violation: {e}")
            return False


# =============================================================================
# Unified Security Manager
# =============================================================================


class SecurityManager:
    """
    Unified security manager for Value Ledger.

    Coordinates SIEM reporting and Boundary Daemon protection.
    """

    _instance: Optional["SecurityManager"] = None
    _instance_lock = threading.Lock()

    def __init__(
        self,
        siem_config: Optional[SIEMConfig] = None,
        daemon_config: Optional[BoundaryDaemonConfig] = None,
    ):
        self.siem = BoundarySIEMClient(siem_config)
        self.daemon = BoundaryDaemonClient(daemon_config)
        self._operation_count = 0
        self._error_count = 0

    @classmethod
    def get_instance(cls) -> "SecurityManager":
        """Get or create singleton instance (thread-safe)."""
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @classmethod
    def configure(
        cls,
        siem_config: Optional[SIEMConfig] = None,
        daemon_config: Optional[BoundaryDaemonConfig] = None,
    ) -> "SecurityManager":
        """Configure and return the singleton instance."""
        cls._instance = cls(siem_config, daemon_config)
        return cls._instance

    def check_and_report(
        self,
        action: str,
        resource: Optional[str] = None,
        context: Optional[Dict] = None,
    ) -> PolicyDecision:
        """
        Check policy and report the access attempt.

        Args:
            action: Action being performed
            resource: Resource being accessed
            context: Additional context

        Returns:
            PolicyDecision

        Raises:
            ConnectionProtectionError: If access is denied
        """
        decision = self.daemon.check_policy(action, resource, context)

        # Report to SIEM (include correlation ID)
        cid = get_correlation_id()
        event = SecurityEvent(
            event_type=(
                SecurityEventType.ACCESS_GRANTED
                if decision.allowed
                else SecurityEventType.ACCESS_DENIED
            ),
            severity=(
                SecurityEventSeverity.INFO if decision.allowed else SecurityEventSeverity.HIGH
            ),
            source="value-ledger",
            message=f"Policy check: {action} on {resource or 'N/A'}",
            outcome="success" if decision.allowed else "failure",
            details={
                "action": action,
                "resource": resource,
                "mode": decision.mode.value,
                "reason": decision.reason,
                "correlation_id": cid,
                **(context or {}),
            },
        )
        self.siem.report(event)

        if not decision.allowed:
            raise ConnectionProtectionError(
                f"Access denied: {decision.reason}",
                policy=decision.mode.value,
                details={"action": action, "resource": resource},
            )

        return decision

    def report_security_event(
        self,
        event_type: SecurityEventType,
        message: str,
        severity: SecurityEventSeverity = SecurityEventSeverity.INFO,
        **kwargs,
    ) -> bool:
        """Report a security event."""
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            source="value-ledger",
            message=message,
            **kwargs,
        )
        return self.siem.report(event)

    def handle_error(
        self,
        error: Exception,
        context: Optional[Dict] = None,
        reraise: bool = True,
    ) -> None:
        """
        Handle and report an error.

        Args:
            error: The exception that occurred
            context: Additional context
            reraise: Whether to re-raise the exception
        """
        self._error_count += 1

        # Report to SIEM
        self.siem.report_error(error, context)

        # Report violation to daemon if security-related
        if isinstance(error, (SecurityError, ConnectionProtectionError)):
            self.daemon.report_violation(
                violation_type=getattr(error, "code", "SECURITY_ERROR"),
                details={
                    "message": str(error),
                    "error_type": type(error).__name__,
                    **(context or {}),
                },
            )

        if reraise:
            raise error

    def flush(self) -> bool:
        """Flush all pending events."""
        return self.siem.flush()


# =============================================================================
# Decorators for Protected Operations
# =============================================================================

F = TypeVar("F", bound=Callable[..., Any])


def protected_operation(
    action: str,
    resource_arg: Optional[str] = None,
    severity: SecurityEventSeverity = SecurityEventSeverity.INFO,
) -> Callable[[F], F]:
    """
    Decorator to protect and audit operations.

    Args:
        action: The action name for policy checks
        resource_arg: Name of the argument containing the resource
        severity: Severity level for the audit event
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            security = SecurityManager.get_instance()

            # Extract resource from arguments if specified
            resource = None
            if resource_arg:
                resource = kwargs.get(resource_arg)
                if resource is None and args:
                    # Try to get from positional args via function signature
                    import inspect

                    sig = inspect.signature(func)
                    params = list(sig.parameters.keys())
                    if resource_arg in params:
                        idx = params.index(resource_arg)
                        if idx < len(args):
                            resource = args[idx]

            try:
                # Check policy
                security.check_and_report(action, str(resource) if resource else None)

                # Execute operation
                result = func(*args, **kwargs)

                # Report success
                security.report_security_event(
                    event_type=(
                        SecurityEventType.LEDGER_WRITE
                        if "write" in action
                        else SecurityEventType.LEDGER_READ
                    ),
                    message=f"Operation completed: {action}",
                    severity=severity,
                    outcome="success",
                )

                return result

            except ConnectionProtectionError:
                raise
            except Exception as e:
                security.handle_error(e, context={"action": action, "resource": resource})

        return wrapper  # type: ignore

    return decorator


@contextmanager
def security_context(
    action: str,
    resource: Optional[str] = None,
):
    """
    Context manager for protected operations.

    Usage:
        with security_context("ledger_write", "/path/to/ledger.jsonl"):
            ledger.append(entry)
    """
    security = SecurityManager.get_instance()

    try:
        security.check_and_report(action, resource)
        yield security
        security.report_security_event(
            event_type=SecurityEventType.LEDGER_WRITE,
            message=f"Operation completed: {action}",
            outcome="success",
        )
    except Exception as e:
        security.handle_error(e, context={"action": action, "resource": resource}, reraise=True)


# =============================================================================
# Convenience Functions
# =============================================================================


def init_security(
    siem_endpoint: Optional[str] = None,
    daemon_socket: Optional[str] = None,
    enabled: bool = True,
) -> SecurityManager:
    """
    Initialize security subsystem.

    Args:
        siem_endpoint: HTTP endpoint for Boundary SIEM
        daemon_socket: Unix socket path for Boundary Daemon
        enabled: Whether security is enabled

    Returns:
        Configured SecurityManager
    """
    dry_run = os.environ.get("VALUE_LEDGER_DRY_RUN", "").lower() in ("1", "true", "yes")

    siem_config = SIEMConfig(
        enabled=enabled,
        http_endpoint=(
            siem_endpoint
            or os.environ.get("VALUE_LEDGER_SIEM_ENDPOINT", "http://localhost:8080/api/v1/events")
        ),
        timeout=float(os.environ.get("VALUE_LEDGER_SIEM_TIMEOUT", "5.0")),
        dry_run=dry_run,
    )

    daemon_config = BoundaryDaemonConfig(
        enabled=enabled,
        socket_path=(
            daemon_socket
            or os.environ.get("VALUE_LEDGER_DAEMON_SOCKET", "/var/run/boundary-daemon/api.sock")
        ),
        http_fallback=os.environ.get("VALUE_LEDGER_DAEMON_HTTP", "http://localhost:9090/api/v1"),
        timeout=float(os.environ.get("VALUE_LEDGER_DAEMON_TIMEOUT", "2.0")),
        dry_run=dry_run,
    )

    return SecurityManager.configure(siem_config, daemon_config)


def report_event(
    event_type: SecurityEventType,
    message: str,
    severity: SecurityEventSeverity = SecurityEventSeverity.INFO,
    **kwargs,
) -> bool:
    """Convenience function to report a security event."""
    return SecurityManager.get_instance().report_security_event(
        event_type, message, severity, **kwargs
    )


def check_connection(host: str, port: int) -> PolicyDecision:
    """Convenience function to check if a connection is allowed."""
    return SecurityManager.get_instance().daemon.protect_connection(host, port)


def check_file_access(path: str, mode: str = "read") -> PolicyDecision:
    """Convenience function to check if file access is allowed."""
    return SecurityManager.get_instance().daemon.protect_file_access(path, mode)
