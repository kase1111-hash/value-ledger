# tests/test_security.py
"""
Tests for Value Ledger Security Integration Module.

Tests cover:
- Custom exception hierarchy
- Security event creation and formatting
- SIEM client functionality
- Boundary Daemon client functionality
- Security manager coordination
- Decorators and context managers
"""

import json
import pytest
import time
from unittest.mock import Mock, patch, MagicMock

from value_ledger.security import (
    # Exceptions
    ValueLedgerError,
    ValidationError,
    StorageError,
    CryptographyError,
    IntegrationError,
    SecurityError,
    ConnectionProtectionError,
    RateLimitError,
    # Event types
    SecurityEventSeverity,
    SecurityEventType,
    SecurityEvent,
    # SIEM integration
    SIEMConfig,
    BoundarySIEMClient,
    # Daemon integration
    BoundaryMode,
    PolicyDecision,
    BoundaryDaemonConfig,
    BoundaryDaemonClient,
    # Security manager
    SecurityManager,
    # Utilities
    protected_operation,
    security_context,
    init_security,
    report_event,
    check_connection,
    check_file_access,
)


# =============================================================================
# Exception Tests
# =============================================================================


class TestValueLedgerError:
    """Tests for base exception class."""

    def test_basic_creation(self):
        error = ValueLedgerError("Test error")
        assert str(error) == "Test error"
        assert error.code == "VL_ERROR"
        assert error.message == "Test error"
        assert error.details == {}
        assert error.timestamp > 0

    def test_with_code_and_details(self):
        error = ValueLedgerError("Test error", code="CUSTOM_CODE", details={"key": "value"})
        assert error.code == "CUSTOM_CODE"
        assert error.details == {"key": "value"}

    def test_to_dict(self):
        error = ValueLedgerError("Test error", code="TEST", details={"foo": "bar"})
        result = error.to_dict()
        assert result["error"] == "TEST"
        assert result["message"] == "Test error"
        assert result["details"]["foo"] == "bar"
        assert "timestamp" in result


class TestValidationError:
    """Tests for ValidationError."""

    def test_creation(self):
        error = ValidationError("Invalid field", field="username")
        assert error.code == "VL_VALIDATION_ERROR"
        assert error.field == "username"
        assert error.details["field"] == "username"


class TestStorageError:
    """Tests for StorageError."""

    def test_creation(self):
        error = StorageError("Write failed", path="/tmp/test.jsonl")
        assert error.code == "VL_STORAGE_ERROR"
        assert error.details["path"] == "/tmp/test.jsonl"


class TestCryptographyError:
    """Tests for CryptographyError."""

    def test_creation(self):
        error = CryptographyError("Encryption failed", operation="encrypt")
        assert error.code == "VL_CRYPTO_ERROR"
        assert error.details["operation"] == "encrypt"


class TestSecurityError:
    """Tests for SecurityError."""

    def test_creation(self):
        error = SecurityError("Access denied", violation_type="path_traversal")
        assert error.code == "VL_SECURITY_ERROR"
        assert error.details["violation_type"] == "path_traversal"


class TestConnectionProtectionError:
    """Tests for ConnectionProtectionError."""

    def test_creation(self):
        error = ConnectionProtectionError("Connection blocked", policy="AIRGAP")
        assert error.code == "VL_CONNECTION_DENIED"
        assert error.details["policy"] == "AIRGAP"


# =============================================================================
# Security Event Tests
# =============================================================================


class TestSecurityEventSeverity:
    """Tests for SecurityEventSeverity enum."""

    def test_severity_values(self):
        assert SecurityEventSeverity.DEBUG.value == 0
        assert SecurityEventSeverity.INFO.value == 1
        assert SecurityEventSeverity.LOW.value == 3
        assert SecurityEventSeverity.MEDIUM.value == 5
        assert SecurityEventSeverity.HIGH.value == 7
        assert SecurityEventSeverity.CRITICAL.value == 9


class TestSecurityEventType:
    """Tests for SecurityEventType enum."""

    def test_event_types_exist(self):
        assert SecurityEventType.ACCESS_GRANTED.value == "access_granted"
        assert SecurityEventType.ACCESS_DENIED.value == "access_denied"
        assert SecurityEventType.LEDGER_WRITE.value == "ledger_write"
        assert SecurityEventType.PATH_TRAVERSAL_ATTEMPT.value == "path_traversal_attempt"
        assert SecurityEventType.SSRF_ATTEMPT.value == "ssrf_attempt"


class TestSecurityEvent:
    """Tests for SecurityEvent dataclass."""

    def test_basic_creation(self):
        event = SecurityEvent(
            event_type=SecurityEventType.ACCESS_GRANTED,
            severity=SecurityEventSeverity.INFO,
            source="test",
            message="Test event",
        )
        assert event.event_type == SecurityEventType.ACCESS_GRANTED
        assert event.severity == SecurityEventSeverity.INFO
        assert event.source == "test"
        assert event.message == "Test event"
        assert event.outcome == "success"
        assert event.event_id is not None
        assert len(event.event_id) == 16

    def test_to_json(self):
        event = SecurityEvent(
            event_type=SecurityEventType.LEDGER_WRITE,
            severity=SecurityEventSeverity.MEDIUM,
            source="value-ledger",
            message="Entry created",
            actor="user123",
            target="/data/ledger.jsonl",
            details={"entry_id": "abc123"},
        )
        result = event.to_json()
        assert result["action"] == "ledger_write"
        assert result["severity"] == 5
        assert result["severity_name"] == "MEDIUM"
        assert result["actor"] == "user123"
        assert result["target"] == "/data/ledger.jsonl"
        assert result["details"]["entry_id"] == "abc123"

    def test_to_cef(self):
        event = SecurityEvent(
            event_type=SecurityEventType.ACCESS_DENIED,
            severity=SecurityEventSeverity.HIGH,
            source="value-ledger",
            message="Access denied to resource",
            actor="attacker",
            target="sensitive-file",
        )
        cef = event.to_cef()
        assert cef.startswith("CEF:0|ValueLedger|Ledger|1.0|")
        assert "access_denied" in cef
        assert "suser=attacker" in cef
        assert "dhost=sensitive-file" in cef
        assert "|7|" in cef  # HIGH severity

    def test_deterministic_event_id(self):
        """Events with same data should have same ID."""
        ts = time.time()
        event1 = SecurityEvent(
            event_type=SecurityEventType.LEDGER_READ,
            severity=SecurityEventSeverity.INFO,
            source="test",
            message="Test",
            timestamp=ts,
        )
        event2 = SecurityEvent(
            event_type=SecurityEventType.LEDGER_READ,
            severity=SecurityEventSeverity.INFO,
            source="test",
            message="Test",
            timestamp=ts,
        )
        assert event1.event_id == event2.event_id


# =============================================================================
# SIEM Client Tests
# =============================================================================


class TestSIEMConfig:
    """Tests for SIEMConfig."""

    def test_defaults(self):
        config = SIEMConfig()
        assert config.enabled is True
        assert "localhost:8080" in config.http_endpoint
        assert config.cef_port == 1514
        assert config.batch_size == 100

    def test_custom_config(self):
        config = SIEMConfig(
            enabled=False,
            http_endpoint="http://siem.example.com/events",
            cef_port=5514,
        )
        assert config.enabled is False
        assert config.http_endpoint == "http://siem.example.com/events"
        assert config.cef_port == 5514


class TestBoundarySIEMClient:
    """Tests for BoundarySIEMClient."""

    def test_disabled_client(self):
        config = SIEMConfig(enabled=False)
        client = BoundarySIEMClient(config)

        event = SecurityEvent(
            event_type=SecurityEventType.LEDGER_WRITE,
            severity=SecurityEventSeverity.INFO,
            source="test",
            message="Test event",
        )

        # Should succeed (no-op)
        assert client.report(event) is True
        assert client.flush() is True

    def test_event_buffering(self):
        config = SIEMConfig(enabled=True, batch_size=5)
        client = BoundarySIEMClient(config)
        client._connected = False  # Force no connection

        for i in range(3):
            event = SecurityEvent(
                event_type=SecurityEventType.LEDGER_READ,
                severity=SecurityEventSeverity.INFO,
                source="test",
                message=f"Event {i}",
            )
            client.report(event)

        assert len(client._event_buffer) == 3

    def test_hash_chain_update(self):
        config = SIEMConfig(enabled=False)
        client = BoundarySIEMClient(config)

        event1 = SecurityEvent(
            event_type=SecurityEventType.LEDGER_WRITE,
            severity=SecurityEventSeverity.INFO,
            source="test",
            message="Event 1",
        )
        event2 = SecurityEvent(
            event_type=SecurityEventType.LEDGER_WRITE,
            severity=SecurityEventSeverity.INFO,
            source="test",
            message="Event 2",
        )

        hash1 = client._update_hash_chain(event1)
        hash2 = client._update_hash_chain(event2)

        assert hash1 is not None
        assert hash2 is not None
        assert hash1 != hash2

    def test_report_error(self):
        config = SIEMConfig(enabled=False)
        client = BoundarySIEMClient(config)

        error = SecurityError("Test security error", violation_type="test")
        result = client.report_error(error, context={"extra": "info"})

        assert result is True


# =============================================================================
# Boundary Daemon Client Tests
# =============================================================================


class TestBoundaryMode:
    """Tests for BoundaryMode enum."""

    def test_modes(self):
        assert BoundaryMode.OPEN.value == "open"
        assert BoundaryMode.RESTRICTED.value == "restricted"
        assert BoundaryMode.TRUSTED.value == "trusted"
        assert BoundaryMode.AIRGAP.value == "airgap"
        assert BoundaryMode.COLDROOM.value == "coldroom"
        assert BoundaryMode.LOCKDOWN.value == "lockdown"


class TestPolicyDecision:
    """Tests for PolicyDecision dataclass."""

    def test_creation(self):
        decision = PolicyDecision(
            allowed=True,
            mode=BoundaryMode.RESTRICTED,
            reason="Policy allows this action",
            constraints={"max_size": 1024},
        )
        assert decision.allowed is True
        assert decision.mode == BoundaryMode.RESTRICTED
        assert decision.reason == "Policy allows this action"
        assert decision.constraints["max_size"] == 1024
        assert decision.timestamp > 0


class TestBoundaryDaemonConfig:
    """Tests for BoundaryDaemonConfig."""

    def test_defaults(self):
        config = BoundaryDaemonConfig()
        assert config.enabled is True
        assert config.default_deny is True
        assert config.cache_ttl == 60.0


class TestBoundaryDaemonClient:
    """Tests for BoundaryDaemonClient."""

    def test_disabled_client(self):
        config = BoundaryDaemonConfig(enabled=False)
        client = BoundaryDaemonClient(config)

        decision = client.check_policy("test_action")
        assert decision.allowed is True
        assert decision.mode == BoundaryMode.OPEN
        assert "disabled" in decision.reason.lower()

    def test_default_deny_when_unavailable(self):
        config = BoundaryDaemonConfig(enabled=True, default_deny=True)
        client = BoundaryDaemonClient(config)
        client._connected = False

        decision = client.check_policy("test_action")
        assert decision.allowed is False
        assert "unavailable" in decision.reason.lower()

    def test_default_allow_when_unavailable(self):
        config = BoundaryDaemonConfig(enabled=True, default_deny=False)
        client = BoundaryDaemonClient(config)
        client._connected = False

        decision = client.check_policy("test_action")
        assert decision.allowed is True

    def test_policy_caching(self):
        config = BoundaryDaemonConfig(enabled=True, cache_ttl=60.0)
        client = BoundaryDaemonClient(config)
        client._connected = False

        # First call
        decision1 = client.check_policy("action1", "resource1")

        # Manually cache a decision
        client._policy_cache["action1:resource1"] = PolicyDecision(
            allowed=True,
            mode=BoundaryMode.TRUSTED,
            reason="Cached",
        )

        # Second call should use cache
        decision2 = client.check_policy("action1", "resource1")
        assert decision2.reason == "Cached"

    def test_protect_connection(self):
        config = BoundaryDaemonConfig(enabled=False)
        client = BoundaryDaemonClient(config)

        decision = client.protect_connection("example.com", 443)
        assert decision.allowed is True

    def test_protect_file_access(self):
        config = BoundaryDaemonConfig(enabled=False)
        client = BoundaryDaemonClient(config)

        decision = client.protect_file_access("/tmp/test.txt", "read")
        assert decision.allowed is True


# =============================================================================
# Security Manager Tests
# =============================================================================


class TestSecurityManager:
    """Tests for SecurityManager."""

    def test_singleton_pattern(self):
        # Reset singleton
        SecurityManager._instance = None

        manager1 = SecurityManager.get_instance()
        manager2 = SecurityManager.get_instance()

        assert manager1 is manager2

    def test_configure(self):
        SecurityManager._instance = None

        siem_config = SIEMConfig(enabled=False)
        daemon_config = BoundaryDaemonConfig(enabled=False)

        manager = SecurityManager.configure(siem_config, daemon_config)

        assert manager.siem.config.enabled is False
        assert manager.daemon.config.enabled is False

    def test_report_security_event(self):
        SecurityManager._instance = None

        siem_config = SIEMConfig(enabled=False)
        daemon_config = BoundaryDaemonConfig(enabled=False)
        manager = SecurityManager.configure(siem_config, daemon_config)

        result = manager.report_security_event(
            event_type=SecurityEventType.LEDGER_WRITE,
            message="Test event",
            severity=SecurityEventSeverity.INFO,
        )

        assert result is True

    def test_check_and_report_allowed(self):
        SecurityManager._instance = None

        siem_config = SIEMConfig(enabled=False)
        daemon_config = BoundaryDaemonConfig(enabled=False)  # Disabled = allow all
        manager = SecurityManager.configure(siem_config, daemon_config)

        decision = manager.check_and_report("test_action", "test_resource")
        assert decision.allowed is True

    def test_check_and_report_denied(self):
        SecurityManager._instance = None

        siem_config = SIEMConfig(enabled=False)
        daemon_config = BoundaryDaemonConfig(enabled=True, default_deny=True)
        manager = SecurityManager.configure(siem_config, daemon_config)
        manager.daemon._connected = False  # Force deny

        with pytest.raises(ConnectionProtectionError):
            manager.check_and_report("test_action", "test_resource")

    def test_handle_error(self):
        SecurityManager._instance = None

        siem_config = SIEMConfig(enabled=False)
        daemon_config = BoundaryDaemonConfig(enabled=False)
        manager = SecurityManager.configure(siem_config, daemon_config)

        error = ValidationError("Test error")

        with pytest.raises(ValidationError):
            manager.handle_error(error, reraise=True)

        assert manager._error_count == 1

    def test_handle_error_no_reraise(self):
        SecurityManager._instance = None

        siem_config = SIEMConfig(enabled=False)
        daemon_config = BoundaryDaemonConfig(enabled=False)
        manager = SecurityManager.configure(siem_config, daemon_config)

        error = ValidationError("Test error")

        # Should not raise
        manager.handle_error(error, reraise=False)
        assert manager._error_count == 1


# =============================================================================
# Decorator and Context Manager Tests
# =============================================================================


class TestProtectedOperationDecorator:
    """Tests for @protected_operation decorator."""

    def test_successful_operation(self):
        SecurityManager._instance = None
        SecurityManager.configure(
            SIEMConfig(enabled=False),
            BoundaryDaemonConfig(enabled=False),
        )

        @protected_operation("test_action")
        def my_function(x, y):
            return x + y

        result = my_function(2, 3)
        assert result == 5

    def test_operation_with_resource_arg(self):
        SecurityManager._instance = None
        SecurityManager.configure(
            SIEMConfig(enabled=False),
            BoundaryDaemonConfig(enabled=False),
        )

        @protected_operation("file_read", resource_arg="path")
        def read_file(path: str):
            return f"contents of {path}"

        result = read_file("/tmp/test.txt")
        assert result == "contents of /tmp/test.txt"


class TestSecurityContext:
    """Tests for security_context context manager."""

    def test_successful_context(self):
        SecurityManager._instance = None
        SecurityManager.configure(
            SIEMConfig(enabled=False),
            BoundaryDaemonConfig(enabled=False),
        )

        with security_context("test_action", "test_resource") as mgr:
            assert mgr is not None
            result = 1 + 1

        assert result == 2

    def test_context_with_exception(self):
        SecurityManager._instance = None
        SecurityManager.configure(
            SIEMConfig(enabled=False),
            BoundaryDaemonConfig(enabled=False),
        )

        with pytest.raises(ValueError):
            with security_context("test_action", "test_resource"):
                raise ValueError("Test error")


# =============================================================================
# Convenience Function Tests
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_init_security(self):
        SecurityManager._instance = None

        manager = init_security(
            siem_endpoint="http://siem.example.com/events",
            daemon_socket="/tmp/test.sock",
            enabled=False,
        )

        assert manager is not None
        assert manager.siem.config.enabled is False

    def test_report_event(self):
        SecurityManager._instance = None
        init_security(enabled=False)

        result = report_event(
            SecurityEventType.LEDGER_WRITE,
            "Test message",
            SecurityEventSeverity.INFO,
        )

        assert result is True

    def test_check_connection(self):
        SecurityManager._instance = None
        init_security(enabled=False)

        decision = check_connection("example.com", 443)
        assert decision.allowed is True

    def test_check_file_access(self):
        SecurityManager._instance = None
        init_security(enabled=False)

        decision = check_file_access("/tmp/test.txt", "read")
        assert decision.allowed is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestSecurityIntegration:
    """Integration tests for security module."""

    def test_full_workflow(self):
        """Test complete security workflow."""
        SecurityManager._instance = None

        # Initialize
        manager = init_security(enabled=False)

        # Report startup event
        manager.report_security_event(
            event_type=SecurityEventType.SERVICE_START,
            message="Value Ledger started",
            severity=SecurityEventSeverity.INFO,
        )

        # Check policy
        decision = manager.daemon.check_policy("ledger_write", "/data/ledger.jsonl")
        assert decision.allowed is True

        # Simulate operation
        with security_context("ledger_write", "/data/ledger.jsonl"):
            pass  # Operation would go here

        # Report shutdown
        manager.report_security_event(
            event_type=SecurityEventType.SERVICE_STOP,
            message="Value Ledger stopped",
            severity=SecurityEventSeverity.INFO,
        )

        # Flush events
        assert manager.flush() is True

    def test_error_handling_workflow(self):
        """Test error handling through security system."""
        SecurityManager._instance = None
        manager = init_security(enabled=False)

        # Simulate various errors
        errors = [
            ValidationError("Invalid input", field="data"),
            StorageError("Write failed", path="/tmp/test"),
            SecurityError("Access denied", violation_type="unauthorized"),
        ]

        for error in errors:
            manager.handle_error(error, reraise=False)

        assert manager._error_count == 3

    def test_cef_format_compliance(self):
        """Verify CEF format is valid."""
        event = SecurityEvent(
            event_type=SecurityEventType.TAMPERING_DETECTED,
            severity=SecurityEventSeverity.CRITICAL,
            source="value-ledger",
            message="Merkle proof verification failed",
            actor="unknown",
            target="entry-abc123",
            details={"expected_hash": "abc", "actual_hash": "def"},
        )

        cef = event.to_cef()

        # Verify CEF structure
        parts = cef.split("|")
        assert parts[0] == "CEF:0"
        assert parts[1] == "ValueLedger"
        assert parts[2] == "Ledger"
        assert parts[3] == "1.0"
        assert parts[4] == "tampering_detected"
        assert parts[6] == "9"  # CRITICAL severity
