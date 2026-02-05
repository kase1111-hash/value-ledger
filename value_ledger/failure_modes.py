# value_ledger/failure_modes.py
"""
Failure Mode Handling for Value Ledger.

This module provides clock monitoring, source validation, and unified failure
mode handling to maintain ledger integrity under adverse conditions.

Phase 2 - 17.5: Failure Mode Handling
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from .core import ValueLedger, LedgerEntry


class ClockMonitor:
    """
    Monitor for clock drift and suspicious timestamps.
    Helps maintain ledger integrity when system clock is unreliable.
    """

    def __init__(
        self,
        max_drift_seconds: float = 300.0,  # 5 minutes
        max_future_seconds: float = 60.0,  # 1 minute into future
    ):
        self.max_drift = max_drift_seconds
        self.max_future = max_future_seconds
        self.last_known_time: Optional[float] = None
        self.drift_events: List[Dict[str, Any]] = []

    def check_timestamp(self, timestamp: float) -> Dict[str, Any]:
        """
        Check if a timestamp is reasonable.

        Returns:
            Dict with 'valid', 'issues', and 'adjusted_timestamp'
        """
        current = time.time()
        issues = []
        adjusted = timestamp

        # Check for future timestamps
        if timestamp > current + self.max_future:
            issues.append(
                {
                    "type": "future_timestamp",
                    "delta": timestamp - current,
                    "severity": "high",
                }
            )
            adjusted = current  # Use current time instead

        # Check for extreme past (clock jumped back)
        if self.last_known_time and timestamp < self.last_known_time - self.max_drift:
            issues.append(
                {
                    "type": "clock_regression",
                    "expected_min": self.last_known_time,
                    "actual": timestamp,
                    "severity": "medium",
                }
            )

        # Check for unrealistic duration (clock skew)
        if self.last_known_time:
            expected_delta = current - self.last_known_time
            actual_delta = timestamp - self.last_known_time
            if abs(expected_delta - actual_delta) > self.max_drift:
                issues.append(
                    {
                        "type": "clock_skew",
                        "expected_delta": expected_delta,
                        "actual_delta": actual_delta,
                        "severity": "low",
                    }
                )

        # Update last known time
        self.last_known_time = current

        result = {
            "valid": len(issues) == 0,
            "original_timestamp": timestamp,
            "adjusted_timestamp": adjusted,
            "issues": issues,
            "checked_at": current,
        }

        if issues:
            self.drift_events.append(result)

        return result

    def get_drift_history(self) -> List[Dict[str, Any]]:
        """Get history of detected drift events."""
        return self.drift_events.copy()

    def reset(self):
        """Reset monitor state."""
        self.last_known_time = None
        self.drift_events = []


class SourceValidator:
    """
    Validate source entries before aggregation or processing.
    Handles classification mismatches and integrity checks.
    """

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.validation_errors: List[Dict[str, Any]] = []

    def validate_entry(self, entry: "LedgerEntry") -> Dict[str, Any]:
        """
        Validate a single entry for integrity issues.

        Returns:
            Dict with 'valid', 'errors', and 'warnings'
        """
        errors = []
        warnings = []

        # Check required fields
        if not entry.id:
            errors.append({"field": "id", "error": "Missing entry ID"})

        if not entry.intent_id:
            errors.append({"field": "intent_id", "error": "Missing intent ID"})

        # Check value vector validity
        vec = entry.value_vector.model_dump()
        for k, v in vec.items():
            if v < 0:
                errors.append(
                    {
                        "field": f"value_vector.{k}",
                        "error": f"Negative value: {v}",
                    }
                )

        # Check classification bounds
        if entry.classification < 0 or entry.classification > 5:
            errors.append(
                {
                    "field": "classification",
                    "error": f"Invalid classification: {entry.classification}",
                }
            )

        # Check for orphaned parent references
        if entry.parent_id and not entry.parent_ids:
            warnings.append(
                {
                    "field": "parent_ids",
                    "warning": "parent_id set but parent_ids empty",
                }
            )

        # Check revocation consistency
        if entry.status == "revoked" and not entry.revoked_at:
            warnings.append(
                {
                    "field": "revoked_at",
                    "warning": "Entry marked revoked but no revocation timestamp",
                }
            )

        result = {
            "valid": len(errors) == 0,
            "entry_id": entry.id,
            "errors": errors,
            "warnings": warnings,
        }

        if errors or (self.strict_mode and warnings):
            self.validation_errors.append(result)

        return result

    def validate_aggregation(
        self,
        entries: List["LedgerEntry"],
        requester_clearance: int = 0,
    ) -> Dict[str, Any]:
        """
        Validate entries for aggregation, checking classification compatibility.

        Returns:
            Dict with 'valid', 'errors', 'max_classification', and 'entries_validated'
        """
        errors = []
        max_classification = 0

        for entry in entries:
            # Check individual entry validity
            entry_result = self.validate_entry(entry)
            if not entry_result["valid"]:
                errors.append(
                    {
                        "entry_id": entry.id,
                        "type": "invalid_entry",
                        "details": entry_result["errors"],
                    }
                )

            # Track max classification
            max_classification = max(max_classification, entry.classification)

            # Check status
            if entry.status == "revoked":
                errors.append(
                    {
                        "entry_id": entry.id,
                        "type": "revoked_entry",
                        "error": "Cannot aggregate revoked entries",
                    }
                )

        # Check requester has sufficient clearance
        if requester_clearance < max_classification:
            errors.append(
                {
                    "type": "insufficient_clearance",
                    "required": max_classification,
                    "actual": requester_clearance,
                    "error": f"Requester clearance {requester_clearance} < required {max_classification}",
                }
            )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "max_classification": max_classification,
            "entries_validated": len(entries),
        }

    def check_classification_mismatch(
        self,
        entry: "LedgerEntry",
        target_classification: int,
    ) -> Dict[str, Any]:
        """
        Check if an entry's classification mismatches a target level.
        Used when entries from different classification levels are being combined.

        Returns:
            Dict with 'compatible', 'requires_upgrade', 'details'
        """
        if entry.classification <= target_classification:
            return {
                "compatible": True,
                "requires_upgrade": False,
                "entry_classification": entry.classification,
                "target_classification": target_classification,
            }
        else:
            return {
                "compatible": False,
                "requires_upgrade": True,
                "entry_classification": entry.classification,
                "target_classification": target_classification,
                "message": f"Entry has classification {entry.classification} but target is {target_classification}",
            }

    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get history of validation errors."""
        return self.validation_errors.copy()

    def reset(self):
        """Reset validator state."""
        self.validation_errors = []


class FailureModeHandler:
    """
    Unified handler for various failure modes in the ledger.
    Coordinates ClockMonitor and SourceValidator with ledger operations.
    """

    def __init__(
        self,
        ledger: "ValueLedger",
        clock_monitor: Optional[ClockMonitor] = None,
        source_validator: Optional[SourceValidator] = None,
    ):
        self.ledger = ledger
        self.clock_monitor = clock_monitor or ClockMonitor()
        self.source_validator = source_validator or SourceValidator()
        self.failure_log: List[Dict[str, Any]] = []

    def safe_accrue(
        self,
        intent_id: str,
        initial_vector: Dict[str, float],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Accrue with failure mode handling.

        Returns:
            Dict with 'success', 'entry_id', 'issues', and 'adjusted_timestamp'
        """
        issues = []

        # Check timestamp
        timestamp = kwargs.get("timestamp", time.time())
        clock_check = self.clock_monitor.check_timestamp(timestamp)
        if not clock_check["valid"]:
            issues.extend(clock_check["issues"])
            # Use adjusted timestamp
            kwargs["timestamp"] = clock_check["adjusted_timestamp"]

        try:
            entry_id = self.ledger.accrue(
                intent_id=intent_id,
                initial_vector=initial_vector,
                **kwargs,
            )
            return {
                "success": True,
                "entry_id": entry_id,
                "issues": issues,
                "adjusted_timestamp": clock_check.get("adjusted_timestamp"),
            }
        except Exception as e:
            failure = {
                "success": False,
                "entry_id": None,
                "error": str(e),
                "issues": issues,
                "timestamp": time.time(),
            }
            self.failure_log.append(failure)
            return failure

    def safe_aggregate(
        self,
        entry_ids: List[str],
        rule: str = "sum",
        requester_clearance: int = 0,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Aggregate with validation and failure mode handling.

        Returns:
            Dict with 'success', 'entry_id', 'validation_result'
        """
        # Gather entries for validation
        entries = []
        for eid in entry_ids:
            entry = self.ledger.get_entry(eid)
            if entry:
                entries.append(entry)

        # Validate entries
        validation = self.source_validator.validate_aggregation(
            entries=entries,
            requester_clearance=requester_clearance,
        )

        if not validation["valid"]:
            failure = {
                "success": False,
                "entry_id": None,
                "validation_result": validation,
                "error": "Validation failed",
                "timestamp": time.time(),
            }
            self.failure_log.append(failure)
            return failure

        try:
            entry_id = self.ledger.aggregate_entries(
                entry_ids=entry_ids,
                rule=rule,
                **kwargs,
            )
            return {
                "success": True,
                "entry_id": entry_id,
                "validation_result": validation,
            }
        except Exception as e:
            failure = {
                "success": False,
                "entry_id": None,
                "error": str(e),
                "validation_result": validation,
                "timestamp": time.time(),
            }
            self.failure_log.append(failure)
            return failure

    def get_failure_log(self) -> List[Dict[str, Any]]:
        """Get log of all failures."""
        return self.failure_log.copy()

    def get_health_report(self) -> Dict[str, Any]:
        """
        Generate a health report for the ledger.

        Returns:
            Dict with clock health, validation health, and overall status
        """
        clock_issues = self.clock_monitor.get_drift_history()
        validation_issues = self.source_validator.get_validation_history()

        return {
            "clock_health": {
                "drift_events": len(clock_issues),
                "recent_drift": clock_issues[-5:] if clock_issues else [],
            },
            "validation_health": {
                "error_count": len(validation_issues),
                "recent_errors": validation_issues[-5:] if validation_issues else [],
            },
            "failure_count": len(self.failure_log),
            "recent_failures": self.failure_log[-5:] if self.failure_log else [],
            "overall_status": (
                "healthy"
                if not (clock_issues or validation_issues or self.failure_log)
                else "degraded"
            ),
            "checked_at": time.time(),
        }

    def reset_monitors(self):
        """Reset all monitors and logs."""
        self.clock_monitor.reset()
        self.source_validator.reset()
        self.failure_log = []
