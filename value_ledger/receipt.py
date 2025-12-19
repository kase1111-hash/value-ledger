# value_ledger/receipt.py
"""
MP-02: Proof-of-Effort Receipt Protocol Implementation.

This module implements the Effort Receipt Protocol as specified in MP-02-spec.md.
Receipts assert that effort occurred with traceable provenance, without asserting
value, ownership, or compensation.

Key Principles (from MP-02):
- Process Over Artifact: Effort is validated as a process unfolding over time
- Continuity Matters: Temporal progression is a primary signal of genuine work
- Receipts, Not Claims: The protocol records evidence, not conclusions about value
- Model Skepticism: LLM assessments are advisory and must be reproducible
- Partial Observability: Uncertainty is preserved, not collapsed
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict, Any, Protocol
from enum import Enum
import json


class SignalType(str, Enum):
    """Types of effort signals that can be observed."""
    VOICE = "voice"
    TEXT = "text"
    COMMAND = "command"
    TOOL = "tool"
    KEYSTROKE = "keystroke"
    NAVIGATION = "navigation"
    UNKNOWN = "unknown"


class CaptureMode(str, Enum):
    """How signals are captured by observers."""
    CONTINUOUS = "continuous"
    INTERMITTENT = "intermittent"
    ON_DEMAND = "on_demand"


@dataclass
class EffortSignal:
    """
    A single observable signal of effort.

    Observers MUST:
    - Time-stamp all signals
    - Preserve ordering
    - Disclose capture modality

    Observers MUST NOT:
    - Alter raw signals
    - Infer intent beyond observed data
    """
    signal_type: str  # SignalType value
    timestamp: float
    hash: str  # SHA-256 hash of signal content
    modality: str  # How signal was captured
    sequence_number: int = 0  # Ordering within segment
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_content(
        cls,
        content: str,
        signal_type: str = SignalType.TEXT,
        modality: str = "direct",
        sequence_number: int = 0,
        metadata: Optional[Dict] = None,
    ) -> "EffortSignal":
        """Create a signal from raw content, computing its hash."""
        return cls(
            signal_type=signal_type,
            timestamp=time.time(),
            hash=hashlib.sha256(content.encode()).hexdigest(),
            modality=modality,
            sequence_number=sequence_number,
            metadata=metadata or {},
        )


@dataclass
class EffortSegment:
    """
    A bounded segment of effort containing multiple signals.

    Segments provide temporal boundaries for validation.
    Segmentation rules must be deterministic and disclosed.
    """
    segment_id: str
    start_time: float
    end_time: float
    signals: List[EffortSignal] = field(default_factory=list)
    segmentation_rule: str = "time_bounded"  # Must be deterministic
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        """Duration of segment in seconds."""
        return self.end_time - self.start_time

    @property
    def signal_count(self) -> int:
        """Number of signals in segment."""
        return len(self.signals)

    def get_signal_hashes(self) -> List[str]:
        """Get ordered list of signal hashes."""
        return [s.hash for s in sorted(self.signals, key=lambda x: x.sequence_number)]

    def compute_segment_hash(self) -> str:
        """Compute deterministic hash of entire segment."""
        data = {
            "segment_id": self.segment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "signal_hashes": self.get_signal_hashes(),
            "segmentation_rule": self.segmentation_rule,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class ValidationMetadata:
    """
    Metadata from validator assessment.

    Validators MUST:
    - Produce deterministic summaries
    - Disclose model identity and version
    - Preserve dissent and uncertainty

    Validators MUST NOT:
    - Declare effort as valuable
    - Assert originality or ownership
    - Collapse ambiguous signals into certainty
    """
    validator_id: str
    model_id: str
    model_version: str
    validation_timestamp: float = field(default_factory=time.time)
    coherence_score: Optional[float] = None  # 0.0-1.0 if computed
    progression_score: Optional[float] = None  # 0.0-1.0 if computed
    uncertainty_markers: List[str] = field(default_factory=list)  # Preserve ambiguity
    adversarial_flags: List[str] = field(default_factory=list)  # Detected patterns
    raw_assessment: Optional[str] = None  # Full validator output for audit


@dataclass
class EffortReceipt:
    """
    A complete Proof-of-Effort Receipt as specified in MP-02.

    The receipt asserts that effort occurred with traceable provenance.
    It does NOT assert value, ownership, or compensation.
    """
    receipt_id: str
    time_bounds: Tuple[float, float]  # (start, end)
    signal_hashes: List[str]  # Ordered hashes of all signals
    effort_summary: str  # Deterministic summary, suitable for NatLangChain
    validation_metadata: ValidationMetadata
    observer_id: str
    validator_id: str
    prior_receipts: List[str] = field(default_factory=list)  # Chain of prior work

    # Failure mode tracking (MP-02 §11)
    observation_gaps: List[Tuple[float, float]] = field(default_factory=list)
    conflicting_validations: List[str] = field(default_factory=list)
    suspected_manipulation: bool = False
    is_incomplete: bool = False

    # Cryptographic anchoring
    receipt_hash: Optional[str] = None
    merkle_ref: Optional[str] = None
    chain_anchor: Optional[str] = None  # NatLangChain reference

    def __post_init__(self):
        """Compute receipt hash if not set."""
        if not self.receipt_hash:
            self.receipt_hash = self.compute_hash()

    def compute_hash(self) -> str:
        """Compute deterministic hash of receipt for verification."""
        data = {
            "receipt_id": self.receipt_id,
            "time_bounds": list(self.time_bounds),
            "signal_hashes": self.signal_hashes,
            "effort_summary": self.effort_summary,
            "observer_id": self.observer_id,
            "validator_id": self.validator_id,
            "prior_receipts": self.prior_receipts,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary for serialization."""
        return {
            "receipt_id": self.receipt_id,
            "time_bounds": list(self.time_bounds),
            "signal_hashes": self.signal_hashes,
            "effort_summary": self.effort_summary,
            "validation_metadata": asdict(self.validation_metadata),
            "observer_id": self.observer_id,
            "validator_id": self.validator_id,
            "prior_receipts": self.prior_receipts,
            "observation_gaps": [list(g) for g in self.observation_gaps],
            "conflicting_validations": self.conflicting_validations,
            "suspected_manipulation": self.suspected_manipulation,
            "is_incomplete": self.is_incomplete,
            "receipt_hash": self.receipt_hash,
            "merkle_ref": self.merkle_ref,
            "chain_anchor": self.chain_anchor,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EffortReceipt":
        """Create receipt from dictionary."""
        vm_data = data.get("validation_metadata", {})
        validation_metadata = ValidationMetadata(
            validator_id=vm_data.get("validator_id", "unknown"),
            model_id=vm_data.get("model_id", "unknown"),
            model_version=vm_data.get("model_version", "unknown"),
            validation_timestamp=vm_data.get("validation_timestamp", time.time()),
            coherence_score=vm_data.get("coherence_score"),
            progression_score=vm_data.get("progression_score"),
            uncertainty_markers=vm_data.get("uncertainty_markers", []),
            adversarial_flags=vm_data.get("adversarial_flags", []),
            raw_assessment=vm_data.get("raw_assessment"),
        )

        return cls(
            receipt_id=data["receipt_id"],
            time_bounds=tuple(data["time_bounds"]),
            signal_hashes=data["signal_hashes"],
            effort_summary=data["effort_summary"],
            validation_metadata=validation_metadata,
            observer_id=data["observer_id"],
            validator_id=data["validator_id"],
            prior_receipts=data.get("prior_receipts", []),
            observation_gaps=[tuple(g) for g in data.get("observation_gaps", [])],
            conflicting_validations=data.get("conflicting_validations", []),
            suspected_manipulation=data.get("suspected_manipulation", False),
            is_incomplete=data.get("is_incomplete", False),
            receipt_hash=data.get("receipt_hash"),
            merkle_ref=data.get("merkle_ref"),
            chain_anchor=data.get("chain_anchor"),
        )


@dataclass
class VerificationResult:
    """Result of third-party receipt verification."""
    valid: bool
    receipt_id: str
    checks_passed: List[str]
    checks_failed: List[str]
    warnings: List[str]
    verified_at: float = field(default_factory=time.time)
    verifier_notes: Optional[str] = None


class Observer(Protocol):
    """
    Protocol for effort observers.

    Observers MUST:
    - Time-stamp all signals
    - Preserve ordering
    - Disclose capture modality

    Observers MUST NOT:
    - Alter raw signals
    - Infer intent beyond observed data
    """

    def capture_signal(self, signal: Any) -> EffortSignal:
        """Capture and return an effort signal."""
        ...

    def get_modality(self) -> str:
        """Return the observation modality."""
        ...

    def get_observer_id(self) -> str:
        """Return unique observer identifier."""
        ...

    def get_capture_mode(self) -> str:
        """Return capture mode: 'continuous', 'intermittent', or 'on_demand'."""
        ...


class Validator(Protocol):
    """
    Protocol for effort validators.

    Validators MUST:
    - Produce deterministic summaries
    - Disclose model identity and version
    - Preserve dissent and uncertainty

    Validators MUST NOT:
    - Declare effort as valuable
    - Assert originality or ownership
    - Collapse ambiguous signals into certainty
    """

    model_id: str
    model_version: str

    def validate_segment(self, segment: EffortSegment) -> ValidationMetadata:
        """Validate an effort segment."""
        ...

    def generate_summary(self, segment: EffortSegment) -> str:
        """Generate deterministic summary of effort."""
        ...

    def assess_coherence(self, segment: EffortSegment) -> float:
        """Assess coherence of effort signals (0.0-1.0)."""
        ...

    def assess_progression(self, segment: EffortSegment) -> float:
        """Assess temporal progression quality (0.0-1.0)."""
        ...

    def detect_adversarial_patterns(self, segment: EffortSegment) -> List[str]:
        """Detect potential adversarial or manipulation patterns."""
        ...


class DefaultObserver:
    """Default implementation of Observer for basic signal capture."""

    def __init__(self, observer_id: str = "default_observer", modality: str = "text"):
        self._observer_id = observer_id
        self._modality = modality
        self._capture_mode = CaptureMode.ON_DEMAND
        self._sequence = 0

    def capture_signal(self, content: str, signal_type: str = SignalType.TEXT) -> EffortSignal:
        """Capture a signal from content."""
        signal = EffortSignal.from_content(
            content=content,
            signal_type=signal_type,
            modality=self._modality,
            sequence_number=self._sequence,
        )
        self._sequence += 1
        return signal

    def get_modality(self) -> str:
        return self._modality

    def get_observer_id(self) -> str:
        return self._observer_id

    def get_capture_mode(self) -> str:
        return self._capture_mode.value


class DefaultValidator:
    """
    Default implementation of Validator using heuristic assessment.

    This validator produces deterministic summaries without LLM calls.
    For production, replace with model-backed validator.
    """

    def __init__(
        self,
        validator_id: str = "default_validator",
        model_id: str = "heuristic",
        model_version: str = "1.0.0",
    ):
        self.validator_id = validator_id
        self.model_id = model_id
        self.model_version = model_version

    def validate_segment(self, segment: EffortSegment) -> ValidationMetadata:
        """Validate a segment and return metadata."""
        coherence = self.assess_coherence(segment)
        progression = self.assess_progression(segment)
        adversarial = self.detect_adversarial_patterns(segment)

        uncertainty = []
        if coherence < 0.5:
            uncertainty.append("low_coherence")
        if progression < 0.5:
            uncertainty.append("weak_progression")
        if segment.signal_count < 3:
            uncertainty.append("insufficient_signals")

        return ValidationMetadata(
            validator_id=self.validator_id,
            model_id=self.model_id,
            model_version=self.model_version,
            coherence_score=coherence,
            progression_score=progression,
            uncertainty_markers=uncertainty,
            adversarial_flags=adversarial,
        )

    def generate_summary(self, segment: EffortSegment) -> str:
        """Generate deterministic summary of effort segment."""
        duration_mins = segment.duration / 60
        signal_types = set(s.signal_type for s in segment.signals)

        return (
            f"Effort segment {segment.segment_id[:8]} spanning {duration_mins:.1f} minutes "
            f"with {segment.signal_count} signals of types: {', '.join(sorted(signal_types))}. "
            f"Segmentation rule: {segment.segmentation_rule}."
        )

    def assess_coherence(self, segment: EffortSegment) -> float:
        """
        Assess coherence based on signal distribution.
        Higher score = more evenly distributed signals.
        """
        if segment.signal_count == 0:
            return 0.0
        if segment.duration == 0:
            return 0.5

        # Check for gaps in signal timestamps
        if segment.signal_count < 2:
            return 0.7  # Single signal is moderately coherent

        timestamps = sorted(s.timestamp for s in segment.signals)
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

        if not gaps:
            return 0.7

        avg_gap = sum(gaps) / len(gaps)
        expected_gap = segment.duration / segment.signal_count

        # Score based on gap consistency
        if avg_gap == 0:
            return 0.5
        consistency = min(1.0, expected_gap / avg_gap) if avg_gap > expected_gap else min(1.0, avg_gap / expected_gap)
        return 0.3 + (0.7 * consistency)

    def assess_progression(self, segment: EffortSegment) -> float:
        """
        Assess temporal progression quality.
        Higher score = clear temporal flow with variety.
        """
        if segment.signal_count == 0:
            return 0.0
        if segment.signal_count < 2:
            return 0.5

        # Check for variety in signal types
        signal_types = set(s.signal_type for s in segment.signals)
        type_variety = min(1.0, len(signal_types) / 3)

        # Check temporal ordering is preserved
        timestamps = [s.timestamp for s in segment.signals]
        sorted_timestamps = sorted(timestamps)
        ordering_score = 1.0 if timestamps == sorted_timestamps else 0.5

        return 0.3 * type_variety + 0.7 * ordering_score

    def detect_adversarial_patterns(self, segment: EffortSegment) -> List[str]:
        """Detect potential manipulation patterns."""
        patterns = []

        if segment.signal_count == 0:
            return patterns

        # Check for suspiciously regular timing
        if segment.signal_count >= 3:
            timestamps = sorted(s.timestamp for s in segment.signals)
            gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            if gaps and len(set(round(g, 2) for g in gaps)) == 1:
                patterns.append("perfectly_regular_timing")

        # Check for duplicate hashes
        hashes = [s.hash for s in segment.signals]
        if len(hashes) != len(set(hashes)):
            patterns.append("duplicate_signal_hashes")

        # Check for unrealistic duration
        if segment.duration < 1.0 and segment.signal_count > 10:
            patterns.append("unrealistic_signal_density")

        return patterns


class ReceiptBuilder:
    """
    Builder for creating EffortReceipts from various sources.
    """

    def __init__(
        self,
        observer: Optional[Observer] = None,
        validator: Optional[Validator] = None,
    ):
        self.observer = observer or DefaultObserver()
        self.validator = validator or DefaultValidator()

    def from_segment(
        self,
        segment: EffortSegment,
        prior_receipts: Optional[List[str]] = None,
    ) -> EffortReceipt:
        """Build a receipt from an effort segment."""
        # Validate the segment
        validation = self.validator.validate_segment(segment)
        summary = self.validator.generate_summary(segment)

        # Generate receipt ID
        receipt_id = hashlib.sha256(
            f"{segment.segment_id}:{segment.compute_segment_hash()}".encode()
        ).hexdigest()

        return EffortReceipt(
            receipt_id=receipt_id,
            time_bounds=(segment.start_time, segment.end_time),
            signal_hashes=segment.get_signal_hashes(),
            effort_summary=summary,
            validation_metadata=validation,
            observer_id=self.observer.get_observer_id(),
            validator_id=validation.validator_id,
            prior_receipts=prior_receipts or [],
            is_incomplete=len(validation.uncertainty_markers) > 2,
            suspected_manipulation=len(validation.adversarial_flags) > 0,
        )

    def from_ledger_entry(
        self,
        entry: "LedgerEntry",
        content: Optional[str] = None,
    ) -> EffortReceipt:
        """Build a receipt from a ValueLedger entry."""
        from .core import LedgerEntry

        # Create a segment from the entry
        segment_id = entry.id or hashlib.sha256(
            f"{entry.intent_id}:{entry.timestamp}".encode()
        ).hexdigest()

        segment = EffortSegment(
            segment_id=segment_id,
            start_time=entry.timestamp - 3600,  # Assume 1 hour if not specified
            end_time=entry.timestamp,
            segmentation_rule="ledger_entry",
        )

        # Add signals from content if available
        if content:
            signal = self.observer.capture_signal(content)
            segment.signals.append(signal)
        elif entry.proof.content_hash:
            # Create signal from hash if content not available
            segment.signals.append(EffortSignal(
                signal_type=SignalType.UNKNOWN,
                timestamp=entry.timestamp,
                hash=entry.proof.content_hash,
                modality="reconstructed",
            ))

        # Build receipt
        receipt = self.from_segment(segment)

        # Add ledger-specific anchoring
        receipt.merkle_ref = entry.proof.merkle_ref

        return receipt

    def verify_receipt(self, receipt: EffortReceipt) -> VerificationResult:
        """
        Verify a receipt per MP-02 §10.

        Third party verification:
        - Recompute receipt hashes
        - Inspect validation metadata
        - Confirm completeness

        Trust in Observer or Validator is NOT required.
        """
        checks_passed = []
        checks_failed = []
        warnings = []

        # Check 1: Receipt hash integrity
        computed_hash = receipt.compute_hash()
        if computed_hash == receipt.receipt_hash:
            checks_passed.append("hash_integrity")
        else:
            checks_failed.append("hash_integrity: computed hash does not match")

        # Check 2: Time bounds validity
        start, end = receipt.time_bounds
        if start < end:
            checks_passed.append("time_bounds_valid")
        else:
            checks_failed.append("time_bounds_invalid: start >= end")

        # Check 3: Signal hashes present
        if receipt.signal_hashes:
            checks_passed.append("signals_present")
        else:
            checks_failed.append("no_signal_hashes")

        # Check 4: Validation metadata complete
        vm = receipt.validation_metadata
        if vm.validator_id and vm.model_id:
            checks_passed.append("validation_metadata_complete")
        else:
            warnings.append("incomplete_validation_metadata")

        # Check 5: No manipulation flags
        if not receipt.suspected_manipulation:
            checks_passed.append("no_manipulation_detected")
        else:
            warnings.append("manipulation_suspected")

        # Check 6: Completeness
        if not receipt.is_incomplete:
            checks_passed.append("receipt_complete")
        else:
            warnings.append("receipt_marked_incomplete")

        return VerificationResult(
            valid=len(checks_failed) == 0,
            receipt_id=receipt.receipt_id,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
        )


def verify_third_party(receipt: EffortReceipt) -> VerificationResult:
    """
    Convenience function for third-party verification per MP-02 §10.

    Trust in Observer or Validator is NOT required.
    """
    builder = ReceiptBuilder()
    return builder.verify_receipt(receipt)
