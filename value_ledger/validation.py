# value_ledger/validation.py
"""
Enhanced Validation Criteria Implementation.

Per MP-02 §7, validators MAY assess:
- Linguistic coherence
- Conceptual progression
- Internal consistency
- Indicators of synthesis vs duplication

This module provides:
- Configurable validation criteria
- Enhanced coherence and progression scoring
- Consistency checking across signals
- Duplication and synthesis detection
- Confidence calculation and reporting
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Validation Criteria Configuration
# =============================================================================

class CriteriaType(str, Enum):
    """Types of validation criteria."""
    COHERENCE = "coherence"
    PROGRESSION = "progression"
    CONSISTENCY = "consistency"
    AUTHENTICITY = "authenticity"
    COMPLETENESS = "completeness"
    TEMPORAL = "temporal"


class SeverityLevel(str, Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationCriterion:
    """A single validation criterion with configurable thresholds."""
    name: str
    criterion_type: str
    description: str = ""
    weight: float = 1.0
    min_threshold: float = 0.0
    warning_threshold: float = 0.5
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationIssue:
    """A validation issue found during assessment."""
    issue_id: str
    criterion: str
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "issue_id": self.issue_id,
            "criterion": self.criterion,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class CriteriaConfig:
    """Configuration for validation criteria."""
    criteria: List[ValidationCriterion] = field(default_factory=list)
    min_signals: int = 2
    min_duration_seconds: float = 10.0
    max_gap_ratio: float = 5.0  # Max gap relative to average
    duplication_threshold: float = 0.8  # Similarity threshold for duplication
    confidence_weights: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "CriteriaConfig":
        """Create default criteria configuration."""
        return cls(
            criteria=[
                ValidationCriterion(
                    name="temporal_coherence",
                    criterion_type=CriteriaType.COHERENCE,
                    description="Signals are temporally distributed",
                    weight=1.0,
                    warning_threshold=0.4,
                ),
                ValidationCriterion(
                    name="signal_progression",
                    criterion_type=CriteriaType.PROGRESSION,
                    description="Signals show clear progression",
                    weight=1.0,
                    warning_threshold=0.5,
                ),
                ValidationCriterion(
                    name="internal_consistency",
                    criterion_type=CriteriaType.CONSISTENCY,
                    description="Signals are internally consistent",
                    weight=0.8,
                    warning_threshold=0.6,
                ),
                ValidationCriterion(
                    name="authenticity",
                    criterion_type=CriteriaType.AUTHENTICITY,
                    description="No signs of synthesis or duplication",
                    weight=1.2,
                    warning_threshold=0.7,
                ),
                ValidationCriterion(
                    name="completeness",
                    criterion_type=CriteriaType.COMPLETENESS,
                    description="Effort segment is complete",
                    weight=0.6,
                    warning_threshold=0.5,
                ),
            ],
            confidence_weights={
                "coherence": 0.25,
                "progression": 0.25,
                "consistency": 0.20,
                "authenticity": 0.20,
                "completeness": 0.10,
            },
        )


# =============================================================================
# Validation Scores
# =============================================================================

@dataclass
class CoherenceScore:
    """Detailed coherence assessment."""
    overall: float = 0.0
    temporal_distribution: float = 0.0
    gap_consistency: float = 0.0
    density_score: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass
class ProgressionScore:
    """Detailed progression assessment."""
    overall: float = 0.0
    temporal_flow: float = 0.0
    type_variety: float = 0.0
    complexity_growth: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass
class ConsistencyScore:
    """Detailed consistency assessment."""
    overall: float = 0.0
    hash_uniqueness: float = 0.0
    metadata_consistency: float = 0.0
    sequence_integrity: float = 0.0
    issues: List[str] = field(default_factory=list)


@dataclass
class AuthenticityScore:
    """Detailed authenticity assessment."""
    overall: float = 0.0
    duplication_score: float = 0.0  # Lower is better (less duplication)
    synthesis_score: float = 0.0  # Lower is better (less synthetic)
    natural_variance: float = 0.0  # Higher is better
    adversarial_patterns: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@dataclass
class CompletenessScore:
    """Detailed completeness assessment."""
    overall: float = 0.0
    signal_sufficiency: float = 0.0
    duration_adequacy: float = 0.0
    coverage: float = 0.0
    gaps: List[Tuple[float, float]] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


# =============================================================================
# Validation Report
# =============================================================================

@dataclass
class ValidationReport:
    """
    Comprehensive validation report.

    Per MP-02 §7: Validators must preserve dissent and uncertainty.
    """
    report_id: str = field(default_factory=lambda: hashlib.sha256(str(time.time()).encode()).hexdigest()[:16])
    segment_id: str = ""
    validator_id: str = ""
    timestamp: float = field(default_factory=time.time)

    # Scores
    coherence: CoherenceScore = field(default_factory=CoherenceScore)
    progression: ProgressionScore = field(default_factory=ProgressionScore)
    consistency: ConsistencyScore = field(default_factory=ConsistencyScore)
    authenticity: AuthenticityScore = field(default_factory=AuthenticityScore)
    completeness: CompletenessScore = field(default_factory=CompletenessScore)

    # Overall assessment
    overall_score: float = 0.0
    confidence: float = 0.0
    issues: List[ValidationIssue] = field(default_factory=list)
    uncertainty_markers: List[str] = field(default_factory=list)

    # Verdict
    valid: bool = False
    verdict_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "report_id": self.report_id,
            "segment_id": self.segment_id,
            "validator_id": self.validator_id,
            "timestamp": self.timestamp,
            "scores": {
                "coherence": {
                    "overall": self.coherence.overall,
                    "temporal_distribution": self.coherence.temporal_distribution,
                    "gap_consistency": self.coherence.gap_consistency,
                    "density_score": self.coherence.density_score,
                },
                "progression": {
                    "overall": self.progression.overall,
                    "temporal_flow": self.progression.temporal_flow,
                    "type_variety": self.progression.type_variety,
                    "complexity_growth": self.progression.complexity_growth,
                },
                "consistency": {
                    "overall": self.consistency.overall,
                    "hash_uniqueness": self.consistency.hash_uniqueness,
                    "metadata_consistency": self.consistency.metadata_consistency,
                    "sequence_integrity": self.consistency.sequence_integrity,
                },
                "authenticity": {
                    "overall": self.authenticity.overall,
                    "duplication_score": self.authenticity.duplication_score,
                    "synthesis_score": self.authenticity.synthesis_score,
                    "natural_variance": self.authenticity.natural_variance,
                    "adversarial_patterns": self.authenticity.adversarial_patterns,
                },
                "completeness": {
                    "overall": self.completeness.overall,
                    "signal_sufficiency": self.completeness.signal_sufficiency,
                    "duration_adequacy": self.completeness.duration_adequacy,
                    "coverage": self.completeness.coverage,
                },
            },
            "overall_score": self.overall_score,
            "confidence": self.confidence,
            "issues": [i.to_dict() for i in self.issues],
            "uncertainty_markers": self.uncertainty_markers,
            "valid": self.valid,
            "verdict_reason": self.verdict_reason,
        }

    def add_issue(
        self,
        criterion: str,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a validation issue."""
        issue = ValidationIssue(
            issue_id=f"{self.report_id}-{len(self.issues)}",
            criterion=criterion,
            severity=severity,
            message=message,
            details=details or {},
        )
        self.issues.append(issue)

        if severity in [SeverityLevel.ERROR, SeverityLevel.CRITICAL]:
            self.uncertainty_markers.append(f"{criterion}:{message}")


# =============================================================================
# Enhanced Validator
# =============================================================================

class EnhancedValidator:
    """
    Enhanced validator with configurable criteria.

    Per MP-02 §7:
    - Validators MUST produce deterministic summaries
    - Validators MUST disclose model identity and version
    - Validators MUST preserve dissent and uncertainty
    - Validators MUST NOT declare effort as valuable
    - Validators MUST NOT assert originality or ownership
    - Validators MUST NOT collapse ambiguous signals into certainty
    """

    def __init__(
        self,
        validator_id: str = "enhanced_validator",
        model_id: str = "enhanced_heuristic",
        model_version: str = "2.0.0",
        config: Optional[CriteriaConfig] = None,
    ):
        self.validator_id = validator_id
        self.model_id = model_id
        self.model_version = model_version
        self.config = config or CriteriaConfig.default()

    def validate(self, segment: "EffortSegment") -> ValidationReport:
        """
        Perform comprehensive validation of an effort segment.

        Args:
            segment: The effort segment to validate

        Returns:
            ValidationReport with detailed scores and issues
        """
        report = ValidationReport(
            segment_id=segment.segment_id,
            validator_id=self.validator_id,
        )

        # Assess each dimension
        report.coherence = self._assess_coherence(segment)
        report.progression = self._assess_progression(segment)
        report.consistency = self._assess_consistency(segment)
        report.authenticity = self._assess_authenticity(segment)
        report.completeness = self._assess_completeness(segment)

        # Calculate overall score and confidence
        report.overall_score = self._calculate_overall_score(report)
        report.confidence = self._calculate_confidence(report, segment)

        # Collect issues from all scores
        self._collect_issues(report)

        # Determine validity
        report.valid = self._determine_validity(report)
        report.verdict_reason = self._generate_verdict_reason(report)

        logger.info(
            f"Validated segment {segment.segment_id}: "
            f"score={report.overall_score:.2f}, confidence={report.confidence:.2f}, "
            f"valid={report.valid}"
        )

        return report

    def _assess_coherence(self, segment: "EffortSegment") -> CoherenceScore:
        """Assess temporal and structural coherence."""
        score = CoherenceScore()

        if segment.signal_count == 0:
            score.issues.append("No signals in segment")
            return score

        # Temporal distribution
        if segment.signal_count >= 2:
            timestamps = sorted(s.timestamp for s in segment.signals)
            gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                max_gap = max(gaps)
                min_gap = min(gaps)

                # Check gap distribution
                if avg_gap > 0:
                    gap_variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)
                    gap_std = math.sqrt(gap_variance)
                    cv = gap_std / avg_gap if avg_gap > 0 else 0

                    # Lower coefficient of variation = more consistent gaps
                    score.gap_consistency = max(0.0, 1.0 - min(cv, 2.0) / 2.0)

                    # Check for excessive gaps
                    if max_gap > avg_gap * self.config.max_gap_ratio:
                        score.issues.append(f"Large gap detected: {max_gap:.1f}s vs avg {avg_gap:.1f}s")
                else:
                    score.gap_consistency = 0.5

                # Temporal distribution score
                expected_span = segment.duration / segment.signal_count
                actual_span = max_gap if gaps else 0
                score.temporal_distribution = max(0.0, 1.0 - abs(actual_span - expected_span) / (expected_span + 1))
        else:
            score.temporal_distribution = 0.5
            score.gap_consistency = 0.5

        # Density score
        if segment.duration > 0:
            density = segment.signal_count / segment.duration
            # Optimal density: 0.1-1.0 signals per second
            if density < 0.01:
                score.density_score = 0.3
                score.issues.append("Very low signal density")
            elif density > 10:
                score.density_score = 0.3
                score.issues.append("Suspiciously high signal density")
            else:
                score.density_score = min(1.0, 0.5 + density)
        else:
            score.density_score = 0.5

        # Overall coherence
        score.overall = (
            score.temporal_distribution * 0.4 +
            score.gap_consistency * 0.4 +
            score.density_score * 0.2
        )

        return score

    def _assess_progression(self, segment: "EffortSegment") -> ProgressionScore:
        """Assess effort progression and flow."""
        score = ProgressionScore()

        if segment.signal_count == 0:
            score.issues.append("No signals to assess progression")
            return score

        # Temporal flow - signals should be in order
        timestamps = [s.timestamp for s in segment.signals]
        sorted_timestamps = sorted(timestamps)
        if timestamps == sorted_timestamps:
            score.temporal_flow = 1.0
        else:
            # Calculate how out-of-order
            inversions = sum(1 for i in range(len(timestamps)-1) if timestamps[i] > timestamps[i+1])
            score.temporal_flow = max(0.0, 1.0 - inversions / max(len(timestamps)-1, 1))
            if inversions > 0:
                score.issues.append(f"{inversions} out-of-order signals detected")

        # Type variety
        signal_types = set(s.signal_type for s in segment.signals)
        score.type_variety = min(1.0, len(signal_types) / 3)

        # Complexity growth - check if later signals are more complex
        # (Using hash entropy as a proxy for complexity)
        if segment.signal_count >= 3:
            hashes = [s.hash for s in sorted(segment.signals, key=lambda x: x.timestamp)]
            early_entropy = self._hash_entropy(hashes[:len(hashes)//2])
            late_entropy = self._hash_entropy(hashes[len(hashes)//2:])

            # Growth is good, but stability is also acceptable
            if late_entropy >= early_entropy:
                score.complexity_growth = 0.8 + 0.2 * min(1.0, (late_entropy - early_entropy) / early_entropy if early_entropy > 0 else 0)
            else:
                score.complexity_growth = 0.6
        else:
            score.complexity_growth = 0.5

        # Overall progression
        score.overall = (
            score.temporal_flow * 0.5 +
            score.type_variety * 0.3 +
            score.complexity_growth * 0.2
        )

        return score

    def _assess_consistency(self, segment: "EffortSegment") -> ConsistencyScore:
        """Assess internal consistency of signals."""
        score = ConsistencyScore()

        if segment.signal_count == 0:
            score.issues.append("No signals to assess consistency")
            return score

        # Hash uniqueness
        hashes = [s.hash for s in segment.signals]
        unique_hashes = set(hashes)
        score.hash_uniqueness = len(unique_hashes) / len(hashes) if hashes else 0.0

        if score.hash_uniqueness < 1.0:
            duplicates = len(hashes) - len(unique_hashes)
            score.issues.append(f"{duplicates} duplicate signal hashes")

        # Metadata consistency
        modalities = [s.modality for s in segment.signals]
        modality_counts = Counter(modalities)
        dominant_modality_ratio = max(modality_counts.values()) / len(modalities) if modalities else 0

        # Some variety in modality is expected, but too much is suspicious
        if 0.3 <= dominant_modality_ratio <= 0.9:
            score.metadata_consistency = 0.9
        else:
            score.metadata_consistency = 0.6
            if dominant_modality_ratio > 0.95:
                score.issues.append("All signals from same modality")

        # Sequence integrity
        sequences = [s.sequence_number for s in segment.signals]
        expected_sequences = list(range(len(sequences)))
        if sorted(sequences) == expected_sequences:
            score.sequence_integrity = 1.0
        else:
            # Check for gaps or duplicates
            unique_sequences = set(sequences)
            if len(unique_sequences) == len(sequences):
                score.sequence_integrity = 0.8
            else:
                score.sequence_integrity = 0.5
                score.issues.append("Sequence number inconsistencies")

        # Overall consistency
        score.overall = (
            score.hash_uniqueness * 0.5 +
            score.metadata_consistency * 0.25 +
            score.sequence_integrity * 0.25
        )

        return score

    def _assess_authenticity(self, segment: "EffortSegment") -> AuthenticityScore:
        """
        Assess authenticity - detect synthesis and duplication.

        Per MP-02 §7: Detect indicators of synthesis vs duplication.
        """
        score = AuthenticityScore()

        if segment.signal_count == 0:
            score.issues.append("No signals to assess authenticity")
            return score

        # Duplication detection
        hashes = [s.hash for s in segment.signals]
        unique_ratio = len(set(hashes)) / len(hashes)
        score.duplication_score = 1.0 - unique_ratio  # Lower is better

        if score.duplication_score > self.config.duplication_threshold:
            score.issues.append("High duplication detected")
            score.adversarial_patterns.append("excessive_duplication")

        # Synthesis detection - check for perfectly regular patterns
        if segment.signal_count >= 3:
            timestamps = sorted(s.timestamp for s in segment.signals)
            gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

            if gaps:
                # Check for perfectly regular timing
                unique_gaps = set(round(g, 2) for g in gaps)
                if len(unique_gaps) == 1 and len(gaps) > 2:
                    score.synthesis_score = 0.8  # High synthesis suspicion
                    score.adversarial_patterns.append("perfectly_regular_timing")
                else:
                    # Check for mechanical patterns
                    gap_variance = sum((g - sum(gaps)/len(gaps)) ** 2 for g in gaps) / len(gaps)
                    if gap_variance < 0.01:  # Very low variance
                        score.synthesis_score = 0.5
                        score.adversarial_patterns.append("low_timing_variance")
                    else:
                        score.synthesis_score = 0.1  # Low synthesis suspicion

        # Natural variance check
        if segment.signal_count >= 2:
            timestamps = [s.timestamp for s in segment.signals]
            if len(timestamps) >= 2:
                variance = sum((t - sum(timestamps)/len(timestamps)) ** 2 for t in timestamps) / len(timestamps)
                # Higher variance suggests more natural behavior
                score.natural_variance = min(1.0, math.sqrt(variance) / 100)
            else:
                score.natural_variance = 0.5
        else:
            score.natural_variance = 0.5

        # Additional adversarial patterns
        # Check for unrealistic burst
        if segment.duration < 1.0 and segment.signal_count > 10:
            score.adversarial_patterns.append("unrealistic_signal_burst")
            score.issues.append("Unrealistic signal burst detected")

        # Check for future timestamps
        now = time.time()
        future_signals = [s for s in segment.signals if s.timestamp > now + 60]
        if future_signals:
            score.adversarial_patterns.append("future_timestamps")
            score.issues.append(f"{len(future_signals)} signals with future timestamps")

        # Overall authenticity (higher is better)
        score.overall = (
            (1.0 - score.duplication_score) * 0.3 +
            (1.0 - score.synthesis_score) * 0.4 +
            score.natural_variance * 0.3
        )

        return score

    def _assess_completeness(self, segment: "EffortSegment") -> CompletenessScore:
        """Assess completeness of the effort segment."""
        score = CompletenessScore()

        # Signal sufficiency
        if segment.signal_count >= self.config.min_signals:
            score.signal_sufficiency = min(1.0, segment.signal_count / (self.config.min_signals * 2))
        else:
            score.signal_sufficiency = segment.signal_count / self.config.min_signals
            score.issues.append(f"Insufficient signals: {segment.signal_count} < {self.config.min_signals}")

        # Duration adequacy
        if segment.duration >= self.config.min_duration_seconds:
            score.duration_adequacy = min(1.0, segment.duration / (self.config.min_duration_seconds * 3))
        else:
            score.duration_adequacy = segment.duration / self.config.min_duration_seconds
            score.issues.append(f"Short duration: {segment.duration:.1f}s < {self.config.min_duration_seconds}s")

        # Coverage - check for gaps in observation
        if segment.signal_count >= 2:
            timestamps = sorted(s.timestamp for s in segment.signals)
            gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

            if gaps:
                avg_gap = sum(gaps) / len(gaps)
                large_gaps = [(timestamps[i], timestamps[i+1]) for i, g in enumerate(gaps) if g > avg_gap * 3]
                score.gaps = large_gaps

                if large_gaps:
                    gap_time = sum(g[1] - g[0] for g in large_gaps)
                    score.coverage = max(0.0, 1.0 - gap_time / segment.duration)
                    score.issues.append(f"{len(large_gaps)} observation gaps totaling {gap_time:.1f}s")
                else:
                    score.coverage = 1.0
            else:
                score.coverage = 1.0
        else:
            score.coverage = 0.5

        # Overall completeness
        score.overall = (
            score.signal_sufficiency * 0.4 +
            score.duration_adequacy * 0.3 +
            score.coverage * 0.3
        )

        return score

    def _calculate_overall_score(self, report: ValidationReport) -> float:
        """Calculate weighted overall score."""
        weights = self.config.confidence_weights

        return (
            report.coherence.overall * weights.get("coherence", 0.25) +
            report.progression.overall * weights.get("progression", 0.25) +
            report.consistency.overall * weights.get("consistency", 0.20) +
            report.authenticity.overall * weights.get("authenticity", 0.20) +
            report.completeness.overall * weights.get("completeness", 0.10)
        )

    def _calculate_confidence(self, report: ValidationReport, segment: "EffortSegment") -> float:
        """
        Calculate confidence in the validation assessment.

        Per MP-02 §5: Partial observability - uncertainty is preserved.
        """
        # Base confidence from signal count
        signal_confidence = min(1.0, segment.signal_count / 10)

        # Duration confidence
        duration_confidence = min(1.0, segment.duration / 300)  # 5 minutes max confidence

        # Issue penalty
        issue_count = len(report.issues)
        issue_penalty = max(0.0, 1.0 - issue_count * 0.1)

        # Adversarial pattern penalty
        adversarial_penalty = max(0.0, 1.0 - len(report.authenticity.adversarial_patterns) * 0.2)

        return (
            signal_confidence * 0.3 +
            duration_confidence * 0.2 +
            issue_penalty * 0.3 +
            adversarial_penalty * 0.2
        )

    def _collect_issues(self, report: ValidationReport) -> None:
        """Collect all issues into the report."""
        # Coherence issues
        for issue in report.coherence.issues:
            report.add_issue(
                CriteriaType.COHERENCE,
                SeverityLevel.WARNING if report.coherence.overall > 0.3 else SeverityLevel.ERROR,
                issue,
            )

        # Progression issues
        for issue in report.progression.issues:
            report.add_issue(
                CriteriaType.PROGRESSION,
                SeverityLevel.WARNING,
                issue,
            )

        # Consistency issues
        for issue in report.consistency.issues:
            report.add_issue(
                CriteriaType.CONSISTENCY,
                SeverityLevel.ERROR if "duplicate" in issue.lower() else SeverityLevel.WARNING,
                issue,
            )

        # Authenticity issues
        for issue in report.authenticity.issues:
            report.add_issue(
                CriteriaType.AUTHENTICITY,
                SeverityLevel.CRITICAL if "future" in issue.lower() else SeverityLevel.ERROR,
                issue,
            )

        # Completeness issues
        for issue in report.completeness.issues:
            report.add_issue(
                CriteriaType.COMPLETENESS,
                SeverityLevel.INFO if "gap" in issue.lower() else SeverityLevel.WARNING,
                issue,
            )

    def _determine_validity(self, report: ValidationReport) -> bool:
        """
        Determine if the segment is valid.

        Per MP-02 §7: Validators MUST NOT collapse ambiguous signals into certainty.
        """
        # Critical issues invalidate
        critical_issues = [i for i in report.issues if i.severity == SeverityLevel.CRITICAL]
        if critical_issues:
            return False

        # Too many errors invalidate
        error_issues = [i for i in report.issues if i.severity == SeverityLevel.ERROR]
        if len(error_issues) > 3:
            return False

        # Low overall score invalidates
        if report.overall_score < 0.3:
            return False

        # Low confidence with low score
        if report.confidence < 0.4 and report.overall_score < 0.5:
            return False

        return True

    def _generate_verdict_reason(self, report: ValidationReport) -> str:
        """Generate human-readable verdict reason."""
        if report.valid:
            return (
                f"Segment validated with score {report.overall_score:.2f} "
                f"and confidence {report.confidence:.2f}. "
                f"{len(report.issues)} issues noted."
            )
        else:
            critical = [i for i in report.issues if i.severity == SeverityLevel.CRITICAL]
            errors = [i for i in report.issues if i.severity == SeverityLevel.ERROR]

            reasons = []
            if critical:
                reasons.append(f"{len(critical)} critical issues")
            if errors:
                reasons.append(f"{len(errors)} errors")
            if report.overall_score < 0.3:
                reasons.append(f"low score ({report.overall_score:.2f})")
            if report.confidence < 0.4:
                reasons.append(f"low confidence ({report.confidence:.2f})")

            return f"Segment invalid: {', '.join(reasons)}"

    def _hash_entropy(self, hashes: List[str]) -> float:
        """Calculate entropy of hash characters as complexity proxy."""
        if not hashes:
            return 0.0

        all_chars = ''.join(hashes)
        char_counts = Counter(all_chars)
        total = len(all_chars)

        entropy = 0.0
        for count in char_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        return entropy


# =============================================================================
# Specialized Checkers
# =============================================================================

class ConsistencyChecker:
    """
    Specialized checker for cross-signal consistency.
    """

    def check_temporal_consistency(self, signals: List["EffortSignal"]) -> Tuple[bool, List[str]]:
        """Check that signal timestamps are consistent."""
        issues = []

        if len(signals) < 2:
            return (True, issues)

        timestamps = [s.timestamp for s in signals]

        # Check for future timestamps
        now = time.time()
        future = [t for t in timestamps if t > now + 60]
        if future:
            issues.append(f"{len(future)} signals have future timestamps")

        # Check for negative time progression
        for i in range(len(signals) - 1):
            if signals[i+1].timestamp < signals[i].timestamp - 1:  # 1 second tolerance
                issues.append(f"Time regression between signals {i} and {i+1}")

        return (len(issues) == 0, issues)

    def check_sequence_consistency(self, signals: List["EffortSignal"]) -> Tuple[bool, List[str]]:
        """Check that sequence numbers are consistent."""
        issues = []

        if len(signals) < 2:
            return (True, issues)

        sequences = [s.sequence_number for s in signals]

        # Check for duplicates
        if len(sequences) != len(set(sequences)):
            issues.append("Duplicate sequence numbers detected")

        # Check for gaps
        sorted_seqs = sorted(sequences)
        for i in range(len(sorted_seqs) - 1):
            if sorted_seqs[i+1] - sorted_seqs[i] > 1:
                issues.append(f"Gap in sequence numbers: {sorted_seqs[i]} to {sorted_seqs[i+1]}")

        return (len(issues) == 0, issues)


class DuplicationDetector:
    """
    Detector for duplicated or synthesized content.
    """

    def __init__(self, similarity_threshold: float = 0.8):
        self.similarity_threshold = similarity_threshold

    def detect_hash_duplication(self, signals: List["EffortSignal"]) -> List[Tuple[int, int]]:
        """Detect signals with identical hashes."""
        duplicates = []
        seen = {}

        for i, signal in enumerate(signals):
            if signal.hash in seen:
                duplicates.append((seen[signal.hash], i))
            else:
                seen[signal.hash] = i

        return duplicates

    def detect_timing_patterns(self, signals: List["EffortSignal"]) -> List[str]:
        """Detect suspicious timing patterns."""
        patterns = []

        if len(signals) < 3:
            return patterns

        timestamps = sorted(s.timestamp for s in signals)
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

        # Perfectly regular timing
        unique_gaps = set(round(g, 3) for g in gaps)
        if len(unique_gaps) == 1:
            patterns.append("perfectly_regular")

        # Exact multiples
        if len(gaps) >= 3:
            min_gap = min(gaps)
            if min_gap > 0:
                ratios = [g / min_gap for g in gaps]
                if all(abs(r - round(r)) < 0.01 for r in ratios):
                    patterns.append("exact_multiples")

        return patterns


class ConfidenceCalculator:
    """
    Calculator for validation confidence scores.
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or {
            "signal_count": 0.25,
            "duration": 0.20,
            "consistency": 0.25,
            "authenticity": 0.30,
        }

    def calculate(
        self,
        signal_count: int,
        duration: float,
        consistency_score: float,
        authenticity_score: float,
        issue_count: int = 0,
    ) -> float:
        """Calculate overall confidence score."""
        # Normalize inputs
        signal_factor = min(1.0, signal_count / 10)
        duration_factor = min(1.0, duration / 300)
        issue_factor = max(0.0, 1.0 - issue_count * 0.1)

        base_confidence = (
            signal_factor * self.weights["signal_count"] +
            duration_factor * self.weights["duration"] +
            consistency_score * self.weights["consistency"] +
            authenticity_score * self.weights["authenticity"]
        )

        return base_confidence * issue_factor


# =============================================================================
# Convenience Functions
# =============================================================================

def create_enhanced_validator(
    validator_id: str = "enhanced_validator",
    config: Optional[CriteriaConfig] = None,
) -> EnhancedValidator:
    """
    Create an enhanced validator with default or custom configuration.

    Args:
        validator_id: Unique validator identifier
        config: Optional custom criteria configuration

    Returns:
        Configured EnhancedValidator
    """
    return EnhancedValidator(
        validator_id=validator_id,
        config=config or CriteriaConfig.default(),
    )


def validate_segment(
    segment: "EffortSegment",
    config: Optional[CriteriaConfig] = None,
) -> ValidationReport:
    """
    Convenience function to validate a segment with default validator.

    Args:
        segment: Effort segment to validate
        config: Optional custom configuration

    Returns:
        ValidationReport with detailed assessment
    """
    validator = create_enhanced_validator(config=config)
    return validator.validate(segment)
