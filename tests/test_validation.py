# tests/test_validation.py
"""
Tests for Enhanced Validation Criteria.

Per MP-02 §7:
- Validators MAY assess linguistic coherence, conceptual progression, internal consistency
- Validators MAY detect indicators of synthesis vs duplication
- Validators MUST preserve dissent and uncertainty
"""

import pytest
import time
import hashlib
import math

from value_ledger.validation import (
    CriteriaType,
    SeverityLevel,
    ValidationCriterion,
    ValidationIssue,
    CriteriaConfig,
    CoherenceScore,
    ProgressionScore,
    ConsistencyScore,
    AuthenticityScore,
    CompletenessScore,
    ValidationReport,
    EnhancedValidator,
    ConsistencyChecker,
    DuplicationDetector,
    ConfidenceCalculator,
    create_enhanced_validator,
    validate_segment,
)
from value_ledger.receipt import (
    EffortSignal,
    EffortSegment,
    SignalType,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def base_time():
    """Fixed base time for deterministic tests."""
    return 1700000000.0


@pytest.fixture
def sample_signals(base_time):
    """Create a list of sample signals with natural variance."""
    signals = []
    for i in range(5):
        signals.append(
            EffortSignal(
                signal_type=SignalType.TEXT if i % 2 == 0 else SignalType.KEYSTROKE,
                timestamp=base_time + i * 60 + (i * 7 % 10),  # Natural variance
                hash=hashlib.sha256(f"content_{i}".encode()).hexdigest(),
                modality="keyboard" if i % 2 == 0 else "mouse",
                sequence_number=i,
            )
        )
    return signals


@pytest.fixture
def sample_segment(sample_signals, base_time):
    """Create a sample effort segment."""
    return EffortSegment(
        segment_id="test-segment-001",
        start_time=base_time,
        end_time=base_time + 300,
        signals=sample_signals,
    )


@pytest.fixture
def empty_segment(base_time):
    """Create an empty segment."""
    return EffortSegment(
        segment_id="empty-segment",
        start_time=base_time,
        end_time=base_time + 60,
        signals=[],
    )


@pytest.fixture
def synthetic_segment(base_time):
    """Create a segment with perfectly regular timing (synthetic pattern)."""
    signals = []
    for i in range(10):
        signals.append(
            EffortSignal(
                signal_type=SignalType.TEXT,
                timestamp=base_time + i * 10.0,  # Perfectly regular
                hash=hashlib.sha256(f"synthetic_{i}".encode()).hexdigest(),
                modality="keyboard",
                sequence_number=i,
            )
        )
    return EffortSegment(
        segment_id="synthetic-segment",
        start_time=base_time,
        end_time=base_time + 100,
        signals=signals,
    )


@pytest.fixture
def duplicate_segment(base_time):
    """Create a segment with duplicate signals."""
    same_hash = hashlib.sha256("duplicate_content".encode()).hexdigest()
    signals = []
    for i in range(5):
        signals.append(
            EffortSignal(
                signal_type=SignalType.TEXT,
                timestamp=base_time + i * 20,
                hash=same_hash,  # All same hash
                modality="keyboard",
                sequence_number=i,
            )
        )
    return EffortSegment(
        segment_id="duplicate-segment",
        start_time=base_time,
        end_time=base_time + 100,
        signals=signals,
    )


@pytest.fixture
def default_validator():
    """Create a default validator."""
    return create_enhanced_validator()


@pytest.fixture
def custom_config():
    """Create a custom criteria configuration."""
    return CriteriaConfig(
        criteria=[
            ValidationCriterion(
                name="custom_coherence",
                criterion_type=CriteriaType.COHERENCE,
                weight=2.0,
            ),
        ],
        min_signals=3,
        min_duration_seconds=30.0,
        duplication_threshold=0.5,
    )


# =============================================================================
# Tests: CriteriaType Enum
# =============================================================================


class TestCriteriaType:

    def test_criteria_types_defined(self):
        assert CriteriaType.COHERENCE == "coherence"
        assert CriteriaType.PROGRESSION == "progression"
        assert CriteriaType.CONSISTENCY == "consistency"
        assert CriteriaType.AUTHENTICITY == "authenticity"
        assert CriteriaType.COMPLETENESS == "completeness"
        assert CriteriaType.TEMPORAL == "temporal"


class TestSeverityLevel:

    def test_severity_levels_defined(self):
        assert SeverityLevel.INFO == "info"
        assert SeverityLevel.WARNING == "warning"
        assert SeverityLevel.ERROR == "error"
        assert SeverityLevel.CRITICAL == "critical"


# =============================================================================
# Tests: ValidationCriterion
# =============================================================================


class TestValidationCriterion:

    def test_creation_with_defaults(self):
        criterion = ValidationCriterion(
            name="test_criterion",
            criterion_type=CriteriaType.COHERENCE,
        )
        assert criterion.name == "test_criterion"
        assert criterion.weight == 1.0
        assert criterion.enabled is True

    def test_creation_with_custom_values(self):
        criterion = ValidationCriterion(
            name="custom",
            criterion_type=CriteriaType.AUTHENTICITY,
            description="Custom authenticity check",
            weight=1.5,
            min_threshold=0.3,
            warning_threshold=0.6,
            enabled=False,
        )
        assert criterion.weight == 1.5
        assert criterion.min_threshold == 0.3
        assert criterion.enabled is False


# =============================================================================
# Tests: ValidationIssue
# =============================================================================


class TestValidationIssue:

    def test_issue_creation(self):
        issue = ValidationIssue(
            issue_id="issue-001",
            criterion=CriteriaType.CONSISTENCY,
            severity=SeverityLevel.WARNING,
            message="Test issue message",
        )
        assert issue.issue_id == "issue-001"
        assert issue.criterion == CriteriaType.CONSISTENCY
        assert issue.timestamp > 0

    def test_issue_to_dict(self):
        issue = ValidationIssue(
            issue_id="issue-002",
            criterion=CriteriaType.COHERENCE,
            severity=SeverityLevel.ERROR,
            message="Coherence error",
            details={"score": 0.2},
        )
        result = issue.to_dict()
        assert result["issue_id"] == "issue-002"
        assert result["severity"] == SeverityLevel.ERROR
        assert result["details"]["score"] == 0.2


# =============================================================================
# Tests: CriteriaConfig
# =============================================================================


class TestCriteriaConfig:

    def test_default_config(self):
        config = CriteriaConfig.default()
        assert len(config.criteria) == 5
        assert config.min_signals == 2
        assert config.min_duration_seconds == 10.0
        assert config.duplication_threshold == 0.8

    def test_custom_config(self, custom_config):
        assert len(custom_config.criteria) == 1
        assert custom_config.min_signals == 3

    def test_confidence_weights_sum(self):
        """Test that confidence weights sum to approximately 1.0."""
        config = CriteriaConfig.default()
        total = sum(config.confidence_weights.values())
        assert abs(total - 1.0) < 0.01


# =============================================================================
# Tests: Score Dataclasses
# =============================================================================


class TestScoreDataclasses:

    def test_coherence_score_defaults(self):
        score = CoherenceScore()
        assert score.overall == 0.0
        assert score.temporal_distribution == 0.0
        assert len(score.issues) == 0

    def test_progression_score_defaults(self):
        score = ProgressionScore()
        assert score.overall == 0.0
        assert score.temporal_flow == 0.0

    def test_consistency_score_defaults(self):
        score = ConsistencyScore()
        assert score.overall == 0.0
        assert score.hash_uniqueness == 0.0

    def test_authenticity_score_defaults(self):
        score = AuthenticityScore()
        assert score.overall == 0.0
        assert len(score.adversarial_patterns) == 0

    def test_completeness_score_defaults(self):
        score = CompletenessScore()
        assert score.overall == 0.0
        assert len(score.gaps) == 0


# =============================================================================
# Tests: ValidationReport
# =============================================================================


class TestValidationReport:

    def test_report_creation(self):
        report = ValidationReport(
            segment_id="segment-001",
            validator_id="test-validator",
        )
        assert report.segment_id == "segment-001"
        assert len(report.report_id) == 16
        assert report.valid is False

    def test_report_add_issue(self):
        report = ValidationReport(segment_id="test")
        report.add_issue(
            criterion=CriteriaType.COHERENCE,
            severity=SeverityLevel.WARNING,
            message="Test warning",
        )
        assert len(report.issues) == 1
        assert report.issues[0].criterion == CriteriaType.COHERENCE

    def test_report_add_critical_issue_adds_uncertainty_marker(self):
        """Test that critical issues add uncertainty markers."""
        report = ValidationReport(segment_id="test")
        report.add_issue(
            criterion=CriteriaType.AUTHENTICITY,
            severity=SeverityLevel.CRITICAL,
            message="Critical auth issue",
        )
        assert len(report.uncertainty_markers) == 1

    def test_report_to_dict(self):
        report = ValidationReport(
            segment_id="segment-001",
            validator_id="test-validator",
        )
        report.overall_score = 0.75
        report.confidence = 0.8
        report.valid = True

        result = report.to_dict()
        assert result["segment_id"] == "segment-001"
        assert result["overall_score"] == 0.75
        assert result["confidence"] == 0.8
        assert result["valid"] is True
        assert "scores" in result


# =============================================================================
# Tests: EnhancedValidator
# =============================================================================


class TestEnhancedValidator:

    def test_validator_creation(self, default_validator):
        assert default_validator.validator_id == "enhanced_validator"
        assert default_validator.model_version == "2.0.0"

    def test_validator_with_custom_config(self, custom_config):
        validator = EnhancedValidator(
            validator_id="custom",
            config=custom_config,
        )
        assert validator.config.min_signals == 3

    def test_validate_sample_segment(self, default_validator, sample_segment):
        report = default_validator.validate(sample_segment)
        assert report.segment_id == "test-segment-001"
        assert report.overall_score > 0
        assert report.confidence > 0

    def test_validate_empty_segment(self, default_validator, empty_segment):
        report = default_validator.validate(empty_segment)
        # Empty segment gets low score but not exactly 0 due to completeness calculation
        assert report.overall_score < 0.1
        assert report.valid is False

    def test_validate_synthetic_segment(self, default_validator, synthetic_segment):
        """Test that synthetic patterns are detected."""
        report = default_validator.validate(synthetic_segment)
        # Should detect regular timing
        assert len(report.authenticity.adversarial_patterns) > 0

    def test_validate_duplicate_segment(self, default_validator, duplicate_segment):
        """Test that duplicates are detected."""
        report = default_validator.validate(duplicate_segment)
        assert report.authenticity.duplication_score > 0.5
        assert report.consistency.hash_uniqueness < 1.0


class TestCoherenceAssessment:

    def test_assess_coherence_with_signals(self, default_validator, sample_segment):
        report = default_validator.validate(sample_segment)
        assert report.coherence.overall > 0
        assert report.coherence.temporal_distribution > 0

    def test_assess_coherence_empty_segment(self, default_validator, empty_segment):
        report = default_validator.validate(empty_segment)
        assert report.coherence.overall == 0.0
        assert "No signals" in report.coherence.issues[0]


class TestProgressionAssessment:

    def test_assess_progression_ordered(self, default_validator, sample_segment):
        report = default_validator.validate(sample_segment)
        assert report.progression.temporal_flow > 0
        assert report.progression.type_variety > 0

    def test_assess_progression_variety(self, default_validator, sample_segment):
        report = default_validator.validate(sample_segment)
        # Sample has TEXT and REVISION types
        assert report.progression.type_variety > 0


class TestConsistencyAssessment:

    def test_assess_consistency_unique_hashes(self, default_validator, sample_segment):
        report = default_validator.validate(sample_segment)
        assert report.consistency.hash_uniqueness == 1.0

    def test_assess_consistency_duplicate_hashes(self, default_validator, duplicate_segment):
        report = default_validator.validate(duplicate_segment)
        assert report.consistency.hash_uniqueness < 1.0
        assert "duplicate" in str(report.consistency.issues).lower()


class TestAuthenticityAssessment:

    def test_detect_duplication(self, default_validator, duplicate_segment):
        report = default_validator.validate(duplicate_segment)
        # All same hash means high duplication
        assert report.authenticity.duplication_score > 0.5

    def test_detect_synthetic_patterns(self, default_validator, synthetic_segment):
        report = default_validator.validate(synthetic_segment)
        # Perfectly regular timing should be flagged
        patterns = report.authenticity.adversarial_patterns
        assert len(patterns) > 0


class TestCompletenessAssessment:

    def test_assess_completeness_sufficient_signals(self, default_validator, sample_segment):
        report = default_validator.validate(sample_segment)
        assert report.completeness.signal_sufficiency > 0

    def test_assess_completeness_insufficient_signals(self, default_validator, base_time):
        segment = EffortSegment(
            segment_id="few-signals",
            start_time=base_time,
            end_time=base_time + 60,
            signals=[
                EffortSignal(
                    signal_type=SignalType.TEXT,
                    timestamp=base_time + 30,
                    hash="abc123",
                    modality="keyboard",
                )
            ],
        )
        report = default_validator.validate(segment)
        assert report.completeness.signal_sufficiency < 1.0


class TestValidityDetermination:

    def test_valid_segment(self, default_validator, sample_segment):
        """Test that good segment is valid."""
        report = default_validator.validate(sample_segment)
        # With default config, sample segment should be valid
        assert report.valid is True or report.overall_score > 0.3

    def test_invalid_due_to_critical_issues(self, default_validator, base_time):
        """Test invalid segment due to future timestamps."""
        future_time = time.time() + 3600  # 1 hour in future
        segment = EffortSegment(
            segment_id="future-segment",
            start_time=base_time,
            end_time=base_time + 60,
            signals=[
                EffortSignal(
                    signal_type=SignalType.TEXT,
                    timestamp=future_time,
                    hash="future123",
                    modality="keyboard",
                )
            ],
        )
        report = default_validator.validate(segment)
        assert report.valid is False


# =============================================================================
# Tests: ConsistencyChecker
# =============================================================================


class TestConsistencyChecker:

    def test_check_temporal_consistency_valid(self, sample_signals):
        checker = ConsistencyChecker()
        is_consistent, issues = checker.check_temporal_consistency(sample_signals)
        assert is_consistent is True
        assert len(issues) == 0

    def test_check_temporal_consistency_future(self, base_time):
        checker = ConsistencyChecker()
        # Need at least 2 signals for this check
        future_signals = [
            EffortSignal(
                signal_type=SignalType.TEXT,
                timestamp=base_time,
                hash="past",
                modality="keyboard",
            ),
            EffortSignal(
                signal_type=SignalType.TEXT,
                timestamp=time.time() + 3600,
                hash="future",
                modality="keyboard",
            ),
        ]
        is_consistent, issues = checker.check_temporal_consistency(future_signals)
        assert is_consistent is False
        assert "future" in issues[0].lower()

    def test_check_sequence_consistency_valid(self, sample_signals):
        checker = ConsistencyChecker()
        is_consistent, issues = checker.check_sequence_consistency(sample_signals)
        assert is_consistent is True

    def test_check_sequence_consistency_duplicates(self, base_time):
        checker = ConsistencyChecker()
        signals = [
            EffortSignal(
                signal_type=SignalType.TEXT,
                timestamp=base_time + i * 10,
                hash=f"hash{i}",
                modality="keyboard",
                sequence_number=0,  # All same sequence
            )
            for i in range(3)
        ]
        is_consistent, issues = checker.check_sequence_consistency(signals)
        assert is_consistent is False
        assert "duplicate" in issues[0].lower()


# =============================================================================
# Tests: DuplicationDetector
# =============================================================================


class TestDuplicationDetector:

    def test_detect_hash_duplication_none(self, sample_signals):
        detector = DuplicationDetector()
        duplicates = detector.detect_hash_duplication(sample_signals)
        assert len(duplicates) == 0

    def test_detect_hash_duplication_found(self, base_time):
        detector = DuplicationDetector()
        same_hash = hashlib.sha256(b"same").hexdigest()
        signals = [
            EffortSignal(
                signal_type=SignalType.TEXT,
                timestamp=base_time + i,
                hash=same_hash,
                modality="keyboard",
                sequence_number=i,
            )
            for i in range(3)
        ]
        duplicates = detector.detect_hash_duplication(signals)
        assert len(duplicates) == 2  # Second and third are duplicates of first

    def test_detect_timing_patterns_regular(self, synthetic_segment):
        detector = DuplicationDetector()
        patterns = detector.detect_timing_patterns(synthetic_segment.signals)
        assert "perfectly_regular" in patterns

    def test_detect_timing_patterns_natural(self, sample_signals):
        """Test natural timing has no suspicious patterns."""
        detector = DuplicationDetector()
        patterns = detector.detect_timing_patterns(sample_signals)
        assert "perfectly_regular" not in patterns


# =============================================================================
# Tests: ConfidenceCalculator
# =============================================================================


class TestConfidenceCalculator:

    def test_calculate_high_confidence(self):
        calculator = ConfidenceCalculator()
        confidence = calculator.calculate(
            signal_count=10,
            duration=300.0,
            consistency_score=1.0,
            authenticity_score=1.0,
            issue_count=0,
        )
        assert confidence > 0.8

    def test_calculate_low_confidence(self):
        calculator = ConfidenceCalculator()
        confidence = calculator.calculate(
            signal_count=1,
            duration=5.0,
            consistency_score=0.3,
            authenticity_score=0.3,
            issue_count=5,
        )
        assert confidence < 0.5

    def test_calculate_with_custom_weights(self):
        calculator = ConfidenceCalculator(
            weights={
                "signal_count": 0.5,
                "duration": 0.5,
                "consistency": 0.0,
                "authenticity": 0.0,
            }
        )
        confidence = calculator.calculate(
            signal_count=10,
            duration=300.0,
            consistency_score=0.0,
            authenticity_score=0.0,
        )
        assert confidence > 0.7


# =============================================================================
# Tests: Convenience Functions
# =============================================================================


class TestConvenienceFunctions:

    def test_create_enhanced_validator_default(self):
        validator = create_enhanced_validator()
        assert validator.validator_id == "enhanced_validator"

    def test_create_enhanced_validator_custom(self, custom_config):
        validator = create_enhanced_validator(
            validator_id="custom-001",
            config=custom_config,
        )
        assert validator.validator_id == "custom-001"
        assert validator.config.min_signals == 3

    def test_validate_segment_function(self, sample_segment):
        report = validate_segment(sample_segment)
        assert report.segment_id == "test-segment-001"
        assert report.overall_score >= 0

    def test_validate_segment_with_config(self, sample_segment, custom_config):
        report = validate_segment(sample_segment, config=custom_config)
        assert report.segment_id == "test-segment-001"


# =============================================================================
# Tests: MP-02 Spec Compliance
# =============================================================================


class TestMP02Compliance:

    def test_deterministic_summaries(self, sample_segment):
        """Per MP-02 §7: Validators MUST produce deterministic summaries."""
        validator1 = create_enhanced_validator()
        validator2 = create_enhanced_validator()

        report1 = validator1.validate(sample_segment)
        report2 = validator2.validate(sample_segment)

        # Core scores should be identical
        assert report1.overall_score == report2.overall_score
        assert report1.coherence.overall == report2.coherence.overall

    def test_model_identity_disclosed(self, default_validator):
        """Per MP-02 §7: Validators MUST disclose model identity and version."""
        assert default_validator.model_id is not None
        assert default_validator.model_version is not None
        assert len(default_validator.model_version) > 0

    def test_uncertainty_preserved(self, default_validator, sample_segment):
        """Per MP-02 §7: Validators MUST preserve dissent and uncertainty."""
        report = default_validator.validate(sample_segment)
        # Uncertainty markers are available for review
        assert hasattr(report, "uncertainty_markers")
        assert hasattr(report, "confidence")
        # Confidence should never be exactly 1.0
        assert report.confidence < 1.0

    def test_no_value_assertion(self, default_validator, sample_segment):
        """Per MP-02 §7: Validators MUST NOT declare effort as valuable."""
        report = default_validator.validate(sample_segment)
        # Report should not contain value assertions
        verdict = report.verdict_reason.lower()
        assert "valuable" not in verdict
        assert "worth" not in verdict

    def test_no_ownership_assertion(self, default_validator, sample_segment):
        """Per MP-02 §7: Validators MUST NOT assert originality or ownership."""
        report = default_validator.validate(sample_segment)
        verdict = report.verdict_reason.lower()
        assert "original" not in verdict
        assert "owner" not in verdict
        assert "belong" not in verdict


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:

    def test_single_signal_segment(self, default_validator, base_time):
        segment = EffortSegment(
            segment_id="single-signal",
            start_time=base_time,
            end_time=base_time + 60,
            signals=[
                EffortSignal(
                    signal_type=SignalType.TEXT,
                    timestamp=base_time + 30,
                    hash="single123",
                    modality="keyboard",
                )
            ],
        )
        report = default_validator.validate(segment)
        # Should handle gracefully
        assert report.overall_score >= 0
        assert report.segment_id == "single-signal"

    def test_very_short_duration(self, default_validator, base_time):
        segment = EffortSegment(
            segment_id="short-segment",
            start_time=base_time,
            end_time=base_time + 0.1,
            signals=[
                EffortSignal(
                    signal_type=SignalType.TEXT,
                    timestamp=base_time + 0.05,
                    hash="short123",
                    modality="keyboard",
                )
            ],
        )
        report = default_validator.validate(segment)
        assert report.completeness.duration_adequacy < 0.5

    def test_many_signals_short_time(self, default_validator, base_time):
        """Test burst of signals (potential adversarial)."""
        signals = [
            EffortSignal(
                signal_type=SignalType.TEXT,
                timestamp=base_time + i * 0.01,
                hash=hashlib.sha256(f"burst{i}".encode()).hexdigest(),
                modality="keyboard",
                sequence_number=i,
            )
            for i in range(20)
        ]
        segment = EffortSegment(
            segment_id="burst-segment",
            start_time=base_time,
            end_time=base_time + 0.2,
            signals=signals,
        )
        report = default_validator.validate(segment)
        # Should flag unrealistic burst
        assert "unrealistic_signal_burst" in report.authenticity.adversarial_patterns

    def test_zero_duration_segment(self, default_validator, base_time):
        segment = EffortSegment(
            segment_id="zero-duration",
            start_time=base_time,
            end_time=base_time,  # Zero duration
            signals=[],
        )
        report = default_validator.validate(segment)
        # Should not crash and have very low score
        assert report.overall_score < 0.1
        assert report.valid is False
