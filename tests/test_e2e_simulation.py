# tests/test_e2e_simulation.py
"""
End-to-end simulation tests for Value Ledger.

Runs full workflow simulations to verify all components work together.
"""

import pytest
import time
import tempfile
import hashlib
import os
import shutil
from pathlib import Path

from value_ledger import (
    # Core
    ValueLedger,
    # Receipt system
    EffortSignal,
    EffortSegment,
    EffortReceipt,
    ValidationMetadata,
    # Validation
    EnhancedValidator,
    ValidationReport,
    CriteriaConfig,
    create_enhanced_validator,
    validate_segment,
    # Privacy
    ConsentRegistry,
    AgencyController,
    PrivacyFilter,
    PrivacyPolicy,
    ObservationConsent,
    ConsentStatus,
    # Compatibility
    ProtocolAdapter,
    AuditExporter,
    AuditFormat,
    LicenseManager,
    LicenseType,
    MP01Formatter,
    create_protocol_adapter,
    export_receipt_for_audit,
)
from value_ledger.receipt import SignalType


class EndToEndSimulation:
    """
    Full end-to-end simulation of the Value Ledger workflow.

    Simulates:
    1. Observer captures effort signals
    2. Signals grouped into segments
    3. Segments validated by enhanced validator
    4. Receipts generated with validation metadata
    5. Privacy filtering applied
    6. Export to various audit formats
    7. Licensing and protocol adaptation
    """

    def __init__(self, run_id: int, temp_dir: str):
        self.run_id = run_id
        self.temp_dir = temp_dir
        self.base_time = time.time() - 300  # Start 5 minutes ago
        self.signals = []
        self.segments = []
        self.receipts = []
        self.validation_reports = []
        self.errors = []

    def run(self) -> dict:
        """Run complete simulation and return results."""
        results = {
            "run_id": self.run_id,
            "success": True,
            "steps_completed": [],
            "errors": [],
            "metrics": {},
        }

        try:
            # Step 1: Setup consent
            self._step_setup_consent(results)

            # Step 2: Capture signals
            self._step_capture_signals(results)

            # Step 3: Create segments
            self._step_create_segments(results)

            # Step 4: Validate segments
            self._step_validate_segments(results)

            # Step 5: Generate receipts
            self._step_generate_receipts(results)

            # Step 6: Apply privacy filtering
            self._step_apply_privacy(results)

            # Step 7: Export to audit formats
            self._step_export_audits(results)

            # Step 8: Test licensing
            self._step_test_licensing(results)

            # Step 9: Test protocol adaptation
            self._step_test_protocol_adapter(results)

            # Step 10: Verify ledger operations
            self._step_verify_ledger(results)

        except Exception as e:
            results["success"] = False
            results["errors"].append(f"Simulation failed: {str(e)}")

        return results

    def _step_setup_consent(self, results: dict):
        """Setup consent registry."""
        consent_path = os.path.join(self.temp_dir, f"consent_{self.run_id}.json")
        registry = ConsentRegistry(storage_path=consent_path)

        # Grant consent for test human
        human_id = f"test_human_{self.run_id}"
        consent = registry.grant_consent(
            human_id=human_id,
            allowed_observers=["test_observer"],
        )

        assert consent.status == ConsentStatus.GRANTED
        assert registry.is_observation_allowed(human_id, "test_observer")

        self.registry = registry
        self.human_id = human_id
        results["steps_completed"].append("setup_consent")

    def _step_capture_signals(self, results: dict):
        """Simulate capturing effort signals."""
        signal_types = [
            SignalType.TEXT,
            SignalType.KEYSTROKE,
            SignalType.COMMAND,
            SignalType.NAVIGATION,
            SignalType.TOOL,
        ]

        # Capture 15-25 signals with natural variance
        import random
        num_signals = random.randint(15, 25)

        for i in range(num_signals):
            # Natural timing variance
            timestamp = self.base_time + i * (10 + random.uniform(-2, 5))
            content = f"signal_content_{self.run_id}_{i}_{random.randint(1000, 9999)}"

            signal = EffortSignal(
                signal_type=random.choice(signal_types),
                timestamp=timestamp,
                hash=hashlib.sha256(content.encode()).hexdigest(),
                modality="keyboard" if i % 3 != 0 else "mouse",
                sequence_number=i,
                metadata={"run_id": self.run_id, "index": i},
            )
            self.signals.append(signal)

        results["metrics"]["signals_captured"] = len(self.signals)
        results["steps_completed"].append("capture_signals")

    def _step_create_segments(self, results: dict):
        """Group signals into segments."""
        # Create 2-3 segments
        segment_size = len(self.signals) // 2

        for seg_idx in range(2):
            start_idx = seg_idx * segment_size
            end_idx = start_idx + segment_size if seg_idx == 0 else len(self.signals)
            segment_signals = self.signals[start_idx:end_idx]

            if not segment_signals:
                continue

            segment = EffortSegment(
                segment_id=f"segment_{self.run_id}_{seg_idx}",
                start_time=segment_signals[0].timestamp,
                end_time=segment_signals[-1].timestamp + 1,
                signals=segment_signals,
                segmentation_rule="time_bounded",
                metadata={"run_id": self.run_id},
            )
            self.segments.append(segment)

        results["metrics"]["segments_created"] = len(self.segments)
        results["steps_completed"].append("create_segments")

    def _step_validate_segments(self, results: dict):
        """Validate segments with enhanced validator."""
        validator = create_enhanced_validator(
            validator_id=f"validator_{self.run_id}"
        )

        for segment in self.segments:
            report = validator.validate(segment)
            self.validation_reports.append(report)

            # Verify report structure
            assert report.segment_id == segment.segment_id
            assert 0 <= report.overall_score <= 1
            assert 0 <= report.confidence <= 1
            assert report.coherence is not None
            assert report.progression is not None
            assert report.consistency is not None
            assert report.authenticity is not None
            assert report.completeness is not None

        valid_count = sum(1 for r in self.validation_reports if r.valid)
        results["metrics"]["segments_validated"] = len(self.validation_reports)
        results["metrics"]["segments_valid"] = valid_count
        results["metrics"]["avg_score"] = sum(r.overall_score for r in self.validation_reports) / len(self.validation_reports)
        results["steps_completed"].append("validate_segments")

    def _step_generate_receipts(self, results: dict):
        """Generate receipts from validated segments."""
        for i, (segment, report) in enumerate(zip(self.segments, self.validation_reports)):
            receipt = EffortReceipt(
                receipt_id=f"receipt_{self.run_id}_{i}",
                time_bounds=(segment.start_time, segment.end_time),
                signal_hashes=segment.get_signal_hashes(),
                effort_summary=f"Effort segment {i} with {segment.signal_count} signals",
                observer_id=f"observer_{self.run_id}",
                validator_id=report.validator_id,
                validation_metadata=ValidationMetadata(
                    validator_id=report.validator_id,
                    model_id="enhanced_heuristic",
                    model_version="2.0.0",
                    validation_timestamp=report.timestamp,
                    coherence_score=report.coherence.overall,
                    uncertainty_markers=[issue.message for issue in report.issues[:3]],
                ),
            )
            self.receipts.append(receipt)

        results["metrics"]["receipts_generated"] = len(self.receipts)
        results["steps_completed"].append("generate_receipts")

    def _step_apply_privacy(self, results: dict):
        """Apply privacy filtering to receipts."""
        policy = PrivacyPolicy(
            allow_content_in_receipts=False,
            redact_patterns=[r"\b\d{4}\b"],  # Redact 4-digit numbers
        )
        privacy_filter = PrivacyFilter(policy=policy)

        filtered_receipts = []
        for receipt in self.receipts:
            # Convert to dict for filtering
            receipt_dict = {
                "receipt_id": receipt.receipt_id,
                "time_bounds": receipt.time_bounds,
                "signal_hashes": receipt.signal_hashes,
                "effort_summary": receipt.effort_summary,
            }
            filtered = privacy_filter.filter_receipt(receipt_dict)
            filtered_receipts.append(filtered)

            # Verify filtering applied
            assert filtered.get("privacy_filtered") is True

        self.filtered_receipts = filtered_receipts
        results["metrics"]["receipts_filtered"] = len(filtered_receipts)
        results["steps_completed"].append("apply_privacy")

    def _step_export_audits(self, results: dict):
        """Export receipts to various audit formats."""
        exporter = AuditExporter(issuer_id=f"issuer_{self.run_id}")

        formats_tested = []
        for receipt in self.receipts:
            # Test all audit formats
            json_ld = exporter.export_receipt(receipt, AuditFormat.JSON_LD)
            assert "@context" in json_ld
            formats_tested.append("json_ld")

            w3c_vc = exporter.export_receipt(receipt, AuditFormat.W3C_VC)
            assert "credentialSubject" in w3c_vc
            formats_tested.append("w3c_vc")

            ots = exporter.export_receipt(receipt, AuditFormat.OPEN_TIMESTAMPS)
            assert "version" in ots
            formats_tested.append("open_timestamps")

            audit_log = exporter.export_receipt(receipt, AuditFormat.AUDIT_LOG)
            assert "entry" in audit_log
            formats_tested.append("audit_log")

        results["metrics"]["audit_formats_tested"] = len(set(formats_tested))
        results["steps_completed"].append("export_audits")

    def _step_test_licensing(self, results: dict):
        """Test licensing functionality."""
        manager = LicenseManager()

        licenses_created = 0
        for receipt in self.receipts:
            # Grant a license
            license_ref = manager.grant_license(
                receipt_ids=[receipt.receipt_id],
                licensor_id=self.human_id,
                licensee_ids=[f"licensee_{self.run_id}"],
                license_type=LicenseType.VIEW_ONLY,
            )
            licenses_created += 1

            # Verify license
            has_access, reason = manager.check_license(
                license_ref.license_id, f"licensee_{self.run_id}"
            )
            assert has_access is True

            # Verify unauthorized access denied
            no_access, reason = manager.check_license(
                license_ref.license_id, "unauthorized_entity"
            )
            assert no_access is False

        results["metrics"]["licenses_created"] = licenses_created
        results["steps_completed"].append("test_licensing")

    def _step_test_protocol_adapter(self, results: dict):
        """Test protocol adaptation (MP-01 compatibility)."""
        adapter = create_protocol_adapter()

        for receipt in self.receipts:
            # Prepare for negotiation
            proposal, formatted = adapter.prepare_for_negotiation(
                receipt=receipt,
                recipient_id=f"recipient_{self.run_id}",
                terms={"usage": "testing"},
            )
            assert proposal is not None

            # Export for audit
            audit_export = export_receipt_for_audit(
                receipt=receipt,
                format=AuditFormat.JSON_LD,
            )
            assert audit_export is not None

        results["metrics"]["protocol_adaptations"] = len(self.receipts)
        results["steps_completed"].append("test_protocol_adapter")

    def _step_verify_ledger(self, results: dict):
        """Verify ledger operations."""
        ledger_path = os.path.join(self.temp_dir, f"ledger_{self.run_id}.jsonl")
        ledger = ValueLedger(storage_path=ledger_path)

        # Verify ledger initialized
        assert ledger.storage_path.exists() or len(ledger.entries) == 0

        results["metrics"]["ledger_initialized"] = True
        results["steps_completed"].append("verify_ledger")


@pytest.fixture
def temp_simulation_dir():
    """Create temporary directory for simulation."""
    temp_dir = tempfile.mkdtemp(prefix="value_ledger_e2e_")
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestEndToEndSimulation:
    """Run end-to-end simulations."""

    @pytest.mark.parametrize("run_id", range(10))
    def test_full_simulation(self, run_id, temp_simulation_dir):
        """Run full end-to-end simulation."""
        simulation = EndToEndSimulation(run_id, temp_simulation_dir)
        results = simulation.run()

        # Verify success
        assert results["success"] is True, f"Simulation {run_id} failed: {results['errors']}"

        # Verify all steps completed
        expected_steps = [
            "setup_consent",
            "capture_signals",
            "create_segments",
            "validate_segments",
            "generate_receipts",
            "apply_privacy",
            "export_audits",
            "test_licensing",
            "test_protocol_adapter",
            "verify_ledger",
        ]
        for step in expected_steps:
            assert step in results["steps_completed"], f"Step '{step}' not completed in run {run_id}"

        # Verify metrics
        assert results["metrics"]["signals_captured"] >= 15
        assert results["metrics"]["segments_created"] >= 1
        assert results["metrics"]["segments_validated"] >= 1
        assert results["metrics"]["receipts_generated"] >= 1
        assert results["metrics"]["avg_score"] > 0

        print(f"\nRun {run_id} completed successfully:")
        print(f"  Signals: {results['metrics']['signals_captured']}")
        print(f"  Segments: {results['metrics']['segments_created']}")
        print(f"  Avg Score: {results['metrics']['avg_score']:.3f}")
        print(f"  Valid Segments: {results['metrics']['segments_valid']}/{results['metrics']['segments_validated']}")


def test_simulation_stress():
    """Run multiple simulations to verify consistency."""
    temp_dir = tempfile.mkdtemp(prefix="value_ledger_stress_")

    try:
        all_results = []
        for i in range(10):
            simulation = EndToEndSimulation(i, temp_dir)
            results = simulation.run()
            all_results.append(results)

        # All should succeed
        success_count = sum(1 for r in all_results if r["success"])
        assert success_count == 10, f"Only {success_count}/10 simulations succeeded"

        # Calculate aggregate stats
        total_signals = sum(r["metrics"]["signals_captured"] for r in all_results)
        total_segments = sum(r["metrics"]["segments_created"] for r in all_results)
        avg_scores = [r["metrics"]["avg_score"] for r in all_results]

        print(f"\n=== Stress Test Results ===")
        print(f"Simulations: 10/10 passed")
        print(f"Total Signals: {total_signals}")
        print(f"Total Segments: {total_segments}")
        print(f"Score Range: {min(avg_scores):.3f} - {max(avg_scores):.3f}")
        print(f"Avg Score: {sum(avg_scores)/len(avg_scores):.3f}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
