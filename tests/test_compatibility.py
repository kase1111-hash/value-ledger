# tests/test_compatibility.py
"""
Tests for MP-02 External Compatibility.

Per MP-02 §14:
- MP-02 receipts are compatible with MP-01 Negotiation & Ratification
- Licensing and delegation modules
- External audit systems
"""

import pytest
import time
import json

from value_ledger.compatibility import (
    NegotiationStatus,
    RatificationMethod,
    MP01Proposal,
    MP01Ratification,
    MP01Formatter,
    LicenseType,
    LicenseReference,
    DelegationRecord,
    LicenseManager,
    AuditFormat,
    AuditEntry,
    AuditExporter,
    ProtocolAdapter,
    create_protocol_adapter,
    export_receipt_for_audit,
)
from value_ledger.receipt import (
    EffortReceipt,
    ValidationMetadata,
)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def sample_receipt():
    """Create a sample receipt for testing."""
    return EffortReceipt(
        receipt_id="test-receipt-001",
        time_bounds=(1000.0, 2000.0),
        signal_hashes=["hash1", "hash2", "hash3"],
        effort_summary="Test effort spanning 1000 seconds with 3 signals",
        validation_metadata=ValidationMetadata(
            validator_id="test-validator",
            model_id="heuristic",
            model_version="1.0.0",
            coherence_score=0.8,
            progression_score=0.7,
        ),
        observer_id="test-observer",
        validator_id="test-validator",
        prior_receipts=["prior-001", "prior-002"],
        receipt_hash="abc123def456",
    )


# =============================================================================
# MP-01 Proposal Tests
# =============================================================================

class TestMP01Proposal:
    """Tests for MP-01 negotiation proposals."""

    def test_proposal_creation(self):
        proposal = MP01Proposal(
            receipt_ids=["receipt-1", "receipt-2"],
            proposer_id="party-a",
            recipient_id="party-b",
            terms={"price": 100, "duration": "30d"},
        )

        assert proposal.status == NegotiationStatus.PROPOSED
        assert len(proposal.receipt_ids) == 2
        assert proposal.proposer_id == "party-a"
        assert proposal.recipient_id == "party-b"

    def test_proposal_hash_is_deterministic(self):
        proposal = MP01Proposal(
            receipt_ids=["r1", "r2"],
            proposer_id="a",
            recipient_id="b",
            terms={"x": 1},
        )

        hash1 = proposal.compute_hash()
        hash2 = proposal.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex

    def test_proposal_serialization(self):
        proposal = MP01Proposal(
            receipt_ids=["r1"],
            proposer_id="a",
            recipient_id="b",
            terms={"key": "value"},
        )

        data = proposal.to_dict()
        restored = MP01Proposal.from_dict(data)

        assert restored.proposal_id == proposal.proposal_id
        assert restored.receipt_ids == proposal.receipt_ids
        assert restored.terms == proposal.terms
        assert data["protocol"] == "MP-01"


class TestMP01Formatter:
    """Tests for MP-01 formatter."""

    def test_create_proposal(self):
        formatter = MP01Formatter(party_id="party-a")

        proposal = formatter.create_proposal(
            receipt_ids=["r1", "r2"],
            recipient_id="party-b",
            terms={"license": "reference"},
        )

        assert proposal.proposer_id == "party-a"
        assert proposal.recipient_id == "party-b"
        assert proposal.status == NegotiationStatus.PROPOSED

    def test_create_proposal_with_expiry(self):
        formatter = MP01Formatter()

        proposal = formatter.create_proposal(
            receipt_ids=["r1"],
            recipient_id="b",
            terms={},
            expires_in_seconds=3600,
        )

        assert proposal.expires_at is not None
        assert proposal.expires_at > time.time()

    def test_accept_proposal(self):
        formatter = MP01Formatter()
        proposal = formatter.create_proposal(["r1"], "b", {})

        formatter.accept_proposal(proposal)

        assert proposal.status == NegotiationStatus.ACCEPTED

    def test_reject_proposal(self):
        formatter = MP01Formatter()
        proposal = formatter.create_proposal(["r1"], "b", {})

        formatter.reject_proposal(proposal, reason="Not interested")

        assert proposal.status == NegotiationStatus.REJECTED
        assert proposal.terms.get("rejection_reason") == "Not interested"

    def test_create_counter_proposal(self):
        formatter = MP01Formatter(party_id="party-b")
        original = MP01Proposal(
            receipt_ids=["r1"],
            proposer_id="party-a",
            recipient_id="party-b",
            terms={"price": 100},
        )

        counter = formatter.create_counter_proposal(original, {"price": 150})

        assert counter.status == NegotiationStatus.COUNTER_OFFERED
        assert original.status == NegotiationStatus.COUNTER_OFFERED
        assert counter.terms["price"] == 150
        assert counter.proposal_id in original.counter_proposals

    def test_ratify_proposal(self):
        formatter = MP01Formatter()
        proposal = formatter.create_proposal(["r1"], "b", {})
        formatter.accept_proposal(proposal)

        ratification = formatter.ratify_proposal(
            proposal,
            ratifiers=["party-a", "party-b"],
        )

        assert proposal.status == NegotiationStatus.RATIFIED
        assert ratification.proposal_id == proposal.proposal_id
        assert "party-a" in ratification.ratified_by
        assert "party-b" in ratification.ratified_by

    def test_format_for_negotiation(self, sample_receipt):
        formatter = MP01Formatter()

        formatted = formatter.format_for_negotiation(sample_receipt)

        assert formatted["format"] == "MP-01/MP-02"
        assert formatted["receipt"]["id"] == sample_receipt.receipt_id
        assert formatted["receipt"]["hash"] == sample_receipt.receipt_hash
        assert formatted["negotiation_metadata"]["verifiable"] is True


# =============================================================================
# License Manager Tests
# =============================================================================

class TestLicenseReference:
    """Tests for license references."""

    def test_license_creation(self):
        license_ref = LicenseReference(
            license_type=LicenseType.REFERENCE,
            receipt_ids=["r1", "r2"],
            licensor_id="owner",
            licensee_ids=["user1", "user2"],
        )

        assert license_ref.is_valid()
        assert license_ref.license_type == LicenseType.REFERENCE
        assert len(license_ref.receipt_ids) == 2

    def test_license_expiration(self):
        license_ref = LicenseReference(
            license_type=LicenseType.VIEW_ONLY,
            receipt_ids=["r1"],
            licensor_id="owner",
            licensee_ids=["user"],
            expires_at=time.time() - 1,  # Already expired
        )

        assert not license_ref.is_valid()

    def test_license_hash_is_deterministic(self):
        license_ref = LicenseReference(
            license_type=LicenseType.FULL,
            receipt_ids=["r1"],
            licensor_id="owner",
            licensee_ids=["user"],
        )

        hash1 = license_ref.compute_hash()
        hash2 = license_ref.compute_hash()

        assert hash1 == hash2

    def test_license_serialization(self):
        license_ref = LicenseReference(
            license_type=LicenseType.DERIVATIVE,
            receipt_ids=["r1"],
            licensor_id="owner",
            licensee_ids=["user"],
            delegation_allowed=True,
            delegation_depth=2,
        )

        data = license_ref.to_dict()
        restored = LicenseReference.from_dict(data)

        assert restored.license_type == license_ref.license_type
        assert restored.delegation_allowed == license_ref.delegation_allowed
        assert restored.delegation_depth == license_ref.delegation_depth


class TestLicenseManager:
    """Tests for license manager."""

    def test_grant_license(self):
        manager = LicenseManager()

        license_ref = manager.grant_license(
            receipt_ids=["r1", "r2"],
            licensor_id="owner",
            licensee_ids=["user1"],
            license_type=LicenseType.REFERENCE,
        )

        assert license_ref.is_valid()
        assert manager.get_license(license_ref.license_id) is not None

    def test_revoke_license(self):
        manager = LicenseManager()
        license_ref = manager.grant_license(
            receipt_ids=["r1"],
            licensor_id="owner",
            licensee_ids=["user"],
        )

        result = manager.revoke_license(license_ref.license_id)

        assert result is True
        assert manager.get_license(license_ref.license_id) is None

    def test_revoke_non_revocable_fails(self):
        manager = LicenseManager()
        license_ref = manager.grant_license(
            receipt_ids=["r1"],
            licensor_id="owner",
            licensee_ids=["user"],
        )
        license_ref.revocable = False

        result = manager.revoke_license(license_ref.license_id)

        assert result is False

    def test_check_license_direct_licensee(self):
        manager = LicenseManager()
        license_ref = manager.grant_license(
            receipt_ids=["r1"],
            licensor_id="owner",
            licensee_ids=["user1", "user2"],
        )

        has_access, reason = manager.check_license(license_ref.license_id, "user1")

        assert has_access is True
        assert reason == "Direct licensee"

    def test_check_license_unauthorized(self):
        manager = LicenseManager()
        license_ref = manager.grant_license(
            receipt_ids=["r1"],
            licensor_id="owner",
            licensee_ids=["user1"],
        )

        has_access, reason = manager.check_license(license_ref.license_id, "user2")

        assert has_access is False
        assert reason == "Not authorized"

    def test_delegate_license(self):
        manager = LicenseManager()
        license_ref = manager.grant_license(
            receipt_ids=["r1"],
            licensor_id="owner",
            licensee_ids=["user1"],
            delegation_allowed=True,
            delegation_depth=2,
        )

        delegation = manager.delegate_license(
            license_id=license_ref.license_id,
            delegator_id="user1",
            delegatee_id="user2",
        )

        assert delegation is not None
        assert delegation.depth == 1

        has_access, reason = manager.check_license(license_ref.license_id, "user2")
        assert has_access is True
        assert "Delegated" in reason

    def test_delegation_depth_limit(self):
        manager = LicenseManager()
        license_ref = manager.grant_license(
            receipt_ids=["r1"],
            licensor_id="owner",
            licensee_ids=["user1"],
            delegation_allowed=True,
            delegation_depth=1,  # Only 1 level
        )

        # First delegation succeeds
        d1 = manager.delegate_license(license_ref.license_id, "user1", "user2")
        assert d1 is not None

        # Second delegation fails (exceeds depth)
        d2 = manager.delegate_license(license_ref.license_id, "user2", "user3")
        assert d2 is None

    def test_list_licenses_for_receipt(self):
        manager = LicenseManager()
        manager.grant_license(["r1", "r2"], "owner", ["u1"])
        manager.grant_license(["r2", "r3"], "owner", ["u2"])
        manager.grant_license(["r3"], "owner", ["u3"])

        licenses = manager.list_licenses_for_receipt("r2")

        assert len(licenses) == 2


# =============================================================================
# Audit Exporter Tests
# =============================================================================

class TestAuditExporter:
    """Tests for audit format exports."""

    def test_export_json_ld(self, sample_receipt):
        exporter = AuditExporter()

        result = exporter.export_receipt(sample_receipt, AuditFormat.JSON_LD)

        assert "@context" in result
        assert "@type" in result
        assert result["@type"] == "mp02:EffortReceipt"
        assert result["mp02:receiptId"] == sample_receipt.receipt_id

    def test_export_w3c_vc(self, sample_receipt):
        exporter = AuditExporter(issuer_id="test-issuer")

        result = exporter.export_receipt(sample_receipt, AuditFormat.W3C_VC)

        assert "VerifiableCredential" in result["type"]
        assert result["issuer"] == "test-issuer"
        assert "credentialSubject" in result
        assert "proof" in result

    def test_export_open_timestamps(self, sample_receipt):
        exporter = AuditExporter()

        result = exporter.export_receipt(sample_receipt, AuditFormat.OPEN_TIMESTAMPS)

        assert result["version"] == 1
        assert result["file_hash"] == sample_receipt.receipt_hash
        assert result["file_hash_type"] == "sha256"
        assert "timestamp" in result
        assert "attestations" in result["timestamp"]

    def test_export_audit_log(self, sample_receipt):
        exporter = AuditExporter()

        result = exporter.export_receipt(sample_receipt, AuditFormat.AUDIT_LOG)

        assert result["log_type"] == "effort_receipt"
        entry = result["entry"]
        assert entry["id"] == sample_receipt.receipt_id
        assert entry["action"] == "effort_recorded"
        assert "evidence" in entry
        assert "integrity" in entry

    def test_create_audit_entry(self, sample_receipt):
        exporter = AuditExporter()

        entry = exporter.create_audit_entry(
            receipt=sample_receipt,
            event_type="verification",
            actor_id="auditor-1",
            action="verify_receipt",
            details={"method": "hash_check"},
        )

        assert entry.subject_id == sample_receipt.receipt_id
        assert entry.actor_id == "auditor-1"
        assert entry.event_type == "verification"

    def test_batch_export(self, sample_receipt):
        exporter = AuditExporter()

        # Create multiple receipts
        receipts = [sample_receipt]
        for i in range(2):
            r = EffortReceipt(
                receipt_id=f"receipt-{i}",
                time_bounds=(1000.0, 2000.0),
                signal_hashes=["h1"],
                effort_summary="Test",
                validation_metadata=ValidationMetadata(
                    validator_id="v", model_id="m", model_version="1"
                ),
                observer_id="o",
                validator_id="v",
            )
            receipts.append(r)

        results = exporter.batch_export(receipts, AuditFormat.AUDIT_LOG)

        assert len(results) == 3


# =============================================================================
# Protocol Adapter Tests
# =============================================================================

class TestProtocolAdapter:
    """Tests for cross-protocol adapter."""

    def test_prepare_for_negotiation(self, sample_receipt):
        adapter = create_protocol_adapter(party_id="party-a")

        proposal, formatted = adapter.prepare_for_negotiation(
            receipt=sample_receipt,
            recipient_id="party-b",
            terms={"license": "reference"},
        )

        assert proposal.proposer_id == "party-a"
        assert proposal.recipient_id == "party-b"
        assert formatted["format"] == "MP-01/MP-02"

    def test_create_licensed_export(self, sample_receipt):
        adapter = create_protocol_adapter()

        result = adapter.create_licensed_export(
            receipt=sample_receipt,
            licensee_ids=["user1", "user2"],
            license_type=LicenseType.REFERENCE,
            audit_format=AuditFormat.JSON_LD,
        )

        assert "license" in result
        assert result["license"]["license_type"] == LicenseType.REFERENCE
        assert "@context" in result  # JSON-LD format

    def test_verify_and_export(self, sample_receipt):
        adapter = create_protocol_adapter()

        result = adapter.verify_and_export(
            receipt=sample_receipt,
            format=AuditFormat.AUDIT_LOG,
            verifier_id="auditor-1",
        )

        assert "verification" in result
        assert result["verification"]["actor_id"] == "auditor-1"


# =============================================================================
# Convenience Function Tests
# =============================================================================

class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_create_protocol_adapter(self):
        adapter = create_protocol_adapter(
            party_id="test-party",
            issuer_id="test-issuer",
        )

        assert adapter.mp01.party_id == "test-party"
        assert adapter.audit.issuer_id == "test-issuer"

    def test_export_receipt_for_audit(self, sample_receipt):
        result = export_receipt_for_audit(sample_receipt, AuditFormat.JSON_LD)

        assert "@context" in result
        assert "@type" in result


# =============================================================================
# Integration Tests
# =============================================================================

class TestCompatibilityIntegration:
    """Integration tests for external compatibility."""

    def test_full_negotiation_workflow(self, sample_receipt):
        """Test complete MP-01 negotiation workflow."""
        # Party A proposes
        formatter_a = MP01Formatter(party_id="party-a")
        proposal = formatter_a.create_proposal(
            receipt_ids=[sample_receipt.receipt_id],
            recipient_id="party-b",
            terms={"license": "reference", "duration": "1y"},
        )

        # Party B counter-offers
        formatter_b = MP01Formatter(party_id="party-b")
        counter = formatter_b.create_counter_proposal(
            proposal,
            {"license": "reference", "duration": "6m"},
        )

        # Party A accepts counter
        formatter_a.accept_proposal(counter)

        # Both parties ratify
        ratification = formatter_a.ratify_proposal(
            counter,
            ratifiers=["party-a", "party-b"],
        )

        assert counter.status == NegotiationStatus.RATIFIED
        assert len(ratification.ratified_by) == 2

    def test_licensed_export_with_delegation(self, sample_receipt):
        """Test licensed export with delegation chain."""
        adapter = create_protocol_adapter()

        # Create license with delegation
        license_ref = adapter.licenses.grant_license(
            receipt_ids=[sample_receipt.receipt_id],
            licensor_id="owner",
            licensee_ids=["user1"],
            delegation_allowed=True,
            delegation_depth=2,
        )

        # Delegate to user2
        adapter.licenses.delegate_license(
            license_ref.license_id, "user1", "user2"
        )

        # Check access
        has_access, _ = adapter.licenses.check_license(
            license_ref.license_id, "user2"
        )

        assert has_access is True

        # Export with license
        export = adapter.create_licensed_export(
            sample_receipt,
            licensee_ids=["user3"],
            audit_format=AuditFormat.W3C_VC,
        )

        assert "license" in export
        assert "proof" in export
