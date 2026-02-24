# tests/test_core.py
"""
Tests for Value Ledger Core Module.

Tests cover:
- ValueVector operations
- LedgerEntry creation and validation
- MerkleTree implementation
- ValueLedger operations (accrue, revoke, aggregate)
- Path validation security
- Proof generation and verification
- ClockMonitor and SourceValidator
- FailureModeHandler
"""

import json
import os
import tempfile
import time
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from value_ledger.core import (
    # Utilities
    generate_entry_id,
    _validate_ledger_path,
    compute_content_hash,
    compute_timestamp_proof,
    # Core classes
    MerkleTree,
    ValueVector,
    ProofData,
    LedgerEntry,
    ValueLedger,
    # Failure mode handling
    ClockMonitor,
    SourceValidator,
    FailureModeHandler,
)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestGenerateEntryId:

    def test_deterministic_id(self):
        """Same input produces same ID."""
        id1 = generate_entry_id("test_data")
        id2 = generate_entry_id("test_data")
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        """Different inputs produce different IDs."""
        id1 = generate_entry_id("data_1")
        id2 = generate_entry_id("data_2")
        assert id1 != id2

    def test_id_is_hex(self):
        """ID is a valid hex string."""
        id1 = generate_entry_id("test")
        assert all(c in "0123456789abcdef" for c in id1)

    def test_id_is_sha256_length(self):
        """ID is 64 characters (SHA-256 hex)."""
        id1 = generate_entry_id("test")
        assert len(id1) == 64


class TestValidateLedgerPath:

    def test_valid_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _validate_ledger_path(f"{tmpdir}/ledger.jsonl")
            assert path.name == "ledger.jsonl"

    def test_null_byte_rejected(self):
        with pytest.raises(ValueError, match="Invalid characters"):
            _validate_ledger_path("/tmp/test\x00.jsonl")

    def test_etc_path_rejected(self):
        with pytest.raises(ValueError, match="sensitive path"):
            _validate_ledger_path("/etc/passwd.jsonl")

    def test_proc_path_rejected(self):
        with pytest.raises(ValueError, match="sensitive path"):
            _validate_ledger_path("/proc/self/environ")

    def test_ssh_path_rejected(self):
        with pytest.raises(ValueError, match="sensitive path"):
            _validate_ledger_path("/home/user/.ssh/id_rsa")

    def test_root_path_rejected(self):
        with pytest.raises(ValueError, match="Cannot use root"):
            _validate_ledger_path("/")

    def test_relative_path_resolved(self):
        path = _validate_ledger_path("./ledger.jsonl")
        assert path.is_absolute()


class TestComputeContentHash:

    def test_hash_content(self):
        hash1 = compute_content_hash("test content")
        assert hash1 is not None
        assert len(hash1) == 64  # SHA-256 hex

    def test_empty_content_returns_none(self):
        assert compute_content_hash("") is None
        assert compute_content_hash(None) is None

    def test_deterministic_hash(self):
        hash1 = compute_content_hash("same content")
        hash2 = compute_content_hash("same content")
        assert hash1 == hash2


class TestComputeTimestampProof:

    def test_proof_generation(self):
        proof = compute_timestamp_proof(1234567890.0, "entry_123")
        assert proof is not None
        assert len(proof) == 64

    def test_deterministic_proof(self):
        proof1 = compute_timestamp_proof(1234567890.0, "entry_123")
        proof2 = compute_timestamp_proof(1234567890.0, "entry_123")
        assert proof1 == proof2

    def test_different_timestamp_different_proof(self):
        proof1 = compute_timestamp_proof(1234567890.0, "entry_123")
        proof2 = compute_timestamp_proof(1234567891.0, "entry_123")
        assert proof1 != proof2


# =============================================================================
# MerkleTree Tests
# =============================================================================


class TestMerkleTree:

    def test_empty_tree_has_no_root(self):
        tree = MerkleTree()
        assert tree.get_root() is None

    def test_single_leaf(self):
        tree = MerkleTree()
        idx = tree.add_leaf("data_1")
        assert idx == 0
        root = tree.get_root()
        assert root is not None
        assert len(root) == 64

    def test_two_leaves(self):
        tree = MerkleTree()
        tree.add_leaf("data_1")
        tree.add_leaf("data_2")
        root = tree.get_root()
        assert root is not None

    def test_three_leaves_odd_handling(self):
        """Odd number of leaves handled correctly (last duplicated)."""
        tree = MerkleTree()
        tree.add_leaf("data_1")
        tree.add_leaf("data_2")
        tree.add_leaf("data_3")
        root = tree.get_root()
        assert root is not None

    def test_power_of_two_leaves(self):
        tree = MerkleTree()
        for i in range(8):
            tree.add_leaf(f"data_{i}")
        root = tree.get_root()
        assert root is not None

    def test_root_cached(self):
        tree = MerkleTree()
        tree.add_leaf("data_1")
        tree.add_leaf("data_2")
        root1 = tree.get_root()
        root2 = tree.get_root()
        assert root1 == root2
        assert tree._root is not None

    def test_cache_invalidated_on_add(self):
        tree = MerkleTree()
        tree.add_leaf("data_1")
        root1 = tree.get_root()
        tree.add_leaf("data_2")
        assert tree._root is None  # Cache invalidated
        root2 = tree.get_root()
        assert root1 != root2

    def test_get_proof_invalid_index(self):
        tree = MerkleTree()
        tree.add_leaf("data_1")
        proof = tree.get_proof(-1)
        assert proof == []
        proof = tree.get_proof(100)
        assert proof == []

    def test_get_proof_single_leaf(self):
        tree = MerkleTree()
        tree.add_leaf("data_1")
        proof = tree.get_proof(0)
        # Single leaf has no siblings, but self-pairs
        assert isinstance(proof, list)

    def test_get_proof_two_leaves(self):
        tree = MerkleTree()
        tree.add_leaf("data_1")
        tree.add_leaf("data_2")
        proof0 = tree.get_proof(0)
        proof1 = tree.get_proof(1)
        assert len(proof0) == 1
        assert len(proof1) == 1
        assert proof0[0]["position"] == "right"
        assert proof1[0]["position"] == "left"

    def test_verify_proof_valid(self):
        tree = MerkleTree()
        tree.add_leaf("data_1")
        tree.add_leaf("data_2")
        tree.add_leaf("data_3")
        tree.add_leaf("data_4")

        root = tree.get_root()

        # Get proof for leaf 2
        proof = tree.get_proof(2)
        leaf_hash = tree.leaves[2]

        assert tree.verify_proof(leaf_hash, proof, root)

    def test_verify_proof_invalid_hash(self):
        tree = MerkleTree()
        tree.add_leaf("data_1")
        tree.add_leaf("data_2")
        root = tree.get_root()
        proof = tree.get_proof(0)
        assert not tree.verify_proof("invalid_hash", proof, root)

    def test_deterministic_tree(self):
        tree1 = MerkleTree()
        tree2 = MerkleTree()
        for i in range(5):
            tree1.add_leaf(f"data_{i}")
            tree2.add_leaf(f"data_{i}")
        assert tree1.get_root() == tree2.get_root()


# =============================================================================
# ValueVector Tests
# =============================================================================


class TestValueVector:

    def test_default_values(self):
        vec = ValueVector()
        assert vec.t == 0.0
        assert vec.e == 0.0
        assert vec.n == 0.0
        assert vec.f == 0.0
        assert vec.r == 0.0
        assert vec.s == 0.0
        assert vec.u == 0.0

    def test_custom_values(self):
        vec = ValueVector(t=1.0, e=2.0, n=3.0, f=4.0, r=5.0, s=6.0, u=7.0)
        assert vec.t == 1.0
        assert vec.e == 2.0
        assert vec.n == 3.0
        assert vec.f == 4.0
        assert vec.r == 5.0
        assert vec.s == 6.0
        assert vec.u == 7.0

    def test_total(self):
        vec = ValueVector(t=1.0, e=2.0, n=3.0, f=4.0, r=5.0, s=6.0, u=7.0)
        assert vec.total() == 28.0

    def test_negative_values_rejected(self):
        """Negative values are rejected by Pydantic validation."""
        with pytest.raises(Exception):  # Pydantic validation error
            ValueVector(t=-1.0)

    def test_apply_subjective_weights(self):
        vec = ValueVector(t=10.0, e=10.0, n=10.0)
        weights = {"t": 2.0, "e": 0.5}
        weighted = vec.apply_subjective_weights(weights)
        assert weighted.t == 20.0
        assert weighted.e == 5.0
        assert weighted.n == 10.0  # No weight = 1.0

    def test_adjust_for_supply(self):
        vec = ValueVector(t=10.0, e=10.0, n=10.0)
        supply = {"t": 2.0, "e": 5.0}
        adjusted = vec.adjust_for_supply(supply)
        assert adjusted.t == 5.0
        assert adjusted.e == 2.0
        assert adjusted.n == 10.0  # No factor = 1.0

    def test_model_dump(self):
        vec = ValueVector(t=1.0, e=2.0)
        data = vec.model_dump()
        assert data["t"] == 1.0
        assert data["e"] == 2.0


# =============================================================================
# LedgerEntry Tests
# =============================================================================


class TestLedgerEntry:

    def test_basic_entry(self):
        entry = LedgerEntry(
            intent_id="test_intent",
            value_vector=ValueVector(t=5.0),
        )
        assert entry.intent_id == "test_intent"
        assert entry.value_vector.t == 5.0
        assert entry.status == "active"
        assert entry.id is not None

    def test_auto_generated_id(self):
        entry = LedgerEntry(intent_id="test", value_vector=ValueVector())
        assert entry.id is not None
        assert len(entry.id) == 64

    def test_explicit_id(self):
        entry = LedgerEntry(
            id="custom_id_12345",
            intent_id="test",
            value_vector=ValueVector(),
        )
        assert entry.id == "custom_id_12345"

    def test_parent_id_sync(self):
        entry = LedgerEntry(
            intent_id="test",
            value_vector=ValueVector(),
            parent_id="parent_1",
        )
        assert "parent_1" in entry.parent_ids

    def test_parent_ids_sync(self):
        entry = LedgerEntry(
            intent_id="test",
            value_vector=ValueVector(),
            parent_ids=["parent_1", "parent_2"],
        )
        assert entry.parent_id == "parent_1"

    def test_classification_bounds(self):
        entry = LedgerEntry(
            intent_id="test",
            value_vector=ValueVector(),
            classification=3,
        )
        assert entry.classification == 3

    def test_proof_data_default(self):
        entry = LedgerEntry(intent_id="test", value_vector=ValueVector())
        assert entry.proof is not None
        assert isinstance(entry.proof, ProofData)


# =============================================================================
# ValueLedger Tests
# =============================================================================


class TestValueLedger:

    @pytest.fixture
    def temp_ledger(self):
        """Create a temporary ledger for testing."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        yield ValueLedger(path)
        # Cleanup
        if os.path.exists(path):
            os.remove(path)

    def test_create_empty_ledger(self, temp_ledger):
        assert len(temp_ledger.entries) == 0

    def test_accrue_entry(self, temp_ledger):
        entry_id = temp_ledger.accrue(
            intent_id="test_intent",
            initial_vector={"t": 5.0, "e": 3.0},
        )
        assert entry_id is not None
        assert len(temp_ledger.entries) == 1
        entry = temp_ledger.get_entry(entry_id)
        assert entry.value_vector.t == 5.0
        assert entry.value_vector.e == 3.0

    def test_accrue_with_content_hash(self, temp_ledger):
        entry_id = temp_ledger.accrue(
            intent_id="test_intent",
            initial_vector={"t": 5.0},
            content_for_proof="Test content for hashing",
        )
        entry = temp_ledger.get_entry(entry_id)
        assert entry.proof.content_hash is not None

    def test_accrue_with_owner(self, temp_ledger):
        entry_id = temp_ledger.accrue(
            intent_id="test_intent",
            initial_vector={"t": 5.0},
            owner="user_123",
            classification=2,
        )
        entry = temp_ledger.get_entry(entry_id)
        assert entry.owner == "user_123"
        assert entry.classification == 2

    def test_accrue_invalid_classification(self, temp_ledger):
        with pytest.raises(ValueError, match="Classification must be 0-5"):
            temp_ledger.accrue(
                intent_id="test",
                initial_vector={"t": 5.0},
                classification=10,
            )

    def test_get_entry_not_found(self, temp_ledger):
        assert temp_ledger.get_entry("nonexistent") is None

    def test_merkle_tree_updated(self, temp_ledger):
        temp_ledger.accrue(intent_id="test1", initial_vector={"t": 1.0})
        root1 = temp_ledger.get_merkle_root()
        temp_ledger.accrue(intent_id="test2", initial_vector={"t": 2.0})
        root2 = temp_ledger.get_merkle_root()
        assert root1 != root2

    def test_get_merkle_proof(self, temp_ledger):
        entry_id = temp_ledger.accrue(
            intent_id="test",
            initial_vector={"t": 5.0},
        )
        proof = temp_ledger.get_merkle_proof(entry_id)
        assert proof is not None
        assert "leaf_hash" in proof
        assert "proof" in proof
        assert "root" in proof

    def test_verify_entry_proof(self, temp_ledger):
        entry_id = temp_ledger.accrue(
            intent_id="test",
            initial_vector={"t": 5.0},
        )
        assert temp_ledger.verify_entry_proof(entry_id)

    def test_persistence(self):
        """Entries persist across ledger instances."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        try:
            # Create and populate ledger
            ledger1 = ValueLedger(path)
            entry_id = ledger1.accrue(
                intent_id="persistent_test",
                initial_vector={"t": 10.0},
            )

            # Create new instance from same file
            ledger2 = ValueLedger(path)
            assert len(ledger2.entries) == 1
            entry = ledger2.get_entry(entry_id)
            assert entry.value_vector.t == 10.0
        finally:
            os.remove(path)


class TestValueLedgerRevocation:

    @pytest.fixture
    def temp_ledger(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        yield ValueLedger(path)
        if os.path.exists(path):
            os.remove(path)

    def test_revoke_entry(self, temp_ledger):
        entry_id = temp_ledger.accrue(
            intent_id="test",
            initial_vector={"t": 5.0},
        )
        result = temp_ledger.revoke(entry_id, reason="Test revocation")
        assert result is True
        entry = temp_ledger.get_entry(entry_id)
        assert entry.status == "revoked"
        assert entry.revocation_reason == "Test revocation"
        assert entry.revoked_at is not None

    def test_revoke_nonexistent_entry(self, temp_ledger):
        result = temp_ledger.revoke("nonexistent")
        assert result is False

    def test_revoke_already_revoked(self, temp_ledger):
        entry_id = temp_ledger.accrue(
            intent_id="test",
            initial_vector={"t": 5.0},
        )
        temp_ledger.revoke(entry_id)
        result = temp_ledger.revoke(entry_id)
        assert result is True

    def test_revoke_with_children(self, temp_ledger):
        parent_id = temp_ledger.accrue(
            intent_id="parent",
            initial_vector={"t": 5.0},
        )
        child_id = temp_ledger.aggregate_correction(
            parent_id=parent_id,
            new_vector=ValueVector(t=6.0),
            notes={"reason": "correction"},
        )
        temp_ledger.revoke(parent_id, revoke_children=True)

        parent = temp_ledger.get_entry(parent_id)
        child = temp_ledger.get_entry(child_id)
        assert parent.status == "revoked"
        assert child.status == "revoked"


class TestValueLedgerAggregation:

    @pytest.fixture
    def temp_ledger(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        yield ValueLedger(path)
        if os.path.exists(path):
            os.remove(path)

    def test_aggregate_sum(self, temp_ledger):
        id1 = temp_ledger.accrue(intent_id="t1", initial_vector={"t": 5.0, "e": 3.0})
        id2 = temp_ledger.accrue(intent_id="t2", initial_vector={"t": 3.0, "e": 2.0})

        agg_id = temp_ledger.aggregate_entries([id1, id2], rule="sum")
        agg = temp_ledger.get_entry(agg_id)

        assert agg.value_vector.t == 8.0
        assert agg.value_vector.e == 5.0
        assert agg.aggregation_rule == "sum"

    def test_aggregate_max(self, temp_ledger):
        id1 = temp_ledger.accrue(intent_id="t1", initial_vector={"t": 5.0, "e": 3.0})
        id2 = temp_ledger.accrue(intent_id="t2", initial_vector={"t": 3.0, "e": 7.0})

        agg_id = temp_ledger.aggregate_entries([id1, id2], rule="max")
        agg = temp_ledger.get_entry(agg_id)

        assert agg.value_vector.t == 5.0
        assert agg.value_vector.e == 7.0
        assert agg.aggregation_rule == "max"

    def test_aggregate_weighted(self, temp_ledger):
        id1 = temp_ledger.accrue(intent_id="t1", initial_vector={"t": 10.0})
        id2 = temp_ledger.accrue(intent_id="t2", initial_vector={"t": 20.0})

        agg_id = temp_ledger.aggregate_entries(
            [id1, id2],
            rule="weighted",
            weights={id1: 1.0, id2: 3.0},  # 25% and 75%
        )
        agg = temp_ledger.get_entry(agg_id)

        # (10 * 0.25) + (20 * 0.75) = 2.5 + 15 = 17.5
        assert abs(agg.value_vector.t - 17.5) < 0.01

    def test_aggregate_invalid_rule(self, temp_ledger):
        id1 = temp_ledger.accrue(intent_id="t1", initial_vector={"t": 5.0})

        with pytest.raises(ValueError, match="Invalid aggregation rule"):
            temp_ledger.aggregate_entries([id1], rule="invalid")

    def test_aggregate_empty_list(self, temp_ledger):
        with pytest.raises(ValueError, match="at least one entry"):
            temp_ledger.aggregate_entries([], rule="sum")

    def test_aggregate_revoked_entry(self, temp_ledger):
        id1 = temp_ledger.accrue(intent_id="t1", initial_vector={"t": 5.0})
        temp_ledger.revoke(id1)

        with pytest.raises(ValueError, match="revoked"):
            temp_ledger.aggregate_entries([id1], rule="sum")

    def test_aggregate_inherits_classification(self, temp_ledger):
        id1 = temp_ledger.accrue(
            intent_id="t1",
            initial_vector={"t": 5.0},
            classification=1,
        )
        id2 = temp_ledger.accrue(
            intent_id="t2",
            initial_vector={"t": 3.0},
            classification=3,
        )

        agg_id = temp_ledger.aggregate_entries([id1, id2], rule="sum")
        agg = temp_ledger.get_entry(agg_id)

        assert agg.classification == 3


class TestValueLedgerAccessControl:

    @pytest.fixture
    def temp_ledger(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        yield ValueLedger(path)
        if os.path.exists(path):
            os.remove(path)

    def test_check_access_owner(self, temp_ledger):
        entry_id = temp_ledger.accrue(
            intent_id="test",
            initial_vector={"t": 5.0},
            owner="user_1",
            classification=5,
        )
        assert temp_ledger.check_access(entry_id, requester_owner="user_1")

    def test_check_access_clearance(self, temp_ledger):
        entry_id = temp_ledger.accrue(
            intent_id="test",
            initial_vector={"t": 5.0},
            owner="user_1",
            classification=3,
        )
        assert not temp_ledger.check_access(
            entry_id, requester_owner="user_2", requester_clearance=2
        )
        assert temp_ledger.check_access(
            entry_id, requester_owner="user_2", requester_clearance=3
        )

    def test_transfer_ownership(self, temp_ledger):
        entry_id = temp_ledger.accrue(
            intent_id="test",
            initial_vector={"t": 5.0},
            owner="user_1",
        )
        result = temp_ledger.transfer_ownership(
            entry_id, new_owner="user_2", current_owner="user_1"
        )
        assert result is True
        entry = temp_ledger.get_entry(entry_id)
        assert entry.owner == "user_2"

    def test_transfer_ownership_wrong_owner(self, temp_ledger):
        entry_id = temp_ledger.accrue(
            intent_id="test",
            initial_vector={"t": 5.0},
            owner="user_1",
        )
        result = temp_ledger.transfer_ownership(
            entry_id, new_owner="user_3", current_owner="user_2"
        )
        assert result is False

    def test_cannot_transfer_revoked(self, temp_ledger):
        entry_id = temp_ledger.accrue(
            intent_id="test",
            initial_vector={"t": 5.0},
            owner="user_1",
        )
        temp_ledger.revoke(entry_id)
        result = temp_ledger.transfer_ownership(entry_id, new_owner="user_2")
        assert result is False


# =============================================================================
# ClockMonitor Tests
# =============================================================================


class TestClockMonitor:

    def test_valid_timestamp(self):
        monitor = ClockMonitor()
        result = monitor.check_timestamp(time.time())
        assert result["valid"]
        assert len(result["issues"]) == 0

    def test_future_timestamp(self):
        monitor = ClockMonitor(max_future_seconds=10)
        future_time = time.time() + 100
        result = monitor.check_timestamp(future_time)
        assert not result["valid"]
        assert any(i["type"] == "future_timestamp" for i in result["issues"])

    def test_clock_regression(self):
        monitor = ClockMonitor(max_drift_seconds=60)
        # Set a known last time
        monitor.last_known_time = time.time()
        # Check a much earlier time
        past_time = time.time() - 1000
        result = monitor.check_timestamp(past_time)
        assert any(i["type"] == "clock_regression" for i in result["issues"])

    def test_drift_history(self):
        monitor = ClockMonitor(max_future_seconds=10)
        monitor.check_timestamp(time.time() + 100)
        history = monitor.get_drift_history()
        assert len(history) == 1

    def test_reset(self):
        monitor = ClockMonitor()
        monitor.check_timestamp(time.time() + 1000)
        monitor.reset()
        assert monitor.last_known_time is None
        assert len(monitor.drift_events) == 0


# =============================================================================
# SourceValidator Tests
# =============================================================================


class TestSourceValidator:

    def test_validate_valid_entry(self):
        validator = SourceValidator()
        entry = LedgerEntry(
            intent_id="test",
            value_vector=ValueVector(t=5.0),
        )
        result = validator.validate_entry(entry)
        assert result["valid"]

    def test_validate_missing_intent_id(self):
        """Empty intent_id is rejected at model validation."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError, match="string_too_short"):
            LedgerEntry(
                intent_id="",
                value_vector=ValueVector(t=5.0),
            )

    def test_validate_aggregation(self):
        validator = SourceValidator()
        entries = [
            LedgerEntry(intent_id="t1", value_vector=ValueVector(t=5.0), classification=2),
            LedgerEntry(intent_id="t2", value_vector=ValueVector(t=3.0), classification=4),
        ]
        result = validator.validate_aggregation(entries, requester_clearance=5)
        assert result["valid"]
        assert result["max_classification"] == 4

    def test_validate_aggregation_insufficient_clearance(self):
        validator = SourceValidator()
        entries = [
            LedgerEntry(intent_id="t1", value_vector=ValueVector(t=5.0), classification=5),
        ]
        result = validator.validate_aggregation(entries, requester_clearance=2)
        assert not result["valid"]
        assert any("insufficient_clearance" in e.get("type", "") for e in result["errors"])


# =============================================================================
# FailureModeHandler Tests
# =============================================================================


class TestFailureModeHandler:

    @pytest.fixture
    def handler(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        ledger = ValueLedger(path)
        handler = FailureModeHandler(ledger)
        yield handler
        if os.path.exists(path):
            os.remove(path)

    def test_safe_accrue_success(self, handler):
        result = handler.safe_accrue(
            intent_id="test",
            initial_vector={"t": 5.0},
        )
        assert result["success"]
        assert result["entry_id"] is not None

    def test_safe_accrue_handles_exceptions(self, handler):
        # Pass an invalid keyword argument to trigger an exception
        result = handler.safe_accrue(
            intent_id="test",
            initial_vector={"t": 5.0},
            invalid_kwarg="should_fail",
        )
        # Should return success=False with error info
        assert result["success"] is False
        assert "error" in result
        assert result["entry_id"] is None

    def test_health_report(self, handler):
        report = handler.get_health_report()
        assert "clock_health" in report
        assert "validation_health" in report
        assert "overall_status" in report

    def test_health_report_healthy(self, handler):
        report = handler.get_health_report()
        assert report["overall_status"] == "healthy"


# =============================================================================
# Parametrized Boundary Value Tests
# =============================================================================


class TestValueVectorBoundaries:

    @pytest.mark.parametrize("field,value,should_pass", [
        ("t", 0.0, True),
        ("t", -0.1, False),
        ("t", 1000.0, True),
        ("e", 0.0, True),
        ("e", -1.0, False),
        ("n", -1.0, False),
        ("n", 0.0, True),
        ("f", 0.0, True),
        ("f", -0.5, False),
        ("r", 0.0, True),
        ("r", -0.01, False),
        ("s", 0.0, True),
        ("s", -999.0, False),
        ("u", 0.0, True),
        ("u", -0.1, False),
    ])
    def test_field_ge_constraint(self, field, value, should_pass):
        if should_pass:
            vv = ValueVector(**{field: value})
            assert getattr(vv, field) == value
        else:
            with pytest.raises(Exception):
                ValueVector(**{field: value})


class TestClassificationBoundaries:

    @pytest.mark.parametrize("classification,valid", [
        (0, True), (1, True), (3, True), (5, True),
        (-1, False), (6, False), (100, False),
    ])
    def test_classification_validation(self, classification, valid):
        if valid:
            entry = LedgerEntry(
                intent_id="test",
                value_vector=ValueVector(),
                classification=classification,
            )
            assert entry.classification == classification
        else:
            with pytest.raises(Exception):
                LedgerEntry(
                    intent_id="test",
                    value_vector=ValueVector(),
                    classification=classification,
                )


class TestAggregationRuleBoundaries:

    @pytest.mark.parametrize("rule", ["sum", "max", "weighted"])
    def test_valid_aggregation_rules(self, rule):
        entry = LedgerEntry(
            intent_id="test",
            value_vector=ValueVector(),
            aggregation_rule=rule,
        )
        assert entry.aggregation_rule == rule

    @pytest.mark.parametrize("rule", ["average", "median", "invalid"])
    def test_invalid_aggregation_rules(self, rule):
        with pytest.raises(Exception):
            LedgerEntry(
                intent_id="test",
                value_vector=ValueVector(),
                aggregation_rule=rule,
            )


class TestMerkleTreeSizes:

    @pytest.mark.parametrize("num_leaves", [1, 2, 3, 4, 7, 8, 15, 16, 100])
    def test_merkle_tree_various_sizes(self, num_leaves):
        tree = MerkleTree()
        for i in range(num_leaves):
            tree.add_leaf(f"leaf_{i}")

        root = tree.get_root()
        assert root is not None
        assert len(root) == 64  # SHA-256 hex length

        # Verify proof for each leaf
        for i in range(num_leaves):
            proof = tree.get_proof(i)
            leaf_hash = tree.leaves[i]
            assert tree.verify_proof(leaf_hash, proof, root)


class TestPathTraversalPrevention:

    @pytest.mark.parametrize("malicious_path", [
        "/etc/passwd",
        "../../../etc/shadow",
        "/proc/self/environ",
        "/dev/null",
        "path\x00injection",
    ])
    def test_path_traversal_blocked(self, malicious_path):
        with pytest.raises(ValueError):
            _validate_ledger_path(malicious_path)
