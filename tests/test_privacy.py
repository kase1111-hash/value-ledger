# tests/test_privacy.py
"""
Tests for MP-02 Privacy & Agency Controls.

Per MP-02 §12:
- Raw signals MAY be encrypted or access-controlled
- Receipts MUST not expose raw content by default
- Humans MAY revoke future observation
- Past receipts remain immutable
"""

import pytest
import time
import tempfile
import os
from pathlib import Path

from value_ledger.privacy import (
    PrivacyLevel,
    ConsentStatus,
    RevocationScope,
    ObservationConsent,
    EncryptedContent,
    SignalEncryptor,
    PrivacyPolicy,
    PrivacyFilter,
    ConsentRegistry,
    AgencyController,
    create_privacy_controller,
    encrypt_signal_content,
    decrypt_signal_content,
    CRYPTO_AVAILABLE,
)


# ============================================================================
# ObservationConsent Tests
# ============================================================================

class TestObservationConsent:
    """Tests for observation consent management."""

    def test_consent_creation_defaults_to_pending(self):
        consent = ObservationConsent(human_id="human-001")
        assert consent.status == ConsentStatus.PENDING
        assert consent.human_id == "human-001"
        assert not consent.is_valid()

    def test_grant_consent(self):
        consent = ObservationConsent(human_id="human-001")
        consent.grant()
        assert consent.status == ConsentStatus.GRANTED
        assert consent.granted_at is not None
        assert consent.is_valid()

    def test_grant_consent_with_expiry(self):
        consent = ObservationConsent(human_id="human-001")
        consent.grant(expires_in_seconds=3600)
        assert consent.is_valid()
        assert consent.expires_at is not None
        assert consent.expires_at > time.time()

    def test_expired_consent_is_not_valid(self):
        consent = ObservationConsent(human_id="human-001")
        consent.grant(expires_in_seconds=0.01)
        time.sleep(0.02)
        assert not consent.is_valid()

    def test_revoke_consent(self):
        consent = ObservationConsent(human_id="human-001")
        consent.grant()
        assert consent.is_valid()

        consent.revoke(reason="Privacy concerns")
        assert consent.status == ConsentStatus.REVOKED
        assert consent.revoked_at is not None
        assert consent.revocation_reason == "Privacy concerns"
        assert not consent.is_valid()


# ============================================================================
# SignalEncryptor Tests
# ============================================================================

@pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
class TestSignalEncryptor:
    """Tests for signal encryption/decryption."""

    def test_encrypt_and_decrypt(self):
        encryptor = SignalEncryptor()
        content = "This is secret signal content"
        password = "test-password-123"

        encrypted = encryptor.encrypt(content, password)
        assert encrypted.ciphertext != content.encode()
        assert encrypted.salt is not None
        assert encrypted.algorithm == "fernet"

        decrypted = encryptor.decrypt(encrypted, password)
        assert decrypted == content

    def test_decrypt_with_wrong_password_fails(self):
        encryptor = SignalEncryptor()
        content = "Secret content"
        password = "correct-password"

        encrypted = encryptor.encrypt(content, password)

        with pytest.raises(ValueError, match="Decryption failed"):
            encryptor.decrypt(encrypted, "wrong-password")

    def test_encrypted_content_serialization(self):
        encryptor = SignalEncryptor()
        content = "Test content"
        password = "password"

        encrypted = encryptor.encrypt(content, password)
        serialized = encrypted.to_dict()

        assert "ciphertext" in serialized
        assert "salt" in serialized
        assert serialized["algorithm"] == "fernet"

        restored = EncryptedContent.from_dict(serialized)
        decrypted = encryptor.decrypt(restored, password)
        assert decrypted == content


class TestSignalEncryptorNoLibrary:
    """Tests for graceful degradation when cryptography is not available."""

    def test_convenience_functions_exist(self):
        # These should be importable even without cryptography
        from value_ledger.privacy import encrypt_signal_content, decrypt_signal_content
        assert callable(encrypt_signal_content)
        assert callable(decrypt_signal_content)


# ============================================================================
# PrivacyPolicy Tests
# ============================================================================

class TestPrivacyPolicy:
    """Tests for privacy policy configuration."""

    def test_default_policy(self):
        policy = PrivacyPolicy()
        assert policy.default_level == PrivacyLevel.HASH_ONLY
        assert policy.allow_content_in_receipts is False
        assert policy.allow_summary_generation is True

    def test_can_access_no_restrictions(self):
        policy = PrivacyPolicy()
        assert policy.can_access("any-reader")

    def test_can_access_with_authorized_readers(self):
        policy = PrivacyPolicy(authorized_readers={"reader-1", "reader-2"})
        assert policy.can_access("reader-1")
        assert policy.can_access("reader-2")
        assert not policy.can_access("reader-3")


# ============================================================================
# PrivacyFilter Tests
# ============================================================================

class TestPrivacyFilter:
    """Tests for privacy filtering of signals and receipts."""

    def test_filter_signal_removes_content_by_default(self):
        filter_ = PrivacyFilter()
        signal_data = {
            "signal_type": "text",
            "timestamp": 1000.0,
            "hash": "abc123",
            "content": "This should be removed",
        }

        filtered = filter_.filter_signal(signal_data)
        assert "content" not in filtered
        assert filtered["hash"] == "abc123"
        assert filtered["privacy_level"] == PrivacyLevel.HASH_ONLY

    def test_filter_signal_unauthorized_reader(self):
        policy = PrivacyPolicy(authorized_readers={"reader-1"})
        filter_ = PrivacyFilter(policy=policy)

        signal_data = {"content": "secret", "hash": "abc"}
        filtered = filter_.filter_signal(signal_data, reader_id="reader-2")

        assert "content" not in filtered
        assert filtered["privacy_filtered"] is True

    def test_filter_receipt_removes_raw_content(self):
        """Per MP-02 §12: Receipts MUST not expose raw content by default."""
        filter_ = PrivacyFilter()
        receipt_data = {
            "receipt_id": "receipt-001",
            "time_bounds": [1000.0, 2000.0],
            "signal_hashes": ["abc", "def"],
            "raw_content": "This should NOT be exposed",
            "signal_contents": ["signal1", "signal2"],
            "validation_metadata": {
                "validator_id": "v1",
                "raw_assessment": "Contains sensitive excerpts",
            },
        }

        filtered = filter_.filter_receipt(receipt_data)

        assert "raw_content" not in filtered
        assert "signal_contents" not in filtered
        assert "raw_assessment" not in filtered["validation_metadata"]
        assert filtered["privacy_filtered"] is True
        assert filtered["receipt_id"] == "receipt-001"

    def test_redact_patterns(self):
        policy = PrivacyPolicy(redact_patterns=[r"\b\d{3}-\d{2}-\d{4}\b"])  # SSN pattern
        filter_ = PrivacyFilter(policy=policy)

        text = "My SSN is 123-45-6789 and my name is John"
        redacted = filter_.redact_patterns(text)

        assert "123-45-6789" not in redacted
        assert "[REDACTED]" in redacted
        assert "John" in redacted


# ============================================================================
# ConsentRegistry Tests
# ============================================================================

class TestConsentRegistry:
    """Tests for consent registry management."""

    def test_grant_consent(self):
        registry = ConsentRegistry()
        consent = registry.grant_consent("human-001")

        assert consent.status == ConsentStatus.GRANTED
        assert consent.is_valid()

    def test_revoke_consent(self):
        registry = ConsentRegistry()
        registry.grant_consent("human-001")

        consent = registry.revoke_consent("human-001", reason="Test revocation")
        assert consent.status == ConsentStatus.REVOKED
        assert consent.revocation_reason == "Test revocation"

    def test_is_observation_allowed_no_consent(self):
        registry = ConsentRegistry()
        assert not registry.is_observation_allowed("unknown-human")

    def test_is_observation_allowed_with_consent(self):
        registry = ConsentRegistry()
        registry.grant_consent("human-001")
        assert registry.is_observation_allowed("human-001")

    def test_is_observation_allowed_revoked(self):
        registry = ConsentRegistry()
        registry.grant_consent("human-001")
        registry.revoke_consent("human-001")
        assert not registry.is_observation_allowed("human-001")

    def test_observer_restrictions(self):
        registry = ConsentRegistry()
        registry.grant_consent(
            "human-001",
            allowed_observers=["observer-1", "observer-2"],
        )

        assert registry.is_observation_allowed("human-001", observer_id="observer-1")
        assert not registry.is_observation_allowed("human-001", observer_id="observer-3")

    def test_signal_type_restrictions(self):
        registry = ConsentRegistry()
        registry.grant_consent(
            "human-001",
            allowed_signal_types=["text", "command"],
        )

        assert registry.is_observation_allowed("human-001", signal_type="text")
        assert not registry.is_observation_allowed("human-001", signal_type="voice")

    def test_list_active_and_revoked(self):
        registry = ConsentRegistry()
        registry.grant_consent("human-001")
        registry.grant_consent("human-002")
        registry.revoke_consent("human-003")

        active = registry.list_active()
        revoked = registry.list_revoked()

        assert len(active) == 2
        assert len(revoked) == 1
        assert revoked[0].human_id == "human-003"

    def test_persistence_to_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage_path = os.path.join(tmpdir, "consent.json")

            # Create and save
            registry1 = ConsentRegistry(storage_path=storage_path)
            registry1.grant_consent("human-001")
            registry1.revoke_consent("human-002", reason="Test")

            # Load in new registry
            registry2 = ConsentRegistry(storage_path=storage_path)

            consent1 = registry2.get_consent("human-001")
            consent2 = registry2.get_consent("human-002")

            assert consent1 is not None
            assert consent1.status == ConsentStatus.GRANTED

            assert consent2 is not None
            assert consent2.status == ConsentStatus.REVOKED


# ============================================================================
# AgencyController Tests
# ============================================================================

class TestAgencyController:
    """Tests for agency controller (human control over observation)."""

    def test_revoke_future_observation(self):
        """Per MP-02 §12: Humans MAY revoke future observation."""
        controller = AgencyController()

        result = controller.revoke_future_observation(
            human_id="human-001",
            reason="No longer want observation",
        )

        assert result["human_id"] == "human-001"
        assert result["future_observation_blocked"] is True
        assert result["past_receipts_affected"] is False  # Per MP-02

    def test_revoke_callback(self):
        controller = AgencyController()
        callback_results = []

        def on_revoke(result):
            callback_results.append(result)

        controller.on_revocation(on_revoke)
        controller.revoke_future_observation("human-001")

        assert len(callback_results) == 1
        assert callback_results[0]["human_id"] == "human-001"

    def test_check_observation_permission_allowed(self):
        controller = AgencyController()
        controller.consent_registry.grant_consent("human-001")

        result = controller.check_observation_permission(
            human_id="human-001",
            observer_id="observer-001",
        )

        assert result["allowed"] is True

    def test_check_observation_permission_denied(self):
        controller = AgencyController()
        controller.consent_registry.revoke_consent("human-001")

        result = controller.check_observation_permission(
            human_id="human-001",
            observer_id="observer-001",
        )

        assert result["allowed"] is False
        assert result["reason"] == "consent_revoked"

    def test_apply_privacy_to_receipt(self):
        """Per MP-02 §12: Receipts MUST not expose raw content by default."""
        controller = AgencyController()

        receipt_data = {
            "receipt_id": "r-001",
            "raw_content": "Should be removed",
            "signal_hashes": ["hash1", "hash2"],
        }

        filtered = controller.apply_privacy_to_receipt(receipt_data)

        assert "raw_content" not in filtered
        assert filtered["signal_hashes"] == ["hash1", "hash2"]
        assert filtered["privacy_filtered"] is True


# ============================================================================
# Integration Tests
# ============================================================================

class TestPrivacyIntegration:
    """Integration tests for privacy controls."""

    def test_full_privacy_workflow(self):
        """Test complete privacy workflow: consent -> observe -> revoke."""
        controller = create_privacy_controller()

        # Step 1: Grant consent
        consent = controller.consent_registry.grant_consent(
            "human-001",
            allowed_observers=["obs-1"],
            allowed_signal_types=["text"],
        )
        assert consent.is_valid()

        # Step 2: Check permission (should be allowed)
        perm = controller.check_observation_permission(
            human_id="human-001",
            observer_id="obs-1",
            signal_type="text",
        )
        assert perm["allowed"]

        # Step 3: Revoke consent
        result = controller.revoke_future_observation(
            human_id="human-001",
            reason="Testing revocation",
        )
        assert result["future_observation_blocked"]

        # Step 4: Check permission again (should be denied)
        perm = controller.check_observation_permission(
            human_id="human-001",
            observer_id="obs-1",
        )
        assert not perm["allowed"]
        assert perm["reason"] == "consent_revoked"

    @pytest.mark.skipif(not CRYPTO_AVAILABLE, reason="cryptography library not available")
    def test_encrypt_decrypt_workflow(self):
        """Test signal encryption and decryption."""
        content = "Sensitive signal content that should be protected"
        password = "secure-password-123"

        # Encrypt
        encrypted = encrypt_signal_content(content, password)
        assert encrypted.ciphertext != content.encode()

        # Decrypt
        decrypted = decrypt_signal_content(encrypted, password)
        assert decrypted == content

    def test_receipt_privacy_preserves_proof_data(self):
        """Ensure privacy filtering preserves cryptographic proof data."""
        policy = PrivacyPolicy()
        filter_ = PrivacyFilter(policy=policy)

        receipt_data = {
            "receipt_id": "receipt-001",
            "receipt_hash": "abc123def456",
            "signal_hashes": ["sig1", "sig2", "sig3"],
            "merkle_ref": "merkle-root-hash",
            "raw_content": "This should be removed",
            "effort_summary": "Summary is allowed",
        }

        filtered = filter_.filter_receipt(receipt_data)

        # Proof data preserved
        assert filtered["receipt_hash"] == "abc123def456"
        assert filtered["signal_hashes"] == ["sig1", "sig2", "sig3"]
        assert filtered["merkle_ref"] == "merkle-root-hash"
        assert filtered["effort_summary"] == "Summary is allowed"

        # Raw content removed
        assert "raw_content" not in filtered
