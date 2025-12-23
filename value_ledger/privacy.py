# value_ledger/privacy.py
"""
MP-02 Privacy & Agency Controls Implementation.

Per MP-02 §12 Privacy and Agency:
- Raw signals MAY be encrypted or access-controlled
- Receipts MUST not expose raw content by default
- Humans MAY revoke future observation
- Past receipts remain immutable

This module provides:
- Signal encryption and access control
- Consent management for observation
- Privacy filtering for receipts
- Agency controls for revoking future observation
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from pathlib import Path

# base64 is always available
import base64

logger = logging.getLogger(__name__)

# Conditional import for cryptography (graceful degradation)
# We need to be extra careful here because cryptography can fail with pyo3 panics
CRYPTO_AVAILABLE = False
_Fernet = None
_hashes = None
_PBKDF2HMAC = None


def _try_import_cryptography():
    """Attempt to import cryptography, returning True if successful."""
    global CRYPTO_AVAILABLE, _Fernet, _hashes, _PBKDF2HMAC
    try:
        # Try importing with a subprocess check first to avoid crashing
        import importlib.util
        spec = importlib.util.find_spec("cryptography")
        if spec is None:
            return False

        # Try the actual imports
        from cryptography.fernet import Fernet
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

        _Fernet = Fernet
        _hashes = hashes
        _PBKDF2HMAC = PBKDF2HMAC
        CRYPTO_AVAILABLE = True
        return True
    except Exception as e:
        logger.debug(f"Cryptography import failed: {e}")
        return False


# Don't auto-import cryptography at module load - it can cause panics
# Instead, lazy-load when encryption is actually needed
def _ensure_crypto():
    """Ensure cryptography is loaded, raising error if not available."""
    global CRYPTO_AVAILABLE
    if not CRYPTO_AVAILABLE and _Fernet is None:
        _try_import_cryptography()
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("Cryptography library required but not available")


def _validate_storage_path(path: str, base_dir: Optional[str] = None) -> Path:
    """
    Validate storage path to prevent path traversal attacks.

    Args:
        path: The path to validate
        base_dir: Optional base directory to restrict paths within

    Returns:
        Resolved Path object

    Raises:
        ValueError: If path is invalid or attempts traversal
    """
    if not path:
        raise ValueError("Storage path cannot be empty")

    # Convert to Path and resolve to absolute
    resolved = Path(path).resolve()

    # Check for null bytes (common attack vector)
    if "\x00" in str(path):
        raise ValueError("Invalid characters in path")

    # If base_dir specified, ensure path is within it
    if base_dir:
        base_resolved = Path(base_dir).resolve()
        try:
            resolved.relative_to(base_resolved)
        except ValueError:
            raise ValueError(f"Path must be within {base_dir}")

    # Block common sensitive paths
    sensitive_patterns = [
        "/etc/", "/proc/", "/sys/", "/dev/",
        "/.ssh/", "/.aws/", "/.config/",
        "/passwd", "/shadow", "/id_rsa",
    ]
    path_str = str(resolved).lower()
    for pattern in sensitive_patterns:
        if pattern in path_str:
            raise ValueError(f"Access to sensitive path not allowed: {pattern}")

    logger.debug(f"Path validated: {resolved}")
    return resolved


class PrivacyLevel(str, Enum):
    """Privacy levels for signals and receipts."""
    PUBLIC = "public"  # No restrictions
    HASH_ONLY = "hash_only"  # Only hash exposed, content hidden
    ENCRYPTED = "encrypted"  # Content encrypted, key required
    REDACTED = "redacted"  # Content permanently removed
    ACCESS_CONTROLLED = "access_controlled"  # Requires authorization


class ConsentStatus(str, Enum):
    """Status of observation consent."""
    GRANTED = "granted"
    REVOKED = "revoked"
    PENDING = "pending"
    EXPIRED = "expired"


class RevocationScope(str, Enum):
    """Scope of observation revocation."""
    ALL = "all"  # Revoke all future observation
    SESSION = "session"  # Revoke for current session only
    SIGNAL_TYPE = "signal_type"  # Revoke specific signal types
    OBSERVER = "observer"  # Revoke specific observer


@dataclass
class ObservationConsent:
    """
    Tracks human consent for observation.

    Per MP-02 §12: Humans MAY revoke future observation.
    """
    human_id: str
    status: str = ConsentStatus.PENDING
    granted_at: Optional[float] = None
    revoked_at: Optional[float] = None
    expires_at: Optional[float] = None
    scope: str = "all"  # What observation is consented to
    allowed_observers: List[str] = field(default_factory=list)
    allowed_signal_types: List[str] = field(default_factory=list)
    revocation_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def is_valid(self) -> bool:
        """Check if consent is currently valid."""
        if self.status != ConsentStatus.GRANTED:
            return False
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True

    def grant(self, expires_in_seconds: Optional[float] = None) -> None:
        """Grant observation consent."""
        self.status = ConsentStatus.GRANTED
        self.granted_at = time.time()
        self.revoked_at = None
        if expires_in_seconds:
            self.expires_at = time.time() + expires_in_seconds
        logger.info(f"Observation consent granted for human {self.human_id}")

    def revoke(self, reason: Optional[str] = None) -> None:
        """
        Revoke future observation.

        Per MP-02 §12: Past receipts remain immutable.
        This only affects FUTURE observation.
        """
        self.status = ConsentStatus.REVOKED
        self.revoked_at = time.time()
        self.revocation_reason = reason
        logger.info(f"Observation consent revoked for human {self.human_id}: {reason}")


@dataclass
class EncryptedContent:
    """
    Wrapper for encrypted signal content.

    Per MP-02 §12: Raw signals MAY be encrypted.
    """
    ciphertext: bytes
    salt: bytes
    nonce: Optional[bytes] = None
    algorithm: str = "fernet"
    key_hint: Optional[str] = None  # Hint for key derivation (not the key itself)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for storage."""
        return {
            "ciphertext": base64.b64encode(self.ciphertext).decode() if CRYPTO_AVAILABLE else "",
            "salt": base64.b64encode(self.salt).decode() if CRYPTO_AVAILABLE else "",
            "nonce": base64.b64encode(self.nonce).decode() if self.nonce and CRYPTO_AVAILABLE else None,
            "algorithm": self.algorithm,
            "key_hint": self.key_hint,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EncryptedContent":
        """Deserialize from storage."""
        if not CRYPTO_AVAILABLE:
            return cls(ciphertext=b"", salt=b"", algorithm=data.get("algorithm", "fernet"))
        return cls(
            ciphertext=base64.b64decode(data["ciphertext"]),
            salt=base64.b64decode(data["salt"]),
            nonce=base64.b64decode(data["nonce"]) if data.get("nonce") else None,
            algorithm=data.get("algorithm", "fernet"),
            key_hint=data.get("key_hint"),
        )


class SignalEncryptor:
    """
    Handles encryption/decryption of signal content.

    Per MP-02 §12: Raw signals MAY be encrypted or access-controlled.
    """

    def __init__(self):
        pass  # Lazy load cryptography when needed

    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password."""
        _ensure_crypto()

        kdf = _PBKDF2HMAC(
            algorithm=_hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))

    def encrypt(self, content: str, password: str) -> EncryptedContent:
        """
        Encrypt signal content.

        Args:
            content: Raw signal content to encrypt
            password: Encryption password

        Returns:
            EncryptedContent with ciphertext and salt
        """
        _ensure_crypto()

        salt = os.urandom(16)
        key = self._derive_key(password, salt)
        fernet = _Fernet(key)
        ciphertext = fernet.encrypt(content.encode())

        logger.debug(f"Encrypted content of length {len(content)}")

        return EncryptedContent(
            ciphertext=ciphertext,
            salt=salt,
            algorithm="fernet",
            key_hint=hashlib.sha256(password.encode()).hexdigest()[:8],
        )

    def decrypt(self, encrypted: EncryptedContent, password: str) -> str:
        """
        Decrypt signal content.

        Args:
            encrypted: EncryptedContent to decrypt
            password: Decryption password

        Returns:
            Decrypted content string

        Raises:
            ValueError: If decryption fails (wrong password)
        """
        _ensure_crypto()

        key = self._derive_key(password, encrypted.salt)
        fernet = _Fernet(key)

        try:
            plaintext = fernet.decrypt(encrypted.ciphertext)
            logger.debug("Successfully decrypted content")
            return plaintext.decode()
        except Exception as e:
            logger.warning(f"Decryption failed: {e}")
            raise ValueError("Decryption failed - invalid password or corrupted data")


@dataclass
class PrivacyPolicy:
    """
    Privacy policy for signals and receipts.

    Defines what information can be exposed and to whom.
    """
    default_level: str = PrivacyLevel.HASH_ONLY
    allow_content_in_receipts: bool = False  # Per MP-02: Receipts MUST not expose raw content by default
    allow_summary_generation: bool = True
    redact_patterns: List[str] = field(default_factory=list)  # Regex patterns to redact
    authorized_readers: Set[str] = field(default_factory=set)
    encrypt_at_rest: bool = True
    retention_days: Optional[int] = None  # Auto-delete after N days

    def can_access(self, reader_id: str) -> bool:
        """Check if reader is authorized to access content."""
        if not self.authorized_readers:
            return True  # No restrictions
        return reader_id in self.authorized_readers


class PrivacyFilter:
    """
    Filters and redacts content from receipts and signals.

    Per MP-02 §12: Receipts MUST not expose raw content by default.
    """

    def __init__(self, policy: Optional[PrivacyPolicy] = None):
        self.policy = policy or PrivacyPolicy()
        self._redaction_marker = "[REDACTED]"

    def filter_signal(self, signal_data: Dict[str, Any], reader_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Filter signal data based on privacy policy.

        Args:
            signal_data: Raw signal data dictionary
            reader_id: ID of the reader requesting access

        Returns:
            Filtered signal data with content removed/redacted as needed
        """
        filtered = signal_data.copy()

        # Check access authorization
        if reader_id and not self.policy.can_access(reader_id):
            logger.info(f"Reader {reader_id} not authorized, applying strict filtering")
            filtered.pop("content", None)
            filtered.pop("raw_data", None)
            filtered["privacy_filtered"] = True
            return filtered

        # Apply privacy level
        if self.policy.default_level == PrivacyLevel.HASH_ONLY:
            filtered.pop("content", None)
            filtered.pop("raw_data", None)

        elif self.policy.default_level == PrivacyLevel.REDACTED:
            if "content" in filtered:
                filtered["content"] = self._redaction_marker

        elif self.policy.default_level == PrivacyLevel.ENCRYPTED:
            # Content should already be encrypted; remove plaintext
            filtered.pop("content", None)

        filtered["privacy_level"] = self.policy.default_level
        return filtered

    def filter_receipt(self, receipt_data: Dict[str, Any], reader_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Filter receipt data to not expose raw content.

        Per MP-02 §12: Receipts MUST not expose raw content by default.

        Args:
            receipt_data: Receipt dictionary
            reader_id: ID of the reader requesting access

        Returns:
            Filtered receipt safe for external sharing
        """
        filtered = receipt_data.copy()

        # Always filter these fields from receipts (per MP-02)
        if not self.policy.allow_content_in_receipts:
            filtered.pop("raw_content", None)
            filtered.pop("signal_contents", None)

            # Filter validation metadata that might leak content
            if "validation_metadata" in filtered:
                vm = filtered["validation_metadata"].copy()
                vm.pop("raw_assessment", None)  # May contain content excerpts
                filtered["validation_metadata"] = vm

        # Mark as filtered
        filtered["privacy_filtered"] = True
        filtered["filtered_at"] = time.time()

        logger.debug(f"Filtered receipt {filtered.get('receipt_id', 'unknown')}")
        return filtered

    def redact_patterns(self, text: str) -> str:
        """Apply pattern-based redaction to text."""
        import re

        result = text
        for pattern in self.policy.redact_patterns:
            try:
                result = re.sub(pattern, self._redaction_marker, result)
            except re.error as e:
                logger.warning(f"Invalid redaction pattern '{pattern}': {e}")

        return result


class ConsentRegistry:
    """
    Registry for managing observation consent across sessions.

    Tracks who has consented to observation and who has revoked.

    Security:
    - Storage paths are validated to prevent path traversal attacks
    """

    def __init__(self, storage_path: Optional[str] = None):
        self._consents: Dict[str, ObservationConsent] = {}
        self._storage_path: Optional[Path] = None

        # Validate storage path if provided
        if storage_path:
            self._storage_path = _validate_storage_path(storage_path)

        self._load_from_storage()

    def _load_from_storage(self) -> None:
        """Load consent records from storage."""
        if not self._storage_path:
            return

        if not self._storage_path.exists():
            return

        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
                for human_id, consent_data in data.items():
                    self._consents[human_id] = ObservationConsent(
                        human_id=human_id,
                        status=consent_data.get("status", ConsentStatus.PENDING),
                        granted_at=consent_data.get("granted_at"),
                        revoked_at=consent_data.get("revoked_at"),
                        expires_at=consent_data.get("expires_at"),
                        scope=consent_data.get("scope", "all"),
                        allowed_observers=consent_data.get("allowed_observers", []),
                        allowed_signal_types=consent_data.get("allowed_signal_types", []),
                        revocation_reason=consent_data.get("revocation_reason"),
                    )
            logger.info(f"Loaded {len(self._consents)} consent records from storage")
        except Exception as e:
            logger.error(f"Failed to load consent records: {e}")

    def _save_to_storage(self) -> None:
        """Save consent records to storage."""
        if not self._storage_path:
            return

        try:
            data = {}
            for human_id, consent in self._consents.items():
                data[human_id] = {
                    "status": consent.status,
                    "granted_at": consent.granted_at,
                    "revoked_at": consent.revoked_at,
                    "expires_at": consent.expires_at,
                    "scope": consent.scope,
                    "allowed_observers": consent.allowed_observers,
                    "allowed_signal_types": consent.allowed_signal_types,
                    "revocation_reason": consent.revocation_reason,
                }

            with open(self._storage_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved {len(data)} consent records to storage")
        except Exception as e:
            logger.error(f"Failed to save consent records: {e}")

    def get_consent(self, human_id: str) -> Optional[ObservationConsent]:
        """Get consent record for a human."""
        return self._consents.get(human_id)

    def grant_consent(
        self,
        human_id: str,
        expires_in_seconds: Optional[float] = None,
        allowed_observers: Optional[List[str]] = None,
        allowed_signal_types: Optional[List[str]] = None,
    ) -> ObservationConsent:
        """
        Grant observation consent for a human.

        Args:
            human_id: Unique identifier for the human
            expires_in_seconds: Optional expiration time
            allowed_observers: Optional list of allowed observer IDs
            allowed_signal_types: Optional list of allowed signal types

        Returns:
            The created/updated consent record
        """
        consent = self._consents.get(human_id) or ObservationConsent(human_id=human_id)
        consent.grant(expires_in_seconds)

        if allowed_observers:
            consent.allowed_observers = allowed_observers
        if allowed_signal_types:
            consent.allowed_signal_types = allowed_signal_types

        self._consents[human_id] = consent
        self._save_to_storage()

        logger.info(f"Granted observation consent for {human_id}")
        return consent

    def revoke_consent(self, human_id: str, reason: Optional[str] = None) -> Optional[ObservationConsent]:
        """
        Revoke future observation consent.

        Per MP-02 §12: Humans MAY revoke future observation.
        Past receipts remain immutable.

        Args:
            human_id: Human identifier
            reason: Optional reason for revocation

        Returns:
            Updated consent record or None if not found
        """
        consent = self._consents.get(human_id)
        if not consent:
            consent = ObservationConsent(human_id=human_id)
            self._consents[human_id] = consent

        consent.revoke(reason)
        self._save_to_storage()

        logger.info(f"Revoked observation consent for {human_id}: {reason}")
        return consent

    def is_observation_allowed(
        self,
        human_id: str,
        observer_id: Optional[str] = None,
        signal_type: Optional[str] = None,
    ) -> bool:
        """
        Check if observation is allowed for a human.

        Args:
            human_id: Human identifier
            observer_id: Optional observer to check
            signal_type: Optional signal type to check

        Returns:
            True if observation is allowed
        """
        consent = self._consents.get(human_id)

        if not consent:
            logger.debug(f"No consent record for {human_id}, defaulting to not allowed")
            return False

        if not consent.is_valid():
            logger.debug(f"Consent for {human_id} is not valid (status={consent.status})")
            return False

        # Check observer restrictions
        if observer_id and consent.allowed_observers:
            if observer_id not in consent.allowed_observers:
                logger.debug(f"Observer {observer_id} not in allowed list for {human_id}")
                return False

        # Check signal type restrictions
        if signal_type and consent.allowed_signal_types:
            if signal_type not in consent.allowed_signal_types:
                logger.debug(f"Signal type {signal_type} not allowed for {human_id}")
                return False

        return True

    def list_revoked(self) -> List[ObservationConsent]:
        """List all revoked consents."""
        return [c for c in self._consents.values() if c.status == ConsentStatus.REVOKED]

    def list_active(self) -> List[ObservationConsent]:
        """List all active (valid) consents."""
        return [c for c in self._consents.values() if c.is_valid()]


class AgencyController:
    """
    Controller for human agency over observation.

    Per MP-02 §12: Humans MAY revoke future observation.
    This controller manages the revocation process and enforces it.
    """

    def __init__(
        self,
        consent_registry: Optional[ConsentRegistry] = None,
        privacy_filter: Optional[PrivacyFilter] = None,
    ):
        self.consent_registry = consent_registry or ConsentRegistry()
        self.privacy_filter = privacy_filter or PrivacyFilter()
        self._revocation_callbacks: List[callable] = []

    def on_revocation(self, callback: callable) -> None:
        """Register a callback for revocation events."""
        self._revocation_callbacks.append(callback)

    def revoke_future_observation(
        self,
        human_id: str,
        reason: Optional[str] = None,
        scope: str = RevocationScope.ALL,
    ) -> Dict[str, Any]:
        """
        Revoke future observation for a human.

        Per MP-02 §12:
        - Humans MAY revoke future observation
        - Past receipts remain immutable

        Args:
            human_id: Human identifier
            reason: Optional reason for revocation
            scope: Scope of revocation (all, session, signal_type, observer)

        Returns:
            Revocation result with details
        """
        consent = self.consent_registry.revoke_consent(human_id, reason)

        result = {
            "human_id": human_id,
            "revoked_at": consent.revoked_at if consent else time.time(),
            "reason": reason,
            "scope": scope,
            "past_receipts_affected": False,  # Per MP-02: Past receipts remain immutable
            "future_observation_blocked": True,
        }

        # Notify callbacks
        for callback in self._revocation_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Revocation callback failed: {e}")

        logger.info(f"Future observation revoked for {human_id} (scope={scope})")
        return result

    def check_observation_permission(
        self,
        human_id: str,
        observer_id: str,
        signal_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Check if observation is permitted.

        Args:
            human_id: Human to observe
            observer_id: Observer requesting permission
            signal_type: Type of signal to observe

        Returns:
            Permission result with allowed status and reason
        """
        allowed = self.consent_registry.is_observation_allowed(
            human_id=human_id,
            observer_id=observer_id,
            signal_type=signal_type,
        )

        consent = self.consent_registry.get_consent(human_id)

        result = {
            "allowed": allowed,
            "human_id": human_id,
            "observer_id": observer_id,
            "signal_type": signal_type,
            "checked_at": time.time(),
        }

        if not allowed:
            if consent and consent.status == ConsentStatus.REVOKED:
                result["reason"] = "consent_revoked"
                result["revoked_at"] = consent.revoked_at
            elif consent and consent.expires_at and time.time() > consent.expires_at:
                result["reason"] = "consent_expired"
            elif not consent:
                result["reason"] = "no_consent"
            else:
                result["reason"] = "not_authorized"

        return result

    def apply_privacy_to_receipt(
        self,
        receipt_data: Dict[str, Any],
        reader_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply privacy controls to a receipt before sharing.

        Per MP-02 §12: Receipts MUST not expose raw content by default.

        Args:
            receipt_data: Receipt data dictionary
            reader_id: Optional reader identifier

        Returns:
            Privacy-filtered receipt
        """
        return self.privacy_filter.filter_receipt(receipt_data, reader_id)


# Convenience functions for module-level access

def create_privacy_controller(
    consent_storage_path: Optional[str] = None,
    privacy_policy: Optional[PrivacyPolicy] = None,
) -> AgencyController:
    """
    Create a fully configured privacy/agency controller.

    Args:
        consent_storage_path: Path for consent storage
        privacy_policy: Optional custom privacy policy

    Returns:
        Configured AgencyController
    """
    registry = ConsentRegistry(storage_path=consent_storage_path)
    filter_ = PrivacyFilter(policy=privacy_policy)
    return AgencyController(consent_registry=registry, privacy_filter=filter_)


def encrypt_signal_content(content: str, password: str) -> EncryptedContent:
    """
    Encrypt signal content for storage.

    Per MP-02 §12: Raw signals MAY be encrypted.

    Args:
        content: Raw content to encrypt
        password: Encryption password

    Returns:
        EncryptedContent object
    """
    encryptor = SignalEncryptor()
    return encryptor.encrypt(content, password)


def decrypt_signal_content(encrypted: EncryptedContent, password: str) -> str:
    """
    Decrypt signal content.

    Args:
        encrypted: EncryptedContent to decrypt
        password: Decryption password

    Returns:
        Decrypted content string
    """
    encryptor = SignalEncryptor()
    return encryptor.decrypt(encrypted, password)
