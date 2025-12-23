# value_ledger/natlangchain.py
"""
NatLangChain Integration for Value Ledger.

NatLangChain is a prose-first distributed ledger where natural language entries
form the immutable substrate. This module provides:

1. Export of ledger entries to NatLangChain format
2. Anchoring receipts to the chain
3. Verification of chain inclusion
4. Proof of Understanding compatibility

See: https://github.com/kase1111-hash/NatLangChain
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
import urllib.request
import urllib.error
import urllib.parse
import ipaddress
import logging
import socket

logger = logging.getLogger(__name__)


def _validate_url(url: str, allow_private: bool = False) -> None:
    """
    Validate URL to prevent SSRF attacks.

    Args:
        url: URL to validate
        allow_private: If True, allow private/internal addresses (for testing)

    Raises:
        ValueError: If URL is invalid or points to restricted address
    """
    parsed = urllib.parse.urlparse(url)

    # Only allow HTTP and HTTPS
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL scheme: {parsed.scheme}. Only http/https allowed.")

    # Must have a hostname
    if not parsed.hostname:
        raise ValueError("URL must have a hostname")

    hostname = parsed.hostname.lower()

    # Block common SSRF targets
    blocked_hosts = {
        "metadata.google.internal",
        "169.254.169.254",  # AWS/GCP metadata
        "metadata.azure.internal",
        "100.100.100.200",  # Alibaba metadata
    }

    if hostname in blocked_hosts:
        raise ValueError(f"Access to {hostname} is not allowed")

    if not allow_private:
        # Try to resolve and check if IP is private
        try:
            # Check if hostname is already an IP
            try:
                ip = ipaddress.ip_address(hostname)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    raise ValueError(f"Private/internal addresses not allowed: {hostname}")
            except ValueError:
                # Not an IP, try to resolve
                resolved = socket.gethostbyname(hostname)
                ip = ipaddress.ip_address(resolved)
                if ip.is_private or ip.is_loopback or ip.is_link_local:
                    raise ValueError(f"URL resolves to private address: {resolved}")
        except socket.gaierror:
            # Can't resolve - allow (will fail at connection time)
            pass

    logger.debug(f"URL validated: {url}")


class AnchorStatus(str, Enum):
    """Status of a chain anchor operation."""
    PENDING = "pending"
    ANCHORED = "anchored"
    FAILED = "failed"
    VERIFIED = "verified"


@dataclass
class NLCRecord:
    """
    A record formatted for NatLangChain submission.

    NatLangChain requires prose-first entries where natural language
    forms the immutable substrate of the ledger.
    """
    record_type: str = "effort_receipt"
    timestamp: float = field(default_factory=time.time)
    proof_hash: str = ""  # SHA-256 compatible with NLC block chaining
    value_summary: Dict[str, Any] = field(default_factory=dict)
    prior_anchors: List[str] = field(default_factory=list)

    # NatLangChain-specific fields
    prose_entry: str = ""  # Human-readable narrative (required by NLC)
    author: str = ""  # Owner/creator identifier
    intent_summary: str = ""  # From validator deterministic summary

    # Anchoring metadata
    anchor_id: Optional[str] = None
    anchor_status: str = AnchorStatus.PENDING
    anchored_at: Optional[float] = None

    def compute_hash(self) -> str:
        """Compute hash for NLC chain linking."""
        data = {
            "record_type": self.record_type,
            "timestamp": self.timestamp,
            "prose_entry": self.prose_entry,
            "author": self.author,
            "intent_summary": self.intent_summary,
            "prior_anchors": self.prior_anchors,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "record_type": self.record_type,
            "timestamp": self.timestamp,
            "proof_hash": self.proof_hash or self.compute_hash(),
            "value_summary": self.value_summary,
            "prior_anchors": self.prior_anchors,
            "prose_entry": self.prose_entry,
            "author": self.author,
            "intent_summary": self.intent_summary,
            "anchor_id": self.anchor_id,
            "anchor_status": self.anchor_status,
            "anchored_at": self.anchored_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NLCRecord":
        """Create from dictionary."""
        return cls(
            record_type=data.get("record_type", "effort_receipt"),
            timestamp=data.get("timestamp", time.time()),
            proof_hash=data.get("proof_hash", ""),
            value_summary=data.get("value_summary", {}),
            prior_anchors=data.get("prior_anchors", []),
            prose_entry=data.get("prose_entry", ""),
            author=data.get("author", ""),
            intent_summary=data.get("intent_summary", ""),
            anchor_id=data.get("anchor_id"),
            anchor_status=data.get("anchor_status", AnchorStatus.PENDING),
            anchored_at=data.get("anchored_at"),
        )


@dataclass
class ValidationResult:
    """Result of pre-anchor validation."""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    prose_quality_score: Optional[float] = None  # 0.0-1.0
    chain_compatible: bool = True


@dataclass
class AnchorResult:
    """Result of anchoring operation."""
    success: bool
    anchor_id: Optional[str] = None
    chain_ref: Optional[str] = None
    block_number: Optional[int] = None
    error: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class InclusionProof:
    """Proof that a record is included in the chain."""
    record_id: str
    anchor_id: str
    block_number: int
    merkle_proof: List[Dict[str, str]]
    chain_root: str
    verified: bool
    verified_at: float = field(default_factory=time.time)


class NLCClient:
    """
    Client for NatLangChain REST API.

    Endpoints (from NatLangChain spec):
    - POST /entry          Submit entry to chain
    - POST /entry/validate Validate entry before submission
    - GET  /chain          Query chain entries
    - GET  /chain/narrative Human-readable chain view

    Security:
    - URL validation prevents SSRF attacks
    - Set allow_private=True only for local development/testing
    """

    # Maximum response size (10 MB)
    MAX_RESPONSE_SIZE = 10 * 1024 * 1024

    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        timeout: float = 30.0,
        allow_private: bool = True,  # Default True for backward compatibility
        max_response_size: int = MAX_RESPONSE_SIZE,
    ):
        """
        Initialize NatLangChain client.

        Args:
            base_url: Base URL of NatLangChain API
            timeout: Request timeout in seconds
            allow_private: If False, block requests to private/internal IPs (SSRF protection)
            max_response_size: Maximum response size in bytes (prevents memory exhaustion)
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.allow_private = allow_private
        self.max_response_size = max_response_size

        # Validate URL on initialization
        _validate_url(self.base_url, allow_private=allow_private)

    def _read_response(self, response) -> bytes:
        """
        Read response with size limit to prevent memory exhaustion.

        Raises:
            ValueError: If response exceeds max_response_size
        """
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) > self.max_response_size:
            raise ValueError(f"Response too large: {content_length} bytes")

        # Read in chunks to enforce limit
        chunks = []
        bytes_read = 0
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > self.max_response_size:
                raise ValueError(f"Response exceeded {self.max_response_size} bytes")
            chunks.append(chunk)

        return b"".join(chunks)

    def submit_entry(self, record: NLCRecord) -> AnchorResult:
        """
        POST /entry - Submit entry to chain.

        Returns anchor_id on success.
        """
        try:
            data = json.dumps(record.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/entry",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(self._read_response(response).decode("utf-8"))
                return AnchorResult(
                    success=True,
                    anchor_id=result.get("anchor_id"),
                    chain_ref=result.get("chain_ref"),
                    block_number=result.get("block_number"),
                )

        except urllib.error.HTTPError as e:
            return AnchorResult(
                success=False,
                error=f"HTTP {e.code}: {e.reason}",
            )
        except urllib.error.URLError as e:
            return AnchorResult(
                success=False,
                error=f"Connection error: {e.reason}",
            )
        except ValueError as e:
            return AnchorResult(
                success=False,
                error=f"Response error: {e}",
            )
        except Exception as e:
            return AnchorResult(
                success=False,
                error=str(e),
            )

    def validate_entry(self, record: NLCRecord) -> ValidationResult:
        """
        POST /entry/validate - Dry-run validation.

        Checks entry format and prose quality without submitting.
        """
        try:
            data = json.dumps(record.to_dict()).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/entry/validate",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(self._read_response(response).decode("utf-8"))
                return ValidationResult(
                    valid=result.get("valid", False),
                    errors=result.get("errors", []),
                    warnings=result.get("warnings", []),
                    prose_quality_score=result.get("prose_quality_score"),
                    chain_compatible=result.get("chain_compatible", True),
                )

        except Exception as e:
            # Fallback to local validation if API unavailable
            return self._local_validate(record)

    def get_chain_narrative(self, limit: int = 50) -> str:
        """
        GET /chain/narrative - Human-readable ledger view.
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/chain/narrative?limit={limit}",
                method="GET",
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                return self._read_response(response).decode("utf-8")

        except Exception as e:
            return f"Error fetching narrative: {e}"

    def search_by_intent(self, intent: str) -> List[NLCRecord]:
        """
        GET /entries/search?intent=... - Semantic search.
        """
        try:
            encoded_intent = urllib.parse.quote(intent)
            req = urllib.request.Request(
                f"{self.base_url}/entries/search?intent={encoded_intent}",
                method="GET",
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(self._read_response(response).decode("utf-8"))
                return [NLCRecord.from_dict(r) for r in result.get("entries", [])]

        except Exception:
            return []

    def check_inclusion(self, anchor_id: str) -> Optional[InclusionProof]:
        """
        GET /chain/proof/{anchor_id} - Get inclusion proof.
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/chain/proof/{anchor_id}",
                method="GET",
            )

            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(self._read_response(response).decode("utf-8"))
                return InclusionProof(
                    record_id=result.get("record_id", ""),
                    anchor_id=anchor_id,
                    block_number=result.get("block_number", 0),
                    merkle_proof=result.get("merkle_proof", []),
                    chain_root=result.get("chain_root", ""),
                    verified=result.get("verified", False),
                )

        except Exception:
            return None

    def _local_validate(self, record: NLCRecord) -> ValidationResult:
        """Local validation when API is unavailable."""
        errors = []
        warnings = []

        # Check required fields
        if not record.prose_entry:
            errors.append("prose_entry is required")
        elif len(record.prose_entry) < 10:
            warnings.append("prose_entry is very short")

        if not record.author:
            warnings.append("author not specified")

        if not record.timestamp:
            errors.append("timestamp is required")

        # Check prose quality heuristically
        prose_score = 0.5
        if record.prose_entry:
            words = record.prose_entry.split()
            if len(words) >= 20:
                prose_score = 0.8
            elif len(words) >= 10:
                prose_score = 0.6

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            prose_quality_score=prose_score,
            chain_compatible=True,
        )


class NatLangChainExporter:
    """
    Export Value Ledger entries to NatLangChain format.
    """

    def __init__(self, client: Optional[NLCClient] = None):
        self.client = client or NLCClient()

    def to_nlc_format(self, entry: "LedgerEntry") -> NLCRecord:
        """Convert a ledger entry to NatLangChain record."""
        from .core import LedgerEntry

        # Generate prose entry
        prose = self.to_prose_entry(entry)

        # Generate intent summary
        intent_summary = self._generate_intent_summary(entry)

        return NLCRecord(
            record_type="effort_receipt",
            timestamp=entry.timestamp,
            proof_hash=entry.proof.content_hash or "",
            value_summary={
                "vector": entry.value_vector.dict(),
                "total": entry.value_vector.total(),
                "status": entry.status,
            },
            prior_anchors=[],  # Will be populated from chain history
            prose_entry=prose,
            author=entry.owner or "anonymous",
            intent_summary=intent_summary,
        )

    def to_prose_entry(self, entry: "LedgerEntry") -> str:
        """
        Convert ledger entry to human-readable prose for NatLangChain.

        The prose forms the immutable substrate of the chain.
        """
        from .core import LedgerEntry

        vec = entry.value_vector
        status_text = {
            "active": "active and accruing",
            "frozen": "frozen",
            "revoked": "revoked",
        }.get(entry.status, entry.status)

        prose = f"""Effort Receipt #{entry.id[:8] if entry.id else 'pending'}

Recorded at timestamp {entry.timestamp:.0f}, this receipt documents cognitive effort
associated with intent "{entry.intent_id}".

Value Assessment:
- Time invested: {vec.t:.1f} units
- Effort expended: {vec.e:.1f} units
- Novelty contribution: {vec.n:.1f} units
- Learning from failure: {vec.f:.1f} units
- Risk exposure: {vec.r:.1f} units
- Strategic insight: {vec.s:.1f} units
- Reusability potential: {vec.u:.1f} units
- Total value: {vec.total():.1f} units

This entry is currently {status_text}."""

        if entry.status == "revoked" and entry.revocation_reason:
            prose += f"\nRevocation reason: {entry.revocation_reason}"

        if entry.owner:
            prose += f"\nOwner: {entry.owner}"

        if entry.classification > 0:
            prose += f"\nClassification level: {entry.classification}"

        return prose

    def _generate_intent_summary(self, entry: "LedgerEntry") -> str:
        """
        Generate deterministic summary suitable for Proof of Understanding.

        This summary must be reproducible by other validators.
        """
        vec = entry.value_vector

        # Identify dominant value dimension
        dimensions = {
            "time": vec.t,
            "effort": vec.e,
            "novelty": vec.n,
            "learning": vec.f,
            "risk": vec.r,
            "strategy": vec.s,
        }
        dominant = max(dimensions.items(), key=lambda x: x[1])

        return (
            f"Effort for intent '{entry.intent_id}' "
            f"with primary contribution in {dominant[0]} ({dominant[1]:.1f}). "
            f"Total assessed value: {vec.total():.1f}."
        )

    def batch_export(self, entries: List["LedgerEntry"]) -> List[NLCRecord]:
        """Export multiple entries to NLC format."""
        records = []
        prior_anchors = []

        for entry in sorted(entries, key=lambda e: e.timestamp):
            record = self.to_nlc_format(entry)
            record.prior_anchors = prior_anchors.copy()
            records.append(record)

            # Track anchors for chaining
            if record.proof_hash:
                prior_anchors.append(record.proof_hash)

        return records

    def validate_before_anchor(self, record: NLCRecord) -> ValidationResult:
        """Validate a record before anchoring."""
        return self.client.validate_entry(record)

    def anchor_to_chain(self, record: NLCRecord) -> AnchorResult:
        """Anchor a record to NatLangChain."""
        # Validate first
        validation = self.validate_before_anchor(record)
        if not validation.valid:
            return AnchorResult(
                success=False,
                error=f"Validation failed: {', '.join(validation.errors)}",
            )

        # Submit to chain
        result = self.client.submit_entry(record)

        # Update record status
        if result.success:
            record.anchor_id = result.anchor_id
            record.anchor_status = AnchorStatus.ANCHORED
            record.anchored_at = time.time()

        return result

    def verify_anchor(self, entry_id: str, chain_ref: str) -> bool:
        """Verify an entry is anchored at the given chain reference."""
        proof = self.client.check_inclusion(chain_ref)
        return proof is not None and proof.verified

    def get_chain_proof(self, anchor_id: str) -> Optional[InclusionProof]:
        """Get inclusion proof for an anchored entry."""
        return self.client.check_inclusion(anchor_id)


class ProofOfUnderstandingValidator:
    """
    Validator compatible with NatLangChain's Proof of Understanding.

    In NatLangChain, validators paraphrase entries to demonstrate comprehension.
    This validator produces deterministic summaries that can serve as
    Proof of Understanding for semantic consensus.
    """

    def __init__(self, validator_id: str = "value_ledger_validator"):
        self.validator_id = validator_id

    def generate_understanding_proof(self, record: NLCRecord) -> str:
        """
        Generate a paraphrase demonstrating understanding of the record.

        This serves as Proof of Understanding for NatLangChain consensus.
        """
        value_summary = record.value_summary
        vec = value_summary.get("vector", {})
        total = value_summary.get("total", 0)

        # Create paraphrase that demonstrates understanding
        paraphrase = f"""
Understanding of Record {record.proof_hash[:8] if record.proof_hash else 'unknown'}:

This effort receipt, authored by {record.author or 'anonymous'}, documents
work characterized primarily by its assessed value components.

The recorded effort shows:
- Temporal investment of {vec.get('t', 0):.1f} units
- Applied effort quantified at {vec.get('e', 0):.1f} units
- Novel contribution rated {vec.get('n', 0):.1f} units
- Learning value of {vec.get('f', 0):.1f} units
- Risk factor of {vec.get('r', 0):.1f} units
- Strategic component of {vec.get('s', 0):.1f} units
- Reusability potential of {vec.get('u', 0):.1f} units

Aggregate assessed value: {total:.1f} units

Intent: {record.intent_summary}

This paraphrase demonstrates comprehension of the original record's content
and value assessment methodology.

Validator: {self.validator_id}
Timestamp: {time.time():.0f}
"""
        return paraphrase.strip()

    def validate_understanding(
        self,
        original: NLCRecord,
        paraphrase: str,
    ) -> Tuple[bool, float, List[str]]:
        """
        Validate that a paraphrase demonstrates understanding.

        Returns:
            (valid, confidence_score, issues)
        """
        issues = []
        score = 0.0

        # Check that key values are mentioned
        vec = original.value_summary.get("vector", {})
        total = original.value_summary.get("total", 0)

        # Check for value mentions (basic heuristic)
        if str(round(total, 1)) in paraphrase:
            score += 0.3
        else:
            issues.append("Total value not accurately reflected")

        # Check for author mention
        if original.author and original.author in paraphrase:
            score += 0.2
        elif not original.author:
            score += 0.1

        # Check for intent mention
        if original.intent_summary and any(
            word in paraphrase.lower()
            for word in original.intent_summary.lower().split()[:5]
        ):
            score += 0.3
        else:
            issues.append("Intent not adequately reflected")

        # Check minimum length
        if len(paraphrase.split()) >= 50:
            score += 0.2
        else:
            issues.append("Paraphrase too brief")

        return (score >= 0.6, score, issues)


def anchor_receipt_to_nlc(
    receipt: "EffortReceipt",
    nlc_client: Optional[NLCClient] = None,
) -> AnchorResult:
    """
    Convenience function to anchor an EffortReceipt to NatLangChain.

    Uses effort_summary as the canonical prose for Proof of Understanding.
    """
    from .receipt import EffortReceipt

    client = nlc_client or NLCClient()
    exporter = NatLangChainExporter(client)

    # Convert receipt to NLC record
    record = NLCRecord(
        record_type="effort_receipt",
        timestamp=receipt.time_bounds[0],
        proof_hash=receipt.receipt_hash or "",
        value_summary={
            "signal_count": len(receipt.signal_hashes),
            "time_bounds": list(receipt.time_bounds),
            "is_complete": not receipt.is_incomplete,
        },
        prior_anchors=receipt.prior_receipts,
        prose_entry=receipt.effort_summary,
        author=receipt.observer_id,
        intent_summary=f"Effort recorded by {receipt.observer_id}, validated by {receipt.validator_id}",
    )

    # Anchor to chain
    result = exporter.anchor_to_chain(record)

    # Update receipt with anchor info
    if result.success:
        receipt.chain_anchor = result.anchor_id

    return result
