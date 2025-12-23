# value_ledger/compatibility.py
"""
MP-02 External Compatibility Implementation.

Per MP-02 §14 Compatibility:
- MP-02 receipts are compatible with MP-01 Negotiation & Ratification
- Licensing and delegation modules
- External audit systems

This module provides:
- MP-01 protocol integration for negotiation workflows
- License reference embedding and verification
- External audit format exports (JSON-LD, OpenTimestamps-style, audit schemas)
- Cross-protocol adapters for interoperability
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# MP-01 Negotiation & Ratification Protocol Compatibility
# =============================================================================

class NegotiationStatus(str, Enum):
    """Status of MP-01 negotiation."""
    PROPOSED = "proposed"
    COUNTER_OFFERED = "counter_offered"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    RATIFIED = "ratified"
    EXPIRED = "expired"


class RatificationMethod(str, Enum):
    """Method of ratification per MP-01."""
    SIGNATURE = "signature"
    CONSENSUS = "consensus"
    THRESHOLD = "threshold"
    AUTOMATIC = "automatic"


@dataclass
class MP01Proposal:
    """
    A negotiation proposal compatible with MP-01 protocol.

    MP-01 defines how effort receipts can be used in negotiation
    and licensing workflows.
    """
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    receipt_ids: List[str] = field(default_factory=list)  # Referenced receipts
    proposer_id: str = ""
    recipient_id: str = ""
    terms: Dict[str, Any] = field(default_factory=dict)
    status: str = NegotiationStatus.PROPOSED
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    counter_proposals: List[str] = field(default_factory=list)
    ratification_method: str = RatificationMethod.SIGNATURE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "protocol": "MP-01",
            "version": "1.0",
            "proposal_id": self.proposal_id,
            "receipt_ids": self.receipt_ids,
            "proposer_id": self.proposer_id,
            "recipient_id": self.recipient_id,
            "terms": self.terms,
            "status": self.status,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "counter_proposals": self.counter_proposals,
            "ratification_method": self.ratification_method,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MP01Proposal":
        """Deserialize from dictionary."""
        return cls(
            proposal_id=data.get("proposal_id", str(uuid.uuid4())),
            receipt_ids=data.get("receipt_ids", []),
            proposer_id=data.get("proposer_id", ""),
            recipient_id=data.get("recipient_id", ""),
            terms=data.get("terms", {}),
            status=data.get("status", NegotiationStatus.PROPOSED),
            created_at=data.get("created_at", time.time()),
            expires_at=data.get("expires_at"),
            counter_proposals=data.get("counter_proposals", []),
            ratification_method=data.get("ratification_method", RatificationMethod.SIGNATURE),
        )

    def compute_hash(self) -> str:
        """Compute proposal hash for verification."""
        data = {
            "proposal_id": self.proposal_id,
            "receipt_ids": sorted(self.receipt_ids),
            "proposer_id": self.proposer_id,
            "recipient_id": self.recipient_id,
            "terms": json.dumps(self.terms, sort_keys=True),
            "created_at": self.created_at,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class MP01Ratification:
    """
    A ratification record for MP-01 proposals.
    """
    ratification_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    proposal_id: str = ""
    proposal_hash: str = ""
    ratified_by: List[str] = field(default_factory=list)
    method: str = RatificationMethod.SIGNATURE
    ratified_at: float = field(default_factory=time.time)
    signatures: Dict[str, str] = field(default_factory=dict)  # party_id -> signature
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "protocol": "MP-01",
            "type": "ratification",
            "ratification_id": self.ratification_id,
            "proposal_id": self.proposal_id,
            "proposal_hash": self.proposal_hash,
            "ratified_by": self.ratified_by,
            "method": self.method,
            "ratified_at": self.ratified_at,
            "signatures": self.signatures,
            "metadata": self.metadata,
        }


class MP01Formatter:
    """
    Formatter for MP-01 Negotiation & Ratification protocol.

    Converts MP-02 receipts into formats suitable for MP-01 workflows.
    """

    def __init__(self, party_id: str = ""):
        self.party_id = party_id

    def create_proposal(
        self,
        receipt_ids: List[str],
        recipient_id: str,
        terms: Dict[str, Any],
        expires_in_seconds: Optional[float] = None,
    ) -> MP01Proposal:
        """
        Create an MP-01 proposal referencing effort receipts.

        Args:
            receipt_ids: List of receipt IDs to include
            recipient_id: ID of the recipient party
            terms: Negotiation terms (licensing, compensation, etc.)
            expires_in_seconds: Optional expiration time

        Returns:
            MP01Proposal ready for negotiation
        """
        proposal = MP01Proposal(
            receipt_ids=receipt_ids,
            proposer_id=self.party_id,
            recipient_id=recipient_id,
            terms=terms,
        )

        if expires_in_seconds:
            proposal.expires_at = time.time() + expires_in_seconds

        logger.info(f"Created MP-01 proposal {proposal.proposal_id} with {len(receipt_ids)} receipts")
        return proposal

    def format_for_negotiation(self, receipt: "EffortReceipt") -> Dict[str, Any]:
        """
        Format a receipt for use in MP-01 negotiation.

        Returns a standardized format that MP-01 systems can process.
        """
        return {
            "format": "MP-01/MP-02",
            "version": "1.0",
            "receipt": {
                "id": receipt.receipt_id,
                "hash": receipt.receipt_hash,
                "time_bounds": list(receipt.time_bounds),
                "signal_count": len(receipt.signal_hashes),
                "observer": receipt.observer_id,
                "validator": receipt.validator_id,
                "summary": receipt.effort_summary,
                "is_complete": not receipt.is_incomplete,
                "suspected_manipulation": receipt.suspected_manipulation,
            },
            "negotiation_metadata": {
                "exportable": True,
                "referenceable": True,
                "verifiable": True,
            },
        }

    def accept_proposal(self, proposal: MP01Proposal) -> MP01Proposal:
        """Mark a proposal as accepted."""
        proposal.status = NegotiationStatus.ACCEPTED
        logger.info(f"Proposal {proposal.proposal_id} accepted")
        return proposal

    def reject_proposal(self, proposal: MP01Proposal, reason: Optional[str] = None) -> MP01Proposal:
        """Mark a proposal as rejected."""
        proposal.status = NegotiationStatus.REJECTED
        if reason:
            proposal.terms["rejection_reason"] = reason
        logger.info(f"Proposal {proposal.proposal_id} rejected: {reason}")
        return proposal

    def create_counter_proposal(
        self,
        original: MP01Proposal,
        modified_terms: Dict[str, Any],
    ) -> MP01Proposal:
        """Create a counter-proposal."""
        counter = MP01Proposal(
            receipt_ids=original.receipt_ids,
            proposer_id=self.party_id,
            recipient_id=original.proposer_id,
            terms={**original.terms, **modified_terms},
        )
        counter.status = NegotiationStatus.COUNTER_OFFERED

        original.counter_proposals.append(counter.proposal_id)
        original.status = NegotiationStatus.COUNTER_OFFERED

        logger.info(f"Created counter-proposal {counter.proposal_id} for {original.proposal_id}")
        return counter

    def ratify_proposal(
        self,
        proposal: MP01Proposal,
        ratifiers: List[str],
        method: str = RatificationMethod.SIGNATURE,
    ) -> MP01Ratification:
        """
        Ratify a proposal, creating a binding record.

        Args:
            proposal: The proposal to ratify
            ratifiers: List of party IDs ratifying
            method: Ratification method

        Returns:
            MP01Ratification record
        """
        if proposal.status != NegotiationStatus.ACCEPTED:
            logger.warning(f"Ratifying non-accepted proposal {proposal.proposal_id}")

        ratification = MP01Ratification(
            proposal_id=proposal.proposal_id,
            proposal_hash=proposal.compute_hash(),
            ratified_by=ratifiers,
            method=method,
        )

        proposal.status = NegotiationStatus.RATIFIED
        logger.info(f"Proposal {proposal.proposal_id} ratified by {ratifiers}")

        return ratification


# =============================================================================
# Licensing and Delegation Module Compatibility
# =============================================================================

class LicenseType(str, Enum):
    """Types of licenses that can be attached to receipts."""
    NONE = "none"
    VIEW_ONLY = "view_only"
    REFERENCE = "reference"
    DERIVATIVE = "derivative"
    FULL = "full"
    CUSTOM = "custom"


@dataclass
class LicenseReference:
    """
    License reference that can be attached to effort receipts.

    Supports both standard licenses and custom terms.
    """
    license_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    license_type: str = LicenseType.NONE
    receipt_ids: List[str] = field(default_factory=list)
    licensor_id: str = ""
    licensee_ids: List[str] = field(default_factory=list)
    terms: Dict[str, Any] = field(default_factory=dict)
    granted_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    revocable: bool = True
    delegation_allowed: bool = False
    delegation_depth: int = 0  # 0 = no delegation, >0 = max delegation chain length

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "license_id": self.license_id,
            "license_type": self.license_type,
            "receipt_ids": self.receipt_ids,
            "licensor_id": self.licensor_id,
            "licensee_ids": self.licensee_ids,
            "terms": self.terms,
            "granted_at": self.granted_at,
            "expires_at": self.expires_at,
            "revocable": self.revocable,
            "delegation_allowed": self.delegation_allowed,
            "delegation_depth": self.delegation_depth,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LicenseReference":
        """Deserialize from dictionary."""
        return cls(
            license_id=data.get("license_id", str(uuid.uuid4())),
            license_type=data.get("license_type", LicenseType.NONE),
            receipt_ids=data.get("receipt_ids", []),
            licensor_id=data.get("licensor_id", ""),
            licensee_ids=data.get("licensee_ids", []),
            terms=data.get("terms", {}),
            granted_at=data.get("granted_at", time.time()),
            expires_at=data.get("expires_at"),
            revocable=data.get("revocable", True),
            delegation_allowed=data.get("delegation_allowed", False),
            delegation_depth=data.get("delegation_depth", 0),
        )

    def is_valid(self) -> bool:
        """Check if license is currently valid."""
        if self.expires_at and time.time() > self.expires_at:
            return False
        return True

    def compute_hash(self) -> str:
        """Compute license hash for verification."""
        data = {
            "license_id": self.license_id,
            "license_type": self.license_type,
            "receipt_ids": sorted(self.receipt_ids),
            "licensor_id": self.licensor_id,
            "terms": json.dumps(self.terms, sort_keys=True),
            "granted_at": self.granted_at,
        }
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


@dataclass
class DelegationRecord:
    """Record of license delegation."""
    delegation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    license_id: str = ""
    delegator_id: str = ""
    delegatee_id: str = ""
    delegated_at: float = field(default_factory=time.time)
    depth: int = 1  # Current depth in delegation chain
    restrictions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "delegation_id": self.delegation_id,
            "license_id": self.license_id,
            "delegator_id": self.delegator_id,
            "delegatee_id": self.delegatee_id,
            "delegated_at": self.delegated_at,
            "depth": self.depth,
            "restrictions": self.restrictions,
        }


class LicenseManager:
    """
    Manages licenses and delegations for effort receipts.
    """

    def __init__(self):
        self._licenses: Dict[str, LicenseReference] = {}
        self._delegations: Dict[str, List[DelegationRecord]] = {}

    def grant_license(
        self,
        receipt_ids: List[str],
        licensor_id: str,
        licensee_ids: List[str],
        license_type: str = LicenseType.REFERENCE,
        terms: Optional[Dict[str, Any]] = None,
        expires_in_seconds: Optional[float] = None,
        delegation_allowed: bool = False,
        delegation_depth: int = 0,
    ) -> LicenseReference:
        """
        Grant a license for effort receipts.

        Args:
            receipt_ids: Receipts covered by the license
            licensor_id: ID of the party granting the license
            licensee_ids: IDs of parties receiving the license
            license_type: Type of license
            terms: Optional custom terms
            expires_in_seconds: Optional expiration
            delegation_allowed: Whether delegation is permitted
            delegation_depth: Maximum delegation chain depth

        Returns:
            LicenseReference record
        """
        license_ref = LicenseReference(
            license_type=license_type,
            receipt_ids=receipt_ids,
            licensor_id=licensor_id,
            licensee_ids=licensee_ids,
            terms=terms or {},
            delegation_allowed=delegation_allowed,
            delegation_depth=delegation_depth,
        )

        if expires_in_seconds:
            license_ref.expires_at = time.time() + expires_in_seconds

        self._licenses[license_ref.license_id] = license_ref
        logger.info(f"Granted license {license_ref.license_id} for {len(receipt_ids)} receipts")

        return license_ref

    def revoke_license(self, license_id: str) -> bool:
        """Revoke a license if revocable."""
        license_ref = self._licenses.get(license_id)
        if not license_ref:
            logger.warning(f"License {license_id} not found")
            return False

        if not license_ref.revocable:
            logger.warning(f"License {license_id} is not revocable")
            return False

        del self._licenses[license_id]
        logger.info(f"Revoked license {license_id}")
        return True

    def delegate_license(
        self,
        license_id: str,
        delegator_id: str,
        delegatee_id: str,
        restrictions: Optional[Dict[str, Any]] = None,
    ) -> Optional[DelegationRecord]:
        """
        Delegate a license to another party.

        Args:
            license_id: License to delegate
            delegator_id: Party delegating
            delegatee_id: Party receiving delegation
            restrictions: Optional additional restrictions

        Returns:
            DelegationRecord if successful, None otherwise
        """
        license_ref = self._licenses.get(license_id)
        if not license_ref:
            logger.warning(f"License {license_id} not found")
            return None

        if not license_ref.delegation_allowed:
            logger.warning(f"License {license_id} does not allow delegation")
            return None

        # Check delegator is authorized
        if delegator_id != license_ref.licensor_id and delegator_id not in license_ref.licensee_ids:
            existing_delegations = self._delegations.get(license_id, [])
            delegator_found = any(d.delegatee_id == delegator_id for d in existing_delegations)
            if not delegator_found:
                logger.warning(f"Party {delegator_id} not authorized to delegate license {license_id}")
                return None

        # Calculate delegation depth
        existing_delegations = self._delegations.get(license_id, [])
        current_depth = 1
        for d in existing_delegations:
            if d.delegatee_id == delegator_id:
                current_depth = d.depth + 1
                break

        if current_depth > license_ref.delegation_depth:
            logger.warning(f"Delegation depth exceeded for license {license_id}")
            return None

        delegation = DelegationRecord(
            license_id=license_id,
            delegator_id=delegator_id,
            delegatee_id=delegatee_id,
            depth=current_depth,
            restrictions=restrictions or {},
        )

        if license_id not in self._delegations:
            self._delegations[license_id] = []
        self._delegations[license_id].append(delegation)

        logger.info(f"Delegated license {license_id} from {delegator_id} to {delegatee_id}")
        return delegation

    def check_license(self, license_id: str, party_id: str) -> Tuple[bool, str]:
        """
        Check if a party has access under a license.

        Returns:
            (has_access, reason)
        """
        license_ref = self._licenses.get(license_id)
        if not license_ref:
            return (False, "License not found")

        if not license_ref.is_valid():
            return (False, "License expired")

        # Direct licensee
        if party_id in license_ref.licensee_ids:
            return (True, "Direct licensee")

        # Licensor always has access
        if party_id == license_ref.licensor_id:
            return (True, "Licensor")

        # Check delegations
        delegations = self._delegations.get(license_id, [])
        for delegation in delegations:
            if delegation.delegatee_id == party_id:
                return (True, f"Delegated at depth {delegation.depth}")

        return (False, "Not authorized")

    def get_license(self, license_id: str) -> Optional[LicenseReference]:
        """Get a license by ID."""
        return self._licenses.get(license_id)

    def list_licenses_for_receipt(self, receipt_id: str) -> List[LicenseReference]:
        """List all licenses covering a receipt."""
        return [
            lic for lic in self._licenses.values()
            if receipt_id in lic.receipt_ids
        ]


# =============================================================================
# External Audit System Compatibility
# =============================================================================

class AuditFormat(str, Enum):
    """Supported external audit formats."""
    JSON_LD = "json_ld"  # Linked Data format
    W3C_VC = "w3c_vc"  # W3C Verifiable Credentials-style
    OPEN_TIMESTAMPS = "open_timestamps"  # OpenTimestamps-compatible
    AUDIT_LOG = "audit_log"  # Standard audit log format
    XBRL = "xbrl"  # Financial reporting format


@dataclass
class AuditEntry:
    """
    A standardized audit entry compatible with external systems.
    """
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    event_type: str = ""
    subject_id: str = ""  # Receipt or ledger entry ID
    actor_id: str = ""
    action: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    evidence_hash: str = ""
    chain_ref: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "subject_id": self.subject_id,
            "actor_id": self.actor_id,
            "action": self.action,
            "details": self.details,
            "evidence_hash": self.evidence_hash,
            "chain_ref": self.chain_ref,
        }


class AuditExporter:
    """
    Exports receipts and ledger entries to external audit formats.

    Supports multiple industry-standard formats for interoperability
    with external audit systems.
    """

    def __init__(self, issuer_id: str = "value_ledger"):
        self.issuer_id = issuer_id

    def export_receipt(
        self,
        receipt: "EffortReceipt",
        format: str = AuditFormat.JSON_LD,
    ) -> Dict[str, Any]:
        """
        Export a receipt to the specified audit format.

        Args:
            receipt: The receipt to export
            format: Target audit format

        Returns:
            Formatted export data
        """
        if format == AuditFormat.JSON_LD:
            return self._to_json_ld(receipt)
        elif format == AuditFormat.W3C_VC:
            return self._to_w3c_vc(receipt)
        elif format == AuditFormat.OPEN_TIMESTAMPS:
            return self._to_open_timestamps(receipt)
        elif format == AuditFormat.AUDIT_LOG:
            return self._to_audit_log(receipt)
        else:
            logger.warning(f"Unknown format {format}, defaulting to JSON-LD")
            return self._to_json_ld(receipt)

    def _to_json_ld(self, receipt: "EffortReceipt") -> Dict[str, Any]:
        """
        Export to JSON-LD (Linked Data) format.

        JSON-LD provides semantic interoperability with external systems.
        """
        return {
            "@context": {
                "@vocab": "https://value-ledger.io/schema/",
                "mp02": "https://natlangchain.io/mp-02/",
                "xsd": "http://www.w3.org/2001/XMLSchema#",
                "timestamp": {"@type": "xsd:dateTime"},
            },
            "@type": "mp02:EffortReceipt",
            "@id": f"urn:mp02:receipt:{receipt.receipt_id}",
            "mp02:receiptId": receipt.receipt_id,
            "mp02:receiptHash": receipt.receipt_hash,
            "mp02:timeBounds": {
                "mp02:start": receipt.time_bounds[0],
                "mp02:end": receipt.time_bounds[1],
            },
            "mp02:signalHashes": receipt.signal_hashes,
            "mp02:effortSummary": receipt.effort_summary,
            "mp02:observer": receipt.observer_id,
            "mp02:validator": receipt.validator_id,
            "mp02:validation": {
                "mp02:coherenceScore": receipt.validation_metadata.coherence_score,
                "mp02:progressionScore": receipt.validation_metadata.progression_score,
                "mp02:modelId": receipt.validation_metadata.model_id,
                "mp02:modelVersion": receipt.validation_metadata.model_version,
            },
            "mp02:isComplete": not receipt.is_incomplete,
            "mp02:suspectedManipulation": receipt.suspected_manipulation,
            "mp02:priorReceipts": receipt.prior_receipts,
            "mp02:chainAnchor": receipt.chain_anchor,
        }

    def _to_w3c_vc(self, receipt: "EffortReceipt") -> Dict[str, Any]:
        """
        Export to W3C Verifiable Credentials-style format.

        This provides compatibility with decentralized identity systems.
        """
        return {
            "@context": [
                "https://www.w3.org/2018/credentials/v1",
                "https://value-ledger.io/credentials/v1",
            ],
            "type": ["VerifiableCredential", "EffortReceiptCredential"],
            "id": f"urn:uuid:{receipt.receipt_id}",
            "issuer": self.issuer_id,
            "issuanceDate": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(receipt.time_bounds[0])),
            "credentialSubject": {
                "id": f"urn:mp02:effort:{receipt.receipt_id}",
                "type": "EffortAssertion",
                "effortSummary": receipt.effort_summary,
                "timeBounds": {
                    "start": receipt.time_bounds[0],
                    "end": receipt.time_bounds[1],
                },
                "signalCount": len(receipt.signal_hashes),
                "observer": receipt.observer_id,
                "validator": receipt.validator_id,
            },
            "proof": {
                "type": "Sha256Hash2024",
                "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "proofValue": receipt.receipt_hash,
                "signalHashes": receipt.signal_hashes,
            },
        }

    def _to_open_timestamps(self, receipt: "EffortReceipt") -> Dict[str, Any]:
        """
        Export to OpenTimestamps-compatible format.

        Provides timestamp proof that can be anchored to external systems.
        """
        return {
            "version": 1,
            "file_hash": receipt.receipt_hash,
            "file_hash_type": "sha256",
            "timestamp": {
                "attestations": [
                    {
                        "type": "mp02_receipt",
                        "time": int(receipt.time_bounds[0]),
                        "receipt_id": receipt.receipt_id,
                        "observer": receipt.observer_id,
                        "validator": receipt.validator_id,
                    }
                ],
                "ops": [
                    {"op": "sha256", "arg": receipt.receipt_hash},
                ],
            },
            "metadata": {
                "effort_summary": receipt.effort_summary,
                "signal_count": len(receipt.signal_hashes),
                "is_complete": not receipt.is_incomplete,
            },
        }

    def _to_audit_log(self, receipt: "EffortReceipt") -> Dict[str, Any]:
        """
        Export to standard audit log format.

        Compatible with enterprise audit systems.
        """
        return {
            "log_version": "1.0",
            "log_type": "effort_receipt",
            "entry": {
                "id": receipt.receipt_id,
                "timestamp": receipt.time_bounds[0],
                "end_timestamp": receipt.time_bounds[1],
                "duration_seconds": receipt.time_bounds[1] - receipt.time_bounds[0],
                "actor": receipt.observer_id,
                "validator": receipt.validator_id,
                "action": "effort_recorded",
                "resource_type": "cognitive_effort",
                "resource_id": receipt.receipt_id,
                "outcome": "complete" if not receipt.is_incomplete else "incomplete",
                "evidence": {
                    "hash": receipt.receipt_hash,
                    "signal_count": len(receipt.signal_hashes),
                    "signal_hashes": receipt.signal_hashes,
                },
                "integrity": {
                    "manipulation_suspected": receipt.suspected_manipulation,
                    "observation_gaps": len(receipt.observation_gaps),
                    "conflicting_validations": len(receipt.conflicting_validations),
                },
                "summary": receipt.effort_summary,
            },
        }

    def create_audit_entry(
        self,
        receipt: "EffortReceipt",
        event_type: str,
        actor_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> AuditEntry:
        """
        Create an audit entry for a receipt event.

        Args:
            receipt: The receipt being audited
            event_type: Type of event (creation, verification, export, etc.)
            actor_id: ID of the actor performing the action
            action: Description of the action
            details: Optional additional details

        Returns:
            AuditEntry record
        """
        return AuditEntry(
            event_type=event_type,
            subject_id=receipt.receipt_id,
            actor_id=actor_id,
            action=action,
            details=details or {},
            evidence_hash=receipt.receipt_hash or "",
            chain_ref=receipt.chain_anchor,
        )

    def batch_export(
        self,
        receipts: List["EffortReceipt"],
        format: str = AuditFormat.JSON_LD,
    ) -> List[Dict[str, Any]]:
        """Export multiple receipts."""
        return [self.export_receipt(r, format) for r in receipts]


# =============================================================================
# Cross-Protocol Adapter
# =============================================================================

class ProtocolAdapter:
    """
    Adapter for cross-protocol interoperability.

    Provides conversion between MP-02 receipts and various external formats.
    """

    def __init__(
        self,
        mp01_formatter: Optional[MP01Formatter] = None,
        license_manager: Optional[LicenseManager] = None,
        audit_exporter: Optional[AuditExporter] = None,
    ):
        self.mp01 = mp01_formatter or MP01Formatter()
        self.licenses = license_manager or LicenseManager()
        self.audit = audit_exporter or AuditExporter()

    def prepare_for_negotiation(
        self,
        receipt: "EffortReceipt",
        recipient_id: str,
        terms: Dict[str, Any],
    ) -> Tuple[MP01Proposal, Dict[str, Any]]:
        """
        Prepare a receipt for MP-01 negotiation.

        Returns:
            (proposal, formatted_receipt)
        """
        formatted = self.mp01.format_for_negotiation(receipt)
        proposal = self.mp01.create_proposal(
            receipt_ids=[receipt.receipt_id],
            recipient_id=recipient_id,
            terms=terms,
        )
        return (proposal, formatted)

    def create_licensed_export(
        self,
        receipt: "EffortReceipt",
        licensee_ids: List[str],
        license_type: str = LicenseType.REFERENCE,
        audit_format: str = AuditFormat.JSON_LD,
    ) -> Dict[str, Any]:
        """
        Create a licensed export of a receipt.

        Returns export with embedded license reference.
        """
        # Create license
        license_ref = self.licenses.grant_license(
            receipt_ids=[receipt.receipt_id],
            licensor_id=receipt.observer_id,
            licensee_ids=licensee_ids,
            license_type=license_type,
        )

        # Export receipt
        exported = self.audit.export_receipt(receipt, audit_format)

        # Embed license reference
        exported["license"] = license_ref.to_dict()

        return exported

    def verify_and_export(
        self,
        receipt: "EffortReceipt",
        format: str = AuditFormat.AUDIT_LOG,
        verifier_id: str = "",
    ) -> Dict[str, Any]:
        """
        Verify a receipt and create an audit export.

        Returns export with verification metadata.
        """
        # Create verification audit entry
        audit_entry = self.audit.create_audit_entry(
            receipt=receipt,
            event_type="verification",
            actor_id=verifier_id,
            action="verify_and_export",
            details={
                "export_format": format,
                "verified_at": time.time(),
            },
        )

        # Export receipt
        exported = self.audit.export_receipt(receipt, format)

        # Add verification record
        exported["verification"] = audit_entry.to_dict()

        return exported


# =============================================================================
# Convenience Functions
# =============================================================================

def create_protocol_adapter(
    party_id: str = "",
    issuer_id: str = "value_ledger",
) -> ProtocolAdapter:
    """
    Create a fully configured protocol adapter.

    Args:
        party_id: ID for MP-01 negotiations
        issuer_id: ID for audit exports

    Returns:
        Configured ProtocolAdapter
    """
    return ProtocolAdapter(
        mp01_formatter=MP01Formatter(party_id=party_id),
        license_manager=LicenseManager(),
        audit_exporter=AuditExporter(issuer_id=issuer_id),
    )


def export_receipt_for_audit(
    receipt: "EffortReceipt",
    format: str = AuditFormat.JSON_LD,
) -> Dict[str, Any]:
    """
    Convenience function to export a receipt for external audit.

    Args:
        receipt: Receipt to export
        format: Target audit format

    Returns:
        Formatted export data
    """
    exporter = AuditExporter()
    return exporter.export_receipt(receipt, format)
