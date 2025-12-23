"""
Value Ledger - Evidentiary accounting layer for cognitive effort in Agent-OS

This module provides the core functionality for tracking and proving the value
of cognitive work, including ideas, effort, time, novelty, failures, strategic insights,
and reusability.
"""

from .core import (
    ValueLedger,
    ValueVector,
    LedgerEntry,
    ProofData,
    MerkleTree,
    ClockMonitor,
    SourceValidator,
    FailureModeHandler,
)
from .heuristics import (
    HeuristicEngine,
    ScoringContext,
    ReusabilityScorer,
)
from .integration import IntentLogConnector, IntentEvent, create_intentlog_listener
from .memory_vault_hook import MemoryVaultHook
from .receipt import (
    EffortSignal,
    EffortSegment,
    EffortReceipt,
    ValidationMetadata,
    VerificationResult,
    ReceiptBuilder,
    DefaultObserver,
    DefaultValidator,
    verify_third_party,
)
from .natlangchain import (
    NLCRecord,
    NLCClient,
    NatLangChainExporter,
    ProofOfUnderstandingValidator,
    anchor_receipt_to_nlc,
)
from .synth_mind import (
    CognitiveTier,
    TierChangeEvent,
    CognitiveTierContext,
    CognitiveTierScorer,
    SynthMindHook,
)
from .interruption import (
    InterruptionType,
    InterruptionEvent,
    InterruptionSummary,
    InterruptionTracker,
    BoundaryDaemonListener,
    MockBoundaryEmitter,
    calculate_effort_factor,
    create_boundary_daemon_hook,
)
from .privacy import (
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
)
from .compatibility import (
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

__version__ = "0.4.0"

__all__ = [
    # Core classes
    "ValueLedger",
    "ValueVector",
    "LedgerEntry",
    "ProofData",
    "MerkleTree",
    # Failure mode handling
    "ClockMonitor",
    "SourceValidator",
    "FailureModeHandler",
    # Heuristics
    "HeuristicEngine",
    "ScoringContext",
    "ReusabilityScorer",
    # Integration
    "IntentLogConnector",
    "IntentEvent",
    "create_intentlog_listener",
    # Memory Vault integration
    "MemoryVaultHook",
    # MP-02 Effort Receipt Protocol
    "EffortSignal",
    "EffortSegment",
    "EffortReceipt",
    "ValidationMetadata",
    "VerificationResult",
    "ReceiptBuilder",
    "DefaultObserver",
    "DefaultValidator",
    "verify_third_party",
    # NatLangChain integration
    "NLCRecord",
    "NLCClient",
    "NatLangChainExporter",
    "ProofOfUnderstandingValidator",
    "anchor_receipt_to_nlc",
    # Synth-Mind integration
    "CognitiveTier",
    "TierChangeEvent",
    "CognitiveTierContext",
    "CognitiveTierScorer",
    "SynthMindHook",
    # Boundary Daemon integration
    "InterruptionType",
    "InterruptionEvent",
    "InterruptionSummary",
    "InterruptionTracker",
    "BoundaryDaemonListener",
    "MockBoundaryEmitter",
    "calculate_effort_factor",
    "create_boundary_daemon_hook",
    # MP-02 Privacy & Agency Controls
    "PrivacyLevel",
    "ConsentStatus",
    "RevocationScope",
    "ObservationConsent",
    "EncryptedContent",
    "SignalEncryptor",
    "PrivacyPolicy",
    "PrivacyFilter",
    "ConsentRegistry",
    "AgencyController",
    "create_privacy_controller",
    "encrypt_signal_content",
    "decrypt_signal_content",
    # MP-02 External Compatibility
    "NegotiationStatus",
    "RatificationMethod",
    "MP01Proposal",
    "MP01Ratification",
    "MP01Formatter",
    "LicenseType",
    "LicenseReference",
    "DelegationRecord",
    "LicenseManager",
    "AuditFormat",
    "AuditEntry",
    "AuditExporter",
    "ProtocolAdapter",
    "create_protocol_adapter",
    "export_receipt_for_audit",
]
