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

__version__ = "0.3.0"

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
]
