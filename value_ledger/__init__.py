"""
Value Ledger - Evidentiary accounting layer for cognitive effort in Agent-OS

This module provides the core functionality for tracking and proving the value
of cognitive work, including ideas, effort, time, novelty, failures, and strategic insights.
"""

from .core import ValueLedger, ValueVector, LedgerEntry
from .heuristics import HeuristicEngine, ScoringContext
from .integration import IntentLogConnector, IntentEvent, create_intentlog_listener
from .memory_vault_hook import MemoryVaultHook

__version__ = "0.1.0"

__all__ = [
    # Core classes
    "ValueLedger",
    "ValueVector",
    "LedgerEntry",
    # Heuristics
    "HeuristicEngine",
    "ScoringContext",
    # Integration
    "IntentLogConnector",
    "IntentEvent",
    "create_intentlog_listener",
    # Memory Vault integration
    "MemoryVaultHook",
]
