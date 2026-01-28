# tests/test_cli.py
"""
Tests for Value Ledger CLI Module.

Tests cover:
- LedgerCLI class operations
- Command parsing
- Stats, query, export, show, revoke, proof commands
- Error handling
"""

import json
import os
import sys
import tempfile
import pytest
from io import StringIO
from unittest.mock import patch, MagicMock

from value_ledger.cli import (
    LedgerCLI,
    format_timestamp,
    format_value_vector,
    main,
    print_usage,
    demo,
)
from value_ledger.core import ValueLedger, ValueVector


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestFormatTimestamp:
    """Tests for timestamp formatting."""

    def test_format_timestamp(self):
        """Timestamp is formatted correctly."""
        # Use a known timestamp: 2023-01-01 00:00:00 UTC
        ts = 1672531200.0
        result = format_timestamp(ts)
        # Should be a date-time string
        assert "2023" in result or "2022" in result  # Timezone dependent
        assert ":" in result

    def test_format_timestamp_float(self):
        """Float timestamp is handled."""
        ts = 1700000000.123
        result = format_timestamp(ts)
        assert isinstance(result, str)


class TestFormatValueVector:
    """Tests for value vector formatting."""

    def test_format_value_vector(self):
        """Value vector is formatted correctly."""
        vec = ValueVector(t=1.5, e=2.3, n=4.7, f=0.5, r=1.0, s=3.2, u=2.1)
        result = format_value_vector(vec)
        assert "T=1.5" in result
        assert "E=2.3" in result
        assert "N=4.7" in result
        assert "F=0.5" in result
        assert "R=1.0" in result
        assert "S=3.2" in result
        assert "U=2.1" in result

    def test_format_value_vector_zeros(self):
        """Zero values are formatted correctly."""
        vec = ValueVector()
        result = format_value_vector(vec)
        assert "T=0.0" in result


# =============================================================================
# LedgerCLI Tests
# =============================================================================


class TestLedgerCLI:
    """Tests for LedgerCLI class."""

    @pytest.fixture
    def cli_with_entries(self):
        """Create CLI with pre-populated ledger."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        # Create ledger and add entries
        ledger = ValueLedger(path)
        ledger.accrue(
            intent_id="test-intent-1",
            initial_vector={"t": 5.0, "e": 3.0},
        )
        ledger.accrue(
            intent_id="test-intent-2",
            initial_vector={"t": 8.0, "e": 6.0, "n": 4.0},
        )

        cli = LedgerCLI(path)
        yield cli

        # Cleanup
        if os.path.exists(path):
            os.remove(path)

    @pytest.fixture
    def empty_cli(self):
        """Create CLI with empty ledger."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name

        cli = LedgerCLI(path)
        yield cli

        # Cleanup
        if os.path.exists(path):
            os.remove(path)


class TestLedgerCLIStats:
    """Tests for stats command."""

    @pytest.fixture
    def cli_with_entries(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        ledger = ValueLedger(path)
        ledger.accrue(intent_id="test-1", initial_vector={"t": 5.0, "e": 3.0})
        ledger.accrue(intent_id="test-2", initial_vector={"t": 8.0, "e": 6.0})
        cli = LedgerCLI(path)
        yield cli
        if os.path.exists(path):
            os.remove(path)

    def test_stats_with_entries(self, cli_with_entries, capsys):
        """Stats command shows statistics."""
        cli_with_entries.stats()
        captured = capsys.readouterr()
        assert "Total entries:" in captured.out or "entries" in captured.out.lower()

    def test_stats_empty_ledger(self, capsys):
        """Stats on empty ledger shows empty message."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            cli = LedgerCLI(path)
            cli.stats()
            captured = capsys.readouterr()
            assert "empty" in captured.out.lower() or "0" in captured.out
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestLedgerCLIQuery:
    """Tests for query command."""

    @pytest.fixture
    def cli_with_entries(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        ledger = ValueLedger(path)
        ledger.accrue(intent_id="query-test-1", initial_vector={"t": 5.0})
        ledger.accrue(intent_id="query-test-2", initial_vector={"t": 8.0})
        cli = LedgerCLI(path)
        yield cli
        if os.path.exists(path):
            os.remove(path)

    def test_query_all(self, cli_with_entries, capsys):
        """Query without filters returns all entries."""
        cli_with_entries.query()
        captured = capsys.readouterr()
        assert "query-test" in captured.out or "2 entries" in captured.out

    def test_query_by_intent_id(self, cli_with_entries, capsys):
        """Query by intent_id filters correctly."""
        cli_with_entries.query(intent_id="query-test-1")
        captured = capsys.readouterr()
        assert "query-test-1" in captured.out or "1 entry" in captured.out.lower()

    def test_query_by_status(self, cli_with_entries, capsys):
        """Query by status filters correctly."""
        cli_with_entries.query(status="active")
        captured = capsys.readouterr()
        # All entries should be active by default
        assert "entries" in captured.out.lower()

    def test_query_no_matches(self, cli_with_entries, capsys):
        """Query with no matches shows appropriate message."""
        cli_with_entries.query(intent_id="nonexistent")
        captured = capsys.readouterr()
        assert "No entries" in captured.out

    def test_query_with_limit(self, cli_with_entries, capsys):
        """Query respects limit parameter."""
        cli_with_entries.query(limit=1)
        captured = capsys.readouterr()
        assert "1 entry" in captured.out.lower() or "Showing 1" in captured.out


class TestLedgerCLIExport:
    """Tests for export command."""

    @pytest.fixture
    def cli_with_entries(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        ledger = ValueLedger(path)
        ledger.accrue(intent_id="export-test-1", initial_vector={"t": 5.0})
        cli = LedgerCLI(path)
        yield cli
        if os.path.exists(path):
            os.remove(path)

    def test_export_json(self, cli_with_entries, capsys):
        """Export to JSON format."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            cli_with_entries.export(output_path, format="json")
            captured = capsys.readouterr()
            assert "Exported" in captured.out or os.path.exists(output_path)

            # Verify JSON is valid
            with open(output_path) as f:
                data = json.load(f)
                assert isinstance(data, list)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_export_csv(self, cli_with_entries, capsys):
        """Export to CSV format."""
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            output_path = f.name

        try:
            cli_with_entries.export(output_path, format="csv")
            captured = capsys.readouterr()
            assert "Exported" in captured.out or os.path.exists(output_path)

            # Verify CSV has content
            with open(output_path) as f:
                content = f.read()
                assert "intent" in content.lower() or "id" in content.lower()
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_export_merkle(self, cli_with_entries, capsys):
        """Export Merkle tree data."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        try:
            cli_with_entries.export(output_path, format="merkle")
            captured = capsys.readouterr()
            assert "Exported" in captured.out or "root" in captured.out.lower()
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_export_invalid_format(self, cli_with_entries, capsys):
        """Invalid format shows error."""
        cli_with_entries.export("output.txt", format="invalid")
        captured = capsys.readouterr()
        assert "Unknown format" in captured.out or "unknown" in captured.out.lower()


class TestLedgerCLIShow:
    """Tests for show command."""

    @pytest.fixture
    def cli_with_entry(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        ledger = ValueLedger(path)
        entry_id = ledger.accrue(intent_id="show-test", initial_vector={"t": 5.0})
        cli = LedgerCLI(path)
        yield cli, entry_id
        if os.path.exists(path):
            os.remove(path)

    def test_show_entry(self, cli_with_entry, capsys):
        """Show displays entry details."""
        cli, entry_id = cli_with_entry
        cli.show(entry_id)
        captured = capsys.readouterr()
        assert entry_id[:8] in captured.out or "show-test" in captured.out

    def test_show_nonexistent(self, cli_with_entry, capsys):
        """Show nonexistent entry shows error."""
        cli, _ = cli_with_entry
        cli.show("nonexistent-id")
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "Entry not found" in captured.out


class TestLedgerCLIRevoke:
    """Tests for revoke command."""

    @pytest.fixture
    def cli_with_entry(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        ledger = ValueLedger(path)
        entry_id = ledger.accrue(intent_id="revoke-test", initial_vector={"t": 5.0})
        cli = LedgerCLI(path)
        yield cli, entry_id
        if os.path.exists(path):
            os.remove(path)

    def test_revoke_entry(self, cli_with_entry, capsys):
        """Revoke marks entry as revoked."""
        cli, entry_id = cli_with_entry
        cli.revoke(entry_id, reason="Test revocation")
        captured = capsys.readouterr()
        assert "Revoked" in captured.out or "revoked" in captured.out.lower()

    def test_revoke_nonexistent(self, cli_with_entry, capsys):
        """Revoke nonexistent entry shows error."""
        cli, _ = cli_with_entry
        cli.revoke("nonexistent-id")
        captured = capsys.readouterr()
        assert "failed" in captured.out.lower() or "not found" in captured.out.lower()


class TestLedgerCLIProof:
    """Tests for proof command."""

    @pytest.fixture
    def cli_with_entry(self):
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        ledger = ValueLedger(path)
        entry_id = ledger.accrue(
            intent_id="proof-test",
            initial_vector={"t": 5.0},
            content_for_proof="Test content for hashing",
        )
        cli = LedgerCLI(path)
        yield cli, entry_id
        if os.path.exists(path):
            os.remove(path)

    def test_proof_entry(self, cli_with_entry, capsys):
        """Proof generates existence proof."""
        cli, entry_id = cli_with_entry
        cli.proof(entry_id)
        captured = capsys.readouterr()
        # Should contain proof-related info
        assert "proof" in captured.out.lower() or "root" in captured.out.lower()

    def test_proof_nonexistent(self, cli_with_entry, capsys):
        """Proof for nonexistent entry shows error."""
        cli, _ = cli_with_entry
        cli.proof("nonexistent-id")
        captured = capsys.readouterr()
        assert "not found" in captured.out.lower() or "Entry not found" in captured.out


# =============================================================================
# Main Function Tests
# =============================================================================


class TestMainFunction:
    """Tests for main() entry point."""

    def test_help_flag(self, capsys):
        """--help shows usage."""
        with patch.object(sys, "argv", ["value-ledger", "--help"]):
            main()
        captured = capsys.readouterr()
        assert "Usage" in captured.out or "usage" in captured.out.lower()

    def test_no_args_shows_help(self, capsys):
        """No arguments shows usage."""
        with patch.object(sys, "argv", ["value-ledger"]):
            main()
        captured = capsys.readouterr()
        assert "Usage" in captured.out or "usage" in captured.out.lower()

    def test_unknown_command(self, capsys):
        """Unknown command shows error."""
        with patch.object(sys, "argv", ["value-ledger", "unknown_command"]):
            main()
        captured = capsys.readouterr()
        assert "Unknown command" in captured.out or "unknown" in captured.out.lower()

    def test_stats_command(self, capsys):
        """Stats command via main()."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with patch.object(sys, "argv", ["value-ledger", "--ledger", path, "stats"]):
                main()
            captured = capsys.readouterr()
            # Should show some output (empty ledger is ok)
            assert captured.out  # Some output was produced
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_query_command_with_options(self, capsys):
        """Query command with options via main()."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with patch.object(
                sys,
                "argv",
                [
                    "value-ledger",
                    "--ledger",
                    path,
                    "query",
                    "--intent-id",
                    "test",
                    "--status",
                    "active",
                    "--limit",
                    "10",
                ],
            ):
                main()
            captured = capsys.readouterr()
            # Should run without error
            assert "error" not in captured.out.lower() or "No entries" in captured.out
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_show_command_missing_arg(self, capsys):
        """Show command without entry_id shows usage."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with patch.object(sys, "argv", ["value-ledger", "--ledger", path, "show"]):
                main()
            captured = capsys.readouterr()
            assert "Usage" in captured.out or "usage" in captured.out.lower()
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_export_command_missing_arg(self, capsys):
        """Export command without output path shows usage."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with patch.object(sys, "argv", ["value-ledger", "--ledger", path, "export"]):
                main()
            captured = capsys.readouterr()
            assert "Usage" in captured.out or "usage" in captured.out.lower()
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_revoke_command_missing_arg(self, capsys):
        """Revoke command without entry_id shows usage."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with patch.object(sys, "argv", ["value-ledger", "--ledger", path, "revoke"]):
                main()
            captured = capsys.readouterr()
            assert "Usage" in captured.out or "usage" in captured.out.lower()
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_proof_command_missing_arg(self, capsys):
        """Proof command without entry_id shows usage."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with patch.object(sys, "argv", ["value-ledger", "--ledger", path, "proof"]):
                main()
            captured = capsys.readouterr()
            assert "Usage" in captured.out or "usage" in captured.out.lower()
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_query_invalid_since_value(self, capsys):
        """Query with invalid --since shows error."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with patch.object(
                sys,
                "argv",
                ["value-ledger", "--ledger", path, "query", "--since", "not-a-number"],
            ):
                main()
            captured = capsys.readouterr()
            assert "Invalid" in captured.out or "Error" in captured.out
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_query_invalid_limit_value(self, capsys):
        """Query with invalid --limit shows error."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            with patch.object(
                sys,
                "argv",
                ["value-ledger", "--ledger", path, "query", "--limit", "not-an-int"],
            ):
                main()
            captured = capsys.readouterr()
            assert "Invalid" in captured.out or "Error" in captured.out
        finally:
            if os.path.exists(path):
                os.remove(path)


class TestDemoCommand:
    """Tests for demo command."""

    def test_demo_runs(self, capsys):
        """Demo command runs without error."""
        # Demo creates its own temp file
        demo()
        captured = capsys.readouterr()
        # Should show demo output
        assert "Demo" in captured.out or "entry" in captured.out.lower()

    def test_demo_via_main(self, capsys):
        """Demo command via main()."""
        with patch.object(sys, "argv", ["value-ledger", "demo"]):
            main()
        captured = capsys.readouterr()
        assert captured.out  # Some output was produced


class TestPrintUsage:
    """Tests for usage printing."""

    def test_print_usage(self, capsys):
        """Usage is printed correctly."""
        print_usage()
        captured = capsys.readouterr()
        assert "Usage" in captured.out or "value-ledger" in captured.out
        assert "stats" in captured.out
        assert "query" in captured.out
        assert "export" in captured.out


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge case tests."""

    def test_ledger_lazy_loading(self):
        """Ledger is lazily loaded on first access."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            cli = LedgerCLI(path)
            assert cli._ledger is None
            _ = cli.ledger  # Access property
            assert cli._ledger is not None
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_cli_with_custom_path(self):
        """CLI can be initialized with custom path."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            cli = LedgerCLI(path)
            assert cli.ledger_path == path
        finally:
            if os.path.exists(path):
                os.remove(path)
