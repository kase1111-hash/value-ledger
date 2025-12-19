"""
value_ledger/cli.py

Admin CLI for Value Ledger management.
Phase 1 - 17.8: query, export, stats commands
"""

import json
import csv
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

# Simple argument parsing without external dependencies
from .core import ValueLedger, ValueVector


def format_timestamp(ts: float) -> str:
    """Format Unix timestamp to human-readable."""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def format_value_vector(v: ValueVector) -> str:
    """Format value vector for display."""
    return f"T={v.t:.1f} E={v.e:.1f} N={v.n:.1f} F={v.f:.1f} R={v.r:.1f} S={v.s:.1f} U={v.u:.1f}"


class LedgerCLI:
    """CLI interface for Value Ledger administration."""

    def __init__(self, ledger_path: str = "ledger.jsonl"):
        self.ledger_path = ledger_path
        self._ledger: Optional[ValueLedger] = None

    @property
    def ledger(self) -> ValueLedger:
        if self._ledger is None:
            self._ledger = ValueLedger(self.ledger_path)
        return self._ledger

    def query(
        self,
        intent_id: Optional[str] = None,
        status: Optional[str] = None,
        since: Optional[float] = None,
        limit: int = 50,
    ) -> None:
        """Query ledger entries with filters."""
        entries = self.ledger.entries

        # Apply filters
        if intent_id:
            entries = [e for e in entries if e.intent_id == intent_id]
        if status:
            entries = [e for e in entries if e.status == status]
        if since:
            entries = [e for e in entries if e.timestamp >= since]

        # Sort by timestamp (newest first) and limit
        entries = sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]

        if not entries:
            print("No entries found matching criteria.")
            return

        print(f"\n{'ID':<20} {'Timestamp':<20} {'Status':<8} {'Intent':<20} {'Value':<40}")
        print("-" * 110)

        for entry in entries:
            print(
                f"{entry.id[:18]:<20} "
                f"{format_timestamp(entry.timestamp):<20} "
                f"{entry.status:<8} "
                f"{entry.intent_id[:18]:<20} "
                f"{format_value_vector(entry.value_vector):<40}"
            )

        print(f"\nShowing {len(entries)} entries")

    def export(
        self,
        output_path: str,
        format: str = "json",
        intent_id: Optional[str] = None,
        include_proofs: bool = False,
    ) -> None:
        """Export ledger data to file."""
        entries = self.ledger.entries

        if intent_id:
            entries = [e for e in entries if e.intent_id == intent_id]

        if format == "json":
            self._export_json(entries, output_path, include_proofs)
        elif format == "csv":
            self._export_csv(entries, output_path)
        elif format == "merkle":
            self._export_merkle(output_path)
        else:
            print(f"Unknown format: {format}. Use json, csv, or merkle.")
            return

        print(f"Exported {len(entries)} entries to {output_path}")

    def _export_json(self, entries, output_path: str, include_proofs: bool) -> None:
        data = []
        for entry in entries:
            entry_data = entry.dict()
            if include_proofs:
                entry_data["merkle_proof"] = self.ledger.get_merkle_proof(entry.id)
            data.append(entry_data)

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def _export_csv(self, entries, output_path: str) -> None:
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "id", "timestamp", "intent_id", "status",
                "t", "e", "n", "f", "r", "s", "total",
                "content_hash", "merkle_ref"
            ])
            for entry in entries:
                v = entry.value_vector
                writer.writerow([
                    entry.id,
                    format_timestamp(entry.timestamp),
                    entry.intent_id,
                    entry.status,
                    f"{v.t:.2f}", f"{v.e:.2f}", f"{v.n:.2f}",
                    f"{v.f:.2f}", f"{v.r:.2f}", f"{v.s:.2f}",
                    f"{v.total():.2f}",
                    entry.proof.content_hash or "",
                    entry.proof.merkle_ref or "",
                ])

    def _export_merkle(self, output_path: str) -> None:
        data = {
            "merkle_root": self.ledger.get_merkle_root(),
            "total_entries": len(self.ledger.entries),
            "leaves": self.ledger.merkle_tree.leaves,
        }
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def stats(self) -> None:
        """Show ledger statistics."""
        entries = self.ledger.entries

        if not entries:
            print("Ledger is empty.")
            return

        # Basic counts
        total = len(entries)
        by_status = {}
        for e in entries:
            by_status[e.status] = by_status.get(e.status, 0) + 1

        # Intent counts
        intents = set(e.intent_id for e in entries)

        # Value aggregation
        total_value = sum(e.value_vector.total() for e in entries)
        avg_value = total_value / total if total > 0 else 0

        # Value by dimension
        sum_t = sum(e.value_vector.t for e in entries)
        sum_e = sum(e.value_vector.e for e in entries)
        sum_n = sum(e.value_vector.n for e in entries)
        sum_f = sum(e.value_vector.f for e in entries)
        sum_r = sum(e.value_vector.r for e in entries)
        sum_s = sum(e.value_vector.s for e in entries)
        sum_u = sum(e.value_vector.u for e in entries)

        # Time range
        oldest = min(entries, key=lambda e: e.timestamp)
        newest = max(entries, key=lambda e: e.timestamp)

        # Top intents by value
        intent_values = {}
        for e in entries:
            if e.status == "active":
                intent_values[e.intent_id] = intent_values.get(e.intent_id, 0) + e.value_vector.total()
        top_intents = sorted(intent_values.items(), key=lambda x: x[1], reverse=True)[:5]

        print("\n" + "=" * 60)
        print("VALUE LEDGER STATISTICS")
        print("=" * 60)

        print(f"\n📊 OVERVIEW")
        print(f"   Total Entries: {total}")
        print(f"   Unique Intents: {len(intents)}")
        print(f"   Merkle Root: {self.ledger.get_merkle_root()[:16]}..." if self.ledger.get_merkle_root() else "   Merkle Root: None")

        print(f"\n📈 STATUS BREAKDOWN")
        for status, count in sorted(by_status.items()):
            pct = (count / total) * 100
            print(f"   {status}: {count} ({pct:.1f}%)")

        print(f"\n💰 VALUE SUMMARY")
        print(f"   Total Value: {total_value:.1f}")
        print(f"   Average per Entry: {avg_value:.1f}")

        print(f"\n📊 VALUE BY DIMENSION")
        print(f"   Time (T):         {sum_t:>8.1f}")
        print(f"   Effort (E):       {sum_e:>8.1f}")
        print(f"   Novelty (N):      {sum_n:>8.1f}")
        print(f"   Failure (F):      {sum_f:>8.1f}")
        print(f"   Risk (R):         {sum_r:>8.1f}")
        print(f"   Strategy (S):     {sum_s:>8.1f}")
        print(f"   Reusability (U):  {sum_u:>8.1f}")

        print(f"\n⏰ TIME RANGE")
        print(f"   Oldest: {format_timestamp(oldest.timestamp)}")
        print(f"   Newest: {format_timestamp(newest.timestamp)}")

        if top_intents:
            print(f"\n🏆 TOP INTENTS BY VALUE")
            for intent, value in top_intents:
                print(f"   {intent[:30]}: {value:.1f}")

        print("\n" + "=" * 60)

    def show(self, entry_id: str) -> None:
        """Show detailed information for a single entry."""
        entry = self.ledger.get_entry(entry_id)
        if not entry:
            # Try prefix match
            matches = [e for e in self.ledger.entries if e.id.startswith(entry_id)]
            if len(matches) == 1:
                entry = matches[0]
            elif len(matches) > 1:
                print(f"Multiple entries match prefix '{entry_id}':")
                for m in matches[:5]:
                    print(f"  {m.id}")
                return
            else:
                print(f"Entry not found: {entry_id}")
                return

        print(f"\n{'='*60}")
        print(f"ENTRY: {entry.id}")
        print(f"{'='*60}")

        print(f"\n📋 BASIC INFO")
        print(f"   Intent ID:  {entry.intent_id}")
        print(f"   Timestamp:  {format_timestamp(entry.timestamp)}")
        print(f"   Status:     {entry.status}")
        if entry.memory_hash:
            print(f"   Memory Hash: {entry.memory_hash}")

        print(f"\n💰 VALUE VECTOR")
        v = entry.value_vector
        print(f"   Time (T):         {v.t:>6.2f}")
        print(f"   Effort (E):       {v.e:>6.2f}")
        print(f"   Novelty (N):      {v.n:>6.2f}")
        print(f"   Failure (F):      {v.f:>6.2f}")
        print(f"   Risk (R):         {v.r:>6.2f}")
        print(f"   Strategy (S):     {v.s:>6.2f}")
        print(f"   Reusability (U):  {v.u:>6.2f}")
        print(f"   ─────────────────────")
        print(f"   TOTAL:            {v.total():>6.2f}")

        print(f"\n🔐 PROOF DATA")
        print(f"   Content Hash:    {entry.proof.content_hash or 'None'}")
        print(f"   Timestamp Proof: {entry.proof.timestamp_proof or 'None'}")
        print(f"   Merkle Ref:      {entry.proof.merkle_ref or 'None'}")

        if entry.status == "revoked":
            print(f"\n⛔ REVOCATION INFO")
            print(f"   Revoked At: {format_timestamp(entry.revoked_at) if entry.revoked_at else 'N/A'}")
            print(f"   Revoked By: {entry.revoked_by or 'N/A'}")
            print(f"   Reason:     {entry.revocation_reason or 'N/A'}")

        if entry.parent_id:
            print(f"\n🔗 LINEAGE")
            print(f"   Parent ID: {entry.parent_id}")
            chain = self.ledger.get_chain(entry.id)
            print(f"   Chain Length: {len(chain)}")

        if entry.metadata:
            print(f"\n📎 METADATA")
            for key, value in entry.metadata.items():
                print(f"   {key}: {value}")

        print()

    def revoke(
        self,
        entry_id: str,
        reason: Optional[str] = None,
        revoked_by: Optional[str] = None,
        cascade: bool = False,
    ) -> None:
        """Revoke an entry."""
        success = self.ledger.revoke(
            entry_id=entry_id,
            reason=reason,
            revoked_by=revoked_by,
            revoke_children=cascade,
        )
        if success:
            print(f"Entry {entry_id[:16]}... revoked successfully.")
        else:
            print(f"Failed to revoke entry {entry_id}")

    def proof(self, entry_id: str) -> None:
        """Export existence proof for an entry."""
        proof = self.ledger.export_existence_proof(entry_id)
        if proof:
            print(json.dumps(proof, indent=2))
        else:
            print(f"Entry not found: {entry_id}")


def print_usage():
    """Print CLI usage information."""
    print("""
Value Ledger CLI - Admin Commands

Usage: python -m value_ledger.cli <command> [options]

Commands:
  stats                           Show ledger statistics
  query [options]                 Query ledger entries
    --intent-id <id>              Filter by intent ID
    --status <status>             Filter by status (active/frozen/revoked)
    --since <timestamp>           Filter entries after timestamp
    --limit <n>                   Limit results (default: 50)

  show <entry_id>                 Show detailed entry info
  export <output> [options]       Export ledger data
    --format <fmt>                Format: json, csv, merkle (default: json)
    --intent-id <id>              Filter by intent
    --include-proofs              Include Merkle proofs in JSON export

  revoke <entry_id> [options]     Revoke an entry
    --reason <text>               Reason for revocation
    --by <identifier>             Who is revoking
    --cascade                     Also revoke child entries

  proof <entry_id>                Export existence proof

  demo                            Run demo with sample data

Options:
  --ledger <path>                 Path to ledger file (default: ledger.jsonl)
  --help                          Show this help message

Examples:
  python -m value_ledger.cli stats
  python -m value_ledger.cli query --status active --limit 10
  python -m value_ledger.cli export output.json --format json
  python -m value_ledger.cli show abc123
  python -m value_ledger.cli revoke abc123 --reason "Duplicate entry"
""")


def demo():
    """Run demo to create sample data."""
    from .core import ValueLedger
    from .heuristics import ScoringContext
    import time

    print("=== Value Ledger Demo ===\n")

    ledger = ValueLedger("demo_ledger.jsonl")

    # Create some sample entries
    print("Creating sample entries...")

    entry1 = ledger.accrue(
        intent_id="demo-intent-001",
        initial_vector={"t": 8.0, "e": 6.5, "n": 7.0, "f": 3.0, "r": 2.0, "s": 5.0},
        memory_hash="sha256:abc123...",
        metadata={"task": "exploring novel architecture"},
        content_for_proof="Sample content for proof generation",
    )
    print(f"  Created entry: {entry1[:16]}...")

    entry2 = ledger.accrue(
        intent_id="demo-intent-002",
        initial_vector={"t": 3.0, "e": 4.0, "n": 9.0, "f": 1.0, "r": 0.5, "s": 2.0},
        metadata={"task": "quick fix"},
        content_for_proof="Another piece of work",
    )
    print(f"  Created entry: {entry2[:16]}...")

    # Show stats
    print("\n")
    cli = LedgerCLI("demo_ledger.jsonl")
    cli.stats()

    # Show an entry
    print("\nDetailed view of first entry:")
    cli.show(entry1)

    # Export proof
    print("Existence proof for first entry:")
    cli.proof(entry1)


def main():
    """Main CLI entry point."""
    args = sys.argv[1:]

    if not args or args[0] in ["--help", "-h", "help"]:
        print_usage()
        return

    # Parse global options
    ledger_path = "ledger.jsonl"
    if "--ledger" in args:
        idx = args.index("--ledger")
        ledger_path = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    command = args[0]
    args = args[1:]

    cli = LedgerCLI(ledger_path)

    if command == "stats":
        cli.stats()

    elif command == "query":
        intent_id = None
        status = None
        since = None
        limit = 50

        i = 0
        while i < len(args):
            if args[i] == "--intent-id" and i + 1 < len(args):
                intent_id = args[i + 1]
                i += 2
            elif args[i] == "--status" and i + 1 < len(args):
                status = args[i + 1]
                i += 2
            elif args[i] == "--since" and i + 1 < len(args):
                since = float(args[i + 1])
                i += 2
            elif args[i] == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
            else:
                i += 1

        cli.query(intent_id=intent_id, status=status, since=since, limit=limit)

    elif command == "show":
        if not args:
            print("Usage: show <entry_id>")
            return
        cli.show(args[0])

    elif command == "export":
        if not args:
            print("Usage: export <output_path> [--format json|csv|merkle]")
            return

        output_path = args[0]
        format = "json"
        intent_id = None
        include_proofs = False

        i = 1
        while i < len(args):
            if args[i] == "--format" and i + 1 < len(args):
                format = args[i + 1]
                i += 2
            elif args[i] == "--intent-id" and i + 1 < len(args):
                intent_id = args[i + 1]
                i += 2
            elif args[i] == "--include-proofs":
                include_proofs = True
                i += 1
            else:
                i += 1

        cli.export(output_path, format=format, intent_id=intent_id, include_proofs=include_proofs)

    elif command == "revoke":
        if not args:
            print("Usage: revoke <entry_id> [--reason <text>] [--by <id>] [--cascade]")
            return

        entry_id = args[0]
        reason = None
        revoked_by = None
        cascade = False

        i = 1
        while i < len(args):
            if args[i] == "--reason" and i + 1 < len(args):
                reason = args[i + 1]
                i += 2
            elif args[i] == "--by" and i + 1 < len(args):
                revoked_by = args[i + 1]
                i += 2
            elif args[i] == "--cascade":
                cascade = True
                i += 1
            else:
                i += 1

        cli.revoke(entry_id, reason=reason, revoked_by=revoked_by, cascade=cascade)

    elif command == "proof":
        if not args:
            print("Usage: proof <entry_id>")
            return
        cli.proof(args[0])

    elif command == "demo":
        demo()

    else:
        print(f"Unknown command: {command}")
        print_usage()


# Export for use in specs
query = LedgerCLI.query
export = LedgerCLI.export
stats = LedgerCLI.stats


if __name__ == "__main__":
    main()
