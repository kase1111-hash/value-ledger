# Remediation Plan — Value Ledger Vibe-Check Audit v2.0

**Project:** value-ledger
**Date:** 2026-02-24
**Based on:** [VIBE_CHECK_REPORT.md](VIBE_CHECK_REPORT.md) (Vibe-Code Confidence: 38.1%)

## Overview

This plan addresses all 17 remediation items from the Vibe-Code Detection Audit,
organized into 5 phases by priority and dependency order. Each item includes exact
file locations, specific code changes, effort estimates, and dependencies.

**Estimated total effort:** ~40-60 focused hours across all phases

---

## Phase 1: Critical Safety & Correctness (Priority: HIGH)

These items address real runtime risks — race conditions, lost stack traces, and
silent failures. Must be done first.

---

### 1.1 Add Exception Chaining (`raise X from e`)

**Audit finding:** Zero `raise X from e` in the entire codebase. Original stack traces are lost on re-raise, making debugging extremely difficult.

**Effort:** ~2 hours
**Dependencies:** None

**Files and changes:**

| File | Line | Current Code | Change To |
|------|------|-------------|-----------|
| `security.py` | 888 | `raise error` | `raise error from error` (or restructure to preserve chain) |
| `security.py` | 959-960 | `except Exception as e: security.handle_error(e, ...)` | Add `from e` to any re-raise inside `handle_error` |
| `security.py` | 989-990 | `except Exception as e: security.handle_error(e, ..., reraise=True)` | Same — ensure `handle_error` chains with `from e` |
| `integration.py` | 187 | `raise` (bare re-raise of RateLimitError) | OK as-is — bare `raise` preserves chain. No change needed. |
| `natlangchain.py` | 84 | `raise ValueError(f"Cannot resolve hostname...{e}")` | `raise ValueError(f"Cannot resolve hostname...") from e` |

**Key change — `security.py:handle_error()`:**
```python
# security.py line 887-888, change:
def handle_error(self, error, context=None, reraise=True):
    ...
    if reraise:
        raise error  # <-- loses chain
# To:
    if reraise:
        raise  # bare raise preserves the original traceback
```

**Also audit every `raise SomeError(...)` that occurs inside an `except` block:**
- `security.py:832-836` — `raise ConnectionProtectionError(...)` inside `check_and_report`. This is raised as a new exception (not re-raise), so add `from None` if intentional, or chain to the policy check context.
- `natlangchain.py:76` — inner `except ValueError` that re-raises the same ValueError is fine (passthrough).
- `privacy.py:77` — `raise RuntimeError(...)` in `_ensure_crypto()` — not inside an except block, no change needed.

---

### 1.2 Add Thread Safety to Lazy-Loaded Singletons

**Audit finding:** Module-level mutable state is mutated without locks. Race conditions possible in multi-threaded environments.

**Effort:** ~1 hour
**Dependencies:** None

**Files and changes:**

**`heuristics.py` — `get_embedding_model()` (lines 29-40):**
```python
# Add at top of file (after imports):
import threading
_EMBEDDING_LOCK = threading.Lock()

# Change get_embedding_model():
def get_embedding_model():
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL  # fast path, no lock
    with _EMBEDDING_LOCK:
        if _EMBEDDING_MODEL is None:  # double-checked locking
            if not _HAS_EMBEDDINGS:
                raise RuntimeError("sentence-transformers not available")
            _EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            if torch.cuda.is_available():
                _EMBEDDING_MODEL = _EMBEDDING_MODEL.to("cuda")
    return _EMBEDDING_MODEL
```

**`privacy.py` — `_try_import_cryptography()` (lines 43-66):**
```python
# Add at top of file (after imports):
import threading
_CRYPTO_LOCK = threading.Lock()

# Change _try_import_cryptography():
def _try_import_cryptography():
    global CRYPTO_AVAILABLE, _Fernet, _hashes, _PBKDF2HMAC
    if CRYPTO_AVAILABLE:
        return True  # fast path
    with _CRYPTO_LOCK:
        if CRYPTO_AVAILABLE:  # double-checked
            return True
        try:
            # ... existing import logic ...
        except Exception as e:
            logger.debug(f"Cryptography import failed: {e}")
            return False

# Also update _ensure_crypto() (line 71-77) to use the lock:
def _ensure_crypto():
    if not CRYPTO_AVAILABLE and _Fernet is None:
        _try_import_cryptography()  # already thread-safe now
    if not CRYPTO_AVAILABLE:
        raise RuntimeError("Cryptography library required but not available")
```

**`security.py` — `SecurityManager.get_instance()` (lines 769-774):**
```python
# Add class-level lock:
class SecurityManager:
    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def get_instance(cls):
        if cls._instance is not None:
            return cls._instance
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance
```

---

### 1.3 Replace `except Exception: pass` with Logged Cleanup

**Audit finding:** `security.py:541-542` silently swallows exceptions during daemon connectivity check.

**Effort:** ~15 minutes
**Dependencies:** None

**File:** `security.py`, lines 540-542

```python
# Current:
        except Exception:
            pass

# Change to:
        except (urllib.error.URLError, socket.error, json.JSONDecodeError, OSError) as e:
            logger.debug(f"Boundary Daemon HTTP fallback unavailable: {e}")
```

---

### 1.4 Narrow Broad `except Exception` Catches

**Audit finding:** 12 broad `except Exception` catches mask unexpected errors.

**Effort:** ~2 hours
**Dependencies:** 1.1 (exception chaining) should be done first

**All locations and specific narrowings:**

| File | Line | Context | Narrow To |
|------|------|---------|-----------|
| `security.py:313` | SIEM health check | `except (urllib.error.URLError, socket.error, OSError) as e:` |
| `security.py:410` | SIEM HTTP send | `except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:` |
| `security.py:434` | SIEM CEF send | `except (socket.error, OSError) as e:` |
| `security.py:541` | Daemon check (covered in 1.3) | Already addressed above |
| `security.py:610` | Daemon socket query | `except (socket.error, json.JSONDecodeError, OSError) as e:` |
| `security.py:616` | Daemon HTTP query | `except (urllib.error.URLError, json.JSONDecodeError, OSError) as e:` |
| `security.py:740` | Violation report | `except (urllib.error.URLError, socket.error, OSError) as e:` |
| `security.py:959` | Protected operation decorator | `except (ConnectionProtectionError,): raise` then `except Exception as e:` — this one is intentionally broad (decorator must catch all). **Keep as-is** but add `from e` to handle_error re-raise. |
| `security.py:989` | Security context manager | Same as above — intentionally broad. **Keep as-is** but add chaining. |
| `natlangchain.py:311` | submit_entry | Already has specific catches above it (296-310). The broad catch at 311 is a final fallback. **Narrow to** `except (TypeError, KeyError, OSError) as e:` |
| `natlangchain.py:342` | validate_entry | `except (urllib.error.URLError, socket.error, OSError):` |
| `natlangchain.py:359` | get_chain_narrative | `except (urllib.error.URLError, socket.error, OSError) as e:` |
| `natlangchain.py:377` | search_by_intent | `except (urllib.error.URLError, socket.error, json.JSONDecodeError, OSError):` |
| `natlangchain.py:401` | check_inclusion | `except (urllib.error.URLError, socket.error, json.JSONDecodeError, OSError):` |

---

### 1.5 Add Input Length Validation to LedgerEntry

**Audit finding:** `intent_id` and other string fields accept arbitrary-length strings, enabling memory abuse.

**Effort:** ~1 hour
**Dependencies:** None

**File:** `core.py`, lines 248-291

```python
# Add Field constraints to LedgerEntry:
class LedgerEntry(BaseModel):
    id: Optional[str] = Field(default=None, max_length=128)
    timestamp: float = Field(default_factory=time.time)
    intent_id: str = Field(..., min_length=1, max_length=256)
    memory_hash: Optional[str] = Field(default=None, max_length=128)
    value_vector: ValueVector
    status: str = Field(default="active", pattern=r"^(active|frozen|revoked)$")
    parent_id: Optional[str] = Field(default=None, max_length=128)
    correction_notes: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    ...
    owner: Optional[str] = Field(default=None, max_length=256)
    contract_id: Optional[str] = Field(default=None, max_length=256)
    revoked_by: Optional[str] = Field(default=None, max_length=256)
    revocation_reason: Optional[str] = Field(default=None, max_length=1024)
    aggregation_rule: Optional[str] = Field(default=None, pattern=r"^(sum|max|weighted)$")
```

**Also add metadata size validation in `accrue()` (line 371):**
```python
def accrue(self, ..., metadata: Optional[Dict] = None, ...):
    # Add metadata size check
    if metadata:
        metadata_str = json.dumps(metadata)
        if len(metadata_str) > 65536:  # 64KB limit
            raise ValueError(f"Metadata too large: {len(metadata_str)} bytes (max 65536)")
    ...
```

---

### 1.6 Add Pagination/Streaming to ValueLedger._load_all()

**Audit finding:** Entire ledger is loaded into memory on initialization. Ledgers with >100K entries will exhaust memory.

**Effort:** ~3 hours
**Dependencies:** None

**File:** `core.py`, lines 303-323

```python
class ValueLedger:
    def __init__(
        self,
        storage_path: str | Path = "ledger.jsonl",
        max_entries: Optional[int] = None,  # New: cap in-memory entries
    ):
        self.storage_path = _validate_ledger_path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._max_entries = max_entries
        self.entries: List[LedgerEntry] = self._load_all()
        self.merkle_tree = MerkleTree()
        self._rebuild_merkle_tree()

    def _load_all(self) -> List[LedgerEntry]:
        if not self.storage_path.exists():
            return []
        entries = []
        error_count = 0
        with open(self.storage_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        data = json.loads(line)
                        entries.append(LedgerEntry(**data))
                    except (json.JSONDecodeError, TypeError, KeyError) as e:
                        error_count += 1
                        logger.warning(f"Failed to load ledger line {line_num}: {e}")
        if error_count > 0:
            logger.warning(f"Skipped {error_count} malformed entries out of {line_num}")
        if self._max_entries and len(entries) > self._max_entries:
            logger.warning(
                f"Ledger has {len(entries)} entries, capping to {self._max_entries} most recent"
            )
            entries = sorted(entries, key=lambda e: e.timestamp)[-self._max_entries:]
        return entries

    @property
    def entry_count(self) -> int:
        """Total entries including those not loaded."""
        # For full count, scan the file
        if not self.storage_path.exists():
            return 0
        with open(self.storage_path, "r") as f:
            return sum(1 for line in f if line.strip())
```

---

## Phase 2: Code Quality & Error Handling (Priority: MEDIUM-HIGH)

These items improve maintainability and developer experience.

---

### 2.1 Add Context Manager Protocol to Network Clients

**Audit finding:** `BoundarySIEMClient` and `BoundaryDaemonClient` open network connections but don't implement `__enter__`/`__exit__`.

**Effort:** ~1 hour
**Dependencies:** None

**File:** `security.py`

**Add to `BoundarySIEMClient` (after line 297):**
```python
class BoundarySIEMClient:
    ...
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self._connected = False
        return False
```

**Add to `BoundaryDaemonClient` (after line 511):**
```python
class BoundaryDaemonClient:
    ...
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._connected = False
        self._policy_cache.clear()
        return False
```

**Add to `NLCClient` in `natlangchain.py` (after line 242):**
```python
class NLCClient:
    ...
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False  # No persistent connections to clean up
```

---

### 2.2 Add Environment Variable Fallbacks

**Audit finding:** Zero `os.environ`/`os.getenv` calls. All config is hardcoded or constructor-only.

**Effort:** ~1.5 hours
**Dependencies:** None

**File:** `security.py`

**Update `init_security()` (lines 998-1024):**
```python
import os

def init_security(
    siem_endpoint: Optional[str] = None,
    daemon_socket: Optional[str] = None,
    enabled: bool = True,
) -> SecurityManager:
    siem_config = SIEMConfig(
        enabled=enabled,
        http_endpoint=(
            siem_endpoint
            or os.environ.get("VALUE_LEDGER_SIEM_ENDPOINT", "http://localhost:8080/api/v1/events")
        ),
        timeout=float(os.environ.get("VALUE_LEDGER_SIEM_TIMEOUT", "5.0")),
    )
    daemon_config = BoundaryDaemonConfig(
        enabled=enabled,
        socket_path=(
            daemon_socket
            or os.environ.get("VALUE_LEDGER_DAEMON_SOCKET", "/var/run/boundary-daemon/api.sock")
        ),
        http_fallback=os.environ.get("VALUE_LEDGER_DAEMON_HTTP", "http://localhost:9090/api/v1"),
        timeout=float(os.environ.get("VALUE_LEDGER_DAEMON_TIMEOUT", "2.0")),
    )
    return SecurityManager.configure(siem_config, daemon_config)
```

**File:** `natlangchain.py`

**Update `NLCClient.__init__()` (line 223):**
```python
import os

class NLCClient:
    def __init__(
        self,
        base_url: str = None,
        timeout: float = None,
        ...
    ):
        self.base_url = (
            base_url
            or os.environ.get("VALUE_LEDGER_NLC_URL", "http://localhost:5000")
        ).rstrip("/")
        self.timeout = timeout or float(os.environ.get("VALUE_LEDGER_NLC_TIMEOUT", "30.0"))
        ...
```

**Env vars to document (add to README):**
| Variable | Default | Description |
|----------|---------|-------------|
| `VALUE_LEDGER_SIEM_ENDPOINT` | `http://localhost:8080/api/v1/events` | SIEM HTTP endpoint |
| `VALUE_LEDGER_SIEM_TIMEOUT` | `5.0` | SIEM request timeout (seconds) |
| `VALUE_LEDGER_DAEMON_SOCKET` | `/var/run/boundary-daemon/api.sock` | Daemon Unix socket path |
| `VALUE_LEDGER_DAEMON_HTTP` | `http://localhost:9090/api/v1` | Daemon HTTP fallback |
| `VALUE_LEDGER_DAEMON_TIMEOUT` | `2.0` | Daemon request timeout (seconds) |
| `VALUE_LEDGER_NLC_URL` | `http://localhost:5000` | NatLangChain API URL |
| `VALUE_LEDGER_NLC_TIMEOUT` | `30.0` | NatLangChain timeout (seconds) |

---

### 2.3 Add Dry-Run/Local-Only Mode for Integration Clients

**Audit finding:** NLCClient, BoundarySIEMClient, BoundaryDaemonClient attempt real network connections to services that don't exist yet, causing silent failures.

**Effort:** ~3 hours
**Dependencies:** 2.2 (env vars)

**Approach:** Add a `dry_run` flag that logs operations to a local file instead of sending over the network.

**File:** `security.py`

**Update `SIEMConfig` (line 270):**
```python
@dataclass
class SIEMConfig:
    ...
    dry_run: bool = False  # Log events locally instead of sending
    dry_run_path: str = "siem_events.jsonl"  # Local event log
```

**Update `BoundarySIEMClient.report()` (line 329):**
```python
def report(self, event: SecurityEvent) -> bool:
    if not self.config.enabled:
        return True
    if self.config.dry_run:
        # Log locally instead of sending to SIEM
        logger.info(f"[DRY RUN] SIEM event: {event.event_type.value}")
        with open(self.config.dry_run_path, "a") as f:
            json.dump(event.to_json(), f)
            f.write("\n")
        return True
    # ... existing code ...
```

**Update `BoundaryDaemonConfig` (line 488):**
```python
@dataclass
class BoundaryDaemonConfig:
    ...
    dry_run: bool = False  # Allow all operations without daemon
```

**Update `BoundaryDaemonClient._check_daemon()` (line 516):**
```python
def _check_daemon(self) -> bool:
    if self.config.dry_run:
        logger.info("[DRY RUN] Boundary Daemon check skipped, all operations allowed")
        self._connected = True
        self._current_mode = BoundaryMode.OPEN
        return True
    # ... existing code ...
```

**File:** `natlangchain.py`

**Update `NLCClient.__init__()` (line 223):**
```python
class NLCClient:
    def __init__(
        self,
        base_url: str = "http://localhost:5000",
        timeout: float = 30.0,
        allow_private: bool = True,
        max_response_size: int = MAX_RESPONSE_SIZE,
        dry_run: bool = False,  # New
    ):
        ...
        self.dry_run = dry_run
        if not dry_run:
            _validate_url(self.base_url, allow_private=allow_private)

    def submit_entry(self, record: NLCRecord) -> AnchorResult:
        if self.dry_run:
            anchor_id = hashlib.sha256(json.dumps(record.to_dict()).encode()).hexdigest()[:16]
            logger.info(f"[DRY RUN] NLC anchor: {anchor_id}")
            return AnchorResult(success=True, anchor_id=f"dry_run_{anchor_id}")
        # ... existing code ...
```

**Wire up env var:** `VALUE_LEDGER_DRY_RUN=true` sets all integration clients to dry-run mode.

---

### 2.4 Add Structured JSON Logging Option

**Audit finding:** Standard Python logging only. No structured output for production log aggregation.

**Effort:** ~2 hours
**Dependencies:** None

**Approach:** Add a `configure_logging()` function that enables JSON-formatted log output.

**New function in `security.py` or new file `value_ledger/logging_config.py`:**

Recommended: Add to existing `security.py` near the top (after line 45), since it already has logger configuration:

```python
import os

def configure_logging(
    json_format: bool = None,
    level: str = None,
):
    """
    Configure logging for all value_ledger modules.

    Args:
        json_format: Use JSON log format. Default: checks VALUE_LEDGER_LOG_FORMAT env var.
        level: Log level. Default: checks VALUE_LEDGER_LOG_LEVEL env var, then INFO.
    """
    if json_format is None:
        json_format = os.environ.get("VALUE_LEDGER_LOG_FORMAT", "").lower() == "json"
    if level is None:
        level = os.environ.get("VALUE_LEDGER_LOG_LEVEL", "INFO")

    root_logger = logging.getLogger("value_ledger")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    if json_format:
        handler = logging.StreamHandler()
        handler.setFormatter(_JSONFormatter())
        root_logger.addHandler(handler)
    else:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        root_logger.addHandler(handler)


class _JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        })
```

**Export from `__init__.py`:**
```python
from .security import configure_logging
```

---

### 2.5 Add Correlation IDs for Cross-Module Tracing

**Audit finding:** Zero correlation IDs. No way to trace related operations across modules.

**Effort:** ~3 hours
**Dependencies:** 2.4 (structured logging)

**Approach:** Use `contextvars` for a thread-safe correlation ID that propagates through call chains.

**New module or addition to `security.py`:**

```python
import contextvars
import uuid

# Thread-local correlation ID
_correlation_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "correlation_id", default=None
)

def get_correlation_id() -> str:
    """Get or create a correlation ID for the current operation."""
    cid = _correlation_id.get()
    if cid is None:
        cid = uuid.uuid4().hex[:12]
        _correlation_id.set(cid)
    return cid

def set_correlation_id(cid: str) -> None:
    """Explicitly set correlation ID (e.g., from external request)."""
    _correlation_id.set(cid)

@contextmanager
def correlation_scope(cid: Optional[str] = None):
    """Context manager that sets a correlation ID for the duration."""
    token = _correlation_id.set(cid or uuid.uuid4().hex[:12])
    try:
        yield _correlation_id.get()
    finally:
        _correlation_id.reset(token)
```

**Integration points:**
- `ValueLedger.accrue()` — set correlation ID in entry metadata
- `SecurityManager.check_and_report()` — include in security events
- `IntentLogConnector.handle_event()` — create correlation scope per event
- JSON logger — include `correlation_id` field automatically

**Update `_JSONFormatter` to include correlation ID:**
```python
class _JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            ...
            "correlation_id": _correlation_id.get(),
        })
```

---

## Phase 3: Code Hygiene & Style (Priority: MEDIUM)

These items address AI-generation artifacts. They improve readability but
don't affect functionality.

---

### 3.1 Remove Decorative Section Dividers

**Audit finding:** 40 `# ====` section divider lines across 4 files are AI formatting artifacts.

**Effort:** ~30 minutes
**Dependencies:** None

**Files and line counts:**
| File | Divider Lines | Lines to Remove |
|------|---------------|-----------------|
| `security.py` | 16 pairs (32 lines) | Lines 48-50, 136-138, 187-189, 264-266, 461-463, 745-747, 895-897, 993-995 |
| `validation.py` | 12 pairs (24 lines) | Lines 33-35, and corresponding dividers throughout |
| `compatibility.py` | 10 pairs (20 lines) | Lines 31-33, 291-293, and remaining |
| `heuristics.py` | 2 lines | Lines 63, 393 |

**Approach:** Remove the `# ===` and `# ---` divider lines and their label comments. Keep the blank line separation between sections. Example:

```python
# Remove:
# =============================================================================
# Custom Exception Hierarchy
# =============================================================================

# Keep only a blank line between sections.
# If the section is large enough to need a label, use a brief comment:
# --- Exception Hierarchy ---
```

---

### 3.2 Add TODO Markers at Integration Stub Points

**Audit finding:** Zero TODO/FIXME markers. Real projects always have loose ends.

**Effort:** ~30 minutes
**Dependencies:** None

**Add markers at these specific locations:**

| File | Line | Add |
|------|------|-----|
| `memory_vault_hook.py:17` | Before stub imports | `# TODO: Replace stubs with real Memory Vault imports when available` |
| `memory_vault_hook.py:51` | `ConsentChecker = None` | `# TODO: Implement consent checking via Learning Contracts` |
| `security.py:300` | `self._test_connection()` | `# TODO: Add retry logic for transient SIEM failures` |
| `security.py:513` | `self._check_daemon()` | `# TODO: Add retry logic for transient daemon failures` |
| `natlangchain.py:227` | `allow_private: bool = True` | `# TODO: Default to False once NatLangChain is deployed publicly` |
| `core.py:95` | `compute_timestamp_proof` | `# TODO: Anchor to external timestamping service (RFC 3161) for production` |
| `integration.py:199` | `intent_updated` comment | `# TODO: Implement partial accruals on intent_updated events` |
| `heuristics.py:401` | NoveltyScorer comment | `# TODO: Add model caching/versioning for reproducible novelty scores` |

---

### 3.3 Consolidate Duplicate Documentation

**Audit finding:** Two contributing guides exist: `CONTRIBUTING.md` (45 lines) and `docs/contributing.md` (359 lines).

**Effort:** ~30 minutes
**Dependencies:** None

**Approach:**
1. Keep `CONTRIBUTING.md` (root — GitHub convention) as the primary file
2. Merge any unique content from `docs/contributing.md` into root `CONTRIBUTING.md`
3. Replace `docs/contributing.md` with a redirect:
   ```markdown
   # Contributing

   See [CONTRIBUTING.md](../CONTRIBUTING.md) in the project root.
   ```
4. Update any links in `README.md` and other docs that point to `docs/contributing.md`

---

## Phase 4: Test Quality Improvements (Priority: MEDIUM)

These items strengthen the test suite and address AI-generation patterns.

---

### 4.1 Replace Formulaic Test Docstrings

**Audit finding:** 100 test functions have `"""Tests for X."""` / `"""Test X."""` docstrings that add no information.

**Effort:** ~4 hours (large batch, mechanical)
**Dependencies:** None

**Approach:** Replace formulaic docstrings with behavior descriptions or remove them entirely. Test function names should describe the scenario; docstrings should add context only when non-obvious.

**Example transformations:**

```python
# Before:
def test_value_vector_total(self):
    """Test value vector total."""

# After (option A — descriptive name, remove docstring):
def test_value_vector_total(self):
    # No docstring needed — function name is self-explanatory

# Before:
def test_revoke_entry(self):
    """Tests for revoking entries."""

# After (option B — add behavior context):
def test_revoke_entry(self):
    """Revoking an entry marks it as revoked but preserves it for audit."""
```

**Files with formulaic docstrings (by count):**
| File | Formulaic Docstrings |
|------|---------------------|
| `tests/test_heuristics.py` | ~20 |
| `tests/test_core.py` | ~18 |
| `tests/test_security.py` | ~15 |
| `tests/test_compatibility.py` | ~12 |
| `tests/test_validation.py` | ~12 |
| `tests/test_cli.py` | ~10 |
| `tests/test_interruption.py` | ~8 |
| `tests/test_privacy.py` | ~5 |

---

### 4.2 Add Parametrized Tests for Boundary Values

**Audit finding:** Only 1 `@pytest.mark.parametrize` across 6,400 lines of tests.

**Effort:** ~4 hours
**Dependencies:** None

**High-value parametrization targets:**

**`tests/test_core.py` — ValueVector field validation:**
```python
@pytest.mark.parametrize("field,value,should_pass", [
    ("t", 0.0, True),
    ("t", -0.1, False),   # ge=0.0 constraint
    ("t", 1000.0, True),
    ("e", 0.0, True),
    ("n", -1.0, False),
])
def test_value_vector_field_constraints(field, value, should_pass):
    ...
```

**`tests/test_core.py` — Classification levels:**
```python
@pytest.mark.parametrize("classification,valid", [
    (0, True), (1, True), (5, True),
    (-1, False), (6, False), (100, False),
])
def test_classification_validation(classification, valid):
    ...
```

**`tests/test_core.py` — Aggregation rules:**
```python
@pytest.mark.parametrize("rule", ["sum", "max", "weighted"])
def test_aggregation_rules(rule):
    ...

@pytest.mark.parametrize("rule", ["average", "median", "", None])
def test_aggregation_invalid_rules(rule):
    ...
```

**`tests/test_core.py` — Merkle tree edge cases:**
```python
@pytest.mark.parametrize("num_leaves", [1, 2, 3, 4, 7, 8, 15, 16, 100])
def test_merkle_tree_various_sizes(num_leaves):
    ...
```

**`tests/test_core.py` — Path traversal prevention:**
```python
@pytest.mark.parametrize("malicious_path", [
    "/etc/passwd", "../../../etc/shadow", "/proc/self/environ",
    "/dev/null", "~/.ssh/id_rsa", "path\x00injection",
])
def test_path_traversal_blocked(malicious_path):
    ...
```

**`tests/test_security.py` — Exception hierarchy:**
```python
@pytest.mark.parametrize("exc_class,code", [
    (ValidationError, "VL_VALIDATION_ERROR"),
    (StorageError, "VL_STORAGE_ERROR"),
    (CryptographyError, "VL_CRYPTO_ERROR"),
    (SecurityError, "VL_SECURITY_ERROR"),
    (ConnectionProtectionError, "VL_CONNECTION_DENIED"),
    (RateLimitError, "VL_RATE_LIMIT"),
])
def test_exception_codes(exc_class, code):
    exc = exc_class("test")
    assert exc.code == code
```

**`tests/test_natlangchain.py` — SSRF blocked hosts:**
```python
@pytest.mark.parametrize("url", [
    "http://169.254.169.254/metadata",
    "http://metadata.google.internal/",
    "http://127.0.0.1/admin",
    "http://[::1]/admin",
    "ftp://example.com/file",
    "file:///etc/passwd",
])
def test_ssrf_blocked_urls(url):
    with pytest.raises(ValueError):
        _validate_url(url, allow_private=False)
```

**`tests/test_heuristics.py` — Scorer boundary values:**
```python
@pytest.mark.parametrize("duration_hours,expected_min,expected_max", [
    (0.0, 0.0, 0.2),     # Zero duration
    (0.01, 0.5, 2.0),    # Very short
    (1.0, 3.0, 8.0),     # One hour
    (8.0, 8.0, 15.0),    # Full day
    (100.0, 14.0, 15.0), # Very long — caps at 15
])
def test_time_scorer_ranges(duration_hours, expected_min, expected_max):
    ...
```

---

## Phase 5: Documentation & Cleanup (Priority: LOW)

These items are cosmetic and organizational.

---

### 5.1 Update README with Environment Variables

**Effort:** ~30 minutes
**Dependencies:** 2.2

Add a "Configuration" section to `README.md` documenting all env vars from Phase 2.2.

---

### 5.2 Add Thread Safety Documentation

**Effort:** ~15 minutes
**Dependencies:** 1.2

Add a "Thread Safety" section to `README.md`:
```markdown
## Thread Safety

The Value Ledger is designed primarily for single-threaded CLI use.
When using in multi-threaded applications:
- Module-level singletons (embedding model, cryptography) are thread-safe via locks
- `ValueLedger` instances are NOT thread-safe — use one instance per thread
- `SecurityManager.get_instance()` is thread-safe
```

---

## Dependency Graph

```
Phase 1 (no dependencies):
  1.1 Exception chaining
  1.2 Thread safety
  1.3 except:pass fix
  1.5 Input validation
  1.6 Pagination

Phase 1 → Phase 1:
  1.1 → 1.4 (narrow exceptions after adding chaining)

Phase 2 (no dependencies):
  2.1 Context managers
  2.4 JSON logging

Phase 2 → Phase 2:
  2.2 → 2.3 (env vars before dry-run mode)
  2.4 → 2.5 (JSON logging before correlation IDs)

Phase 3 (no dependencies):
  3.1 Remove dividers
  3.2 Add TODOs
  3.3 Consolidate docs

Phase 4 (no dependencies):
  4.1 Fix docstrings
  4.2 Add parametrized tests

Phase 5 (depends on earlier phases):
  2.2 → 5.1 (env vars → README update)
  1.2 → 5.2 (thread safety → README update)
```

---

## Implementation Order (Recommended)

For maximum impact with minimum risk, implement in this order:

1. **1.1** Exception chaining (~2h) — immediate debugging benefit
2. **1.3** except:pass fix (~15min) — trivial, high value
3. **1.2** Thread safety (~1h) — prevents race conditions
4. **1.4** Narrow exceptions (~2h) — depends on 1.1
5. **1.5** Input validation (~1h) — prevents abuse
6. **1.6** Pagination (~3h) — prevents memory issues
7. **2.1** Context managers (~1h) — resource cleanup
8. **2.2** Env var fallbacks (~1.5h) — runtime configurability
9. **2.3** Dry-run mode (~3h) — depends on 2.2
10. **2.4** JSON logging (~2h) — observability
11. **2.5** Correlation IDs (~3h) — depends on 2.4
12. **3.1** Remove dividers (~30min) — cosmetic
13. **3.2** Add TODOs (~30min) — honesty markers
14. **3.3** Consolidate docs (~30min) — reduce confusion
15. **4.1** Fix docstrings (~4h) — batch job
16. **4.2** Parametrized tests (~4h) — test depth
17. **5.1-5.2** README updates (~45min) — depends on earlier phases

---

## Expected Impact on Vibe-Code Score

If all items are implemented:

| Domain | Current | Expected | Change |
|--------|---------|----------|--------|
| A. Surface Provenance | 57.1% (12/21) | ~71% (15/21) | +14% |
| B. Behavioral Integrity | 66.7% (14/21) | ~86% (18/21) | +19% |
| C. Interface Authenticity | 57.1% (12/21) | ~76% (16/21) | +19% |

**Expected Weighted Authenticity:** ~80%
**Expected Vibe-Code Confidence:** ~20%
**Expected Classification:** AI-Assisted (16-35 range)

The project would move from "Substantially Vibe-Coded" to "AI-Assisted" — which
accurately reflects a codebase where AI generated the initial code but a human has
reviewed, hardened, and made it production-ready.
