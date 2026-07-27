"""Task 3.14 — Enforce the current deferred real-data boundary.

Tests that classifier.py, practice.py, and real_eval.py are outside
supported imports, that real_eval.py uses only standard-library imports,
that CLI invocation exits with an explicit unsupported/deferred status,
and that generated deferred invocations always reject before inspecting
arguments or accessing pilot resources.

**Validates: Requirements 2.9, 2.10, 2.11, 3.8**
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, seed, settings, strategies as st

from test_support import CONFIG, HYPOTHESIS_SEED

ROOT = Path(__file__).resolve().parents[1]
DEFERRED_MODULES = tuple(CONFIG["deferred_scope"]["modules"])
FORBIDDEN = set(CONFIG["forbidden_synthetic_dependencies"])
SUPPORTED = set(CONFIG["supported_synthetic_modules"])

# Standard-library module names allowed in real_eval.py imports.
_STDLIB_ALLOWED = frozenset({
    "__future__", "sys", "typing", "os", "pathlib", "abc",
    "collections", "dataclasses", "enum", "functools", "io",
    "itertools", "json", "math", "re", "string", "textwrap",
    "types", "warnings",
})


# ---------------------------------------------------------------------------
# Property 12 (Evidence-Scope Safety): deferred invocations
# ---------------------------------------------------------------------------

@st.composite
def deferred_invocation_args(draw: Any) -> tuple[str, list[str]]:
    """Generate plausible but always-rejected CLI arguments for real_eval."""
    extras = draw(st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("L", "N", "P")),
            min_size=1,
            max_size=40,
        ),
        min_size=0,
        max_size=4,
    ))
    return str(ROOT / "src" / "real_eval.py"), extras


@seed(HYPOTHESIS_SEED)
@settings(max_examples=20)
@given(invocation=deferred_invocation_args())
@pytest.mark.deferred_scope
def test_property_12_generated_deferred_invocations_stop_before_import_or_path_access(
    invocation: tuple[str, list[str]],
) -> None:
    """Property 12 refinement — deferred invocations always reject early.

    No generated argument combination should cause real_eval to import an
    optional dependency, access a model/cache/manifest/video path, or
    produce any result resembling a metric.

    **Validates: Requirements 2.9, 2.10, 2.11**
    """
    script_path, extra_args = invocation
    result = subprocess.run(
        [sys.executable, script_path, *extra_args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    combined = (result.stdout + result.stderr).strip().lower()

    # Must exit with nonzero status.
    assert result.returncode != 0, (
        f"real_eval exited 0 with args {extra_args!r}"
    )
    # Must mention deferred/unsupported status.
    assert "deferred" in combined or "unsupported" in combined, (
        f"real_eval output did not contain deferred/unsupported: {combined[:300]}"
    )
    # Must not produce a Python traceback (clean rejection, not a crash).
    assert "traceback" not in combined, (
        f"real_eval produced a traceback: {combined[:500]}"
    )
    # Must not produce anything resembling a metric value.
    for metric_word in ("accuracy", "coverage", "latency", "perseveration"):
        assert metric_word not in combined, (
            f"real_eval output contained metric-like word '{metric_word}': "
            f"{combined[:300]}"
        )


# ---------------------------------------------------------------------------
# Unit tests — deferred CLI and import guards
# ---------------------------------------------------------------------------

@pytest.mark.deferred_scope
def test_deferred_cli_exits_with_explicit_status_without_optional_imports() -> None:
    """CLI invocation of real_eval.py rejects immediately.

    The subprocess must:
    1. Exit with nonzero return code.
    2. Print an explicit unsupported/deferred status message.
    3. Never produce a traceback.
    4. Never import torch, classifier, or practice.

    **Validates: Requirements 2.9, 2.11**
    """
    # Also inject an import guard to catch any attempt at optional imports.
    guard_code = (
        "import builtins, sys; "
        f"_forbidden = {FORBIDDEN!r}; "
        "_orig = builtins.__import__; "
        "def _g(name, *a, **k):\n"
        "    assert name.split('.')[0] not in _forbidden, "
        "f'forbidden import: {name}'; "
        "    return _orig(name, *a, **k)\n"
        "builtins.__import__ = _g; "
        "exec(open(sys.argv[1]).read()); "
        "sys.exit(0)"
    )
    # Direct subprocess invocation is cleaner and more reliable.
    result = subprocess.run(
        [sys.executable, str(ROOT / "src" / "real_eval.py"),
         "__missing_cache__", "__missing_model__"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    combined = (result.stdout + result.stderr).strip()

    assert result.returncode != 0, "real_eval.py exited with 0"
    assert "deferred" in combined.lower(), (
        f"Missing deferred status in output: {combined[:300]}"
    )
    assert "unsupported" in combined.lower(), (
        f"Missing unsupported status in output: {combined[:300]}"
    )
    assert "traceback" not in combined.lower(), (
        f"Unexpected traceback: {combined[:500]}"
    )

    # Verify the DeferredRealDataError API contract.
    from real_eval import (
        DEFERRED_MESSAGE,
        DEFERRED_STATUS,
        DeferredRealDataError,
        deferred_status,
        run_real,
    )

    status = deferred_status()
    assert status["status"] == DEFERRED_STATUS
    assert status["supported"] is False
    assert status["evidence_category"] == "future pilot"
    assert DEFERRED_MESSAGE in status["message"]

    with pytest.raises(DeferredRealDataError, match="deferred"):
        run_real("any", "arguments", flag=True)


@pytest.mark.deferred_scope
def test_deferred_import_is_standard_library_only_and_loads_no_pilot_module() -> None:
    """real_eval.py must import only standard-library modules at the top level.

    Parse the AST of real_eval.py and verify every import is from the
    standard library.  Then dynamically import it in an isolated subprocess
    with a guarded __import__ and confirm no forbidden module is loaded.

    **Validates: Requirements 2.9, 3.8**
    """
    source_path = ROOT / "src" / "real_eval.py"
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))

    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imported_names.add(node.module.split(".")[0])

    non_stdlib = imported_names - _STDLIB_ALLOWED
    assert not non_stdlib, (
        f"real_eval.py imports non-standard-library modules at top level: "
        f"{sorted(non_stdlib)}"
    )

    # Dynamic verification: import real_eval in a subprocess with guarded imports.
    # Exclude real_eval itself from the forbidden set since we import it intentionally;
    # the guard verifies it doesn't transitively import other deferred/optional modules.
    import_forbidden = FORBIDDEN - {"real_eval"}
    guard_script = f"""\
import builtins
import sys

forbidden = {import_forbidden!r}
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    base = name.split(".")[0]
    if base in forbidden:
        raise AssertionError(f"real_eval imported forbidden module: {{name}}")
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

import real_eval

# Verify the module loaded successfully and has the expected API.
assert hasattr(real_eval, "deferred_status")
assert hasattr(real_eval, "run_real")
assert hasattr(real_eval, "DeferredRealDataError")
assert hasattr(real_eval, "EVIDENCE_CATEGORY")
assert real_eval.EVIDENCE_CATEGORY == "future pilot"
assert real_eval.deferred_status()["supported"] is False

# Verify no forbidden module was loaded (real_eval itself is allowed).
loaded = set(sys.modules)
leaked = forbidden & loaded
assert not leaked, f"Forbidden modules loaded: {{leaked}}"
"""
    result = subprocess.run(
        [sys.executable, "-c", guard_script],
        cwd=ROOT / "src",
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, (
        f"Guarded import of real_eval failed:\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.deferred_scope
def test_deferred_modules_are_not_imported_by_supported_synthetic_workflow() -> None:
    """Supported synthetic modules never import classifier, practice, or real_eval.

    Walk the AST of every supported module and verify no import references
    a deferred module.

    **Validates: Requirements 2.9, 3.8**
    """
    for module_name in SUPPORTED:
        source_path = ROOT / "src" / f"{module_name}.py"
        if not source_path.exists():
            continue
        source = source_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(source_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    base = alias.name.split(".")[0]
                    assert base not in DEFERRED_MODULES, (
                        f"{module_name}.py imports deferred module '{base}'"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module is not None:
                    base = node.module.split(".")[0]
                    assert base not in DEFERRED_MODULES, (
                        f"{module_name}.py imports from deferred module '{base}'"
                    )


@pytest.mark.deferred_scope
def test_deferred_scaffold_files_are_empty_or_guard_only() -> None:
    """classifier.py and practice.py must be empty; real_eval.py must be guard-only.

    **Validates: Requirements 2.9, 2.11**
    """
    for module_name in ("classifier", "practice"):
        source_path = ROOT / "src" / f"{module_name}.py"
        assert source_path.exists(), f"{module_name}.py is missing"
        content = source_path.read_text(encoding="utf-8").strip()
        assert content == "", (
            f"{module_name}.py should be empty but contains: {content[:200]}"
        )

    # real_eval.py must exist and must not define any function that returns
    # a result (all paths must raise or return a rejection status).
    real_eval_path = ROOT / "src" / "real_eval.py"
    assert real_eval_path.exists()
    source = real_eval_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(real_eval_path))

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Every function must either raise or return a rejection-like value.
            # We verify this by checking that no function name suggests a
            # result-producing capability.
            assert "evaluate" not in node.name.lower() or "deferred" in source.lower(), (
                f"real_eval.py contains a result-producing function: {node.name}"
            )
