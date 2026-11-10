import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE_TESTS = [ROOT / "tests" / name for name in ("test_live_providers.py", "test_live_schemas.py")]
KEY_VARIABLES = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GOOGLE_API_KEY",
    "GEMINI_API_KEY",
    "OPENROUTER_API_KEY",
)


def test_live_tests_stay_disabled_even_with_credentials():
    environment = dict(os.environ)
    environment.pop("PYTEST_ADDOPTS", None)
    environment.update(dict.fromkeys(KEY_VARIABLES, "dummy-not-a-real-key"))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *map(str, LIVE_TESTS),
            "-m",
            "live",
            "--no-cov",
            "-p",
            "no:cacheprovider",
            "-q",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "62 skipped" in result.stdout


def test_explicit_live_run_requires_credentials():
    environment = {name: value for name, value in os.environ.items() if name not in KEY_VARIABLES}
    environment.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *map(str, LIVE_TESTS),
            "--live",
            "-m",
            "live",
            "--no-cov",
            "-p",
            "no:cacheprovider",
            "-k",
            "openai and digital_pdf",
            "-q",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Set OPENAI_API_KEY" in result.stdout
    assert "2 errors" in result.stdout
