#!/usr/bin/env bash
# Build the openextract CLI for $TARGET and write a release archive under dist/.
# .github/workflows/release.yml runs this on each release target.

set -euo pipefail

if [ -z "${TARGET:-}" ] || [ -z "${VERSION:-}" ]; then
  echo "TARGET and VERSION are required" >&2
  exit 1
fi

case "$VERSION" in
  *[!0-9A-Za-z.+_-]*)
    echo "refusing unsafe version: ${VERSION}" >&2
    exit 1
    ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

case "${RUNNER_OS:-}" in
  Linux | macOS | Windows) ;;
  "")
    case "$(uname -s)" in
      Linux) RUNNER_OS=Linux ;;
      Darwin) RUNNER_OS=macOS ;;
      MINGW* | MSYS* | CYGWIN*) RUNNER_OS=Windows ;;
      *)
        echo "set RUNNER_OS (Linux, macOS, or Windows)" >&2
        exit 1
        ;;
    esac
    ;;
  *)
    echo "unsupported RUNNER_OS=${RUNNER_OS}" >&2
    exit 1
    ;;
esac

bin_name="openextract"
ext="tar.gz"
case "$TARGET" in
  x86_64-unknown-linux-musl | aarch64-unknown-linux-musl) ;;
  x86_64-apple-darwin | aarch64-apple-darwin) ;;
  x86_64-pc-windows-msvc)
    bin_name="openextract.exe"
    ext="zip"
    ;;
  *)
    echo "unsupported target: ${TARGET}" >&2
    exit 1
    ;;
esac

if [[ "$TARGET" == *-linux-musl ]]; then
  if ! command -v musl-gcc >/dev/null 2>&1; then
    echo "musl-gcc is required to link ${TARGET} (install the musl-tools package)" >&2
    exit 1
  fi
  upper="$(printf '%s' "$TARGET" | tr '[:lower:]' '[:upper:]' | tr '-' '_')"
  export "CARGO_TARGET_${upper}_LINKER=musl-gcc"
  export "CC_$(printf '%s' "$TARGET" | tr '-' '_')=musl-gcc"
  # Musl targets default to a static CRT. Set it explicitly so a parent RUSTFLAGS
  # cannot drop it, and so the archive is one file with no libmusl dependency.
  export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }-C target-feature=+crt-static"
fi

(
  cd rust
  cargo build --locked --release --target "$TARGET"
)

bin="rust/target/${TARGET}/release/${bin_name}"
if [ ! -f "$bin" ]; then
  echo "missing binary: ${bin}" >&2
  exit 1
fi

if [[ "$TARGET" == *-linux-musl ]]; then
  if ! command -v file >/dev/null 2>&1; then
    echo "file(1) is required to check that the musl binary is statically linked" >&2
    exit 1
  fi
  file "$bin"
  if file "$bin" | grep -q 'dynamically linked'; then
    echo "musl binary is dynamically linked" >&2
    exit 1
  fi
  if ! file "$bin" | grep -Eq 'statically linked|static-pie'; then
    echo "musl binary is not statically linked" >&2
    exit 1
  fi
  # glibc ldd says "not a dynamic executable" or "statically linked" for static-pie.
  ldd_out="$(ldd "$bin" 2>&1 || true)"
  printf '%s\n' "$ldd_out"
  if printf '%s\n' "$ldd_out" | grep -Eq '=>'; then
    echo "musl binary links shared libraries" >&2
    exit 1
  fi
  if ! printf '%s\n' "$ldd_out" | grep -Eq 'not a dynamic executable|statically linked'; then
    echo "musl binary is not reported as static" >&2
    exit 1
  fi
fi

stage="$(mktemp -d)"
extract="$(mktemp -d)"
cleanup() {
  rm -rf "$stage" "$extract"
}
trap cleanup EXIT

cp "$bin" "${stage}/${bin_name}"
cp LICENSE "${stage}/LICENSE"
if [ "$RUNNER_OS" != "Windows" ]; then
  chmod 755 "${stage}/${bin_name}"
fi

mkdir -p dist
archive="dist/openextract-${VERSION}-${TARGET}.${ext}"
rm -f "$archive"
if [ "$ext" = "zip" ]; then
  # Python is on every GitHub-hosted runner. zipfile avoids depending on
  # bsdtar’s zip support, which GNU tar does not have.
  if python3 -c 'import sys' >/dev/null 2>&1; then
    py=python3
  elif python -c 'import sys' >/dev/null 2>&1; then
    py=python
  else
    echo "python is required to pack the Windows zip" >&2
    exit 1
  fi
  "$py" - "$stage" "$archive" "$bin_name" <<'PY'
import sys
import zipfile
from pathlib import Path

stage, archive, bin_name = sys.argv[1:]
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
    for name in (bin_name, "LICENSE"):
        zf.write(Path(stage) / name, arcname=name)
PY
  listing="$("$py" - "$archive" <<'PY'
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as zf:
    for name in sorted(zf.namelist()):
        print(name)
PY
)"
  "$py" - "$archive" "$extract" <<'PY'
import sys
import zipfile

archive, dest = sys.argv[1:]
with zipfile.ZipFile(archive) as zf:
    zf.extractall(dest)
PY
else
  tar -czf "$archive" -C "$stage" "$bin_name" LICENSE
  listing="$(tar -tzf "$archive" | sed 's#^\./##' | sort)"
  tar -xzf "$archive" -C "$extract"
fi
expected="$(printf '%s\n' LICENSE "$bin_name" | sort)"
if [ "$listing" != "$expected" ]; then
  echo "unexpected archive contents:" >&2
  printf '%s\n' "$listing" >&2
  exit 1
fi
cmp "$bin" "${extract}/${bin_name}"

run_smoke=false
case "$TARGET" in
  x86_64-unknown-linux-musl)
    if [ "$(uname -m)" != "x86_64" ]; then
      echo "x86_64 musl build must run on an x86_64 runner" >&2
      exit 1
    fi
    run_smoke=true
    ;;
  aarch64-unknown-linux-musl)
    if [ "$(uname -m)" != "aarch64" ]; then
      echo "aarch64 musl build must run on an aarch64 runner" >&2
      exit 1
    fi
    run_smoke=true
    ;;
  aarch64-apple-darwin)
    if [ "$(uname -m)" != "arm64" ]; then
      echo "Apple silicon build must run on an arm64 runner" >&2
      exit 1
    fi
    run_smoke=true
    ;;
  x86_64-apple-darwin)
    if command -v file >/dev/null 2>&1; then
      file "$bin" | grep -q 'x86_64'
    fi
    ;;
  x86_64-pc-windows-msvc)
    if [ "$RUNNER_OS" != "Windows" ]; then
      echo "Windows build must run on Windows" >&2
      exit 1
    fi
    run_smoke=true
    ;;
esac

if [ "$run_smoke" = true ]; then
  output="$("${extract}/${bin_name}" --list-providers)"
  printf '%s\n' "$output" | grep -q '"prefix"'
fi

echo "packed ${archive}"
