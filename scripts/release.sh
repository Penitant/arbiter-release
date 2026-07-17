#!/usr/bin/env bash
# Build and publish an Arbiter release.
#
#   scripts/release.sh v1.0.0
#
# Produces a clean server bundle, creates a GitHub release with the bundle, the classifier
# checkpoint, and their checksums, then publishes the npm package.
#
# Requirements: git, tar, gh (authenticated), npm, sha256sum (or shasum), a committed
# server/triage/weights/checkpoint.pt.

set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  echo "usage: $0 <version-tag>   e.g. $0 v1.0.0" >&2
  exit 1
fi
if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+([.-].+)?$ ]]; then
  echo "error: version must look like v1.2.3 (got '$VERSION')" >&2
  exit 1
fi
BARE="${VERSION#v}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECKPOINT="server/triage/weights/checkpoint.pt"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "error: $CHECKPOINT not found — nothing to attach as the classifier weights" >&2
  exit 1
fi

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT

TARBALL="arbiter-server-${VERSION}.tar.gz"
DIST="$BUILD/dist"
mkdir -p "$DIST"

echo "› staging clean server bundle…"
# Copy server/ minus secrets, caches, local state, weights, and internal docs.
tar \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='.env' \
  --exclude='.pytest_cache' \
  --exclude='db/*.db' \
  --exclude='cache' \
  --exclude='.venv' \
  --exclude='triage/weights/checkpoint.pt' \
  --exclude='triage/data/*.jsonl' \
  --exclude='triage/data/*.csv' \
  --exclude='triage/data/embeddings.pt' \
  --exclude='triage/data/*.log' \
  --exclude='*.log' \
  -cf - server | ( cd "$BUILD" && tar -xf - )

# Top-level release docs travel with the bundle.
for f in README.md LICENSE CHANGELOG.md; do
  [[ -f "$f" ]] && cp "$f" "$BUILD/server/" || true
done

echo "› packaging ${TARBALL}…"
tar -czf "$DIST/$TARBALL" -C "$BUILD" server

# --- checksums ---------------------------------------------------------------
checksum() {  # checksum <file> -> "<hex>  <basename>"
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    ( cd "$(dirname "$f")" && sha256sum "$(basename "$f")" )
  else
    ( cd "$(dirname "$f")" && shasum -a 256 "$(basename "$f")" )
  fi
}

cp "$CHECKPOINT" "$DIST/checkpoint.pt"
checksum "$DIST/$TARBALL"       > "$DIST/${TARBALL}.sha256"
checksum "$DIST/checkpoint.pt"  > "$DIST/checkpoint.pt.sha256"

ASSETS=(
  "$DIST/$TARBALL"
  "$DIST/${TARBALL}.sha256"
  "$DIST/checkpoint.pt"
  "$DIST/checkpoint.pt.sha256"
)

# Idempotent: create the release, or refresh assets on an existing one.
if gh release view "$VERSION" >/dev/null 2>&1; then
  echo "› release ${VERSION} exists — refreshing assets…"
  gh release upload "$VERSION" --clobber "${ASSETS[@]}"
else
  echo "› creating GitHub release ${VERSION}…"
  gh release create "$VERSION" \
    --title "Arbiter ${VERSION}" \
    --notes "Release ${VERSION}. See CHANGELOG.md." \
    "${ASSETS[@]}"
fi

# Publish npm only when auth is available; otherwise skip (CI without a token, or a local run).
if [[ -n "${NODE_AUTH_TOKEN:-}" ]] || npm whoami >/dev/null 2>&1; then
  echo "› publishing npm package (opencode-arbiter@${BARE})…"
  ( cd notary
    npm version "$BARE" --no-git-tag-version --allow-same-version >/dev/null
    npm publish --access public
  )
else
  echo "⚠ skipping npm publish: no npm auth. Set NPM_TOKEN (CI) or run 'npm login', then:"
  echo "    (cd notary && npm publish --access public)"
fi

echo "✓ released ${VERSION}"
