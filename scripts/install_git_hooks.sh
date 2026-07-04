#!/usr/bin/env bash
set -euo pipefail

install_hook() {
  local repo_root hook_path
  repo_root="$(git rev-parse --show-toplevel)"
  hook_path="$repo_root/.git/hooks/pre-commit"
  install -m 0755 "$0" "$hook_path"
  echo "Installed Stage 1 freeze hook at $hook_path"
}

if [[ "${1:-}" == "--install" ]]; then
  install_hook
  exit 0
fi

protected_files="$(
  git diff --cached --name-only -- \
    ':(top)eval/**' \
    ':(top)dataguard/**'
)"

if [[ -z "$protected_files" ]]; then
  exit 0
fi

if [[ "${ALLOW_PROTECTED_CHANGES:-0}" == "1" ]]; then
  cat >&2 <<'MESSAGE'
Protected-change override accepted.
eval/ and dataguard/ are frozen; this override is valid only with explicit human approval.
Rerun the full unit tests and raw-data validation, document the reason and diff, and record the commit hash.
MESSAGE
  exit 0
fi

cat >&2 <<'MESSAGE'
COMMIT BLOCKED: eval/ and dataguard/ are frozen Stage 1 controls.

Protected changes require explicit human approval.
After an approved change, rerun the full unit tests and raw-data validation and document the reason, files, diff summary, results, and commit hash.

Use ALLOW_PROTECTED_CHANGES=1 only with explicit approval.
MESSAGE

printf '\nProtected staged files:\n%s\n' "$protected_files" >&2
exit 1
