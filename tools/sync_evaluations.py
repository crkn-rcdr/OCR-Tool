#!/usr/bin/env python3
"""Batch Synchronization Tool for CRKN OCR-Evaluations GitHub Pages Repository.

Safely commits and pushes large evaluation assets (HTML viewers, summary JSON/CSV,
and reference page scans) to GitHub Pages in bite-sized batches (publication by publication)
to permanently prevent HTTP 500 / curl 22 payload disconnect errors on GitHub.

Usage:
    python tools/sync_evaluations.py
    python tools/sync_evaluations.py --dry-run
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_git(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    """Execute a git command with UTF-8 encoding and output capture."""
    res = subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace"
    )
    if check and res.returncode != 0:
        raise RuntimeError(f"Git command failed: {' '.join(cmd)}\nError: {res.stderr or res.stdout}")
    return res


def sync_evaluations(eval_repo_dir: Path, dry_run: bool = False) -> bool:
    """Commit and push OCR-Evaluations in granular batches to avoid GitHub HTTP payload timeouts."""
    if not eval_repo_dir.exists() or not (eval_repo_dir / ".git").exists():
        print(f"[ERROR] OCR-Evaluations repository not found at: {eval_repo_dir}")
        return False

    print("==================================================================")
    print("      CRKN OCR-EVALUATIONS ROBUST BATCH PUSH SYNCHRONIZER         ")
    print("==================================================================")
    print(f"Target Repo: {eval_repo_dir}")
    print(f"Dry Run    : {dry_run}")
    print("==================================================================")

    # 1. Ensure git network buffers are configured
    try:
        run_git(["git", "config", "http.postBuffer", "524288000"], eval_repo_dir, check=False)
        run_git(["git", "config", "http.maxRequestBuffer", "524288000"], eval_repo_dir, check=False)
        run_git(["git", "config", "http.version", "HTTP/1.1"], eval_repo_dir, check=False)
    except Exception:
        pass

    docs_dir = eval_repo_dir / "docs"
    if not docs_dir.exists():
        print(f"[ERROR] docs/ directory not found in {eval_repo_dir}")
        return False

    # 2. Sync Top-Level Dashboard Files First
    top_level_files = [
        "docs/index.html",
        "docs/comparison_summary.json",
        "docs/comparison_summary.csv",
    ]
    staged_any = False
    for tlf in top_level_files:
        if (eval_repo_dir / tlf).exists():
            run_git(["git", "add", tlf], eval_repo_dir, check=False)
            staged_any = True

    diff_staged = run_git(["git", "diff", "--cached", "--name-only"], eval_repo_dir, check=False)
    if diff_staged.stdout.strip():
        print("\n[Batch 1] Pushing Master AI Dashboard and summary tables...")
        if not dry_run:
            run_git(["git", "commit", "-m", "Auto-update Master AI Dashboard and summary tables"], eval_repo_dir)
            push_res = run_git(["git", "push", "origin", "main"], eval_repo_dir, check=False)
            if push_res.returncode != 0:
                print(f"[WARN] Retrying push for dashboard files: {push_res.stderr}")
                time.sleep(2)
                run_git(["git", "push", "origin", "main"], eval_repo_dir)
            print("  [OK] Successfully pushed Master AI Dashboard.")
    else:
        print("\n[Info] Master AI Dashboard is already up-to-date on origin.")

    # 3. Discover All Modified or Untracked Publication Folders Under docs/
    # Folders follow the structure: docs/<Collection>/<Publication_Name>
    status_out = run_git(["git", "status", "--porcelain", "-uall", "docs/"], eval_repo_dir, check=False).stdout
    if not status_out.strip():
        print("\n[CRKN Sync] All evaluation docs and scans are 100% up-to-date with origin/main!")
        return True

    # Identify distinct publication folders needing sync
    target_folders = set()
    for line in status_out.splitlines():
        parts = line.strip().split(maxsplit=1)
        if len(parts) < 2:
            continue
        rel_path = Path(parts[1].strip('"'))
        # rel_path typically starts with docs/Collection/Publication/...
        subparts = rel_path.parts
        if len(subparts) >= 3 and subparts[0] == "docs":
            # docs / Collection / Publication
            pub_folder = f"{subparts[0]}/{subparts[1]}/{subparts[2]}"
            target_folders.add(pub_folder)
        elif len(subparts) >= 2 and subparts[0] == "docs":
            pub_folder = f"{subparts[0]}/{subparts[1]}"
            target_folders.add(pub_folder)

    sorted_folders = sorted(list(target_folders))
    print(f"\n[CRKN Sync] Found {len(sorted_folders)} publication batches needing sync.")

    success_count = 0
    fail_count = 0

    for idx, folder in enumerate(sorted_folders, 1):
        print(f"\n[{idx}/{len(sorted_folders)}] Processing: {folder}...")
        if dry_run:
            print(f"  [DRY-RUN] Would stage, commit, and push: {folder}")
            continue

        try:
            # Stage publication folder
            run_git(["git", "add", folder], eval_repo_dir)
            diff_check = run_git(["git", "diff", "--cached", "--name-only"], eval_repo_dir, check=False)
            if not diff_check.stdout.strip():
                print("  • No changes staged (skipped).")
                continue

            # Count files in this commit
            staged_files = [f for f in diff_check.stdout.splitlines() if f.strip()]
            commit_msg = f"Add comparison pages and scan images for {folder.replace('docs/', '')}"
            run_git(["git", "commit", "-m", commit_msg], eval_repo_dir)

            # Push batch to origin main
            push_res = run_git(["git", "push", "origin", "main"], eval_repo_dir, check=False)
            if push_res.returncode != 0:
                print(f"  [WARN] Initial push failed; retrying in 3 seconds: {push_res.stderr.strip()[:120]}")
                time.sleep(3)
                push_res = run_git(["git", "push", "origin", "main"], eval_repo_dir, check=False)

            if push_res.returncode == 0:
                print(f"  [OK] Successfully pushed {len(staged_files)} assets for {folder}.")
                success_count += 1
            else:
                print(f"  [FAIL] Failed to push {folder}: {push_res.stderr}")
                fail_count += 1

        except Exception as e:
            print(f"  [FAIL] Error syncing {folder}: {e}")
            fail_count += 1

    print("\n==================================================================")
    print("                      BATCH SYNC COMPLETE                         ")
    print("==================================================================")
    print(f"Successfully Pushed: {success_count} batches")
    print(f"Failed Batches     : {fail_count}")
    print("==================================================================")
    return fail_count == 0


def main():
    parser = argparse.ArgumentParser(description="Safely sync OCR-Evaluations to GitHub Pages in bite-sized batches.")
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path.home() / "Documents" / "OCR-Evaluations",
        help="Path to local OCR-Evaluations clone"
    )
    parser.add_argument("--dry-run", action="store_true", help="Simulate without pushing")
    args = parser.parse_args()

    success = sync_evaluations(args.eval_dir, dry_run=args.dry_run)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
