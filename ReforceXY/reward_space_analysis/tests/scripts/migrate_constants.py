#!/usr/bin/env python3
"""Script to migrate test constants from self.TEST_* to PARAMS/TOL/SEEDS imports.

This script performs automated refactoring to standardize constant access patterns
across the test suite, replacing class-based access (self.TEST_*) with direct imports
from the constants module.

Usage:
    python migrate_constants.py --dry-run  # Preview changes
    python migrate_constants.py --apply    # Apply changes

Features:
    - Automatic detection of self.TEST_* patterns
    - Smart import injection
    - Preserves formatting and comments
    - Generates detailed migration report
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


# Mapping: self.TEST_* -> (module, constant)
CONSTANT_MAPPINGS = {
    # Test parameters
    "self.TEST_BASE_FACTOR": ("PARAMS", "BASE_FACTOR"),
    "self.TEST_PROFIT_AIM": ("PARAMS", "PROFIT_AIM"),
    "self.TEST_RR": ("PARAMS", "RISK_REWARD_RATIO"),
    "self.TEST_RR_HIGH": ("PARAMS", "RISK_REWARD_RATIO_HIGH"),
    "self.TEST_PNL_STD": ("PARAMS", "PNL_STD"),
    "self.TEST_PNL_DUR_VOL_SCALE": ("PARAMS", "PNL_DUR_VOL_SCALE"),
    # Tolerances
    "self.TOL_IDENTITY_STRICT": ("TOLERANCE", "IDENTITY_STRICT"),
    "self.TOL_IDENTITY_RELAXED": ("TOLERANCE", "IDENTITY_RELAXED"),
    "self.TOL_GENERIC_EQ": ("TOLERANCE", "GENERIC_EQ"),
    "self.TOL_DISTRIB_SHAPE": ("TOLERANCE", "DISTRIB_SHAPE"),
    "self.TOL_NEGLIGIBLE": ("TOLERANCE", "NEGLIGIBLE"),
    "self.TOL_NUMERIC_GUARD": ("TOLERANCE", "NUMERIC_GUARD"),
    "self.TOL_RELATIVE": ("TOLERANCE", "RELATIVE"),
    # Seeds
    "self.SEED": ("SEEDS", "BASE"),
    "self.SEED_REPRODUCIBILITY": ("SEEDS", "REPRODUCIBILITY"),
    "self.SEED_BOOTSTRAP": ("SEEDS", "BOOTSTRAP"),
    "self.SEED_HETEROSCEDASTICITY": ("SEEDS", "HETEROSCEDASTICITY"),
    # Scenarios
    "self.TEST_SAMPLES": ("SCENARIOS", "SAMPLE_SIZE_SMALL"),
    "self.BOOTSTRAP_DEFAULT_ITERATIONS": ("SCENARIOS", "BOOTSTRAP_ITERATIONS"),
    # Continuity
    "self.CONTINUITY_EPS_SMALL": ("CONTINUITY", "EPS_SMALL"),
    "self.CONTINUITY_EPS_LARGE": ("CONTINUITY", "EPS_LARGE"),
    # Exit factor
    "self.EXIT_FACTOR_SCALING_RATIO_MIN": ("EXIT_FACTOR", "SCALING_RATIO_MIN"),
    "self.EXIT_FACTOR_SCALING_RATIO_MAX": ("EXIT_FACTOR", "SCALING_RATIO_MAX"),
    "self.MIN_EXIT_POWER_TAU": ("EXIT_FACTOR", "MIN_POWER_TAU"),
    # Statistical
    "self.BH_FP_RATE_THRESHOLD": ("STATISTICAL", "BH_FP_RATE_THRESHOLD"),
    # Default params (special case - keep as is)
    "self.DEFAULT_PARAMS": (None, None),  # Skip migration
}


def find_test_files(base_path: Path) -> List[Path]:
    """Find all test Python files recursively."""
    return sorted(base_path.rglob("test_*.py"))


def analyze_file(file_path: Path) -> Tuple[Dict[str, int], Set[str]]:
    """Analyze a file for self.TEST_* patterns.

    Returns:
        Tuple of (pattern_counts, required_imports)
    """
    content = file_path.read_text(encoding="utf-8")
    pattern_counts = defaultdict(int)
    required_imports = set()

    for pattern, (module, _) in CONSTANT_MAPPINGS.items():
        if module is None:
            continue  # Skip special cases
        matches = re.findall(re.escape(pattern), content)
        if matches:
            pattern_counts[pattern] = len(matches)
            required_imports.add(module)

    return dict(pattern_counts), required_imports


def check_existing_imports(content: str) -> Set[str]:
    """Check which constants are already imported."""
    imported = set()
    # Look for: from ..constants import PARAMS, TOL, ...
    import_pattern = r"from\s+\.\.constants\s+import\s+([^\n]+)"
    matches = re.findall(import_pattern, content)
    for match in matches:
        # Parse comma-separated imports
        for item in match.split(","):
            item = item.strip()
            if item and not item.startswith("("):
                imported.add(item)
    return imported


def inject_imports(content: str, required_imports: Set[str]) -> str:
    """Inject missing imports at the top of the file."""
    existing_imports = check_existing_imports(content)
    missing_imports = required_imports - existing_imports

    if not missing_imports:
        return content

    # Find the position to inject (after existing ..constants import or after last import)
    lines = content.split("\n")
    inject_pos = None

    # Try to find existing ..constants import
    for i, line in enumerate(lines):
        if "from ..constants import" in line or "from tests.constants import" in line:
            inject_pos = i
            # If it's a multi-line import, find the end
            if "(" in line and ")" not in line:
                for j in range(i + 1, len(lines)):
                    if ")" in lines[j]:
                        inject_pos = j
                        break
            break

    # If no ..constants import found, find last import
    if inject_pos is None:
        for i, line in enumerate(lines):
            if line.strip().startswith("import ") or line.strip().startswith("from "):
                inject_pos = i

    # If still no position found, inject after docstring/shebang
    if inject_pos is None:
        for i, line in enumerate(lines):
            if (
                line.strip()
                and not line.strip().startswith("#")
                and not line.strip().startswith('"""')
            ):
                inject_pos = i - 1
                break

    # Build import statement
    sorted_imports = sorted(missing_imports)

    if inject_pos is not None:
        # Check if there's already a constants import to extend
        constants_import_line = None
        for i in range(max(0, inject_pos - 3), min(len(lines), inject_pos + 3)):
            if "from ..constants import" in lines[i] or "from tests.constants import" in lines[i]:
                constants_import_line = i
                break

        if constants_import_line is not None:
            # Extend existing import
            line = lines[constants_import_line]
            if "(" in line:
                # Multi-line import - find closing paren
                end_line = constants_import_line
                for j in range(constants_import_line, len(lines)):
                    if ")" in lines[j]:
                        end_line = j
                        break
                # Add new imports before closing paren
                indent = "    "
                new_imports = ",\n".join(f"{indent}{imp}" for imp in sorted_imports)
                lines[end_line] = f"{indent}{', '.join(sorted_imports)},\n" + lines[end_line]
            else:
                # Single line import - extend it
                # Extract existing imports
                match = re.search(r"from\s+\.\.constants\s+import\s+(.+)", line)
                if match:
                    existing = [x.strip() for x in match.group(1).split(",")]
                    all_imports = sorted(set(existing + sorted_imports))
                    lines[constants_import_line] = (
                        f"from ..constants import {', '.join(all_imports)}"
                    )
        else:
            # Inject new import
            import_line = f"from ..constants import {', '.join(sorted_imports)}"
            lines.insert(inject_pos + 1, import_line)

    return "\n".join(lines)


def migrate_file(file_path: Path, dry_run: bool = True) -> Dict[str, Any]:
    """Migrate a single file.

    Returns:
        Migration report dict
    """
    content = file_path.read_text(encoding="utf-8")
    original_content = content

    pattern_counts, required_imports = analyze_file(file_path)

    if not pattern_counts:
        return {
            "path": str(file_path),
            "changed": False,
            "patterns": {},
            "imports_added": [],
        }

    # Inject imports first
    content = inject_imports(content, required_imports)

    # Replace patterns
    replacements = {}
    for pattern, (module, constant) in CONSTANT_MAPPINGS.items():
        if module is None or pattern not in pattern_counts:
            continue

        replacement = f"{module}.{constant}"
        count = content.count(pattern)
        content = content.replace(pattern, replacement)
        replacements[pattern] = (replacement, count)

    changed = content != original_content

    if changed and not dry_run:
        file_path.write_text(content, encoding="utf-8")

    return {
        "path": str(file_path),
        "changed": changed,
        "patterns": replacements,
        "imports_added": sorted(required_imports),
        "preview": content if dry_run else None,
    }


def generate_report(results: List[Dict], dry_run: bool):
    """Generate migration report."""
    print("\n" + "=" * 80)
    print(f"MIGRATION REPORT ({'DRY RUN' if dry_run else 'APPLIED'})")
    print("=" * 80 + "\n")

    changed_files = [r for r in results if r["changed"]]
    unchanged_files = [r for r in results if not r["changed"]]

    print(f"Files analyzed: {len(results)}")
    print(f"Files to migrate: {len(changed_files)}")
    print(f"Files unchanged: {len(unchanged_files)}")
    print()

    if changed_files:
        print("=" * 80)
        print("CHANGED FILES")
        print("=" * 80 + "\n")

        for result in changed_files:
            print(f"\n📝 {result['path']}")
            print(f"   Imports added: {', '.join(result['imports_added']) or 'None'}")
            print(f"   Pattern replacements:")

            for pattern, (replacement, count) in result["patterns"].items():
                print(f"      {pattern} -> {replacement} ({count} occurrences)")

    # Summary statistics
    print("\n" + "=" * 80)
    print("STATISTICS")
    print("=" * 80 + "\n")

    total_replacements = sum(
        sum(count for _, count in r["patterns"].values()) for r in changed_files
    )

    pattern_summary = defaultdict(int)
    for result in changed_files:
        for pattern, (_, count) in result["patterns"].items():
            pattern_summary[pattern] += count

    print(f"Total replacements: {total_replacements}")
    print(f"\nBy pattern:")
    for pattern, count in sorted(pattern_summary.items(), key=lambda x: -x[1]):
        print(f"   {pattern}: {count}")

    if dry_run:
        print("\n" + "=" * 80)
        print("⚠️  DRY RUN MODE - No files were modified")
        print("   Run with --apply to apply changes")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Migrate test constants to standardized imports")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without applying")
    parser.add_argument("--apply", action="store_true", help="Apply changes to files")
    parser.add_argument(
        "--path", type=str, default="tests", help="Base path to search for test files"
    )

    args = parser.parse_args()

    if not args.dry_run and not args.apply:
        parser.error("Must specify either --dry-run or --apply")

    # Get the project root (reward_space_analysis/)
    script_dir = Path(__file__).parent  # tests/scripts/
    tests_dir = script_dir.parent  # tests/
    base_path = tests_dir if args.path == "tests" else tests_dir.parent / args.path

    if not base_path.exists():
        print(f"❌ Path not found: {base_path}", file=sys.stderr)
        sys.exit(1)

    print(f"🔍 Scanning for test files in: {base_path}")
    test_files = find_test_files(base_path)
    print(f"   Found {len(test_files)} test files\n")

    if not test_files:
        print("❌ No test files found", file=sys.stderr)
        sys.exit(1)

    dry_run = args.dry_run
    results = []

    for test_file in test_files:
        print(f"Processing: {test_file.relative_to(base_path.parent)}", end="... ")
        result = migrate_file(test_file, dry_run=dry_run)
        results.append(result)
        print("✓" if result["changed"] else "○")

    generate_report(results, dry_run)

    if not dry_run:
        print("\n✅ Migration complete!")
        print("   Run tests to verify: pytest tests/ -v")


if __name__ == "__main__":
    main()
