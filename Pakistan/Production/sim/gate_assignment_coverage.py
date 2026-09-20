"""Supplement the established gate with OBI, QT_2t@20 and DROP coverage."""
# Read the selected assignment and output paths without duplicating gate options.
import argparse
# Read the completed manifests for explicit scope verification.
import json
# Resolve direct-script imports independently of the caller's directory.
from pathlib import Path
# Keep the same interpreter path rules as the strict runner.
import sys
# Make Production importable for direct invocation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Reuse the identical frozen runner and worker function.
from sim import gate_refresh as runner
# Use the production assignment validator.
from venues.psx_config import LiveConfig


# Select representative names from the actual supplied assignment.
def main():
    # Parse only the fields this supplementary scope needs.
    parser = argparse.ArgumentParser(add_help=False)
    # Require the production assignment whose uncovered groups will be selected.
    parser.add_argument("--assignment", type=Path, required=True)
    # Keep manual interventions consistent with the full assigned gate.
    parser.add_argument("--overrides", type=Path)
    # Locate the original runner's evidence directory.
    parser.add_argument("--output-dir", type=Path, required=True)
    # Reject a historical profile in this assignment-specific supplement.
    parser.add_argument("--profile", choices=("assigned",), required=True)
    # Inspect scope without swallowing the original runner's other arguments.
    args, remainder = parser.parse_known_args()
    # Coverage is across the full twenty dates, not a smoke substitute.
    if "--smoke" in remainder:
        # Prevent the two-name smoke selection from omitting the DROP case.
        parser.error("assignment coverage requires the full twenty-date scope")
    # Resolve labels using the production loader, including overrides.
    config = LiveConfig(args.assignment, args.overrides)
    # The established twelve-name sample happens to cover only QT_2t@15.
    labels = ("OBI", "QT_2t@20", "DROP")
    # Keep selected names and expected labels explicit.
    selected = {}
    # Require a representative of each additional category.
    for label in labels:
        # Sorted production symbols make selection deterministic.
        candidates = [s for s in config.symbols(quoting_only=False) if config.params_for(s)["label"] == label]
        # Missing groups cannot be described as tested.
        if not candidates:
            # Explain the coverage limitation before any replay.
            raise ValueError(f"assignment has no representative for {label}")
        # Select one representative without altering its assigned parameters.
        selected[candidates[0]] = label
    # Change only the declared sample; replay and risk logic remain identical.
    runner.G.GATE_NAMES = list(selected)
    # The standard runner records exact per-cell names, parameters and exclusions.
    status = runner.main()
    # Read the completed evidence to label this supplementary scope accurately.
    path = args.output_dir / "summary.json"
    # The underlying default label describes the standard twelve-name scope.
    summary = json.loads(path.read_text())
    # Replace that label with this explicitly different coverage scope.
    summary["scope"] = "assignment_group_supplement_20_dates"
    # Preserve the selected representative and its expected assigned group.
    summary["representatives"] = selected
    # Verify the complete requested active and excluded scope.
    valid_scope = summary["planned"] == 40 and len(summary["excluded"]) == 20
    # Scope failure invalidates the supplementary gate.
    summary["passed"] = summary["passed"] and valid_scope
    # Persist the corrected, authoritative scope description.
    runner.save(path, summary)
    # Print the final supplementary verdict after the runner's default summary.
    print(json.dumps(summary), flush=True)
    # Fail the shell command on a mismatch or incomplete coverage.
    return status if summary["passed"] else 1


# Spawned workers must not dispatch a second parent run.
if __name__ == "__main__":
    # Preserve a nonzero gate result for shell automation.
    raise SystemExit(main())
