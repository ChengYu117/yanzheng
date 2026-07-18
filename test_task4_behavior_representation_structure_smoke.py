"""Smoke checks for the reviewed Task 4 grouping definition."""

from __future__ import annotations

from src.nlp_re_base.task4_behavior_structure import (
    COMPONENTS,
    LABEL_ORDER,
    NONBEHAVIOR_COMPONENTS,
    NONBEHAVIOR_UNASSIGNED,
    UNASSIGNED,
)


def main() -> None:
    assert set(COMPONENTS) == set(LABEL_ORDER)
    assert set(UNASSIGNED) == set(LABEL_ORDER)
    assert set(NONBEHAVIOR_COMPONENTS) == set(LABEL_ORDER)
    assert set(NONBEHAVIOR_UNASSIGNED) == set(LABEL_ORDER)
    for label in LABEL_ORDER:
        assigned = []
        for component in COMPONENTS[label]:
            assert len(component["members"]) >= 2
            assigned.extend(component["members"])
        assigned.extend(UNASSIGNED[label])
        assert len(assigned) == len(set(assigned)), f"duplicate assignment in {label}"
        nonbehavior_assigned = []
        for component in NONBEHAVIOR_COMPONENTS[label]:
            assert len(component["members"]) >= 2
            nonbehavior_assigned.extend(component["members"])
        nonbehavior_assigned.extend(NONBEHAVIOR_UNASSIGNED[label])
        assert len(nonbehavior_assigned) == len(set(nonbehavior_assigned)), (
            f"duplicate nonbehavior assignment in {label}"
        )
    print("Task 4 reviewed-grouping smoke test passed")


if __name__ == "__main__":
    main()
