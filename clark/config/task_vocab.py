"""
Standard task type definitions for Clark.

Every facility has access to this vocabulary. Tasks are opt-in via the
facility config's `tasks.enabled` list. The three core tasks (pick, pack,
idle) are always present and cannot be disabled.

Custom tasks can be added via `tasks.custom` in the facility YAML.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskDef:
    """Immutable definition of a task type."""
    task_id: str            # canonical ID used in code and config
    display_name: str       # human-readable label
    output_type: str        # "orders_per_hour" | "units_per_hour" | "hours" | "none"
    hustle_eligible: bool   # can a worker hustle this task?
    always_enabled: bool = False  # True for pick/pack/idle — cannot be disabled


# ─── Core (always present) ────────────────────────────────────────────────────

PICK = TaskDef(
    task_id="pick",
    display_name="Picking",
    output_type="orders_per_hour",
    hustle_eligible=True,
    always_enabled=True,
)

PACK = TaskDef(
    task_id="pack",
    display_name="Packing",
    output_type="orders_per_hour",
    hustle_eligible=True,
    always_enabled=True,
)

IDLE = TaskDef(
    task_id="idle",
    display_name="Idle",
    output_type="none",
    hustle_eligible=False,
    always_enabled=True,
)

# ─── Operations (optional) ───────────────────────────────────────────────────

RESTOCK = TaskDef(
    task_id="restock",
    display_name="Restocking",
    output_type="hours",
    hustle_eligible=True,
)

MANAGEMENT = TaskDef(
    task_id="management",
    display_name="Management",
    output_type="hours",
    hustle_eligible=False,  # admin work can't be rushed
)

CYCLE_COUNT = TaskDef(
    task_id="cycle_count",
    display_name="Cycle Count",
    output_type="hours",
    hustle_eligible=False,  # audits require careful attention
)

SIDE_PROJECT = TaskDef(
    task_id="side_project",
    display_name="Side Project",
    output_type="hours",
    hustle_eligible=True,
)

RECEIVING = TaskDef(
    task_id="receiving",
    display_name="Receiving",
    output_type="units_per_hour",
    hustle_eligible=True,
)

# ─── Extended (less common) ──────────────────────────────────────────────────

LOADING = TaskDef(
    task_id="loading",
    display_name="Loading",
    output_type="units_per_hour",
    hustle_eligible=False,
)

RETURNS_PROCESSING = TaskDef(
    task_id="returns_processing",
    display_name="Returns Processing",
    output_type="units_per_hour",
    hustle_eligible=False,
)

QUALITY_CHECK = TaskDef(
    task_id="quality_check",
    display_name="Quality Check",
    output_type="orders_per_hour",
    hustle_eligible=False,
)

TRAINING = TaskDef(
    task_id="training",
    display_name="Training",
    output_type="hours",
    hustle_eligible=False,
)


# ─── Registry ────────────────────────────────────────────────────────────────

STANDARD_VOCAB: dict[str, TaskDef] = {
    t.task_id: t for t in [
        PICK, PACK, IDLE,
        RESTOCK, MANAGEMENT, CYCLE_COUNT, SIDE_PROJECT, RECEIVING,
        LOADING, RETURNS_PROCESSING, QUALITY_CHECK, TRAINING,
    ]
}

CORE_TASK_IDS: frozenset[str] = frozenset(
    t.task_id for t in STANDARD_VOCAB.values() if t.always_enabled
)

HUSTLE_BLOCKED_IDS: frozenset[str] = frozenset(
    t.task_id for t in STANDARD_VOCAB.values() if not t.hustle_eligible
)


def get_task(task_id: str) -> TaskDef:
    """Return a standard task def by ID. Raises KeyError for unknown IDs."""
    return STANDARD_VOCAB[task_id]
