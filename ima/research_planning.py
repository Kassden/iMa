"""Deterministic paid-planning admission, independent of worker admission."""
import math


def planning_reason(*, now, last_plan, terminal_since_plan, terminal_threshold,
                    remaining_trials, low_watermark, active_jobs, available_program_slots,
                    checkpoint_seconds, lane_refill_required=False, material_change=False):
    values = (now, last_plan, checkpoint_seconds)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Finite nonnegative planning times required")
    if min(terminal_since_plan, remaining_trials, active_jobs, available_program_slots, low_watermark) < 0 or terminal_threshold < 1:
        raise ValueError("Invalid planning counters")
    if last_plan == 0:
        return "startup_review"
    elapsed = now - last_plan
    # A missing mandatory lane prevents dispatch, unlike a healthy backlog.
    if lane_refill_required and remaining_trials > 0 and available_program_slots and elapsed >= 5:
        return "required_lane_refill"
    if elapsed < min(max(checkpoint_seconds, 1), 60):
        return None
    if material_change:
        return "material_result_change"
    if terminal_since_plan >= terminal_threshold:
        return "completed_trial_batch"
    if not active_jobs and elapsed >= max(checkpoint_seconds, 900):
        # Queued but undispatchable work needs a review, not indefinite silence.
        return "idle_queue_review"
    if remaining_trials <= low_watermark and available_program_slots:
        if terminal_since_plan > 0:
            return "queue_refill_after_results"
    # A timer tick is not new research evidence while useful work remains.
    return None


def observe_material_result(result, recipe, best_values):
    """Notify on a new comparable champion or execution failure, not raw scores."""
    if result.get("status") != "completed":
        return True
    value = result.get("objective_value")
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return False
    from .research_telemetry import comparison_key
    key = comparison_key(result, recipe)
    previous = best_values.get(key)
    best_values[key] = min(value, previous) if previous is not None else value
    return previous is None or value < previous
