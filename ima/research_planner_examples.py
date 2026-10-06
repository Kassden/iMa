"""Executable adapter recipes and compatibility guidance for research planning."""


def adapter_recipe_examples() -> list[dict]:
    examples = []
    for model, distribution in (
        ("ridge_regressor", "shared_residual"),
        ("hist_gradient_regressor", "shared_residual"),
        ("catboost_regressor", "catboost_uncertainty"),
    ):
        graph = {
            "graph_id": f"{model}-speed-distribution-to-win",
            "primary_node_id": "speed", "fundamental_node_id": "win",
            "output_node_id": "win", "date_column": "date",
            "nodes": [
                {"node_id": "speed", "kind": "estimator", "parameters": {
                    "model_kind": model,
                    "performance_distribution": {
                        "kind": distribution, "coordinate": "log_speed_mps",
                    },
                }, "output": {"kind": "performance_distribution", "target": "speed",
                              "unit": "log_mps"}},
                {"node_id": "win", "kind": "probabilistic_adapter", "inputs": ["speed"]},
            ],
        }
        examples.append({
            "schema_version": 3, "target": {"kind": "win_probability"},
            "model": {"kind": model}, "calibration": {"kind": "none"},
            "blend": {"kind": "none"}, "pipeline_graph": graph,
        })
    return examples


MODEL_TARGET_GUIDANCE = {
    "bare_regressor_win": "Invalid: ridge_regressor, hist_gradient_regressor and catboost_regressor "
        "cannot directly predict win_probability. Use a complete adapter graph example; "
        "its speed estimator learns observed physical speed, while the outer target remains win_probability.",
    "speed_distribution": "Use output target=speed, kind=performance_distribution, unit=log_mps. "
        "shared_residual learns scale from later chronological calibration observations; "
        "catboost_uncertainty learns conditional mean and variance. Speed labels are training-only.",
    "adapter": "probabilistic_adapter takes one performance_distribution parent and produces "
        "race-normalized win probabilities. Normalizing raw point predictions is not a substitute.",
    "native_probit": "gaussian_probit directly supports win_probability, including its heteroscedastic "
        "option; it does not require an explicit adapter node.",
    "ranking": "Bare ranking models use ranking_strength. A supported graph ranking_score estimator "
        "can feed race_normalize, which fits a chronological probability adapter. Do not assume every "
        "standalone ranking backend is executable inside a graph.",
    "graph_model": "Outer model.kind must equal the primary estimator model_kind; use registered "
        "model_parameters there. Graph recipes use outer calibration=none and blend=none. "
        "Any market stage must be explicit, never a fundamental ancestor.",
    "selection": "Examples describe available choices, not required allocations. Select only when "
        "the dataset supplies the observed target and the hypothesis justifies the cost. Preserve 80/20 policy.",
}
