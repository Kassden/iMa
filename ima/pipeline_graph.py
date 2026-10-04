"""Typed model DAGs with recursive chronological cross-fitting and replay state.

Every learned downstream node sees forward predictions of its entire upstream
graph. Final ancestor refits use all supplied past rows, as in standard stacking.
No score frame is accepted by a fitting method.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import copy
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd
import scipy
import sklearn

from .feature_sets import FeatureSchema
from .modeling import MarketBlend, RaceProbabilityModel, TemperatureCalibrator, normalize_by_race
from .prediction_store import PredictionArtifact, PredictionStore, frame_fingerprint, prediction_key
from .probabilistic_adapters import distribution_to_win, distribution_to_joint, ranking_scores_to_probabilities, win_to_joint


@dataclass(frozen=True)
class OutputContract:
    kind: str = "win_probability"
    target: str = "win_probability"
    unit: str = "probability"
    market: bool = False

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        return cls(**dict(value or {}))

    def validate(self):
        allowed = {"win_probability": {"probability"}, "ranking_score": {"score"},
                   "point_prediction": {"mps", "seconds", "log_seconds", "latent"},
                   "performance_distribution": {"log_mps", "log_seconds", "latent"},
                   "feature_frame": {"features"}, "joint_order": {"probability"}}
        if self.kind not in allowed or self.unit not in allowed[self.kind]:
            raise ValueError(f"Incompatible output kind/unit: {self.kind}/{self.unit}")
        if self.kind == "win_probability" and self.target != "win_probability":
            raise ValueError("Win output must have win_probability target")


@dataclass(frozen=True)
class GraphNode:
    node_id: str
    kind: str
    inputs: tuple[str, ...] = ()
    parameters: dict = field(default_factory=dict)
    output: OutputContract = field(default_factory=OutputContract)
    fit_scope: str = "forward_oof"

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        raw = dict(value)
        raw["inputs"] = tuple(raw.get("inputs", ()))
        raw["output"] = OutputContract.from_dict(raw.get("output"))
        return cls(**raw)


def _learned_stage(node, nodes):
    return node.kind in {"meta_estimator", "calibrate", "market_blend"} or (
        node.kind == "race_normalize" and nodes[node.inputs[0]].output.kind == "ranking_score"
    )


@dataclass(frozen=True)
class PipelineGraph:
    graph_id: str
    nodes: tuple[GraphNode, ...]
    output_node_id: str
    fundamental_node_id: str | None = None
    primary_node_id: str | None = None
    joint_node_id: str | None = None
    date_column: str = "date"
    horse_column: str = "horse_id"
    n_splits: int = 3
    min_train_dates: int = 2
    schema_version: int = 1
    dataset_id: str = "unspecified"
    target_contract_id: str = "win_probability"
    evaluation_contract_id: str = "chronological"
    fit_budget: int | None = None
    component_refs: tuple = ()

    @classmethod
    def from_dict(cls, value):
        if isinstance(value, cls):
            return value
        raw = copy.deepcopy(value)
        edges = raw.pop("edges", ())
        inputs = {}
        for edge in edges:
            source, target = (edge["source"], edge["target"]) if isinstance(edge, dict) else edge
            inputs.setdefault(target, []).append(source)
        raw["nodes"] = tuple(GraphNode.from_dict(dict(node, inputs=node.get("inputs", inputs.get(node["node_id"], ()))))
                             if isinstance(node, dict) else node for node in raw["nodes"])
        if "output_node_id" not in raw:
            raw["output_node_id"] = raw.pop("output", raw["nodes"][-1].node_id)
        raw["component_refs"] = tuple(raw.get("component_refs", ()))
        return cls(**raw)

    def validate(self):
        nodes = {n.node_id: n for n in self.nodes}
        if len(nodes) != len(self.nodes) or not nodes:
            raise ValueError("Graph node IDs must be unique and nonempty")
        if self.n_splits < 2 or self.min_train_dates < 1 or self.schema_version != 1:
            raise ValueError("Invalid chronological graph protocol/version")
        if self.component_refs:
            raise ValueError("External fitted references are unsupported; supply refittable estimator factories")
        if self.primary_node_id and (self.primary_node_id not in nodes or nodes[self.primary_node_id].kind != "estimator"):
            raise ValueError("primary_node_id must name an estimator")
        if self.output_node_id not in nodes:
            raise ValueError("Unknown evaluation output")
        if self.joint_node_id and (self.joint_node_id not in nodes or nodes[self.joint_node_id].output.kind != "joint_order"):
            raise ValueError("joint_node_id must identify complete joint orders")
        visited, active = set(), set()

        def visit(key):
            if key in active:
                raise ValueError("Pipeline graph cycle")
            if key in visited:
                return
            if key not in nodes:
                raise ValueError(f"Unknown graph input: {key}")
            active.add(key)
            node = nodes[key]
            node.output.validate()
            if node.fit_scope not in {"forward_oof", "training_only", "none"}:
                raise ValueError("In-sample/future fit scope is forbidden")
            for parent in node.inputs:
                visit(parent)
            active.remove(key)
            visited.add(key)

        for key in nodes:
            visit(key)
        fundamental = self.fundamental_node_id or self.output_node_id
        if fundamental not in nodes:
            raise ValueError("Unknown fundamental output")
        reachable = set()

        def collect(key):
            if key not in reachable:
                reachable.add(key)
                for parent in nodes[key].inputs:
                    collect(parent)

        for root in (self.output_node_id, fundamental, self.joint_node_id):
            if root is not None:
                collect(root)
        if self.primary_node_id is not None and self.primary_node_id not in reachable:
            raise ValueError("primary_node_id must be used by an output, fundamental, or joint endpoint")
        supported = {"estimator", "feature_view", "race_normalize", "calibrate",
                     "weighted_probability_pool", "log_probability_pool", "meta_estimator",
                     "market_blend", "probabilistic_adapter", "forward_oof_predict", "rank_distribution"}
        for node in nodes.values():
            if node.kind not in supported:
                raise ValueError(f"Unsupported graph stage: {node.kind}")
            parents = [nodes[i] for i in node.inputs]
            if node.kind in {"weighted_probability_pool", "log_probability_pool", "meta_estimator"}:
                if not parents or node.output.kind != "win_probability" or any(p.output != parents[0].output for p in parents):
                    raise ValueError("Pool/stack inputs require identical target/unit/market contracts")
                if parents[0].output.kind != "win_probability" or node.output != parents[0].output:
                    raise ValueError("Pool/stack is only defined for compatible win probabilities")
            elif node.kind in {"calibrate", "market_blend", "forward_oof_predict"}:
                if len(parents) != 1 or parents[0].output.kind != "win_probability":
                    raise ValueError("Calibration/market inputs require one win probability")
                if node.kind != "market_blend" and node.output != parents[0].output:
                    raise ValueError("Pass-through/calibration cannot change contracts")
                if node.kind == "market_blend" and (parents[0].output.market or not node.output.market or node.output.kind != "win_probability"):
                    raise ValueError("Market blend requires fundamental input and declared market output")
            elif node.kind == "race_normalize":
                if len(parents) != 1 or parents[0].output.kind not in {"ranking_score", "win_probability"} or node.output.kind != "win_probability":
                    raise ValueError("Race normalization requires rank/probability input")
            elif node.kind == "probabilistic_adapter":
                if len(parents) != 1 or parents[0].output.kind != "performance_distribution" or node.output.kind != "win_probability":
                    raise ValueError("Distribution adapter requires a native distribution input")
            elif node.kind == "rank_distribution":
                if len(parents) != 1 or parents[0].output.kind not in {"performance_distribution", "win_probability"} or node.output.kind != "joint_order":
                    raise ValueError("Joint-order nodes require one native distribution or explicit win-strength adapter")
            elif node.kind == "estimator":
                if len(parents) > 1 or any(p.kind != "feature_view" for p in parents):
                    raise ValueError("Estimator inputs may only select a feature view")
            elif node.kind == "feature_view" and (parents or node.output.kind != "feature_frame"):
                raise ValueError("Feature view must be a feature-frame root")
            learned = _learned_stage(node, nodes)
            if learned and node.fit_scope != "forward_oof":
                raise ValueError("Learned downstream stages require forward_oof fit scope")
        def market_ancestor(key):
            return nodes[key].output.market or nodes[key].kind == "market_blend" or any(market_ancestor(p) for p in nodes[key].inputs)
        if market_ancestor(fundamental):
            raise ValueError("Fundamental output has a market ancestor")
        if any(not n.output.market and any(nodes[p].output.market for p in n.inputs) for n in nodes.values()):
            raise ValueError("Market dependency cannot be relabeled fundamental")
        return self


def _validated_probabilities(values, frame):
    p = np.asarray(values, float)
    if p.shape != (len(frame),) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Invalid graph probability shape/support")
    if "field_size" in frame:
        sizes = frame.groupby("race_id")["race_id"].transform("size").to_numpy()
        if not np.array_equal(sizes, pd.to_numeric(frame["field_size"], errors="coerce").to_numpy()):
            raise ValueError("Graph win predictions require complete fields")
    totals = pd.Series(p).groupby(frame["race_id"].to_numpy()).sum()
    if not np.allclose(totals, 1, atol=1e-8, rtol=0):
        raise ValueError("Graph win probabilities are not race-normalized")
    return p


@dataclass
class FittedGraphNode:
    spec: GraphNode
    parents: tuple["FittedGraphNode", ...] = ()
    state: Any = None
    feature_columns: tuple[str, ...] = ()

    def predict(self, frame):
        node = self.spec
        if node.kind == "feature_view":
            return frame[list(self.feature_columns)]
        if node.kind == "estimator":
            if node.output.kind == "performance_distribution":
                distribution = self.state.predict_distribution(frame)
                expected_unit = {"latent_strength": "latent", "log_speed_mps": "log_mps", "log_time_seconds": "log_seconds"}[distribution.coordinate]
                if node.output.unit != expected_unit:
                    raise ValueError("Native distribution violates the declared coordinate/unit")
                return distribution
            method = self.state.predict_proba if node.output.kind == "win_probability" else self.state.predict
            values = method(frame)
        else:
            outputs = [p.predict(frame) for p in self.parents]
            if node.kind in {"weighted_probability_pool", "log_probability_pool"}:
                weights = np.asarray(node.parameters.get("weights", np.ones(len(outputs))), float)
                if weights.shape != (len(outputs),) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
                    raise ValueError("Pool weights must be finite, nonnegative and nonzero")
                matrix = np.column_stack([_validated_probabilities(p, frame) for p in outputs])
                if node.kind == "log_probability_pool":
                    scores = np.log(np.maximum(matrix, np.finfo(float).tiny)) @ (weights/weights.sum())
                    values = ranking_scores_to_probabilities(scores, frame["race_id"])
                else:
                    values = normalize_by_race(matrix @ (weights/weights.sum()), frame["race_id"])
            elif node.kind == "meta_estimator":
                values = self.state.predict_proba(self._meta_frame(frame, outputs))
            elif node.kind == "calibrate":
                values = self.state.transform(outputs[0], frame["race_id"])
            elif node.kind == "market_blend":
                market = _validated_probabilities(frame[node.parameters.get("market_column", "market_probability")].to_numpy(), frame)
                availability = node.parameters.get("availability_column")
                if availability and not frame[availability].fillna(False).all():
                    raise ValueError("Market quote unavailable at prediction cutoff")
                values = self.state.transform(outputs[0], market, frame["race_id"])
            elif node.kind == "race_normalize":
                if self.parents[0].spec.output.kind == "ranking_score":
                    if self.state is None:
                        raise ValueError("Ranking probability adapter lacks fitted forward-OOF calibration")
                    values = self.state.transform(ranking_scores_to_probabilities(outputs[0], frame["race_id"]), frame["race_id"])
                else:
                    values = normalize_by_race(outputs[0], frame["race_id"])
            elif node.kind == "probabilistic_adapter":
                values = distribution_to_win(outputs[0], frame["race_id"])
            elif node.kind == "rank_distribution":
                from .performance_distributions import PerformanceDistribution
                result = {}
                codes, races = pd.factorize(frame["race_id"], sort=False)
                horse_column = node.parameters.get("horse_column", "horse_id")
                settings = {k: v for k, v in node.parameters.items() if k != "horse_column"}
                for code, race in enumerate(races):
                    rows = np.flatnonzero(codes == code)
                    runners = frame.iloc[rows][horse_column].astype(str).tolist()
                    if self.parents[0].spec.output.kind == "performance_distribution":
                        dist = outputs[0]
                        selected = PerformanceDistribution(dist.location[rows], dist.scale[rows], dist.coordinate,
                            dist.direction, metadata=dist.metadata.copy())
                        result[str(race)] = distribution_to_joint(selected, runners, **settings)
                    else:
                        result[str(race)] = win_to_joint(outputs[0][rows], runners, **settings)
                return result
            elif node.kind == "forward_oof_predict":
                values = outputs[0]
            else:
                raise ValueError(f"Unsupported inference stage: {node.kind}")
        if node.output.kind == "win_probability":
            return _validated_probabilities(values, frame)
        values = np.asarray(values, float)
        if values.shape != (len(frame),) or not np.isfinite(values).all():
            raise ValueError("Invalid graph prediction shape/values")
        if node.output.unit in {"mps", "seconds"} and np.any(values <= 0):
            raise ValueError("Physical point prediction must have positive support")
        return values

    def _meta_frame(self, frame, outputs):
        result = frame[[c for c in ("race_id", "target_win") if c in frame]].copy()
        for column, values in zip(self.feature_columns, outputs):
            result[column] = np.log(np.maximum(values, 1e-12))
        return result


@dataclass
class FittedPipelineGraph:
    graph: PipelineGraph
    output: FittedGraphNode
    fundamental: FittedGraphNode
    fit_report: dict
    joint: FittedGraphNode | None = None

    def _check_cutoff(self, frame):
        dates = pd.to_datetime(frame[self.graph.date_column], errors="raise")
        if len(frame) == 0 or dates.isna().any() or dates.min() <= pd.Timestamp(self.fit_report["training_cutoff"]):
            raise ValueError("Graph inference must follow the fitted training cutoff")

    def predict_fundamental_proba(self, frame):
        self._check_cutoff(frame)
        if self.fundamental.spec.output.kind != "win_probability":
            raise TypeError("Graph has no declared fundamental win endpoint")
        return _validated_probabilities(self.fundamental.predict(frame), frame)

    def predict_proba(self, frame):
        self._check_cutoff(frame)
        if self.output.spec.output.kind != "win_probability":
            raise TypeError("Graph output is not win_probability")
        return _validated_probabilities(self.output.predict(frame), frame)

    def predict(self, frame):
        self._check_cutoff(frame)
        result = self.output.predict(frame)
        return result.physical_mean() if self.output.spec.output.kind == "performance_distribution" else result

    def predict_distribution(self, frame):
        self._check_cutoff(frame)
        if self.output.spec.output.kind != "performance_distribution":
            raise TypeError("Graph output is not a performance distribution")
        return self.output.predict(frame)

    def predict_joint(self, frame):
        self._check_cutoff(frame)
        joint = self.joint or (self.output if self.output.spec.output.kind == "joint_order" else None)
        if joint is None:
            raise TypeError("Graph has no declared joint-order endpoint")
        return joint.predict(frame)


def _schema_subset(schema, columns):
    columns = tuple(columns)
    if not columns or not set(columns) <= set(schema.features):
        raise ValueError("Feature view must select declared schema columns")
    return FeatureSchema(name="graph_view", numeric=tuple(c for c in schema.numeric if c in columns),
                         categorical=tuple(c for c in schema.categorical if c in columns))


def fit_pipeline_graph(graph, frame, *, feature_schema, seed=42, estimator_factories=None,
                       prediction_store: PredictionStore | None = None):
    graph = PipelineGraph.from_dict(graph).validate()
    frame = frame.copy()
    if "target_probability" not in frame and "target_win" in frame:
        frame["target_probability"] = frame["target_win"]
    dates = pd.to_datetime(frame[graph.date_column], errors="raise")
    if dates.isna().any() or frame[["race_id", graph.horse_column]].isna().any().any():
        raise ValueError("Graph rows require exact race/horse/date keys")
    if frame.groupby("race_id")[graph.date_column].nunique().max() != 1:
        raise ValueError("A race cannot cross chronological boundaries")
    if frame.duplicated(["race_id", graph.horse_column]).any():
        raise ValueError("Duplicate graph race/horse rows")
    if graph.fit_budget is not None and graph.fit_budget < 1:
        raise ValueError("Physical fit budget must be positive")
    nodes = {node.node_id: node for node in graph.nodes}
    factories = dict(estimator_factories or {})
    memo = {}
    report = {"physical_fits": 0, "cache_hits": 0, "oof_artifacts": [], "oof_populations": [],
              "component_fits": [], "training_cutoff": str(dates.max()),
              "seed": seed, "schema_version": graph.schema_version,
              "dependency_versions": {"numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__}}
    schema_dict = asdict(feature_schema)
    graph_identity = asdict(graph)

    def component_contract(key):
        node = nodes[key]
        contract = {"kind": node.kind, "parameters": node.parameters, "output": asdict(node.output),
                "fit_scope": node.fit_scope, "parents": [component_contract(p) for p in node.inputs]}
        if node.kind == "race_normalize" and _learned_stage(node, nodes):
            contract["adapter_version"] = "ranking_temperature_forward_oof_v1"
        return contract

    def consume(node, data):
        if graph.fit_budget is not None and report["physical_fits"] >= graph.fit_budget:
            raise RuntimeError("Graph physical-fit budget exhausted")
        report["physical_fits"] += 1
        report["component_fits"].append({"node_id": node.node_id, "training_cutoff": str(pd.to_datetime(data[graph.date_column]).max()),
            "rows": len(data), "fit_scope": node.fit_scope,
            "fit_kind": "ranking_probability_adapter" if node.kind == "race_normalize" else node.kind})

    def build(key, data):
        node = nodes[key]
        fingerprint = frame_fingerprint(data)
        # Identical compatible component nodes share fits, but downstream lineage stays explicit.
        signature = prediction_key({"node": asdict(node) | {"node_id": ""}, "frame": fingerprint,
                                    "schema": schema_dict, "seed": seed,
                                    "factory_id": node.node_id if factories else None})
        if signature in memo:
            return memo[signature]
        parents = tuple(build(p, data) for p in node.inputs)
        fitted = FittedGraphNode(node, parents)
        if node.kind == "feature_view":
            fitted.feature_columns = tuple(node.parameters.get("features", feature_schema.features))
            _schema_subset(feature_schema, fitted.feature_columns)
        elif node.kind == "estimator":
            schema = _schema_subset(feature_schema, parents[0].feature_columns) if parents else feature_schema
            current_market = {"market_probability", "win_odds", "place_odds", "odds", "final_odds", "target_win", "target_probability"}
            if not node.output.market and any(c.lower() in current_market or c.lower().startswith("current_market") for c in schema.features):
                raise ValueError("Fundamental estimator includes a market/odds feature")
            consume(node, data)
            if node.node_id in factories:
                fitted.state = factories[node.node_id]().fit(data)
            else:
                parameters = dict(node.parameters)
                kind = parameters.pop("model_kind", "benter_conditional_logit")
                model_parameters = parameters.pop("model_parameters", parameters.pop("parameters", {}))
                distribution_options = parameters.pop("performance_distribution", None)
                if parameters:
                    raise ValueError(f"Unsupported estimator options: {sorted(parameters)}")
                if node.output.kind == "performance_distribution" and kind != "gaussian_probit":
                    from .probabilistic_adapters import fit_performance_distribution
                    coordinates = {"latent": "latent_strength", "log_mps": "log_speed_mps", "log_seconds": "log_time_seconds"}
                    options = dict(distribution_options or {})
                    options.setdefault("kind", "catboost_uncertainty" if kind == "catboost_uncertainty" else "shared_residual")
                    options.setdefault("coordinate", coordinates[node.output.unit])
                    options.setdefault("date_column", graph.date_column)
                    if options["kind"] == "catboost_uncertainty":
                        options["parameters"] = dict(options.get("parameters", {})) | model_parameters
                    time = pd.to_datetime(data[graph.date_column])
                    unique = np.sort(time.unique())
                    if len(unique) < 2:
                        raise ValueError("Distribution nodes need a disjoint past residual window")
                    boundary = unique[max(1, int(0.8*len(unique)))]
                    fitted.state = fit_performance_distribution(options, data.loc[time < boundary],
                        data.loc[time >= boundary], schema, label_column=node.output.target, seed=seed,
                        model_spec={"kind": kind, "parameters": model_parameters})
                elif kind == "gaussian_probit":
                    from .performance_probit import fit_probit
                    fitted.state = fit_probit(data, schema, model_parameters)
                elif node.output.kind == "win_probability":
                    fitted.state = RaceProbabilityModel(kind=kind, parameters=model_parameters,
                                                       random_state=seed, feature_schema=schema).fit(data)
                elif node.output.kind in {"point_prediction", "ranking_score"}:
                    from .research_models import ResearchRegressor
                    label = node.output.target
                    fitted.state = ResearchRegressor(kind=kind, parameters=model_parameters).fit(data, schema, label)
                else:
                    raise ValueError("Native distributions need an explicitly refittable estimator factory")
        elif _learned_stage(node, nodes):
            indices, outputs = forward_inputs(node.inputs, data)
            learning = data.iloc[indices].copy()
            consume(node, learning)
            if node.kind == "meta_estimator":
                fitted.feature_columns = tuple(f"component_{i}" for i in range(len(outputs)))
                meta = fitted._meta_frame(learning, outputs)
                schema = FeatureSchema(name="graph_meta", numeric=fitted.feature_columns, categorical=())
                fitted.state = RaceProbabilityModel(kind="benter_conditional_logit",
                    parameters=dict(node.parameters.get("model_parameters", {})), feature_schema=schema,
                    random_state=seed).fit(meta)
            elif node.kind == "calibrate":
                fitted.state = TemperatureCalibrator.fit(outputs[0], learning)
            elif node.kind == "race_normalize":
                fitted.state = TemperatureCalibrator.fit(
                    ranking_scores_to_probabilities(outputs[0], learning["race_id"]), learning)
                report["component_fits"][-1]["fitted_temperature"] = fitted.state.temperature
            else:
                column = node.parameters.get("market_column", "market_probability")
                availability = node.parameters.get("availability_column")
                if availability and not learning[availability].fillna(False).all():
                    raise ValueError("Unavailable market quotes in blend training")
                market = _validated_probabilities(learning[column].to_numpy(), learning)
                fitted.state = MarketBlend.fit(outputs[0], market, learning)
        memo[signature] = fitted
        return fitted

    def forward_inputs(parent_ids, data):
        time = pd.to_datetime(data[graph.date_column])
        unique = np.sort(time.unique())
        if len(unique) < graph.min_train_dates + graph.n_splits:
            raise ValueError("Insufficient chronological history for nested learned graph")
        blocks = np.array_split(unique[graph.min_train_dates:], graph.n_splits)
        indices, columns = [], [[] for _ in parent_ids]
        for fold_id, block in enumerate(blocks):
            training = data.loc[time < block[0]]
            positions = np.flatnonzero(time.isin(block).to_numpy())
            scoring = data.iloc[positions]
            row_keys = tuple(zip(scoring["race_id"].astype(str), scoring[graph.horse_column].astype(str)))
            fold_outputs = []
            # Nested learned ancestors need enough earlier dates; exclude initial unsupported folds.
            if any(has_learned_ancestor(p) for p in parent_ids) and len(pd.to_datetime(training[graph.date_column]).unique()) < required_dates(parent_ids):
                continue
            for parent_id in parent_ids:
                component = component_contract(parent_id)
                metadata = {"fit_scope": "forward_oof", "component_id": prediction_key(component),
                    "component_contract": component, "feature_schema": schema_dict, "seed": seed,
                    "protocol": {"n_splits": graph.n_splits, "min_train_dates": graph.min_train_dates,
                        "date_column": graph.date_column, "horse_column": graph.horse_column,
                        "dataset_id": graph.dataset_id, "target_contract_id": graph.target_contract_id,
                        "evaluation_contract_id": graph.evaluation_contract_id},
                    "training_fingerprint": frame_fingerprint(training),
                    "prediction_fingerprint": frame_fingerprint(scoring), "fold_id": fold_id,
                    "training_cutoff": str(pd.to_datetime(training[graph.date_column]).max()),
                    "prediction_start": str(pd.to_datetime(scoring[graph.date_column]).min()),
                    "versions": report["dependency_versions"]}
                cache_key = prediction_key(metadata)
                # Custom factory identity cannot be inferred from a callable; disable persistent reuse.
                artifact = prediction_store.get(cache_key, row_keys) if prediction_store and not factories else None
                if artifact is not None:
                    values = artifact.values
                    report["cache_hits"] += 1
                else:
                    fitted = build(parent_id, training)
                    values = fitted.predict(scoring)
                    if nodes[parent_id].output.kind == "win_probability":
                        values = _validated_probabilities(values, scoring)
                    elif nodes[parent_id].output.kind == "ranking_score":
                        values = np.asarray(values, float)
                        if values.shape != (len(scoring),) or not np.isfinite(values).all():
                            raise ValueError("Invalid forward ranking scores")
                    else:
                        raise ValueError("Learned graph stages require win probabilities or registered ranking scores")
                    artifact = PredictionArtifact(values, row_keys, metadata)
                    if prediction_store and not factories:
                        prediction_store.put(cache_key, artifact)
                report["oof_artifacts"].append({"key": cache_key, "component_id": parent_id,
                    "training_cutoff": metadata["training_cutoff"], "prediction_start": metadata["prediction_start"],
                    "rows": len(row_keys), "row_keys_hash": prediction_key({"row_keys": row_keys}),
                    "training_fingerprint": metadata["training_fingerprint"],
                    "prediction_fingerprint": metadata["prediction_fingerprint"], "fold_id": fold_id})
                fold_outputs.append(values)
            indices.extend(positions.tolist())
            for i, values in enumerate(fold_outputs):
                columns[i].append(values)
        if not indices:
            raise ValueError("No safe forward rows remain for learned graph stage")
        report["oof_populations"].append({"inputs": parent_ids, "rows": len(data),
            "eligible_rows": len(indices), "excluded_rows": len(data)-len(indices),
            "excluded_position_ranges": _position_ranges(sorted(set(range(len(data)))-set(indices))),
            "exclusion_reason": "initial history has no safe earlier fitted ancestor"})
        return np.asarray(indices), [np.concatenate(c) for c in columns]

    def has_learned_ancestor(key):
        return _learned_stage(nodes[key], nodes) or any(has_learned_ancestor(p) for p in nodes[key].inputs)

    def learned_depth(key):
        return int(_learned_stage(nodes[key], nodes)) + max([learned_depth(p) for p in nodes[key].inputs] or [0])

    def required_dates(parent_ids):
        # Every learned level consumes at least n_splits future dates.
        return graph.min_train_dates + graph.n_splits*max(learned_depth(p) for p in parent_ids)

    output = build(graph.output_node_id, frame)
    fundamental = build(graph.fundamental_node_id or graph.output_node_id, frame)
    report["fit_report_id"] = hashlib.sha256(json.dumps(graph_identity, sort_keys=True).encode()).hexdigest()
    joint = build(graph.joint_node_id, frame) if graph.joint_node_id else None
    return FittedPipelineGraph(graph, output, fundamental, report, joint)


def _position_ranges(positions):
    """Compact half-open row-mask spans in the immutable input frame order."""
    ranges = []
    for position in positions:
        if ranges and ranges[-1][1] == position:
            ranges[-1][1] += 1
        else:
            ranges.append([position, position+1])
    return ranges


def fit_graph(graph, train, calibration, feature_schema, seed=42, *, model_spec=None,
              estimator_factories=None, prediction_store=None):
    """Schema3 dispatch. Final refits use train+calibration; scoring stays separate.

    primary_node_id binds outer model.parameters to one declared estimator.
    Learned stages fit recursive forward OOF predictions from these past rows.
    """
    graph = PipelineGraph.from_dict(graph)
    if model_spec is not None:
        if graph.primary_node_id is None:
            raise ValueError("Outer model tuning requires an explicit primary_node_id")
        kind = model_spec.get("kind") if isinstance(model_spec, dict) else model_spec.kind
        parameters = model_spec.get("parameters", {}) if isinstance(model_spec, dict) else model_spec.parameters
        nodes = []
        for node in graph.nodes:
            if node.node_id == graph.primary_node_id:
                options = dict(node.parameters)
                declared_kind = options.get("model_kind", kind)
                if declared_kind != kind:
                    raise ValueError("Outer tuning cannot silently change graph estimator family")
                options["model_kind"] = kind
                options["model_parameters"] = dict(options.get("model_parameters", {})) | dict(parameters or {})
                node = GraphNode(node.node_id, node.kind, node.inputs, options, node.output, node.fit_scope)
            nodes.append(node)
        graph = PipelineGraph(**(graph.__dict__ | {"nodes": tuple(nodes)}))
    if calibration is not None and len(calibration):
        if pd.to_datetime(train[graph.date_column]).max() >= pd.to_datetime(calibration[graph.date_column]).min():
            raise ValueError("Graph calibration window must strictly follow training")
        frame = pd.concat([train, calibration], ignore_index=True)
    else:
        frame = train
    return fit_pipeline_graph(graph, frame, feature_schema=feature_schema, seed=seed,
        estimator_factories=estimator_factories, prediction_store=prediction_store)
