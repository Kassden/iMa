"""Simplex pools fitted exclusively from lineage-checked forward predictions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .prediction_store import frame_fingerprint, prediction_key


SELECTION_MODES = {"development_selected", "nested_selection"}


def validate_selection(mode, cutoff):
    if mode not in SELECTION_MODES:
        raise ValueError("Unknown pool selection mode")
    if mode == "nested_selection" and cutoff is None:
        raise ValueError("nested_selection requires an explicit selection_cutoff")
    if mode == 'nested_selection':
        raise ValueError('nested_selection is not qualified: source-linked component/feature selection evidence is required; use development_selected')
    if cutoff is not None and pd.isna(pd.Timestamp(cutoff)):
        raise ValueError("Invalid selection_cutoff")


def probability_matrix(outputs, frame):
    matrix = np.column_stack(outputs)
    if matrix.shape[0] != len(frame) or not np.isfinite(matrix).all() or np.any(matrix < 0):
        raise ValueError("Invalid pool probabilities")
    totals = pd.DataFrame(matrix).groupby(frame.race_id.to_numpy()).sum().to_numpy()
    if not np.allclose(totals, 1, atol=1e-8, rtol=0):
        raise ValueError("Pool inputs require complete normalized races")
    return matrix


@dataclass
class ForwardSimplexPool:
    weights: np.ndarray
    report: dict

    @classmethod
    def fit(cls, frame, blocks, *, horse_column="horse_id", date_column="date",
            selection_mode="development_selected", selection_cutoff=None,
            outer_training_cutoff):
        validate_selection(selection_mode, selection_cutoff)
        cutoff = pd.Timestamp(outer_training_cutoff)
        selected = pd.Timestamp(selection_cutoff) if selection_cutoff is not None else cutoff
        dates = pd.to_datetime(frame[date_column], errors="raise")
        if dates.isna().any() or pd.isna(cutoff) or dates.max() > cutoff or selected > cutoff:
            raise ValueError("Pool fit/selection exceeds outer training cutoff")
        keys = tuple(zip(frame.race_id.astype(str), frame[horse_column].astype(str)))
        if not keys or len(set(keys)) != len(keys) or frame[["race_id", horse_column]].isna().any().any():
            raise ValueError("Pool requires unique nonmissing runner keys")
        if "field_size" in frame and not np.array_equal(
            frame.groupby("race_id").race_id.transform("size"), frame.field_size
        ):
            raise ValueError("Pool requires complete fields")
        columns, receipts, offset, identities, output_contract = None, [], 0, None, None
        previous_end = None
        for artifacts in blocks:
            if not artifacts:
                raise ValueError("Missing component predictions")
            block_keys = artifacts[0].row_keys
            scoring = frame.iloc[offset:offset+len(block_keys)]
            if block_keys != keys[offset:offset+len(block_keys)] or not block_keys:
                raise ValueError("Missing/reordered pool runner keys")
            start = pd.to_datetime(scoring[date_column]).min().normalize()
            end = pd.to_datetime(scoring[date_column]).max().normalize()
            if previous_end is not None and start <= previous_end:
                raise ValueError("Forward blocks must contain whole disjoint meetings")
            previous_end = end
            if selection_mode == "nested_selection" and selected.normalize() >= start:
                raise ValueError("Selection cutoff must precede every nested prediction meeting")
            block_ids, values = [], []
            common = None
            for artifact in artifacts:
                artifact.validate()
                metadata = artifact.metadata
                if artifact.row_keys != block_keys:
                    raise ValueError("Component runner keys differ")
                required = {"component_id", "component_contract", "training_fingerprint",
                            "prediction_fingerprint", "training_cutoff", "prediction_start",
                            "selection_mode", "selection_cutoff"}
                if not required <= metadata.keys():
                    raise ValueError("Missing forward prediction lineage")
                if metadata["component_id"] != prediction_key(metadata["component_contract"]):
                    raise ValueError("Corrupt component lineage")
                declared_output = metadata["component_contract"].get("output")
                if (not isinstance(declared_output, dict)
                    or declared_output.get("kind") != "win_probability"):
                    raise ValueError("Pool components must declare a win-probability output contract")
                if output_contract is not None and declared_output != output_contract:
                    raise ValueError("Pool components have incompatible target/probability contracts")
                output_contract = declared_output
                if metadata["selection_mode"] != selection_mode or pd.Timestamp(metadata["selection_cutoff"]) != selected:
                    raise ValueError("Forward prediction selection lineage differs from pool contract")
                if (metadata["prediction_fingerprint"] != frame_fingerprint(scoring)
                    or pd.Timestamp(metadata["prediction_start"]).normalize() != start
                    or pd.Timestamp(metadata["training_cutoff"]).normalize() >= start):
                    raise ValueError("Invalid forward meeting/fit lineage")
                shared = (metadata["training_fingerprint"], metadata["training_cutoff"],
                          metadata.get("protocol"), metadata.get("feature_schema"), metadata.get("seed"))
                if common is not None and shared != common:
                    raise ValueError("Components have incompatible forward populations")
                common = shared
                block_ids.append(metadata["component_id"])
                values.append(artifact.values)
            if identities is not None and identities != block_ids:
                raise ValueError("Component identity changed between forward blocks")
            identities = block_ids
            matrix = probability_matrix(values, scoring)
            if columns is None:
                columns = [[] for _ in artifacts]
            for column, values in zip(columns, matrix.T):
                column.extend(values)
            receipts.append({"rows": len(block_keys), "prediction_start": str(start),
                             "training_cutoff": common[1],
                             "artifact_keys": [prediction_key(a.metadata) for a in artifacts]})
            offset += len(block_keys)
        if offset != len(frame) or columns is None:
            raise ValueError("Missing pool forward prediction rows")
        matrix = probability_matrix(columns, frame)
        labels = frame.target_probability.to_numpy(float)
        if not np.isfinite(labels).all() or np.any(labels < 0) or not np.allclose(
            pd.Series(labels).groupby(frame.race_id.to_numpy()).sum(), 1, atol=1e-8, rtol=0
        ):
            raise ValueError("Invalid pool race labels")
        races = frame.race_id.nunique()
        tiny = np.finfo(float).tiny

        def objective(weights):
            return -float(labels @ np.log(np.maximum(matrix @ weights, tiny))) / races

        def gradient(weights):
            return -(matrix.T @ (labels / np.maximum(matrix @ weights, tiny))) / races

        count = matrix.shape[1]
        result = minimize(objective, np.full(count, 1/count), jac=gradient, method="SLSQP",
                          bounds=[(0., 1.)]*count,
                          constraints={"type": "eq", "fun": lambda w: w.sum()-1,
                                       "jac": lambda w: np.ones(count)},
                          options={"ftol": 1e-12, "maxiter": 1000})
        if not result.success or not np.isfinite(result.fun):
            raise ValueError(f"Simplex optimization failed: {result.message}")
        weights = np.clip(result.x, 0, 1)
        weights[weights < 1e-12] = 0
        weights /= weights.sum()
        controls = {"equal": objective(np.full(count, 1/count)),
                    **{f"pure_{i}": objective(w) for i, w in enumerate(np.eye(count))}}
        report = {"selection_mode": selection_mode, "selection_cutoff": str(selected),
                  "fit_label_cutoff": str(dates.max()), "outer_training_cutoff": str(cutoff),
                  "weights": weights.tolist(), "race_log_loss": objective(weights),
                  "controls": controls, "rows": len(frame), "races": races,
                  "components": identities, "forward_blocks": receipts,
                  "optimizer": "scipy.SLSQP", "iterations": int(result.nit)}
        return cls(weights, report)

    def predict(self, outputs, frame):
        matrix = probability_matrix(outputs, frame)
        if matrix.shape[1] != len(self.weights):
            raise ValueError("Pool component count changed")
        return matrix @ self.weights
