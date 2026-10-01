"""Chronological, outer-fold-fitted adjusted-performance history."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .feature_discovery_specs import content_id


COVARIATES = ("distance", "actual_weight", "race_class", "horse_age", "draw")


def covariates(frame):
    return frame.reindex(columns=COVARIATES).apply(pd.to_numeric, errors="coerce")


def records(frame, residuals):
    dates = pd.to_datetime(frame.date).dt.normalize() + pd.Timedelta(days=1)
    if "observed_at" in frame:
        dates = pd.concat([dates, pd.to_datetime(frame.observed_at)], axis=1).max(axis=1)
    return pd.DataFrame({"race_id": frame.race_id.astype(str), "horse_no": frame.horse_no,
                         "horse_id": frame.horse_id.astype(str), "available_at": dates,
                         "distance_band": (pd.to_numeric(frame.distance) / 400).round(),
                         "residual": residuals}).reset_index(drop=True)


@dataclass
class AdjustedSpeedHistory:
    model: object
    history: pd.DataFrame
    shrinkage: float
    columns: tuple[str, str]
    report: dict

    @classmethod
    def fit(cls, train, spec, cutoff=None):
        if "horse_id" not in train or train.horse_id.isna().any():
            raise ValueError("Adjusted speed requires complete historical horse identities")
        dates = pd.to_datetime(train.date).dt.normalize()
        speed = pd.to_numeric(train.distance) / pd.to_numeric(train.finish_seconds).where(lambda x: x > 0)
        valid = speed.notna() & np.isfinite(speed)
        availability = records(train, speed.to_numpy()).available_at.to_numpy()
        residuals = np.full(len(train), np.nan)
        unique = sorted(dates.unique())
        # Expanding chronological blocks provide OOF residuals without fitting future results.
        blocks = np.array_split(np.array(unique), min(12, len(unique)))
        fits = []
        for block in blocks:
            past = (availability < block[0]) & valid
            held = dates.isin(block) & valid
            if past.sum() < 30 or not held.any():
                continue
            model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=10))
            model.fit(covariates(train.loc[past]), speed.loc[past])
            residuals[np.flatnonzero(held)] = speed.loc[held] - model.predict(covariates(train.loc[held]))
            fits.append({"fit_through": str(dates.loc[past].max()), "predict_from": str(block[0]), "fit_rows": int(past.sum())})
        if valid.sum() < 30 or not fits:
            raise ValueError("Insufficient chronological history for adjusted-speed OOF features")
        model = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), Ridge(alpha=10))
        final_valid = valid & (availability < np.datetime64(cutoff if cutoff is not None else dates.max() + pd.Timedelta(days=1)))
        if final_valid.sum() < 30:
            raise ValueError("Insufficient available auxiliary history at fold cutoff")
        model.fit(covariates(train.loc[final_valid]), speed.loc[final_valid])
        columns = tuple("dfs_" + content_id({"family": "adjusted_speed", "distance_conditioned": conditioned, "shrinkage": spec.residual_shrinkage}) for conditioned in (False, True))
        report = {"family": "adjusted_speed", "auxiliary": "Ridge(alpha=10)", "covariates": list(COVARIATES), "oof_blocks": fits, "fit_races_hash": content_id(sorted(train.race_id.astype(str).unique())), "columns": list(columns), "support_shrinkage": spec.residual_shrinkage}
        return cls(model, records(train, residuals), spec.residual_shrinkage, columns, report)

    def transform(self, frame):
        output = frame.copy()
        speed = pd.to_numeric(frame.distance) / pd.to_numeric(frame.finish_seconds).where(lambda x: x > 0) if "finish_seconds" in frame else pd.Series(np.nan, index=frame.index)
        current = records(frame, speed.to_numpy() - self.model.predict(covariates(frame)))
        history = pd.concat([self.history, current]).drop_duplicates(["race_id", "horse_no"], keep="first")
        cutoffs = pd.to_datetime(frame.date).dt.normalize().to_numpy()
        for conditioned, column in zip((False, True), self.columns):
            keys = ["horse_id", "distance_band"] if conditioned else ["horse_id"]
            values = np.full(len(frame), np.nan)
            groups = history.dropna(subset=["residual"]).groupby(keys, sort=False)
            lookup = {key if isinstance(key, tuple) else (key,): group.sort_values("available_at", kind="stable") for key, group in groups}
            for key, positions in current.groupby(keys, sort=False).indices.items():
                key = key if isinstance(key, tuple) else (key,)
                past = lookup.get(key)
                if past is None:
                    continue
                positions = np.asarray(positions)
                counts = np.searchsorted(past.available_at.to_numpy(), cutoffs[positions], side="left")
                sums = np.concatenate([[0], past.residual.cumsum().to_numpy()])
                values[positions] = sums[counts] / np.maximum(counts + self.shrinkage, 1)
            output[column] = values
        return output
