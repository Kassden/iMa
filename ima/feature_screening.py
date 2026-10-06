"""Training-only discovery selection using established sklearn-compatible tools."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures, SmartCorrelatedSelection
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import ndcg_score

from .feature_discovery_specs import DiscoverySpec, content_id


class RaceScreeningEstimator(RegressorMixin,BaseEstimator):
    def __init__(self, target_kind="win_probability"):
        self.target_kind=target_kind

    def fit(self,x,y):
        classifier=self.target_kind in {"win_probability","placing_top_k"}
        estimator=LogisticRegression(C=.1,max_iter=500) if classifier else Ridge(alpha=1)
        self.model_=make_pipeline(StandardScaler(),estimator).fit(x,np.asarray(y)[:,0])
        return self

    def predict(self,x):
        if self.target_kind in {"win_probability","placing_top_k"}:
            return self.model_.predict_proba(x)[:,1]
        return self.model_.predict(x)

    def score(self,x,y):
        labels,codes=np.asarray(y)[:,0],np.asarray(y)[:,1]
        prediction=self.predict(x)
        scores=[]
        for code in np.unique(codes):
            mask=codes==code
            truth,p=labels[mask],prediction[mask]
            if self.target_kind=="win_probability":
                p=np.clip(p,1e-12,None);p=p/p.sum()
                scores.append(float(truth@np.log(p)))
            elif self.target_kind=="ranking_strength":
                scores.append(float(ndcg_score([truth],[p],k=3)))
            elif self.target_kind=="placing_top_k":
                scores.append(-float(np.mean((truth-p)**2)))
            else:
                scores.append(-float(np.mean(np.abs(truth-p))))
        return float(np.mean(scores))


def chronological_inner_splits(train):
    dates=pd.to_datetime(train.date).dt.normalize()
    ordered=sorted(dates.unique())
    if len(ordered)<8:
        return []
    edges=np.linspace(len(ordered)//2,len(ordered),4,dtype=int)
    result=[]
    for start,end in zip(edges[:-1],edges[1:]):
        left=np.flatnonzero(dates<ordered[start])
        right=np.flatnonzero(dates.isin(ordered[start:end]))
        if len(left) and len(right):result.append((left,right))
    return result


@dataclass
class DiscoverySelection:
    columns: tuple[str, ...]
    report: dict
    fitted_selectors: tuple = ()

    @classmethod
    def fit(cls, train: pd.DataFrame, spec: DiscoverySpec, target_kind: str, label: str):
        generated = sorted(c for c in train if c.startswith("dfs_"))
        coverage = train[generated].notna().mean()
        usable = [c for c in generated if coverage[c] >= 1 - spec.missingness_limit and train[c].nunique(dropna=True) > 1]
        dropped = {c: "missing_or_constant" for c in generated if c not in usable}
        fitted = []
        redundancy_groups = []
        near_constant = [c for c in usable if train[c].value_counts(normalize=True,dropna=False).iloc[0]>.995]
        values = train[usable].replace([np.inf, -np.inf], np.nan)
        for selector in (DropConstantFeatures(tol=1, missing_values="ignore"), DropDuplicateFeatures(missing_values="ignore"), SmartCorrelatedSelection(method="spearman", threshold=spec.correlation_threshold, selection_method="missing_values", missing_values="ignore")):
            if values.shape[1] < 2:
                break
            before = set(values)
            values = selector.fit_transform(values)
            fitted.append(selector)
            redundancy_groups.extend([sorted(group) for group in getattr(selector,"correlated_feature_sets_",[])])
            dropped.update({c: type(selector).__name__ for c in before - set(values)})
        ranking = [(c, 0.0) for c in values]
        if spec.selection != "quality" and len(values) and train[label].nunique() > 1:
            x = values.fillna(values.median()).fillna(0)
            method = mutual_info_classif if target_kind in {"win_probability", "placing_top_k"} else mutual_info_regression
            scores = method(x, train[label].to_numpy(), random_state=spec.seed)
            ranking = sorted(zip(values.columns, map(float, scores)), key=lambda item: (-item[1], item[0]))
        v2 = spec.schema_version == 2
        requested = spec.max_selected
        count = len(ranking) if requested is None else min(requested, len(ranking))
        columns = tuple(c for c, _ in ranking[:count])
        inner=chronological_inner_splits(train)
        selector_note=None
        if spec.selection in {"embedded","sequential"} and len(ranking)>1 and train[label].nunique()>1:
            limit = getattr(spec, "selection_shortlist", None) if v2 else (12 if spec.selection=="sequential" else 32)
            shortlist=[c for c,_ in ranking[:limit]]
            x=values[shortlist].fillna(0).to_numpy()
            labels=train[label].to_numpy(dtype=float)
            codes=pd.factorize(train.race_id)[0]
            grouped_y=np.column_stack([labels,codes])
            estimator=RaceScreeningEstimator(target_kind)
            if spec.selection=="sequential":
                if not inner:
                    raise ValueError("Sequential selection requires at least eight training dates; no random-CV fallback")
                desired = count if v2 else min(spec.max_selected,max(1,len(shortlist)//2))
                fit_count = len(inner) * sum(range(len(shortlist)-desired+1, len(shortlist)+1))
                budget = getattr(spec, "selection_fit_budget", None)
                if desired == len(shortlist):
                    columns = tuple(shortlist)
                    selector_note = "All eligible columns requested; no subset search needed"
                    selector = None
                elif budget is not None and fit_count > budget:
                    raise SelectionDeferred({"reason": "selection_fit_budget", "estimated_fits": fit_count, "budget": budget, "requested": desired, "eligible": len(shortlist)})
                else:
                    selector=SequentialFeatureSelector(estimator,n_features_to_select=desired,cv=inner,n_jobs=1)
                    selector.fit(x,grouped_y)
            else:
                estimator.fit(x,grouped_y)
                selector=SelectFromModel(estimator.model_[-1],prefit=True,max_features=count,threshold=-np.inf if v2 else None)
            if selector is not None:
                columns=tuple(c for c,keep in zip(shortlist,selector.get_support()) if keep)
                selector_note="Race-weighted chronological proxy selection; final target model must confirm"
                fitted.append(selector)
        fit_hash = content_id(sorted(train.race_id.astype(str).unique()))
        report = {"selector_id": content_id({"spec": spec.model_dump(mode="json"), "fit_races": fit_hash, "columns": columns, "target": target_kind}), "fit_races_hash": fit_hash, "target_kind": target_kind, "selected": columns, "ranking": ranking, "rejected": dropped, "coverage": coverage.to_dict(), "method": spec.selection, "caveat": "Univariate MI is screening evidence, not race-level statistical significance"}
        report.update(inner_protocol_id=content_id([(train.iloc[a].race_id.unique().tolist(),train.iloc[b].race_id.unique().tolist()) for a,b in inner]),inner_fold_count=len(inner),selector_note=selector_note)
        report.update(near_constant_flags=near_constant,redundancy_groups=redundancy_groups,selection_exposure="Outer training labels used for screening; inner proxy scores are selection diagnostics, not independent confirmation")
        if v2:
            report.update(requested_budget=requested, eligible_count=len(ranking), selected_count=len(columns), shortlist_limit=getattr(spec,"selection_shortlist",None), estimated_proxy_fits=selection_workload(spec,len(ranking),len(inner))["estimated_fits"])
        return cls(columns, report,tuple(fitted))


class SelectionDeferred(ValueError):
    def __init__(self, report):
        self.report = report
        super().__init__(str(report))


def selection_workload(spec, eligible_count: int, inner_folds: int) -> dict:
    shortlist = getattr(spec, "selection_shortlist", None)
    n = min(eligible_count, shortlist) if shortlist is not None else eligible_count
    k = n if spec.max_selected is None else min(n, spec.max_selected)
    fits = inner_folds * sum(range(n-k+1,n+1)) if spec.selection == "sequential" and k < n else int(spec.selection == "embedded" and n > 1)
    return {"method":spec.selection,"eligible":eligible_count,"shortlist":n,"requested":spec.max_selected,"selected_budget":k,"estimated_fits":fits}
