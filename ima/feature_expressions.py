"""Typed numeric expressions; explicit dependencies replace executable planner code."""
from __future__ import annotations

import math
import ast
from functools import lru_cache
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator


class ExpressionError(ValueError):
    def __init__(self, reason: str, detail: str):
        self.report = {"reason": reason, "detail": detail}
        super().__init__(f"{reason}: {detail}")


class NumericExpression(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    op: Literal["ref", "constant", "add", "subtract", "multiply", "divide", "negate", "abs", "log", "sqrt", "minimum", "maximum", "power"]
    args: tuple[NumericExpression, ...] = ()
    ref: str | None = None
    value: float | None = None

    @model_validator(mode="after")
    def arity(self):
        n = {"ref":0,"constant":0,"negate":1,"abs":1,"log":1,"sqrt":1}.get(self.op, 2)
        if len(self.args) != n:
            raise ValueError(f"{self.op} requires {n} arguments")
        if self.op == "ref":
            if not self.ref or not self.ref.isidentifier() or self.value is not None:
                raise ValueError("Invalid numeric symbol")
        elif self.op == "constant":
            if self.value is None or not math.isfinite(self.value) or self.ref is not None:
                raise ValueError("Constants must be finite")
        elif self.ref is not None or self.value is not None:
            raise ValueError("Only leaves carry ref/value")
        return self


class InputMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    unit: str = "1"
    dtype: Literal["float32", "float64", "int32", "int64"] = "float64"
    temporal_scope: Literal["pre_race", "historical", "current_outcome", "future"] = "pre_race"
    target_tainted: bool = False
    available_at_column: str | None = None
    source_family: str = "validated_dataset"


_UNITS = {"1":{}, "m":{"length":1}, "km":{"length":1}, "s":{"time":1}, "sec":{"time":1}, "seconds":{"time":1}, "ms":{"time":1}, "day":{"time":1}, "days":{"time":1}, "kg":{"mass":1}, "g":{"mass":1}, "lb":{"mass":1}, "m/s":{"length":1,"time":-1}, "mps":{"length":1,"time":-1}, "lengths":{"beaten_lengths":1}, "rating":{"rating":1}}


@lru_cache(maxsize=256)
def _unit_signature(unit):
    scales = {"km":1000.,"ms":.001,"day":86400.,"days":86400.,"g":.001,"lb":.45359237}
    if unit in _UNITS:
        return tuple(sorted(_UNITS[unit].items())),scales.get(unit,1.)
    try:
        root = ast.parse(unit.replace("^","**"),mode="eval").body
    except SyntaxError as exc:
        raise ExpressionError("unit_syntax",unit) from exc
    if len(list(ast.walk(root))) > 64:
        raise ExpressionError("unit_complexity",unit)
    def visit(node):
        if isinstance(node,ast.Name):
            return dict(_UNITS.get(node.id,{node.id:1})),scales.get(node.id,1.)
        if isinstance(node,ast.Constant) and node.value == 1:
            return {},1.
        if isinstance(node,ast.BinOp) and isinstance(node.op,(ast.Mult,ast.Div)):
            a,scale = visit(node.left)
            b,right_scale = visit(node.right)
            sign = -1 if isinstance(node.op,ast.Div) else 1
            for key,value in b.items():
                a[key] = a.get(key,0)+sign*value
            return {k:v for k,v in a.items() if v},scale/right_scale if sign == -1 else scale*right_scale
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow):
            exponent = node.right
            sign = 1
            if isinstance(exponent,ast.UnaryOp) and isinstance(exponent.op,ast.USub):
                sign,exponent = -1,exponent.operand
            if not isinstance(exponent,ast.Constant) or type(exponent.value) not in {int,float} or not math.isfinite(exponent.value) or abs(exponent.value)>8:
                raise ExpressionError("unit_exponent",unit)
            a,scale = visit(node.left)
            return {k:v*sign*exponent.value for k,v in a.items() if v*exponent.value},scale**(sign*exponent.value)
        raise ExpressionError("unit_syntax",unit)
    dimensions,scale = visit(root)
    return tuple(sorted(dimensions.items())),scale


def unit_dimensions(unit: str) -> dict:
    # Equal dimensions do not imply equal scales: km and m cannot be added directly.
    return dict(_unit_signature(unit)[0])


def unit_scale(unit: str) -> float:
    return _unit_signature(unit)[1]


def validate_expression(expression, metadata, *, max_nodes=256, max_depth=24):
    metadata = {k: v if isinstance(v,InputMetadata) else InputMetadata.model_validate(v) for k,v in metadata.items()}
    dependencies = set()
    count = 0

    def visit(node, depth):
        nonlocal count
        count += 1
        if count > max_nodes or depth > max_depth:
            raise ExpressionError("complexity", "Explicit expression resource budget exceeded")
        if node.op == "ref":
            if node.ref not in metadata:
                raise ExpressionError("unknown_dependency", node.ref)
            meta = metadata[node.ref]
            if meta.target_tainted or meta.temporal_scope in {"current_outcome","future"}:
                raise ExpressionError("target_or_temporal_taint", node.ref)
            if meta.temporal_scope == "historical" and not meta.available_at_column:
                raise ExpressionError("unverified_availability", node.ref)
            dependencies.add(node.ref)
            return unit_dimensions(meta.unit), unit_scale(meta.unit)
        if node.op == "constant":
            return {}, 1.
        children = [visit(arg,depth+1) for arg in node.args]
        a, au = children[0]
        if node.op in {"negate","abs"}:
            return a, au
        if node.op in {"log","sqrt"}:
            if node.op == "log" and (a or au != 1):
                raise ExpressionError("unit_mismatch", "log requires dimensionless input")
            return ({k:v/2 for k,v in a.items()}, au**.5) if node.op == "sqrt" else ({},1.)
        b, bu = children[1]
        if node.op in {"add","subtract","minimum","maximum"}:
            if a != b or au != bu:
                raise ExpressionError("unit_mismatch", "Additive inputs require identical units/scales")
            return a, au
        if node.op == "power":
            exponent = node.args[1]
            if exponent.op != "constant" or abs(exponent.value) > 4:
                raise ExpressionError("numeric_domain", "Power requires constant exponent in [-4,4]")
            return {k:v*exponent.value for k,v in a.items() if v*exponent.value}, au**exponent.value
        result = dict(a)
        for k,v in b.items():
            result[k] = result.get(k,0) + v * (-1 if node.op == "divide" else 1)
        return {k:v for k,v in result.items() if v}, au/bu if node.op == "divide" else au*bu

    dimensions, scale = visit(expression,1)
    return {"dependencies":sorted(dependencies),"dimensions":dimensions,"unit_scale":scale,"nodes":count}


def evaluate_expression(expression: NumericExpression, bindings: dict, length: int) -> np.ndarray:
    def evaluate(node):
        if node.op == "ref":
            return np.asarray(bindings[node.ref],dtype=float)
        if node.op == "constant":
            return np.full(length,node.value,dtype=float)
        values = [evaluate(arg) for arg in node.args]
        unary = {"negate":np.negative,"abs":np.abs,"log":np.log,"sqrt":np.sqrt}
        binary = {"add":np.add,"subtract":np.subtract,"multiply":np.multiply,"divide":np.divide,"minimum":np.minimum,"maximum":np.maximum,"power":np.power}
        with np.errstate(all="ignore"):
            result = unary[node.op](*values) if node.op in unary else binary[node.op](*values)
        return np.where(np.isfinite(result),result,np.nan)
    result = evaluate(expression)
    if result.shape != (length,):
        raise ExpressionError("shape_mismatch", str(result.shape))
    return np.where(np.isfinite(result), result, np.nan)
