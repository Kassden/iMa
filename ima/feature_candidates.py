"""Deterministic, unfitted hypotheses from explicitly eligible numeric inputs."""
from itertools import combinations

from .feature_definitions import FeatureDefinition, validate_definition
from .feature_expressions import NumericExpression, unit_dimensions, unit_scale
from .feature_lifecycle import _numeric_metadata, deterministic_candidate_definitions


def candidate_definitions(catalog, *, exclude_ids=(), limit=32) -> list[FeatureDefinition]:
    """Return the next bounded batch; exclusions never impose a lifetime cap.

    Named domain hypotheses precede lexical family/unit/input contrasts. Units
    are not converted: compatible dimensions must also have identical scales.
    """
    if type(limit) is not int or limit <= 0:
        raise ValueError("Candidate limit must be a positive integer")
    metadata = {
        name: meta for name, meta in _numeric_metadata(catalog).items()
        if catalog[name].get("nonmissing") != 0
        and catalog[name].get("distinct") not in (0, 1)
    }
    seen = set(exclude_ids)
    seen.update(row.get("definition_id") for row in catalog.values() if isinstance(row, dict))
    proposals = []

    def accept(definition):
        if definition.name in catalog:
            return
        try:
            validate_definition(definition, metadata)
        except ValueError:
            return
        identifier = definition.content_id()
        if identifier not in seen:
            seen.add(identifier)
            proposals.append(definition)

    def ref(name):
        return NumericExpression(op="ref", ref=name)

    def generated():
        yield from deterministic_candidate_definitions(catalog, limit=8)
        groups = {}
        for name, meta in metadata.items():
            signature = (meta.source_family, tuple(sorted(unit_dimensions(meta.unit).items())), unit_scale(meta.unit))
            groups.setdefault(signature, []).append(name)
        for signature in sorted(groups):
            for left, right in combinations(groups[signature], 2):
                for op, name, unit, hypothesis in (
                    ("divide", f"relative_{left}_to_{right}", "1",
                     "Relative measurements within one source family may reveal pre-race imbalance."),
                    ("subtract", f"difference_{left}_minus_{right}", metadata[left].unit,
                     "A difference within one source family may capture pre-race deviation from its reference."),
                ):
                    yield FeatureDefinition(
                        name=name, expression_ast=NumericExpression(op=op, args=(ref(left), ref(right))),
                        input_refs=(left, right), unit=unit, source_families=(signature[0],),
                        producer="deterministic_feature_candidates", hypothesis=hypothesis,
                    )
        for name, meta in metadata.items():
            if (unit_dimensions(meta.unit) or unit_scale(meta.unit) != 1
                    or not name.endswith(("_starts", "_count", "_support", "_observations"))):
                continue
            shifted = NumericExpression(op="add", args=(
                NumericExpression(op="abs", args=(ref(name),)),
                NumericExpression(op="constant", value=1),
            ))
            yield FeatureDefinition(
                name="log_abs_support_" + name, expression_ast=NumericExpression(op="log", args=(shifted,)),
                input_refs=(name,), unit="1", source_families=(meta.source_family,),
                producer="deterministic_feature_candidates",
                hypothesis="Diminishing support magnitude may encode experience without a fitted transform.",
            )

    for definition in generated():
        accept(definition)
        if len(proposals) == limit:
            break
    return proposals
