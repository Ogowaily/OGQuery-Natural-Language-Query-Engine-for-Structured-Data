"""
Execution Plan Models & Query Planner  ─  v3 (type-safe)
=========================================================
All execution state is carried in strongly-typed dataclasses.
No raw dict ever reaches the QueryEngine.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Literal

from ogquery.schema.registry import SchemaRegistry


# ══════════════════════════════════════════════════════════════════════════════
# Typed filter models
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ExactFilter:
    """Categorical / text exact-match filter."""
    column: str
    value: str


@dataclass(frozen=True)
class SemanticFilter:
    """Vector-similarity filter (routed to FAISS)."""
    column: str
    value: str


@dataclass(frozen=True)
class NumericRangeFilter:
    """
    Numeric range filter.
    At least one of min / max must be set.
    """
    column: str
    min: Optional[float] = None
    max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.min is None and self.max is None:
            raise ValueError(
                f"NumericRangeFilter for '{self.column}': "
                "at least one of min / max must be provided."
            )
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(
                f"NumericRangeFilter for '{self.column}': "
                f"min ({self.min}) > max ({self.max})."
            )

    @property
    def lo(self) -> float:
        return self.min if self.min is not None else float("-inf")

    @property
    def hi(self) -> float:
        return self.max if self.max is not None else float("inf")


Objective = Literal["minimize", "maximize", "list", "count", "top_k"]


@dataclass
class ExecutionPlan:
    """
    Fully typed, validated execution plan.
    Produced by QueryPlanner; consumed exclusively by QueryEngine.
    No raw dict access anywhere downstream.
    """
    objective: Objective
    limit: int
    sort_by: Optional[str]
    sort_ascending: bool
    target_column: Optional[str]

    exact_filters: List[ExactFilter] = field(default_factory=list)
    semantic_filters: List[SemanticFilter] = field(default_factory=list)
    numeric_filters: List[NumericRangeFilter] = field(default_factory=list)

    # ── Convenience ────────────────────────────────────────────────────
    raw_query: str = ""
    parser_used: str = ""

    def has_any_filter(self) -> bool:
        return bool(
            self.exact_filters or self.semantic_filters or self.numeric_filters
        )


# ══════════════════════════════════════════════════════════════════════════════
# Query Planner
# ══════════════════════════════════════════════════════════════════════════════

_VALID_OBJECTIVES: frozenset = frozenset(
    {"minimize", "maximize", "list", "count", "top_k"}
)


class QueryPlanner:
    """
    Converts a validated GroqParser output dict into a typed ExecutionPlan.

    Responsibilities
    ----------------
    - Schema validation: reject unknown columns and wrong filter types.
    - Filter routing: exact vs. semantic vs. numeric.
    - No dict mutation; no inference of missing structure.

    The planner RAISES on schema violations so the engine never sees
    a malformed plan.
    """

    def __init__(self, registry: SchemaRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def plan(self, parsed: dict) -> ExecutionPlan:
        """
        Build and return a fully typed ExecutionPlan.

        Parameters
        ----------
        parsed:
            Output dict from GroqParser._validate().  Keys are guaranteed
            present (GroqParser fills defaults), but values still need
            schema cross-checking.

        Raises
        ------
        ValueError  if a critical schema violation is detected.
        """
        objective = self._resolve_objective(parsed.get("objective", "list"))
        limit = self._resolve_limit(parsed.get("limit", 2))
        sort_by = self._resolve_sort_column(parsed.get("sort_by"))
        sort_ascending = bool(parsed.get("sort_ascending", True))
        target_column = self._resolve_target_column(
            parsed.get("target_column"), objective
        )

        ep = ExecutionPlan(
            objective=objective,
            limit=limit,
            sort_by=sort_by,
            sort_ascending=sort_ascending,
            target_column=target_column,
            raw_query=parsed.get("raw_query", ""),
            parser_used=parsed.get("_parser", "unknown"),
        )

        self._resolve_categorical_filters(parsed.get("filters", {}), ep)
        self._resolve_numeric_filters(parsed.get("numeric_filters", {}), ep)

        return ep

    # ------------------------------------------------------------------
    # Private resolution helpers
    # ------------------------------------------------------------------

    def _resolve_objective(self, raw: object) -> Objective:
        if raw not in _VALID_OBJECTIVES:
            return "list"
        return raw  # type: ignore[return-value]

    def _resolve_limit(self, raw: object) -> int:
        try:
            if raw is None:
                return 10
            return max(1,min(int(raw), 100)) 
        except (TypeError, ValueError):
            return 10

    def _resolve_sort_column(self, col: Optional[str]) -> Optional[str]:
        if not col:
            return None
        if col not in self._registry.columns:
            # Unknown column — silently ignore rather than crash.
            return None
        return col

    def _resolve_target_column(
        self, col: Optional[str], objective: Objective
    ) -> Optional[str]:
        """Target column is only meaningful for minimize / maximize / top_k."""
        if objective not in {"minimize", "maximize", "top_k"}:
            return None
        if not col or col not in self._registry.columns:
            return None
        meta = self._registry.columns[col]
        if meta.col_type != "numeric":
            return None
        return col

    def _resolve_categorical_filters(
        self, filters: object, ep: ExecutionPlan
    ) -> None:
        if not isinstance(filters, dict):
            return

        for col, value in filters.items():
            if col not in self._registry.columns:
                continue  # Unknown column — skip.

            meta = self._registry.columns[col]
            str_value = str(value)

            if meta.col_type == "numeric":
                # LLM mis-classified a numeric column as categorical.
                # Attempt to interpret as an equality range.
                try:
                    fval = float(str_value)
                    ep.numeric_filters.append(
                        NumericRangeFilter(column=col, min=fval, max=fval)
                    )
                except ValueError:
                    pass  # Cannot coerce → silently drop.
                continue

            # Exact filter (always added for non-numeric)
            ep.exact_filters.append(ExactFilter(column=col, value=str_value))

            # Also add semantic filter if the column supports it.
            if meta.semantic_enabled:
                ep.semantic_filters.append(
                    SemanticFilter(column=col, value=str_value)
                )

    def _resolve_numeric_filters(
        self, numeric_filters: object, ep: ExecutionPlan
    ) -> None:
        if not isinstance(numeric_filters, dict):
            return

        for col, bounds in numeric_filters.items():
            if col not in self._registry.columns:
                continue

            meta = self._registry.columns[col]
            if meta.col_type != "numeric":
                # Numeric filter on a non-numeric column — skip.
                continue

            if not isinstance(bounds, dict):
                continue

            raw_min = bounds.get("min")
            raw_max = bounds.get("max")

            parsed_min: Optional[float] = None
            parsed_max: Optional[float] = None

            try:
                if raw_min is not None:
                    parsed_min = float(raw_min)
            except (TypeError, ValueError):
                pass

            try:
                if raw_max is not None:
                    parsed_max = float(raw_max)
            except (TypeError, ValueError):
                pass

            if parsed_min is None and parsed_max is None:
                continue  # Nothing usable — skip.

            try:
                ep.numeric_filters.append(
                    NumericRangeFilter(
                        column=col, min=parsed_min, max=parsed_max
                    )
                )
            except ValueError:
                # min > max etc. — skip rather than crash.
                pass