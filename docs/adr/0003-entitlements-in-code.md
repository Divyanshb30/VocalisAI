# ADR 0003: Compute passenger entitlements in code; the LLM only explains

**Status:** accepted

## Context
Compensation depends on distance bands, delay thresholds, notice periods and exceptions. LLMs are unreliable at this arithmetic.

## Decision
Deterministic per-jurisdiction engines (DGCA, UK261/EU261, GCAA) return entitlements with clause citations. Retrieval supplies airline policy text; the LLM phrases arguments.

## Alternatives considered
Let the LLM reason over retrieved regulation text.

## Consequences
Entitlements are unit-tested against gold cases; every figure the agent cites is traceable to a clause.
