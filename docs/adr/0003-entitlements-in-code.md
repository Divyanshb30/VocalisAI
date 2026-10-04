# ADR 0003: Compute passenger entitlements in code; the LLM only explains

**Status:** accepted

## Context
Compensation depends on distance bands, delay thresholds, notice periods and exceptions. LLMs are unreliable at this arithmetic.

## Decision
Deterministic per-jurisdiction engines (DGCA, UK261/EU261, GCAA) return entitlements with clause citations. Hybrid retrieval (BM25 + dense + RRF) supplies the regulation text behind each entitlement; the LLM phrases arguments.

## Alternatives considered
Let the LLM reason over retrieved regulation text.

## Consequences
Entitlements are unit-tested against gold cases; every figure the agent cites is traceable to a clause.
