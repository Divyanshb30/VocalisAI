# ADR 0006: LangGraph for the case workflow, not the audio loop

**Status:** accepted

## Context
A case spans extraction, rights assessment, human approval, a call that may last an hour, and a report.

## Decision
LangGraph with checkpoints and interrupts orchestrates the case; the call itself runs in Pipecat.

## Alternatives considered
LangGraph inside the audio loop (adds latency, not frame-oriented); CrewAI (less control); Temporal (heavy).

## Consequences
Human approvals are first-class interrupts. Checkpoints are in memory today (`InMemorySaver`); durable Postgres checkpoints are needed before a case can survive a restart.
