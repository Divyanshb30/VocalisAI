# ADR 0008: Own simulation harness (SimAir) instead of Pipecat's built-in evals

**Status:** accepted

## Context
Pipecat ships a simulation eval framework, but it models an inbound caller persona talking to a bot.

## Decision
Build SimAir: the agent is the caller; SimAir provides IVR trees, hold, adversarial rep personas with hidden policies, scripted events, and planted canaries. Scoring is deterministic where possible, with an LLM judge for tone and persistence (calibration against human labels is pending).

## Alternatives considered
Pipecat evals; paid platforms (Hamming, Coval, Cekura); manual testing.

## Consequences
Covers outbound IVR/hold/DTMF and leak testing; results are reproducible. A CI safety gate on top of it is not built yet.
