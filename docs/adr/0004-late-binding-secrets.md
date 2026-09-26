# ADR 0004: Late-binding secrets and a deterministic output guard

**Status:** accepted

## Context
The agent talks to adversarial or socially engineering reps. Prompt instructions alone cannot guarantee zero leaks.

## Decision
Data is tiered. T1 values are never in the LLM context: the model emits placeholders, a renderer substitutes them only if the passenger allowed it. T2 values (OTP, card, passport) are never stored. A regex/Luhn/canary output guard runs on every utterance and DTMF digit.

## Alternatives considered
Instruction-only safety; LLM guard models (added latency, can be jailbroken).

## Consequences
Leak-freedom holds by construction for values outside the model's context, and is measured with canaries in every eval run.
