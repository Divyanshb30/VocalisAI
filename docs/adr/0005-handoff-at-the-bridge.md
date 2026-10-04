# ADR 0005: Human handoff by muting and bridging legs

**Status:** accepted, not built yet. Today the simulated call loop routes the step to a simulated passenger and the agent's context only gets a summary note.

## Context
OTPs, payments and identity checks must be done by the passenger, live, without dropping the call.

## Decision
A CallBridge models the call as legs (agent, remote, owner). Handoff mutes the agent leg and bridges the owner leg; the agent keeps listening and can take the call back. On Twilio this maps to call updates.

## Alternatives considered
Conference calls (extra paid legs); cold transfer (agent loses the call).

## Consequences
Handoff and hand-back are the same operation in simulation and on real telephony.
