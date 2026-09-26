# ADR 0002: Pipecat with Pipecat Flows for the real-time loop

**Status:** accepted

## Context
We need streaming audio, interruption handling, telephony serializers, IVR navigation and tracing.

## Decision
Pipecat for the pipeline, Pipecat Flows (now part of Pipecat core) for conversation state.

## Alternatives considered
LiveKit Agents (excellent for WebRTC rooms, needs a media server/SIP); managed platforms such as Vapi/Retell (lock-in, cost, little to learn); DIY asyncio (months of edge cases).

## Consequences
Built-in IVR navigator, voicemail detector, smart-turn, OpenTelemetry spans. Flows keeps each node's prompt small, which matters on 8k-context free tiers.
