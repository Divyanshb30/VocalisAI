# ADR 0002: Pipecat with Pipecat Flows for the real-time loop

**Status:** accepted

## Context
We need streaming audio, interruption handling, telephony serializers, IVR navigation and tracing.

## Decision
Pipecat for the pipeline, Pipecat Flows (now part of Pipecat core) for conversation state.

## Alternatives considered
LiveKit Agents (excellent for WebRTC rooms, needs a media server/SIP); managed platforms such as Vapi/Retell (lock-in, cost, little to learn); DIY asyncio (months of edge cases).

## Consequences
Flows keeps each node's prompt small, which matters on small-context free tiers. Built so far: a text-mode Pipecat pipeline with Flows nodes ivr → negotiate → confirm. Pipecat's smart-turn and OpenTelemetry tracing are not wired in yet; they come with the real-time audio pipeline.
