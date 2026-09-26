# ADR 0001: Cascaded STT → LLM → TTS instead of speech-to-speech

**Status:** accepted

## Context
The agent speaks to untrusted third parties and must never leak sensitive data or make unauthorised commitments.

## Decision
Use a cascaded pipeline: streaming STT, a text LLM, streaming TTS.

## Alternatives considered
Speech-to-speech models (OpenAI Realtime, Gemini Live) give lower latency and better prosody, but produce audio directly, so there is no text checkpoint to filter before the rep hears it.

## Consequences
Every utterance passes the output guard before TTS; each stage is independently swappable and measurable. Latency must be won back with streaming, smart-turn and fast inference.
