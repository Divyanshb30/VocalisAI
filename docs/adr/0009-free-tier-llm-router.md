# ADR 0009: Quota-aware multi-provider router on free tiers

**Status:** accepted

## Context
The project has a $0 LLM budget, and free tiers have daily token and request caps.

## Decision
A LiteLLM-based router spreads load across Cerebras, Groq, Gemini and local Ollama with fallbacks; a fast talker handles turns, a stronger planner runs asynchronously.

## Alternatives considered
One paid provider; self-hosting a large model.

## Consequences
Zero cost and provider failover; context must stay small (Cerebras free tier caps at 8k), which Flows enforces.
