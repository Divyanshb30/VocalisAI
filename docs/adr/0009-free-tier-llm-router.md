# ADR 0009: Quota-aware multi-provider router on free tiers

**Status:** accepted

## Context
The project has a $0 LLM budget, and free tiers have daily token and request caps.

## Decision
A LiteLLM-based router spreads load across Cerebras, Groq, Gemini and local Ollama with fallbacks; a fast talker handles turns. A stronger asynchronous planner is planned but not built.

## Alternatives considered
One paid provider; self-hosting a large model.

## Consequences
Zero cost and provider failover. Free tiers have tight caps that differ per model (Cerebras gpt-oss-120b 5 requests/min, qwen-3.8-27b 450 requests/min; Groq 8K tokens/min), so prompts stay small (Flows), the quota ledger is kept per model, and the eval harness paces each talker by its own limits.
