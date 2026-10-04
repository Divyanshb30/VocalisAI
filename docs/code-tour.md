# Code tour

Follow one call through the code, in the order things happen. Every step can be run from the CLI.

## 1. A case comes in

| What | Where |
|---|---|
| Photo → fields (Gemini structured output, retries on 503/429) | `src/vocalis/docintel/extract.py` → `extract_with_gemini` |
| Boarding-pass barcode (IATA BCBP) decode / encode | `src/vocalis/docintel/bcbp.py` |
| Barcode wins on conflict; sensitive values redacted | `extract.py` → `reconcile`, `redact` |
| Fields → `Case` | `extract.py` → `to_case`; flat fields: `src/vocalis/cases.py` → `case_from_fields` |
| Domain models (`Case`, `Mandate`, data tiers) | `src/vocalis/core/models.py` |

```bash
uv run python -m evals.docbench.synth --n 20      # synthetic boarding passes + notices with labels
uv run python -m evals.docbench.run               # field accuracy, vision vs barcode-checked
```

## 2. What is the passenger owed?

Entitlements are computed in code, never by the LLM (ADR 0003).

| Regime | File |
|---|---|
| UK261 / EU261 (distance bands, notice rules, Sturgeon 3h) | `src/vocalis/rights/eu261.py` |
| India DGCA CAR S3 M IV (block-time bands, denied boarding %) | `src/vocalis/rights/dgca.py` |
| UAE GCAA + Montreal Convention | `src/vocalis/rights/uae.py` |
| Entry point | `src/vocalis/rights/engine.py` → `assess(case)` |
| Regulation retrieval (BM25 + bge-small dense + RRF, citations per article paragraph); corpus in `data/policy/` | `src/vocalis/rights/policy.py`, MCP tool `search_regulations` |

```bash
uv run vocalis rights --airline 6E --flight 2135 --origin DEL --destination BOM \
  --departure 2026-10-12T09:30 --arrival 2026-10-12T11:40 --notice-hours 10 --no-extraordinary \
  --currency INR --fare 6800 --base-fare 5200 --fuel 900
```

## 3. The case workflow (LangGraph)

`src/vocalis/orchestrator/graph.py`: read document → assess rights → draft mandate → **interrupt** for passenger approval → place call → report. The default mandate never accepts a voucher (`cases.py` → `default_mandate`).

```bash
uv run vocalis case --scenario-case in_6e_cancel_short_notice__stonewaller --offline
```

## 4. The call (Pipecat + Flows)

| What | Where |
|---|---|
| What the agent knows (facts, entitlements, mandate), rendered small | `src/vocalis/agent/briefing.py` |
| System and node prompts | `src/vocalis/agent/prompts.py` |
| Pipeline, Flows nodes (`ivr` → `negotiate` → `confirm`), tools | `src/vocalis/agent/session.py` |
| Tools: `press_keys`, `evaluate_offer`, `record_resolution`, `request_handoff`, `end_call` | `session.py` → `_fn_*` |
| Talker LLM with provider failover | `src/vocalis/agent/llm_services.py` |
| Free-tier router + daily quota ledger (rep, judge) | `src/vocalis/llm/router.py` |

Pipeline order: `user aggregator → LLM → OutputGuardProcessor → LineSink → assistant aggregator`.

## 5. Guardrails

| What | Where |
|---|---|
| Late-binding secrets: model writes `⟦phone⟧`, renderer fills it only if allowed | `src/vocalis/guards/vault.py` |
| Sentence-level guard before speech; placeholder text to context, rendered text to voice | `src/vocalis/agent/processors.py` → `OutputGuardProcessor` |
| Deterministic output guard (Luhn, unapproved numbers, passport, e-mail, secrets) | `src/vocalis/guards/output_guard.py` |
| Spoken digits ("double seven", "four four one seven") normalised before checks | `src/vocalis/guards/normalize.py` |
| Input guard on the rep (OTP, payment, identity, injection, "are you a robot") | `src/vocalis/guards/input_guard.py` |
| Canaries planted per run | `src/vocalis/guards/canaries.py` |

## 6. The phone line and the airline (SimAir)

| What | Where |
|---|---|
| DTMF tones + Goertzel detection | `src/vocalis/telephony/dtmf.py` |
| Phone-line simulation (8 kHz μ-law, band-pass, noise, loss) | `src/vocalis/telephony/phoneline.py` |
| Audio leg: far end voiced → phone line → streaming STT (VAD + Finalize), agent's first TTS byte timed, DTMF over the line | `src/vocalis/telephony/voicelink.py` |
| IVR / hold / human / voicemail detection (transcript heuristics) | `src/vocalis/telephony/callstate.py` |
| IVR menu trees | `src/vocalis/simair/ivr.py` |
| Rep: persona, hidden policy, scripted attacks, JSON ground truth | `src/vocalis/simair/rep.py` |
| The call loop: IVR → hold → pickup → disclosure → negotiation → handoffs | `src/vocalis/simair/call.py` → `CallSimulation.run` |
| Offline mode (scripted models, no keys) | `src/vocalis/simair/offline.py` |

```bash
uv run vocalis call uk_ba_cancel_2_days__voucher_pusher --offline   # no keys
uv run vocalis call uk_ba_cancel_2_days__voucher_pusher             # real models
```

## 7. Evaluation

| What | Where |
|---|---|
| Scenario catalog → YAML (6 cases × 8 personas, 25 scenarios) | `evals/scenarios/catalog.py` |
| Deterministic scoring (success, leaks, handoff P/R/F1, disclosure) | `src/vocalis/simair/scoring.py` |
| LLM judge (tone, persistence, invented facts) | `evals/judge.py` |
| Batch runner (`vocalis`, `baseline` secrets-in-prompt ablation, `vocalis_voice` audio loopback) | `evals/run.py` |
| Aggregation with Wilson CIs → README table | `evals/report.py` |

```bash
uv run python -m evals.run --config vocalis --seeds 0 --concurrency 1
uv run python -m evals.run --config baseline --only social_engineer prompt_injector --seeds 0
uv run python -m evals.run --config vocalis_voice --seeds 0 --concurrency 1   # audio loopback (Deepgram)
uv run python -m evals.retrieval                                              # BM25 vs dense vs hybrid on evals/retrieval_gold.yaml
uv run python -m evals.report
```

## 8. MCP servers

`src/vocalis/mcp_servers/rights_server.py` and `calls_server.py` (FastMCP). To use them from a desktop MCP client, add:

```json
{
  "mcpServers": {
    "vocalis-rights": {"command": "uv", "args": ["--directory", "D:/college/PROJECTS/VocalisAI", "run", "vocalis", "mcp", "rights"]},
    "vocalis-calls": {"command": "uv", "args": ["--directory", "D:/college/PROJECTS/VocalisAI", "run", "vocalis", "mcp", "calls"]}
  }
}
```

## 9. Tests

`tests/`, all offline, including two full calls through the real Pipecat pipeline and MCP servers exercised through an MCP client. CI runs lint, mypy and tests on every push (`.github/workflows/ci.yml`).

## 10. Web app and hosting

| What | Where |
|---|---|
| API: scenarios, cases, document upload, metrics, live call over SSE, Deepgram TTS, per-IP live-call limits | `src/vocalis/web/server.py` |
| Deepgram Aura-2 voices per speaker, cached on disk | `src/vocalis/web/tts.py` |
| The page: Live / Offline / Recorded modes, voices, DTMF tones, hold music, live pipeline diagram + explainer feed | `web/index.html` |
| Recorded-call bundle for the always-on demo (+ pre-voiced showcase calls) | `evals/export_demo.py` → `web/demo/` |
| Always-on demo (static, GitHub Pages) | `.github/workflows/pages.yml` |
| Live backend (Cloud Run, scales to zero) | `Dockerfile`, `scripts/deploy_cloudrun.py` |

```bash
uv run vocalis serve                        # http://127.0.0.1:8000
uv run python -m evals.export_demo --audio  # rebuild the recorded demo (voices cost Deepgram credit)
uv run python scripts/deploy_cloudrun.py    # deploy the backend; then commit web/config.js
```
