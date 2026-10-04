# Claims ledger

Every public claim in the README, mapped to the code that implements it, the tests that pin it down, and the measurement behind it. `tests/test_claims.py` checks this file against the repository on every push: each path listed here must exist, each test named here must exist, each metric key must be present in `evals/results/summary.json`, and the README metrics table must equal what `evals/report.py` generates from that file.

Metric keys are paths into `evals/results/summary.json`; `<text>`, `<voice>` and `<baseline>` stand for the configs named in its `primary` entry.

Scoring version: **v1** (the checks described in `src/vocalis/simair/scoring.py` at the commit that produced the runs). Known limits of v1 scoring are listed at the end; they are being closed before any number is republished.

## What it does

| Claim | Code | Tests | Evidence |
|---|---|---|---|
| A photo of a boarding pass, e-ticket or cancellation email becomes a typed `Case` | `src/vocalis/docintel/extract.py` | `tests/test_mcp_and_graph.py::test_calls_server_case_to_report_offline` | `docbench.summary` |
| Vision output is cross-checked against the IATA boarding-pass barcode | `src/vocalis/docintel/bcbp.py`, `src/vocalis/docintel/extract.py` | `tests/test_bcbp.py::test_decode_iata_sample`, `tests/test_bcbp.py::test_barcode_image_roundtrip` | `docbench.summary` |
| Entitlements are computed in code for DGCA, UK261/EU261 and GCAA (+ Montreal) | `src/vocalis/rights/engine.py`, `src/vocalis/rights/dgca.py`, `src/vocalis/rights/eu261.py`, `src/vocalis/rights/uae.py` | `tests/test_rights.py::test_dgca_cancellation_bands`, `tests/test_rights.py::test_uk261_long_haul_cancellation_short_notice`, `tests/test_rights.py::test_uae_cancellation_refund_care_and_damages` | — |
| Each entitlement is tied to the clause it comes from | `src/vocalis/rights/models.py` | `tests/test_rights.py::test_every_entitlement_has_a_citation` | — |
| Hybrid search (BM25 + dense + RRF) over the regulation text | `src/vocalis/rights/policy.py`, `data/policy/eu261.txt`, `data/policy/uk261.txt`, `evals/retrieval.py` | `tests/test_policy.py::test_hybrid_fuses_and_filters_by_jurisdiction`, `tests/test_policy.py::test_bm25_finds_exact_clause_words` | `retrieval.results` |
| Navigates the phone menu with keypress tones | `src/vocalis/telephony/dtmf.py`, `src/vocalis/simair/ivr.py`, `src/vocalis/agent/session.py` | `tests/test_telephony.py::test_dtmf_survives_phone_line`, `tests/test_voicelink.py::test_keypresses_survive_the_line` | `<text>.ivr_reached_agent`, `<voice>.voice.dtmf_keys_decoded` |
| Waits through hold with the LLM switched off | `src/vocalis/simair/call.py` | `tests/test_call_offline.py::test_full_call_voucher_pusher_with_card_request` | `<text>.llm_calls_during_hold` |
| Discloses that it is an AI, and never denies it | `src/vocalis/agent/briefing.py`, `src/vocalis/simair/call.py` | `tests/test_call_offline.py::test_full_call_voucher_pusher_with_card_request` | `<text>.disclosure_first_utterance`, `<text>.honest_when_asked_if_human` |
| Negotiates against the passenger's mandate and checks every offer against it | `src/vocalis/agent/session.py`, `src/vocalis/core/models.py` | `tests/test_call_offline.py::test_full_call_voucher_pusher_with_card_request` | `<text>.task_success`, `<text>.mandate_violations` |
| Hands over for OTPs, payments, identity checks; sensitive values never enter the model's context | `src/vocalis/guards/input_guard.py`, `src/vocalis/guards/vault.py`, `src/vocalis/simair/call.py` | `tests/test_guards.py::test_input_guard_flags`, `tests/test_guards.py::test_vault_renders_only_allowed_t1`, `tests/test_call_offline.py::test_social_engineer_triggers_guard_handoffs` | `<text>.handoff`, `<text>.leak_runs` |
| Every utterance and keypress passes a deterministic output guard | `src/vocalis/guards/output_guard.py`, `src/vocalis/agent/processors.py` | `tests/test_guards.py::test_output_guard_blocks_secrets_even_when_spelled_out`, `tests/test_guards.py::test_output_guard_dtmf`, `tests/test_sim_fixes.py::test_tool_call_text_is_never_spoken` | `<text>.leak_runs` |
| Canary secrets are planted in every run | `src/vocalis/guards/canaries.py` | `tests/test_guards.py::test_canaries_detect_leaks_and_are_well_formed` | `<text>.leak_runs` |
| The reference is read back phonetically and confirmed by the rep | `src/vocalis/agent/session.py`, `src/vocalis/agent/prompts.py`, `src/vocalis/simair/rep.py` | `tests/test_readback.py::test_readback_detection`, `tests/test_readback.py::test_readback_correction_updates_the_record` | `readback.before`, `readback.after` |
| Simulated phone line: 8 kHz μ-law, band-pass, noise, in-band DTMF | `src/vocalis/telephony/phoneline.py`, `src/vocalis/telephony/dtmf.py` | `tests/test_telephony.py::test_mulaw_roundtrip_error_small`, `tests/test_telephony.py::test_dtmf_survives_phone_line` | — |
| Audio loopback: Aura-2 voices over the simulated line into streaming STT, end of turn by a local VAD and `Finalize` | `src/vocalis/telephony/voicelink.py` | `tests/test_voicelink.py::test_spelled_codes_read_one_character_at_a_time`, `tests/test_voicelink.py::test_wer_normalisation` | `<voice>.voice.voice_to_voice_s`, `<voice>.voice.wer_all` |
| LangGraph case workflow pauses for passenger approval | `src/vocalis/orchestrator/graph.py` | `tests/test_mcp_and_graph.py::test_graph_pauses_for_approval_then_calls`, `tests/test_mcp_and_graph.py::test_graph_declined_places_no_call` | — |
| MCP servers `vocalis-calls` and `vocalis-rights` | `src/vocalis/mcp_servers/calls_server.py`, `src/vocalis/mcp_servers/rights_server.py` | `tests/test_mcp_and_graph.py::test_rights_server_lists_tools_and_computes_dgca`, `tests/test_mcp_and_graph.py::test_calls_server_case_to_report_offline` | — |
| Quota-aware free-tier router with fallback | `src/vocalis/llm/router.py`, `src/vocalis/agent/llm_services.py` | — | `<text>.talker_models`, `<text>.talker_served` |
| Web page with Live / Offline / Recorded modes, FastAPI backend | `web/index.html`, `src/vocalis/web/server.py`, `evals/export_demo.py` | — | — |
| Talker comparison: same code and scenarios, different talker model | `evals/report.py` | — | `vocalis`, `archive/v2/vocalis_qwen` |

## Not built (must not appear in the README)

These are design decisions or sprint items that are not implemented yet. `tests/test_claims.py` fails if the README describes any of them as part of the system; each comes back to the README in the change that ships it.

CallBridge · Silero VAD / smart-turn · async planner · calibrated LLM judge · CI safety gate · Postgres / pgvector · airline-policy retrieval · mandate approval or take-over in the web page · Twilio / real phone calls · calendar or mail MCP servers · Opik tracing of the voice pipeline · Sentry · CloudWatch · Whisper fine-tune · call-state classifier model.

## Known limits of scoring v1

Being fixed in this order before anything is republished:

1. Handoff false positives are judged against scripted events only; a request the simulated rep improvises is not ground truth.
2. A mandate violation is taken from the agent's own record of the outcome, not from what the simulator granted.
3. The leak detector matches a fixed set of formats for each canary.
4. The simulated rep always corrects a wrong read-back.
5. Runs do not yet record the git commit or the model id that actually served each reply.
