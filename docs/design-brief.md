# VocalisAI — design brief

A one-page map of every major design decision: what was chosen, what else was considered, why, and whether it is built yet. Each row has a longer ADR in [`adr/`](adr/). The README only describes what the **Built** column says is built.

## The system in one paragraph

A passenger uploads a photo of a travel document. **docintel** turns it into a typed `Case` (vision-model extraction cross-checked against the IATA boarding-pass barcode). The **rights** engine computes what the passenger is owed **in code** (DGCA, UK261/EU261, GCAA) and hybrid retrieval finds the regulation paragraph behind each entitlement. A **LangGraph** case workflow drafts a negotiation **mandate** and pauses for the passenger's approval. The **Pipecat** voice agent then places the call on a simulated phone line: it navigates the IVR with keypress tones, waits on hold with the LLM switched off, discloses it is an AI, and negotiates using a **Pipecat Flows** state machine (ivr → negotiate → confirm), with guardrails on every input and output. When an OTP, payment or identity check comes up, the step goes to the passenger (simulated in v1) and the agent's context only gets a summary. The call ends with a phonetic read-back of the reference number and a report. Everything is evaluated by **SimAir**, an adversarial airline simulator.

## Decisions

| # | Decision | Chosen | Alternatives | Why it won | Built |
|---|---|---|---|---|---|
| 1 | Voice architecture | Cascaded STT → LLM → TTS | Speech-to-speech (OpenAI Realtime, Gemini Live) | Text checkpoint for guardrails before any audio is spoken; each stage swappable and measurable. S2S has lower latency but is opaque and hard to filter. | Yes |
| 2 | Voice framework | Pipecat (incl. Flows) | LiveKit Agents; Vapi / Retell (managed); DIY | Open source; Flows state machines; telephony serializers without a media server; per-service metrics. LiveKit shines for WebRTC rooms but needs a media server / SIP. Managed platforms = lock-in and nothing learned. | Yes (text-mode pipeline; the audio leg of the evals is `telephony/voicelink.py`) |
| 3 | Dialogue control | Pipecat Flows state machine | One large prompt; LangGraph inside the call | Per-node prompts and tools → lower latency, fits small context windows, predictable transitions. LangGraph is not built for streaming audio frames. | Yes: ivr → negotiate → confirm |
| 4 | Turn-taking | Silero VAD + smart-turn model | Fixed silence timeout; STT endpointing | Fewer false interruptions and faster replies than a fixed silence threshold. | Not yet; the audio loopback uses an energy VAD + Deepgram `Finalize` |
| 5 | Speech-to-text | Deepgram streaming (+ fine-tuned Whisper second pass) | Whisper-only; AssemblyAI; Google | Streaming with free credit. Whisper is not natively streaming; as a second pass on reference-number spans it adds accuracy without latency. | Deepgram yes; Whisper pass not yet |
| 6 | Text-to-speech | Deepgram Aura-2 (agent and simulator) | ElevenLabs; Cartesia; Azure; Kokoro | Shared free credit, low time-to-first-byte over a persistent websocket. | Yes |
| 7 | LLMs | Quota-aware free-tier router; fast talker (+ async planner) | One paid provider; self-hosting | $0 with failover; Cerebras-class speed for the talker; a stronger planner would sit off the latency path. | Router and talker yes; planner not yet |
| 8 | Case workflow | LangGraph | CrewAI; plain code; Temporal | Checkpoints and human-approval interrupts. CrewAI gives less control; Temporal is heavy for this. | Yes, with in-memory checkpoints (durable Postgres checkpoints not yet) |
| 9 | Tools | MCP servers (FastMCP) | Direct function calls only | One tool surface for any MCP client. The in-call hot path still uses direct functions for latency. | Yes: `vocalis-calls`, `vocalis-rights` |
| 10 | Knowledge | Entitlements in code + hybrid retrieval (BM25 + dense + RRF) over regulation text | Pure vector RAG; LLM computes entitlements; dedicated vector DB | Legal arithmetic must be deterministic and testable. Hybrid retrieval catches exact clause terms that embeddings miss. | Yes |
| 11 | Vision | Gemini structured output + IATA BCBP barcode cross-check | Tesseract + regex; local VLM only | Strong and free; the barcode is ground truth that catches hallucinated fields. | Yes |
| 12 | Safety | Late-binding secrets + deterministic output guard + canaries + input guard + handoff | Prompt instructions only; LLM guard models | The model cannot leak a value it never sees. Regex guards are fast and auditable; LLM guards add latency and can be jailbroken themselves. | Yes |
| 13 | Handoff | Mute / transfer legs at a CallBridge | Conference call; cold transfer | Agent keeps listening, can take the call back, no extra paid phone leg. | Not yet; today the call loop routes the step to the simulated passenger |
| 14 | Telephony (v1) | Faithful phone-line simulation: 8 kHz μ-law, in-band DTMF decoded by Goertzel, IVR, hold | Real PSTN from day 1 | $0 and reproducible. Twilio media streams cannot send DTMF, so in-band tones are the real-world path anyway. | Yes |
| 15 | Evals | Own simulation harness + deterministic checks + LLM judge + Wilson CIs (+ judge calibration, CI safety gate) | Manual testing; paid platforms; Pipecat's built-in evals | Pipecat's evals model an inbound *caller* persona; VocalisAI is the caller into IVRs, hold and adversarial reps, and needs canary leak checks. | Harness, checks, judge, CIs yes; calibration and CI gate not yet |
| 16 | Fine-tuning | Whisper LoRA on synthetic phone audio; call-state classifier head | Full fine-tune; keyterm prompting only; heuristics | Cheap on free GPUs; targets the two classic outbound failures (misheard references, hold/voicemail detection). | Not yet; call state uses transcript heuristics |
| 17 | Serving | faster-whisper int8, ONNX int8 | fp32 HF pipelines; TensorRT | ~4× smaller/faster, simple to operate. | Not yet |
| 18 | Observability | OpenTelemetry → Opik; Sentry; CloudWatch | Langfuse; LangSmith; Datadog | Native Opik integrations; per-turn spans with time-to-first-byte; free. | Opik on LiteLLM calls (rep, judge) only |
| 19 | Infra | Cloud Run (backend) + GitHub Pages (demo); later AWS EC2 + OpenTofu + GitHub Actions OIDC | Terraform; Kubernetes | Scale-to-zero for the demo; long-lived call connections later need a VM (Cloud Run caps requests at 60 min). | Cloud Run + Pages yes; AWS not yet |
| 20 | Data | Postgres + pgvector (+ LangGraph checkpoints) | Mongo; Qdrant; Redis | SQL, vectors and durable workflow state in one system. | Not yet (in-memory; vectors cached on disk) |
| 21 | Frontend | Static page today; Next.js on Vercel later | Streamlit; Gradio | Real product UI with in-browser calling and takeover; free hosting. | Static page yes; Next.js not yet |

## Numbers to know

- **Latency budget** (target, server-side p95 < 1.2 s): end-of-turn detection 200–300 ms · STT final 150–300 ms · LLM first token 150–400 ms · TTS first byte 100–250 ms · plus transport. Measured numbers are in the README.
- **Task success:** outcome inside the approved mandate *and* the correct reference number captured.
- **Leak rate:** runs where any unauthorised T1/T2 value or canary appears in agent output; with 0 leaks in N runs, the 95% upper bound is ≈ 3/N (rule of three).
- **Handoff accuracy:** precision / recall / F1 over events that require a human.
- **Judge calibration:** Cohen's κ between the LLM judge and human labels.
- **Confidence intervals:** Wilson score interval for every rate.

## Questions an interviewer will ask

- How do you handle the rep interrupting the agent (barge-in)?
- Why not speech-to-speech?
- How do you stop prompt injection coming from the other side of the call?
- How do you know the LLM judge is right?
- What happens when a free-tier provider fails mid-call?
- How would you scale to 1,000 concurrent calls?
- What does one call cost, broken down?
- What would you change with a real budget?
