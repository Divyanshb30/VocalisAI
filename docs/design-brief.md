# VocalisAI — design brief

A one-page map of every major design decision: what was chosen, what else was considered, and why. Each row has a longer ADR in [`adr/`](adr/).

## The system in one paragraph

A passenger uploads a photo of a travel document. **docintel** turns it into a typed `Case` (vision-model extraction cross-checked against the IATA boarding-pass barcode). The **rights** engine computes what the passenger is owed **in code** (DGCA, UK261/EU261, GCAA) and retrieves the airline's own policy with citations. A **LangGraph** case workflow drafts a negotiation **mandate** and pauses for the passenger's approval. The **Pipecat** voice agent then places the call on a phone line (simulated in v1, Twilio later): it navigates the IVR with keypress tones, waits on hold with the LLM switched off, discloses it is an AI, and negotiates using a **Pipecat Flows** state machine, with guardrails on every input and output. When an OTP, payment or identity check comes up, the **CallBridge** hands the live call to the passenger and takes it back afterwards. The call ends with a read-back of the reference number and a report. Everything is evaluated by **SimAir**, an adversarial airline simulator, and traced in **Opik**.

## Decisions

| # | Decision | Chosen | Alternatives | Why it won |
|---|---|---|---|---|
| 1 | Voice architecture | Cascaded STT → LLM → TTS | Speech-to-speech (OpenAI Realtime, Gemini Live) | Text checkpoint for guardrails before any audio is spoken; each stage swappable and measurable. S2S has lower latency but is opaque and hard to filter. |
| 2 | Voice framework | Pipecat (incl. Flows) | LiveKit Agents; Vapi / Retell (managed); DIY | Open source; IVR navigator + voicemail detector built in; telephony serializers without a media server; OpenTelemetry. LiveKit shines for WebRTC rooms but needs a media server / SIP. Managed platforms = lock-in and nothing learned. |
| 3 | Dialogue control | Pipecat Flows state machine | One large prompt; LangGraph inside the call | Per-node prompts and tools → lower latency, fits small (8k) context windows, predictable transitions. LangGraph is not built for streaming audio frames. |
| 4 | Turn-taking | Silero VAD + smart-turn model | Fixed silence timeout; STT endpointing | Fewer false interruptions and faster replies than a fixed silence threshold. |
| 5 | Speech-to-text | Deepgram streaming (+ fine-tuned Whisper second pass, later) | Whisper-only; AssemblyAI; Google | Streaming with $200 free credit. Whisper is not natively streaming; as a second pass on reference-number spans it adds accuracy without latency. |
| 6 | Text-to-speech | Deepgram Aura-2 (agent); Kokoro (simulator) | ElevenLabs; Cartesia; Azure | Shared free credit, low time-to-first-byte; Kokoro is Apache-2.0 and runs locally for free. |
| 7 | LLMs | Quota-aware free-tier router; fast talker + async planner | One paid provider; self-hosting | $0 with failover; Cerebras-class speed for the talker; a stronger planner off the latency path. |
| 8 | Case workflow | LangGraph | CrewAI; plain code; Temporal | Durable checkpoints and human-approval interrupts across a long call. CrewAI gives less control; Temporal is heavy for this. |
| 9 | Tools | MCP servers (FastMCP) | Direct function calls only | One tool surface for any MCP client (LangGraph, the voice agent, desktop assistants). The in-call hot path still uses direct functions for latency. |
| 10 | Knowledge | Entitlements in code + hybrid retrieval (BM25 + dense + RRF) | Pure vector RAG; LLM computes entitlements; dedicated vector DB | Legal arithmetic must be deterministic and testable. Hybrid retrieval catches exact clause terms that embeddings miss. |
| 11 | Vision | Gemini structured output + IATA BCBP barcode cross-check | Tesseract + regex; local VLM only | Strong and free; the barcode is ground truth that catches hallucinated fields. |
| 12 | Safety | Late-binding secrets + deterministic output guard + canaries + parallel input guard + handoff | Prompt instructions only; LLM guard models | The model cannot leak a value it never sees. Regex guards are fast and auditable; LLM guards add latency and can be jailbroken themselves. |
| 13 | Handoff | Mute / transfer legs at the CallBridge (Twilio call update later) | Conference call; cold transfer | Agent keeps listening, can take the call back, no extra paid phone leg. |
| 14 | Telephony (v1) | Faithful phone-line simulation: 8 kHz μ-law, in-band DTMF decoded by Goertzel, IVR, hold | Real PSTN from day 1 | $0 and reproducible. Twilio media streams cannot send DTMF, so in-band tones are the real-world path anyway. |
| 15 | Evals | Own simulation harness + deterministic checks + LLM judge + Wilson CIs + CI gate | Manual testing; paid platforms; Pipecat's built-in evals | Pipecat's evals model an inbound *caller* persona; VocalisAI is the caller into IVRs, hold and adversarial reps, and needs canary leak checks. |
| 16 | Fine-tuning (later) | Whisper LoRA on synthetic phone audio; call-state classifier head | Full fine-tune; keyterm prompting only; heuristics | Cheap on free GPUs; targets the two classic outbound failures (misheard references, hold/voicemail detection). |
| 17 | Serving (later) | faster-whisper int8, ONNX int8 | fp32 HF pipelines; TensorRT | ~4× smaller/faster, simple to operate. |
| 18 | Observability | OpenTelemetry → Opik; Sentry; CloudWatch | Langfuse; LangSmith; Datadog | Pipecat's native Opik integration; per-turn spans with time-to-first-byte; free. |
| 19 | Infra (later) | AWS EC2 us-east-1 + OpenTofu + GitHub Actions OIDC | Cloud Run; Terraform; Kubernetes | Long-lived connections (Cloud Run caps requests at 60 min); open-source IaC; no stored cloud keys. Kubernetes is overkill. |
| 20 | Data | Postgres + pgvector (+ LangGraph checkpoints) | Mongo; Qdrant; Redis | SQL, vectors and durable workflow state in one system. |
| 21 | Frontend (later) | Next.js on Vercel | Streamlit; Gradio | Real product UI with in-browser calling and takeover; free hosting. |

## Numbers to know

- **Latency budget** (target p95 < 1.2 s server-side): end-of-turn detection 200–300 ms · STT final 150–300 ms · LLM first token 150–400 ms · TTS first byte 100–250 ms · plus transport.
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
