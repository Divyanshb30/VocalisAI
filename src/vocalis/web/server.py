"""Web app: one page (web/index.html) plus a small API.

    uv run vocalis serve            # http://127.0.0.1:8000

Modes the page offers:
- live     real models (needs keys): the call streams line by line over SSE
- offline  the same Pipecat pipeline with scripted models (no keys)
- replay   recorded eval calls from web/demo/calls.json (works with no server at all)
"""

from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from vocalis.calls import build_report, summary_text
from vocalis.cases import default_mandate, load_case, save_case, save_report, scenario_for_case
from vocalis.core.geo import airport, carrier_name
from vocalis.core.models import Case, Mandate
from vocalis.core.settings import get_settings
from vocalis.rights.engine import assess
from vocalis.simair.call import CallSimulation, Line, RunConfig
from vocalis.simair.scenario import Persona, Scenario, load_scenarios

ROOT = Path(__file__).resolve().parents[3]
WEB = ROOT / "web"
SCENARIO_DIR = ROOT / "evals" / "scenarios"
SUMMARY = ROOT / "evals" / "results" / "summary.json"

app = FastAPI(title="VocalisAI")


def case_view(case: Case, mandate: Mandate) -> dict[str, Any]:
    """Everything the page shows about a case, in display form."""
    seg = case.first_segment
    o, d = airport(seg.origin), airport(case.last_segment.destination)
    dis = case.disruption
    what = dis.type.value.replace("_", " ")
    if dis.notice_hours is not None:
        what += f", notified {dis.notice_hours:.0f}h before departure"
    if dis.arrival_delay_minutes:
        what += f", arrived {dis.arrival_delay_minutes // 60}h{dis.arrival_delay_minutes % 60:02d} late"
    if dis.reason_given:
        what += f" ({dis.reason_given})"
    return {
        "case_id": case.id,
        "passenger": case.passenger.full_name,
        "booking_reference": case.booking_reference,
        "airline": carrier_name(case.airline),
        "flight": seg.designator,
        "route": f"{o.city} ({o.iata}) → {d.city} ({d.iata})",
        "date": seg.scheduled_departure.strftime("%d %b %Y, %H:%M"),
        "disruption": what,
        "fare": f"{case.fare.total:,.0f} {case.fare.currency}" if case.fare else None,
        "entitlements": assess(case).summary_lines(),
        "mandate": {
            "target": mandate.target.value,
            "acceptable": [x.value for x in mandate.acceptable],
            "forbidden": [x.value for x in mandate.forbidden],
        },
    }


def _scenarios() -> dict[str, Scenario]:
    return {s.id: s for s in load_scenarios(SCENARIO_DIR)}


@app.get("/api/health")
def health() -> dict[str, Any]:
    s = get_settings()
    return {"live": bool(s.cerebras_api_key or s.groq_api_key or s.google_api_key), "offline": True}


@app.get("/api/scenarios")
def scenarios() -> list[dict[str, str]]:
    return [
        {"id": s.id, "title": s.title, "jurisdiction": s.jurisdiction, "persona": s.rep.persona.value}
        for s in _scenarios().values()
    ]


@app.get("/api/case/{scenario_id}")
def scenario_case(scenario_id: str) -> dict[str, Any]:
    sc = _scenarios().get(scenario_id)
    if not sc:
        raise HTTPException(404, "unknown scenario")
    return case_view(sc.case, sc.mandate)


@app.post("/api/extract")
async def extract_document(file: UploadFile = File(...)) -> dict[str, Any]:
    """Photo of a boarding pass / cancellation notice -> case (needs GOOGLE_API_KEY)."""
    from vocalis.docintel.extract import extract, to_case

    suffix = Path(file.filename or "doc.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
    try:
        result = await extract(tmp.name)
        case = to_case(result.fields)
    except Exception as exc:
        raise HTTPException(422, f"Could not read the document: {str(exc)[:200]}") from exc
    finally:
        Path(tmp.name).unlink(missing_ok=True)
    mandate = default_mandate(assess(case))
    save_case(case, mandate)
    view = case_view(case, mandate)
    view["barcode_found"] = result.barcode is not None
    view["conflicts"] = [c.__dict__ for c in result.conflicts]
    return view


@app.get("/api/metrics")
def metrics() -> JSONResponse:
    return JSONResponse(json.loads(SUMMARY.read_text(encoding="utf-8")) if SUMMARY.exists() else {})


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"


@app.get("/api/call")
async def call(
    scenario: str | None = None,
    case_id: str | None = None,
    mode: str = "offline",
    persona: str = "stonewaller",
) -> StreamingResponse:
    if scenario:
        sc = _scenarios().get(scenario)
        if not sc:
            raise HTTPException(404, "unknown scenario")
    elif case_id:
        case, mandate = load_case(case_id)
        sc = scenario_for_case(case, mandate or default_mandate(assess(case)), Persona(persona))
    else:
        raise HTTPException(400, "give scenario or case_id")

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    def on_line(line: Line) -> None:
        queue.put_nowait(asdict(line))

    if mode == "live":
        cfg, router = RunConfig(seed=0, label="live", on_line=on_line), None
    else:
        from vocalis.simair.offline import ScriptedAgentLLM, ScriptedRepRouter

        cfg = RunConfig(
            seed=0,
            label="offline",
            on_line=on_line,
            llm_factory=lambda: (ScriptedAgentLLM(), ["offline/scripted-agent"]),
        )
        router = ScriptedRepRouter()

    async def run() -> dict[str, Any]:
        result = await CallSimulation(sc, cfg, router=router).run()  # type: ignore[arg-type]
        call_id = uuid.uuid4().hex[:10]
        report = build_report(call_id, sc, result)
        save_report(call_id, report)
        return report

    async def stream():  # type: ignore[no-untyped-def]
        yield _sse("case", case_view(sc.case, sc.mandate))
        task = asyncio.create_task(run())
        while not (task.done() and queue.empty()):
            try:
                yield _sse("line", await asyncio.wait_for(queue.get(), timeout=0.5))
            except TimeoutError:
                yield ": keep-alive\n\n"
        try:
            report = task.result()
        except Exception as exc:
            yield _sse("error", {"message": f"{type(exc).__name__}: {exc}"[:300]})
            return
        if report.get("error"):
            yield _sse("error", {"message": report["error"][:300]})
        report.pop("transcript", None)
        report["summary"] = summary_text(report)
        yield _sse("report", report)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/", StaticFiles(directory=WEB), name="web")
