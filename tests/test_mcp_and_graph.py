"""MCP servers through a real MCP client (in-memory transport), and the LangGraph workflow."""

import uuid
from pathlib import Path

import pytest
from fastmcp import Client
from langgraph.types import Command

from vocalis.mcp_servers.calls_server import mcp as calls_mcp
from vocalis.mcp_servers.rights_server import mcp as rights_mcp
from vocalis.orchestrator.graph import build_graph
from vocalis.simair.scenario import load_scenarios


def _data(result):
    return result.data if getattr(result, "data", None) is not None else result.structured_content


async def test_rights_server_lists_tools_and_computes_dgca():
    async with Client(rights_mcp) as client:
        names = {t.name for t in await client.list_tools()}
        assert {"check_entitlements", "explain_regime", "supported_airports_and_airlines"} <= names
        res = await client.call_tool(
            "check_entitlements",
            {
                "airline": "6E",
                "flight_number": "2135",
                "origin": "DEL",
                "destination": "BOM",
                "departure": "2026-10-12T09:30",
                "arrival": "2026-10-12T11:40",
                "disruption": "cancellation",
                "notice_hours": 10,
                "extraordinary_circumstances": False,
                "fare_currency": "INR",
                "fare_total": 6800,
                "base_fare": 5200,
                "fuel_charge": 900,
            },
        )
        data = _data(res)
        assert [r["regime"] for r in data["applicable_regimes"]] == ["DGCA"]
        comp = [e for e in data["entitlements"] if e["kind"] == "compensation"]
        assert comp and comp[0]["amount"] == "6100" and comp[0]["currency"] == "INR"


@pytest.mark.timeout(120)
async def test_calls_server_case_to_report_offline(tmp_path, monkeypatch):
    monkeypatch.setattr("vocalis.cases.STORE", tmp_path)
    async with Client(calls_mcp) as client:
        case = _data(
            await client.call_tool(
                "create_case",
                {
                    "passenger_first_name": "Emily",
                    "passenger_last_name": "Clarke",
                    "booking_reference": "W9RD3L",
                    "airline": "BA",
                    "flight_number": "1446",
                    "origin": "LHR",
                    "destination": "EDI",
                    "departure": "2026-11-03T17:15",
                    "disruption": "cancellation",
                    "notice_hours": 48,
                    "extraordinary_circumstances": False,
                },
            )
        )
        assert case["mandate"]["target"] == "refund_and_compensation"
        call = _data(await client.call_tool("place_call", {"case_id": case["case_id"], "offline": True}))
        assert call["error"] is None and call["reference_verified"]
        transcript = _data(await client.call_tool("get_transcript", {"call_id": call["call_id"]}))
        text = transcript if isinstance(transcript, str) else transcript["result"]
        assert "AI assistant" in text


@pytest.mark.timeout(120)
async def test_graph_pauses_for_approval_then_calls(tmp_path, monkeypatch):
    monkeypatch.setattr("vocalis.cases.STORE", tmp_path)
    sc = {s.id: s for s in load_scenarios(Path("evals/scenarios"))}["uk_ba_cancel_2_days__cooperative"]
    graph = build_graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    first = await graph.ainvoke(
        {"case": sc.case.model_dump(mode="json"), "offline": True, "rep_persona": "cooperative"}, config
    )
    assert "__interrupt__" in first and "report" not in first
    assert first["mandate"]["forbidden"] == ["voucher"]
    done = await graph.ainvoke(Command(resume={"approved": True}), config)
    assert done["report"]["reference_verified"]


async def test_graph_declined_places_no_call(tmp_path, monkeypatch):
    monkeypatch.setattr("vocalis.cases.STORE", tmp_path)
    sc = load_scenarios(Path("evals/scenarios"))[0]
    graph = build_graph()
    config = {"configurable": {"thread_id": uuid.uuid4().hex}}
    await graph.ainvoke({"case": sc.case.model_dump(mode="json"), "offline": True}, config)
    done = await graph.ainvoke(Command(resume={"approved": False}), config)
    assert "report" not in done
