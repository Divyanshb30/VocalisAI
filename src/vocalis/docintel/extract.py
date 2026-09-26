"""Photo of a travel document -> structured fields -> Case.

1. Decode the IATA boarding-pass barcode if there is one (ground truth).
2. Ask a vision model for the same fields as structured JSON.
3. Reconcile: barcode fields win; disagreements are reported for the passenger to confirm.
4. Redact anything sensitive before it is stored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

from vocalis.core.models import (
    Case,
    DocType,
    Disruption,
    DisruptionType,
    Fare,
    FlightSegment,
    Passenger,
)
from vocalis.core.settings import get_settings
from vocalis.docintel.bcbp import BCBP, read_image

EXTRACTION_PROMPT = """\
You are reading a photo of an airline travel document (boarding pass, booking confirmation, \
cancellation or schedule-change notice, or receipt). Extract the fields exactly as printed. \
Use null for anything not visible. Do not guess. Dates as YYYY-MM-DD, times as HH:MM (24h). \
Airline as the 2-character IATA code (e.g. 6E, AI, BA, EK). Airports as 3-letter IATA codes. \
flight_number is digits only, without the airline code. Never extract card numbers, CVVs, \
passport numbers or one-time codes; if any are present, list their kind in sensitive_fields_present."""


class DisruptionKind(StrEnum):
    cancellation = "cancellation"
    delay = "delay"
    schedule_change = "schedule_change"
    denied_boarding = "denied_boarding"
    none = "none"


class ExtractedDocument(BaseModel):
    """Flat schema: friendly to vision models' structured-output modes."""

    doc_type: DocType
    passenger_first_name: str | None = None
    passenger_last_name: str | None = None
    booking_reference: str | None = Field(default=None, description="PNR, 6 alphanumerics")
    ticket_number: str | None = None
    airline_code: str | None = None
    flight_number: str | None = None
    origin_iata: str | None = None
    destination_iata: str | None = None
    departure_date: str | None = None
    departure_time: str | None = None
    arrival_time: str | None = None
    disruption: DisruptionKind = DisruptionKind.none
    notice_date: str | None = Field(default=None, description="Date the passenger was notified")
    new_departure_time: str | None = None
    delay_minutes: int | None = None
    reason_given: str | None = None
    fare_total: str | None = None
    base_fare: str | None = None
    fuel_charge: str | None = None
    currency: str | None = None
    email: str | None = None
    sensitive_fields_present: list[str] = Field(default_factory=list)


@dataclass
class Conflict:
    field: str
    vision: str | None
    barcode: str


@dataclass
class ExtractionResult:
    fields: ExtractedDocument
    barcode: BCBP | None
    conflicts: list[Conflict] = field(default_factory=list)
    model: str | None = None

    @property
    def needs_confirmation(self) -> bool:
        return bool(self.conflicts)


_CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PASSPORT = re.compile(r"\b[A-Z][0-9]{7}\b")


def redact(doc: ExtractedDocument) -> ExtractedDocument:
    data = doc.model_dump()
    for k, v in data.items():
        if isinstance(v, str) and k not in ("ticket_number",):
            v = _CARD.sub("[redacted]", v)
            data[k] = _PASSPORT.sub("[redacted]", v)
    return ExtractedDocument.model_validate(data)


def _norm(s: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def reconcile(doc: ExtractedDocument, bp: BCBP | None, reference: date | None = None) -> ExtractionResult:
    if bp is None or not bp.legs:
        return ExtractionResult(fields=doc, barcode=None)
    leg = bp.legs[0]
    truth = {
        "passenger_last_name": bp.last_name,
        "passenger_first_name": bp.first_name,
        "booking_reference": leg.pnr,
        "airline_code": leg.carrier,
        "flight_number": leg.flight_number,
        "origin_iata": leg.origin,
        "destination_iata": leg.destination,
        "departure_date": leg.flight_date(reference).isoformat(),
    }
    conflicts = []
    data = doc.model_dump()
    for key, value in truth.items():
        seen = data.get(key)
        same = _norm(seen) == _norm(value) or (
            key == "flight_number" and _norm(seen).lstrip("0") == _norm(value).lstrip("0")
        )
        if seen and not same:
            conflicts.append(Conflict(key, seen, value))
        data[key] = value.title() if key.startswith("passenger_") else value
    return ExtractionResult(fields=ExtractedDocument.model_validate(data), barcode=bp, conflicts=conflicts)


async def extract_with_gemini(image_bytes: bytes, mime_type: str = "image/png", model: str | None = None) -> ExtractedDocument:
    from google import genai
    from google.genai import types

    s = get_settings()
    if not s.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set")
    client = genai.Client(api_key=s.google_api_key)
    resp = await client.aio.models.generate_content(
        model=model or s.vocalis_vision_model,
        contents=[types.Part.from_bytes(data=image_bytes, mime_type=mime_type), EXTRACTION_PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ExtractedDocument,
            temperature=0,
        ),
    )
    parsed = resp.parsed
    if isinstance(parsed, ExtractedDocument):
        return parsed
    return ExtractedDocument.model_validate_json(resp.text or "{}")


async def extract(path: str | Path, model: str | None = None) -> ExtractionResult:
    p = Path(path)
    mime = "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/png"
    doc = redact(await extract_with_gemini(p.read_bytes(), mime, model))
    result = reconcile(doc, read_image(p))
    result.model = model or get_settings().vocalis_vision_model
    return result


def _dec(v: str | None) -> Decimal | None:
    if not v:
        return None
    try:
        return Decimal(re.sub(r"[^\d.]", "", v))
    except InvalidOperation:
        return None


def to_case(doc: ExtractedDocument) -> Case:
    if not (doc.booking_reference and doc.airline_code and doc.origin_iata and doc.destination_iata and doc.departure_date):
        raise ValueError("document is missing booking reference, airline, route or date")
    dep = datetime.fromisoformat(f"{doc.departure_date}T{doc.departure_time or '00:00'}")
    arr = datetime.fromisoformat(f"{doc.departure_date}T{doc.arrival_time}") if doc.arrival_time else None
    if arr and arr < dep:
        arr = arr + timedelta(days=1)  # overnight arrival
    notice_hours = None
    if doc.notice_date:
        notice_hours = (dep - datetime.fromisoformat(f"{doc.notice_date}T12:00")).total_seconds() / 3600
    fare = None
    if doc.fare_total and doc.currency:
        fare = Fare(
            currency=doc.currency.upper(),
            total=_dec(doc.fare_total) or Decimal(0),
            base_fare=_dec(doc.base_fare),
            fuel_charge=_dec(doc.fuel_charge),
        )
    return Case(
        passenger=Passenger(
            first_name=(doc.passenger_first_name or "").title(),
            last_name=(doc.passenger_last_name or "").title(),
        ),
        booking_reference=_norm(doc.booking_reference),
        airline=doc.airline_code.upper(),
        segments=[
            FlightSegment(
                carrier=doc.airline_code.upper(),
                flight_number=_norm(doc.flight_number) or "0",
                origin=doc.origin_iata.upper(),
                destination=doc.destination_iata.upper(),
                scheduled_departure=dep,
                scheduled_arrival=arr,
            )
        ],
        disruption=Disruption(
            type=DisruptionType(doc.disruption.value),
            notice_hours=notice_hours,
            departure_delay_minutes=doc.delay_minutes,
            arrival_delay_minutes=doc.delay_minutes,
            reason_given=doc.reason_given,
        ),
        fare=fare,
        ticket_number=doc.ticket_number,
        booking_email=doc.email,
        source_documents=[doc.doc_type],
    )
