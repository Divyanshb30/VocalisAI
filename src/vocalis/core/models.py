"""Domain models shared by every VocalisAI component."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import IntEnum, StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field, SecretStr, computed_field


class DocType(StrEnum):
    BOARDING_PASS = "boarding_pass"
    BOOKING_CONFIRMATION = "booking_confirmation"
    CANCELLATION_NOTICE = "cancellation_notice"
    RECEIPT = "receipt"


class DisruptionType(StrEnum):
    CANCELLATION = "cancellation"
    DELAY = "delay"
    DENIED_BOARDING = "denied_boarding"
    SCHEDULE_CHANGE = "schedule_change"
    NONE = "none"


class OutcomeType(StrEnum):
    """What the passenger ends up with after a call."""

    CASH_REFUND = "cash_refund"
    COMPENSATION = "compensation"
    REFUND_AND_COMPENSATION = "refund_and_compensation"
    REBOOKING = "rebooking"
    VOUCHER = "voucher"
    CALLBACK = "callback"
    NOTHING = "nothing"


class DataTier(IntEnum):
    """Sensitivity tiers. See docs/adr/0004-late-binding-secrets.md."""

    T0 = 0  # shareable: in the LLM context
    T1 = 1  # permissioned: placeholder only, rendered if the passenger allowed it
    T2 = 2  # never: not stored, any request triggers a handoff


class Passenger(BaseModel):
    first_name: str
    last_name: str

    @computed_field  # type: ignore[prop-decorator]
    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


class FlightSegment(BaseModel):
    carrier: str = Field(description="Operating carrier IATA code, e.g. 6E, BA, EK")
    flight_number: str = Field(description="Number without the carrier prefix, e.g. 2135")
    origin: str = Field(description="IATA airport code")
    destination: str = Field(description="IATA airport code")
    scheduled_departure: datetime
    scheduled_arrival: datetime | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def designator(self) -> str:
        return f"{self.carrier}{self.flight_number}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def block_minutes(self) -> int | None:
        if self.scheduled_arrival is None:
            return None
        return int((self.scheduled_arrival - self.scheduled_departure).total_seconds() // 60)


class AlternativeOffer(BaseModel):
    """A replacement flight the airline offered, relative to the original schedule."""

    departs_minutes_after_original: int
    arrives_minutes_after_original: int


class Disruption(BaseModel):
    type: DisruptionType = DisruptionType.NONE
    notice_hours: float | None = Field(
        default=None, description="Hours between notification and scheduled departure"
    )
    arrival_delay_minutes: int | None = None
    departure_delay_minutes: int | None = None
    alternative: AlternativeOffer | None = None
    extraordinary_circumstances: bool | None = Field(
        default=None, description="Weather, ATC, security... None when unknown"
    )
    reason_given: str | None = None
    passenger_declined_alternative: bool = False


class Fare(BaseModel):
    currency: str
    total: Decimal
    base_fare: Decimal | None = None
    fuel_charge: Decimal | None = None


class Case(BaseModel):
    """Everything known about one passenger's problem with one booking."""

    id: str = Field(default_factory=lambda: uuid4().hex[:12])
    passenger: Passenger
    booking_reference: str = Field(description="PNR, 6 alphanumeric characters")
    airline: str = Field(description="Marketing carrier IATA code")
    segments: list[FlightSegment]
    disruption: Disruption = Field(default_factory=Disruption)
    fare: Fare | None = None
    ticket_number: str | None = None
    booking_email: str | None = None
    source_documents: list[DocType] = Field(default_factory=list)

    @property
    def first_segment(self) -> FlightSegment:
        return self.segments[0]

    @property
    def last_segment(self) -> FlightSegment:
        return self.segments[-1]


class VaultEntry(BaseModel):
    key: str
    value: SecretStr
    tier: DataTier
    allowed: bool = Field(default=False, description="Passenger allowed this T1 value to be shared")


class Mandate(BaseModel):
    """What the passenger authorised the agent to negotiate for."""

    target: OutcomeType
    acceptable: list[OutcomeType] = Field(default_factory=list)
    forbidden: list[OutcomeType] = Field(default_factory=lambda: [OutcomeType.VOUCHER])
    min_amount: Decimal | None = None
    currency: str | None = None
    notes: str = ""

    def permits(self, outcome: OutcomeType) -> bool:
        if outcome in self.forbidden:
            return False
        return outcome == self.target or outcome in self.acceptable


class Handoff(BaseModel):
    reason: str
    trigger_text: str
    at_turn: int


class CallOutcome(BaseModel):
    outcome: OutcomeType = OutcomeType.NOTHING
    amount: Decimal | None = None
    currency: str | None = None
    reference_number: str | None = None
    summary: str = ""
    handoffs: list[Handoff] = Field(default_factory=list)
    ended_at: datetime | None = None
