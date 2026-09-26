"""IATA Bar Coded Boarding Pass (BCBP, Resolution 792) — encode, decode and read from images.

The PDF417/Aztec/QR barcode on a boarding pass carries the passenger name, PNR, route,
carrier, flight number and date in fixed-width fields. It is ground truth: when the vision
model and the barcode disagree, the barcode wins.

Mandatory items, first leg (60 chars):
  format 'M'(1) · legs(1) · name(20) · e-ticket 'E'(1) · PNR(7) · from(3) · to(3) · carrier(3)
  · flight(5) · julian date(3) · compartment(1) · seat(4) · check-in seq(5) · status(1)
  · conditional-size hex(2)
Each further leg repeats PNR..status plus its own size field.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


class BCBPError(ValueError):
    pass


@dataclass(frozen=True)
class BCBPLeg:
    pnr: str
    origin: str
    destination: str
    carrier: str
    flight_number: str
    julian_day: int
    compartment: str
    seat: str
    checkin_sequence: str
    passenger_status: str

    def flight_date(self, reference: date | None = None) -> date:
        """Resolve the day-of-year to the date nearest ``reference`` (BCBP omits the year)."""
        ref = reference or date.today()
        candidates = [date(y, 1, 1) + timedelta(days=self.julian_day - 1) for y in (ref.year - 1, ref.year, ref.year + 1)]
        return min(candidates, key=lambda d: abs((d - ref).days))


@dataclass(frozen=True)
class BCBP:
    last_name: str
    first_name: str
    eticket: bool
    legs: list[BCBPLeg]
    raw: str


def _leg(s: str) -> tuple[BCBPLeg, int]:
    if len(s) < 37:
        raise BCBPError("leg too short")
    leg = BCBPLeg(
        pnr=s[0:7].strip(),
        origin=s[7:10].strip(),
        destination=s[10:13].strip(),
        carrier=s[13:16].strip(),
        flight_number=s[16:21].strip().lstrip("0") or "0",
        julian_day=int(s[21:24]),
        compartment=s[24:25],
        seat=s[25:29].strip().lstrip("0"),
        checkin_sequence=s[29:34].strip().lstrip("0"),
        passenger_status=s[34:35],
    )
    cond_size = int(s[35:37], 16)
    return leg, 37 + cond_size


def decode(raw: str) -> BCBP:
    s = raw.rstrip("\r\n")
    if not s or s[0] != "M":
        raise BCBPError("not an IATA BCBP 'M' format string")
    try:
        n_legs = int(s[1])
    except ValueError as exc:
        raise BCBPError("bad leg count") from exc
    name = s[2:22].strip()
    last, _, first = name.partition("/")
    eticket = s[22] == "E"
    pos = 23
    legs: list[BCBPLeg] = []
    for _ in range(n_legs):
        leg, used = _leg(s[pos:])
        legs.append(leg)
        pos += used
    return BCBP(last_name=last.strip(), first_name=first.strip(), eticket=eticket, legs=legs, raw=s)


def encode(
    last_name: str,
    first_name: str,
    pnr: str,
    origin: str,
    destination: str,
    carrier: str,
    flight_number: str,
    flight_date: date,
    seat: str = "12A",
    compartment: str = "Y",
    sequence: int = 42,
) -> str:
    """Build a single-leg BCBP string (used for synthetic boarding passes and tests)."""
    name = f"{last_name.upper()}/{first_name.upper()}"[:20].ljust(20)
    julian = flight_date.timetuple().tm_yday
    return (
        "M1"
        + name
        + "E"
        + pnr.upper().ljust(7)
        + origin.upper()
        + destination.upper()
        + carrier.upper().ljust(3)
        + flight_number.rjust(4, "0").ljust(5)
        + f"{julian:03d}"
        + compartment
        + seat.rjust(4, "0")
        + f"{sequence:04d}".ljust(5)
        + "1"
        + "00"
    )


def read_image(path: str | Path) -> BCBP | None:
    """Find a boarding-pass barcode in an image and decode it; None if there isn't one."""
    import zxingcpp
    from PIL import Image

    img = Image.open(path)
    for result in zxingcpp.read_barcodes(img):
        text = result.text
        if text.startswith("M") and len(text) >= 60:
            try:
                return decode(text)
            except BCBPError:
                continue
    return None


def render_barcode(text: str, scale: int = 3):  # type: ignore[no-untyped-def]
    """PDF417 image of a BCBP string (PIL image), for synthetic documents."""
    import zxingcpp
    from PIL import Image

    barcode = zxingcpp.create_barcode(text, zxingcpp.BarcodeFormat.PDF417)
    img = barcode.to_image(scale=scale)
    return Image.fromarray(img) if not isinstance(img, Image.Image) else img
