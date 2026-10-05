"""Fabricated airline emails and e-tickets, and phone-capture damage at three severities.

Every name, booking reference, address and amount is generated; nothing comes from a real person or
booking. The documents imitate the layout of an email client and an itinerary receipt (header bar,
sender line, banner, footer), and are then "photographed": perspective, screen moire, glare, low light,
cropping, blur and JPEG compression, scaled by severity, so the same document can be read at each level.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from evals.docbench.synth import FIRST, LAST, REASONS, ROUTES, _fare, _font, _pnr
from vocalis.core.geo import carrier_name

SEVERITIES = ("light", "medium", "heavy")
_LEVEL = {"light": 0.3, "medium": 0.65, "heavy": 1.0}


def _flight(rng: random.Random) -> dict:
    carrier, o, d, mins, cur = rng.choice(ROUTES)
    day = date(2026, 10, 1) + timedelta(days=rng.randint(0, 90))
    dep = datetime(day.year, day.month, day.day, rng.randint(5, 22), rng.choice([0, 15, 30, 45]))
    base, fuel, total = _fare(rng, cur, mins)
    first, last = rng.choice(FIRST), rng.choice(LAST)
    return {
        "carrier": carrier, "o": o, "d": d, "mins": mins, "cur": cur, "dep": dep, "fno": str(rng.randint(100, 2999)),
        "pnr": _pnr(rng), "first": first, "last": last, "base": base, "fuel": fuel, "total": total,
        "email": f"{first.lower()}.{last.lower()}{rng.randint(1, 99)}@example.com",
    }  # fmt: skip


def _mail_frame(
    subject: str, sender: str, to: str, sent: datetime, body: list[tuple[str, int]]
) -> Image.Image:
    """An email as a mail client shows it: toolbar, subject, sender block, banner, body, footer."""
    W = 1500
    H = 330 + sum(s + 14 for _, s in body) + 120
    img = Image.new("RGB", (W, H), "white")
    g = ImageDraw.Draw(img)
    g.rectangle([0, 0, W, 54], fill=(242, 243, 245))
    g.text((24, 14), "Inbox    Archive    Reply    Forward", font=_font(20), fill=(90, 90, 100))
    g.text((32, 76), subject, font=_font(32), fill=(15, 15, 20))
    g.ellipse([32, 132, 80, 180], fill=(200, 205, 215))
    g.text((96, 132), sender, font=_font(21), fill=(30, 30, 35))
    g.text(
        (96, 160), f"to {to}  ·  {sent.strftime('%a %d %b %Y, %H:%M')}", font=_font(18), fill=(110, 110, 120)
    )
    g.rectangle([32, 206, W - 32, 266], fill=(20, 40, 90))
    g.text((52, 220), sender.split("<")[0].strip().upper(), font=_font(28), fill="white")
    y = 290
    for text, size in body:
        g.text((52, y), text, font=_font(size), fill=(20, 20, 20))
        y += size + 14
    g.line([32, y + 20, W - 32, y + 20], fill=(220, 220, 225))
    g.text((52, y + 36), "This is an automated message. Please do not reply. Manage your booking online.", font=_font(17),
           fill=(130, 130, 140))  # fmt: skip
    return img


def email_cancellation(rng: random.Random) -> tuple[Image.Image, dict]:
    f = _flight(rng)
    notice = f["dep"] - timedelta(hours=rng.choice([4, 10, 20, 40, 72, 150, 400]))
    name = carrier_name(f["carrier"])
    reason = rng.choice(REASONS)
    body = [
        (f"Dear {f['first']} {f['last']},", 24),
        (f"We are sorry to tell you that flight {f['carrier']} {f['fno']} from {f['o']} to {f['d']}", 24),
        (
            f"on {f['dep'].strftime('%d %B %Y')} at {f['dep'].strftime('%H:%M')} has been cancelled because of {reason}.",
            24,
        ),
        ("", 8),
        (f"Booking reference: {f['pnr']}", 26),
        (
            f"Amount paid: {f['cur']} {f['total']:,} (base fare {f['cur']} {f['base']:,}, fuel {f['cur']} {f['fuel']:,})",
            24,
        ),
        ("", 8),
        ("You can choose a refund to your original payment method or a free change to another flight.", 22),
        ("We apologise for the inconvenience caused.", 22),
    ]
    img = _mail_frame(
        f"Flight {f['carrier']} {f['fno']} cancelled - action needed",
        f"{name} <bookings@{name.lower().replace(' ', '')}.example>",
        f["email"],
        notice,
        body,
    )
    return img, _truth(f, "cancellation_notice", "cancellation", notice=notice)


def email_delay(rng: random.Random) -> tuple[Image.Image, dict]:
    f = _flight(rng)
    delay = rng.choice([150, 200, 260, 330, 420, 600])
    new_dep = f["dep"] + timedelta(minutes=delay)
    sent = f["dep"] - timedelta(hours=rng.choice([2, 5, 12]))
    name = carrier_name(f["carrier"])
    body = [
        (f"Hello {f['first']} {f['last']},", 24),
        (f"Your flight {f['carrier']} {f['fno']} from {f['o']} to {f['d']} on {f['dep'].strftime('%d %B %Y')} is delayed.", 24),
        (f"Original departure: {f['dep'].strftime('%H:%M')}    New departure: {new_dep.strftime('%H:%M')}", 26),
        (f"Booking reference: {f['pnr']}", 26),
        (f"Fare paid: {f['cur']} {f['total']:,}", 24),
        ("Meal vouchers are available at the transfer desk. We apologise for the delay.", 22),
    ]  # fmt: skip
    img = _mail_frame(
        f"Delay to your flight {f['carrier']} {f['fno']}",
        f"{name} <updates@{name.lower().replace(' ', '')}.example>",
        f["email"],
        sent,
        body,
    )
    truth = _truth(f, "cancellation_notice", "delay", notice=sent)
    truth["doc_type"] = "delay_notice"
    return img, truth


def eticket(rng: random.Random) -> tuple[Image.Image, dict]:
    """An itinerary receipt: two-column table with the ticket number and fare breakdown."""
    f = _flight(rng)
    name = carrier_name(f["carrier"])
    arr = f["dep"] + timedelta(minutes=f["mins"])
    ticket = f"{rng.randint(100, 999)}-{rng.randint(10**9, 10**10 - 1)}"
    W, H = 1400, 980
    img = Image.new("RGB", (W, H), (252, 252, 250))
    g = ImageDraw.Draw(img)
    g.rectangle([0, 0, W, 90], fill=(120, 20, 30))
    g.text((32, 24), f"{name.upper()}   E-TICKET ITINERARY / RECEIPT", font=_font(32), fill="white")
    rows = [
        ("Passenger", f"{f['last'].upper()}/{f['first'].upper()} MR/MS"),
        ("Booking reference", f["pnr"]),
        ("Ticket number", ticket),
        ("Flight", f"{f['carrier']} {f['fno']}"),
        ("From", f["o"]),
        ("To", f["d"]),
        ("Date", f["dep"].strftime("%d%b%Y").upper()),
        ("Departure", f["dep"].strftime("%H:%M")),
        ("Arrival", arr.strftime("%H:%M")),
        ("Fare", f"{f['cur']} {f['base']:,}"),
        ("Carrier-imposed fuel charge", f"{f['cur']} {f['fuel']:,}"),
        ("Total", f"{f['cur']} {f['total']:,}"),
    ]
    y = 130
    for k, v in rows:
        g.text((48, y), k, font=_font(24), fill=(100, 100, 105))
        g.text((520, y), v, font=_font(28), fill=(15, 15, 20))
        g.line([48, y + 50, W - 48, y + 50], fill=(228, 228, 232))
        y += 64
    g.text(
        (48, y + 20),
        "Baggage: 1 piece 23kg. Conditions of carriage apply.",
        font=_font(18),
        fill=(130, 130, 140),
    )
    truth = _truth(f, "receipt", "none")
    truth["arrival_time"] = arr.strftime("%H:%M")
    return img, truth


def _truth(f: dict, doc_type: str, disruption: str, notice: datetime | None = None) -> dict:
    t = {
        "doc_type": doc_type,
        "passenger_first_name": f["first"],
        "passenger_last_name": f["last"],
        "booking_reference": f["pnr"],
        "airline_code": f["carrier"],
        "flight_number": f["fno"],
        "origin_iata": f["o"],
        "destination_iata": f["d"],
        "departure_date": f["dep"].date().isoformat(),
        "departure_time": f["dep"].strftime("%H:%M"),
        "disruption": disruption,
        "fare_total": str(f["total"]),
        "currency": f["cur"],
    }
    if notice is not None:  # the emails: notified on the sent date, to the passenger's address
        t["notice_date"] = notice.date().isoformat()
        t["email"] = f["email"]
    return t


# ------------------------------------------------------------------------------------ capture damage


def _perspective(img: Image.Image, rng: random.Random, k: float) -> Image.Image:
    w, h = img.size
    d = k * 0.09
    jitter = [rng.uniform(0, d) for _ in range(8)]
    quad = (
        w * jitter[0], h * jitter[1], w * jitter[2], h * (1 - jitter[3]),
        w * (1 - jitter[4]), h * (1 - jitter[5]), w * (1 - jitter[6]), h * jitter[7],
    )  # fmt: skip
    return img.transform((w, h), Image.Transform.QUAD, quad, Image.Resampling.BICUBIC, fillcolor=(60, 60, 64))


def _moire(arr: np.ndarray, rng: random.Random, k: float) -> np.ndarray:
    """Screen-door pattern from photographing a display."""
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    angle = rng.uniform(0, math.pi)
    freq = rng.uniform(0.25, 0.45)
    wave = np.sin((xx * math.cos(angle) + yy * math.sin(angle)) * freq)
    return arr + (wave[..., None] * 18 * k)


def _glare(arr: np.ndarray, rng: random.Random, k: float) -> np.ndarray:
    h, w = arr.shape[:2]
    cy, cx = rng.uniform(0.2, 0.8) * h, rng.uniform(0.2, 0.8) * w
    r = max(h, w) * rng.uniform(0.15, 0.3)
    yy, xx = np.mgrid[0:h, 0:w]
    spot = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
    return arr + spot[..., None] * 170 * k


def capture(img: Image.Image, seed: int, severity: str) -> Image.Image:
    """The same document photographed with more or less care. Deterministic per (seed, severity)."""
    rng = random.Random(f"{seed}-{severity}")
    k = _LEVEL[severity]
    img = img.convert("RGB")
    img = img.rotate(rng.uniform(-6, 6) * k, expand=True, fillcolor=(60, 60, 64))
    img = _perspective(img, rng, k)
    w, h = img.size
    crop = rng.uniform(0, 0.04) * k  # a phone photo rarely frames the page exactly
    img = img.crop((int(w * crop), int(h * crop * 0.5), w - int(w * crop * 0.3), h))
    arr = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.5 + 0.4 * k:
        arr = _moire(arr, rng, k)
    if rng.random() < 0.3 + 0.6 * k:
        arr = _glare(arr, rng, k)
    arr = arr * (1 - 0.45 * k * rng.uniform(0.5, 1))  # low light
    arr = arr + np.random.default_rng(seed).normal(0, 4 + 14 * k, arr.shape)  # sensor noise
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    img = img.filter(ImageFilter.GaussianBlur(0.3 + 1.6 * k * rng.uniform(0.6, 1)))
    img = ImageEnhance.Contrast(img).enhance(1 - 0.25 * k)
    scale = 1 - 0.45 * k  # farther away, fewer pixels on the text
    img = img.resize((int(img.width * scale), int(img.height * scale)), Image.Resampling.BILINEAR)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(85 - 50 * k))
    return Image.open(io.BytesIO(buf.getvalue()))


def generate(n_base: int, out: Path, seed: int = 11) -> None:
    """``n_base`` fabricated documents, each saved at every severity: doc_000_light.jpg, ..."""
    from evals.docbench.synth import boarding_pass

    makers = {"boarding_pass": boarding_pass, "email_cancellation": email_cancellation, "email_delay": email_delay,
              "eticket": eticket}  # fmt: skip
    kinds = list(makers)
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    labels = {}
    for i in range(n_base):
        kind = kinds[i % len(kinds)]
        img, truth = makers[kind](rng)
        for sev in SEVERITIES:
            name = f"doc_{i:03d}_{sev}.jpg"
            capture(img, seed * 1000 + i, sev).save(out / name, "JPEG")
            labels[name] = {**truth, "kind": kind, "severity": sev, "base": i}
    (out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="base documents; each is saved at 3 severities")
    ap.add_argument("--out", type=Path, default=Path("evals/docbench/captured"))
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    generate(a.n, a.out, a.seed)
    print(f"wrote {a.n * len(SEVERITIES)} images ({a.n} documents x {len(SEVERITIES)} severities) to {a.out}")
