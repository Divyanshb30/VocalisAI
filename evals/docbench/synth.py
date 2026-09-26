"""Synthetic travel documents with ground-truth labels, for the extraction benchmark.

Boarding passes (with a real IATA BCBP PDF417 barcode) and cancellation notices, rendered
with PIL and then degraded like phone photos (rotation, blur, noise, JPEG compression).

    uv run python -m evals.docbench.synth --n 20 --out evals/docbench/data
"""

from __future__ import annotations

import argparse
import io
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from vocalis.core.geo import carrier_name
from vocalis.docintel.bcbp import encode, render_barcode

ROUTES = [
    ("6E", "DEL", "BOM", 130, "INR"), ("6E", "BLR", "DEL", 165, "INR"), ("AI", "DEL", "LHR", 570, "INR"),
    ("QP", "BOM", "GOI", 75, "INR"), ("BA", "LHR", "DEL", 540, "GBP"), ("BA", "LHR", "EDI", 85, "GBP"),
    ("VS", "LHR", "BOM", 560, "GBP"), ("EK", "DXB", "BOM", 190, "AED"), ("FZ", "DXB", "DEL", 215, "AED"),
    ("EY", "AUH", "LHR", 460, "AED"), ("AI", "BOM", "DXB", 195, "INR"), ("6E", "HYD", "DXB", 240, "INR"),
]
FIRST = ["Asha", "Rohan", "Priya", "Arjun", "Meera", "Kabir", "Sara", "Omar", "Leila", "James", "Emily", "Vikram"]
LAST = ["Rao", "Sharma", "Iyer", "Khan", "Patel", "Singh", "Ahmed", "Hassan", "Smith", "Clarke", "Mehta", "Das"]
REASONS = ["operational reasons", "crew availability", "adverse weather", "a technical issue with the aircraft", "air traffic restrictions"]
ALNUM = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in ("C:/Windows/Fonts/arial.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _pnr(rng: random.Random) -> str:
    return "".join(rng.choice(ALNUM) for _ in range(6))


def _fare(rng: random.Random, cur: str, mins: int) -> tuple[int, int, int]:
    base = {"INR": 25, "GBP": 0.35, "AED": 1.1}[cur] * mins * rng.uniform(0.8, 1.6)
    base = int(base) if cur != "INR" else int(base / 10) * 10
    fuel = int(base * 0.18)
    total = int((base + fuel) * 1.12)
    return base, fuel, total


def _degrade(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")
    img = img.rotate(rng.uniform(-4, 4), expand=True, fillcolor=(235, 235, 230))
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.1)))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.8, 1.15))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.8, 1.1))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=rng.randint(45, 85))
    return Image.open(io.BytesIO(buf.getvalue()))


def boarding_pass(rng: random.Random) -> tuple[Image.Image, dict]:
    carrier, o, d, mins, cur = rng.choice(ROUTES)
    first, last = rng.choice(FIRST), rng.choice(LAST)
    pnr = _pnr(rng)
    fno = str(rng.randint(100, 2999))
    day = date(2026, 10, 1) + timedelta(days=rng.randint(0, 90))
    dep = datetime(day.year, day.month, day.day, rng.randint(5, 22), rng.choice([0, 15, 30, 45]))
    arr = dep + timedelta(minutes=mins)
    seat = f"{rng.randint(1, 38)}{rng.choice('ABCDEF')}"
    W, H = 1400, 560
    img = Image.new("RGB", (W, H), (250, 250, 247))
    g = ImageDraw.Draw(img)
    g.rectangle([0, 0, W, 70], fill=(20, 40, 90))
    g.text((30, 18), f"{carrier_name(carrier).upper()}   BOARDING PASS", font=_font(32), fill="white")
    rows = [
        ("PASSENGER", f"{last.upper()}/{first.upper()}"), ("FLIGHT", f"{carrier} {fno}"),
        ("FROM", o), ("TO", d), ("DATE", dep.strftime("%d %b %Y").upper()),
        ("DEPARTS", dep.strftime("%H:%M")), ("ARRIVES", arr.strftime("%H:%M")),
        ("SEAT", seat), ("PNR", pnr),
    ]
    x, y = 40, 100
    for i, (k, v) in enumerate(rows):
        cx, cy = x + (i % 3) * 300, y + (i // 3) * 110
        g.text((cx, cy), k, font=_font(20), fill=(90, 90, 90))
        g.text((cx, cy + 28), v, font=_font(34), fill=(10, 10, 10))
    code = render_barcode(encode(last, first, pnr, o, d, carrier, fno, day, seat=seat), scale=2).convert("RGB")
    code.thumbnail((440, 420))
    img.paste(code, (W - code.width - 30, 100))
    truth = {
        "doc_type": "boarding_pass", "passenger_first_name": first, "passenger_last_name": last,
        "booking_reference": pnr, "airline_code": carrier, "flight_number": fno,
        "origin_iata": o, "destination_iata": d, "departure_date": dep.date().isoformat(),
        "departure_time": dep.strftime("%H:%M"), "arrival_time": arr.strftime("%H:%M"),
        "disruption": "none",
    }
    return img, truth


def cancellation_notice(rng: random.Random) -> tuple[Image.Image, dict]:
    carrier, o, d, mins, cur = rng.choice(ROUTES)
    first, last = rng.choice(FIRST), rng.choice(LAST)
    pnr = _pnr(rng)
    fno = str(rng.randint(100, 2999))
    day = date(2026, 10, 1) + timedelta(days=rng.randint(0, 90))
    dep = datetime(day.year, day.month, day.day, rng.randint(5, 22), rng.choice([0, 30]))
    notice = dep - timedelta(hours=rng.choice([3, 10, 20, 50, 120, 400]))
    base, fuel, total = _fare(rng, cur, mins)
    reason = rng.choice(REASONS)
    email = f"{first.lower()}.{last.lower()}@example.com"
    lines = [
        (f"From: {carrier_name(carrier)} <noreply@{carrier_name(carrier).lower().replace(' ', '')}.example>", 20),
        (f"To: {email}", 20),
        (f"Sent: {notice.strftime('%d %B %Y %H:%M')}", 20),
        ("", 10),
        (f"Important: your flight {carrier} {fno} has been cancelled", 30),
        ("", 10),
        (f"Dear {first} {last},", 24),
        (f"We regret to inform you that flight {carrier} {fno} from {o} to {d},", 24),
        (f"scheduled to depart on {dep.strftime('%d %B %Y at %H:%M')}, has been cancelled due to {reason}.", 24),
        (f"Booking reference (PNR): {pnr}", 24),
        (f"Fare paid: {cur} {total:,}  (base fare {cur} {base:,}, fuel surcharge {cur} {fuel:,})", 24),
        ("You may request a refund or rebook through our customer service team.", 24),
    ]
    W, H = 1500, 60 + sum(s + 16 for _, s in lines)
    img = Image.new("RGB", (W, H), "white")
    g = ImageDraw.Draw(img)
    y = 30
    for text, size in lines:
        g.text((40, y), text, font=_font(size), fill=(20, 20, 20))
        y += size + 16
    truth = {
        "doc_type": "cancellation_notice", "passenger_first_name": first, "passenger_last_name": last,
        "booking_reference": pnr, "airline_code": carrier, "flight_number": fno,
        "origin_iata": o, "destination_iata": d, "departure_date": dep.date().isoformat(),
        "departure_time": dep.strftime("%H:%M"), "disruption": "cancellation",
        "notice_date": notice.date().isoformat(), "fare_total": str(total), "base_fare": str(base),
        "fuel_charge": str(fuel), "currency": cur, "email": email,
    }
    return img, truth


def generate(n: int, out: Path, seed: int = 7) -> None:
    rng = random.Random(seed)
    out.mkdir(parents=True, exist_ok=True)
    labels = {}
    for i in range(n):
        make = boarding_pass if i % 2 == 0 else cancellation_notice
        img, truth = make(rng)
        name = f"doc_{i:03d}.jpg"
        _degrade(img, rng).save(out / name, "JPEG")
        labels[name] = truth
    (out / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("evals/docbench/data"))
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    generate(a.n, a.out, a.seed)
    print(f"wrote {a.n} documents to {a.out}")
