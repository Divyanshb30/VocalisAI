from datetime import date

import pytest

from vocalis.docintel.bcbp import BCBPError, decode, encode, read_image, render_barcode

# Example from the IATA BCBP implementation guide (mandatory items only).
IATA_SAMPLE = "M1DESMARAIS/LUC       EABC123 YULFRAAC 0834 326J001A0025 100"


def test_decode_iata_sample():
    bp = decode(IATA_SAMPLE)
    assert (bp.last_name, bp.first_name) == ("DESMARAIS", "LUC")
    assert bp.eticket
    [leg] = bp.legs
    assert (leg.pnr, leg.origin, leg.destination, leg.carrier) == ("ABC123", "YUL", "FRA", "AC")
    assert leg.flight_number == "834"
    assert leg.julian_day == 326
    assert leg.seat == "1A"


def test_encode_decode_roundtrip():
    s = encode("Rao", "Asha", "X7K2QB", "DEL", "BOM", "6E", "2135", date(2026, 10, 1), seat="14C")
    assert len(s) == 60
    bp = decode(s)
    leg = bp.legs[0]
    assert (bp.last_name, bp.first_name, leg.pnr) == ("RAO", "ASHA", "X7K2QB")
    assert (leg.origin, leg.destination, leg.carrier, leg.flight_number) == ("DEL", "BOM", "6E", "2135")
    assert leg.flight_date(date(2026, 9, 20)) == date(2026, 10, 1)
    assert leg.seat == "14C"


def test_julian_date_year_rollover():
    s = encode("A", "B", "PNR123", "LHR", "DXB", "EK", "2", date(2027, 1, 3))
    assert decode(s).legs[0].flight_date(date(2026, 12, 28)) == date(2027, 1, 3)


def test_rejects_non_bcbp():
    with pytest.raises(BCBPError):
        decode("hello world")


def test_barcode_image_roundtrip(tmp_path):
    s = encode("Rao", "Asha", "X7K2QB", "DEL", "BOM", "6E", "2135", date(2026, 10, 1))
    path = tmp_path / "bp.png"
    render_barcode(s).save(path)
    bp = read_image(path)
    assert bp is not None and bp.legs[0].pnr == "X7K2QB"
