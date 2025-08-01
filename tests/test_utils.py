from utils.helpers import parse_price


def test_parse_price_valid():
    assert parse_price("$123") == 123


def test_parse_price_invalid():
    assert parse_price("unavailable") is None
