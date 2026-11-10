import hashlib
from decimal import Decimal

import pytest

from tests.media_fixtures import FIXTURE_DIR, MEDIA_FIXTURES


@pytest.mark.parametrize("fixture", MEDIA_FIXTURES, ids=lambda fixture: fixture.id)
def test_media_fixture_integrity(fixture):
    assert fixture.path.resolve().is_relative_to(FIXTURE_DIR.resolve())
    data = fixture.path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == fixture.sha256
    signatures = {
        "application/pdf": b"%PDF-",
        "image/png": b"\x89PNG\r\n\x1a\n",
        "image/jpeg": b"\xff\xd8\xff",
    }
    assert data.startswith(signatures[fixture.media_type])
    expected = fixture.expected_invoice()
    assert expected.items
    total = sum(Decimal(str(item.unit_price)) * item.quantity for item in expected.items)
    assert Decimal(str(expected.total)) == total


def test_media_fixture_ids_and_paths_are_unique():
    assert len({fixture.id for fixture in MEDIA_FIXTURES}) == len(MEDIA_FIXTURES)
    assert len({fixture.file for fixture in MEDIA_FIXTURES}) == len(MEDIA_FIXTURES)
