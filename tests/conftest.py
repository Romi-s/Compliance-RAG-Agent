import pytest


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    """Keep the test suite hermetic: never run real corpus seeding (which would
    call OpenAI and spend the key now present in a local .env)."""
    monkeypatch.setattr("app.api.routes.ensure_seeded", lambda: None)
