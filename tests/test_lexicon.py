"""Tests for Lexicon CRUD endpoints."""


def test_list_lexicon(client):
    """GET /api/tts/lexicon returns active lexicon entries."""
    response = client.get("/api/tts/lexicon")
    assert response.status_code == 200
    data = response.json()
    assert "entries" in data
    assert "total" in data
    assert "Reactory" in data["entries"]


def test_lexicon_crud_flow(client):
    """Test full CRUD cycle for a custom pronunciation."""
    # 1. Create / update
    response = client.post(
        "/api/tts/lexicon",
        json={
            "word": "CockroachDB",
            "replacement": "cock-roach D B",
            "ipa": "kˈɑːkɹoʊtʃ dˌiːbˈiː"
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert data["word"] == "CockroachDB"
    assert data["replacement"] == "cock-roach D B"

    # 2. Get single
    response = client.get("/api/tts/lexicon/CockroachDB")
    assert response.status_code == 200
    assert response.json()["replacement"] == "cock-roach D B"

    # 3. List custom
    response = client.get("/api/tts/lexicon/custom")
    assert response.status_code == 200
    assert "CockroachDB" in response.json()["entries"]

    # 4. Delete
    response = client.delete("/api/tts/lexicon/CockroachDB")
    assert response.status_code == 200
    assert response.json()["status"] == "deleted"

    # 5. Verify deleted
    response = client.get("/api/tts/lexicon/CockroachDB")
    assert response.status_code == 404
