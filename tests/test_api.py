import json
import os
import pickle

import pytest
import torch

from chat_api import app


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test health check endpoint."""
    rv = client.get("/health")
    assert rv.status_code == 200
    assert b"ok" in rv.data


def test_static_files(client):
    """Test static file serving."""
    rv = client.get("/")
    assert rv.status_code == 200
    assert b"Void Z1" in rv.data


def test_chat_endpoint(client):
    """Test chat endpoint."""
    data = {"prompt": "Hello, how are you?", "max_new_tokens": 50, "temperature": 0.8}
    rv = client.post("/chat", data=json.dumps(data), content_type="application/json")

    assert rv.status_code == 200
    response = json.loads(rv.data)
    assert "text" in response
    assert len(response["text"]) > 0


def test_model_loading():
    """Test if model files can be loaded."""
    model_path = os.path.join(os.path.dirname(__file__), "out/model.pt")
    vocab_path = os.path.join(os.path.dirname(__file__), "data/void/vocab.pkl")
    meta_path = os.path.join(os.path.dirname(__file__), "data/void/meta.pkl")

    assert os.path.exists(model_path), "Model file missing"
    assert os.path.exists(vocab_path), "Vocabulary file missing"
    assert os.path.exists(meta_path), "Meta file missing"

    # Test loading vocab
    with open(vocab_path, "rb") as f:
        chars, stoi = pickle.load(f)
    assert len(chars) > 0
    assert len(stoi) > 0

    # Test loading meta
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    assert "vocab_size" in meta

    # Test loading model (if enough memory)
    try:
        model = torch.load(model_path, map_location="cpu")
        assert model is not None
    except Exception as e:
        pytest.skip(f"Skipping full model load test: {str(e)}")


def test_error_handling(client):
    """Test error handling."""
    # Test invalid JSON
    rv = client.post("/chat", data="invalid json", content_type="application/json")
    assert rv.status_code == 400

    # Test missing required field
    rv = client.post("/chat", data=json.dumps({}), content_type="application/json")
    assert rv.status_code == 400

    # Test invalid parameters
    rv = client.post(
        "/chat",
        data=json.dumps({"prompt": "test", "temperature": -1}),
        content_type="application/json",
    )
    assert rv.status_code == 400
