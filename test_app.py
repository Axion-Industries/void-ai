import pytest
from chat_api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_root_endpoint(client):
    rv = client.get('/')
    assert rv.status_code == 200

def test_chat_endpoint_no_prompt(client):
    rv = client.post('/chat', json={})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'response' in json_data
    assert json_data['response'] == 'No prompt provided.'

def test_chat_endpoint_with_prompt(client):
    rv = client.post('/chat', json={'prompt': 'Hello'})
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'response' in json_data
    assert isinstance(json_data['response'], str)

def test_static_files(client):
    rv = client.get('/index.html')
    assert rv.status_code == 200
    assert b'<!DOCTYPE html>' in rv.data
