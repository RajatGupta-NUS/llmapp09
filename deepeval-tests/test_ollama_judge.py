from pydantic import BaseModel

from conftest import OllamaJudgeModel


class JudgeResponse(BaseModel):
    score: int


def test_ollama_judge_sends_credentials_and_validates_schema(monkeypatch):
    request = {}

    class MockResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": '{"score": 1}'}}

    def mock_post(url, **kwargs):
        request["url"] = url
        request.update(kwargs)
        return MockResponse()

    monkeypatch.setattr("conftest.requests.post", mock_post)
    judge = OllamaJudgeModel(
        model="gemma4:31b",
        base_url="https://ollama.example/",
        api_key="test-key",
    )

    result = judge.generate("Evaluate this output", schema=JudgeResponse)

    assert result == JudgeResponse(score=1)
    assert request["url"] == "https://ollama.example/api/chat"
    assert request["headers"]["Authorization"].startswith("Bearer ")
    assert request["json"]["model"] == "gemma4:31b"
    assert request["json"]["format"] == JudgeResponse.model_json_schema()
