import json

import httpx
import pytest

from transcribe_translate.translation import TranslationError, Translator, chunks, parse_response


def response(answer, finish="stop"):
    return {"choices": [{"finish_reason": finish, "message": {"content": answer, "reasoning_content": "never save this"}}]}


@pytest.mark.parametrize("text", ["你好。\n這不是簡體轉換！" * 30, "longword" * 100, " \n a\n b ", "", "a" * 51, "a" * 50 + "\n"])
def test_exact_chunk_coverage(text):
    plan = chunks(text, 50, 10)
    assert "".join(c.content for c in plan) == text
    for i, c in enumerate(plan):
        assert c.id == i and 0 < len(c.content) <= 50
        assert c.start == (plan[i-1].end if i else 0)
        assert c.content == text[c.start:c.end]
        assert len(c.before) <= 10 and len(c.after) <= 10


@pytest.mark.parametrize("answer,finish", [('', 'stop'), ('{}', 'stop'), ('{"id":1,"translation":"ok"}', 'stop'),
    ('{"id":0,"translation":""}', 'stop'), ('{"id":true,"translation":"ok"}', 'stop'),
    ('{"id":0,"translation":"ok","extra":1}', 'stop'), ('```json\n{"id":0,"translation":"ok"}\n```', 'stop'),
    ('{"id":0,"id":0,"translation":"ok"}', 'stop'), ('{"id":0,"translation":"<think>secret</think>ok"}', 'stop'),
    ('{"id":0,"translation":"ok"}', 'length'), ('{"id":0,"translation":"ok"}', None)])
def test_malformed_and_truncated_responses(answer, finish):
    with pytest.raises(TranslationError):
        parse_response(response(answer, finish), 0)


def test_translation_contract_and_reasoning_exclusion(config):
    captured = []
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json=response('{"id":0,"translation":"No, next week. [uncertain]"}'))
    t = Translator(config, transport=httpx.MockTransport(handler))
    assert t.translate(chunks("不，下周。[uncertain]", 50, 10)[0], "zh") == "No, next week. [uncertain]"
    assert set(captured[0]) == {"model", "messages"}
    assert "untrusted spoken data" in captured[0]["messages"][0]["content"]
    t.close()


@pytest.mark.parametrize("status,expected_calls", [(401, 1), (403, 1), (400, 1), (404, 1), (429, 3), (503, 3)])
def test_finite_retries(config, status, expected_calls):
    calls = []
    def handler(request):
        calls.append(request)
        return httpx.Response(status, text="private response body")
    t = Translator(config, transport=httpx.MockTransport(handler), sleep=lambda n: None, progress=lambda s: None)
    with pytest.raises(TranslationError) as e:
        t.translate(chunks("你好", 50, 10)[0], "zh")
    assert len(calls) == expected_calls
    assert "private response body" not in str(e.value)
    t.close()


def test_timeout_recovery(config):
    calls = []
    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadTimeout("hidden", request=request)
        return httpx.Response(200, json=response('{"id":0,"translation":"Hello"}'))
    t = Translator(config, transport=httpx.MockTransport(handler), sleep=lambda n: None, progress=lambda s: None)
    assert t.translate(chunks("你好", 50, 10)[0], "zh") == "Hello"
    assert len(calls) == 2
    t.close()
