from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from stage2_rag_assistant.eval.ragas_llm import FakeRagasLLM, build_ragas_llm
from stage2_rag_assistant.eval.eval_config import JudgeLLMConfig


class _Dummy(BaseModel):
    value: int


def test_scripted_responses_return_in_order(fake_ragas_llm):
    llm = fake_ragas_llm(responses=[_Dummy(value=1), _Dummy(value=2)])
    assert llm.generate("p1", _Dummy) == _Dummy(value=1)
    assert asyncio.run(llm.agenerate("p2", _Dummy)) == _Dummy(value=2)


def test_call_log_records_prompt_and_response_model(fake_ragas_llm):
    llm = fake_ragas_llm(responses=[_Dummy(value=1)])
    llm.generate("hello", _Dummy)
    assert llm.call_log == [{"prompt": "hello", "response_model": _Dummy}]


def test_exhaustion_raises_runtime_error(fake_ragas_llm):
    llm = fake_ragas_llm(responses=[_Dummy(value=1)])
    llm.generate("p", _Dummy)
    with pytest.raises(RuntimeError):
        llm.generate("p", _Dummy)


def test_responder_callable_is_used_instead_of_a_list(fake_ragas_llm):
    llm = fake_ragas_llm(responder=lambda prompt, model: model(value=len(prompt)))
    assert llm.generate("abc", _Dummy) == _Dummy(value=3)


def test_responses_and_responder_are_mutually_exclusive():
    with pytest.raises(ValueError):
        FakeRagasLLM(responses=[_Dummy(value=1)], responder=lambda p, m: m(value=0))


def test_factory_returns_fake_for_fake_provider():
    llm = build_ragas_llm(JudgeLLMConfig(provider="fake", model="x"))
    assert isinstance(llm, FakeRagasLLM)


def test_factory_raises_for_unimplemented_provider():
    with pytest.raises(NotImplementedError):
        build_ragas_llm(JudgeLLMConfig(provider="anthropic", model="x"))
