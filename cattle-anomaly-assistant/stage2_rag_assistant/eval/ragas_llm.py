"""Deterministic, network-free stand-in for a real Ragas judge LLM. Mirrors
stage2_rag_assistant/llm/providers/fake_provider.py's FakeLLMClient shape:
scripted responses or a responder callable, a call_log, RuntimeError on
exhaustion. Wired as eval_config.yaml's actual default (judge_llm.provider:
fake) — same "no real provider yet" policy as the main pipeline's
build_llm_client(), not just a test double.

Unlike FakeLLMClient (which answers one fixed JSON shape), each Layer-2
metric's internal multi-step pipeline asks the judge LLM for a different,
metric-specific pydantic response_model per call (confirmed by tracing
Faithfulness: it asks for StatementGeneratorOutput, then NLIStatementOutput
— two different models in one ascore() call). Hardcoding a per-metric
default would mean silently breaking on ragas's next internal refactor, so
the default responder instead introspects WHATEVER response_model it's
asked for and constructs a syntactically-valid instance generically,
recursing into nested models and list-of-model fields. This is what makes
run_eval.py's un-scripted demo-mode run actually produce a score instead
of erroring on every call — found the hard way: a bare FakeRagasLLM()
with no default here raised on the very first call, so every metric on
every case errored out silently until this was added.
"""

from __future__ import annotations

import typing
from collections.abc import Callable
from typing import TYPE_CHECKING, TypeVar

from pydantic import BaseModel

from stage2_rag_assistant.eval import _ragas_compat  # noqa: F401  (import for its side effect, before ragas)
from ragas.llms.base import InstructorBaseRagasLLM

if TYPE_CHECKING:
    from stage2_rag_assistant.eval.eval_config import JudgeLLMConfig

T = TypeVar("T", bound=BaseModel)
Responder = Callable[[str, type], BaseModel]


def auto_fill_response_model(model_cls: type[BaseModel]) -> BaseModel:
    """Constructs a syntactically-valid (semantically arbitrary) instance of
    whatever pydantic model is requested. Produces placeholder content, not
    a meaningful judgment — see this module's docstring and
    tests/eval/test_score_case.py's docstring for why that's an acceptable,
    clearly-labeled limitation of running without a real judge.
    """
    hints = typing.get_type_hints(model_cls)
    kwargs = {name: _auto_fill_value(hints.get(name, str)) for name in model_cls.model_fields}
    return model_cls(**kwargs)


def _auto_fill_value(annotation):
    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    if annotation is float:
        return 0.7
    if annotation is int:
        return 1
    if annotation is bool:
        return True
    if annotation is str:
        return "placeholder text"
    if origin is list:
        inner = args[0] if args else str
        return [_auto_fill_value(inner)]
    if origin is typing.Union:
        non_none = [a for a in args if a is not type(None)]
        return _auto_fill_value(non_none[0]) if non_none else None
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return auto_fill_response_model(annotation)
    return "placeholder"


def _auto_fill_responder(prompt: str, response_model: type[T]) -> T:  # noqa: ARG001 - prompt unused by design
    return auto_fill_response_model(response_model)  # type: ignore[return-value]


class FakeRagasLLM(InstructorBaseRagasLLM):
    def __init__(
        self,
        responses: list[BaseModel] | None = None,
        responder: Responder | None = None,
    ) -> None:
        if responses is not None and responder is not None:
            raise ValueError("pass at most one of responses, responder")
        self._responses = list(responses) if responses is not None else None
        self._responder = responder or (None if responses is not None else _auto_fill_responder)
        self.call_log: list[dict] = []

    def _next(self, prompt: str, response_model: type[T]) -> T:
        self.call_log.append({"prompt": prompt, "response_model": response_model})
        if self._responder is not None:
            return self._responder(prompt, response_model)  # type: ignore[return-value]
        if not self._responses:
            raise RuntimeError("FakeRagasLLM has no more scripted responses")
        return self._responses.pop(0)  # type: ignore[return-value]

    def generate(self, prompt: str, response_model: type[T]) -> T:
        return self._next(prompt, response_model)

    async def agenerate(self, prompt: str, response_model: type[T]) -> T:
        return self._next(prompt, response_model)


def build_ragas_llm(config: "JudgeLLMConfig") -> InstructorBaseRagasLLM:
    """Mirrors llm/client.py's build_llm_client() factory shape exactly —
    one branch point, 'fake' works today, anything else is deferred."""
    if config.provider == "fake":
        return FakeRagasLLM()
    raise NotImplementedError(
        f"Judge LLM provider {config.provider!r} is not implemented yet — see LLM_ASSISTANT_STATUS.md Open "
        "Question 2 (same deferred-provider policy as stage2_rag_assistant/llm/client.py). Set "
        "judge_llm.provider: fake in eval_config.yaml until a real provider is chosen and built."
    )
