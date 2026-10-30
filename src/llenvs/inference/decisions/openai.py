"""OpenAI's synchronous Decisions API, with per-request durable receipts."""

import asyncio
import time
from copy import deepcopy

from llenvs.inference.backends.api import OpenAIBackend, _notify, _run_concurrent
from llenvs.inference.decisions.protocol import (
    DecisionBackend,
    DecisionCapabilities,
    DecisionResult,
    normalize_probabilities,
)
from llenvs.inference.protocol import RefusedByPolicyError
from llenvs.inference.records import normalize_usage, plain


class OpenAIDecisionBackend(DecisionBackend):
    def __init__(self, model="gpt-6-luna", **kwargs):
        from openai import AsyncOpenAI

        if not hasattr(AsyncOpenAI, "decisions"):
            raise ImportError(
                "OpenAI Decisions requires the decisions extra in a separate environment (SDK >=3.26)"
            )
        kwargs.setdefault("max_retries", 0)
        self._transport = OpenAIBackend(model=model, **kwargs)
        self._model = model
        self._client = self._transport._client
        self._async_client = self._transport._async_client

    @property
    def model_name(self):
        return self._model

    @property
    def capabilities(self):
        return DecisionCapabilities()

    def descriptor(self):
        descriptor = self._transport.descriptor()
        descriptor["identity"].update(
            backend="openai_decisions",
            protocol_version=1,
            endpoint_path="/v1/decisions",
            version_pinned=False,
        )
        return descriptor

    def _request_kwargs(self, request):
        cache = {}
        if request.cache_mode is not None:
            cache["prompt_cache_options"] = {"mode": request.cache_mode}
        if request.cache_key is not None:
            cache["prompt_cache_key"] = request.cache_key
        return {
            "model": self._model,
            "input": deepcopy(request.context),
            "questions": [
                {
                    "type": "choice",
                    "name": "decision",
                    "instructions": request.question,
                    "choices": [
                        {
                            "value": option.value,
                            **(
                                {"description": option.description}
                                if option.description is not None
                                else {}
                            ),
                        }
                        for option in request.options
                    ],
                }
            ],
        } | cache

    def reparse_receipt(self, receipt):
        request, response = receipt["request"], receipt["response"]
        answers = response.get("answers", [])
        if len(answers) != 1 or answers[0].get("name") != "decision":
            raise ValueError("Decisions response must contain exactly the named answer")
        answer = answers[0]
        if answer.get("type") == "refusal":
            raise RefusedByPolicyError("Decision model refused the question")
        if answer.get("type") != "choice" or answer.get("error"):
            raise ValueError("Decisions response is not a successful choice")
        probabilities, normalization = normalize_probabilities(
            request, [(entry["value"], entry["probability"]) for entry in answer["probabilities"]]
        )
        return DecisionResult(
            probabilities,
            tuple(option.value for option in request.options),
            response["model"],
            deepcopy(response),
            **normalize_usage(response.get("usage")),
            metadata={
                **receipt.get("metadata", {}),
                **normalization,
                "raw_usage": deepcopy(response.get("usage")),
            },
        )

    async def _decide(self, index, request, observer):
        kwargs = self._request_kwargs(request)
        attempts, start = [], time.monotonic()
        while True:
            await _notify(observer, "started", index, len(attempts))
            attempt_start = time.monotonic()
            try:
                response = await self._async_client.decisions.create(**kwargs)
            except Exception as exc:
                await asyncio.sleep(
                    await self._transport._failed_attempt_async(
                        exc, attempt_start, attempts, start, observer, index
                    )
                )
                continue
            raw = plain(response)
            attempts.append(
                {
                    "status_code": 200,
                    "request_id": getattr(response, "_request_id", None),
                    "usage": raw.get("usage"),
                    "elapsed_seconds": time.monotonic() - attempt_start,
                }
            )
            receipt = {
                "request": request,
                "response": raw,
                "metadata": {
                    "attempts": attempts,
                    "elapsed_seconds": time.monotonic() - start,
                    "request": {
                        "model": self._model,
                        "question_type": "choice",
                        "option_count": len(request.options),
                    },
                    "endpoint": "/v1/decisions",
                    "version_pinned": False,
                },
            }
            await _notify(observer, "received", index, receipt)
            try:
                result = self.reparse_receipt(receipt)
            except Exception as exc:
                # A response that cannot be read, a refusal among them, was billed.
                setattr(exc, "attempts", deepcopy(attempts))
                await _notify(observer, "failed", index, exc)
                raise
            await _notify(observer, "completed", index, result)
            return result

    def decide_batch(self, requests, *, observer=None):
        for request in requests:
            self._request_kwargs(request)
        return _run_concurrent(
            lambda item: self._decide(item[0], item[1], observer),
            list(enumerate(requests)),
            self._transport._max_concurrency,
            return_partial=True,
            runner=self._transport._async_runner,
        )

    def close(self):
        self._transport.close()
