"""A durable receipt is reusable; an unobserved remote outcome is not."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from llenvs.inference.persistence import (
    CallStore,
    CheckpointCorruptionError,
    CheckpointInUseError,
    PersistentBackend,
    UnknownOutcomeError,
)
from llenvs.inference.protocol import (
    BackendCapabilities,
    ChatMessage,
    GenerationResult,
    SamplingParams,
    ScoringResult,
)


class Backend:
    capabilities = BackendCapabilities(supports_request_progress=True)
    model_name = "offline"
    calls = 0

    def descriptor(self):
        return {
            "identity": {"model": self.model_name, "precision": "fp32"},
            "provenance": {"port": 1234},
        }

    def generate_chat_batch(self, batch, params, *, observer=None):
        results = []
        for index, messages in enumerate(batch):
            observer.started(index, 0)
            self.calls += 1
            receipt = {"text": messages[0].content, "draw": self.calls}
            observer.received(index, receipt)
            result = self.reparse_receipt(receipt, params)
            observer.completed(index, result)
            results.append(result)
        return results

    def reparse_receipt(self, receipt, params):
        return GenerationResult(text=receipt["text"], metadata={"draw": receipt["draw"]})


def messages(text):
    return [ChatMessage("user", text)]


def native_result(result):
    return replace(
        result, metadata={k: v for k, v in result.metadata.items() if k != "persistence"}
    )


def test_result_identifies_draw_and_replay_without_changing_usage_or_stored_result(tmp_path):
    backend = Backend()
    native = GenerationResult(
        text="answer",
        prompt_tokens=100,
        completion_tokens=20,
        cached_prompt_tokens=60,
        cache_write_prompt_tokens=30,
        reasoning_tokens=10,
        metadata={"raw_usage": {"input": 100}, "request": {"model": "offline"}},
    )
    backend.reparse_receipt = lambda *args: native
    with CallStore.open(tmp_path, run_descriptor={"run": "accounting"}) as store:
        wrapped = PersistentBackend(backend, store)
        first = wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id="a")
        key = first.metadata["persistence"]["call_key"]
        fingerprint = store.fingerprint
        assert first.metadata["persistence"] == {
            "run_fingerprint": fingerprint,
            "call_key": key,
            "source": "backend",
            "previous_attempts": 0,
        }
        assert store._records[key]["result"] == native
        assert "persistence" not in native.metadata
        assert native_result(first) == native
        first.metadata["persistence"]["source"] = "tampered"
        first.metadata["raw_usage"]["input"] = 999
        replay = wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id="a")
        assert replay.metadata["persistence"] == {
            "run_fingerprint": fingerprint,
            "call_key": key,
            "source": "store",
            "previous_attempts": 0,
        }
        assert native_result(replay) == native
        assert store._records[key]["result"] == native
    with CallStore.open(tmp_path, run_descriptor={"run": "accounting"}) as store:
        reopened = PersistentBackend(backend, store).generate_chat(
            messages("a"), SamplingParams(), repeat_id="a"
        )
        assert reopened == replay
        assert backend.calls == 1


def test_mixed_batch_provenance_is_per_result_and_duplicate_positions_share_a_key(tmp_path):
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        backend = Backend()
        wrapped = PersistentBackend(backend, store)
        first = wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id="a")
        results = wrapped.generate_chat_batch(
            [messages("a"), messages("b"), messages("b")],
            SamplingParams(),
            repeat_ids=["a", "b", "b"],
        )
        provenance = [result.metadata["persistence"] for result in results]
        assert [item["source"] for item in provenance] == ["store", "backend", "shared"]
        assert provenance[0]["call_key"] == first.metadata["persistence"]["call_key"]
        assert provenance[1]["call_key"] == provenance[2]["call_key"] != provenance[0]["call_key"]
        assert native_result(results[1]) == native_result(results[2])
        assert backend.calls == 2


def test_decisions_have_the_same_store_identity_and_replay_provenance(tmp_path):
    from llenvs.inference.decisions import DecisionOption, DecisionRequest, DecisionResult

    class Decisions(Backend):
        def decide_batch(self, requests, *, observer):
            results = []
            for index, request in enumerate(requests):
                observer.started(index, 0)
                self.calls += 1
                result = DecisionResult(
                    probabilities=(0.25, 0.75),
                    labels=("a", "b"),
                    model="offline",
                    raw_response={"probabilities": [0.25, 0.75]},
                    prompt_tokens=12,
                )
                observer.completed(index, result)
                results.append(result)
            return results

    request = DecisionRequest("context", "question", (DecisionOption("a"), DecisionOption("b")))
    backend = Decisions()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store)
        first = wrapped.decide_batch([request], repeat_ids=["a"])[0]
        replay = wrapped.decide_batch([request], repeat_ids=["a"])[0]
        assert first.metadata["persistence"]["source"] == "backend"
        assert replay.metadata["persistence"] == {
            **first.metadata["persistence"],
            "source": "store",
        }
        assert native_result(first) == native_result(replay)
        assert replay.prompt_tokens == 12 and backend.calls == 1


@pytest.mark.parametrize("operation", ["completion", "tools", "prefix", "chat_prefix", "score"])
def test_other_persistent_operations_return_provenance(tmp_path, operation):
    backend = Backend()
    outcome = (
        ScoringResult((), prompt_tokens=2)
        if operation == "score"
        else GenerationResult(text="answer")
    )
    calls = []

    def execute(*args):
        calls.append(args)
        return [outcome]

    backend.generate = backend.generate_with_tools_batch = execute
    backend.continue_from_prefix = backend.generate_chat_with_prefix_batch = execute
    backend.score_chat_batch = execute
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store, checkpoint_granularity="batch")
        methods = {
            "completion": lambda: wrapped.generate_single(
                "prompt", SamplingParams(), repeat_id="a"
            ),
            "tools": lambda: wrapped.generate_with_tools(
                messages("a"), [], SamplingParams(), repeat_id="a"
            ),
            "prefix": lambda: wrapped.continue_from_prefix(
                "prefix", SamplingParams(), repeat_ids=["a"]
            )[0],
            "chat_prefix": lambda: wrapped.generate_chat_with_prefix(
                messages("a"), "prefix", SamplingParams(), repeat_id="a"
            ),
            "score": lambda: wrapped.score_chat(messages("a"), "answer", repeat_id="a"),
        }
        first = methods[operation]()
        replay = methods[operation]()
        assert first.metadata["persistence"]["source"] == "backend"
        assert replay.metadata["persistence"] == {
            **first.metadata["persistence"],
            "source": "store",
        }
        assert native_result(first) == native_result(replay) == outcome
        assert "persistence" not in outcome.metadata
        assert len(calls) == 1


def test_successful_partial_batch_slots_keep_provenance_and_usage(tmp_path):
    from llenvs.inference.protocol import PartialBatchError

    backend = Backend()
    outcome = GenerationResult(text="answer", prompt_tokens=5, completion_tokens=2)

    def execute(batch, params, *, observer):
        observer.started(0, 0)
        observer.completed(0, outcome)
        observer.started(1, 0)
        error = ValueError("rejected")
        error.status_code = 400
        raise PartialBatchError([outcome, error], {1: error})

    backend.generate_chat_batch = execute
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store)
        with pytest.raises(PartialBatchError) as caught:
            wrapped.generate_chat_batch(
                [messages("a"), messages("b")], SamplingParams(), repeat_ids=["a", "b"]
            )
        first = caught.value.results[0]
        assert first.metadata["persistence"]["source"] == "backend"
        assert native_result(first) == outcome
        replay = wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id="a")
        assert replay.metadata["persistence"] == {
            **first.metadata["persistence"],
            "source": "store",
        }
        assert native_result(replay) == outcome


@pytest.mark.parametrize("kind", ["sdk", "malformed", "oserror"])
def test_failed_positions_have_detached_identity_source_and_replay_metadata(tmp_path, kind):
    from llenvs.inference.protocol import MalformedResponseError, PartialBatchError

    if kind == "sdk":
        import httpx
        from openai import BadRequestError

        error = BadRequestError(
            "rejected",
            response=httpx.Response(400, request=httpx.Request("POST", "https://offline.invalid")),
            body={"error": "rejected"},
        )
    elif kind == "malformed":
        error = MalformedResponseError("rejected", provider_error={"code": 400})
    else:
        error = OSError(5, "rejected", "offline")
        error.status_code = 400
    error.metadata = {"provider": "offline"}
    backend = Backend()

    def execute(batch, params, *, observer):
        backend.calls += len(batch)
        for index in range(len(batch)):
            observer.started(index, 0)
        raise PartialBatchError([error] * len(batch), dict.fromkeys(range(len(batch)), error))

    backend.generate_chat_batch = execute
    batch = [messages("a"), messages("b"), messages("b")]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store)
        with pytest.raises(PartialBatchError) as caught:
            wrapped.generate_chat_batch(batch, SamplingParams(), repeat_ids=["a", "b", "b"])
        failures = caught.value.failures
        records = [failure.metadata["persistence"] for failure in failures.values()]
        assert [record["source"] for record in records] == ["backend", "backend", "shared"]
        assert records[0]["call_key"] != records[1]["call_key"] == records[2]["call_key"]
        assert all(record["run_fingerprint"] == store.fingerprint for record in records)
        assert all(record["previous_attempts"] == 0 for record in records)
        assert all(
            type(failure) is type(error) and failure.status_code == 400
            for failure in failures.values()
        )
        assert all(caught.value.results[i] is failure for i, failure in failures.items())
        assert len({id(failure) for failure in failures.values()}) == 3
        assert error.metadata == {"provider": "offline"}
        if kind == "sdk":
            assert failures[0].response is error.response and failures[0].body == error.body
        if kind == "oserror":
            assert failures[0].filename == "offline" and failures[0].errno == 5
        failures[1].metadata["persistence"]["source"] = "tampered"
        assert failures[2].metadata["persistence"]["source"] == "shared"
        keys = [record["call_key"] for record in records]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        with pytest.raises(PartialBatchError) as caught:
            PersistentBackend(backend, store).generate_chat_batch(
                batch, SamplingParams(), repeat_ids=["a", "b", "b"]
            )
        for index, failure in caught.value.failures.items():
            assert failure.metadata["persistence"] == {
                "run_fingerprint": store.fingerprint,
                "call_key": keys[index],
                "source": "store",
                "previous_attempts": 0,
            }
            assert failure.status_code == 400
        assert backend.calls == 2


def test_attempt_count_is_history_at_entry_not_retries_during_the_invocation(tmp_path):
    from llenvs.inference.protocol import PartialBatchError

    backend = Backend()
    original = backend.generate_chat_batch
    error = ValueError("retry budget exhausted")
    error.status_code = 503
    error.attempts = [{"status_code": 503}] * 3

    def execute(batch, params, *, observer):
        for attempt in range(3):
            observer.started(0, attempt)
            backend.calls += 1
            observer.failed(0, error)
        raise error

    backend.generate_chat_batch = execute
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store)
        with pytest.raises(PartialBatchError) as caught:
            wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id="a")
        failure = caught.value.failures[0]
        provenance = failure.metadata["persistence"]
        assert provenance["previous_attempts"] == 0
        assert failure.attempts == error.attempts
        key = provenance["call_key"]
        with pytest.raises(PartialBatchError) as replayed:
            wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id="a")
        assert replayed.value.failures[0].metadata["persistence"] == {
            **provenance,
            "source": "store",
            "previous_attempts": 2,
        }
        assert backend.calls == 3
        store.acknowledge_resubmission(
            key, reason="Retry definite failure", accept_duplicate_expenditure=True
        )
        backend.generate_chat_batch = original
        result = wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id="a")
        assert result.metadata["persistence"] == {**provenance, "previous_attempts": 3}
        assert wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id="a").metadata[
            "persistence"
        ] == {
            **result.metadata["persistence"],
            "source": "store",
        }
        assert backend.calls == 4


def test_successful_retry_and_shared_position_keep_their_entry_history_count(tmp_path):
    backend = Backend()
    original = backend.generate_chat_batch
    error = ValueError("retryable")
    error.status_code = 503

    def execute(batch, params, *, observer):
        for attempt in range(2):
            observer.started(0, attempt)
            observer.failed(0, error)
        return original(batch, params, observer=observer)

    backend.generate_chat_batch = execute
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store)
        results = wrapped.generate_chat_batch(
            [messages("a"), messages("a")], SamplingParams(), repeat_ids=["a", "a"]
        )
        records = [result.metadata["persistence"] for result in results]
        assert [record["source"] for record in records] == ["backend", "shared"]
        assert [record["previous_attempts"] for record in records] == [0, 0]
        replay = wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id="a")
        assert replay.metadata["persistence"] == {
            **records[0],
            "source": "store",
            "previous_attempts": 2,
        }


def test_repeated_parser_errors_identify_receipt_reuse_without_another_attempt(tmp_path):
    from llenvs.inference.protocol import PartialBatchError

    backend = Backend()
    backend.reparse_receipt = lambda *args: (_ for _ in ()).throw(ValueError("parser failed"))
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store)
        for source in ("backend", "receipt", "receipt"):
            with pytest.raises(PartialBatchError) as caught:
                wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id="a")
            record = caught.value.failures[0].metadata["persistence"]
            assert record["source"] == source and record["previous_attempts"] == 0
            assert store._records[record["call_key"]]["state"] == "received"
        assert backend.calls == 1


def test_explicit_draws_survive_regrouping_subsets_and_duplicate_positions(tmp_path):
    backend = Backend()
    with CallStore.open(tmp_path, run_descriptor={"run": "test"}) as store:
        wrapped = PersistentBackend(backend, store)
        results = wrapped.generate_chat_batch(
            [messages("a"), messages("b"), messages("a")],
            SamplingParams(),
            repeat_ids=["a", "b", "a"],
        )
        assert backend.calls == 2
        assert native_result(results[0]) == native_result(results[2])
    with CallStore.open(tmp_path, run_descriptor={"run": "test"}) as store:
        backend.descriptor = lambda: {
            "identity": {"model": "offline", "precision": "fp32"},
            "provenance": {"port": 9999},
        }
        wrapped = PersistentBackend(backend, store)
        assert native_result(
            wrapped.generate_chat(messages("b"), SamplingParams(), repeat_id="b")
        ) == native_result(results[1])
        assert backend.calls == 2
        assert (
            wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id="other") != results[0]
        )


def test_receipt_is_reparsed_locally_after_parser_failure(tmp_path):
    backend = Backend()
    parse = backend.reparse_receipt
    backend.reparse_receipt = lambda *args: (_ for _ in ()).throw(ValueError("parser interrupted"))
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store)
        with pytest.raises(Exception):
            wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
    backend.reparse_receipt = parse
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        result = PersistentBackend(backend, store).generate_chat(
            messages("a"), SamplingParams(), repeat_id=1
        )
        assert result.text == "a" and backend.calls == 1
        provenance = result.metadata["persistence"]
        assert provenance["source"] == "receipt"
        assert store._records[provenance["call_key"]]["state"] == "completed"
        assert provenance["run_fingerprint"] == store.fingerprint


@pytest.mark.parametrize("retries", [0, 2])
def test_started_without_receipt_stops_resubmission(tmp_path, retries):
    backend = Backend()

    def interrupted(batch, params, *, observer):
        for attempt in range(retries + 1):
            observer.started(0, attempt)
        raise TimeoutError("remote execution may have occurred")

    backend.generate_chat_batch = interrupted
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        with pytest.raises(UnknownOutcomeError) as caught:
            PersistentBackend(backend, store).generate_chat(
                messages("a"), SamplingParams(), repeat_id=1
            )
        provenance = caught.value.metadata["persistence"]
        assert provenance == {
            "run_fingerprint": store.fingerprint,
            "call_key": caught.value.key,
            "source": "backend",
            "previous_attempts": 0,
        }
        assert store._inflight[caught.value.key].done()
        with pytest.raises(UnknownOutcomeError) as replayed:
            PersistentBackend(backend, store).generate_chat(
                messages("a"), SamplingParams(), repeat_id=1
            )
        assert replayed.value.metadata["persistence"] == {
            **provenance,
            "source": "store",
            "previous_attempts": retries,
        }
    backend.generate_chat_batch = lambda *a, **kw: pytest.fail("resubmitted unknown outcome")
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        with pytest.raises(UnknownOutcomeError) as caught:
            PersistentBackend(backend, store).generate_chat(
                messages("a"), SamplingParams(), repeat_id=1
            )
        assert caught.value.metadata["persistence"] == {
            **provenance,
            "source": "store",
            "previous_attempts": retries,
        }


def test_unknown_phase_error_identifies_the_child_that_requires_recovery(tmp_path):
    class Phased(Backend):
        def generate_chat_batch(self, batch, params, *, observer):
            observer.begin_phases(0)

            def execute(child):
                child.started(0, 0)
                self.calls += 1
                raise TimeoutError("unknown phase outcome")

            observer.run_phase(0, "reasoning", {"ids": [1]}, execute, lambda receipt: receipt)

    backend = Phased()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store)
        with pytest.raises(UnknownOutcomeError) as caught:
            wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
        failure = caught.value
        provenance = failure.metadata["persistence"]
        child = store._records[failure.key]
        assert child["request"]["phase"] == "reasoning"
        assert provenance == {
            "run_fingerprint": store.fingerprint,
            "call_key": failure.key,
            "source": "backend",
            "previous_attempts": 0,
        }
        parent = child["request"]["parent"]
        assert parent not in store._inflight or store._inflight[parent].done()
        with pytest.raises(UnknownOutcomeError) as replayed:
            wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
        assert replayed.value.metadata["persistence"] == {**provenance, "source": "store"}
        assert backend.calls == 1
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        with pytest.raises(UnknownOutcomeError) as replayed:
            PersistentBackend(backend, store).generate_chat(
                messages("a"), SamplingParams(), repeat_id=1
            )
        assert replayed.value.metadata["persistence"] == {**provenance, "source": "store"}
        assert backend.calls == 1


def test_distinct_calls_run_concurrently_and_inflight_duplicate_coalesces(tmp_path):
    backend = Backend()
    barrier = threading.Barrier(2)
    original = backend.generate_chat_batch

    def concurrent(*args, **kwargs):
        barrier.wait(timeout=5)
        return original(*args, **kwargs)

    backend.generate_chat_batch = concurrent
    with CallStore.open(tmp_path, run_descriptor={}) as store, ThreadPoolExecutor(3) as pool:
        wrapped = PersistentBackend(backend, store)
        first = pool.submit(wrapped.generate_chat, messages("a"), SamplingParams(), repeat_id=1)
        duplicate = pool.submit(wrapped.generate_chat, messages("a"), SamplingParams(), repeat_id=1)
        second = pool.submit(wrapped.generate_chat, messages("b"), SamplingParams(), repeat_id=1)
        original_result = first.result(timeout=10)
        shared_result = duplicate.result(timeout=10)
        assert native_result(original_result) == native_result(shared_result)
        assert (
            original_result.metadata["persistence"]["call_key"]
            == shared_result.metadata["persistence"]["call_key"]
        )
        assert {
            original_result.metadata["persistence"]["source"],
            shared_result.metadata["persistence"]["source"],
        } == {"backend", "shared"}
        assert second.result(timeout=10).text == "b"
        assert backend.calls == 2


def test_exclusive_run_lock(tmp_path):
    with CallStore.open(tmp_path, run_descriptor={}):
        with pytest.raises(CheckpointInUseError):
            CallStore.open(tmp_path, run_descriptor={})


def test_corruption_fails_on_open_before_any_delegate(tmp_path):
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        PersistentBackend(Backend(), store).generate_chat(
            messages("a"), SamplingParams(), repeat_id=1
        )
        record = next((store.run_dir / "calls").rglob("*.pkl"))
    record.write_bytes(b"broken")
    with pytest.raises(CheckpointCorruptionError):
        CallStore.open(tmp_path, run_descriptor={})


def test_lazy_open_skips_call_payloads_but_validates_a_replayed_record(tmp_path, monkeypatch):
    import llenvs.inference.persistence.store as mod

    backend = Backend()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        PersistentBackend(backend, store).generate_chat(
            messages("a"), SamplingParams(), repeat_id=1
        )
    load = mod.CallStore._load
    reads = []

    def tracked(store, path):
        reads.append(path)
        return load(store, path)

    monkeypatch.setattr(mod.CallStore, "_load", tracked)
    with CallStore.open(tmp_path, run_descriptor={}, validation="lazy") as store:
        assert reads == []
        replay = PersistentBackend(backend, store).generate_chat(
            messages("a"), SamplingParams(), repeat_id=1
        )
        assert replay.text == "a" and len(reads) == 1
        assert replay.metadata["persistence"]["source"] == "store"
        assert backend.calls == 1


def test_lazy_corruption_fails_on_access_and_prevents_further_dispatch(tmp_path):
    from llenvs.inference.persistence import CheckpointError

    backend = Backend()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        PersistentBackend(backend, store).generate_chat(
            messages("a"), SamplingParams(), repeat_id=1
        )
        path = next((store.run_dir / "calls").rglob("*.pkl"))
    path.write_bytes(b"corrupt")
    with CallStore.open(tmp_path, run_descriptor={}, validation="lazy") as store:
        wrapped = PersistentBackend(backend, store)
        with pytest.raises(CheckpointCorruptionError):
            wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
        with pytest.raises(CheckpointError, match="poisoned"):
            wrapped.generate_chat(messages("b"), SamplingParams(), repeat_id=2)
        assert backend.calls == 1
    with pytest.raises(CheckpointCorruptionError):
        CallStore.open(tmp_path, run_descriptor={})


@pytest.mark.parametrize("state", ["completed", "received", "started", "failed"])
def test_lazy_reopen_preserves_recovery_without_reissuing_the_call(tmp_path, state):
    from llenvs.inference.protocol import PartialBatchError

    backend = Backend()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        result = PersistentBackend(backend, store).generate_chat(
            messages("a"), SamplingParams(), repeat_id=1
        )
        key = result.metadata["persistence"]["call_key"]
        store.update(
            key, state, error={"type": "ValueError", "message": "rejected", "status_code": 503}
        )
    with CallStore.open(tmp_path, run_descriptor={}, validation="lazy") as store:
        wrapped = PersistentBackend(backend, store)
        if state == "started":
            with pytest.raises(UnknownOutcomeError):
                wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
        elif state == "failed":
            with pytest.raises(PartialBatchError) as caught:
                wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
            assert caught.value.failures[0].metadata["persistence"]["source"] == "store"
        else:
            replay = wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
            assert replay.text == "a"
            assert replay.metadata["persistence"]["source"] == (
                "store" if state == "completed" else "receipt"
            )
        assert backend.calls == 1


def test_invalid_validation_mode_fails_before_creating_a_store(tmp_path):
    with pytest.raises(ValueError, match="validation"):
        CallStore.open(tmp_path, run_descriptor={}, validation="skip")
    assert list(tmp_path.iterdir()) == []


def test_default_request_granularity_rejects_native_batch_only_backend(tmp_path):
    backend = Backend()
    backend.capabilities = BackendCapabilities()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        with pytest.raises(ValueError, match="granularity"):
            PersistentBackend(backend, store)


def test_provider_error_is_not_cached_success(tmp_path):
    backend = Backend()
    backend.reparse_receipt = lambda *args: GenerationResult(
        text="", metadata={"provider_error": {"message": "bad"}}
    )
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        with pytest.raises(Exception):
            PersistentBackend(backend, store).generate_chat(
                messages("a"), SamplingParams(), repeat_id=1
            )


@pytest.mark.parametrize("phase_state", ["completed", "received"])
def test_completed_reasoning_phase_is_reused_after_phase_gap(tmp_path, phase_state):
    from llenvs.inference.controlled import TokenPhase

    class Phased(Backend):
        phase_calls = []
        interrupt = True

        def reparse_token_receipt(self, receipt):
            return receipt

        def generate_chat_batch(self, batch, params, *, observer):
            observer.begin_phases(0)
            outcomes = []
            for name in ["reasoning", "answer"]:
                if name == "answer" and self.interrupt:
                    raise RuntimeError("interrupted between phases")

                def execute(child, name=name):
                    child.started(0, 0)
                    self.phase_calls.append(name)
                    phase = TokenPhase((1,), GenerationResult(text=name))
                    child.received(0, phase)
                    if name == "reasoning" and phase_state == "received" and self.interrupt:
                        raise ValueError("interrupted phase parser")
                    child.completed(0, phase)
                    return phase

                outcomes.append(
                    observer.run_phase(0, name, {"ids": [1]}, execute, self.reparse_token_receipt)
                )
            result = GenerationResult(
                text=outcomes[-1].result.text,
                metadata={"phase_results": [phase.result for phase in outcomes]},
            )
            observer.completed(0, result)
            return [result]

    backend = Phased()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        with pytest.raises(Exception):
            PersistentBackend(backend, store).generate_chat(
                messages("a"), SamplingParams(), repeat_id=1
            )
    backend.interrupt = False
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        result = PersistentBackend(backend, store).generate_chat(
            messages("a"), SamplingParams(), repeat_id=1
        )
    assert result.text == "answer"
    assert backend.phase_calls == ["reasoning", "answer"]
    phases = result.metadata["phase_results"]
    assert [phase.metadata["persistence"]["source"] for phase in phases] == [
        "store" if phase_state == "completed" else "receipt",
        "backend",
    ]
    assert len({phase.metadata["persistence"]["call_key"] for phase in phases}) == 2


def _crash_worker(root, mode, connection, release):
    class CrashBackend(Backend):
        def generate_chat_batch(self, batch, params, *, observer):
            started = threading.Barrier(2)

            def one(index):
                observer.started(index, 0)
                started.wait(timeout=10)
                if index == 1:
                    release.wait()
                    return
                receipt = {"text": "durable", "draw": 1}
                observer.received(index, receipt)
                if mode == "completed":
                    observer.completed(index, self.reparse_receipt(receipt, params))
                connection.send("durable")
                release.wait()

            with ThreadPoolExecutor(2) as pool:
                list(pool.map(one, range(len(batch))))

    with CallStore.open(root, run_descriptor={}) as store:
        PersistentBackend(CrashBackend(), store).generate_chat_batch(
            [messages("a"), messages("b")], SamplingParams(), repeat_ids=[1, 2]
        )


@pytest.mark.parametrize("mode", ["received", "completed"])
def test_process_death_retains_individual_receipt_while_sibling_is_blocked(tmp_path, mode):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    release = context.Event()
    process = context.Process(target=_crash_worker, args=(str(tmp_path), mode, child, release))
    process.start()
    try:
        assert parent.poll(15), "Child did not acknowledge the fsynced receipt"
        assert parent.recv() == "durable"
        process.kill()
        process.join(timeout=10)
        assert not process.is_alive()
        backend = Backend()
        with CallStore.open(tmp_path, run_descriptor={}) as store:
            wrapped = PersistentBackend(backend, store)
            assert (
                wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1).text
                == "durable"
            )
            assert backend.calls == 0
            with pytest.raises(UnknownOutcomeError):
                wrapped.generate_chat(messages("b"), SamplingParams(), repeat_id=2)
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        parent.close()
        child.close()


def test_storage_failure_is_fatal_and_poisoned_store_dispatches_nothing(tmp_path, monkeypatch):
    import llenvs.inference.persistence.store as mod
    from llenvs.inference.persistence import CheckpointError

    backend = Backend()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store)
        monkeypatch.setattr(
            mod, "_fsync_directory", lambda path: (_ for _ in ()).throw(OSError("disk"))
        )
        with pytest.raises(CheckpointError):
            wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
        with pytest.raises(CheckpointError, match="poisoned"):
            wrapped.generate_chat(messages("b"), SamplingParams(), repeat_id=2)
        assert backend.calls == 0


def test_recovery_decision_clears_current_attempt_but_retains_diagnostics(tmp_path):
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        key, _, _ = store.reserve({"operation": "test"}, 1)
        store.update(
            key, "started", receipt=None, error={"message": "timeout"}, outcome_unknown=True
        )
        with pytest.raises(ValueError):
            store.acknowledge_resubmission(key, reason="", accept_duplicate_expenditure=True)
        store.acknowledge_resubmission(
            key, reason="provider investigation", accept_duplicate_expenditure=True
        )
        record = store._records[key]
        assert record["state"] == "reserved"
        assert record.get("error") is None and record.get("outcome_unknown") is False
        assert record["attempts"][0]["error"]["message"] == "timeout"


def test_identity_scope_parameters_and_implicit_occurrences_separate_draws(tmp_path):
    backend = Backend()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        first = PersistentBackend(backend, store, scope="first")
        second = PersistentBackend(backend, store, scope="second")
        a = first.generate_chat(messages("a"), SamplingParams(), repeat_id="same")
        assert native_result(
            first.generate_chat(messages("a"), SamplingParams(), repeat_id="same")
        ) == native_result(a)
        other_scope = second.generate_chat(messages("a"), SamplingParams(), repeat_id="same")
        assert (
            other_scope.metadata["persistence"]["call_key"] != a.metadata["persistence"]["call_key"]
        )
        first.generate_chat(messages("a"), SamplingParams(temperature=0.2), repeat_id="same")
        first.generate_chat(messages("a"), SamplingParams())
        first.generate_chat(messages("a"), SamplingParams())
        backend.model_name = "another-model"
        first.generate_chat(messages("a"), SamplingParams(), repeat_id="same")
        assert backend.calls == 6


def test_backend_close_records_separate_lifecycle_and_is_idempotent(tmp_path):
    backend = Backend()
    closed = []
    backend.close = lambda: closed.append(True)
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store)
        wrapped.close()
        wrapped.close()
        assert closed == [True]
        assert len(list((store.run_dir / "lifecycle").glob("*.pkl"))) == 2
        store.mark_complete()
        assert (store.run_dir / "manifest.json").exists()
        lifecycle = next((store.run_dir / "lifecycle").glob("*.pkl"))
    lifecycle.write_bytes(b"bad lifecycle")
    with pytest.raises(CheckpointCorruptionError):
        CallStore.open(tmp_path, run_descriptor={})


def test_phase_predicate_bypasses_the_store(tmp_path):
    backend = Backend()
    backend.generate_chat_batch = lambda batch, params, **kwargs: [
        GenerationResult(text="uncached") for _ in batch
    ]
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(
            backend, store, scope="excluded", persist=lambda scope: scope == "included"
        )
        result = wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
        assert result.text == "uncached"
        assert "persistence" not in result.metadata
        assert not list(store._records)


def test_nonsemantic_callbacks_are_excluded_before_copying_private_request(tmp_path):
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        key, _, _ = store.reserve(
            {
                "arguments": {
                    "params": SamplingParams(
                        extra={"callbacks": [threading.Lock()], "cache_key": "keep"}
                    )
                }
            },
            1,
        )
        assert store._records[key]["request"]["arguments"]["params"].extra == {"cache_key": "keep"}


def test_persistent_policy_callables_without_stable_identity_raise_before_dispatch(tmp_path):
    backend = Backend()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        with pytest.raises(TypeError, match="callable"):
            PersistentBackend(backend, store).generate_chat(
                messages("a"),
                SamplingParams(extra={"logits_processor": [lambda value: value]}),
                repeat_id=1,
            )
        assert backend.calls == 0


def test_explicit_provider_failure_retains_receipt_and_replays_error_policy(tmp_path):
    from llenvs.inference.protocol import MalformedResponseError, PartialBatchError

    backend = Backend()
    parse = backend.reparse_receipt
    backend.reparse_receipt = lambda *args: (_ for _ in ()).throw(
        MalformedResponseError("Provider failed", provider_error={"code": 429})
    )
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store)
        with pytest.raises(PartialBatchError):
            wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
        record = next(iter(store._records.values()))
        assert record["state"] == "failed" and record["receipt"]["text"] == "a"
        backend.reparse_receipt = parse
        with pytest.raises(PartialBatchError) as caught:
            wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
        assert isinstance(caught.value.failures[0], MalformedResponseError)
        assert backend.calls == 1
        store.acknowledge_resubmission(
            record["key"], reason="Retry definite failure", accept_duplicate_expenditure=True
        )
        assert wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1).text == "a"
        assert backend.calls == 2


def test_checksum_failure_while_open_is_fatal_before_another_dispatch(tmp_path):
    backend = Backend()
    with CallStore.open(tmp_path, run_descriptor={}) as store:
        wrapped = PersistentBackend(backend, store)
        wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
        path = next((store.run_dir / "calls").rglob("*.pkl"))
        store._records.cache.clear()
        path.write_bytes(b"corrupt")
        with pytest.raises(CheckpointCorruptionError):
            wrapped.generate_chat(messages("a"), SamplingParams(), repeat_id=1)
        assert backend.calls == 1


def _try_run_lock(root, connection):
    try:
        with CallStore.open(root, run_descriptor={}):
            connection.send("acquired")
    except CheckpointInUseError:
        connection.send("refused")


def test_run_lock_refuses_a_second_process(tmp_path):
    import multiprocessing

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    with CallStore.open(tmp_path, run_descriptor={}):
        process = context.Process(target=_try_run_lock, args=(str(tmp_path), child))
        process.start()
        try:
            assert parent.poll(15)
            assert parent.recv() == "refused"
            process.join(timeout=10)
            assert process.exitcode == 0
        finally:
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
            parent.close()
            child.close()


def _phase_crash_worker(root, connection, release):
    from llenvs.inference.controlled import TokenPhase

    class Phased(Backend):
        def generate_chat_batch(self, batch, params, *, observer):
            observer.begin_phases(0)

            def execute(child):
                child.started(0, 0)
                phase = TokenPhase((1,), GenerationResult(text="reasoning"))
                child.received(0, phase)
                child.completed(0, phase)
                return phase

            observer.run_phase(0, "reasoning", {"ids": [1]}, execute, lambda receipt: receipt)
            connection.send("phase durable")
            release.wait()

    with CallStore.open(root, run_descriptor={}) as store:
        PersistentBackend(Phased(), store).generate_chat(
            messages("a"), SamplingParams(), repeat_id=1
        )


def test_process_death_between_phases_replays_reasoning_without_dispatch(tmp_path):
    import multiprocessing

    from llenvs.inference.controlled import TokenPhase

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    release = context.Event()
    process = context.Process(target=_phase_crash_worker, args=(str(tmp_path), child, release))
    process.start()
    try:
        assert parent.poll(15), "Reasoning was not acknowledged"
        assert parent.recv() == "phase durable"
        process.kill()
        process.join(timeout=10)
        assert not process.is_alive()

        class Resumed(Backend):
            def generate_chat_batch(self, batch, params, *, observer):
                observer.begin_phases(0)
                phase = observer.run_phase(
                    0,
                    "reasoning",
                    {"ids": [1]},
                    lambda child: pytest.fail("Completed reasoning was dispatched again"),
                    lambda receipt: receipt,
                )
                assert isinstance(phase, TokenPhase)
                result = GenerationResult(text="answer")
                observer.completed(0, result)
                return [result]

        with CallStore.open(tmp_path, run_descriptor={}) as store:
            assert (
                PersistentBackend(Resumed(), store)
                .generate_chat(messages("a"), SamplingParams(), repeat_id=1)
                .text
                == "answer"
            )
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        parent.close()
        child.close()
