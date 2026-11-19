"""Context ownership, device selection and deterministic-policy restoration."""

from types import SimpleNamespace

import pytest
import torch

from dicechess_training import runtime as execution


class Model:
    def __init__(self):
        self.device = None

    def to(self, device):
        self.device = device
        return self


class Store:
    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value

    def wait(self, keys, _timeout=None):
        # Lifecycle unit cases model an already-ready peer. Real two-worker
        # recovery remains covered by the fresh-process integration tests.
        for key in keys:
            self.values.setdefault(key, b"ready")


@pytest.fixture
def backend(monkeypatch):
    state = SimpleNamespace(
        active=False, prefixes=[], backend=None, device=None, wrapped=None, pending=[]
    )
    original = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )

    def initialize(*, backend, store, rank, world_size, timeout):
        assert store is not None
        assert rank == 0
        assert world_size == 2
        assert timeout.total_seconds() > 0
        state.active = True
        state.backend = backend

    def destroy():
        state.active = False

    def synchronize():
        pending, state.pending = state.pending, []
        for complete in pending:
            complete()

    def prefix(name, store):
        state.prefixes.append(name)
        return store

    def wrap(model, *, device_ids, broadcast_buffers):
        state.wrapped = SimpleNamespace(
            model=model, device_ids=device_ids, broadcast_buffers=broadcast_buffers
        )
        return state.wrapped

    monkeypatch.setattr(execution.dist, "is_initialized", lambda: state.active)
    monkeypatch.setattr(execution.dist, "rendezvous", lambda *a, **kw: iter([(Store(), 0, 2)]))
    monkeypatch.setattr(execution.dist, "PrefixStore", prefix)
    monkeypatch.setattr(execution.dist, "init_process_group", initialize)
    monkeypatch.setattr(execution.dist, "destroy_process_group", destroy)
    monkeypatch.setattr(execution.dist, "barrier", synchronize)
    monkeypatch.setattr(execution, "DistributedDataParallel", wrap)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: setattr(state, "device", device))
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("LOCAL_RANK", "0")
    yield state
    torch.use_deterministic_algorithms(original[0], warn_only=original[1])


def _exercise(context, model, state, error):
    with context as entered:
        assert entered is context
        state.during = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            state.active,
        )
        state.result = context.wrap(model)
        if error is not None:
            raise error


@pytest.mark.parametrize(
    ("requested", "available", "selected"),
    [("cpu", False, "cpu"), ("cuda", True, "cuda"), ("auto", False, "cpu"), ("auto", True, "cuda")],
)
@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("policy", [(False, False), (False, True), (True, False), (True, True)])
@pytest.mark.parametrize("error_type", [None, ValueError, execution.TrainingPaused])
def test_context_restores_policy_device_and_ownership(
    backend, monkeypatch, requested, available, selected, world_size, policy, error_type
):
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])
    context = execution.Runtime(execution.RunOptions(device=requested))
    model = Model()
    error = error_type("authored body failure") if error_type else None
    if error_type:
        with pytest.raises(error_type) as caught:
            _exercise(context, model, backend, error)
        assert caught.value is error
    else:
        _exercise(context, model, backend, error)
    assert backend.during == (True, False, world_size == 2)
    assert model.device.type == selected
    assert not backend.active
    assert torch.are_deterministic_algorithms_enabled() is policy[0]
    assert torch.is_deterministic_algorithms_warn_only_enabled() is policy[1]
    if world_size == 2:
        assert backend.backend == ("nccl" if selected == "cuda" else "gloo")
        assert backend.result.model is model
        assert backend.result.device_ids == ([0] if selected == "cuda" else None)
        assert backend.result.broadcast_buffers is False
    else:
        assert backend.result is model
        assert backend.prefixes == []


def test_sequential_owned_contexts_use_distinct_store_namespaces(backend, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    for _ in range(3):
        with execution.Runtime(execution.RunOptions(device="cpu")):
            assert backend.active
        assert not backend.active
    assert len(backend.prefixes) == 3
    assert len(set(backend.prefixes)) == 3


def test_existing_process_group_remains_owned_by_its_caller(backend, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    backend.active = True
    context = execution.Runtime(execution.RunOptions(device="cpu"))
    with pytest.raises(execution.TrainingRuntimeError, match="own process group"):
        context.__enter__()
    assert backend.active


@pytest.mark.parametrize("error_type", [None, execution.TrainingPaused])
def test_cpu_context_finishes_pending_work_while_wrapper_survives(backend, monkeypatch, error_type):
    monkeypatch.setenv("WORLD_SIZE", "2")
    finished = []
    wrapper = None
    error = error_type("authored pause") if error_type else None
    try:
        with execution.Runtime(execution.RunOptions(device="cpu")) as context:
            wrapper = context.wrap(Model())
            backend.pending.append(lambda: finished.append("completed"))
            if error is not None:
                raise error
    except execution.TrainingPaused as caught:
        assert caught is error
    assert wrapper is backend.wrapped
    assert finished == ["completed"]
    assert not backend.active


@pytest.mark.parametrize("stage", ["foreign-group", "rendezvous", "initialization", "cuda"])
@pytest.mark.parametrize("policy", [(False, False), (False, True), (True, False), (True, True)])
def test_failed_context_entry_restores_callers_policy(backend, monkeypatch, stage, policy):
    monkeypatch.setenv("WORLD_SIZE", "2")
    torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])
    failure = RuntimeError("authored setup failure")

    def fail(*args, **kwargs):
        raise failure

    requested = "cpu"
    if stage == "foreign-group":
        backend.active = True
    elif stage == "rendezvous":
        monkeypatch.setattr(execution.dist, "rendezvous", fail)
    elif stage == "initialization":
        monkeypatch.setattr(execution.dist, "init_process_group", fail)
    else:
        requested = "cuda"
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "set_device", fail)
    context = execution.Runtime(execution.RunOptions(device=requested))
    expected = execution.TrainingRuntimeError if stage == "foreign-group" else RuntimeError
    with pytest.raises(expected) as caught:
        context.__enter__()
    if stage != "foreign-group":
        assert caught.value is failure
    assert backend.active is (stage == "foreign-group")
    assert torch.are_deterministic_algorithms_enabled() is policy[0]
    assert torch.is_deterministic_algorithms_warn_only_enabled() is policy[1]


@pytest.mark.parametrize("stage", ["pending-work", "destruction"])
@pytest.mark.parametrize("policy", [(False, False), (False, True), (True, False), (True, True)])
def test_failed_context_cleanup_restores_callers_policy(backend, monkeypatch, stage, policy):
    monkeypatch.setenv("WORLD_SIZE", "2")
    torch.use_deterministic_algorithms(policy[0], warn_only=policy[1])
    failure = RuntimeError("authored cleanup failure")

    def fail(*args, **kwargs):
        raise failure

    if stage == "pending-work":
        backend.pending.append(fail)
    else:
        monkeypatch.setattr(execution.dist, "destroy_process_group", fail)
    with (
        pytest.raises(RuntimeError) as caught,
        execution.Runtime(execution.RunOptions(device="cpu")),
    ):
        assert backend.active
    assert caught.value is failure
    assert backend.active is (stage == "destruction")
    assert torch.are_deterministic_algorithms_enabled() is policy[0]
    assert torch.is_deterministic_algorithms_warn_only_enabled() is policy[1]


def test_next_cpu_group_waits_for_peer_to_finish_previous_trainer(backend, monkeypatch):
    import threading

    monkeypatch.setenv("WORLD_SIZE", "2")
    awaiting_peer = threading.Event()
    peer_finished = threading.Event()
    entered = threading.Event()
    failures = []

    class PeerStore(Store):
        def wait(self, keys, timeout=None):
            awaiting_peer.set()
            if not peer_finished.wait(2):
                raise RuntimeError("authored peer did not finish")
            super().wait(keys, timeout)

    monkeypatch.setattr(execution.dist, "rendezvous", lambda *a, **kw: iter([(PeerStore(), 0, 2)]))

    def train_next():
        try:
            with execution.Runtime(execution.RunOptions(device="cpu")):
                entered.set()
        except BaseException as error:
            failures.append(error)

    worker = threading.Thread(target=train_next)
    worker.start()
    try:
        assert awaiting_peer.wait(1)
        assert not entered.is_set()
        assert not backend.active
    finally:
        peer_finished.set()
        worker.join(3)
    assert not worker.is_alive()
    assert not failures
    assert entered.is_set()
    assert not backend.active
