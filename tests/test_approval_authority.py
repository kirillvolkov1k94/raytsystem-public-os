"""Workflow approval issuance is derived from trusted, live workflow state."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path
from threading import Event, get_ident
from typing import Any

import pytest
from pydantic import ValidationError

from platform_helpers import make_platform_workspace
from raytsystem.authority import AuthorityError
from raytsystem.contracts import (
    ApprovalRecord,
    PendingWorkflowApproval,
    WorkflowApprovalGate,
    WorkflowDefinition,
    WorkflowNode,
    WorkflowRevision,
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
from raytsystem.workflows import ApprovalAuthorityService, WorkflowService

pytestmark = pytest.mark.filterwarnings("error")

ACTOR = "user_local_test"
APPROVER = "user_approval_operator"
WAIT_STARTED = datetime(2040, 1, 2, 3, 4, 5, tzinfo=UTC)
GATE = WorkflowApprovalGate(
    approval_gate_id="wgate_authority_test",
    action="workflow_approval",
    scope_sha256="c" * 64,
    required_role="role_operator",
    expires_after_seconds=120,
)


def _approval_node(gate: WorkflowApprovalGate = GATE) -> WorkflowNode:
    return WorkflowNode(
        node_id="step_gate",
        node_type=WorkflowNodeType.APPROVAL,
        name="Approval gate",
        input_schema_sha256="a" * 64,
        output_schema_sha256="b" * 64,
        approval_gate_id=gate.approval_gate_id,
    )


def _revision(gate: WorkflowApprovalGate = GATE) -> WorkflowRevision:
    draft = WorkflowRevision(
        revision_id="wrev_authority_test",
        workflow_id="wf_authority_test",
        version="1.0.0",
        trigger_ids=(),
        nodes=(_approval_node(gate),),
        edges=(),
        approval_gate_ids=(gate.approval_gate_id,),
        manifest_sha256="0" * 64,
        created_at=datetime(2040, 1, 1, tzinfo=UTC),
    )
    manifest = sha256_hex(
        canonical_json_bytes(draft.model_dump(mode="json", exclude={"manifest_sha256"}))
    )
    return draft.model_copy(update={"manifest_sha256": manifest})


def _register(
    root: Path, gate: WorkflowApprovalGate = GATE
) -> tuple[WorkflowService, WorkflowRevision]:
    service = WorkflowService(root)
    revision = _revision(gate)
    service.register(
        WorkflowDefinition(
            workflow_id=revision.workflow_id,
            name="Authority workflow",
            description="Approval authority integration fixture",
            enabled=True,
        ),
        revision,
        actor_id=ACTOR,
        approval_gates=(gate,),
    )
    return service, revision


def _start_waiting(
    root: Path,
    *,
    idempotency_key: str = "workflow_authority_case",
    inputs: dict[str, Any] | None = None,
) -> tuple[WorkflowService, str]:
    service, revision = _register(root)
    run = service.start(
        revision.revision_id,
        inputs or {"seed": "value"},
        actor_id=ACTOR,
        idempotency_key=idempotency_key,
    )
    service.run_ready_steps(run.workflow_run_id, at=WAIT_STARTED)
    return service, run.workflow_run_id


def _approval_count(root: Path) -> int:
    store = open_platform_store_read_only(root)
    assert store is not None
    with store:
        return len(store.list_heads("authority_approval", limit=500))


def _receipt_count(root: Path) -> int:
    store = open_platform_store_read_only(root)
    assert store is not None
    with store:
        row = store.connection.execute(
            "SELECT COUNT(*) FROM idempotency_receipts "
            "WHERE scope LIKE 'workflow_approval_issuance_%'"
        ).fetchone()
    assert row is not None
    return int(row[0])


def test_inspect_pending_returns_exact_immutable_trusted_binding(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    _, workflow_run_id = _start_waiting(root)
    local_at = (WAIT_STARTED + timedelta(seconds=10)).astimezone(
        timezone(timedelta(hours=7))
    )

    pending = ApprovalAuthorityService(root).inspect_pending(
        workflow_run_id, "step_gate", at=local_at
    )

    assert pending == PendingWorkflowApproval(
        workflow_run_id=workflow_run_id,
        step_run_id=derive_id(
            "wstep", {"workflow_run_id": workflow_run_id, "node_id": "step_gate"}
        ),
        node_id="step_gate",
        approval_gate_id=GATE.approval_gate_id,
        action=GATE.action,
        target_id=derive_id(
            "wfappr", {"node_id": "step_gate", "workflow_run_id": workflow_run_id}
        ),
        input_sha256=sha256_hex(canonical_json_bytes({"seed": "value"})),
        scope_sha256=GATE.scope_sha256,
        required_role=GATE.required_role,
        expires_at=WAIT_STARTED + timedelta(seconds=GATE.expires_after_seconds),
    )
    assert pending.expires_at.tzinfo is UTC
    with pytest.raises(ValidationError, match="frozen"):
        pending.node_id = "step_other"  # type: ignore[misc]


def test_exact_issuance_is_persisted_idempotently_without_transition(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    _, workflow_run_id = _start_waiting(root)
    authority = ApprovalAuthorityService(root)

    first = authority.issue_approval(
        workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key="approval_authority_case",
        at=WAIT_STARTED + timedelta(seconds=10),
    )
    replay = authority.issue_approval(
        workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key="approval_authority_case",
        at=WAIT_STARTED + timedelta(seconds=20),
    )

    assert isinstance(first, ApprovalRecord)
    assert replay == first
    assert first.action == GATE.action
    assert first.target_id == derive_id(
        "wfappr", {"node_id": "step_gate", "workflow_run_id": workflow_run_id}
    )
    assert first.artifact_sha256 == sha256_hex(canonical_json_bytes({"seed": "value"}))
    assert first.scope == (GATE.required_role,)
    assert first.policy_version == GATE.schema_version
    assert first.policy_sha256 == GATE.scope_sha256
    assert first.approver == APPROVER
    assert first.approved_at == WAIT_STARTED + timedelta(seconds=10)
    assert first.expires_at == WAIT_STARTED + timedelta(seconds=GATE.expires_after_seconds)
    assert _approval_count(root) == 1
    assert _receipt_count(root) == 2
    with initialize_platform_store(root) as store:
        step = store.head("workflow_step", derive_id(
            "wstep", {"workflow_run_id": workflow_run_id, "node_id": "step_gate"}
        ))
    assert step is not None
    assert step.state == "waiting"


def test_changed_actor_key_or_request_binding_cannot_issue_again(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, workflow_run_id = _start_waiting(root)
    authority = ApprovalAuthorityService(root)
    authority.issue_approval(
        workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key="approval_authority_binding",
        at=WAIT_STARTED + timedelta(seconds=10),
    )

    with pytest.raises(AuthorityError, match=r"binding|idempotency"):
        authority.issue_approval(
            workflow_run_id,
            "step_gate",
            approver="user_other_operator",
            idempotency_key="approval_authority_binding",
            at=WAIT_STARTED + timedelta(seconds=11),
        )
    with pytest.raises(AuthorityError, match=r"binding|idempotency"):
        authority.issue_approval(
            workflow_run_id,
            "step_gate",
            approver=APPROVER,
            idempotency_key="approval_authority_other_key",
            at=WAIT_STARTED + timedelta(seconds=11),
        )

    second = service.start(
        _revision().revision_id,
        {"seed": "second"},
        actor_id=ACTOR,
        idempotency_key="workflow_authority_second",
    )
    service.run_ready_steps(second.workflow_run_id, at=WAIT_STARTED)
    with pytest.raises(AuthorityError, match=r"binding|idempotency"):
        authority.issue_approval(
            second.workflow_run_id,
            "step_gate",
            approver=APPROVER,
            idempotency_key="approval_authority_binding",
            at=WAIT_STARTED + timedelta(seconds=12),
        )
    assert _approval_count(root) == 1


def test_expired_or_pre_request_gate_rejects_issuance(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    _, workflow_run_id = _start_waiting(root)
    authority = ApprovalAuthorityService(root)

    for at in (
        WAIT_STARTED - timedelta(microseconds=1),
        WAIT_STARTED + timedelta(seconds=GATE.expires_after_seconds),
    ):
        with pytest.raises(AuthorityError, match=r"active|expired"):
            authority.inspect_pending(workflow_run_id, "step_gate", at=at)
        with pytest.raises(AuthorityError, match=r"active|expired"):
            authority.issue_approval(
                workflow_run_id,
                "step_gate",
                approver=APPROVER,
                idempotency_key="approval_authority_expired",
                at=at,
            )
    assert _approval_count(root) == 0


def test_naive_times_fail_closed_and_offset_times_normalize_to_utc(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    _, workflow_run_id = _start_waiting(root)
    authority = ApprovalAuthorityService(root)
    naive = datetime(2040, 1, 2, 3, 4, 15)

    with pytest.raises(AuthorityError, match="timezone"):
        authority.inspect_pending(workflow_run_id, "step_gate", at=naive)
    with pytest.raises(AuthorityError, match="timezone"):
        authority.issue_approval(
            workflow_run_id,
            "step_gate",
            approver=APPROVER,
            idempotency_key="approval_authority_naive",
            at=naive,
        )

    local_at = (WAIT_STARTED + timedelta(seconds=10)).astimezone(
        timezone(timedelta(hours=-5))
    )
    approval = authority.issue_approval(
        workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key="approval_authority_offset",
        at=local_at,
    )
    assert approval.approved_at == WAIT_STARTED + timedelta(seconds=10)
    assert approval.approved_at.tzinfo is UTC


def test_changed_workflow_input_or_gate_fails_closed_without_second_record(
    tmp_path: Path,
) -> None:
    root = make_platform_workspace(tmp_path)
    _, workflow_run_id = _start_waiting(root)
    authority = ApprovalAuthorityService(root)
    authority.issue_approval(
        workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key="approval_authority_integrity",
        at=WAIT_STARTED + timedelta(seconds=10),
    )
    with initialize_platform_store(root) as store:
        run = store.head("workflow_run", workflow_run_id)
        assert run is not None
        changed_inputs = {"seed": "changed"}
        changed = dict(run.payload)
        changed["inputs"] = changed_inputs
        changed["input_sha256"] = sha256_hex(canonical_json_bytes(changed_inputs))
        store.append_record(
            kind="workflow_run",
            record_id=workflow_run_id,
            payload=changed,
            state=run.state,
            expected_revision=run.revision,
        )

    with pytest.raises(AuthorityError, match=r"input|binding"):
        authority.issue_approval(
            workflow_run_id,
            "step_gate",
            approver=APPROVER,
            idempotency_key="approval_authority_integrity",
            at=WAIT_STARTED + timedelta(seconds=20),
        )
    assert _approval_count(root) == 1

    gate_root = make_platform_workspace(tmp_path / "gate")
    _, gate_run_id = _start_waiting(gate_root)
    gate_authority = ApprovalAuthorityService(gate_root)
    gate_authority.issue_approval(
        gate_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key="approval_authority_gate_integrity",
        at=WAIT_STARTED + timedelta(seconds=10),
    )
    with initialize_platform_store(gate_root) as store:
        gate = store.head("workflow_approval_gate", GATE.approval_gate_id)
        assert gate is not None
        changed_gate = dict(gate.payload)
        changed_gate["required_role"] = "role_reviewer"
        store.append_record(
            kind="workflow_approval_gate",
            record_id=GATE.approval_gate_id,
            payload=changed_gate,
            state=gate.state,
            expected_revision=gate.revision,
        )
    with pytest.raises(AuthorityError, match=r"gate|immutable"):
        gate_authority.inspect_pending(
            gate_run_id, "step_gate", at=WAIT_STARTED + timedelta(seconds=21)
        )
    assert _approval_count(gate_root) == 1


def test_wrong_node_and_nonwaiting_step_fail_without_authority_record(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, revision = _register(root)
    run = service.start(
        revision.revision_id,
        {"seed": "value"},
        actor_id=ACTOR,
        idempotency_key="workflow_authority_nonwaiting",
    )
    authority = ApprovalAuthorityService(root)

    for node_id in ("step_missing", "step_gate"):
        with pytest.raises(AuthorityError, match=r"node|waiting"):
            authority.issue_approval(
                run.workflow_run_id,
                node_id,
                approver=APPROVER,
                idempotency_key=f"approval_authority_{node_id}",
                at=WAIT_STARTED,
            )
    assert _approval_count(root) == 0

    service.run_ready_steps(run.workflow_run_id, at=WAIT_STARTED)
    approval = authority.issue_approval(
        run.workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key="approval_authority_transition",
        at=WAIT_STARTED + timedelta(seconds=10),
    )
    service.grant_approval(
        run.workflow_run_id,
        "step_gate",
        approval_id=approval.approval_id,
        actor_id=ACTOR,
        at=WAIT_STARTED + timedelta(seconds=11),
    )
    with pytest.raises(AuthorityError, match="waiting"):
        authority.issue_approval(
            run.workflow_run_id,
            "step_gate",
            approver=APPROVER,
            idempotency_key="approval_authority_transition",
            at=WAIT_STARTED + timedelta(seconds=12),
        )
    assert _approval_count(root) == 1


def test_concurrent_exact_issuance_creates_one_record(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    _, workflow_run_id = _start_waiting(root)

    def _issue(_: int) -> str:
        approval = ApprovalAuthorityService(root).issue_approval(
            workflow_run_id,
            "step_gate",
            approver=APPROVER,
            idempotency_key="approval_authority_concurrent",
            at=WAIT_STARTED + timedelta(seconds=10),
        )
        return approval.approval_id

    with ThreadPoolExecutor(max_workers=8) as executor:
        approval_ids = tuple(executor.map(_issue, range(16)))

    assert len(set(approval_ids)) == 1
    assert _approval_count(root) == 1
    assert _receipt_count(root) == 2


def test_crash_rolls_back_approval_and_both_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_platform_workspace(tmp_path)
    _, workflow_run_id = _start_waiting(root)
    authority = ApprovalAuthorityService(root)
    original = PlatformStore.idempotent_receipt

    class SimulatedCrash(RuntimeError):
        pass

    def _crash_after_target_binding(
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
        if receipt is not None and scope == "workflow_approval_issuance_target":
            raise SimulatedCrash("crash after first persisted binding")
        return result

    with monkeypatch.context() as patch:
        patch.setattr(PlatformStore, "idempotent_receipt", _crash_after_target_binding)
        with pytest.raises(SimulatedCrash):
            authority.issue_approval(
                workflow_run_id,
                "step_gate",
                approver=APPROVER,
                idempotency_key="approval_authority_crash",
                at=WAIT_STARTED + timedelta(seconds=10),
            )

    assert _approval_count(root) == 0
    assert _receipt_count(root) == 0
    approval = authority.issue_approval(
        workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key="approval_authority_crash",
        at=WAIT_STARTED + timedelta(seconds=11),
    )
    assert approval.approved_at == WAIT_STARTED + timedelta(seconds=11)
    assert _approval_count(root) == 1
    assert _receipt_count(root) == 2


@pytest.mark.parametrize("operation", ["inspect", "issue"])
def test_default_time_is_sampled_after_writer_lock_before_expiry_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    root = make_platform_workspace(tmp_path / operation)
    short_gate = GATE.model_copy(
        update={
            "approval_gate_id": f"wgate_authority_{operation}",
            "expires_after_seconds": 1,
        }
    )
    service, revision = _register(root, short_gate)
    run = service.start(
        revision.revision_id,
        {"seed": "value"},
        actor_id=ACTOR,
        idempotency_key=f"workflow_authority_lock_{operation}",
    )
    service.run_ready_steps(run.workflow_run_id, at=datetime.now(UTC))
    authority = ApprovalAuthorityService(root)
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

    def _call() -> object:
        if operation == "inspect":
            return authority.inspect_pending(run.workflow_run_id, "step_gate")
        return authority.issue_approval(
            run.workflow_run_id,
            "step_gate",
            approver=APPROVER,
            idempotency_key=f"approval_authority_lock_{operation}",
        )

    with ThreadPoolExecutor(max_workers=1) as executor, initialize_platform_store(
        root
    ) as blocker:
        with blocker.transaction():
            result = executor.submit(_call)
            assert transaction_attempted.wait(timeout=1)
            time.sleep(1.2)
        with pytest.raises(AuthorityError, match="expired"):
            result.result(timeout=5)
    assert _approval_count(root) == 0
