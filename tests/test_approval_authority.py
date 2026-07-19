"""Workflow approval issuance is derived from trusted, live workflow state."""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from threading import Event, get_ident
from typing import Any

import pytest
from pydantic import ValidationError

import raytsystem.authority as authority_module
import raytsystem.contracts as contracts_module
import raytsystem.workflows as workflows_module
from platform_helpers import make_platform_workspace
from raytsystem.authority import AuthorityError, AuthorityResolver
from raytsystem.contracts import (
    ApprovalRecord,
    PendingWorkflowApproval,
    PendingWorkflowApprovalPage,
    WorkflowApprovalGate,
    WorkflowDefinition,
    WorkflowNode,
    WorkflowRevision,
    canonical_json_bytes,
    derive_id,
    sha256_hex,
)
from raytsystem.contracts.execution import ExecutionApproval
from raytsystem.contracts.governance import EmergencyAction
from raytsystem.contracts.workflows import WorkflowNodeType
from raytsystem.emergency import EmergencyService
from raytsystem.execution.store import ExecutionStore
from raytsystem.platform_store import (
    PlatformStore,
    StoredRecord,
    initialize_platform_store,
    open_platform_store_read_only,
)
from raytsystem.workflows import ApprovalAuthorityService, WorkflowError, WorkflowService

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


def _historical_pending_runs(
    root: Path,
    *,
    revision_count: int,
    runs_per_revision: int,
) -> tuple[tuple[str, str], ...]:
    service = WorkflowService(root)
    runs: list[tuple[str, str]] = []
    previous_revision_id: str | None = None
    for revision_index in range(revision_count):
        revision_id = f"wrev_authority_history_{revision_index:03d}"
        draft = _revision().model_copy(
            update={
                "revision_id": revision_id,
                "version": f"1.{revision_index}.0",
                "previous_revision_id": previous_revision_id,
                "created_at": datetime(2040, 1, 1, tzinfo=UTC) + timedelta(seconds=revision_index),
                "manifest_sha256": "0" * 64,
            }
        )
        manifest = sha256_hex(
            canonical_json_bytes(draft.model_dump(mode="json", exclude={"manifest_sha256"}))
        )
        revision = draft.model_copy(update={"manifest_sha256": manifest})
        service.register(
            WorkflowDefinition(
                workflow_id=revision.workflow_id,
                name="Authority workflow",
                description="Approval authority history fixture",
                enabled=True,
            ),
            revision,
            actor_id=ACTOR,
            approval_gates=(GATE,),
        )
        for run_index in range(runs_per_revision):
            key = f"workflow_history_{revision_index:03d}_{run_index:03d}"
            run = service.start(
                revision.revision_id,
                {"seed": key},
                actor_id=ACTOR,
                idempotency_key=key,
            )
            service.run_ready_steps(run.workflow_run_id, at=WAIT_STARTED)
            runs.append((run.workflow_run_id, revision.revision_id))
        previous_revision_id = revision.revision_id
    return tuple(runs)


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


def _forged_workflow_approval(
    workflow_run_id: str,
    *,
    approved_at: datetime = WAIT_STARTED + timedelta(seconds=10),
    expires_at: datetime = WAIT_STARTED + timedelta(seconds=GATE.expires_after_seconds),
    scope: tuple[str, ...] = (GATE.required_role,),
    policy_version: str = GATE.schema_version,
    policy_sha256: str = GATE.scope_sha256,
    conditions: tuple[str, ...] = (),
) -> ApprovalRecord:
    return ApprovalRecord.create(
        action=GATE.action,
        target_id=derive_id("wfappr", {"node_id": "step_gate", "workflow_run_id": workflow_run_id}),
        artifact_sha256=sha256_hex(canonical_json_bytes({"seed": "value"})),
        scope=scope,
        policy_version=policy_version,
        policy_sha256=policy_sha256,
        approver=APPROVER,
        approved_at=approved_at,
        expires_at=expires_at,
        conditions=conditions,
    )


def _store_authority_approval(
    root: Path,
    approval: ApprovalRecord,
    *,
    record_id: str | None = None,
    state: str = "accepted",
    revisions: int = 1,
) -> str:
    stored_id = record_id or approval.approval_id
    with initialize_platform_store(root) as store:
        expected_revision: int | None = None
        for revision in range(revisions):
            store.append_record(
                kind="authority_approval",
                record_id=stored_id,
                payload=approval.model_dump(mode="json"),
                state=state,
                expected_revision=expected_revision,
            )
            expected_revision = revision + 1
    return stored_id


def _step_head(root: Path, workflow_run_id: str) -> StoredRecord:
    with initialize_platform_store(root) as store:
        record = store.head(
            "workflow_step",
            derive_id("wstep", {"workflow_run_id": workflow_run_id, "node_id": "step_gate"}),
        )
    assert record is not None
    return record


def _make_gate_head_noncanonical(root: Path, mutation: str) -> None:
    with initialize_platform_store(root) as store:
        record = store.head("workflow_approval_gate", GATE.approval_gate_id)
        assert record is not None
        if mutation == "revision":
            store.append_record(
                kind=record.kind,
                record_id=record.record_id,
                payload=record.payload,
                state=record.state,
                expected_revision=record.revision,
            )
            return
        if mutation == "state":
            with store.transaction():
                store.connection.execute(
                    "UPDATE records SET state='rejected' "
                    "WHERE kind=? AND record_id=? AND revision=?",
                    (record.kind, record.record_id, record.revision),
                )
            return
        changed = dict(record.payload)
        if mutation == "payload_id":
            changed["approval_gate_id"] = "wgate_authority_alias"
        else:
            changed["action"] = "delete_data"
        rendered = canonical_json_bytes(changed)
        digest = sha256_hex(rendered)
        with store.transaction():
            store.connection.execute(
                "UPDATE records SET payload_json=?, payload_sha256=? "
                "WHERE kind=? AND record_id=? AND revision=?",
                (
                    rendered.decode("utf-8"),
                    digest,
                    record.kind,
                    record.record_id,
                    record.revision,
                ),
            )
            store.connection.execute(
                "UPDATE record_heads SET payload_sha256=? WHERE kind=? AND record_id=?",
                (digest, record.kind, record.record_id),
            )


class _NoOffsetTimezone(tzinfo):
    def utcoffset(self, _value: datetime | None) -> None:
        return None

    def dst(self, _value: datetime | None) -> None:
        return None

    def tzname(self, _value: datetime | None) -> str:
        return "no-offset"


def test_inspect_pending_returns_exact_immutable_trusted_binding(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    _, workflow_run_id = _start_waiting(root)
    local_at = (WAIT_STARTED + timedelta(seconds=10)).astimezone(timezone(timedelta(hours=7)))

    pending = ApprovalAuthorityService(root).inspect_pending(
        workflow_run_id, "step_gate", at=local_at
    )

    assert pending == PendingWorkflowApproval(
        workflow_run_id=workflow_run_id,
        revision_id="wrev_authority_test",
        step_run_id=derive_id(
            "wstep", {"workflow_run_id": workflow_run_id, "node_id": "step_gate"}
        ),
        node_id="step_gate",
        approval_gate_id=GATE.approval_gate_id,
        action=GATE.action,
        target_id=derive_id("wfappr", {"node_id": "step_gate", "workflow_run_id": workflow_run_id}),
        input_sha256=sha256_hex(canonical_json_bytes({"seed": "value"})),
        scope_sha256=GATE.scope_sha256,
        policy_version=GATE.schema_version,
        required_role=GATE.required_role,
        expires_at=WAIT_STARTED + timedelta(seconds=GATE.expires_after_seconds),
    )
    assert pending.expires_at.tzinfo is UTC
    with pytest.raises(ValidationError, match="frozen"):
        pending.node_id = "step_other"  # type: ignore[misc]


def test_pending_workflow_approval_binds_workflow_and_policy_revisions() -> None:
    assert "revision_id" in PendingWorkflowApproval.model_fields
    assert "policy_version" in PendingWorkflowApproval.model_fields


def test_pending_approval_page_is_a_public_frozen_contract() -> None:
    page_type = getattr(contracts_module, "PendingWorkflowApprovalPage", None)

    assert page_type is not None
    assert getattr(workflows_module, "PendingWorkflowApprovalPage", None) is page_type
    assert tuple(page_type.model_fields) == (
        "schema_name",
        "schema_version",
        "id_scheme_version",
        "extensions",
        "items",
        "next_cursor",
        "snapshot_id",
        "observed_at",
    )


def test_list_pending_returns_an_exact_consistency_bound_page(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    _, workflow_run_id = _start_waiting(root)
    authority = ApprovalAuthorityService(root)
    observed_at = WAIT_STARTED + timedelta(seconds=10)

    page = authority.list_pending(limit=1, at=observed_at)

    assert type(page) is PendingWorkflowApprovalPage
    assert page.items == (authority.inspect_pending(workflow_run_id, "step_gate", at=observed_at),)
    assert page.next_cursor is None
    assert page.snapshot_id.startswith("pview_")
    assert page.observed_at == observed_at
    with pytest.raises(ValidationError, match="frozen"):
        page.next_cursor = "changed"  # type: ignore[misc]


def test_list_pending_traverses_204_runs_across_51_historical_revisions(
    tmp_path: Path,
) -> None:
    root = make_platform_workspace(tmp_path)
    expected_runs = _historical_pending_runs(
        root,
        revision_count=51,
        runs_per_revision=4,
    )
    authority = ApprovalAuthorityService(root)
    observed_at = WAIT_STARTED + timedelta(seconds=10)
    cursor: str | None = None
    pages: list[PendingWorkflowApprovalPage] = []
    observed: list[PendingWorkflowApproval] = []

    while True:
        page = authority.list_pending(
            limit=37,
            cursor=cursor,
            at=observed_at if cursor is None else None,
        )
        pages.append(page)
        observed.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            break

    assert len(expected_runs) == 204
    assert len({revision_id for _, revision_id in expected_runs}) == 51
    assert len(observed) == 204
    assert len({item.target_id for item in observed}) == 204
    assert [(item.workflow_run_id, item.node_id, item.step_run_id) for item in observed] == sorted(
        (item.workflow_run_id, item.node_id, item.step_run_id) for item in observed
    )
    assert {item.revision_id for item in observed} == {
        revision_id for _, revision_id in expected_runs
    }
    assert len(pages) == 6
    assert [len(page.items) for page in pages] == [37, 37, 37, 37, 37, 19]
    assert len({page.snapshot_id for page in pages}) == 1
    assert all(page.observed_at == observed_at for page in pages)


def test_list_pending_preserves_default_observation_microseconds_across_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_platform_workspace(tmp_path)
    run_ids = [
        _start_waiting(root, idempotency_key=f"workflow_microseconds_{index}")[1]
        for index in range(2)
    ]
    observed_at = WAIT_STARTED + timedelta(seconds=10, microseconds=654321)
    waiting_since = observed_at - timedelta(microseconds=1)
    for workflow_run_id in run_ids:
        step = _step_head(root, workflow_run_id)
        changed = dict(step.payload)
        changed["started_at"] = waiting_since.isoformat().replace("+00:00", "Z")
        rendered = canonical_json_bytes(changed)
        digest = sha256_hex(rendered)
        with initialize_platform_store(root) as store, store.transaction():
            store.connection.execute(
                "UPDATE records SET payload_json=?, payload_sha256=? "
                "WHERE kind=? AND record_id=? AND revision=?",
                (
                    rendered.decode("utf-8"),
                    digest,
                    step.kind,
                    step.record_id,
                    step.revision,
                ),
            )
            store.connection.execute(
                "UPDATE record_heads SET payload_sha256=? WHERE kind=? AND record_id=?",
                (digest, step.kind, step.record_id),
            )

    class _ObservedDatetime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:
            return observed_at if tz is not None else observed_at.replace(tzinfo=None)

    monkeypatch.setattr(authority_module, "datetime", _ObservedDatetime)
    authority = ApprovalAuthorityService(root)

    first = authority.list_pending(limit=1)
    assert first.next_cursor is not None
    second = authority.list_pending(limit=1, cursor=first.next_cursor)

    assert first.observed_at == observed_at
    assert second.observed_at == observed_at
    assert len(first.items) == len(second.items) == 1
    assert first.items[0].workflow_run_id != second.items[0].workflow_run_id


def test_list_pending_resumes_from_cursor_run_without_rescanning_prior_heads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = make_platform_workspace(tmp_path)
    _historical_pending_runs(root, revision_count=51, runs_per_revision=4)
    authority = ApprovalAuthorityService(root)
    observed_at = WAIT_STARTED + timedelta(seconds=10)
    first = authority.list_pending(limit=37, at=observed_at)
    assert first.next_cursor is not None
    after_run_id = first.items[-1].workflow_run_id
    original = PlatformStore.iter_heads
    calls: list[tuple[str | None, int]] = []
    yielded_run_ids: list[str] = []
    head_queries: list[str] = []

    def _tracked_iter_heads(
        self: PlatformStore,
        kind: str,
        *,
        after_record_id: str | None = None,
        batch_size: int = 500,
    ) -> Any:
        calls.append((after_record_id, batch_size))
        self.connection.set_trace_callback(
            lambda statement: (
                head_queries.append(statement)
                if "FROM record_heads h LEFT JOIN records r" in statement
                else None
            )
        )
        try:
            for record in original(
                self,
                kind,
                after_record_id=after_record_id,
                batch_size=batch_size,
            ):
                yielded_run_ids.append(record.record_id)
                yield record
        finally:
            self.connection.set_trace_callback(None)

    monkeypatch.setattr(PlatformStore, "iter_heads", _tracked_iter_heads)

    second = authority.list_pending(limit=37, cursor=first.next_cursor)

    assert len(second.items) == 37
    assert calls == [(after_run_id, 38)]
    assert len(head_queries) == 1
    assert len(yielded_run_ids) <= 38
    assert all(record_id > after_run_id for record_id in yielded_run_ids)


def test_list_pending_cursor_is_opaque_and_tampering_fails_closed(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    for index in range(3):
        _start_waiting(root, idempotency_key=f"workflow_cursor_{index}")
    authority = ApprovalAuthorityService(root)
    observed_at = WAIT_STARTED + timedelta(seconds=10)

    first = authority.list_pending(limit=1, at=observed_at)

    assert first.next_cursor is not None
    assert first.items[0].workflow_run_id not in first.next_cursor
    assert first.items[0].node_id not in first.next_cursor
    replacement = "A" if first.next_cursor[-1] != "A" else "B"
    tampered = f"{first.next_cursor[:-1]}{replacement}"
    with pytest.raises(AuthorityError, match="cursor"):
        authority.list_pending(limit=1, cursor=tampered)

    second = authority.list_pending(limit=1, cursor=first.next_cursor)
    assert second.snapshot_id == first.snapshot_id
    assert second.observed_at == first.observed_at
    assert second.items[0].target_id != first.items[0].target_id


def test_list_pending_rejects_noncanonical_cursor_encoding(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    for index in range(2):
        _start_waiting(root, idempotency_key=f"workflow_cursor_alias_{index}")
    first = ApprovalAuthorityService(root).list_pending(
        limit=1,
        at=WAIT_STARTED + timedelta(seconds=10),
    )
    assert first.next_cursor is not None
    encoded_payload, encoded_signature = first.next_cursor.split(".")
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    final_index = alphabet.index(encoded_signature[-1])
    assert final_index % 4 == 0
    alias = alphabet[final_index + 1]
    noncanonical = f"{encoded_payload}.{encoded_signature[:-1]}{alias}"

    with pytest.raises(AuthorityError, match="cursor"):
        ApprovalAuthorityService(root).list_pending(limit=1, cursor=noncanonical)


def test_list_pending_cursor_rejects_concurrent_transition(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    for index in range(3):
        _start_waiting(root, idempotency_key=f"workflow_concurrent_page_{index}")
    authority = ApprovalAuthorityService(root)
    observed_at = WAIT_STARTED + timedelta(seconds=10)
    first = authority.list_pending(limit=1, at=observed_at)
    assert first.next_cursor is not None
    expected = first.items[0]

    WorkflowService(root).deny_approval(
        expected.workflow_run_id,
        expected.node_id,
        expected=expected,
        actor_id=ACTOR,
        idempotency_key="workflow_concurrent_page_deny",
        at=observed_at + timedelta(seconds=1),
    )

    with pytest.raises(AuthorityError, match=r"concurrent|changed"):
        authority.list_pending(limit=1, cursor=first.next_cursor)


def test_list_pending_cursor_rejects_changed_observation_time(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    for index in range(2):
        _start_waiting(root, idempotency_key=f"workflow_observation_page_{index}")
    authority = ApprovalAuthorityService(root)
    observed_at = WAIT_STARTED + timedelta(seconds=10)
    first = authority.list_pending(limit=1, at=observed_at)
    assert first.next_cursor is not None

    with pytest.raises(AuthorityError, match=r"observation|consistent"):
        authority.list_pending(
            limit=1,
            cursor=first.next_cursor,
            at=observed_at + timedelta(seconds=1),
        )


def test_list_pending_fails_closed_on_corrupt_canonical_step(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    _, workflow_run_id = _start_waiting(root)
    step = _step_head(root, workflow_run_id)
    changed = dict(step.payload)
    changed["workflow_run_id"] = "wrun_foreign_binding"
    rendered = canonical_json_bytes(changed)
    digest = sha256_hex(rendered)
    with initialize_platform_store(root) as store, store.transaction():
        store.connection.execute(
            "UPDATE records SET payload_json=?, payload_sha256=? "
            "WHERE kind=? AND record_id=? AND revision=?",
            (
                rendered.decode("utf-8"),
                digest,
                step.kind,
                step.record_id,
                step.revision,
            ),
        )
        store.connection.execute(
            "UPDATE record_heads SET payload_sha256=? WHERE kind=? AND record_id=?",
            (digest, step.kind, step.record_id),
        )

    with pytest.raises(AuthorityError, match=r"invalid|binding"):
        ApprovalAuthorityService(root).list_pending(
            limit=10,
            at=WAIT_STARTED + timedelta(seconds=10),
        )


def test_list_pending_fails_closed_on_orphan_canonical_run_head(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    _, workflow_run_id = _start_waiting(root)
    with initialize_platform_store(root) as store:
        head = store.head("workflow_run", workflow_run_id)
        assert head is not None
        store.connection.execute("PRAGMA foreign_keys=OFF")
        with store.transaction():
            store.connection.execute(
                "DELETE FROM records WHERE kind=? AND record_id=? AND revision=?",
                (head.kind, head.record_id, head.revision),
            )

    with pytest.raises(AuthorityError):
        ApprovalAuthorityService(root).list_pending(
            limit=10,
            at=WAIT_STARTED + timedelta(seconds=10),
        )


@pytest.mark.parametrize("limit", [0, 501, True, "10"])
def test_list_pending_rejects_unbounded_or_malformed_limit(
    tmp_path: Path,
    limit: object,
) -> None:
    root = make_platform_workspace(tmp_path)

    with pytest.raises(AuthorityError, match="limit"):
        ApprovalAuthorityService(root).list_pending(limit=limit)  # type: ignore[arg-type]


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
        step = store.head(
            "workflow_step",
            derive_id("wstep", {"workflow_run_id": workflow_run_id, "node_id": "step_gate"}),
        )
    assert step is not None
    assert step.state == "waiting"


def test_pre_revision_pending_issuance_receipts_replay_compatibly(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    _, workflow_run_id = _start_waiting(root)
    authority = ApprovalAuthorityService(root)
    idempotency_key = "approval_authority_legacy_pending"
    pending = authority.inspect_pending(
        workflow_run_id,
        "step_gate",
        at=WAIT_STARTED + timedelta(seconds=5),
    )
    issued = authority.issue_approval(
        workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key=idempotency_key,
        at=WAIT_STARTED + timedelta(seconds=10),
    )
    legacy_pending = pending.model_dump(mode="json")
    legacy_pending.pop("revision_id")
    legacy_pending.pop("policy_version")
    legacy_request = {
        "pending": legacy_pending,
        "gate_schema_version": GATE.schema_version,
        "approver": APPROVER,
        "idempotency_key": idempotency_key,
    }
    legacy_sha256 = sha256_hex(canonical_json_bytes(legacy_request))
    with initialize_platform_store(root) as store, store.transaction():
        store.connection.execute(
            "UPDATE idempotency_receipts SET request_sha256=? "
            "WHERE scope LIKE 'workflow_approval_issuance_%'",
            (legacy_sha256,),
        )

    replay = authority.issue_approval(
        workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key=idempotency_key,
        at=WAIT_STARTED + timedelta(seconds=20),
    )

    assert replay == issued
    assert _approval_count(root) == 1
    assert _receipt_count(root) == 2


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

    local_at = (WAIT_STARTED + timedelta(seconds=10)).astimezone(timezone(timedelta(hours=-5)))
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
        idempotency_key="approval_authority_transition_decision",
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


@pytest.mark.parametrize(
    "invalid_binding",
    [
        "pre_wait",
        "wrong_role",
        "extra_role",
        "wrong_policy_version",
        "wrong_policy_sha256",
        "shortened_expiry",
        "extended_expiry",
        "condition",
    ],
)
def test_grant_rejects_approval_outside_exact_gate_authority(
    tmp_path: Path, invalid_binding: str
) -> None:
    root = make_platform_workspace(tmp_path / invalid_binding)
    service, workflow_run_id = _start_waiting(root)
    approved_at = WAIT_STARTED + timedelta(seconds=10)
    expires_at = WAIT_STARTED + timedelta(seconds=GATE.expires_after_seconds)
    scope = (GATE.required_role,)
    policy_version = GATE.schema_version
    policy_sha256 = GATE.scope_sha256
    conditions: tuple[str, ...] = ()
    if invalid_binding == "pre_wait":
        approved_at = WAIT_STARTED - timedelta(seconds=1)
    elif invalid_binding == "wrong_role":
        scope = ("role_admin",)
    elif invalid_binding == "extra_role":
        scope = (GATE.required_role, "role_admin")
    elif invalid_binding == "wrong_policy_version":
        policy_version = "0.9.0"
    elif invalid_binding == "wrong_policy_sha256":
        policy_sha256 = "d" * 64
    elif invalid_binding == "shortened_expiry":
        expires_at -= timedelta(seconds=1)
    elif invalid_binding == "extended_expiry":
        expires_at += timedelta(seconds=1)
    else:
        conditions = ("manual_follow_up",)
    approval = _forged_workflow_approval(
        workflow_run_id,
        approved_at=approved_at,
        expires_at=expires_at,
        scope=scope,
        policy_version=policy_version,
        policy_sha256=policy_sha256,
        conditions=conditions,
    )
    _store_authority_approval(root, approval)

    with pytest.raises(WorkflowError, match="authority"):
        service.grant_approval(
            workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key=f"approval_authority_invalid_{invalid_binding}",
            at=WAIT_STARTED + timedelta(seconds=20),
        )

    assert _step_head(root, workflow_run_id).state == "waiting"


@pytest.mark.parametrize("invalid_head", ["alias", "rejected", "revision"])
def test_grant_rejects_noncanonical_approval_record_head(tmp_path: Path, invalid_head: str) -> None:
    root = make_platform_workspace(tmp_path / invalid_head)
    service, workflow_run_id = _start_waiting(root)
    approval = _forged_workflow_approval(workflow_run_id)
    record_id = approval.approval_id
    state = "accepted"
    revisions = 1
    if invalid_head == "alias":
        record_id = "apr_authority_alias"
    elif invalid_head == "rejected":
        state = "rejected"
    else:
        revisions = 2
    requested_id = _store_authority_approval(
        root,
        approval,
        record_id=record_id,
        state=state,
        revisions=revisions,
    )

    with pytest.raises(WorkflowError, match="authority"):
        service.grant_approval(
            workflow_run_id,
            "step_gate",
            approval_id=requested_id,
            actor_id=ACTOR,
            idempotency_key=f"approval_authority_head_{invalid_head}",
            at=WAIT_STARTED + timedelta(seconds=20),
        )

    assert _step_head(root, workflow_run_id).state == "waiting"


@pytest.mark.parametrize("invalid_gate", ["wrong_action", "payload_id", "state", "revision"])
def test_grant_rejects_noncanonical_approval_gate_head(tmp_path: Path, invalid_gate: str) -> None:
    root = make_platform_workspace(tmp_path / invalid_gate)
    service, workflow_run_id = _start_waiting(root)
    approval = ApprovalAuthorityService(root).issue_approval(
        workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key=f"approval_authority_gate_{invalid_gate}",
        at=WAIT_STARTED + timedelta(seconds=10),
    )
    _make_gate_head_noncanonical(root, invalid_gate)

    with pytest.raises(WorkflowError, match="gate"):
        service.grant_approval(
            workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key=f"approval_authority_gate_decision_{invalid_gate}",
            at=WAIT_STARTED + timedelta(seconds=20),
        )

    step = _step_head(root, workflow_run_id)
    assert step.state == "waiting"
    assert step.payload.get("approval_id") is None
    assert "output" not in step.payload
    with initialize_platform_store(root) as store:
        event_types = tuple(event["event_type"] for event in store.list_events(workflow_run_id))
    assert "workflow_approval_granted" not in event_types


@pytest.mark.parametrize(
    "invalid_at",
    [
        datetime(2040, 1, 2, 10, 4, 25),
        datetime(2040, 1, 2, 10, 4, 25, tzinfo=_NoOffsetTimezone()),
    ],
)
def test_grant_rejects_explicit_time_without_utc_offset(
    tmp_path: Path, invalid_at: datetime
) -> None:
    root = make_platform_workspace(tmp_path)
    service, workflow_run_id = _start_waiting(root)
    approval = ApprovalAuthorityService(root).issue_approval(
        workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key="approval_authority_grant_time",
        at=WAIT_STARTED + timedelta(seconds=10),
    )

    with pytest.raises(WorkflowError, match="timezone"):
        service.grant_approval(
            workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key="approval_authority_invalid_time_decision",
            at=invalid_at,
        )

    assert _step_head(root, workflow_run_id).state == "waiting"


def test_grant_rejects_execution_approval_without_generic_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_platform_workspace(tmp_path)
    service, workflow_run_id = _start_waiting(root)
    seed = ExecutionApproval(
        approval_id="xapr_pending",
        action=GATE.action,
        payload_sha256=sha256_hex(canonical_json_bytes({"seed": "value"})),
        run_id=derive_id("wfappr", {"node_id": "step_gate", "workflow_run_id": workflow_run_id}),
        scope=(GATE.required_role,),
        approved_by=APPROVER,
        approved_at=WAIT_STARTED + timedelta(seconds=10),
        expires_at=WAIT_STARTED + timedelta(seconds=GATE.expires_after_seconds),
    )
    approval = seed.model_copy(update={"approval_id": derive_id("xapr", seed.identity_payload())})
    with ExecutionStore.open_for_write(root / "ops" / "control.sqlite") as store:
        store.put(approval, expected_revision=None)

    def _generic_fallback_forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("workflow grant consulted the generic authority fallback")

    monkeypatch.setattr(AuthorityResolver, "require_approval", _generic_fallback_forbidden)

    with pytest.raises(WorkflowError, match="authority"):
        service.grant_approval(
            workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key="approval_authority_execution_decision",
            at=WAIT_STARTED + timedelta(seconds=20),
        )

    assert _step_head(root, workflow_run_id).state == "waiting"


def test_grant_rejects_file_only_approval_fallback(tmp_path: Path) -> None:
    root = make_platform_workspace(tmp_path)
    service, workflow_run_id = _start_waiting(root)
    approval = _forged_workflow_approval(workflow_run_id)
    accepted = root / "ops" / "approvals" / "accepted"
    accepted.mkdir(parents=True, exist_ok=True)
    (accepted / f"{approval.approval_id}.json").write_text(
        json.dumps(approval.model_dump(mode="json"), sort_keys=True),
        encoding="utf-8",
    )
    (accepted / f"{approval.approval_id}.verification.json").write_text(
        json.dumps({"approval_id": approval.approval_id}),
        encoding="utf-8",
    )

    with pytest.raises(WorkflowError, match="authority"):
        service.grant_approval(
            workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key="approval_authority_file_decision",
            at=WAIT_STARTED + timedelta(seconds=20),
        )

    assert _step_head(root, workflow_run_id).state == "waiting"


def test_grant_preserves_emergency_revocation_in_locked_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_platform_workspace(tmp_path)
    service, revision = _register(root)
    run = service.start(
        revision.revision_id,
        {"seed": "value"},
        actor_id=ACTOR,
        idempotency_key="workflow_authority_revoked_grant",
    )
    service.run_ready_steps(run.workflow_run_id, at=datetime.now(UTC))
    approval = ApprovalAuthorityService(root).issue_approval(
        run.workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key="approval_authority_revoked_grant",
    )
    time.sleep(0.002)
    EmergencyService(root).activate(
        (EmergencyAction.REVOKE_PENDING_APPROVALS,),
        reason="approval provenance incident",
        actor_id=ACTOR,
        idempotency_key="emergency_revoke_workflow_approval",
    )
    original = AuthorityResolver.require_workflow_approval
    observed_locked_store = False

    def _observe_locked_store(
        self: AuthorityResolver,
        store: PlatformStore,
        approval_id: str,
        **kwargs: Any,
    ) -> ApprovalRecord:
        nonlocal observed_locked_store
        observed_locked_store = store.connection.in_transaction
        return original(self, store, approval_id, **kwargs)

    def _read_only_authority_forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("workflow grant opened a second authority snapshot")

    monkeypatch.setattr(AuthorityResolver, "require_workflow_approval", _observe_locked_store)
    monkeypatch.setattr(
        authority_module,
        "open_platform_store_read_only",
        _read_only_authority_forbidden,
    )

    with pytest.raises(WorkflowError, match="authority"):
        service.grant_approval(
            run.workflow_run_id,
            "step_gate",
            approval_id=approval.approval_id,
            actor_id=ACTOR,
            idempotency_key="approval_authority_revoked_decision",
        )

    assert observed_locked_store is True
    assert _step_head(root, run.workflow_run_id).state == "waiting"


def test_grant_default_time_is_sampled_after_writer_lock_before_expiry_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = make_platform_workspace(tmp_path)
    short_gate = GATE.model_copy(
        update={
            "approval_gate_id": "wgate_authority_grant_lock",
            "expires_after_seconds": 1,
        }
    )
    service, revision = _register(root, short_gate)
    run = service.start(
        revision.revision_id,
        {"seed": "value"},
        actor_id=ACTOR,
        idempotency_key="workflow_authority_grant_lock",
    )
    service.run_ready_steps(run.workflow_run_id, at=datetime.now(UTC))
    approval = ApprovalAuthorityService(root).issue_approval(
        run.workflow_run_id,
        "step_gate",
        approver=APPROVER,
        idempotency_key="approval_authority_grant_lock",
    )
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
                service.grant_approval,
                run.workflow_run_id,
                "step_gate",
                approval_id=approval.approval_id,
                actor_id=ACTOR,
                idempotency_key="approval_authority_lock_decision",
            )
            assert transaction_attempted.wait(timeout=1)
            time.sleep(1.2)
        with pytest.raises(WorkflowError, match="expired"):
            result.result(timeout=5)

    record = _step_head(root, run.workflow_run_id)
    assert record.state == "failed"
    assert record.payload["failure_reason"] == "approval_expired"
    assert record.payload.get("approval_id") is None
    assert "output" not in record.payload
    with initialize_platform_store(root) as store:
        step_states = tuple(
            str(row[0])
            for row in store.connection.execute(
                "SELECT state FROM records WHERE kind='workflow_step' AND record_id=?",
                (record.record_id,),
            ).fetchall()
        )
        event_types = tuple(event["event_type"] for event in store.list_events(run.workflow_run_id))
    assert "succeeded" not in step_states
    assert "workflow_approval_granted" not in event_types


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

    with ThreadPoolExecutor(max_workers=1) as executor, initialize_platform_store(root) as blocker:
        with blocker.transaction():
            result = executor.submit(_call)
            assert transaction_attempted.wait(timeout=1)
            time.sleep(1.2)
        with pytest.raises(AuthorityError, match="expired"):
            result.result(timeout=5)
    assert _approval_count(root) == 0
