from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from raytsystem.authority import (
    ApprovalAuthorityService,
    AuthorityError,
    AuthorityResolver,
    workflow_approval_target,
)
from raytsystem.contracts import (
    PendingWorkflowApproval,
    WorkflowApprovalGate,
    WorkflowDefinition,
    WorkflowNode,
    WorkflowRetryPolicy,
    WorkflowRevision,
    WorkflowRun,
    WorkflowStepRun,
    canonical_json_bytes,
    derive_id,
    sha256_hex,
)
from raytsystem.contracts.workflows import WorkflowNodeType
from raytsystem.emergency import EmergencyService
from raytsystem.features import FeatureConfig, load_feature_config
from raytsystem.platform_store import (
    PlatformStore,
    PlatformStoreError,
    StoredRecord,
    initialize_platform_store,
    open_platform_store_read_only,
)
from raytsystem.security.sensitivity import SecretScanner

Operation = Callable[[dict[str, Any]], dict[str, Any]]

_APPROVAL_ACTION = "workflow_approval"
_ENGINE_ACTOR = "raytsystem_workflow_engine"
_CANCELLABLE_RUN_STATES = frozenset({"planned", "running", "paused"})
_DONE_STEP_STATES = frozenset({"succeeded", "skipped"})
_STEP_EXTRA_KEYS = frozenset({"output", "failure_reason"})
_APPROVAL_DECISION_SCOPE = "workflow_approval_decision"


class _WorkflowRunProjection(BaseModel):
    """Typed immutable identity of a run across legitimate state progression."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_name: Literal["WorkflowRunV1"]
    schema_version: str
    id_scheme_version: str
    extensions: dict[str, Any]
    workflow_run_id: str
    revision_id: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    step_run_ids: tuple[str, ...]
    replay_of_run_id: str | None
    started_at: datetime

    @model_validator(mode="after")
    def _valid_workflow_run_projection(self) -> _WorkflowRunProjection:
        WorkflowRun.model_validate(
            self.model_dump(mode="python") | {"state": "running", "completed_at": None}
        )
        return self

    @classmethod
    def from_run(cls, run: WorkflowRun) -> _WorkflowRunProjection:
        return cls.model_validate(run.model_dump(mode="python", exclude={"state", "completed_at"}))


class _WorkflowDecisionReceipt(BaseModel):
    """Immutable internal proof for one exact workflow approval transition."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=False)

    receipt_id: str
    idempotency_key: str = Field(min_length=1, max_length=256)
    request_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    decision: Literal["grant", "deny"]
    workflow_run_id: str
    step_run_id: str
    node_id: str
    approval_id: str | None
    actor_id: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    approval_gate_id: str
    gate_action: str
    required_role: str
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_version: str
    gate_expires_at: datetime
    run_projection: _WorkflowRunProjection
    expected_pending: PendingWorkflowApproval | None = None
    decided_at: datetime
    event_id: str
    terminal_step_state: Literal["succeeded", "failed"]
    step_revision: int = Field(ge=1)
    run_revision: int = Field(ge=1)
    result: WorkflowRun

    @model_validator(mode="after")
    def _exact_binding(self) -> _WorkflowDecisionReceipt:
        if (
            not self.idempotency_key
            or any(character.isspace() for character in self.idempotency_key)
            or self.gate_expires_at.tzinfo is None
            or self.gate_expires_at.utcoffset() is None
            or self.decided_at.tzinfo is None
            or self.decided_at.utcoffset() is None
            or self.result.workflow_run_id != self.workflow_run_id
            or _WorkflowRunProjection.from_run(self.result) != self.run_projection
        ):
            raise ValueError("Workflow decision receipt binding is invalid")
        expected_state = "succeeded" if self.decision == "grant" else "failed"
        expected_run_state = "running" if self.decision == "grant" else "failed"
        if (
            self.terminal_step_state != expected_state
            or self.result.state != expected_run_state
            or (self.decision == "grant") != (self.approval_id is not None)
            or (self.decision == "grant") != (self.expected_pending is None)
            or (self.decision == "grant" and self.decided_at >= self.gate_expires_at)
        ):
            raise ValueError("Workflow decision receipt result is invalid")
        if self.decision == "deny":
            pending = self.expected_pending
            if pending is None or (
                pending.workflow_run_id != self.workflow_run_id
                or pending.revision_id != self.run_projection.revision_id
                or pending.step_run_id != self.step_run_id
                or pending.node_id != self.node_id
                or pending.approval_gate_id != self.approval_gate_id
                or pending.action != self.gate_action
                or pending.target_id != workflow_approval_target(self.workflow_run_id, self.node_id)
                or pending.input_sha256 != self.input_sha256
                or pending.scope_sha256 != self.policy_sha256
                or pending.policy_version != self.policy_version
                or pending.required_role != self.required_role
                or pending.expires_at != self.gate_expires_at
            ):
                raise ValueError("Workflow denial receipt expected binding is invalid")
        if self.request_sha256 != sha256_hex(canonical_json_bytes(self.request_payload())):
            raise ValueError("Workflow decision receipt request hash is invalid")
        return self

    def request_payload(self) -> dict[str, Any]:
        payload = {
            "workflow_run_id": self.workflow_run_id,
            "step_run_id": self.step_run_id,
            "node_id": self.node_id,
            "decision": self.decision,
            "approval_id": self.approval_id,
            "actor_id": self.actor_id,
            "input_sha256": self.input_sha256,
            "approval_gate_id": self.approval_gate_id,
            "gate_action": self.gate_action,
            "required_role": self.required_role,
            "policy_sha256": self.policy_sha256,
            "policy_version": self.policy_version,
            "gate_expires_at": self.gate_expires_at,
            "run_projection": self.run_projection.model_dump(mode="python"),
        }
        if self.decision == "deny":
            if self.expected_pending is None:
                raise ValueError("Workflow denial receipt is missing its expected binding")
            payload["expected_pending"] = self.expected_pending.model_dump(mode="python")
        return payload

    def identity_payload(self) -> dict[str, Any]:
        excluded = {"receipt_id"}
        if self.expected_pending is None:
            excluded.add("expected_pending")
        return self.model_dump(mode="python", exclude=excluded)

    def storage_payload(self) -> dict[str, Any]:
        excluded: set[str] = set()
        if self.expected_pending is None:
            excluded.add("expected_pending")
        return self.model_dump(mode="json", exclude=excluded)

    def verify_id(self) -> bool:
        return self.receipt_id == derive_id("wdec", self.identity_payload())


@dataclass(frozen=True)
class _WorkflowDecisionContext:
    run: WorkflowRun
    run_record: StoredRecord
    inputs: dict[str, Any]
    node: WorkflowNode
    step: WorkflowStepRun
    step_record: StoredRecord
    gate: WorkflowApprovalGate
    gate_expires_at: datetime


def _identity_operation(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"input_sha256": sha256_hex(canonical_json_bytes(inputs))}


def _summarize_keys_operation(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"keys": sorted(str(key) for key in inputs)}


_BUILTIN_OPERATIONS: dict[str, Operation] = {
    "identity": _identity_operation,
    "summarize_keys": _summarize_keys_operation,
}


def _retry_delay_ms(policy: WorkflowRetryPolicy, attempt: int) -> int:
    if policy.backoff == "linear":
        delay = policy.initial_delay_ms * attempt
    elif policy.backoff == "exponential":
        delay = policy.initial_delay_ms * (2 ** (attempt - 1))
    else:
        delay = policy.initial_delay_ms
    return min(delay, policy.maximum_delay_ms) if policy.maximum_delay_ms else delay


class WorkflowError(RuntimeError):
    """Workflow validation or execution violates DAG, policy, or idempotency constraints."""


class WorkflowService:
    def __init__(
        self,
        root: Path,
        *,
        operations: dict[str, Operation] | None = None,
        features: FeatureConfig | None = None,
        scanner: SecretScanner | None = None,
    ) -> None:
        self.root = root.resolve()
        self.features = features or load_feature_config(self.root)
        if operations:
            raise WorkflowError("Custom workflow callables are forbidden")
        self.operations = dict(_BUILTIN_OPERATIONS)
        self.scanner = scanner or SecretScanner()

    def register(
        self,
        definition: WorkflowDefinition,
        revision: WorkflowRevision,
        *,
        actor_id: str,
        retry_policies: tuple[WorkflowRetryPolicy, ...] = (),
        approval_gates: tuple[WorkflowApprovalGate, ...] = (),
    ) -> WorkflowRevision:
        self._require_enabled()
        if revision.workflow_id != definition.workflow_id:
            raise WorkflowError("Workflow definition and revision IDs differ")
        self.validate_dag(revision)
        calculated = sha256_hex(
            canonical_json_bytes(revision.model_dump(mode="json", exclude={"manifest_sha256"}))
        )
        if calculated != revision.manifest_sha256:
            raise WorkflowError("Workflow revision manifest hash is invalid")
        with initialize_platform_store(self.root) as store:
            for policy in retry_policies:
                self._store_immutable(
                    store,
                    kind="workflow_retry_policy",
                    record_id=policy.retry_policy_id,
                    payload=policy.model_dump(mode="json"),
                )
            for gate in approval_gates:
                self._store_immutable(
                    store,
                    kind="workflow_approval_gate",
                    record_id=gate.approval_gate_id,
                    payload=gate.model_dump(mode="json"),
                )
            self._require_node_references(store, revision)
            existing = store.head("workflow_revision", revision.revision_id)
            if existing is not None:
                if existing.payload_sha256 != sha256_hex(
                    canonical_json_bytes(revision.model_dump(mode="json"))
                ):
                    raise WorkflowError("Workflow revision is immutable")
                return revision
            prior_definition = store.head("workflow", definition.workflow_id)
            store.append_record(
                kind="workflow",
                record_id=definition.workflow_id,
                payload=definition.model_copy(
                    update={"current_revision_id": revision.revision_id}
                ).model_dump(mode="json"),
                state="enabled" if definition.enabled else "disabled",
                expected_revision=None if prior_definition is None else prior_definition.revision,
            )
            store.append_record(
                kind="workflow_revision",
                record_id=revision.revision_id,
                payload=revision.model_dump(mode="json"),
                state="validated",
                expected_revision=None,
            )
            store.append_event(
                stream_id=definition.workflow_id,
                aggregate_id=revision.revision_id,
                event_type="workflow_registered",
                actor_id=actor_id,
                payload_schema="workflow_revision_v1",
                payload={
                    "workflow_id": definition.workflow_id,
                    "revision_id": revision.revision_id,
                    "manifest_sha256": revision.manifest_sha256,
                },
            )
        return revision

    def validate_dag(self, revision: WorkflowRevision) -> tuple[str, ...]:
        nodes = {node.node_id: node for node in revision.nodes}
        indegree = {node_id: 0 for node_id in nodes}
        outgoing: dict[str, list[str]] = defaultdict(list)
        edge_pairs: set[tuple[str, str]] = set()
        for edge in revision.edges:
            pair = (edge.source_node_id, edge.target_node_id)
            if pair in edge_pairs:
                raise WorkflowError("Duplicate workflow edge")
            edge_pairs.add(pair)
            outgoing[edge.source_node_id].append(edge.target_node_id)
            indegree[edge.target_node_id] += 1
        queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
        ordered: list[str] = []
        while queue:
            node_id = queue.popleft()
            ordered.append(node_id)
            for target in sorted(outgoing[node_id]):
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)
        if len(ordered) != len(nodes):
            raise WorkflowError("Workflow DAG contains a cycle")
        for node in nodes.values():
            if node.node_type is WorkflowNodeType.DETERMINISTIC_COMMAND:
                if node.operation_id is None or node.operation_id not in self.operations:
                    raise WorkflowError("Deterministic commands require a registered operation ID")
                if any(token in node.operation_id for token in (" ", ";", "|", "&", "$", "`")):
                    raise WorkflowError("Raw shell commands are forbidden")
        return tuple(ordered)

    def start(
        self,
        revision_id: str,
        inputs: dict[str, Any],
        *,
        actor_id: str,
        idempotency_key: str,
        replay_of_run_id: str | None = None,
    ) -> WorkflowRun:
        self._require_enabled()
        EmergencyService(self.root, features=self.features).assert_runtime_allowed()
        input_bytes = self._safe_payload(inputs, label="Workflow inputs")
        input_sha256 = sha256_hex(input_bytes)
        request = {
            "revision_id": revision_id,
            "inputs_sha256": input_sha256,
            "replay_of_run_id": replay_of_run_id,
        }
        workflow_run_id = derive_id("wrun", {"idempotency_key": idempotency_key, **request})
        with initialize_platform_store(self.root) as store:
            existing = store.head("workflow_run", workflow_run_id)
            if existing is not None:
                existing_payload = dict(existing.payload)
                existing_payload.pop("inputs", None)
                return WorkflowRun.model_validate(existing_payload)
            revision = self._revision(store, revision_id)
            definition_record = store.head("workflow", revision.workflow_id)
            if (
                definition_record is None
                or definition_record.state != "enabled"
                or definition_record.payload.get("current_revision_id") != revision_id
            ):
                raise WorkflowError("Workflow revision is not the enabled current revision")
            order = self.validate_dag(revision)
            now = datetime.now(UTC)
            steps: list[WorkflowStepRun] = []
            for node_id in order:
                step = WorkflowStepRun(
                    step_run_id=derive_id(
                        "wstep", {"workflow_run_id": workflow_run_id, "node_id": node_id}
                    ),
                    workflow_run_id=workflow_run_id,
                    node_id=node_id,
                    state="pending",
                    idempotency_key=derive_id(
                        "idem", {"workflow_run_id": workflow_run_id, "node_id": node_id}
                    ),
                    input_sha256=input_sha256,
                )
                steps.append(step)
                store.append_record(
                    kind="workflow_step",
                    record_id=step.step_run_id,
                    payload=step.model_dump(mode="json"),
                    state=step.state,
                    expected_revision=None,
                )
            run = WorkflowRun(
                workflow_run_id=workflow_run_id,
                revision_id=revision_id,
                state="running",
                input_sha256=input_sha256,
                step_run_ids=tuple(step.step_run_id for step in steps),
                replay_of_run_id=replay_of_run_id,
                started_at=now,
            )
            store.append_record(
                kind="workflow_run",
                record_id=workflow_run_id,
                payload=run.model_dump(mode="json") | {"inputs": inputs},
                state=run.state,
                expected_revision=None,
            )
            store.append_event(
                stream_id=workflow_run_id,
                aggregate_id=workflow_run_id,
                event_type="workflow_started",
                actor_id=actor_id,
                payload_schema="workflow_run_v1",
                payload={"revision_id": revision_id, "step_count": len(steps)},
            )
            return run

    def run_ready_steps(self, workflow_run_id: str, *, at: datetime | None = None) -> WorkflowRun:
        self._require_enabled()
        EmergencyService(self.root, features=self.features).assert_runtime_allowed()
        now = (at or datetime.now(UTC)).astimezone(UTC)
        with initialize_platform_store(self.root) as store:
            run, run_record, initial_inputs = self._load_run(store, workflow_run_id)
            if run.state != "running":
                return run
            revision = self._revision(store, run.revision_id)
            order = self.validate_dag(revision)
            node_by_id = {node.node_id: node for node in revision.nodes}
            dependencies: dict[str, set[str]] = defaultdict(set)
            for edge in revision.edges:
                dependencies[edge.target_node_id].add(edge.source_node_id)
            records = self._step_records(store, run)
            states = {node_id: record.state for node_id, record in records.items()}
            inputs = dict(initial_inputs)
            for node_id in order:
                if records[node_id].state == "succeeded":
                    output = records[node_id].payload.get("output")
                    if isinstance(output, dict):
                        inputs[node_id] = output
            failed = False
            blocked = False
            for node_id in order:
                state = states[node_id]
                if state in _DONE_STEP_STATES:
                    continue
                if state in {"failed", "cancelled"}:
                    failed = True
                    break
                if not all(states[parent] in _DONE_STEP_STATES for parent in dependencies[node_id]):
                    continue
                node = node_by_id[node_id]
                record = records[node_id]
                step = self._step_from_record(record)
                if state in {"waiting", "running"}:
                    if self._expire_step(store, node, step, record, now):
                        failed = True
                    else:
                        blocked = True
                    break
                if state == "paused":
                    blocked = True
                    break
                if node.node_type in {WorkflowNodeType.APPROVAL, WorkflowNodeType.WAIT}:
                    waiting = step.model_copy(update={"state": "waiting", "started_at": now})
                    self._persist_step(store, waiting, record.revision)
                    states[node_id] = "waiting"
                    blocked = True
                    break
                if node.node_type is not WorkflowNodeType.DETERMINISTIC_COMMAND:
                    paused_step = step.model_copy(update={"state": "paused", "started_at": now})
                    self._persist_step(store, paused_step, record.revision)
                    states[node_id] = "paused"
                    blocked = True
                    break
                output = self._drive_deterministic(store, node, step, record, dict(inputs), now)
                if output is None:
                    failed = True
                    break
                states[node_id] = "succeeded"
                inputs[node_id] = output
            final_state = (
                "failed"
                if failed
                else "succeeded"
                if not blocked and all(state in _DONE_STEP_STATES for state in states.values())
                else "running"
            )
            updated = run.model_copy(
                update={
                    "state": final_state,
                    "completed_at": now if final_state in {"failed", "succeeded"} else None,
                }
            )
            self._persist_run(store, updated, run_record.revision, initial_inputs)
            store.append_event(
                stream_id=workflow_run_id,
                aggregate_id=workflow_run_id,
                event_type="workflow_state_changed",
                actor_id=_ENGINE_ACTOR,
                payload_schema="workflow_run_v1",
                payload={"state": final_state},
            )
            return updated

    def grant_approval(
        self,
        workflow_run_id: str,
        node_id: str,
        *,
        approval_id: str,
        actor_id: str,
        idempotency_key: str,
        at: datetime | None = None,
    ) -> WorkflowRun:
        self._require_enabled()
        EmergencyService(self.root, features=self.features).assert_runtime_allowed()
        self._require_decision_key(idempotency_key)
        if not approval_id:
            raise WorkflowError("Workflow approval requires an explicit approval ID")
        requested_at = self._decision_time(at)
        expired = False
        granted_run: WorkflowRun | None = None
        with initialize_platform_store(self.root) as store, store.transaction():
            now = requested_at or datetime.now(UTC)
            context = self._workflow_decision_context(store, workflow_run_id, node_id)
            request = self._workflow_decision_request(
                context,
                decision="grant",
                approval_id=approval_id,
                actor_id=actor_id,
            )
            prior = self._workflow_decision_receipt(
                store,
                idempotency_key=idempotency_key,
                request=request,
            )
            if prior is not None:
                return self._replay_workflow_decision(store, context, prior)
            self._require_waiting_decision(context)
            waiting_since = context.step.started_at
            if waiting_since is None:
                raise WorkflowError("Workflow approval waiting time is invalid")
            if now >= context.gate_expires_at:
                failed = context.step.model_copy(update={"state": "failed", "completed_at": now})
                self._persist_step(
                    store,
                    failed,
                    context.step_record.revision,
                    extra={"failure_reason": "approval_expired"},
                )
                expired = True
            else:
                try:
                    approval = AuthorityResolver(self.root).require_workflow_approval(
                        store,
                        approval_id,
                        action=context.gate.action,
                        target_id=workflow_approval_target(workflow_run_id, node_id),
                        artifact_sha256=context.run.input_sha256,
                        required_role=context.gate.required_role,
                        policy_version=context.gate.schema_version,
                        policy_sha256=context.gate.scope_sha256,
                        waiting_since=waiting_since,
                        gate_expires_at=context.gate_expires_at,
                        at=now,
                    )
                except AuthorityError as error:
                    raise WorkflowError("Workflow approval authority is invalid") from error
                output = {"approval_id": approval.approval_id, "granted_by": actor_id}
                output_bytes = self._safe_payload(output, label="Workflow output")
                granted = context.step.model_copy(
                    update={
                        "state": "succeeded",
                        "approval_id": approval.approval_id,
                        "output_sha256": sha256_hex(output_bytes),
                        "completed_at": now,
                    }
                )
                persisted_step = self._persist_step(
                    store,
                    granted,
                    context.step_record.revision,
                    extra={"output": output},
                )
                event = store.append_event(
                    stream_id=workflow_run_id,
                    aggregate_id=context.step.step_run_id,
                    event_type="workflow_approval_granted",
                    actor_id=actor_id,
                    payload_schema="workflow_step_run_v1",
                    payload={"node_id": node_id, "approval_id": approval.approval_id},
                )
                receipt = self._new_workflow_decision_receipt(
                    idempotency_key=idempotency_key,
                    request=request,
                    context=context,
                    decided_at=now,
                    event_id=str(event["event_id"]),
                    terminal_step_state="succeeded",
                    step_revision=persisted_step.revision,
                    run_revision=context.run_record.revision,
                    result=context.run,
                )
                self._store_workflow_decision_receipt(store, receipt)
                granted_run = context.run
        if expired:
            raise WorkflowError("Workflow approval gate has expired")
        if granted_run is None:
            raise WorkflowError("Workflow approval transition did not complete")
        return granted_run

    def deny_approval(
        self,
        workflow_run_id: str,
        node_id: str,
        *,
        expected: PendingWorkflowApproval,
        actor_id: str,
        idempotency_key: str,
        at: datetime | None = None,
    ) -> WorkflowRun:
        self._require_enabled()
        EmergencyService(self.root, features=self.features).assert_runtime_allowed()
        self._require_decision_key(idempotency_key)
        self._require_expected_pending(expected)
        requested_at = self._decision_time(at)
        with initialize_platform_store(self.root) as store, store.transaction():
            now = requested_at or datetime.now(UTC)
            context = self._workflow_decision_context(store, workflow_run_id, node_id)
            request = self._workflow_decision_request(
                context,
                decision="deny",
                approval_id=None,
                actor_id=actor_id,
                expected=expected,
            )
            prior = self._workflow_decision_receipt(
                store,
                idempotency_key=idempotency_key,
                request=request,
            )
            if prior is not None:
                return self._replay_workflow_decision(store, context, prior)
            try:
                live = (
                    ApprovalAuthorityService(self.root)
                    ._pending_context(
                        store,
                        workflow_run_id,
                        node_id,
                        now,
                    )
                    .pending
                )
            except AuthorityError as error:
                raise WorkflowError("Workflow denial pending binding is unavailable") from error
            if live != expected:
                raise WorkflowError("Workflow denial expected binding does not match live state")
            self._require_waiting_decision(context)
            denied = context.step.model_copy(update={"state": "failed", "completed_at": now})
            persisted_step = self._persist_step(
                store,
                denied,
                context.step_record.revision,
                extra={"failure_reason": "approval_denied"},
            )
            failed_run = context.run.model_copy(update={"state": "failed", "completed_at": now})
            persisted_run = self._persist_run(
                store,
                failed_run,
                context.run_record.revision,
                context.inputs,
            )
            event = store.append_event(
                stream_id=workflow_run_id,
                aggregate_id=context.step.step_run_id,
                event_type="workflow_approval_denied",
                actor_id=actor_id,
                payload_schema="workflow_step_run_v1",
                payload={"node_id": node_id},
            )
            receipt = self._new_workflow_decision_receipt(
                idempotency_key=idempotency_key,
                request=request,
                context=context,
                decided_at=now,
                event_id=str(event["event_id"]),
                terminal_step_state="failed",
                step_revision=persisted_step.revision,
                run_revision=persisted_run.revision,
                result=failed_run,
            )
            self._store_workflow_decision_receipt(store, receipt)
            return failed_run

    def wake(
        self,
        workflow_run_id: str,
        node_id: str,
        *,
        actor_id: str,
        at: datetime | None = None,
    ) -> WorkflowRun:
        self._require_enabled()
        EmergencyService(self.root, features=self.features).assert_runtime_allowed()
        now = (at or datetime.now(UTC)).astimezone(UTC)
        with initialize_platform_store(self.root) as store:
            run, _, _ = self._load_run(store, workflow_run_id, expected_state="running")
            node, step, record = self._waiting_step(
                store, run, node_id, expected=WorkflowNodeType.WAIT
            )
            deadline = timedelta(seconds=node.timeout_seconds)
            if step.started_at is not None and now >= step.started_at + deadline:
                timed_out = step.model_copy(update={"state": "failed", "completed_at": now})
                self._persist_step(
                    store, timed_out, record.revision, extra={"failure_reason": "timeout"}
                )
                raise WorkflowError("Workflow wait step has timed out")
            output = {"woken_by": actor_id}
            output_bytes = self._safe_payload(output, label="Workflow output")
            woken = step.model_copy(
                update={
                    "state": "succeeded",
                    "output_sha256": sha256_hex(output_bytes),
                    "completed_at": now,
                }
            )
            self._persist_step(store, woken, record.revision, extra={"output": output})
            store.append_event(
                stream_id=workflow_run_id,
                aggregate_id=step.step_run_id,
                event_type="workflow_step_woken",
                actor_id=actor_id,
                payload_schema="workflow_step_run_v1",
                payload={"node_id": node_id},
            )
            return run

    def pause(self, workflow_run_id: str, *, actor_id: str) -> WorkflowRun:
        self._require_enabled()
        return self._transition_run(
            workflow_run_id,
            actor_id=actor_id,
            expected_state="running",
            next_state="paused",
            event_type="workflow_paused",
        )

    def resume(self, workflow_run_id: str, *, actor_id: str) -> WorkflowRun:
        self._require_enabled()
        return self._transition_run(
            workflow_run_id,
            actor_id=actor_id,
            expected_state="paused",
            next_state="running",
            event_type="workflow_resumed",
        )

    def cancel(self, workflow_run_id: str, *, actor_id: str) -> WorkflowRun:
        self._require_enabled()
        now = datetime.now(UTC)
        with initialize_platform_store(self.root) as store:
            run, run_record, inputs = self._load_run(store, workflow_run_id)
            if run.state not in _CANCELLABLE_RUN_STATES:
                raise WorkflowError("Terminal workflow runs cannot be cancelled")
            for step_id in run.step_run_ids:
                record = store.head("workflow_step", step_id)
                if record is None:
                    raise WorkflowError("Workflow step record is missing")
                if record.state in {"pending", "waiting", "paused", "running"}:
                    step = self._step_from_record(record)
                    cancelled_step = step.model_copy(
                        update={"state": "cancelled", "completed_at": now}
                    )
                    self._persist_step(store, cancelled_step, record.revision)
            cancelled = run.model_copy(update={"state": "cancelled", "completed_at": now})
            self._persist_run(store, cancelled, run_record.revision, inputs)
            store.append_event(
                stream_id=workflow_run_id,
                aggregate_id=workflow_run_id,
                event_type="workflow_cancelled",
                actor_id=actor_id,
                payload_schema="workflow_run_v1",
                payload={"workflow_run_id": workflow_run_id},
            )
            return cancelled

    def snapshot(self) -> dict[str, Any]:
        store = open_platform_store_read_only(self.root)
        if store is None:
            return {
                "snapshot_id": "pview_unavailable",
                "state": "unavailable",
                "workflows": [],
                "runs": [],
                "graph": [],
            }
        with store:
            return {
                "snapshot_id": store.snapshot_id(),
                "state": "ready"
                if self.features.enabled("workflow_engine_enabled")
                else "disabled",
                "workflows": [record.payload for record in store.list_heads("workflow", limit=200)],
                "runs": [
                    {key: value for key, value in record.payload.items() if key != "inputs"}
                    for record in store.list_heads("workflow_run", limit=200)
                ],
                "graph": self._graph_payload(store),
            }

    def _graph_payload(self, store: PlatformStore) -> list[dict[str, Any]]:
        graphs: list[dict[str, Any]] = []
        for workflow in store.list_heads("workflow", limit=50):
            revision_id = workflow.payload.get("current_revision_id")
            if not isinstance(revision_id, str):
                continue
            record = store.head("workflow_revision", revision_id)
            if record is None:
                continue
            try:
                revision = WorkflowRevision.model_validate(record.payload)
            except ValidationError:
                continue
            graphs.append(
                {
                    "workflow_id": revision.workflow_id,
                    "revision_id": revision.revision_id,
                    "nodes": [
                        {
                            "node_id": node.node_id,
                            "node_type": node.node_type.value,
                            "name": node.name,
                            "operation_id": node.operation_id,
                            "timeout_seconds": node.timeout_seconds,
                        }
                        for node in revision.nodes[:200]
                    ],
                    "edges": [
                        {
                            "edge_id": edge.edge_id,
                            "source_node_id": edge.source_node_id,
                            "target_node_id": edge.target_node_id,
                        }
                        for edge in revision.edges[:400]
                    ],
                }
            )
        return graphs

    def _drive_deterministic(
        self,
        store: PlatformStore,
        node: WorkflowNode,
        step: WorkflowStepRun,
        record: StoredRecord,
        inputs: dict[str, Any],
        now: datetime,
    ) -> dict[str, Any] | None:
        operation = self.operations.get(str(node.operation_id))
        if operation is None:
            raise WorkflowError("Registered operation disappeared")
        timeout = timedelta(seconds=node.timeout_seconds)
        if step.started_at is not None and now >= step.started_at + timeout:
            self._fail_step(store, step, record.revision, now, reason="timeout")
            return None
        policy = self._retry_policy(store, node)
        max_attempts = 1 if policy is None else policy.max_attempts
        attempt = step.attempt
        revision = record.revision
        current = step
        while True:
            try:
                output = operation(dict(inputs))
            except Exception:
                if policy is not None and attempt < max_attempts:
                    delay_ms = _retry_delay_ms(policy, attempt)
                    attempt += 1
                    current = current.model_copy(update={"attempt": attempt, "started_at": now})
                    revision = self._persist_step(store, current, revision).revision
                    store.append_event(
                        stream_id=step.workflow_run_id,
                        aggregate_id=step.step_run_id,
                        event_type="workflow_step_retry",
                        actor_id=_ENGINE_ACTOR,
                        payload_schema="workflow_step_run_v1",
                        payload={
                            "node_id": step.node_id,
                            "attempt": attempt,
                            "delay_ms": delay_ms,
                        },
                    )
                    continue
                self._fail_step(store, current, revision, now, reason="operation_error")
                return None
            if not isinstance(output, dict):
                self._fail_step(store, current, revision, now, reason="invalid_output")
                return None
            try:
                output_bytes = self._safe_payload(output, label="Workflow output")
            except WorkflowError:
                self._fail_step(store, current, revision, now, reason="output_rejected")
                return None
            completed = current.model_copy(
                update={
                    "state": "succeeded",
                    "output_sha256": sha256_hex(output_bytes),
                    "started_at": now,
                    "completed_at": now,
                }
            )
            self._persist_step(store, completed, revision, extra={"output": output})
            store.append_event(
                stream_id=step.workflow_run_id,
                aggregate_id=step.step_run_id,
                event_type="workflow_step_succeeded",
                actor_id=_ENGINE_ACTOR,
                payload_schema="workflow_step_run_v1",
                payload={"node_id": step.node_id, "output_sha256": completed.output_sha256},
            )
            return output

    def _expire_step(
        self,
        store: PlatformStore,
        node: WorkflowNode,
        step: WorkflowStepRun,
        record: StoredRecord,
        now: datetime,
    ) -> bool:
        if node.node_type is WorkflowNodeType.APPROVAL:
            gate = self._approval_gate(store, node)
            limit = timedelta(seconds=gate.expires_after_seconds)
            reason = "approval_expired"
        else:
            limit = timedelta(seconds=node.timeout_seconds)
            reason = "timeout"
        if step.started_at is None or now < step.started_at + limit:
            return False
        self._fail_step(store, step, record.revision, now, reason=reason)
        return True

    def _fail_step(
        self,
        store: PlatformStore,
        step: WorkflowStepRun,
        expected_revision: int,
        now: datetime,
        *,
        reason: str,
    ) -> None:
        failed = step.model_copy(
            update={
                "state": "failed",
                "started_at": step.started_at or now,
                "completed_at": now,
            }
        )
        self._persist_step(store, failed, expected_revision, extra={"failure_reason": reason})
        store.append_event(
            stream_id=step.workflow_run_id,
            aggregate_id=step.step_run_id,
            event_type="workflow_step_failed",
            actor_id=_ENGINE_ACTOR,
            payload_schema="workflow_step_run_v1",
            payload={"node_id": step.node_id, "reason": reason, "attempt": step.attempt},
        )

    def _transition_run(
        self,
        workflow_run_id: str,
        *,
        actor_id: str,
        expected_state: str,
        next_state: str,
        event_type: str,
    ) -> WorkflowRun:
        with initialize_platform_store(self.root) as store:
            run, run_record, inputs = self._load_run(store, workflow_run_id)
            if run.state != expected_state:
                raise WorkflowError(f"Workflow run must be {expected_state} for this transition")
            updated = run.model_copy(update={"state": next_state})
            self._persist_run(store, updated, run_record.revision, inputs)
            store.append_event(
                stream_id=workflow_run_id,
                aggregate_id=workflow_run_id,
                event_type=event_type,
                actor_id=actor_id,
                payload_schema="workflow_run_v1",
                payload={"state": next_state},
            )
            return updated

    def _load_run(
        self,
        store: PlatformStore,
        workflow_run_id: str,
        *,
        expected_state: str | None = None,
    ) -> tuple[WorkflowRun, StoredRecord, dict[str, Any]]:
        record = store.head("workflow_run", workflow_run_id)
        if record is None:
            raise WorkflowError("Workflow run does not exist")
        payload = dict(record.payload)
        raw_inputs = payload.pop("inputs", {})
        inputs = dict(raw_inputs) if isinstance(raw_inputs, dict) else {}
        run = WorkflowRun.model_validate(payload)
        if expected_state is not None and run.state != expected_state:
            raise WorkflowError(f"Workflow run must be {expected_state} for this transition")
        return run, record, inputs

    def _persist_run(
        self,
        store: PlatformStore,
        run: WorkflowRun,
        expected_revision: int,
        inputs: dict[str, Any],
    ) -> StoredRecord:
        return store.append_record(
            kind="workflow_run",
            record_id=run.workflow_run_id,
            payload=run.model_dump(mode="json") | {"inputs": inputs},
            state=run.state,
            expected_revision=expected_revision,
        )

    def _persist_step(
        self,
        store: PlatformStore,
        step: WorkflowStepRun,
        expected_revision: int,
        *,
        extra: dict[str, Any] | None = None,
    ) -> StoredRecord:
        return store.append_record(
            kind="workflow_step",
            record_id=step.step_run_id,
            payload=step.model_dump(mode="json") | (extra or {}),
            state=step.state,
            expected_revision=expected_revision,
        )

    def _step_records(self, store: PlatformStore, run: WorkflowRun) -> dict[str, StoredRecord]:
        records: dict[str, StoredRecord] = {}
        for step_id in run.step_run_ids:
            record = store.head("workflow_step", step_id)
            if record is None:
                raise WorkflowError("Workflow step record is missing")
            records[str(record.payload["node_id"])] = record
        return records

    def _step_from_record(self, record: StoredRecord) -> WorkflowStepRun:
        payload = {
            key: value for key, value in record.payload.items() if key not in _STEP_EXTRA_KEYS
        }
        return WorkflowStepRun.model_validate(payload)

    def _workflow_decision_context(
        self,
        store: PlatformStore,
        workflow_run_id: str,
        node_id: str,
    ) -> _WorkflowDecisionContext:
        try:
            run, run_record, inputs = self._load_run(store, workflow_run_id)
            if (
                run_record.kind != "workflow_run"
                or run_record.record_id != workflow_run_id
                or run_record.state != run.state
                or run.workflow_run_id != workflow_run_id
                or not isinstance(run_record.payload.get("inputs"), dict)
                or sha256_hex(canonical_json_bytes(inputs)) != run.input_sha256
            ):
                raise WorkflowError("Workflow run input binding is invalid")
            revision = self._revision(store, run.revision_id)
            node = next((item for item in revision.nodes if item.node_id == node_id), None)
            if node is None or node.node_type is not WorkflowNodeType.APPROVAL:
                raise WorkflowError("Workflow node does not accept this transition")
            matching_steps: list[StoredRecord] = []
            for step_run_id in run.step_run_ids:
                record = store.head("workflow_step", step_run_id)
                if record is None:
                    raise WorkflowError("Workflow step record is missing")
                if record.payload.get("node_id") == node_id:
                    matching_steps.append(record)
            if len(matching_steps) != 1:
                raise WorkflowError("Workflow node has no unique approval step")
            step_record = matching_steps[0]
            step = self._step_from_record(step_record)
            if (
                step_record.kind != "workflow_step"
                or step_record.record_id != step.step_run_id
                or step_record.state != step.state
                or step.step_run_id not in run.step_run_ids
                or step.workflow_run_id != workflow_run_id
                or step.node_id != node_id
                or step.input_sha256 != run.input_sha256
                or step.started_at is None
            ):
                raise WorkflowError("Workflow approval step binding is invalid")
            gate = self._canonical_approval_gate(store, node)
            gate_expires_at = step.started_at.astimezone(UTC) + timedelta(
                seconds=gate.expires_after_seconds
            )
        except WorkflowError:
            raise
        except (PlatformStoreError, ValidationError, TypeError, ValueError) as error:
            raise WorkflowError("Workflow approval decision binding is invalid") from error
        return _WorkflowDecisionContext(
            run=run,
            run_record=run_record,
            inputs=inputs,
            node=node,
            step=step,
            step_record=step_record,
            gate=gate,
            gate_expires_at=gate_expires_at,
        )

    @staticmethod
    def _workflow_decision_request(
        context: _WorkflowDecisionContext,
        *,
        decision: Literal["grant", "deny"],
        approval_id: str | None,
        actor_id: str,
        expected: PendingWorkflowApproval | None = None,
    ) -> dict[str, Any]:
        request = {
            "workflow_run_id": context.run.workflow_run_id,
            "step_run_id": context.step.step_run_id,
            "node_id": context.node.node_id,
            "decision": decision,
            "approval_id": approval_id,
            "actor_id": actor_id,
            "input_sha256": context.run.input_sha256,
            "approval_gate_id": context.gate.approval_gate_id,
            "gate_action": context.gate.action,
            "required_role": context.gate.required_role,
            "policy_sha256": context.gate.scope_sha256,
            "policy_version": context.gate.schema_version,
            "gate_expires_at": context.gate_expires_at,
            "run_projection": _WorkflowRunProjection.from_run(context.run).model_dump(
                mode="python"
            ),
        }
        if decision == "deny":
            if expected is None:
                raise WorkflowError("Workflow denial requires an expected pending binding")
            request["expected_pending"] = expected.model_dump(mode="python")
        return request

    def _workflow_decision_receipt(
        self,
        store: PlatformStore,
        *,
        idempotency_key: str,
        request: dict[str, Any],
    ) -> _WorkflowDecisionReceipt | None:
        try:
            stored = store.idempotent_receipt(
                scope=_APPROVAL_DECISION_SCOPE,
                idempotency_key=idempotency_key,
                request=request,
            )
            if stored is None:
                return None
            receipt = _WorkflowDecisionReceipt.model_validate(stored)
            if (
                receipt.idempotency_key != idempotency_key
                or receipt.request_payload() != request
                or not receipt.verify_id()
            ):
                raise WorkflowError("Workflow approval decision receipt is corrupt")
            return receipt
        except WorkflowError:
            raise
        except (PlatformStoreError, ValidationError, TypeError, ValueError) as error:
            raise WorkflowError(
                "Workflow approval decision idempotency receipt is invalid"
            ) from error

    @staticmethod
    def _new_workflow_decision_receipt(
        *,
        idempotency_key: str,
        request: dict[str, Any],
        context: _WorkflowDecisionContext,
        decided_at: datetime,
        event_id: str,
        terminal_step_state: Literal["succeeded", "failed"],
        step_revision: int,
        run_revision: int,
        result: WorkflowRun,
    ) -> _WorkflowDecisionReceipt:
        draft = _WorkflowDecisionReceipt(
            receipt_id="wdec_pending",
            idempotency_key=idempotency_key,
            request_sha256=sha256_hex(canonical_json_bytes(request)),
            decision=request["decision"],
            workflow_run_id=context.run.workflow_run_id,
            step_run_id=context.step.step_run_id,
            node_id=context.node.node_id,
            approval_id=request["approval_id"],
            actor_id=request["actor_id"],
            input_sha256=context.run.input_sha256,
            approval_gate_id=context.gate.approval_gate_id,
            gate_action=context.gate.action,
            required_role=context.gate.required_role,
            policy_sha256=context.gate.scope_sha256,
            policy_version=context.gate.schema_version,
            gate_expires_at=context.gate_expires_at,
            run_projection=request["run_projection"],
            expected_pending=request.get("expected_pending"),
            decided_at=decided_at,
            event_id=event_id,
            terminal_step_state=terminal_step_state,
            step_revision=step_revision,
            run_revision=run_revision,
            result=result,
        )
        return draft.model_copy(update={"receipt_id": derive_id("wdec", draft.identity_payload())})

    @staticmethod
    def _store_workflow_decision_receipt(
        store: PlatformStore,
        receipt: _WorkflowDecisionReceipt,
    ) -> None:
        stored_payload = receipt.storage_payload()
        stored = store.idempotent_receipt(
            scope=_APPROVAL_DECISION_SCOPE,
            idempotency_key=receipt.idempotency_key,
            request=receipt.request_payload(),
            receipt=stored_payload,
        )
        if stored != stored_payload:
            raise WorkflowError("Workflow approval decision receipt changed concurrently")

    def _replay_workflow_decision(
        self,
        store: PlatformStore,
        context: _WorkflowDecisionContext,
        receipt: _WorkflowDecisionReceipt,
    ) -> WorkflowRun:
        try:
            if _WorkflowRunProjection.from_run(context.run) != receipt.run_projection:
                raise WorkflowError("Workflow approval decision run binding is invalid")
            historical_run = store.record(
                "workflow_run", receipt.workflow_run_id, receipt.run_revision
            )
            if historical_run is None:
                raise WorkflowError("Workflow approval decision receipt is orphaned")
            historical_payload = dict(historical_run.payload)
            historical_inputs = historical_payload.pop("inputs", None)
            if (
                not isinstance(historical_inputs, dict)
                or historical_run.kind != "workflow_run"
                or historical_run.record_id != receipt.workflow_run_id
                or historical_run.state != receipt.result.state
                or WorkflowRun.model_validate(historical_payload) != receipt.result
                or sha256_hex(canonical_json_bytes(historical_inputs)) != receipt.input_sha256
            ):
                raise WorkflowError("Workflow approval decision result receipt is corrupt")
            step = context.step
            step_record = context.step_record
            expected_output = {
                "approval_id": receipt.approval_id,
                "granted_by": receipt.actor_id,
            }
            if (
                step_record.revision != receipt.step_revision
                or step_record.state != receipt.terminal_step_state
                or step.state != receipt.terminal_step_state
                or step.completed_at != receipt.decided_at
            ):
                raise WorkflowError("Workflow approval decision terminal receipt is orphaned")
            if receipt.decision == "grant":
                expected_output_sha256 = sha256_hex(canonical_json_bytes(expected_output))
                if (
                    step.approval_id != receipt.approval_id
                    or step.output_sha256 != expected_output_sha256
                    or step_record.payload.get("output") != expected_output
                    or "failure_reason" in step_record.payload
                ):
                    raise WorkflowError("Workflow approval grant receipt is corrupt")
                waiting_since = step.started_at
                if waiting_since is None or receipt.approval_id is None:
                    raise WorkflowError("Workflow approval grant receipt is corrupt")
                try:
                    AuthorityResolver(self.root).require_workflow_approval(
                        store,
                        receipt.approval_id,
                        action=receipt.gate_action,
                        target_id=workflow_approval_target(
                            receipt.workflow_run_id, receipt.node_id
                        ),
                        artifact_sha256=receipt.input_sha256,
                        required_role=receipt.required_role,
                        policy_version=receipt.policy_version,
                        policy_sha256=receipt.policy_sha256,
                        waiting_since=waiting_since,
                        gate_expires_at=receipt.gate_expires_at,
                        at=receipt.decided_at,
                    )
                except AuthorityError as error:
                    raise WorkflowError(
                        "Workflow approval grant receipt authority is invalid"
                    ) from error
                expected_event_type = "workflow_approval_granted"
                expected_event_payload = {
                    "node_id": receipt.node_id,
                    "approval_id": receipt.approval_id,
                }
            else:
                if (
                    step.approval_id is not None
                    or step.output_sha256 is not None
                    or step_record.payload.get("failure_reason") != "approval_denied"
                    or "output" in step_record.payload
                    or context.run_record.revision != receipt.run_revision
                    or context.run != receipt.result
                ):
                    raise WorkflowError("Workflow approval denial receipt is corrupt")
                expected_event_type = "workflow_approval_denied"
                expected_event_payload = {"node_id": receipt.node_id}
            event = store.event(receipt.event_id)
            if (
                event is None
                or not store.verify_event_stream(receipt.workflow_run_id)
                or event["stream_id"] != receipt.workflow_run_id
                or event["aggregate_id"] != receipt.step_run_id
                or event["event_type"] != expected_event_type
                or event["actor_id"] != receipt.actor_id
                or event["payload_schema"] != "workflow_step_run_v1"
                or event["payload"] != expected_event_payload
            ):
                raise WorkflowError("Workflow approval decision event receipt is orphaned")
            return receipt.result
        except WorkflowError:
            raise
        except (PlatformStoreError, ValidationError, TypeError, ValueError) as error:
            raise WorkflowError("Workflow approval decision receipt is corrupt") from error

    @staticmethod
    def _require_waiting_decision(context: _WorkflowDecisionContext) -> None:
        if context.run.state != "running" or context.run_record.state != "running":
            raise WorkflowError("Workflow run must be running for this transition")
        if (
            context.step.state != "waiting"
            or context.step_record.state != "waiting"
            or context.step.completed_at is not None
            or context.step.approval_id is not None
            or context.step.output_sha256 is not None
            or "output" in context.step_record.payload
            or "failure_reason" in context.step_record.payload
        ):
            raise WorkflowError("Workflow step is not waiting")

    @staticmethod
    def _require_decision_key(idempotency_key: str) -> None:
        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key
            or len(idempotency_key) > 256
            or any(character.isspace() for character in idempotency_key)
        ):
            raise WorkflowError("Workflow approval decision requires an exact idempotency key")

    @staticmethod
    def _require_expected_pending(expected: PendingWorkflowApproval) -> None:
        if type(expected) is not PendingWorkflowApproval:
            raise WorkflowError("Workflow denial requires a typed expected pending binding")
        try:
            validated = PendingWorkflowApproval.model_validate(expected.model_dump(mode="python"))
        except (ValidationError, TypeError, ValueError) as error:
            raise WorkflowError("Workflow denial expected binding is malformed") from error
        if validated != expected:
            raise WorkflowError("Workflow denial expected binding is malformed")

    @staticmethod
    def _decision_time(at: datetime | None) -> datetime | None:
        if at is None:
            return None
        try:
            if at.tzinfo is None or at.utcoffset() is None:
                raise WorkflowError("Workflow approval time must be timezone-aware")
            return at.astimezone(UTC)
        except (TypeError, ValueError) as error:
            raise WorkflowError("Workflow approval time must be timezone-aware") from error

    def _waiting_step(
        self,
        store: PlatformStore,
        run: WorkflowRun,
        node_id: str,
        *,
        expected: WorkflowNodeType,
    ) -> tuple[WorkflowNode, WorkflowStepRun, StoredRecord]:
        revision = self._revision(store, run.revision_id)
        node = next((item for item in revision.nodes if item.node_id == node_id), None)
        if node is None or node.node_type is not expected:
            raise WorkflowError("Workflow node does not accept this transition")
        record = self._step_records(store, run).get(node_id)
        if record is None or record.state != "waiting":
            raise WorkflowError("Workflow step is not waiting")
        return node, self._step_from_record(record), record

    def _revision(self, store: PlatformStore, revision_id: str) -> WorkflowRevision:
        record = store.head("workflow_revision", revision_id)
        if record is None:
            raise WorkflowError("Workflow revision does not exist")
        if (
            record.kind != "workflow_revision"
            or record.record_id != revision_id
            or record.revision != 1
            or record.state != "validated"
        ):
            raise WorkflowError("Workflow revision is not canonical")
        try:
            revision = WorkflowRevision.model_validate(record.payload)
        except ValidationError as error:
            raise WorkflowError("Workflow revision contract is invalid") from error
        manifest = sha256_hex(
            canonical_json_bytes(revision.model_dump(mode="json", exclude={"manifest_sha256"}))
        )
        if revision.revision_id != revision_id or manifest != revision.manifest_sha256:
            raise WorkflowError("Workflow revision binding is invalid")
        return revision

    def _retry_policy(self, store: PlatformStore, node: WorkflowNode) -> WorkflowRetryPolicy | None:
        if node.retry_policy_id is None:
            return None
        record = store.head("workflow_retry_policy", node.retry_policy_id)
        if record is None:
            raise WorkflowError("Workflow retry policy is missing")
        return WorkflowRetryPolicy.model_validate(record.payload)

    def _approval_gate(self, store: PlatformStore, node: WorkflowNode) -> WorkflowApprovalGate:
        if node.approval_gate_id is None:
            raise WorkflowError("Approval nodes require a typed approval gate")
        record = store.head("workflow_approval_gate", node.approval_gate_id)
        if record is None:
            raise WorkflowError("Workflow approval gate is missing")
        return WorkflowApprovalGate.model_validate(record.payload)

    def _canonical_approval_gate(
        self, store: PlatformStore, node: WorkflowNode
    ) -> WorkflowApprovalGate:
        gate_id = node.approval_gate_id
        if gate_id is None:
            raise WorkflowError("Approval nodes require a typed approval gate")
        record = store.head("workflow_approval_gate", gate_id)
        if (
            record is None
            or record.kind != "workflow_approval_gate"
            or record.record_id != gate_id
            or record.revision != 1
            or record.state != "registered"
        ):
            raise WorkflowError("Workflow approval gate is not canonical")
        try:
            gate = WorkflowApprovalGate.model_validate(record.payload)
        except ValidationError as error:
            raise WorkflowError("Workflow approval gate contract is invalid") from error
        if gate.approval_gate_id != gate_id or gate.action != _APPROVAL_ACTION:
            raise WorkflowError("Workflow approval gate binding is invalid")
        return gate

    def _require_node_references(self, store: PlatformStore, revision: WorkflowRevision) -> None:
        for node in revision.nodes:
            if node.retry_policy_id is not None and (
                store.head("workflow_retry_policy", node.retry_policy_id) is None
            ):
                raise WorkflowError("Workflow retry policies must be registered")
            if node.node_type is WorkflowNodeType.APPROVAL:
                gate_id = node.approval_gate_id
                if gate_id is None or store.head("workflow_approval_gate", gate_id) is None:
                    raise WorkflowError("Workflow approval gates must be registered")

    def _store_immutable(
        self,
        store: PlatformStore,
        *,
        kind: str,
        record_id: str,
        payload: dict[str, Any],
    ) -> None:
        existing = store.head(kind, record_id)
        rendered = sha256_hex(canonical_json_bytes(payload))
        if existing is not None:
            if existing.payload_sha256 != rendered:
                raise WorkflowError("Workflow reference records are immutable")
            return
        store.append_record(
            kind=kind,
            record_id=record_id,
            payload=payload,
            state="registered",
            expected_revision=None,
        )

    def _require_enabled(self) -> None:
        if not self.features.enabled("workflow_engine_enabled"):
            raise WorkflowError("Workflow engine is disabled")

    def _safe_payload(self, payload: dict[str, Any], *, label: str) -> bytes:
        try:
            rendered = canonical_json_bytes(payload)
        except (TypeError, ValueError) as error:
            raise WorkflowError(f"{label} must be canonical JSON") from error
        if len(rendered) > 256 * 1024:
            raise WorkflowError(f"{label} exceed the bounded size")
        if self.scanner.scan(rendered).blocks_processing:
            raise WorkflowError(f"{label} contain restricted data")
        return rendered
