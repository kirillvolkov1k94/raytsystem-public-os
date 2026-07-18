from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from raytsystem.contracts import (
    ApprovalRecord,
    PendingWorkflowApproval,
    PolicyDecision,
    PolicyOutcome,
    WorkflowApprovalGate,
    WorkflowRevision,
    WorkflowRun,
    WorkflowStepRun,
    canonical_json_bytes,
    derive_id,
    sha256_hex,
)
from raytsystem.contracts.execution import ExecutionApproval
from raytsystem.contracts.governance import EmergencyAction
from raytsystem.contracts.workflows import WorkflowNodeType
from raytsystem.execution.store import ExecutionStore, ExecutionStoreError
from raytsystem.platform_store import (
    PlatformStore,
    PlatformStoreError,
    initialize_platform_store,
    open_platform_store_read_only,
)
from raytsystem.security.paths import PathPolicyError, read_regular_file

_WORKFLOW_APPROVAL_ACTION = "workflow_approval"
_ISSUANCE_TARGET_SCOPE = "workflow_approval_issuance_target"
_ISSUANCE_KEY_SCOPE = "workflow_approval_issuance_key"


class AuthorityError(RuntimeError):
    """A policy decision or approval is missing, forged, stale, or out of scope."""


def workflow_approval_target(workflow_run_id: str, node_id: str) -> str:
    """Return the exact authority target for one workflow approval node."""

    return derive_id("wfappr", {"node_id": node_id, "workflow_run_id": workflow_run_id})


@dataclass(frozen=True)
class _PendingWorkflowContext:
    pending: PendingWorkflowApproval
    waiting_since: datetime
    gate_schema_version: str


class ApprovalAuthorityService:
    """Inspect and issue approvals from trusted workflow state without transitioning it."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def inspect_pending(
        self,
        workflow_run_id: str,
        node_id: str,
        *,
        at: datetime | None = None,
    ) -> PendingWorkflowApproval:
        requested_at = None if at is None else self._utc_at(at)
        try:
            with initialize_platform_store(self.root) as store, store.transaction():
                now = requested_at or datetime.now(UTC)
                return self._pending_context(store, workflow_run_id, node_id, now).pending
        except AuthorityError:
            raise
        except PlatformStoreError as error:
            raise AuthorityError("Workflow approval state is unavailable") from error
        except (OSError, sqlite3.Error, ValidationError, TypeError, ValueError) as error:
            raise AuthorityError("Workflow approval state is invalid") from error

    def issue_approval(
        self,
        workflow_run_id: str,
        node_id: str,
        *,
        approver: str,
        idempotency_key: str,
        at: datetime | None = None,
    ) -> ApprovalRecord:
        requested_at = None if at is None else self._utc_at(at)
        try:
            with initialize_platform_store(self.root) as store, store.transaction():
                now = requested_at or datetime.now(UTC)
                context = self._pending_context(store, workflow_run_id, node_id, now)
                request = {
                    "pending": context.pending.model_dump(mode="json"),
                    "gate_schema_version": context.gate_schema_version,
                    "approver": approver,
                    "idempotency_key": idempotency_key,
                }
                target_receipt = store.idempotent_receipt(
                    scope=_ISSUANCE_TARGET_SCOPE,
                    idempotency_key=context.pending.target_id,
                    request=request,
                )
                key_receipt = store.idempotent_receipt(
                    scope=_ISSUANCE_KEY_SCOPE,
                    idempotency_key=idempotency_key,
                    request=request,
                )
                if target_receipt is not None or key_receipt is not None:
                    if target_receipt is None or target_receipt != key_receipt:
                        raise AuthorityError(
                            "Workflow approval idempotency binding is incomplete"
                        )
                    return self._stored_approval(store, target_receipt, context, approver, now)

                approval = ApprovalRecord.create(
                    action=context.pending.action,
                    target_id=context.pending.target_id,
                    artifact_sha256=context.pending.input_sha256,
                    scope=(context.pending.required_role,),
                    policy_version=context.gate_schema_version,
                    policy_sha256=context.pending.scope_sha256,
                    approver=approver,
                    approved_at=now,
                    expires_at=context.pending.expires_at,
                )
                store.append_record(
                    kind="authority_approval",
                    record_id=approval.approval_id,
                    payload=approval.model_dump(mode="json"),
                    state="accepted",
                    expected_revision=None,
                )
                receipt = {"approval_id": approval.approval_id}
                store.idempotent_receipt(
                    scope=_ISSUANCE_TARGET_SCOPE,
                    idempotency_key=context.pending.target_id,
                    request=request,
                    receipt=receipt,
                )
                store.idempotent_receipt(
                    scope=_ISSUANCE_KEY_SCOPE,
                    idempotency_key=idempotency_key,
                    request=request,
                    receipt=receipt,
                )
                return approval
        except AuthorityError:
            raise
        except PlatformStoreError as error:
            raise AuthorityError(
                "Workflow approval idempotency binding is invalid"
            ) from error
        except (OSError, sqlite3.Error, ValidationError, TypeError, ValueError) as error:
            raise AuthorityError("Workflow approval issuance is invalid") from error

    def _pending_context(
        self,
        store: PlatformStore,
        workflow_run_id: str,
        node_id: str,
        now: datetime,
    ) -> _PendingWorkflowContext:
        run_record = store.head("workflow_run", workflow_run_id)
        if run_record is None:
            raise AuthorityError("Workflow run does not exist")
        run_payload = dict(run_record.payload)
        raw_inputs = run_payload.pop("inputs", None)
        if not isinstance(raw_inputs, dict):
            raise AuthorityError("Workflow run input binding is invalid")
        run = WorkflowRun.model_validate(run_payload)
        if (
            run.workflow_run_id != workflow_run_id
            or run.state != "running"
            or run_record.state != run.state
            or sha256_hex(canonical_json_bytes(raw_inputs)) != run.input_sha256
        ):
            raise AuthorityError("Workflow run input binding is invalid")

        revision_record = store.head("workflow_revision", run.revision_id)
        if (
            revision_record is None
            or revision_record.revision != 1
            or revision_record.state != "validated"
        ):
            raise AuthorityError("Workflow revision is not immutable and validated")
        revision = WorkflowRevision.model_validate(revision_record.payload)
        manifest = sha256_hex(
            canonical_json_bytes(
                revision.model_dump(mode="json", exclude={"manifest_sha256"})
            )
        )
        if manifest != revision.manifest_sha256:
            raise AuthorityError("Workflow revision manifest is invalid")
        node = next((item for item in revision.nodes if item.node_id == node_id), None)
        if node is None or node.node_type is not WorkflowNodeType.APPROVAL:
            raise AuthorityError("Workflow node is not an approval node")

        matching_steps = []
        for step_run_id in run.step_run_ids:
            step_record = store.head("workflow_step", step_run_id)
            if step_record is None:
                raise AuthorityError("Workflow step record is missing")
            if step_record.payload.get("node_id") == node_id:
                matching_steps.append(step_record)
        if len(matching_steps) != 1:
            raise AuthorityError("Workflow node has no unique waiting step")
        step_record = matching_steps[0]
        if step_record.state != "waiting":
            raise AuthorityError("Workflow step is not waiting on approval")
        step = WorkflowStepRun.model_validate(step_record.payload)
        if (
            step_record.record_id != step.step_run_id
            or step.workflow_run_id != workflow_run_id
            or step.node_id != node_id
            or step.state != "waiting"
            or step.started_at is None
            or step.completed_at is not None
            or step.approval_id is not None
            or step.input_sha256 != run.input_sha256
        ):
            raise AuthorityError("Workflow step is not waiting on the trusted input")

        gate_id = node.approval_gate_id
        gate_record = (
            None if gate_id is None else store.head("workflow_approval_gate", gate_id)
        )
        if (
            gate_record is None
            or gate_record.revision != 1
            or gate_record.state != "registered"
        ):
            raise AuthorityError("Workflow approval gate is not immutable and registered")
        gate = WorkflowApprovalGate.model_validate(gate_record.payload)
        if gate.approval_gate_id != gate_id or gate.action != _WORKFLOW_APPROVAL_ACTION:
            raise AuthorityError("Workflow approval gate binding is invalid")

        waiting_since = step.started_at.astimezone(UTC)
        expires_at = waiting_since + timedelta(seconds=gate.expires_after_seconds)
        if now < waiting_since:
            raise AuthorityError("Workflow approval gate is not active yet")
        if now >= expires_at:
            raise AuthorityError("Workflow approval gate has expired")
        pending = PendingWorkflowApproval(
            workflow_run_id=workflow_run_id,
            step_run_id=step.step_run_id,
            node_id=node_id,
            approval_gate_id=gate.approval_gate_id,
            action=gate.action,
            target_id=workflow_approval_target(workflow_run_id, node_id),
            input_sha256=run.input_sha256,
            scope_sha256=gate.scope_sha256,
            required_role=gate.required_role,
            expires_at=expires_at,
        )
        return _PendingWorkflowContext(
            pending=pending,
            waiting_since=waiting_since,
            gate_schema_version=gate.schema_version,
        )

    def _stored_approval(
        self,
        store: PlatformStore,
        receipt: dict[str, object],
        context: _PendingWorkflowContext,
        approver: str,
        now: datetime,
    ) -> ApprovalRecord:
        approval_id = receipt.get("approval_id")
        if not isinstance(approval_id, str):
            raise AuthorityError("Workflow approval idempotency receipt is invalid")
        record = store.head("authority_approval", approval_id)
        if record is None or record.revision != 1 or record.state != "accepted":
            raise AuthorityError("Workflow approval idempotency receipt is orphaned")
        approval = ApprovalRecord.model_validate(record.payload)
        expected = ApprovalRecord.create(
            action=context.pending.action,
            target_id=context.pending.target_id,
            artifact_sha256=context.pending.input_sha256,
            scope=(context.pending.required_role,),
            policy_version=context.gate_schema_version,
            policy_sha256=context.pending.scope_sha256,
            approver=approver,
            approved_at=approval.approved_at,
            expires_at=context.pending.expires_at,
        )
        if (
            approval != expected
            or approval.approval_id != approval_id
            or approval.approved_at < context.waiting_since
            or approval.approved_at > now
        ):
            raise AuthorityError("Stored workflow approval binding is invalid")
        return approval

    @staticmethod
    def _utc_at(at: datetime) -> datetime:
        if at.tzinfo is None or at.utcoffset() is None:
            raise AuthorityError("Workflow approval time must be timezone-aware")
        return at.astimezone(UTC)


class AuthorityResolver:
    """Resolve hash-bound authority records from trusted local stores; fail closed."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def require_approval(
        self,
        approval_id: str,
        *,
        action: str,
        target_id: str,
        artifact_sha256: str,
        destination: str | None = None,
        required_scope: frozenset[str] = frozenset(),
        policy_sha256: str | None = None,
        at: datetime | None = None,
    ) -> ApprovalRecord | ExecutionApproval:
        approval = self._approval(approval_id)
        now = (at or datetime.now(UTC)).astimezone(UTC)
        if isinstance(approval, ApprovalRecord):
            expected = ApprovalRecord.create(
                action=approval.action,
                target_id=approval.target_id,
                artifact_sha256=approval.artifact_sha256,
                destination=approval.destination,
                scope=approval.scope,
                policy_version=approval.policy_version,
                policy_sha256=approval.policy_sha256,
                approver=approval.approver,
                approved_at=approval.approved_at,
                expires_at=approval.expires_at,
                conditions=approval.conditions,
            )
            valid = (
                expected.approval_id == approval.approval_id
                and approval.is_valid_for(
                    action=action,
                    target_id=target_id,
                    artifact_sha256=artifact_sha256,
                    destination=destination,
                    at=now,
                )
                and required_scope.issubset(approval.scope)
                and (policy_sha256 is None or approval.policy_sha256 == policy_sha256)
            )
        else:
            target_bindings = {
                value
                for value in (
                    approval.employee_id,
                    approval.task_id,
                    approval.run_id,
                    approval.workspace_id,
                )
                if value is not None
            }
            valid = bool(
                approval.verify_id()
                and approval.action == action
                and approval.payload_sha256 == artifact_sha256
                and approval.destination == destination
                and target_id in target_bindings
                and required_scope.issubset(approval.scope)
                and approval.approved_at <= now < approval.expires_at
            )
        if not valid:
            raise AuthorityError("Approval does not match the exact action scope")
        revoked_after = self._pending_approvals_revoked_at()
        if revoked_after is not None and approval.approved_at < revoked_after:
            raise AuthorityError("Approval was revoked by an active emergency control")
        return approval

    def require_workflow_approval(
        self,
        store: PlatformStore,
        approval_id: str,
        *,
        action: str,
        target_id: str,
        artifact_sha256: str,
        required_role: str,
        policy_version: str,
        policy_sha256: str,
        waiting_since: datetime,
        gate_expires_at: datetime,
        at: datetime,
    ) -> ApprovalRecord:
        """Resolve one exact workflow approval from the caller's locked store snapshot."""

        if store.root != self.root or not store.connection.in_transaction:
            raise AuthorityError("Workflow approval requires a locked platform transaction")
        if action != _WORKFLOW_APPROVAL_ACTION:
            raise AuthorityError("Workflow approval action is unsupported")
        try:
            record = store.head("authority_approval", approval_id)
        except (PlatformStoreError, TypeError, ValueError) as error:
            raise AuthorityError("Workflow approval authority record is invalid") from error
        if (
            record is None
            or record.kind != "authority_approval"
            or record.record_id != approval_id
            or record.revision != 1
            or record.state != "accepted"
        ):
            raise AuthorityError("Workflow approval authority record is not canonical")
        try:
            approval = ApprovalRecord.model_validate(record.payload)
            expected = ApprovalRecord.create(
                action=action,
                target_id=target_id,
                artifact_sha256=artifact_sha256,
                scope=(required_role,),
                policy_version=policy_version,
                policy_sha256=policy_sha256,
                approver=approval.approver,
                approved_at=approval.approved_at,
                expires_at=gate_expires_at,
            )
        except (ValidationError, TypeError, ValueError) as error:
            raise AuthorityError("Stored workflow approval contract is invalid") from error
        if (
            approval != expected
            or approval.approval_id != approval_id
            or approval.approved_at < waiting_since
            or approval.approved_at > at
            or at >= gate_expires_at
        ):
            raise AuthorityError("Workflow approval does not match the exact gate authority")
        revoked_after = self._pending_approvals_revoked_at(store)
        if revoked_after is not None and approval.approved_at < revoked_after:
            raise AuthorityError("Approval was revoked by an active emergency control")
        return approval

    def require_policy_decision(
        self,
        policy_decision_id: str,
        *,
        action: str,
        target_id: str,
        payload_sha256: str,
        destination: str | None = None,
        policy_sha256: str | None = None,
        allow_requires_approval: bool = True,
    ) -> PolicyDecision:
        decision = self._policy_decision(policy_decision_id)
        material = {
            "action": decision.action,
            "target_id": decision.target_id,
            "payload_sha256": decision.payload_sha256,
            "destination": decision.destination,
            "policy_version": decision.policy_version,
            "policy_sha256": decision.policy_sha256,
            "outcome": decision.outcome,
            "reason_codes": decision.reason_codes,
            "required_approval_scope": decision.required_approval_scope,
            "evaluated_at": decision.evaluated_at,
        }
        allowed_outcomes = {PolicyOutcome.ALLOW}
        if allow_requires_approval:
            allowed_outcomes.add(PolicyOutcome.REQUIRE_APPROVAL)
        if (
            derive_id("pdec", material) != decision.policy_decision_id
            or decision.action != action
            or decision.target_id != target_id
            or decision.payload_sha256 != payload_sha256
            or decision.destination != destination
            or decision.outcome not in allowed_outcomes
            or (policy_sha256 is not None and decision.policy_sha256 != policy_sha256)
        ):
            raise AuthorityError("Policy decision does not match the exact action scope")
        return decision

    def _approval(self, approval_id: str) -> ApprovalRecord | ExecutionApproval:
        payload = self._platform_payload("authority_approval", approval_id)
        if payload is not None:
            try:
                return ApprovalRecord.model_validate(payload)
            except ValidationError as error:
                raise AuthorityError("Stored approval contract is invalid") from error
        execution = self._execution_record(ExecutionApproval, approval_id)
        if isinstance(execution, ExecutionApproval):
            return execution
        accepted = self._accepted_approval(approval_id)
        if accepted is not None:
            return accepted
        raise AuthorityError("Approval authority record is unavailable")

    def _policy_decision(self, decision_id: str) -> PolicyDecision:
        payload = self._platform_payload("authority_policy", decision_id)
        if payload is not None:
            try:
                return PolicyDecision.model_validate(payload)
            except ValidationError as error:
                raise AuthorityError("Stored policy decision is invalid") from error
        execution = self._execution_record(PolicyDecision, decision_id)
        if not isinstance(execution, PolicyDecision):
            raise AuthorityError("Policy decision authority record is unavailable")
        return execution

    def _pending_approvals_revoked_at(self, store: PlatformStore | None = None) -> datetime | None:
        if store is None:
            read_store = open_platform_store_read_only(self.root)
            if read_store is None:
                return None
            with read_store:
                return self._revocation_timestamp(read_store)
        return self._revocation_timestamp(store)

    @staticmethod
    def _revocation_timestamp(store: PlatformStore) -> datetime | None:
        try:
            record = store.head("emergency", "emergency_global")
        except (PlatformStoreError, sqlite3.Error) as error:
            raise AuthorityError("Emergency state is unreadable; approvals fail closed") from error
        if record is None or record.state != "active":
            return None
        active = {str(value) for value in record.payload.get("active_actions", [])}
        if EmergencyAction.REVOKE_PENDING_APPROVALS.value not in active:
            return None
        stamp: object = record.payload.get("activated_at")
        extensions = record.payload.get("extensions")
        if isinstance(extensions, dict):
            per_action = extensions.get("action_activated_at")
            if isinstance(per_action, dict):
                stamp = per_action.get(EmergencyAction.REVOKE_PENDING_APPROVALS.value, stamp)
        try:
            revoked_at = datetime.fromisoformat(str(stamp))
        except (TypeError, ValueError) as error:
            raise AuthorityError("Emergency revocation timestamp is invalid") from error
        if revoked_at.tzinfo is None:
            revoked_at = revoked_at.replace(tzinfo=UTC)
        return revoked_at.astimezone(UTC)

    def _platform_payload(self, kind: str, record_id: str) -> dict[str, object] | None:
        store = open_platform_store_read_only(self.root)
        if store is None:
            return None
        with store:
            record = store.head(kind, record_id)
            return None if record is None else dict(record.payload)

    def _execution_record(
        self,
        model: type[PolicyDecision] | type[ExecutionApproval],
        record_id: str,
    ) -> PolicyDecision | ExecutionApproval | None:
        try:
            store = ExecutionStore.open_for_read(self.root / "ops" / "control.sqlite")
        except (OSError, ExecutionStoreError):
            return None
        if store is None:
            return None
        with store:
            return store.get(model, record_id)

    def _accepted_approval(self, approval_id: str) -> ApprovalRecord | None:
        relative = f"ops/approvals/accepted/{approval_id}.json"
        verification_relative = f"ops/approvals/accepted/{approval_id}.verification.json"
        try:
            payload = json.loads(read_regular_file(self.root, relative, max_bytes=256_000).data)
            verification = json.loads(
                read_regular_file(self.root, verification_relative, max_bytes=256_000).data
            )
            approval = ApprovalRecord.model_validate(payload)
        except (OSError, PathPolicyError, json.JSONDecodeError, ValidationError):
            return None
        if not isinstance(verification, dict) or verification.get("approval_id") != approval_id:
            return None
        return approval
