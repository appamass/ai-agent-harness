"""Incident responder: on-call agent for alert diagnosis and approved remediation.

- Stage 1: Autonomy.READ_ONLY for diagnosis (query_logs, get_metrics, recent_deploys).
- Stage 2: Autonomy.ACT_WITH_APPROVAL for remediation (rollback_deploy, Risk.IRREVERSIBLE).
- Credentials passed strictly via deps (DeployClient), never serialized to checkpoints.
- Emergency release freeze via KillSwitch.
- Checkpoints via SqliteCheckpointStore (cross-platform).
- Deterministic --demo mode without an API key. Use `build_incident_agent` with your own provider for real alerts.

Usage:
    python examples/incident_responder.py --demo   # offline demo (Stage 1 + Stage 2)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

from ai_agent_harness import (
    Agent,
    Autonomy,
    BaseTool,
    CheckpointStore,
    KillSwitch,
    Limits,
    ModelProvider,
    Risk,
    RunState,
    RunStatus,
    SqliteCheckpointStore,
    resume,
    start,
)
from ai_agent_harness.testing import FakeProvider, text_response, tool_response

SYSTEM_PROMPT = (
    "You are the On-Call Incident Responder. Investigate alerts by querying logs, metrics, and recent deploys.\n"
    "Diagnose the root cause concisely. In Stage 2, propose a rollback only if a recent deploy caused the incident.\n"
    "Never propose actions on healthy services."
)


# ==============================================================================
# Runtime Dependencies (deps) – Not Serialized to Checkpoints
# ==============================================================================


@dataclass
class DeployClient:
    """Deployment platform API client. Passed per execution via deps, never stored in checkpoints."""

    token: str
    rollbacks: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])

    def rollback(self, service: str, to_deploy: str) -> str:
        self.rollbacks.append((service, to_deploy))
        return f"Successfully rolled back {service} to {to_deploy}."


@dataclass
class IncidentDeps:
    deploys: DeployClient


# ==============================================================================
# Diagnostic and Remediation Tools
# ==============================================================================


ServiceName = Annotated[str, Field(pattern=r"^[a-z0-9-]{1,30}$", description="Target service name")]


class _Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LogQueryInput(_Input):
    service: ServiceName
    query: str = Field(min_length=2, max_length=100, description="Log search query")


class QueryLogs(BaseTool[LogQueryInput]):
    name: ClassVar[str] = "query_logs"
    description: ClassVar[str] = "Queries recent service logs for errors, stack traces, and warnings."
    input_model: ClassVar[type[BaseModel]] = LogQueryInput
    risk: ClassVar[Risk] = Risk.READ

    async def run(self, args: LogQueryInput, state: RunState) -> str:
        if "500" in args.query or "error" in args.query:
            return (
                f"Logs for '{args.service}':\n"
                "[ERROR] 03:02:11 DB pool connection timeout (timeout=5000ms)\n"
                "[FATAL] 03:02:12 Unhandled exception in /checkout/pay: ConnectionTimeout"
            )
        return f"Logs for '{args.service}': 0 errors found matching query."


class MetricsInput(_Input):
    service: ServiceName
    metric: str = Field(min_length=2, max_length=50, description="Metric name (e.g. latency_p99, error_rate)")


class GetMetrics(BaseTool[MetricsInput]):
    name: ClassVar[str] = "get_metrics"
    description: ClassVar[str] = "Fetches metrics for error rates, latency percentiles, and resource utilization."
    input_model: ClassVar[type[BaseModel]] = MetricsInput
    risk: ClassVar[Risk] = Risk.READ

    async def run(self, args: MetricsInput, state: RunState) -> str:
        return f"Metric '{args.metric}' on '{args.service}': current=520ms, baseline=45ms (11x spike)"


class ServiceInput(_Input):
    service: ServiceName


class RecentDeploys(BaseTool[ServiceInput]):
    name: ClassVar[str] = "recent_deploys"
    description: ClassVar[str] = "Lists recent deployments for a service with commit hash, timestamp, and status."
    input_model: ClassVar[type[BaseModel]] = ServiceInput
    risk: ClassVar[Risk] = Risk.READ

    async def run(self, args: ServiceInput, state: RunState) -> str:
        return (
            f"Recent deployments for '{args.service}':\n"
            "- d-1042 (deployed 14 minutes ago by alice@, commit 8f3a1c, active)\n"
            "- d-1041 (deployed 2 days ago by bob@, commit e4d2b9, stable)"
        )


class RollbackInput(_Input):
    service: ServiceName
    to_deploy: str = Field(pattern=r"^d-\d{1,8}$", description="Target deployment ID to roll back to (e.g. d-1041)")
    reason: str = Field(min_length=10, max_length=500, description="Explanation for the on-call engineer")


class RollbackDeploy(BaseTool[RollbackInput]):
    name: ClassVar[str] = "rollback_deploy"
    description: ClassVar[str] = "Rolls back a production service to an earlier stable deployment."
    input_model: ClassVar[type[BaseModel]] = RollbackInput
    risk: ClassVar[Risk] = Risk.IRREVERSIBLE

    async def run(self, args: RollbackInput, state: RunState) -> str:
        return state.deps_as(IncidentDeps).deploys.rollback(args.service, args.to_deploy)


# ==============================================================================
# Agent Assembly
# ==============================================================================


def build_incident_agent(
    model: ModelProvider,
    *,
    stage: int = 1,
    checkpoints: CheckpointStore | None = None,
    freeze_file: Path | None = None,
) -> Agent:
    """Builds the Incident Responder agent with Stage 1 (read-only) or Stage 2 (approval) autonomy."""
    tools: list[BaseTool[Any]] = [QueryLogs(), GetMetrics(), RecentDeploys()]
    autonomy = Autonomy.READ_ONLY

    if stage >= 2:
        tools.append(RollbackDeploy())
        autonomy = Autonomy.ACT_WITH_APPROVAL

    return Agent.create(
        model,
        SYSTEM_PROMPT,
        name="incident_responder",
        tools=tools,
        autonomy=autonomy,
        hooks=[KillSwitch(lambda _state: freeze_file.exists())] if freeze_file is not None else [],
        checkpoints=checkpoints,
        limits=Limits(max_steps=10, max_seconds=180),
    )


# ==============================================================================
# Demo and CLI Execution
# ==============================================================================


async def run_demo() -> int:
    """Runs a complete end-to-end demo of both Stage 1 (read-only) and Stage 2 (approved rollback)."""
    print("=" * 70)
    print("INCIDENT RESPONDER DEMO: On-Call Investigation & Remediation")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        store_path = Path(tmp_dir) / "checkpoints.db"
        store = SqliteCheckpointStore(store_path)
        await store.setup()
        deploy_token = os.environ.get("DEPLOY_API_TOKEN", "demo-token")
        deploy_client = DeployClient(token=deploy_token)
        deps = IncidentDeps(deploys=deploy_client)

        # --------------------------------------------------
        # Stage 1: Read-Only Alert Diagnosis
        # --------------------------------------------------
        print("\n--- STAGE 1: Automated Read-Only Investigation (Autonomy.READ_ONLY) ---")
        stage1_provider = FakeProvider(
            [
                tool_response(("query_logs", {"service": "checkout", "query": "500 error"})),
                tool_response(("get_metrics", {"service": "checkout", "metric": "latency_p99"})),
                tool_response(("recent_deploys", {"service": "checkout"})),
                text_response(
                    "DIAGNOSIS: checkout service latency spiked to 520ms immediately after deployment d-1042. "
                    "DB connection pool timeout in /checkout/pay. "
                    "RECOMMENDATION: Roll back checkout to deployment d-1041."
                ),
            ]
        )
        agent_stage1 = build_incident_agent(stage1_provider, stage=1, checkpoints=store)
        state1 = await start(
            agent_stage1,
            "ALERT: checkout service error rate > 5%",
            tenant_id="ops",
            run_id="inc-9001",
            deps=deps,
        )

        print(f"Run ID:  {state1.run_id}")
        print(f"Status:  {state1.status}")
        print(f"Report:\n{state1.last_text}")
        assert state1.status is RunStatus.DONE
        state1.check_protocol()

        # --------------------------------------------------
        # Stage 2: Remediation with Mandatory Human Approval
        # --------------------------------------------------
        print("\n--- STAGE 2: Remediation Proposal & Approval Flow (Autonomy.ACT_WITH_APPROVAL) ---")
        stage2_provider = FakeProvider(
            [
                tool_response(
                    (
                        "rollback_deploy",
                        {
                            "service": "checkout",
                            "to_deploy": "d-1041",
                            "reason": "Reverting d-1042 due to DB connection pool exhaustion",
                        },
                    )
                ),
                text_response("Remediation verified: checkout rolled back to d-1041. Traffic stable."),
            ]
        )
        agent_stage2 = build_incident_agent(stage2_provider, stage=2, checkpoints=store)

        state2 = await start(
            agent_stage2,
            "Remediate checkout alert based on d-1042 investigation findings",
            tenant_id="ops",
            run_id="inc-9002",
            deps=deps,
        )

        print(f"Run ID:     {state2.run_id}")
        print(f"Status:     {state2.status}")
        print(f"Prompt:     {state2.status_reason}")
        print(f"Pending ID: {state2.pending_id}")
        assert state2.status is RunStatus.WAITING_FOR_APPROVAL
        assert state2.pending_id is not None
        state2.check_protocol()

        # Simulate on-call engineer approving the rollback
        print("\n[On-Call Engineer Review]: Approving rollback...")
        resumed_state = await resume(
            agent_stage2,
            state2,
            approved=True,
            pending_id=state2.pending_id,
            deps=deps,
        )

        print(f"Final Status: {resumed_state.status}")
        print(f"Final Output: {resumed_state.last_text}")
        assert resumed_state.status is RunStatus.DONE
        assert len(deploy_client.rollbacks) == 1
        assert deploy_client.rollbacks[0] == ("checkout", "d-1041")
        resumed_state.check_protocol()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE: All harness guarantees, security boundaries, and approval binding verified.")
    print("=" * 70)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Incident Responder pilot agent")
    parser.add_argument("--demo", action="store_true", help="Run the offline demonstration")
    if not parser.parse_args(argv).demo:
        parser.print_help()
        return 2
    return asyncio.run(run_demo())


if __name__ == "__main__":
    sys.exit(main())
