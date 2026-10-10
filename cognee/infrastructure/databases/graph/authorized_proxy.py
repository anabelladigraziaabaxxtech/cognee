import base64
import json
import os
from typing import Any

import httpx

from cognee.shared.logging_utils import get_logger

logger = get_logger("AgentsPPGraphAuth")


class GraphAuthorizationError(RuntimeError):
    """Raised when Agents++ authorization denies a graph read operation."""


class AuthorizedGraphEngineProxy:
    """Decorator proxy that enforces Agents++ authorization on graph reads."""

    _agentspp_authorized_proxy = True

    def __init__(self, graph_engine: Any):
        self._graph_engine = graph_engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph_engine, name)

    async def query(self, query: str, params: dict):
        await self._authorize_graph_action(
            action="query",
            resource="graph.query",
            constraints={
                "queryLength": len(query or ""),
                "hasParams": bool(params),
            },
        )
        return await self._graph_engine.query(query, params)

    async def get_graph_data(self):
        await self._authorize_graph_action(
            action="read_graph",
            resource="graph.get_graph_data",
            constraints={},
        )
        return await self._graph_engine.get_graph_data()

    async def get_id_filtered_graph_data(self, target_ids: list[str] | None = None):
        await self._authorize_graph_action(
            action="read_graph",
            resource="graph.get_id_filtered_graph_data",
            constraints={
                "targetIdsCount": len(target_ids or []),
            },
        )
        if hasattr(self._graph_engine, "get_id_filtered_graph_data"):
            return await self._graph_engine.get_id_filtered_graph_data(target_ids=target_ids)
        return await self._graph_engine.get_graph_data()

    async def get_filtered_graph_data(self, attribute_filters=None):
        await self._authorize_graph_action(
            action="read_graph",
            resource="graph.get_filtered_graph_data",
            constraints={
                "filterCount": len(attribute_filters or []),
            },
        )
        return await self._graph_engine.get_filtered_graph_data(attribute_filters=attribute_filters)

    async def get_nodeset_subgraph(self, node_type, node_name, node_name_filter_operator="OR"):
        await self._authorize_graph_action(
            action="read_graph",
            resource="graph.get_nodeset_subgraph",
            constraints={
                "nodeType": getattr(node_type, "__name__", str(node_type)),
                "nodeName": node_name or [],
                "nodeNameFilterOperator": node_name_filter_operator,
            },
        )
        return await self._graph_engine.get_nodeset_subgraph(
            node_type=node_type,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
        )

    async def get_neighborhood(self, node_ids: list[str], depth: int = 1, edge_types=None):
        await self._authorize_graph_action(
            action="read_graph",
            resource="graph.get_neighborhood",
            constraints={
                "seedNodeCount": len(node_ids or []),
                "depth": depth,
                "edgeTypes": edge_types or [],
            },
        )
        return await self._graph_engine.get_neighborhood(
            node_ids=node_ids,
            depth=depth,
            edge_types=edge_types,
        )

    async def _authorize_graph_action(self, *, action: str, resource: str, constraints: dict) -> None:
        mode = os.getenv("AGENTSPP_GRAPH_ENFORCEMENT", "off").strip().lower()
        if mode == "off":
            return

        fail_open = os.getenv("AGENTSPP_GRAPH_FAIL_OPEN", "true").strip().lower() == "true"
        gateway_url = os.getenv("AGENTSPP_GATEWAY_URL", "http://localhost:8787").rstrip("/")
        agent_did = os.getenv("AGENTSPP_AGENT_DID", "").strip()
        credential = os.getenv("AGENTSPP_CREDENTIAL_JWT", "").strip()

        if not agent_did or not credential:
            message = (
                "Agents++ graph enforcement missing AGENTSPP_AGENT_DID or "
                "AGENTSPP_CREDENTIAL_JWT"
            )
            if mode == "monitor":
                logger.warning("%s (monitor mode; allowing request)", message)
                return
            if fail_open:
                logger.warning("%s (fail-open=true; allowing request)", message)
                return
            raise GraphAuthorizationError(message)

        credential_for_authorize = await _resolve_authorization_credential(
            gateway_url=gateway_url,
            agent_did=agent_did,
            credential=credential,
        )

        effective_constraints = dict(constraints)
        inferred_domain = _infer_domain_from_constraints(effective_constraints)
        if inferred_domain:
            effective_constraints.setdefault("domain", inferred_domain)
            effective_constraints.setdefault("department", inferred_domain)

        payload = {
            "agentDid": agent_did,
            "credential": credential_for_authorize,
            "plane": "graph",
            "action": action,
            "resource": resource,
            "constraints": effective_constraints,
        }

        timeout_s = float(os.getenv("AGENTSPP_AUTH_TIMEOUT_SECONDS", "2.0"))

        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.post(f"{gateway_url}/authorize", json=payload)
            body = response.json()

            allowed = bool(body.get("allowed"))
            reason = body.get("reasonCode", "UNKNOWN")
            detail = body.get("detail", "")

            if allowed:
                return

            message = f"Agents++ denied graph {action} on {resource}: {reason} {detail}".strip()
            if mode == "monitor":
                logger.warning("%s (monitor mode; allowing request)", message)
                return
            raise GraphAuthorizationError(message)

        except Exception as error:
            if mode == "monitor":
                logger.warning(
                    "Agents++ graph auth monitor error (allowing request): %s",
                    error,
                    exc_info=True,
                )
                return
            if fail_open:
                logger.warning(
                    "Agents++ graph auth error with fail-open=true (allowing request): %s",
                    error,
                    exc_info=True,
                )
                return
            raise


async def _resolve_authorization_credential(*, gateway_url: str, agent_did: str, credential: str) -> str:
    if _looks_like_vp_jwt(credential):
        return credential

    payload = {
        "agentDid": agent_did,
        "vcJwt": credential,
        "lifetime": "60s",
    }

    async with httpx.AsyncClient(timeout=3.0) as client:
        response = await client.post(f"{gateway_url}/demo/presentations", json=payload)
    body = response.json()
    presentation = body.get("presentation")
    if isinstance(presentation, str) and presentation.strip():
        return presentation

    raise GraphAuthorizationError("Failed to mint VP for authorization from vcJwt")


def _looks_like_vp_jwt(jwt_token: str) -> bool:
    try:
        parts = jwt_token.split(".")
        if len(parts) < 2:
            return False
        payload_b64 = parts[1]
        padding = "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64 + padding).decode("utf-8"))
        vp = payload.get("vp")
        if not isinstance(vp, dict):
            return False
        types = vp.get("type")
        return isinstance(types, list) and "VerifiablePresentation" in types
    except Exception:
        return False


def maybe_wrap_graph_engine(graph_engine: Any) -> Any:
    mode = os.getenv("AGENTSPP_GRAPH_ENFORCEMENT", "off").strip().lower()
    if mode == "off":
        return graph_engine
    if getattr(graph_engine, "_agentspp_authorized_proxy", False):
        return graph_engine
    return AuthorizedGraphEngineProxy(graph_engine)


def _infer_domain_from_constraints(constraints: dict) -> str | None:
    node_name = constraints.get("nodeName")
    if isinstance(node_name, list):
        lowered = [str(value).lower() for value in node_name]
        if any("finance" in value for value in lowered):
            return "finance"
        if any(value in {"hr", "human_resources", "human-resources"} or "hr" in value for value in lowered):
            return "hr"

    explicit = os.getenv("AGENTSPP_GRAPH_DOMAIN", "").strip().lower()
    if explicit in {"finance", "hr"}:
        return explicit

    generic = os.getenv("AGENTSPP_DEFAULT_DOMAIN", "").strip().lower()
    if generic in {"finance", "hr"}:
        return generic

    return None
