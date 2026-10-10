import base64
import json
import os
from typing import Any

import httpx

from cognee.shared.logging_utils import get_logger

logger = get_logger("AgentsPPVectorAuth")


class VectorAuthorizationError(RuntimeError):
    """Raised when Agents++ authorization denies a vector operation."""


class AuthorizedVectorEngineProxy:
    """Decorator proxy that enforces Agents++ authorization on vector reads.

    Enabled only when AGENTSPP_VECTOR_ENFORCEMENT is not "off".
    Modes:
      - off: bypass gateway
      - monitor: call gateway, log denies/errors, continue
      - enforce: deny on policy rejection (and on gateway errors when fail-open disabled)
    """

    _agentspp_authorized_proxy = True

    def __init__(self, vector_engine: Any):
        self._vector_engine = vector_engine

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vector_engine, name)

    async def search(
        self,
        collection_name: str,
        query_text: str | None = None,
        query_vector: list[float] | None = None,
        limit: int | None = None,
        with_vector: bool = False,
        include_payload: bool = False,
        node_name: list[str] | None = None,
        node_name_filter_operator: str = "OR",
    ):
        await self._authorize_vector_action(
            action="search",
            collection_name=collection_name,
            limit=limit,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
            batch_size=None,
        )
        return await self._vector_engine.search(
            collection_name,
            query_text,
            query_vector,
            limit,
            with_vector=with_vector,
            include_payload=include_payload,
            node_name=node_name,
            node_name_filter_operator=node_name_filter_operator,
        )

    async def batch_search(
        self,
        collection_name: str,
        query_texts: list[str],
        limit: int | None,
        with_vectors: bool = False,
        include_payload: bool = False,
        node_name: list[str] | None = None,
    ):
        await self._authorize_vector_action(
            action="batch_search",
            collection_name=collection_name,
            limit=limit,
            node_name=node_name,
            node_name_filter_operator="OR",
            batch_size=len(query_texts),
        )
        return await self._vector_engine.batch_search(
            collection_name,
            query_texts,
            limit,
            with_vectors=with_vectors,
            include_payload=include_payload,
            node_name=node_name,
        )

    async def _authorize_vector_action(
        self,
        *,
        action: str,
        collection_name: str,
        limit: int | None,
        node_name: list[str] | None,
        node_name_filter_operator: str,
        batch_size: int | None,
    ) -> None:
        mode = os.getenv("AGENTSPP_VECTOR_ENFORCEMENT", "off").strip().lower()
        if mode == "off":
            return

        fail_open = os.getenv("AGENTSPP_VECTOR_FAIL_OPEN", "true").strip().lower() == "true"
        gateway_url = os.getenv("AGENTSPP_GATEWAY_URL", "http://localhost:8787").rstrip("/")
        agent_did = os.getenv("AGENTSPP_AGENT_DID", "").strip()
        credential = os.getenv("AGENTSPP_CREDENTIAL_JWT", "").strip()

        if not agent_did or not credential:
            msg = (
                "Agents++ vector enforcement missing AGENTSPP_AGENT_DID or "
                "AGENTSPP_CREDENTIAL_JWT"
            )
            if mode == "monitor":
                logger.warning("%s (monitor mode; allowing request)", msg)
                return
            if fail_open:
                logger.warning("%s (fail-open=true; allowing request)", msg)
                return
            raise VectorAuthorizationError(msg)

        credential_for_authorize = await _resolve_authorization_credential(
            gateway_url=gateway_url,
            agent_did=agent_did,
            credential=credential,
        )

        inferred_domain = _infer_domain(node_name)
        constraints: dict[str, Any] = {
            "collection": collection_name,
            "nodeName": node_name or [],
            "nodeNameFilterOperator": node_name_filter_operator,
        }
        if limit is not None:
            constraints["topK"] = limit
        if batch_size is not None:
            constraints["batchSize"] = batch_size
        if inferred_domain:
            constraints["domain"] = inferred_domain
            constraints["department"] = inferred_domain

        payload = {
            "agentDid": agent_did,
            "credential": credential_for_authorize,
            "plane": "vector",
            "action": action,
            "resource": collection_name,
            "constraints": constraints,
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

            message = f"Agents++ denied vector {action} on {collection_name}: {reason} {detail}".strip()
            if mode == "monitor":
                logger.warning("%s (monitor mode; allowing request)", message)
                return
            raise VectorAuthorizationError(message)

        except Exception as error:
            if mode == "monitor":
                logger.warning(
                    "Agents++ vector auth monitor error (allowing request): %s",
                    error,
                    exc_info=True,
                )
                return
            if fail_open:
                logger.warning(
                    "Agents++ vector auth error with fail-open=true (allowing request): %s",
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

    raise VectorAuthorizationError("Failed to mint VP for authorization from vcJwt")


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


def maybe_wrap_vector_engine(vector_engine: Any) -> Any:
    mode = os.getenv("AGENTSPP_VECTOR_ENFORCEMENT", "off").strip().lower()
    if mode == "off":
        return vector_engine
    if getattr(vector_engine, "_agentspp_authorized_proxy", False):
        return vector_engine
    return AuthorizedVectorEngineProxy(vector_engine)


def _infer_domain(node_names: list[str] | None) -> str | None:
    if node_names:
        lowered = [value.lower() for value in node_names]
        if any("finance" in value for value in lowered):
            return "finance"
        if any(value in {"hr", "human_resources", "human-resources"} or "hr" in value for value in lowered):
            return "hr"

    explicit = os.getenv("AGENTSPP_VECTOR_DOMAIN", "").strip().lower()
    if explicit in {"finance", "hr"}:
        return explicit

    generic = os.getenv("AGENTSPP_DEFAULT_DOMAIN", "").strip().lower()
    if generic in {"finance", "hr"}:
        return generic

    return None
