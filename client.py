"""
CustomerSupportEnv client.

Provides both HTTP and WebSocket interfaces to the environment server,
following the OpenEnv EnvClient pattern.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

import httpx

try:
    import websockets
    _HAS_WEBSOCKETS = True
except ImportError:
    _HAS_WEBSOCKETS = False

from models import (
    EnvironmentState,
    ResetRequest,
    StepRequest,
    TriageAction,
    TriageObservation,
)


class CustomerSupportEnvClient:
    """
    HTTP client for the Customer Support Triage RL environment.

    Wraps the REST API for use in training loops and evaluation scripts.

    Usage::

        client = CustomerSupportEnvClient(base_url="http://localhost:8000")
        obs = client.reset("easy", seed=42)
        while not obs.done:
            action = my_agent.act(obs)
            obs = client.step(action)
        print("Episode reward:", obs.episode_stats.total_reward)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("ENV_BASE_URL", "http://localhost:8000")
        ).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def reset(
        self,
        difficulty: str = "easy",
        seed: Optional[int] = None,
    ) -> TriageObservation:
        """Start a new episode. Returns the first observation."""
        payload = ResetRequest(difficulty=difficulty, seed=seed)  # type: ignore[arg-type]
        response = self._client.post("/reset", json=payload.model_dump())
        response.raise_for_status()
        return TriageObservation.model_validate(response.json())

    def step(self, action: TriageAction) -> TriageObservation:
        """Submit a triage action. Returns the next observation + reward."""
        payload = StepRequest(action=action)
        response = self._client.post("/step", json=payload.model_dump())
        response.raise_for_status()
        return TriageObservation.model_validate(response.json())

    def state(self) -> EnvironmentState:
        """Get current environment state snapshot."""
        response = self._client.get("/state")
        response.raise_for_status()
        return EnvironmentState.model_validate(response.json())

    def health(self) -> dict[str, Any]:
        """Health check."""
        response = self._client.get("/health")
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> "CustomerSupportEnvClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying HTTP connection."""
        self._client.close()


class CustomerSupportEnvWSClient:
    """
    Asynchronous WebSocket client for the Customer Support Triage environment.

    Provides lower-latency interaction for high-frequency training loops.

    Usage (async)::

        async with CustomerSupportEnvWSClient("ws://localhost:8000") as client:
            obs = await client.reset("easy", seed=42)
            while not obs.done:
                action = my_agent.act(obs)
                obs = await client.step(action)
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> None:
        if not _HAS_WEBSOCKETS:
            raise ImportError("Install `websockets` to use the WebSocket client.")
        import uuid

        base = (
            base_url
            or os.environ.get("ENV_WS_URL", "ws://localhost:8000")
        ).rstrip("/")
        self._session_id = session_id or str(uuid.uuid4())
        self._url = f"{base}/ws/{self._session_id}"
        self._ws = None

    async def connect(self) -> None:
        import websockets as ws

        self._ws = await ws.connect(self._url)

    async def disconnect(self) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def reset(
        self, difficulty: str = "easy", seed: Optional[int] = None
    ) -> TriageObservation:
        msg = json.dumps({"type": "reset", "difficulty": difficulty, "seed": seed})
        await self._ws.send(msg)
        raw = await self._ws.recv()
        return TriageObservation.model_validate_json(raw)

    async def step(self, action: TriageAction) -> TriageObservation:
        msg = json.dumps({"type": "step", "action": action.model_dump()})
        await self._ws.send(msg)
        raw = await self._ws.recv()
        return TriageObservation.model_validate_json(raw)

    async def state(self) -> EnvironmentState:
        msg = json.dumps({"type": "state"})
        await self._ws.send(msg)
        raw = await self._ws.recv()
        return EnvironmentState.model_validate_json(raw)

    async def __aenter__(self) -> "CustomerSupportEnvWSClient":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.disconnect()
