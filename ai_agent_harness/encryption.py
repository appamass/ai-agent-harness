"""Optional checkpoint encryption for stores at rest.

Install with: `pip install "ai-agent-harness[encryption]"`
"""

from __future__ import annotations

import base64
import json
import os
from collections.abc import Mapping
from typing import Final, Protocol, cast

from .core.errors import ConfigError, InvalidStateError

ENCRYPTED_PREFIX: Final = "enc:v1:"

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
except ImportError as exc:
    raise ConfigError("Checkpoint encryption needs cryptography: pip install 'ai-agent-harness[encryption]'") from exc


class StateCodec(Protocol):
    """Protocol for encrypting and decrypting serialized RunState payloads."""

    def encode(self, state_json: str, *, tenant_id: str, run_id: str, revision: int) -> str:
        """Encrypts `state_json` into an authenticated envelope string."""
        ...

    def decode(self, payload: str, *, tenant_id: str, run_id: str) -> str:
        """Decrypts `payload` back into the plaintext `state_json`."""
        ...


class AesGcmCodec:
    """Envelope encryption using AES-256-GCM with authenticated metadata header.

    Key rotation: Pass multiple keys in `keys` mapping; new checkpoints are encrypted with `active`.
    Each key must be exactly 32 bytes (256 bits).
    """

    def __init__(self, keys: Mapping[str, bytes], *, active: str) -> None:
        if not keys:
            raise ConfigError("AesGcmCodec requires at least one key")
        if active not in keys:
            raise ConfigError(f"Active key {active!r} is not in keys")

        self._active = active
        self._ciphers: dict[str, AESGCM] = {}

        for kid, key in keys.items():
            if not isinstance(key, bytes) or len(key) != 32:
                raise ConfigError(f"Key {kid!r} must be exactly 32 bytes for AES-256")
            try:
                self._ciphers[kid] = AESGCM(key)
            except ValueError as exc:
                raise ConfigError(f"Invalid key configuration for key id {kid!r}") from exc

    def __repr__(self) -> str:
        return f"AesGcmCodec(keys={sorted(self._ciphers.keys())!r}, active={self._active!r})"

    def encode(self, state_json: str, *, tenant_id: str, run_id: str, revision: int) -> str:
        nonce_bytes = os.urandom(12)
        nonce_b64 = base64.b64encode(nonce_bytes).decode("ascii")
        header = {
            "kid": self._active,
            "nonce": nonce_b64,
            "revision": revision,
            "run_id": run_id,
            "tenant_id": tenant_id,
        }
        header_json = json.dumps(header, separators=(",", ":"), sort_keys=True)
        aad = f"{ENCRYPTED_PREFIX}{header_json}".encode()
        cipher = self._ciphers[self._active]
        ciphertext_bytes = cipher.encrypt(nonce_bytes, state_json.encode(), aad)
        ciphertext_b64 = base64.b64encode(ciphertext_bytes).decode("ascii")
        return f"{ENCRYPTED_PREFIX}{header_json}\n{ciphertext_b64}"

    def decode(self, payload: str, *, tenant_id: str, run_id: str) -> str:
        if not payload.startswith(ENCRYPTED_PREFIX):
            raise InvalidStateError("Payload is not encrypted")
        body = payload[len(ENCRYPTED_PREFIX) :]
        header_json, sep, ciphertext_b64 = body.partition("\n")
        if not sep:
            raise InvalidStateError("Invalid encrypted envelope: missing separator")

        try:
            header = json.loads(header_json)
        except Exception as exc:
            raise InvalidStateError("Invalid encrypted envelope: corrupted header") from exc

        if not isinstance(header, dict):
            raise InvalidStateError("Invalid encrypted envelope: header must be a JSON object")

        header_dict = cast(dict[str, object], header)
        kid = header_dict.get("kid")
        hdr_tenant = header_dict.get("tenant_id")
        hdr_run = header_dict.get("run_id")
        hdr_revision = header_dict.get("revision")
        nonce_b64 = header_dict.get("nonce")

        if (
            not isinstance(kid, str)
            or not isinstance(hdr_tenant, str)
            or not isinstance(hdr_run, str)
            or not isinstance(hdr_revision, int)
            or not isinstance(nonce_b64, str)
        ):
            raise InvalidStateError("Invalid encrypted envelope: missing or invalid header fields")

        if (hdr_tenant, hdr_run) != (tenant_id, run_id):
            raise InvalidStateError(
                f"Checkpoint envelope belongs to {hdr_tenant}/{hdr_run}, expected {tenant_id}/{run_id}"
            )

        cipher = self._ciphers.get(kid)
        if cipher is None:
            raise ConfigError(f"Unknown key id {kid!r}")

        try:
            nonce_bytes = base64.b64decode(nonce_b64, validate=True)
        except Exception as exc:
            raise InvalidStateError("Invalid nonce encoding in envelope") from exc

        try:
            ciphertext_bytes = base64.b64decode(ciphertext_b64, validate=True)
        except Exception as exc:
            raise InvalidStateError("Invalid ciphertext encoding in envelope") from exc

        aad = f"{ENCRYPTED_PREFIX}{header_json}".encode()
        try:
            plaintext_bytes = cipher.decrypt(nonce_bytes, ciphertext_bytes, aad)
        except Exception as exc:
            raise InvalidStateError("Checkpoint decryption failed: corrupted ciphertext or authentication tag") from exc

        return plaintext_bytes.decode("utf-8")
