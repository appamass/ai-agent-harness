"""Tests for optional checkpoint encryption (AesGcmCodec) and envelope integrity."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from ai_agent_harness import (
    InMemoryCheckpointStore,
    JsonFileCheckpointStore,
)
from ai_agent_harness.core import InvalidStateError
from ai_agent_harness.core.errors import ConfigError
from ai_agent_harness.encryption import ENCRYPTED_PREFIX, AesGcmCodec
from ai_agent_harness.persistence_sqlite import SqliteCheckpointStore

from .conftest import run_sync, waiting_state


def test_codec_validates_key_requirements() -> None:
    valid_key = os.urandom(32)

    with pytest.raises(ConfigError, match="requires at least one key"):
        AesGcmCodec({}, active="k1")

    with pytest.raises(ConfigError, match="Active key 'k2' is not in keys"):
        AesGcmCodec({"k1": valid_key}, active="k2")

    with pytest.raises(ConfigError, match="must be exactly 32 bytes"):
        AesGcmCodec({"k1": b"too-short"}, active="k1")

    with pytest.raises(ConfigError, match="must be exactly 32 bytes"):
        AesGcmCodec({"k1": valid_key + b"x"}, active="k1")

    from typing import cast

    with pytest.raises(ConfigError, match="must be exactly 32 bytes"):
        AesGcmCodec(cast(dict[str, bytes], {"k1": "string-key" * 4}), active="k1")


def test_codec_repr_does_not_reveal_key_material() -> None:
    secret_key = os.urandom(32)
    codec = AesGcmCodec({"main": secret_key}, active="main")
    repr_str = repr(codec)
    assert "main" in repr_str
    assert secret_key.hex() not in repr_str
    assert str(secret_key) not in repr_str


def test_codec_encode_decode_roundtrip() -> None:
    key = os.urandom(32)
    codec = AesGcmCodec({"k1": key}, active="k1")
    state = waiting_state("t1", "r1")
    raw_json = state.model_dump_json()

    encoded1 = codec.encode(raw_json, tenant_id="t1", run_id="r1", revision=1)
    encoded2 = codec.encode(raw_json, tenant_id="t1", run_id="r1", revision=1)

    assert encoded1.startswith(ENCRYPTED_PREFIX)
    assert encoded1 != encoded2  # Random nonces produce unique ciphertexts
    assert codec.decode(encoded1, tenant_id="t1", run_id="r1") == raw_json
    assert codec.decode(encoded2, tenant_id="t1", run_id="r1") == raw_json


def test_aad_and_tamper_detection() -> None:
    key = os.urandom(32)
    codec = AesGcmCodec({"k1": key}, active="k1")
    state = waiting_state("t1", "r1")
    raw_json = state.model_dump_json()

    encoded = codec.encode(raw_json, tenant_id="t1", run_id="r1", revision=1)
    body = encoded[len(ENCRYPTED_PREFIX) :]
    header_json, _, ciphertext_b64 = body.partition("\n")
    header = json.loads(header_json)

    # 1. Tampering with header fields
    tampered_header = dict(header)
    tampered_header["revision"] = 2
    tampered_encoded = f"{ENCRYPTED_PREFIX}{json.dumps(tampered_header, sort_keys=True)}\n{ciphertext_b64}"
    with pytest.raises(InvalidStateError, match="decryption failed"):
        codec.decode(tampered_encoded, tenant_id="t1", run_id="r1")

    # 2. Tampering with tenant_id in header
    tampered_header = dict(header)
    tampered_header["tenant_id"] = "t2"
    tampered_encoded = f"{ENCRYPTED_PREFIX}{json.dumps(tampered_header, sort_keys=True)}\n{ciphertext_b64}"
    with pytest.raises(InvalidStateError, match="belongs to"):
        codec.decode(tampered_encoded, tenant_id="t1", run_id="r1")

    # 3. Tampering with ciphertext
    corrupted_cipher = ciphertext_b64[:-4] + "AAAA"
    tampered_encoded = f"{ENCRYPTED_PREFIX}{header_json}\n{corrupted_cipher}"
    with pytest.raises(InvalidStateError, match="decryption failed"):
        codec.decode(tampered_encoded, tenant_id="t1", run_id="r1")

    # 4. Truncated envelope
    with pytest.raises(InvalidStateError, match="missing separator"):
        codec.decode(f"{ENCRYPTED_PREFIX}{header_json}", tenant_id="t1", run_id="r1")

    # 5. Non-encrypted payload
    with pytest.raises(InvalidStateError, match="not encrypted"):
        codec.decode(raw_json, tenant_id="t1", run_id="r1")


def test_key_rotation() -> None:
    key_old = os.urandom(32)
    key_new = os.urandom(32)

    codec_old = AesGcmCodec({"k1": key_old}, active="k1")
    codec_rotated = AesGcmCodec({"k1": key_old, "k2": key_new}, active="k2")

    state = waiting_state("t1", "r1")
    raw_json = state.model_dump_json()

    # Encrypted under old key k1
    enc_old = codec_old.encode(raw_json, tenant_id="t1", run_id="r1", revision=1)
    assert json.loads(enc_old[len(ENCRYPTED_PREFIX) :].partition("\n")[0])["kid"] == "k1"

    # Rotated codec can decrypt old key k1
    assert codec_rotated.decode(enc_old, tenant_id="t1", run_id="r1") == raw_json

    # New writes use active key k2
    enc_new = codec_rotated.encode(raw_json, tenant_id="t1", run_id="r1", revision=2)
    assert json.loads(enc_new[len(ENCRYPTED_PREFIX) :].partition("\n")[0])["kid"] == "k2"
    assert codec_rotated.decode(enc_new, tenant_id="t1", run_id="r1") == raw_json

    # Codec with key k1 revoked cannot decrypt enc_new
    codec_revoked = AesGcmCodec({"k2": key_new}, active="k2")
    with pytest.raises(ConfigError, match="Unknown key id 'k1'"):
        codec_revoked.decode(enc_old, tenant_id="t1", run_id="r1")


def test_unencrypted_state_rejected_when_codec_is_set(tmp_path: Path) -> None:
    codec = AesGcmCodec({"k1": os.urandom(32)}, active="k1")
    plain_store = SqliteCheckpointStore(tmp_path / "plain.sqlite")
    run_sync(plain_store.setup())
    state = waiting_state("t1", "r1")
    run_sync(plain_store.save(state))

    # Reopening the database with an encrypted store rejects unencrypted rows
    enc_store = SqliteCheckpointStore(tmp_path / "plain.sqlite", codec=codec)
    with pytest.raises(InvalidStateError, match="Unencrypted checkpoint rejected"):
        run_sync(enc_store.load("t1", "r1"))


def test_encrypted_state_rejected_when_codec_is_not_set(tmp_path: Path) -> None:
    codec = AesGcmCodec({"k1": os.urandom(32)}, active="k1")
    enc_store = SqliteCheckpointStore(tmp_path / "enc.sqlite", codec=codec)
    run_sync(enc_store.setup())
    state = waiting_state("t1", "r1")
    run_sync(enc_store.save(state))

    # Reopening without codec rejects encrypted rows
    plain_store = SqliteCheckpointStore(tmp_path / "enc.sqlite")
    with pytest.raises(InvalidStateError, match="Encrypted checkpoint cannot be loaded without a codec"):
        run_sync(plain_store.load("t1", "r1"))


def test_migration_recipe_roundtrip(tmp_path: Path) -> None:
    codec = AesGcmCodec({"k1": os.urandom(32)}, active="k1")
    db_path = tmp_path / "migrate.sqlite"
    plain_store = SqliteCheckpointStore(db_path)
    run_sync(plain_store.setup())

    # Write 3 plain runs
    for run_id in ("r1", "r2", "r3"):
        st = waiting_state("tenant1", run_id)
        run_sync(plain_store.save(st))

    # Migration recipe: open with plain store and encrypted store, load -> save
    enc_store = SqliteCheckpointStore(db_path, codec=codec)
    for run_info in run_sync(plain_store.list_runs("tenant1")):
        current_state = run_sync(plain_store.load("tenant1", run_info.run_id))
        assert current_state is not None
        run_sync(enc_store.save(current_state))

    # Now all runs load cleanly from encrypted store
    for run_id in ("r1", "r2", "r3"):
        loaded = run_sync(enc_store.load("tenant1", run_id))
        assert loaded is not None
        assert loaded.run_id == run_id

    # Plain store now sees encrypted rows and refuses to load them without codec
    with pytest.raises(InvalidStateError, match="Encrypted checkpoint cannot be loaded"):
        run_sync(plain_store.load("tenant1", "r1"))


def test_json_file_store_with_encryption(tmp_path: Path) -> None:
    codec = AesGcmCodec({"k1": os.urandom(32)}, active="k1")
    dir_path = tmp_path / "file_enc"
    store = JsonFileCheckpointStore(dir_path, codec=codec)

    state = waiting_state("t1", "r1")
    run_sync(store.save(state))
    assert state.revision == 1

    file_text = (dir_path / "t1" / "r1.json").read_text(encoding="utf-8")
    assert file_text.startswith(ENCRYPTED_PREFIX)
    # The actual state JSON is not present in cleartext
    assert '"agent_name":"a"' not in file_text

    loaded = run_sync(store.load("t1", "r1"))
    assert loaded is not None
    assert loaded.run_id == "r1"
    assert loaded.revision == 1

    # Second save increments revision and stays encrypted
    run_sync(store.save(state))
    assert state.revision == 2
    loaded2 = run_sync(store.load("t1", "r1"))
    assert loaded2 is not None and loaded2.revision == 2


def test_in_memory_store_with_encryption() -> None:
    codec = AesGcmCodec({"k1": os.urandom(32)}, active="k1")
    store = InMemoryCheckpointStore(codec=codec)

    state = waiting_state("t1", "r1")
    run_sync(store.save(state))
    assert state.revision == 1

    loaded = run_sync(store.load("t1", "r1"))
    assert loaded is not None
    assert loaded.run_id == "r1"
    assert loaded.revision == 1
