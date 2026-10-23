"""Built-in guardrails (implemented as hooks). Custom guardrails follow the same pattern:
a class that inherits from `Hook` and overrides only the required method.
"""

from __future__ import annotations

import base64
import binascii
import functools
import inspect
import ipaddress
import math
import posixpath
import re
import urllib.parse
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Iterator
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, TypeVar, get_args
from urllib.parse import urlsplit

from pydantic import BaseModel
from pydantic.networks import AnyUrl

from .core.errors import ConfigError
from .core.hooks import CONTINUE, Decision, Hook, ToolInvocation
from .core.state import RunState
from .core.types import ModelRequest, ModelResponse, Risk, ToolResult

_T = TypeVar("_T")


def _at_least_one(items: tuple[_T, ...], guard: str) -> tuple[_T, ...]:
    if not items:
        raise ValueError(f"{guard} requires at least one pattern")
    return items


DEFAULT_PROTECTED: tuple[str, ...] = (
    "tests",
    "tests/*",
    "test/*",
    "conftest.py",
    ".github/*",
    ".gitlab-ci.yml",
    ".pre-commit-config.yaml",
    "evals/*",
    "*thresholds*",
    "pyproject.toml",
    "setup.cfg",
    "pytest.ini",
    "tox.ini",
)

KillCheck = Callable[[RunState], bool | Awaitable[bool]]


class KillSwitch(Hook):
    """Emergency stop: halts runs before the next model call, before every tool, and on every input.

    `check(state)` returns True when the kill switch is active — synchronously or async, e.g., a
    feature flag, a database row, or a file. Via `state.tenant_id`, it can be toggled per tenant.
    Fail-closed: if `check` raises (flag service unreachable) or returns a non-bool, the run blocks too.

    An already running tool is not interrupted; the run stops immediately afterward.
    If the kill switch hits an already approved action, it is answered as blocked: the approval
    is consumed, and after turning the switch off the model must request approval again.
    """

    def __init__(self, check: KillCheck, reason: str = "Kill switch is active") -> None:
        if not callable(check):
            raise TypeError("KillSwitch requires a check function")
        self._check = check
        self._reason = reason

    async def _decide(self, state: RunState) -> Decision:
        value = self._check(state)
        if inspect.isawaitable(value):
            value = await value
        if not isinstance(value, bool):
            raise TypeError(f"KillSwitch check must return bool, got {type(value).__name__}")
        return Decision.block(self._reason) if value else CONTINUE

    async def on_user_input(self, state: RunState, text: str) -> Decision:
        return await self._decide(state)

    async def before_model(self, state: RunState, request: ModelRequest) -> Decision:
        return await self._decide(state)

    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        return await self._decide(state)


_PATH_KEY = re.compile(
    r"^(paths?|files?|filename|filenames|target|destination|dest|dst|src|source|dir|directory)$|_(path|paths|file|files|dir)$"
)


class ProtectedPaths(Hook):
    """Reward-hacking protection: write tools must not touch protected paths.

    This prevents an agent from modifying the evaluators it is measured against (tests, CI, thresholds).
    For write tools (risk WRITE/IRREVERSIBLE), it checks — including inside lists and nested objects —
    - all arguments whose field name looks like a path (`path`, `files`, `target_path`, ...)
      or appears in `extra_keys`, and
    - **every** string value without whitespace, regardless of field name (`location`, `save_to`, `uri`, ...).
    Specify paths containing spaces under custom field names via `extra_keys`.

    Comparison is hardened against bypasses: case-insensitivity (important on macOS/Windows),
    `./`, `..`, backslashes, and absolute paths (every path suffix is checked, so `/repo/tests/x.py`
    matches `tests/*`).

    Boundary: shell-like tools accepting free-form command strings cannot be secured this way.
    Run such tools only inside a sandbox with read-only mounts.
    """

    def __init__(
        self,
        patterns: Iterable[str] = DEFAULT_PROTECTED,
        *,
        extra_keys: Iterable[str] = (),
        root: str | Path | None = None,
    ) -> None:
        self._patterns = _at_least_one(tuple(p.casefold().strip("/") for p in patterns), "ProtectedPaths")
        self._extra_keys = frozenset(k.casefold() for k in extra_keys)
        self._root: Path | None = None
        if root is not None:
            resolved = Path(root).resolve(strict=False)
            if not resolved.is_dir():
                raise ConfigError(f"ProtectedPaths: root directory does not exist: {resolved}")
            self._root = resolved

    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        if invocation.risk is Risk.READ:
            return CONTINUE
        for value in _path_values(invocation.args.model_dump(), self._extra_keys):
            if self.is_protected(value):
                return Decision.block(f"'{value}' is protected and must not be modified by the agent")
        return CONTINUE

    def is_protected(self, raw: str) -> bool:
        if _matches_patterns(raw, self._patterns):
            return True
        if self._root is not None:
            target = (self._root / raw.replace("\\", "/")).resolve(strict=False)
            if not target.is_relative_to(self._root):
                return True
            rel = target.relative_to(self._root).as_posix()
            if _matches_patterns(rel, self._patterns):
                return True
        return False


def _matches_patterns(raw: str, patterns: tuple[str, ...]) -> bool:
    normalized = posixpath.normpath(raw.replace("\\", "/")).casefold()
    segments = [s for s in normalized.split("/") if s not in ("", ".")]
    for start in range(len(segments)):
        candidate = "/".join(segments[start:])
        if any(fnmatchcase(candidate, pattern) for pattern in patterns):
            return True
    return False


DEFAULT_SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    ("private_key", r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"),
    (
        "api_token",
        r"(?<![A-Za-z0-9])(?:hf_[A-Za-z0-9]{20,}"
        r"|(?<!-)sk-proj-[A-Za-z0-9_-]{20,}"
        r"|(?<!-)sk-ant-[A-Za-z0-9_-]{20,}"
        r"|(?<!-)sk-svcacct-[A-Za-z0-9_-]{20,}"
        r"|(?<!-)sk-admin-[A-Za-z0-9_-]{20,}"
        r"|(?<!-)sk-[A-Za-z0-9]{20,}"
        r"|gh[pousr]_[A-Za-z0-9]{20,}"
        r"|github_pat_[A-Za-z0-9_]{20,}"
        r"|glpat-[A-Za-z0-9_-]{20,}"
        r"|xox[baprs]-[A-Za-z0-9-]{10,}"
        r"|AIza[A-Za-z0-9_-]{35}"
        r"|npm_[A-Za-z0-9]{30,}"
        r"|pypi-AgE[A-Za-z0-9_-]{30,}"
        r"|[sr]k_live_[A-Za-z0-9]{16,})",
    ),
    ("aws_access_key", r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])"),
    ("auth_bearer", r"(?i:\bBearer\s+)[A-Za-z0-9._~+/=-]{20,}"),
    ("auth_basic", r"(?i:\bAuthorization:\s*Basic\s+)[A-Za-z0-9+/=]{8,}"),
    ("jwt", r"(?<![A-Za-z0-9_-])eyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ("url_credentials", r"\b[a-zA-Z][a-zA-Z0-9+.-]{0,31}://[^/\s:@]+:[^/\s:@]+@[^\s/]+"),
    ("sendgrid_key", r"\bSG\.[A-Za-z0-9_-]{22}\.[A-Za-z0-9_-]{43}\b"),
    ("twilio_key", r"\b(?:SK|AC)[0-9a-fA-F]{32}\b"),
    ("slack_webhook", r"https://hooks\.slack\.com/services/T[A-Za-z0-9_]+/B[A-Za-z0-9_]+/[A-Za-z0-9_]+"),
    ("azure_connection_string", r"DefaultEndpointsProtocol=https;AccountName=[^;]+;AccountKey=[A-Za-z0-9+/=]{40,}"),
)

_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SECRET_FIELD_KEY = re.compile(
    r"(?:^|[^a-z])(?:pass(?:word)?|secret|token|api_?key|auth_?token|client_?secret|access_?token)s?(?:$|[^a-z])"
)


def _key_words(key: str) -> str:
    """A field name as lower-case words (`dbPassword` becomes `db_password`): one pattern covers both spellings."""
    return _CAMEL_BOUNDARY.sub("_", key).casefold()


def _is_secret_key(key: str) -> bool:
    """True for field names such as `password`, `api_key`, `dbPassword`, or `refreshToken`."""
    return _SECRET_FIELD_KEY.search(_key_words(key)) is not None


_ASSIGNMENT_REGEX = re.compile(
    # A word ending in a credential name (`db_password`, `dbPassword`), an optional closing quote (JSON), `=` or `:`
    r"""\b(?i:[a-z0-9_]*(?:password|secret|token|api_?key|client_?secret|access_?token))['"]?\s*[:=]\s*['"]?([^\s'"]{8,})['"]?"""
)
_PLACEHOLDER_WORDS = frozenset(
    {"changeme", "example", "placeholder", "dummy", "test", "none", "null", "admin", "password", "secret", "token"}
)


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    total = len(s)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


def _is_placeholder(val: str) -> bool:
    s = val.strip().casefold()
    if s in _PLACEHOLDER_WORDS:
        return True
    return bool(
        (s.startswith("${") and s.endswith("}")) or (s.startswith("<") and s.endswith(">")) or s.startswith("***")
    )


def _looks_like_secret(value: str) -> bool:
    """A value that is no placeholder, is long enough, and has enough variety to be a credential."""
    return not _is_placeholder(value) and len(value) >= 8 and _shannon_entropy(value) >= 2.5


def _find_assignment_secret(text: str) -> str | None:
    if "=" not in text and ":" not in text:  # cheap: an assignment needs one of them
        return None
    for match in _ASSIGNMENT_REGEX.finditer(text):
        if _looks_like_secret(match.group(1)):
            return "assignment"
    return None


_BASE64_CANDIDATE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{16,}=*")
"""A whole base64 run with any `=` after it. One start per run and no lookahead keep the search linear.
A `=` before the run is allowed: `key=<base64>` is the common case."""
_HEX_CANDIDATE = re.compile(r"\b[0-9a-fA-F]{32,}\b")
_PERCENT_ESCAPE = re.compile(r"%[0-9a-fA-F]{2}")
_PRINTABLE_RUN = re.compile(rb"[\t\n\r\x20-\x7e]{12,}")
_DECODE_BUDGET = 50_000
"""Characters of decoded text that `decode=True` scans per text, so a large input cannot make the scan slow."""


def _printable_runs(raw: bytes) -> Iterator[str]:
    """The readable stretches of `raw` (12 characters or more). Binary data, such as a hash, has none."""
    for run in _PRINTABLE_RUN.findall(raw):
        yield run.decode("ascii")


def _decode_base64(candidate: str) -> Iterator[str]:
    """Readable text in a base64 run, decoded at each of the 4 offsets.

    A path before the data is made of base64 characters too (`example/api/v1/upload/<base64>`): it is part of the run
    and moves the 4-character blocks. The path only turns into a few garbage bytes in front of the data, which
    `_printable_runs` drops, so the data is found at the offset where its blocks line up. The `/` that ends the
    path decodes to a byte that is never a letter or digit, so it cannot hide the start of a token.
    """
    for offset in range(4):
        core = candidate[offset:].rstrip("=")  # lenient decoders accept missing or extra padding, so the check does too
        try:
            raw = base64.b64decode(core + "=" * (-len(core) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        yield from _printable_runs(raw)


def _decoded_texts(text: str) -> Iterator[str]:
    """The text that one step of URL, base64, or hex decoding reveals in `text`."""
    if _PERCENT_ESCAPE.search(text):
        unquoted = urllib.parse.unquote(text)
        if unquoted != text:
            yield unquoted
    for match in _BASE64_CANDIDATE.finditer(text):
        yield from _decode_base64(match.group(0))
    for match in _HEX_CANDIDATE.finditer(text):
        digits = match.group(0)
        if len(digits) % 2 == 0:
            yield from _printable_runs(bytes.fromhex(digits))


def _decode_and_scan(text: str, scanner: Callable[[str], str | None]) -> str | None:
    """Scans what `_decoded_texts` finds, once, up to `_DECODE_BUDGET` characters in all."""
    pieces: list[str] = []
    left = _DECODE_BUDGET
    for piece in _decoded_texts(text):
        pieces.append(piece[:left])
        left -= len(pieces[-1])
        if left <= 0:
            break
    return scanner("\n".join(pieces)) if pieces else None


class SecretGuard(Hook):
    """Prevents credentials and private keys from leaking through tool arguments, tool outputs, or inputs.

    Usable both as a `Hook` inside `HookPipeline` and standalone via `find_secret(text)` and `redact(text)`.
    Error reasons report only the pattern category name, never the matched secret value. Model responses are
    checked too (`check_model_output`), so a secret the model repeats does not reach the user or the history.
    """

    def __init__(
        self,
        patterns: Iterable[tuple[str, str]] = DEFAULT_SECRET_PATTERNS,
        *,
        check_user_input: bool = False,
        check_model_output: bool = True,
        assignments: bool = False,
        decode: bool = False,
    ) -> None:
        listed = _at_least_one(tuple(patterns), "SecretGuard")
        self._patterns = tuple((name, re.compile(expr)) for name, expr in listed)
        self._check_user_input = check_user_input
        self._check_model_output = check_model_output
        self._assignments = assignments
        self._decode = decode

    def find_secret(self, text: str) -> str | None:
        """Returns the category name of the first detected secret in `text`, or `None`."""
        direct = self._find_secret_direct(text)
        if direct is not None:
            return direct
        if self._assignments:
            assigned = _find_assignment_secret(text)
            if assigned is not None:
                return assigned
        if self._decode:
            decoded = _decode_and_scan(text, self._find_secret_direct)
            if decoded is not None:
                return decoded
        return None

    def _find_secret_direct(self, text: str) -> str | None:
        for name, pattern in self._patterns:
            if pattern.search(text) is not None:
                return name
        return None

    def redact(self, text: str, *, replacement: str = "[REDACTED]") -> str:
        """Replaces all detected secrets in `text` with `replacement`."""
        cleaned = text
        for _, pattern in self._patterns:
            cleaned = pattern.sub(replacement, cleaned)
        return cleaned

    def _check(self, texts: Iterable[str], blocked: str) -> Decision:
        """Blocks with `blocked` and the category of the first secret found in `texts` (never the secret itself)."""
        for text in texts:
            kind = self.find_secret(text)
            if kind is not None:
                return Decision.block(f"{blocked}: contains potential secret ({kind})")
        return CONTINUE

    async def on_user_input(self, state: RunState, text: str) -> Decision:
        if not self._check_user_input:
            return CONTINUE
        return self._check([text], "User input blocked")

    async def after_model(self, state: RunState, response: ModelResponse) -> Decision:
        if not self._check_model_output:
            return CONTINUE
        return self._check([response.message.text], "Model response blocked")

    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        args = invocation.args.model_dump(mode="json")
        if self._assignments:
            for key, value, is_dict_key in _walk_strings(args):
                if key and value and not is_dict_key and _is_secret_key(key) and _looks_like_secret(value):
                    return Decision.block("Tool arguments blocked: contains potential secret (assignment)")
        return self._check(_all_strings(args), "Tool arguments blocked")

    async def after_tool(self, state: RunState, invocation: ToolInvocation, result: ToolResult) -> Decision:
        return self._check([result.content], "Tool output withheld")


_WHATWG_SPECIAL_SCHEMES: tuple[str, ...] = ("https", "http", "wss", "ws", "ftp", "file")


_OPAQUE_URL_BRANCHES: tuple[tuple[str, str], ...] = (
    # (marker, branch) for URLs without a slash after the scheme; each branch starts with its marker in any case
    ("mailto:", r"\b(?ai:mailto):[^\s<>()\"'\]@]*@[^\s<>()\"'\]]*"),
    ("data:", r"\b(?ai:data):(?:[a-zA-Z0-9.+-]+/[a-zA-Z0-9.+-]+|[;,])[^\s<>()\"'\]]*"),
    ("javascript:", r"\b(?ai:javascript):[^\s<>()\"'\]]+"),
)


def _build_url_patterns() -> tuple[re.Pattern[str], tuple[str, ...]]:
    """`_URL_IN_TEXT` and its quick pre-check `_URL_MARKERS` from one table, so they cannot drift apart."""
    special = "|".join(_WHATWG_SPECIAL_SCHEMES)
    branches = (
        # Schemes are case-insensitive in ASCII (WHATWG): `HTTPS:/host` is a URL with a host, like `https://host`
        rf"\b(?ai:{special}):[/\\]+[^\s<>()\"'\]]*",
        r"\b[a-zA-Z][a-zA-Z0-9+.-]{1,31}:[/\\]{2,}[^\s<>()\"'\]]*",
        *(branch for _, branch in _OPAQUE_URL_BRANCHES),
    )
    # The first two branches always contain `:/` or `:\`; the others start with their marker
    markers = (":/", ":\\", *(marker for marker, _ in _OPAQUE_URL_BRANCHES))
    return re.compile("|".join(branches)), markers


_URL_IN_TEXT, _URL_MARKERS = _build_url_patterns()
"""Every `_URL_IN_TEXT` match contains one of these once the text is lower-cased: text without them has no URL."""
_TRAILING_PUNCT = ".,;:!?'\")"
_URL_KEY_WORD = re.compile(r"(?:^|[^a-z])(?:url|uri|href|endpoint|webhook)s?(?:$|[^a-z])")
_HOST_CHARS = re.compile(r"[a-z0-9.-]+")


def _is_url_key(key: str | None, extra: frozenset[str]) -> bool:
    if key is None:
        return False
    folded = key.casefold()
    return folded in extra or _URL_KEY_WORD.search(_key_words(key)) is not None


def _is_url_typed_field(annotation: Any) -> bool:
    if annotation is None:
        return False
    candidates = [annotation, *get_args(annotation)]
    for cand in candidates:
        if isinstance(cand, type) and (issubclass(cand, AnyUrl) or "url" in cand.__name__.casefold()):
            return True
    return False


@functools.lru_cache(maxsize=256)
def _url_field_names(model: type[BaseModel]) -> frozenset[str]:
    """Lower-case names of the fields of `model` that are typed as URLs. It depends on the class only."""
    return frozenset(
        name.casefold() for name, info in model.model_fields.items() if _is_url_typed_field(info.annotation)
    )


class EgressGuard(Hook):
    """Prevents data exfiltration and SSRF through URLs in tool arguments and model responses.

    `allowed_domains` is required, so the policy is always a deliberate choice. Pass `()` to block every URL.
    By default only `https://` URLs pointing to explicitly allowed domains are permitted.
    Direct IP addresses, `localhost`, `.local`/`.internal` hosts, and URLs with embedded credentials
    are always rejected. Usable both as a `Hook` and standalone via `check_url(url)` / `check_text(text)`.

    Two checks apply to tool arguments. Free text is scanned for URL-like parts (`scheme://`, `scheme:/`,
    `scheme:\\`, and `mailto:`, `data:`, `file:`, and similar). A value in a field whose name says it is a
    URL (`url`, `uri`, `href`, `endpoint`, `webhook`, also as `base_url` or `callbackUrl`; extend with
    `url_keys`) must be a complete allowed URL, so `evil.example/x` and `//evil.example/x` are blocked there.
    A bare host or path in a differently named field is not found. Name such fields in `url_keys`.
    """

    def __init__(
        self,
        allowed_domains: Iterable[str],
        *,
        allow_http: bool = False,
        check_model_output: bool = True,
        url_keys: Iterable[str] = (),
        ignore_keys: Iterable[str] = (),
    ) -> None:
        if isinstance(allowed_domains, str):
            raise ConfigError("EgressGuard: allowed_domains must be a list of domains, not a single string")
        self._domains = tuple(d.strip().casefold().lstrip("*") for d in allowed_domains if d.strip())
        self._allow_http = allow_http
        self._check_model_output = check_model_output
        self._url_keys = frozenset(k.casefold() for k in url_keys)
        self._ignore_keys = frozenset(k.casefold() for k in ignore_keys)

    def check_url(self, url: str) -> str | None:
        """Returns `None` if `url` is allowed, or a concise reason string if blocked."""
        if any(c.isspace() or not c.isprintable() or c == "\\" for c in url):
            return "URL contains whitespace, control characters, or a backslash"
        try:
            parsed = urlsplit(url)
            _ = parsed.port  # reading the port raises ValueError when it is invalid
        except ValueError:
            return "Malformed URL"
        scheme = parsed.scheme.casefold()
        allowed_schemes = ("https", "http") if self._allow_http else ("https",)
        if scheme not in allowed_schemes:
            return f"URL scheme '{scheme or 'none'}' is not allowed"
        if parsed.username is not None or parsed.password is not None:
            return "URL must not contain embedded credentials"
        host = (parsed.hostname or "").casefold().rstrip(".")
        if not host:
            return "URL is missing a hostname"
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            return "Host is a local or internal address"
        if _is_ip_literal(host):
            return "Direct IP addresses are not allowed"
        if _HOST_CHARS.fullmatch(host) is None:
            return "Host contains characters that are not allowed"
        if not _domain_matches(host, self._domains):
            return "Host is not in the allowed domains list"
        return None

    def check_text(self, text: str) -> str | None:
        """Scans `text` for URLs and returns the first violation reason, or `None`."""
        lowered = text.lower()  # ASCII letters keep their place, so a scheme in any case shows its marker
        if not any(marker in lowered for marker in _URL_MARKERS):  # cheap: most text holds no URL at all
            return None
        for match in _URL_IN_TEXT.finditer(text):
            raw_url = match.group(0).rstrip(_TRAILING_PUNCT)
            reason = self.check_url(raw_url)
            if reason is not None:
                return reason
        return None

    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        url_fields = _url_field_names(type(invocation.args))
        for key, value, is_dict_key in _walk_strings(invocation.args.model_dump(mode="json")):
            if not value:
                continue
            key_fold = key.casefold() if key else ""
            is_ignored = key_fold in self._ignore_keys
            is_strict_url = (
                not is_dict_key and not is_ignored and (key_fold in url_fields or _is_url_key(key, self._url_keys))
            )
            if is_strict_url:
                reason = self.check_url(value)
                if reason is not None:
                    return Decision.block(f"Outbound URL blocked: {reason}")
            else:
                reason = self.check_text(value)
                if reason is not None:
                    if is_ignored and key:
                        return Decision.block(f"Outbound URL blocked in field '{key}': {reason}")
                    return Decision.block(f"Outbound URL blocked: {reason}")
        return CONTINUE

    async def after_model(self, state: RunState, response: ModelResponse) -> Decision:
        if not self._check_model_output:
            return CONTINUE
        reason = self.check_text(response.message.text)
        if reason is not None:
            return Decision.block(f"Model response blocked by egress policy: {reason}")
        return CONTINUE


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
    except ValueError:
        return False
    return True


def _domain_matches(host: str, allowed: tuple[str, ...]) -> bool:
    for pattern in allowed:
        if pattern.startswith("."):
            suffix = pattern[1:]
            if host == suffix or host.endswith(pattern):
                return True
        elif host == pattern:
            return True
    return False


DEFAULT_WEAKENING_PATTERNS: tuple[str, ...] = (
    r"#\s*type:\s*ignore",
    r"#\s*pyright:\s*ignore",
    r"#\s*noqa\b",
    r"#\s*(?:ruff|flake8)\s*:\s*noqa\b",
    r"#\s*pylint\s*:\s*(?:disable|skip-file)",
    r"#\s*mypy\s*:\s*ignore-errors",
    r"#\s*nosec\b",
    r"#\s*pragma:\s*no\s*cover",
    r"\bpytest\.mark\.(?:skipif|skip|xfail)\b",
    r"\bpytest\.(?:skip\s*\(|importorskip\b)",
    r"\bunittest\.skip(?:If|Unless)?\b",
    r"//\s*@ts-(?:ignore|nocheck|expect-error)\b",
    r"\bsys\.exit\s*\(\s*0\s*\)",
    r"eslint-disable\b",
    r"//\s*nolint\b",
    r"#\[allow\(",
    r"#\[ignore\]",
    r"@SuppressWarnings\b",
    r"@Disabled\b",
    r"\b(?:it|describe)\.skip\s*\(",
    r"\bxit\s*\(",
    r"\bt\.Skip(?:Now|f)?\s*\(",
)


class WeakeningGuard(Hook):
    """Prevents write tools from inserting directives that silence linters, typecheckers, or tests."""

    def __init__(self, patterns: Iterable[str] = DEFAULT_WEAKENING_PATTERNS) -> None:
        self._patterns = _at_least_one(tuple(re.compile(p, re.IGNORECASE) for p in patterns), "WeakeningGuard")

    def check_text(self, text: str) -> str | None:
        """Returns the first matched check-weakening directive in `text`, or `None`."""
        for pattern in self._patterns:
            match = pattern.search(text)
            if match is not None:
                return match.group(0).strip()
        return None

    async def before_tool(self, state: RunState, invocation: ToolInvocation) -> Decision:
        if invocation.risk is Risk.READ:
            return CONTINUE
        for value in _all_strings(invocation.args.model_dump()):
            found = self.check_text(value)
            if found is not None:
                return Decision.block(f"Write blocked: contains check-weakening directive '{found}'")
        return CONTINUE


def _walk_strings(data: Any, key: str | None = None, *, top: bool = True) -> Iterator[tuple[str | None, str, bool]]:
    """Yields `(field_key, string_value, is_nested_dict_key)` across arbitrary JSON-like structures."""
    if isinstance(data, dict):
        mapping: dict[Any, Any] = data  # pyright: ignore[reportUnknownVariableType] – JSON-like data
        for k, v in mapping.items():
            if not top:
                yield (key, str(k), True)
            yield from _walk_strings(v, str(k), top=False)
    elif isinstance(data, list | tuple | set):
        items: list[Any] | tuple[Any, ...] | set[Any] = data  # pyright: ignore[reportUnknownVariableType]
        for item in items:
            yield from _walk_strings(item, key, top=False)
    elif isinstance(data, str):
        yield (key, data, False)


def _all_strings(data: Any) -> Iterator[str]:
    for _, value, _ in _walk_strings(data):
        if value:
            yield value


def _path_values(data: Any, extra_keys: frozenset[str]) -> Iterator[str]:
    """All values that could be a path: always under path-like field names, otherwise any string without whitespace.

    Free text (email body, comment) almost always contains whitespace and remains allowed. A single
    token like `pyproject.toml` under any field name is checked — fail-closed, better to block once
    too often than allow a modified test file. Keys of nested dictionaries also count when they look
    like paths (`files={"tests/x.py": "..."}`: containing `/`, `\\`, or `.`) — plain names like `tests`
    and top-level tool model field names do not.
    """
    for key, value, is_dict_key in _walk_strings(data):
        if is_dict_key:
            if any(sep in value for sep in ("/", "\\", ".")):
                yield value
        else:
            named = key is not None and (_PATH_KEY.search(key.casefold()) is not None or key.casefold() in extra_keys)
            if named or (value and not any(c.isspace() for c in value)):
                yield value
