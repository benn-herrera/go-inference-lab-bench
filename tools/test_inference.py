#!/usr/bin/env python3
"""INVARIANT: stdlib only. No third-party dependencies. Ever.
Adding `pip install` of anything is not on the table — if you reach for one,
stop and find a stdlib path, or talk it through with the maintainer first.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
from pathlib import Path

from prompt_parser import parse_prompt_content, PromptParseError

THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parent.parent

API_CONFIG_TOML = PROJECT_DIR / "config" / "api_config.toml"
ARCH_TOML_DIR = PROJECT_DIR / "models" / "arch"
MODELS_DIR = PROJECT_DIR / "models"
BENCH_BIN = PROJECT_DIR / "bin" / "bench"


def read_bench_port() -> str:
    pat = re.compile(r"^[ \t]*port[ \t]*=[ \t]*([^\s#]+)")
    with API_CONFIG_TOML.open("r", encoding="utf-8") as f:
        for line in f:
            m = pat.match(line)
            if m:
                return m.group(1).strip()
    raise RuntimeError(f"no port found in {API_CONFIG_TOML}")


def scan_diffusion_arch_names() -> list[str]:
    pat = re.compile(r'generation\s*=\s*"diffusion"')
    names: list[str] = []
    for path in sorted(ARCH_TOML_DIR.glob("*.arch.toml")):
        try:
            with path.open("r", encoding="utf-8") as f:
                if any(pat.search(line) for line in f):
                    names.append(path.name[: -len(".arch.toml")])
        except OSError:
            continue
    return names


def env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def env_bool(name: str, default: bool) -> bool:
    """Bash semantics: literal 'true' is True; anything else (or unset) is False.

    `default` applies only when the env var is unset.
    """
    raw = os.environ.get(name)
    return (raw == "true") if raw is not None else default


def env_tristate(name: str, default: bool | None) -> bool | None:
    """Bash semantics: literal 'true'/'false'/'null' map to True/False/None.

    `default` applies only when the env var is unset. Any other value also
    falls back to `default` (matches the script's permissive parsing).
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    return {"true": True, "false": False, "null": None}.get(raw, default)


def jbool(x: bool | None) -> str:
    """Render a bool/Optional[bool] as a JSON literal: 'true' / 'false' / 'null'.

    Used both for loop-mode state echoing (matches bash output) and inline
    JSON payload construction.
    """
    return json.dumps(x)


def toggle(v: bool) -> bool:
    return not v


def rotate(v: bool | None) -> bool | None:
    """True → False → None → True. Mirrors the bash `rotate` helper."""
    if v is True:
        return False
    if v is False:
        return None
    return True


class State:
    def __init__(self) -> None:
        self.LOOP_MODE = False

        self.ALL_MODELS_NO_DIFFUSION = env_bool("ALL_MODELS_NO_DIFFUSION", False)
        self.ALL_MODELS_MMPROJ = env_bool("ALL_MODELS_MMPROJ", False)
        self.ALL_MODELS = True if (self.ALL_MODELS_NO_DIFFUSION or self.ALL_MODELS_MMPROJ) else env_bool("ALL_MODELS", False)

        self.USE_LLAMA = env_bool("USE_LLAMA", False)
        self.PREFER_ST = env_bool("PREFER_ST", False)
        # GGUF_ONLY drops safetensors (.st) models from the enumerated list.
        # llama.cpp cannot load .st, so any cross-backend test (e.g.
        # test_vision_equiv.sh) that compares bench vs llama-server must
        # restrict to GGUF decoders — otherwise the .st rows run on bench
        # but have no llama counterpart.
        self.GGUF_ONLY = env_bool("GGUF_ONLY", False)
        # mmproj auto-discovery is on by default for the test harness;
        # set DISABLE_MMPROJ=true to opt out. Off-by-default lives on
        # the bench server itself (--auto-mmproj); the harness inverts
        # that default because vision iteration is the common test
        # path right now.
        self.DISABLE_MMPROJ = env_bool("DISABLE_MMPROJ", False)

        self.TOP_LOGPROBS = env_str("TOP_LOGPROBS", "0")
        try:
            top_lp_int = int(self.TOP_LOGPROBS)
        except ValueError:
            top_lp_int = 0
        self.LOGPROBS = env_bool("LOGPROBS", top_lp_int > 0)

        self.BENCH_PORT = read_bench_port()
        self.BENCH_BASE_URL = f"http://localhost:{self.BENCH_PORT}"
        self.BENCH_API_BASE_URL = f"{self.BENCH_BASE_URL}/api/v1"
        self.BENCH_CTL_URL = f"{self.BENCH_BASE_URL}/ctl"

        self.LLAMA_PORT = env_str("LLAMA_PORT", "8080")
        self.LLAMA_BASE_URL = f"http://localhost:{self.LLAMA_PORT}"
        self.LLAMA_API_BASE_URL = f"{self.LLAMA_BASE_URL}/v1"

        self.STATELESS = env_tristate("STATELESS", None)
        self.THINK = env_tristate("THINK", None)
        self.ELIDE_THINK = env_tristate("ELIDE_THINK", None)
        self.FLASH = env_tristate("FLASH", None)
        self.MAX_TOKENS = env_str("MAX_TOKENS", "4096")
        self.TEMPERATURE = env_str("TEMPERATURE", "0")
        self.MODEL = env_str("MODEL", "default")
        self.DEBUG_POST = env_bool("DEBUG_POST", False)
        self.DEBUG_RESPONSE = env_bool("DEBUG_RESPONSE", False)
        self.DIFFUSION_STEPS = env_str("DIFFUSION_STEPS", "32")
        self.DIFFUSION_BLOCK_LENGTH = env_str("DIFFUSION_BLOCK_LENGTH", "64")
        self.DIFFUSION_TOKENS = env_str("DIFFUSION_TOKENS", "128")
        self.LLAMA_DIFFUSION_NGL = env_str("LLAMA_DIFUSE_NGL", "99")
        self.LLAMA_DIFFUSION_UB = env_str("LLAMA_DIFUSE_UB", "512")
        self.FORCE_DIFFUSION_CLI = env_bool("FORCE_DIFFUSION_CLI", False)
        self.USE_RLB_GEN = env_bool("USE_RLB_GEN", False)
        self.RLB_PREFILL = env_bool("RLB_PREFILL", False)
        self.RLB_ALPHA = env_str("RLB_ALPHA", "auto")
        self.RLB_HALT_RULE = env_str("RLB_HALT_RULE", "")
        self.RLB_TERMINAL_HALT_RULE = env_str("RLB_TERMINAL_HALT_RULE", "")
        self.RLB_MAG_NORM = env_bool("RLB_MAG_NORM", False)

        self.USE_PMM = env_bool("USE_PMM", False)
        # PMM_THINK_CAP / PMM_BLEND_ALPHA: empty = use server default; otherwise
        # a numeric string emitted into the payload as a number.
        self.PMM_THINK_CAP = env_str("PMM_THINK_CAP", "")
        self.PMM_THINK_DISABLED = env_bool("PMM_THINK_DISABLED", False)
        self.PMM_BLEND_ALPHA = env_str("PMM_BLEND_ALPHA", "")

        self.DIFFUSION_ARCH_NAMES = scan_diffusion_arch_names()

        force_list = env_str("FORCE_MODEL_LIST", "")
        # Bash splits on commas-or-whitespace via `(${VAR//,/ })`; mirror that
        # so callers passing either separator (incl. test_chat_equiv.sh, which
        # passes space-joined `${ARR[*]}`) get the expected list.
        self.FORCE_MODEL_LIST: list[str] = force_list.replace(",", " ").split()
        self.MODEL_LIST: list[str] = []
        self.LOG = os.environ.get("LOG")
        self.FORCE_NEW_SERVER = env_bool("FORCE_NEW_SERVER", False)

        self.API_BASE_URL = ""

        self.SERVER_PID: int | None = None
        self.LLAMA_PID: int | None = None


def usage(this_script: str) -> None:
    sys.stderr.write(f"Usage: {this_script} --loop|<message>\n")
    sys.exit(1)


def parse_models(resp_bytes: bytes) -> list[str]:
    try:
        data = json.loads(resp_bytes.decode("utf-8"))
        return [entry["id"] for entry in data["data"]]
    except Exception:
        return []


def _normalize_model_id(name: str) -> str:
    """Reduce a model id to its comparison key, matching how
    update_model_list normalizes: take the basename (llama may report a
    full path) and strip a trailing .gguf/.st. Bench reports the bare id
    already; llama --alias reports the decoder stem. Both land on the same
    key (e.g. 'gemma-4-E4B-it')."""
    base = Path(name).name
    for ext in (".gguf", ".st"):
        if base.endswith(ext):
            base = base[: -len(ext)]
    return base


def assert_serving_model(state: State) -> int:
    """Guard: confirm the server actually serving requests is loaded with
    state.MODEL before we send a chat request. Single-model llama-server
    ignores the payload `model` field and serves whatever was loaded, with
    no error on mismatch — so the harness must police identity itself and
    must NOT trust the backend. Returns 0 if the served model set contains
    state.MODEL, 1 (with an [ERR] line) otherwise."""
    resp = query_models(state)
    if resp is None:
        sys.stderr.write(
            f"[ERR] identity check: could not GET {state.API_BASE_URL}/models\n"
        )
        return 1
    served = parse_models(resp)
    want = _normalize_model_id(state.MODEL)
    served_keys = {_normalize_model_id(s) for s in served}
    if want in served_keys:
        return 0
    sys.stderr.write(
        f"[ERR] server is serving model {sorted(served_keys)} but test "
        f"intends {want}\n"
    )
    return 1


def query_models(state: State) -> bytes | None:
    url = f"{state.API_BASE_URL}/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            if resp.status >= 400:
                return None
            return resp.read()
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError):
        return None


def is_diffusion_model(state: State, model_name: str) -> bool:
    if state.FORCE_DIFFUSION_CLI:
        return True
    name_lower = model_name.lower()
    for arch in state.DIFFUSION_ARCH_NAMES:
        if arch in name_lower:
            return True
    return False


def update_model_list(state: State) -> int:
    use_api = not (state.LOGPROBS or state.USE_LLAMA)

    if use_api:
        resp = query_models(state)
        if resp is None:
            return 1
        names = parse_models(resp)
        models_text = "\n".join(names)
    else:
        found: list[str] = []
        if MODELS_DIR.is_dir():
            for entry in MODELS_DIR.iterdir():
                low = entry.name.lower()
                if not (low.endswith(".gguf") or low.endswith(".st")):
                    continue
                # mmproj-*.gguf are vision/audio tower sidecars paired with a
                # decoder GGUF (llama.cpp convention from --mmproj). They are
                # not standalone models — including them yields 404s on chat
                # completion. Mirrors the bench's server-side filter in
                # internal/model/manager.go's isMMProjGGUF.
                if low.startswith("mmproj-"):
                    continue
                found.append(entry.name)
        models_text = "\n".join(found)

    models_text = models_text.replace(".gguf", "")
    models_text = models_text.replace(".st", "")

    seen = set()
    uniq = []
    for line in sorted(models_text.split("\n")):
        if not line:
            continue
        if line in seen:
            continue
        seen.add(line)
        uniq.append(line)

    state.MODEL_LIST = list(uniq)
    if state.GGUF_ONLY:
        # Keep only models backed by a local .gguf decoder. A safetensors
        # model dir <id>.st enumerates as the bare id <id> (no -f16/-f32
        # suffix), which has no <id>.gguf; GGUF decoders keep their suffixed
        # id (<id>-f16.gguf exists). llama-server can't load .st, so the
        # cross-backend equiv test excludes them.
        state.MODEL_LIST = [
            m for m in state.MODEL_LIST if (MODELS_DIR / f"{m}.gguf").is_file()
        ]
    if state.ALL_MODELS_NO_DIFFUSION:
        state.MODEL_LIST = [m for m in state.MODEL_LIST if not is_diffusion_model(state, m)]
    if state.ALL_MODELS_MMPROJ:
        # Keep only models that load with multimodal capability. The
        # check is format-aware (mirrors Manager.scan's PREFER_ST-driven
        # dispatch): GGUF → matching mmproj sidecar; safetensors →
        # vision_config/audio_config in config.json. The .st check looks
        # at the model's own metadata rather than its arch.toml because
        # a given HF assembly may implement only a subset of the arch's
        # multimodal ceiling (e.g. only gemma-4-E2B has the audio tower
        # even though the gemma4 arch declares audio support).
        state.MODEL_LIST = [
            m for m in state.MODEL_LIST
            if _model_is_multimodal(m, state.PREFER_ST)
        ]

    if state.FORCE_MODEL_LIST:
        state.MODEL_LIST = list(state.FORCE_MODEL_LIST)

    if state.MODEL == "default" and state.MODEL_LIST:
        state.MODEL = state.MODEL_LIST[0]

    return 0


def show_models(state: State) -> None:
    print("models")
    print("======")
    for i, model in enumerate(state.MODEL_LIST, start=1):
        if model == state.MODEL:
            print(f"{i}) {model}*")
        else:
            print(f"{i}) {model}")
    print("")


def wait_for_starting_server(state: State) -> int:
    for _ in range(60):
        if query_models(state) is not None:
            update_model_list(state)
            return 0
        time.sleep(0.5)
    sys.stderr.write(f"failed getting models from {state.API_BASE_URL}/models\n")
    return 1


def start_api_server(state: State) -> int:
    if not (BENCH_BIN.exists() and os.access(BENCH_BIN, os.X_OK)):
        sys.stderr.write("./bin/bench binary missing. 'make all' first.\n")
        return 1
    state.API_BASE_URL = state.BENCH_API_BASE_URL
    log_path = state.LOG if state.LOG else "bin/test_inference.log"
    args = [str(BENCH_BIN), "serve-api", "--log", log_path, "--log-level", "NONE"]
    if state.PREFER_ST:
        args.append("--prefer-st")
    if not state.DISABLE_MMPROJ:
        args.append("--auto-mmproj")
    log_file = open(os.devnull, "wb")
    proc = subprocess.Popen(
        args,
        cwd=str(PROJECT_DIR),
        stdout=log_file,
        stderr=log_file,
        start_new_session=True,
    )
    state.SERVER_PID = proc.pid
    return wait_for_starting_server(state)


# Keep these two lists in lockstep with src/internal/model/manager.go's
# modelDescriptorTokens / quantFormatTokens — same auto-matching logic
# powers the bench side and the llama-server launch.
_MODEL_DESCRIPTOR_TOKENS = frozenset({"it", "instruct", "chat", "coder"})
_QUANT_FORMAT_TOKENS = ("MXFP4", "BF16", "F16", "F32", "GGUF")


def _is_descriptor_token(tok: str) -> bool:
    if not tok:
        return False
    if tok.lower() in _MODEL_DESCRIPTOR_TOKENS:
        return True
    # Q<digit> prefix (Q4_K_M, Q5_K_S, Q8_0, ...)
    if len(tok) >= 2 and tok[0] in ("Q", "q") and tok[1].isdigit():
        return True
    upper = tok.upper()
    for qf in _QUANT_FORMAT_TOKENS:
        if upper == qf or upper.startswith(qf + "_"):
            return True
    return False


def _strip_model_descriptors(name: str) -> str:
    """Truncate at the leftmost descriptor token. Mirrors Go's
    stripModelDescriptors. Tokens are hyphen-separated."""
    tokens = name.split("-")
    for i, tok in enumerate(tokens):
        if _is_descriptor_token(tok):
            if i == 0:
                return name  # never strip the whole name to empty
            return "-".join(tokens[:i])
    return name


def _mmproj_match_key(filename: str) -> str:
    """Strip mmproj decorations from a sidecar filename to derive its
    'what model is this for' key. Mirrors Go's mmprojMatchKey."""
    base = Path(filename).name
    if base.endswith(".gguf"):
        base = base[: -len(".gguf")]
    base = base.replace("mmproj-", "").replace("-mmproj", "").replace("mmproj", "")
    return base


def _find_mmproj_for_gguf(gguf_path: Path) -> Path | None:
    """Smart match: strip the decoder name, glob *mmproj*.gguf in the
    same directory, return the candidate whose stripped key starts with
    the stripped decoder name (with hyphen-boundary discipline).
    Mirrors Go's Manager.findMmprojForGGUF."""
    decoder_id = gguf_path.stem
    stripped = _strip_model_descriptors(decoder_id)
    if not stripped:
        return None
    candidates = sorted(gguf_path.parent.glob("*mmproj*.gguf"))
    # First pass: exact full-stem match (incl. precision/quant suffix like
    # -f16/-f32), so e.g. gemma-4-E4B-it-f32 binds the -f32 mmproj rather than
    # the sorted-first -f16 one (the stripped fallback truncates at "it").
    for c in candidates:
        if _mmproj_match_key(c.name) == decoder_id:
            return c
    for c in candidates:
        key = _mmproj_match_key(c.name)
        if not key.startswith(stripped):
            continue
        if len(key) > len(stripped) and key[len(stripped)] != "-":
            continue
        return c
    return None


# Multimodal capability marker keys in the safetensors model's config.json.
# This list must stay in lockstep with the stmap's derived_metadata
# `config_key_present` ops (see e.g. models/arch/gemma4.arch.stmap.toml).
# Mirrors handleConfigKeyPresent's emptiness rule in
# src/internal/inference/arch/model_reader_safetensors_derived.go: a value
# is "present" iff non-null and (when string / list / dict) non-empty.
_ST_MULTIMODAL_CONFIG_KEYS = ("vision_config", "audio_config", "video_config")


def _st_is_multimodal(st_dir: Path) -> bool:
    """Read the safetensors model's config.json and report whether any
    multimodal capability key (vision/audio/video) is declared present.
    Canonical signal — the Go safetensors loader's
    `vision.has_encoder` flag is derived from the same config.json
    presence check, so a particular HF assembly that lacks an audio
    tower (e.g. gemma-4-E4B has vision but no audio) is correctly
    classified by its own metadata rather than by the arch's
    capability ceiling."""
    try:
        cfg = json.loads((st_dir / "config.json").read_bytes())
    except (OSError, json.JSONDecodeError):
        return False
    for key in _ST_MULTIMODAL_CONFIG_KEYS:
        v = cfg.get(key)
        if v is None:
            continue
        if isinstance(v, (str, list, dict)) and not v:
            continue
        return True
    return False


def _model_is_multimodal(model_name: str, prefer_st: bool) -> bool:
    """Resolve whether `model_name` would load as multimodal under the
    current PREFER_ST setting, mirroring Manager.scan's format-priority
    rule. For the chosen format, applies the format-appropriate marker:
    .gguf → presence of a matching mmproj sidecar (via
    _find_mmproj_for_gguf); .st → vision_config/audio_config/video_config
    in config.json (via _st_is_multimodal)."""
    gguf = MODELS_DIR / f"{model_name}.gguf"
    st_dir = MODELS_DIR / f"{model_name}.st"
    has_gguf = gguf.is_file()
    has_st = st_dir.is_dir()
    if prefer_st:
        if has_st:
            return _st_is_multimodal(st_dir)
        if has_gguf:
            return _find_mmproj_for_gguf(gguf) is not None
    else:
        if has_gguf:
            return _find_mmproj_for_gguf(gguf) is not None
        if has_st:
            return _st_is_multimodal(st_dir)
    return False


def _resolve_llama_mmproj(state: State) -> tuple[Path | None, Path | None]:
    """Return (mmproj_path, model_path) for the active decoder if
    DISABLE_MMPROJ is not set and a matching sidecar exists; otherwise
    (None, None) so start_llama_server falls back to --models-dir.

    Decoder selection: state.MODEL if set + the corresponding .gguf
    exists, else auto-pick the first models/*.gguf (non-mmproj) with a
    smart-matching sidecar. Picking ANY paired decoder is fine for the
    --mmproj launch — the test request specifies the model by name.
    """
    if state.DISABLE_MMPROJ:
        return None, None
    models_dir = PROJECT_DIR / "models"
    candidate_names: list[str] = []
    if state.MODEL:
        candidate_names.append(state.MODEL)
    for p in sorted(models_dir.glob("*.gguf")):
        name = p.stem
        if name.startswith("mmproj-") or "mmproj" in name.lower():
            continue
        candidate_names.append(name)
    seen: set[str] = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        model_path = models_dir / f"{name}.gguf"
        if not model_path.exists():
            continue
        mmproj_path = _find_mmproj_for_gguf(model_path)
        if mmproj_path is not None:
            return mmproj_path, model_path
    return None, None


def start_llama_server(state: State) -> int:
    if shutil.which("llama-server") is None:
        sys.stderr.write("llama-server not installed.\n")
        return 1
    state.API_BASE_URL = state.LLAMA_API_BASE_URL
    log_path = state.LOG if state.LOG else "bin/test_inference_llama.log"
    # If the selected model has a paired mmproj-*.gguf sidecar, switch
    # from --models-dir (text-only, multi-model) to single-model mode
    # with --mmproj. llama-server only supports one --mmproj at a time,
    # so multi-model + multimodal is mutually exclusive at this layer.
    mmproj_path, model_path = _resolve_llama_mmproj(state)
    args = ["llama-server", "--port", str(state.LLAMA_PORT), "--ctx-size", "8192"]
    if mmproj_path is not None and model_path is not None:
        # --alias makes /v1/models and the response `model` field self-report
        # the *loaded* decoder, independent of the request payload's `model`
        # (which single-model llama-server silently ignores). The alias is the
        # loaded decoder's stem (NOT state.MODEL): if _resolve_llama_mmproj had
        # to fall back to a different paired decoder, we want the guard to SEE
        # that divergence, not paper over it. assert_serving_model compares the
        # self-reported stem against state.MODEL and fails loudly on mismatch.
        args += ["-m", str(model_path), "--mmproj", str(mmproj_path),
                 "--jinja", "-a", model_path.stem]
    else:
        args += ["--models-dir", str(PROJECT_DIR / "models")]
    log_file = open(log_path, "ab")
    proc = subprocess.Popen(
        args,
        cwd=str(PROJECT_DIR),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    state.LLAMA_PID = proc.pid
    return wait_for_starting_server(state)


def quit_api_server(state: State) -> None:
    try:
        with urllib.request.urlopen(f"{state.BENCH_CTL_URL}/?quit&now", timeout=2) as r:
            r.read()
        time.sleep(1)
    except Exception:
        try:
            ps = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=False)
            for line in ps.stdout.splitlines():
                if "awk" in line:
                    continue
                if "bin/bench" in line:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            os.kill(int(parts[1]), signal.SIGTERM)
                        except (ProcessLookupError, PermissionError, ValueError):
                            pass
        except Exception:
            pass
    state.SERVER_PID = None


def quit_llama_server(state: State) -> None:
    try:
        ps = subprocess.run(["ps", "aux"], capture_output=True, text=True, check=False)
        for line in ps.stdout.splitlines():
            if "awk" in line:
                continue
            if "llama-server" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        os.kill(int(parts[1]), signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, ValueError):
                        pass
    except Exception:
        pass
    state.LLAMA_PID = None


def quit_server(state: State) -> None:
    if state.SERVER_PID is not None:
        quit_api_server(state)
    if state.LLAMA_PID is not None:
        quit_llama_server(state)


def cycle_server(state: State) -> int:
    quit_server(state)
    if state.USE_LLAMA:
        return start_llama_server(state)
    return start_api_server(state)


def force_cycle_server(state: State) -> None:
    quit_api_server(state)
    quit_llama_server(state)
    if state.USE_LLAMA:
        rc = start_llama_server(state)
        print("new llama server started." if rc == 0 else "failed to start llama server.")
    else:
        rc = start_api_server(state)
        print("new api server started." if rc == 0 else "failed to start api server.")


def reassemble_stream(line_iter, debug_sink=None) -> dict:
    content_parts: list[str] = []
    finish_reason: str | None = None
    usage = None
    logprobs = None
    role = "assistant"
    for raw in line_iter:
        if isinstance(raw, bytes):
            try:
                raw_str = raw.decode("utf-8")
            except UnicodeDecodeError:
                raw_str = raw.decode("utf-8", errors="replace")
        else:
            raw_str = raw
        if debug_sink is not None:
            out = raw_str if raw_str.endswith("\n") else raw_str + "\n"
            debug_sink.write(out)
            debug_sink.flush()
        line = raw_str.rstrip("\r\n")
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            continue
        payload = line[len("data:") :].lstrip(" ")
        if payload == "[DONE]":
            break
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(obj.get("error"), dict):
            sys.stderr.write("error: mid-stream error event: " + payload + "\n")
            sys.exit(2)
        if obj.get("usage") is not None:
            usage = obj["usage"]
        choices = obj.get("choices") or []
        if not choices:
            continue
        ch = choices[0]
        delta = ch.get("delta") or {}
        if isinstance(delta.get("role"), str):
            role = delta["role"]
        c = delta.get("content")
        if isinstance(c, str):
            content_parts.append(c)
        if ch.get("finish_reason") is not None:
            finish_reason = ch["finish_reason"]
        if isinstance(ch.get("logprobs"), dict):
            logprobs = ch["logprobs"]
    out = {
        "choices": [
            {
                "index": 0,
                "message": {"role": role, "content": "".join(content_parts)},
                "finish_reason": finish_reason or "stop",
            }
        ],
    }
    if logprobs is not None:
        out["choices"][0]["logprobs"] = logprobs
    if usage is not None:
        out["usage"] = usage
    return out


def parse_response(state: State, resp_json: str, is_diffusion: bool) -> None:
    try:
        resp_all = json.loads(resp_json)
    except Exception:
        resp_all = {}
    printed = False
    if "choices" in resp_all:
        ch0 = resp_all["choices"][0]
        resp_text = ch0["message"]["content"]
        resp_text = resp_text if resp_text.strip() else "<NO-MODEL-RESPONSE>"
        usage = resp_all.get("usage", {}) or {}
        itps = usage.get("prompt_tokens_per_sec", 0.0)
        otps = usage.get("completion_tokens_per_sec", 0.0)
        ttps = usage.get("total_tokens_per_sec", 0.0)
        secs = usage.get("total_seconds", 0.0)
        ctok = usage.get("completion_tokens", 0)
        ttok = usage.get("thinking_tokens", 0)
        stats = f"itps:{itps:.2f}, otps:{otps:.2f}, ttps:{ttps:.2f}, s:{secs:.2f}"
        if ctok > 0:
            stats += f", otok:{ctok}, think:{ttok}"
        print(f"{resp_text} {{{stats}}}")
        printed = True
        choices_list = resp_all.get("choices", [])
        first_choice = choices_list[0] if choices_list else {}
        lp_block = first_choice.get("logprobs", {}) if isinstance(first_choice, dict) else {}
        if lp_block:
            model_type = "diffusion" if is_diffusion else "autoregression"
            try:
                top_lps = lp_block["content"][0]["top_logprobs"]
                lp_json = json.dumps(top_lps, sort_keys=True)
                print(f"{state.MODEL}|{model_type}|{lp_json}|{resp_text}")
            except (KeyError, IndexError, TypeError):
                pass

    if (not printed) or state.DEBUG_RESPONSE:
        if resp_all:
            print(json.dumps(resp_all, indent=2))
        elif resp_json.strip():
            print(resp_json)
        else:
            print("<NO-API-RESPONSE>")


def filter_diffusion_cli_output(line_iter, debug: bool):
    if debug:
        for line in line_iter:
            yield line
        return
    block = False
    printing = False
    for raw in line_iter:
        line = raw.rstrip("\r\n") if isinstance(raw, str) else raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not block and line.startswith("load_backend: loaded"):
            block = True
        if block:
            if line.startswith("total time: "):
                block = False
            continue
        if line.startswith("~") or line.startswith("ggml_metal_free:"):
            continue
        if line.strip() == "":
            if not printing:
                continue
        yield line + "\n"
        printing = True


def query_one_diffusion_cli(state: State, msg: str) -> int:
    model_path = MODELS_DIR / f"{state.MODEL}.gguf"
    if not model_path.is_file():
        model_path = MODELS_DIR / f"{state.MODEL}.st"
    if not (model_path.is_file() or model_path.is_dir()):
        sys.stderr.write(f"[ERR] model {state.MODEL} not found as .gguf or .st\n")
        return 1

    cli_args = [
        "llama-diffusion-cli",
        "-m",
        str(model_path),
        "-p",
        msg,
        "-n",
        state.DIFFUSION_TOKENS,
        "-ngl",
        state.LLAMA_DIFFUSION_NGL,
        "-ub",
        state.LLAMA_DIFFUSION_UB,
        "-c",
        state.DIFFUSION_TOKENS,
        "--diffusion-steps",
        state.DIFFUSION_STEPS,
        "--diffusion-block-length",
        state.DIFFUSION_BLOCK_LENGTH,
        "--temp",
        state.TEMPERATURE,
        "-fa",
        "on" if state.FLASH else "off",
    ]

    if state.DEBUG_POST:
        print(f"[DBG] llama-diffusion-cli {' '.join(cli_args[1:])}")

    debug = state.DEBUG_RESPONSE

    def run_and_filter() -> str:
        proc = subprocess.Popen(
            cli_args,
            cwd=str(PROJECT_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert proc.stdout is not None
        captured = []
        for line in filter_diffusion_cli_output(proc.stdout, debug):
            sys.stdout.write(line)
            sys.stdout.flush()
            captured.append(line)
        proc.wait()
        return "".join(captured)

    run_and_filter()
    if state.LOGPROBS:
        out = run_and_filter()
        print(out, end="" if out.endswith("\n") else "\n")
        print(f"{state.MODEL}|diffusion|<logprob not supported>|{out}")
    else:
        run_and_filter()
    return 0


def _num(s: str, default: int | float = 0) -> int | float:
    """Parse a numeric string preserving int-vs-float lexical type.

    `json.loads` is the cheapest way to do this: '4096' → int 4096, '0.5' →
    float 0.5. Garbage falls back to `default`.
    """
    try:
        v = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return default
    return v if isinstance(v, (int, float)) else default


def build_payload(state: State, msg: str, is_diffusion: bool) -> tuple[str, int | float]:
    max_tokens: int | float = _num(state.MAX_TOKENS, 4096)

    # Resolve @FILENAME refs into typed content parts (image_url for
    # images, inline text for text files). Plain-text prompts pass
    # through unchanged. \@ in the prompt is consumed as a literal @.
    content = parse_prompt_content(msg)

    payload: dict[str, object] = {
        "messages": [{"role": "user", "content": content}],
    }

    if state.THINK is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": state.THINK}

    if not state.USE_LLAMA:
        bench_custom: dict[str, object] = {}

        if state.STATELESS is not None:
            bench_custom["stateless"] = state.STATELESS
        if state.ELIDE_THINK is not None:
            bench_custom["elide_thinking"] = state.ELIDE_THINK
        if state.FLASH is not None:
            bench_custom["flash_attention"] = state.FLASH

        if is_diffusion:
            bench_custom["diffusion"] = {
                "steps": _num(state.DIFFUSION_STEPS, 32),
                "block_length": _num(state.DIFFUSION_BLOCK_LENGTH, 64),
            }
            max_tokens = _num(state.DIFFUSION_TOKENS, 128)

        if state.USE_RLB_GEN:
            bench_custom["use_rlb_gen"] = True
            # Bash always emitted enable_rlb_on_prefill when USE_RLB_GEN=true
            # (its `[[ -n ${RLB_PREFILL} ]]` is always true since the default
            # 'false' is non-empty). Preserve that here.
            bench_custom["enable_rlb_on_prefill"] = state.RLB_PREFILL
            # -1.0 is the server-side sentinel for "auto" (per generate_rlb).
            rlb_alpha = "-1.0" if state.RLB_ALPHA == "auto" else state.RLB_ALPHA
            if rlb_alpha:
                bench_custom["rlb_alpha"] = _num(rlb_alpha, 0)
            if state.RLB_HALT_RULE:
                bench_custom["rlb_halt_rule"] = state.RLB_HALT_RULE
            if state.RLB_TERMINAL_HALT_RULE:
                bench_custom["rlb_terminal_halt_rule"] = state.RLB_TERMINAL_HALT_RULE
            if state.RLB_MAG_NORM:
                bench_custom["rlb_magnitude_norm"] = True

        if state.USE_PMM:
            bench_custom["use_pmm"] = True
            if state.PMM_THINK_CAP:
                bench_custom["pmm_think_cap"] = _num(state.PMM_THINK_CAP, 0)
            if state.PMM_THINK_DISABLED:
                bench_custom["pmm_think_disabled"] = True
            if state.PMM_BLEND_ALPHA:
                bench_custom["pmm_blend_alpha"] = _num(state.PMM_BLEND_ALPHA, 1.0)

        if bench_custom:
            payload["bench_custom"] = bench_custom

    if state.LOGPROBS:
        payload["logprobs"] = True
        payload["top_logprobs"] = _num(state.TOP_LOGPROBS, 0)

    payload["model"] = state.MODEL
    payload["max_tokens"] = max_tokens
    payload["temperature"] = _num(state.TEMPERATURE, 0)
    payload["stream"] = True
    payload["stream_options"] = {"include_usage": True}

    return json.dumps(payload, indent=2), max_tokens


def query_one(state: State, msg: str) -> int:
    is_diffusion = is_diffusion_model(state, state.MODEL)

    if state.USE_LLAMA and is_diffusion:
        # llama-diffusion-cli loads state.MODEL directly by path (no server),
        # so there is no served-vs-intended divergence to guard against.
        return query_one_diffusion_cli(state, msg)

    # Identity guard. Before sending the chat request, confirm the server is
    # serving state.MODEL. Single-model llama-server (the -m/--mmproj vision
    # path) ignores the payload `model` and serves whatever was loaded; if a
    # cycle/off-by-one left the wrong model resident, this catches it. On
    # mismatch we print a NON-sentinel response token (forces a semantic FAIL
    # in the equiv parsers, not a '—' skip) and emit NO logprob fingerprint
    # line, so a contaminated logprob is never recorded. Return non-zero.
    if assert_serving_model(state) != 0:
        print("[IDENTITY-MISMATCH] {itps:0.00, otps:0.00, ttps:0.00, s:0.00}")
        return 1

    payload, _ = build_payload(state, msg, is_diffusion)

    completions_url = f"{state.API_BASE_URL}/chat/completions"
    if state.DEBUG_POST:
        print(f"[DBG] {completions_url} POST payload: {payload}")

    debug_resp = state.DEBUG_RESPONSE
    debug_sink = None
    if debug_resp:
        sys.stderr.write("[DBG] SSE chunks:\n")
        sys.stderr.flush()
        debug_sink = sys.stderr

    req = urllib.request.Request(
        completions_url,
        data=payload.encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Connection": "close",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            line_iter = (line.decode("utf-8", errors="replace") for line in resp)
            reassembled = reassemble_stream(line_iter, debug_sink=debug_sink)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        sys.stderr.write(f"HTTP {e.code} from {completions_url}: {body}\n")
        return 1
    except (urllib.error.URLError, OSError) as e:
        sys.stderr.write(f"request failed: {e}\n")
        return 1

    resp_str = json.dumps(reassembled)

    if debug_resp:
        sys.stderr.write("[DBG] reassembled JSON:\n")
        sys.stderr.write(resp_str + "\n")
        sys.stderr.flush()

    parse_response(state, resp_str, is_diffusion)
    return 0


def query(state: State, msg: str) -> int:
    # Pre-validate @FILENAME refs up-front so a typo in loop mode doesn't
    # kill the REPL — and so we don't start the request just to abort.
    try:
        parse_prompt_content(msg)
    except PromptParseError as e:
        sys.stderr.write(f"prompt parse error: {e}\n")
        return 2
    if state.ALL_MODELS:
        # The server must be built for state.MODEL *before* the query: the
        # llama-server mmproj path is single-model (-m/--mmproj resolved from
        # state.MODEL at launch), so the model name in the request payload is
        # ignored. Cycling after the query instead of before created an
        # off-by-one — each model's request was served by the *previous*
        # model's server (e.g. a Gemma prompt answered by a resident Qwen),
        # silently corrupting per-model vision logprobs. The initial server
        # (started in main for MODEL_LIST[0]) is already correct for the first
        # iteration, so only re-cycle on an actual model change.
        prev_model = state.MODEL
        first = True
        for model in list(state.MODEL_LIST):
            print(f'{model} <-- "{msg}"')
            state.MODEL = model
            if state.FORCE_NEW_SERVER and not (first and model == prev_model):
                cycle_server(state)
            first = False
            query_one(state, msg)
        return 0
    return query_one(state, msg)


def loop_help(state: State) -> None:
    text = textwrap.dedent(f"""\
        Acontextual Loop Mode
        =====================
        /all-models: toggle all-models mode (currently: {jbool(state.ALL_MODELS)})
        /cls: clear the screen
        /debug-post: toggle display of post json (currently: {jbool(state.DEBUG_POST)})
        /debug-response: toggle display of response json (currently: {jbool(state.DEBUG_RESPONSE)})
        /diffusion-block-length [length]: show or set DIFFUSION_BLOCK_LENGTH (currently: {state.DIFFUSION_BLOCK_LENGTH})
        /diffusion-steps [steps]: show or set DIFFUSION_STEPS (currently: {state.DIFFUSION_STEPS})
        /diffusion-tokens [count]: show or set DIFFUSION_TOKENS (currently: {state.DIFFUSION_TOKENS})
        /elide-think: toggle elision of thinking output (currently: {jbool(state.ELIDE_THINK)})
        /flash: toggle flash attention mode (currently: {jbool(state.FLASH)})
        /help: show this help message
        /llama: toggle between using llama-server and bench serve-api. (currently: USE_LLAMA={jbool(state.USE_LLAMA)})
        /max-tokens [token_count]: show or set MAX_TOKENS (currently: {state.MAX_TOKENS})
        /model [index]: show or set current model (currently: {state.MODEL})
        /new-server: shuts down old server and starts a new one.
        /prefer-st: toggle prefer safetensors mode (currently: {jbool(state.PREFER_ST)})
        /rlb-alpha <0-1>|auto: set RLB SSM state alpha blending factor (currently: {state.RLB_ALPHA})
        /rlb-gen: toggle RLB generation mode (currently: {jbool(state.USE_RLB_GEN)})
        /rlb-halt-rule [name]: set RLB halt rule (currently: {state.RLB_HALT_RULE})
        /rlb-mag-norm: toggle RLB magnitude normalization (currently: {jbool(state.RLB_MAG_NORM)})
        /rlb-prefill: toggle RLB during prefill (currently: {jbool(state.RLB_PREFILL)})
        /rlb-terminal-halt-rule [name|-]: set RLB halt rule for terminal block only; "-" to clear (currently: {state.RLB_TERMINAL_HALT_RULE or '<reuse halt-rule>'})
        /pmm: toggle PMM two-stream decode mode (currently: {jbool(state.USE_PMM)})
        /pmm-think-cap [n]: set PMM think-phase slab cap; empty to clear (currently: {state.PMM_THINK_CAP or '<default>'})
        /pmm-think-disabled: toggle skipping Stream B during the think phase (currently: {jbool(state.PMM_THINK_DISABLED)})
        /pmm-blend-alpha [α|-]: set shadow blend weight 0.0-1.0; "-" to clear/reset to 1.0 (currently: {state.PMM_BLEND_ALPHA or '<default=1.0>'})
        /stateless: toggle stateless mode (currently: {jbool(state.STATELESS)})
        /temperature [temperature]: show or set TEMPERATURE (currently: {state.TEMPERATURE})
        /think: toggle think mode (currently: {jbool(state.THINK)})
        /quit: exit loop mode
        """)
    print(text)
    if state.USE_LLAMA:
        names = " ".join(state.DIFFUSION_ARCH_NAMES)
        print(textwrap.dedent(f"""\
            Note: with USE_LLAMA=true, diffusion models ({names}) invoke llama-diffusion-cli
            fresh for each prompt — no persistent server, so each query loads the model from scratch.
            """))


def loop_split(line: str) -> tuple[str, list[str]]:
    parts = line.split()
    if not parts:
        return "", []
    return parts[0], parts[1:]


def cmd_set_positive_int(line: str, current: str) -> str:
    _, args = loop_split(line)
    arg = args[0] if args else ""
    try:
        n = int(arg) if arg else 0
    except ValueError:
        n = 0
    if n > 0:
        return str(n)
    return current


def cmd_set_nonneg_float(line: str, current: str) -> str:
    _, args = loop_split(line)
    arg = args[0] if args else ""
    try:
        f = float(arg) if arg else 0.0
    except ValueError:
        return current
    if f >= 0:
        return arg
    return current


def loop_mode(state: State) -> None:
    try:
        import readline  # noqa: F401
    except ImportError:
        pass

    loop_help(state)
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print("")
            break
        line = line.rstrip(" ")

        if line == "/all-models":
            state.ALL_MODELS = toggle(state.ALL_MODELS)
            print(f"ALL_MODELS={jbool(state.ALL_MODELS)}")
            continue
        if line == "/cls":
            os.system("clear")
            continue
        if line == "/new-server":
            force_cycle_server(state)
            continue
        if line == "/llama":
            state.USE_LLAMA = toggle(state.USE_LLAMA)
            print(f"USE_LLAMA={jbool(state.USE_LLAMA)}")
            force_cycle_server(state)
            continue
        if line in ("/help", "/h", "/", "/?"):
            loop_help(state)
            continue
        if line.startswith("/model") or line == "/m":
            _, args = loop_split(line)
            mi_str = args[0] if args else ""
            try:
                mi = int(mi_str) if mi_str else 0
            except ValueError:
                mi = 0
            if mi > 0 and 1 <= mi <= len(state.MODEL_LIST):
                state.MODEL = state.MODEL_LIST[mi - 1]
            else:
                update_model_list(state)
                show_models(state)
            print(f"MODEL={state.MODEL}")
            continue
        if line == "/prefer-st":
            state.PREFER_ST = toggle(state.PREFER_ST)
            print(f"PREFER_ST={jbool(state.PREFER_ST)}")
            force_cycle_server(state)
            continue
        if line == "/flash":
            state.FLASH = rotate(state.FLASH)
            print(f"FLASH={jbool(state.FLASH)}")
            continue
        if line == "/stateless":
            state.STATELESS = rotate(state.STATELESS)
            print(f"STATELESS={jbool(state.STATELESS)}")
            continue
        if line == "/debug-post":
            state.DEBUG_POST = toggle(state.DEBUG_POST)
            print(f"DEBUG_POST={jbool(state.DEBUG_POST)}")
            continue
        if line == "/debug-response":
            state.DEBUG_RESPONSE = toggle(state.DEBUG_RESPONSE)
            print(f"DEBUG_RESPONSE={jbool(state.DEBUG_RESPONSE)}")
            continue
        if line == "/think":
            state.THINK = rotate(state.THINK)
            print(f"THINK={jbool(state.THINK)}")
            continue
        if line == "/elide-think":
            state.ELIDE_THINK = rotate(state.ELIDE_THINK)
            print(f"ELIDE_THINK={jbool(state.ELIDE_THINK)}")
            continue
        if line.startswith("/diffusion-steps"):
            state.DIFFUSION_STEPS = cmd_set_positive_int(line, state.DIFFUSION_STEPS)
            print(f"DIFFUSION_STEPS={state.DIFFUSION_STEPS}")
            continue
        if line.startswith("/diffusion-block-length"):
            state.DIFFUSION_BLOCK_LENGTH = cmd_set_positive_int(line, state.DIFFUSION_BLOCK_LENGTH)
            print(f"DIFFUSION_BLOCK_LENGTH={state.DIFFUSION_BLOCK_LENGTH}")
            continue
        if line.startswith("/diffusion-tokens"):
            state.DIFFUSION_TOKENS = cmd_set_positive_int(line, state.DIFFUSION_TOKENS)
            print(f"DIFFUSION_TOKENS={state.DIFFUSION_TOKENS}")
            continue
        if line.startswith("/max-tokens"):
            state.MAX_TOKENS = cmd_set_positive_int(line, state.MAX_TOKENS)
            print(f"MAX_TOKENS={state.MAX_TOKENS}")
            continue
        if line.startswith("/temperature"):
            state.TEMPERATURE = cmd_set_nonneg_float(line, state.TEMPERATURE)
            print(f"TEMPERATURE={state.TEMPERATURE}")
            continue
        if line == "/rlb-gen":
            state.USE_RLB_GEN = toggle(state.USE_RLB_GEN)
            print(f"USE_RLB_GEN={jbool(state.USE_RLB_GEN)}")
            continue
        if line == "/rlb-prefill":
            state.RLB_PREFILL = toggle(state.RLB_PREFILL)
            print(f"RLB_PREFILL={jbool(state.RLB_PREFILL)}")
            continue
        if line.startswith("/rlb-alpha"):
            _, args = loop_split(line)
            ra = args[0] if args else ""
            if ra == "auto":
                state.RLB_ALPHA = ra
            else:
                try:
                    f = float(ra) if ra else 1.0
                    if 0.0 <= f <= 1.0:
                        state.RLB_ALPHA = ra
                except ValueError:
                    pass
            print(f"RLB_ALPHA={state.RLB_ALPHA}")
            continue
        if line.startswith("/rlb-terminal-halt-rule"):
            _, args = loop_split(line)
            thr = args[0] if args else ""
            if thr == "":
                pass
            elif thr == "-":
                state.RLB_TERMINAL_HALT_RULE = ""
            else:
                state.RLB_TERMINAL_HALT_RULE = thr
            disp = state.RLB_TERMINAL_HALT_RULE or "<reuse halt-rule>"
            print(f"RLB_TERMINAL_HALT_RULE={disp}")
            continue
        if line.startswith("/rlb-halt-rule"):
            _, args = loop_split(line)
            hr = args[0] if args else ""
            if hr:
                state.RLB_HALT_RULE = hr
            print(f"RLB_HALT_RULE={state.RLB_HALT_RULE}")
            continue
        if line == "/rlb-mag-norm":
            state.RLB_MAG_NORM = toggle(state.RLB_MAG_NORM)
            print(f"RLB_MAG_NORM={jbool(state.RLB_MAG_NORM)}")
            continue
        if line == "/pmm-think-disabled":
            state.PMM_THINK_DISABLED = toggle(state.PMM_THINK_DISABLED)
            print(f"PMM_THINK_DISABLED={jbool(state.PMM_THINK_DISABLED)}")
            continue
        if line.startswith("/pmm-blend-alpha"):
            _, args = loop_split(line)
            pba = args[0] if args else ""
            if pba == "":
                pass  # no arg: just print current
            elif pba == "-":
                state.PMM_BLEND_ALPHA = ""  # clear → server default 1.0
            else:
                state.PMM_BLEND_ALPHA = pba
            disp = state.PMM_BLEND_ALPHA or "<default=1.0>"
            print(f"PMM_BLEND_ALPHA={disp}")
            continue
        if line.startswith("/pmm-think-cap"):
            _, args = loop_split(line)
            ptc = args[0] if args else ""
            if ptc == "":
                pass
            elif ptc == "-":
                state.PMM_THINK_CAP = ""
            else:
                state.PMM_THINK_CAP = ptc
            disp = state.PMM_THINK_CAP or "<default>"
            print(f"PMM_THINK_CAP={disp}")
            continue
        if line == "/pmm":
            state.USE_PMM = toggle(state.USE_PMM)
            print(f"USE_PMM={jbool(state.USE_PMM)}")
            continue
        if line in ("quit", "exit", "/quit", "/exit"):
            break
        if line.startswith("/"):
            sys.stderr.write(f"unknown / command {line}\n")
            loop_help(state)
            continue
        if line == "":
            continue

        query(state, line)


def main() -> int:
    this_script = Path(sys.argv[0]).name or "test_inference.py"
    args = sys.argv[1:]

    os.chdir(PROJECT_DIR)

    state = State()
    msg = ""

    if args and args[0] == "--loop":
        state.LOOP_MODE = True
        args = args[1:]
    elif args and args[0].startswith("-"):
        sys.stderr.write(f"unknown option: {args[0]}\n")
        usage(this_script)
    elif not args:
        usage(this_script)
    else:
        if args[0] == "--all-models":
            state.ALL_MODELS = True
            args = args[1:]
        msg = " ".join(args)

    # Cleanup policy: if THIS invocation started a server (FORCE_NEW_SERVER
    # cycle, or loop-mode session that brought one up), it owns the lifetime
    # and must tear it down at exit — whether normal exit, SIGINT, or
    # SIGTERM. quit_server() gates on SERVER_PID/LLAMA_PID being set, both
    # of which are populated only when start_*_server() ran in *this*
    # process, so reused-server invocations correctly leave the existing
    # server alone.
    def _sig_handler(signum, frame):
        quit_server(state)
        sys.exit(128 + signum)

    try:
        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)
    except (ValueError, OSError):
        pass

    atexit.register(quit_server, state)

    if state.FORCE_NEW_SERVER:
        if state.USE_LLAMA:
            quit_llama_server(state)
            start_llama_server(state)
        else:
            quit_api_server(state)
            start_api_server(state)
    else:
        if state.USE_LLAMA:
            state.API_BASE_URL = state.LLAMA_API_BASE_URL
        else:
            state.API_BASE_URL = state.BENCH_API_BASE_URL
        update_model_list(state)

    if state.LOOP_MODE:
        loop_mode(state)
    else:
        rc = query(state, msg)
        return rc
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:
        sys.exit(0)
