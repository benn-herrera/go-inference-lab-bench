#!/usr/bin/env bash
# Vision equivalence gate. Tabulated + exit-coded (0 = all pass, 1 = any FAIL).
# Usage: test_vision_equiv.sh [llama|gguf-st]   (default: llama)
#   llama   Cross-backend: bench (GGUF) vs llama-server (GGUF); GGUF_ONLY set automatically.
#   gguf-st Cross-format, bench only: bench .st vision (CHECK) vs bench GGUF vision (REF).
#
# Env knobs:
#   MODEL              restrict to one model (default: all mmproj-capable / all GGUF+.st pairs)
#   IMAGES             image paths, comma/space-separated (default: vision_test.png, resize-exercising;
#                        vision_test_960x624.png is the no-resize variant for isolating resize bugs)
#   SKIP_BENCH         skip bench side (llama mode only)
#   SKIP_LLAMA         skip llama-server side (llama mode only)
#   LOG_DIR            log directory (default: bin/test_vision_equiv_logs)
#   VISION_PASS_THRESH max |check_lp - ref_lp| before logprob FAIL (default: 0.0075; see CONVENTIONS.md)
#
# See CONVENTIONS.md (section "test_vision_equiv.sh") for prompt requirements, the
# threshold/F16-floor rationale, and the prompt-framing findings.

set -ou pipefail

THIS_SCRIPT=$(basename "${0}")
THIS_DIR=$(cd "$(dirname "${0}")" && pwd)
cd "${THIS_DIR}"

EQUIV="${1:-llama}"
EQUIV_FN="${EQUIV//-/_}"

# use vvv to eliminate image scaling from equation if you are hunting a llama/bench difference
# IMAGES_RAW="${IMAGES:-test_data/vision_test_960x624.png}"
IMAGES_RAW="${IMAGES:-test_data/vision_test.png}"
LOG_DIR="${LOG_DIR:-bin/test_vision_equiv_logs}"

IFS=', ' read -r -a IMAGE_LIST <<< "${IMAGES_RAW}"
for img in "${IMAGE_LIST[@]}"; do
  [[ -f "${img}" ]] || { echo "error: test image not found: ${img}" 1>&2; exit 1; }
done

# Fixed dir, no per-run subdir — matches sibling scripts (test_chat_equiv.sh
# writes ./bin/test_<mode>_equiv.log). Overwriting prior run logs is fine.
run_dir="${LOG_DIR}"
mkdir -p "${run_dir}"

# Single-token-answer prompts only (two cats: blue-striped tie right, green-striped tie left,
# color photograph). Free-form prose and "which side is darker" are excluded — see CONVENTIONS.md.
#
# IMAGE-FIRST (required): image marker leads every prompt so both engines process the image
# through KV cache before question text. Mid-sentence image gives bench and llama different
# causal layouts — a real layout difference, not FP. See CONVENTIONS.md for the full story.
#
# FORMAT-UNAMBIGUOUS (required): avoid phrasings that split mass over formatting alternatives
# (e.g. "left or right" forces a lowercase competitor against natural "Right" → 0.0186 delta,
# vs 0.0019 without). Effect scales with model depth; deeper models amplify it hard (gemma-31B
# color 0.20 → 0.00000 once pinned). See CONVENTIONS.md for the three-instance record.
# UPSHOT: the prompts *are* tuned a bit to reduce this but it is less cheating
# and more acknowledging the pragmatics of keeping two *very*
# complex sets of mathematical operations in sync.
PROMPT_TEMPLATES=(
  "@%s Identify the subject. Answer in one word."
  "@%s Which side of the image is the cat wearing the blue tie on? Answer in one word."
  #"@%s Which side of the image is the cat wearing the green tie on? Answer in one word."
  "@%s Describe the image colorspace. Answer in one word with exactly one of 'color' or 'grayscale'."
)
PROMPT_LABELS=(
  "subject"
  "blue-tie-side"
  #"green-tie-side"
  "color-or-gray"
)

# Force mmproj on regardless of caller env — vision is the script's whole point.
export DISABLE_MMPROJ="false"
# Deterministic output across the matrix.
export TEMPERATURE="${TEMPERATURE:-0}"
# Bound response length; prompts already request short answers.
export MAX_TOKENS="${MAX_TOKENS:-16}"
# Gemma-4 and other reasoning-by-default models will otherwise burn the
# entire MAX_TOKENS budget in <think> and leave visible content empty
# (→ <NO-MODEL-RESPONSE>). Disable thinking unless the caller overrode.
export THINK="${THINK:-false}"
# Emit the per-model "<model>|autoregression|[{...logprob...}]|<answer>"
# fingerprint line (answer-token top-logprob), same mechanism as
# test_chat_equiv.sh, so we can compare check vs ref logprob per row in
# addition to the semantic word check.
export TOP_LOGPROBS="${TOP_LOGPROBS:-1}"

# Threshold for the answer-token logprob comparison. Looser than text-equiv's
# 0.001 because the vision path is longer and there's just more room
# for numbers to drift. vision is simply more sensitive to small differences
# in code that are theoretically equivalent, but result in tiny deltas in practice.
#
# 0.0075 (not 0.004): the Qwen3.5 F16 vision floor is ~0.005-0.006 on contested
# tail tokens (its 27-layer encoder accumulates more sub-LSB residual than
# gemma's), AND that floor is sensitive to the *reference binary's build* — the
# brew llama-server reference is unpinned, so a llama.cpp rebuild shifts tail
# logprobs ~0.001 even when no vision/Metal op changed (measured: brew 9410->9430
# moved Qwen subject/color from <0.004 to ~0.0057/0.0043 with bench byte-identical
# and the reference's Qwen vision code unchanged). 0.004 sat below that floor and
# was not robustly reproducible across reference rebuilds. See CONVENTIONS.md.
VISION_PASS_THRESH="${VISION_PASS_THRESH:-0.0075}"

# Server log file. test_inference.py launches `bench serve-api --log "${LOG}"`,
# and the log package writes ALL levels to the file regardless of stderr level
# (--log-level NONE only silences stderr). So the per-load
# "ModelReader[gguf|safetensors] created" lines land here — this is what the
# gguf-st loader assertions grep, exactly as test_chat_equiv.sh does. Every
# run_backend invocation appends to this one accumulating file.
export LOG="${run_dir}/test_${EQUIV}_equiv.server.log"
rm -f "${LOG}" 2>/dev/null

PYTHON=$(command -v python3 2>/dev/null) || \
  PYTHON=$(command -v python 2>/dev/null) || \
  { echo "neither python3 nor python found." 1>&2; exit 1; }

# =============================================================================
# Matrix machinery (shared by both modes)
# =============================================================================

function slug() {
  echo "${1}" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_'
}

# Parse a captured test_inference.sh log into one TSV row per model:
#   <model>\t<response>\t<answer_token_logprob>
# In single-model mode the harness emits only a "<resp> {itps:...}" line;
# in all-models mode each run is preceded by a `<model> <-- "msg"` marker.
#
# The answer-token logprob comes from the fingerprint line emitted by
# test_inference.py when TOP_LOGPROBS>0 (same shape as test_chat_equiv.sh):
#   <model>|autoregression|[{"logprob": X, "token": "...", ...}, ...]|<answer>
# We take element [0]["logprob"] (the top logprob of the first answer token).
# The join is done here in Python (the model names contain dots — e.g.
# "Qwen3.5-9B" — which break bash associative-array keys and arithmetic, and
# macOS's default bash 3.2 has no associative arrays at all).
function extract_responses() {
  "${PYTHON}" - "${1}" "${2:-default}" <<'PY'
import re, sys, json
path, default_model = sys.argv[1], sys.argv[2]
text = open(path).read()

def trim(s: str) -> str:
    s = s.strip().replace("\t", " ")
    return s if s else "<EMPTY>"

def resp_of(chunk: str) -> str:
    j = chunk.rfind(" {itps:")
    if j < 0:
        return "<NO-RESPONSE>"
    ls = chunk.rfind("\n", 0, j) + 1
    return trim(chunk[ls:j])

def lp_of(chunk: str) -> str:
    """First answer-token top-logprob from the fingerprint line in this
    chunk, or "" if none present."""
    for line in chunk.splitlines():
        parts = line.split("|")
        if len(parts) < 4 or parts[1] not in ("autoregression", "diffusion"):
            continue
        try:
            return repr(json.loads(parts[2])[0]["logprob"])
        except (ValueError, KeyError, IndexError, TypeError):
            continue
    return ""

marker = re.compile(r'^(\S+) <-- "', re.MULTILINE)
matches = list(marker.finditer(text))
if not matches:
    print(f"{default_model}\t{resp_of(text)}\t{lp_of(text)}")
else:
    for idx, m in enumerate(matches):
        model = m.group(1)
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chunk = text[start:end]
        print(f"{model}\t{resp_of(chunk)}\t{lp_of(chunk)}")
PY
}

# results.tsv rows: backend \t model \t image \t prompt \t response \t logprob
# (logprob is the answer-token top-logprob, or empty if the backend emitted no
# fingerprint line for that model — the EQUIV block treats empty as a skip.)
RESULTS_TSV="${run_dir}/results.tsv"
: > "${RESULTS_TSV}"

# run_backend <backend_label> <use_llama> <prefer_st> <model_override> <result_key>
#   model_override: a specific MODEL id to serve this run, or "" to inherit the
#     ambient MODEL/ALL_MODELS_MMPROJ enumeration (the llama-mode default path).
#   result_key: the model name recorded in results.tsv (and passed to
#     extract_responses as its single-model fallback). In gguf-st mode this is
#     the LOGICAL key so the GGUF and ST runs pair on one (model,image,prompt).
function run_backend() {
  local backend="${1}" use_llama="${2}" prefer_st="${3}" model_override="${4}" result_key="${5}"
  printf '\n========== backend=%s ==========\n' "${backend}"
  for img in "${IMAGE_LIST[@]}"; do
    local img_slug; img_slug=$(slug "${img}")
    printf '\n---------- image=%s ----------\n' "${img}"
    for i in "${!PROMPT_TEMPLATES[@]}"; do
      local label="${PROMPT_LABELS[${i}]}"
      local prompt
      printf -v prompt "${PROMPT_TEMPLATES[${i}]}" "\"${img}\""
      local out="${run_dir}/${backend}__${img_slug}__${label}.log"
      printf '\n----- [%s | %s | %s] -----\n' "${backend}" "${img}" "${label}"
      # Each call cycles its own server end-to-end: test_inference.py owns
      # the lifecycle and tears it down at exit, so reuse across calls
      # isn't on the table. The wall-clock cost is acceptable for an
      # equivalence gate that runs occasionally.
      # MODEL is set only when overridden; otherwise it is inherited from the
      # ambient env (the llama-mode all-models path). Build the env-assignment
      # list explicitly so an empty override doesn't clobber an inherited MODEL.
      local -a env_assign=(
        "USE_LLAMA=${use_llama}"
        "PREFER_ST=${prefer_st}"
        "FORCE_NEW_SERVER=true"
      )
      [[ -n "${model_override}" ]] && env_assign+=("MODEL=${model_override}")
      env "${env_assign[@]}" \
        "${THIS_DIR}/test_inference.sh" "${prompt}" 2>&1 | tee "${out}"
      # extract_responses emits "<model>\t<response>\t<logprob>" per model
      # (logprob may be empty); append backend/image/prompt context to each.
      # We force the recorded model name to result_key so GGUF and ST runs of
      # the same logical model land on the same comparison key. (In all-models
      # llama mode result_key is empty → fall back to the harness's per-run
      # "<model> <--" marker, preserving the original behavior.)
      while IFS=$'\t' read -r model resp lp; do
        [[ -n "${model}" ]] || continue
        [[ -n "${result_key}" ]] && model="${result_key}"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${backend}" "${model}" "${img}" "${label}" "${resp}" "${lp}" >> "${RESULTS_TSV}"
      done < <(extract_responses "${out}" "${result_key:-${MODEL:-default}}")
    done
  done
}

# Count CHECK / REF rows currently in RESULTS_TSV by backend label. The
# accumulating TSV means a phase reads its own contribution by counting rows
# carrying one of its backend labels.
function count_rows() {
  local labels_re="${1}"
  local n; n=$(grep -cE "^(${labels_re})	" "${RESULTS_TSV}" 2>/dev/null)
  echo "${n:-0}"
}

# =============================================================================
# Per-mode phases. See header for the contract; the driver at the bottom
# sequences setup → check → interstitial → ref → finish per mode.
#
# Cross-function state globals: CHECK_ROWS / REF_ROWS (row counts produced by
# the check / ref collection), CHECK_BACKENDS / REF_BACKENDS (the backend
# labels each side uses, a regex alternation for count_rows), MODELS (gguf-st
# logical_key<TAB>gguf_id pairs), ST_CHECK (gguf-st loader-line cross-check).
# =============================================================================

CHECK_BACKENDS=""
REF_BACKENDS=""

# ---- llama: bench (GGUF) vs llama-server (GGUF) -----------------------------
function setup_equiv_llama() {
  # llama.cpp cannot load safetensors (.st) models; including them would
  # enumerate .st rows that run on bench but have no llama counterpart.
  export GGUF_ONLY=true
  # Iterate all mmproj-capable models unless a specific MODEL was requested.
  [[ -n "${MODEL:-}" ]] || export ALL_MODELS_MMPROJ=true
  CHECK_BACKENDS="bench"
  REF_BACKENDS="llama"
}
function gather_check_llama() {
  # CHECK = bench (system under test). MODEL inherited from ambient env;
  # result_key empty → per-model rows key off the harness's run markers.
  [[ -n "${SKIP_BENCH:-}" ]] || run_backend "bench" "false" "false" "" ""
}
function gather_ref_llama() {
  # REF = llama-server (the authoritative reference).
  [[ -n "${SKIP_LLAMA:-}" ]] || run_backend "llama" "true" "false" "" ""
}

# ---- gguf-st: bench .st vision (check) vs bench GGUF vision (ref) -----------
# Deliberately does NOT set GGUF_ONLY (it NEEDS the .st models) and does NOT
# use ALL_MODELS_MMPROJ — it enumerates logical models that have both formats
# and drives each via an explicit per-run MODEL + PREFER_ST.
function setup_equiv_gguf_st() {
  CHECK_BACKENDS="bench-st"
  REF_BACKENDS="bench-gguf"
  # Enumerate (logical_key<TAB>gguf_id) pairs. A logical key is the GGUF id
  # with a trailing -f16/-f32 precision suffix stripped; the .st dir is named
  # by that bare key. A complete pair needs the .st dir with config.json +
  # tokenizer.gguf AND a matching GGUF decoder. Honors MODEL= as a single
  # filter on the logical key or gguf id: incomplete in auto mode = skip,
  # incomplete with explicit MODEL = error.
  local require_complete="false"
  [[ -n "${MODEL:-}" ]] && require_complete="true"
  MODELS=()
  while IFS=$'\t' read -r logical_key gguf_id; do
    [[ -n "${logical_key}" ]] || continue
    if [[ ! -f "models/${logical_key}.st/config.json" ]]; then
      ${require_complete} && { echo "${logical_key} is not a safetensor model." 1>&2; exit 1; }
      echo "${logical_key} is not a safetensor model. skipping."; continue
    fi
    if [[ ! -f "models/${logical_key}.st/tokenizer.gguf" ]]; then
      ${require_complete} && { echo "${logical_key} is missing tokenizer.gguf. ('make st-tok-ggufs')" 1>&2; exit 1; }
      echo "${logical_key} is missing tokenizer.gguf. skipping. ('make st-tok-ggufs')"; continue
    fi
    MODELS+=("${logical_key}	${gguf_id}")
  done < <(enumerate_gguf_st_pairs)
  if [[ ${#MODELS[@]} -eq 0 ]]; then
    echo "gguf-st: no logical model has BOTH a GGUF decoder and a complete .st dir in models/ (nothing to compare)." 1>&2
    [[ -n "${MODEL:-}" ]] && echo "  (MODEL=${MODEL} filter is active)" 1>&2
    exit 1
  fi
}
# Enumerate (logical_key<TAB>gguf_id) pairs whose .st dir exists. Honors MODEL=
# as a filter on the logical key or the full gguf id.
function enumerate_gguf_st_pairs() {
  "${PYTHON}" - "${MODEL:-}" <<'PY'
import re, sys
from pathlib import Path
models_dir = Path("models")
only = sys.argv[1] if len(sys.argv) > 1 else ""
_PRECISION = re.compile(r"-(f16|f32)$", re.IGNORECASE)
seen = set()
for gguf in sorted(models_dir.glob("*.gguf")):
    gid = gguf.stem
    if gid.lower().startswith("mmproj-") or "mmproj" in gid.lower():
        continue
    key = _PRECISION.sub("", gid)
    if (models_dir / f"{key}.st").is_dir() and key not in seen:
        if only and only not in (key, gid):
            continue
        seen.add(key)
        print(f"{key}\t{gid}")
PY
}
function gather_check_gguf_st() {
  # CHECK = bench's safetensors (.st) vision tower (system under test).
  local logical_key gguf_id
  for pair in "${MODELS[@]}"; do
    IFS=$'\t' read -r logical_key gguf_id <<< "${pair}"
    printf '\n########## logical model=%s (st=%s.st) ##########\n' \
      "${logical_key}" "${logical_key}"
    run_backend "bench-st" "false" "true" "${logical_key}" "${logical_key}"
  done
  # Assert the check run actually loaded models as safetensors (not gguf).
  # The loader lines land in the accumulating server ${LOG}; the check run is
  # first, so it owns every line present so far. Qwen3.5 .st vision is
  # unimplemented: it still loads via the safetensors reader, so this assertion
  # holds; an absent vision answer surfaces as a '—' (non-comparison) row in
  # finish, not a crash.
  local gguf_check; gguf_check=$(grep 'ModelReader\[gguf\] created' "${LOG}")
  ST_CHECK=$(grep 'ModelReader\[safetensors\] created' "${LOG}") # global: re-read by gather_ref_gguf_st
  [[ -z "${gguf_check}" && -n "${ST_CHECK}" ]] || {
    echo "FAIL: gguf-st check run should have loaded models as safetensors but loaded gguf." 1>&2
    exit 1
  }
}
function gather_ref_gguf_st() {
  # REF = bench's GGUF vision tower (the known-good reference).
  local logical_key gguf_id
  for pair in "${MODELS[@]}"; do
    IFS=$'\t' read -r logical_key gguf_id <<< "${pair}"
    printf '\n########## logical model=%s (gguf=%s) ##########\n' \
      "${logical_key}" "${gguf_id}"
    run_backend "bench-gguf" "false" "false" "${gguf_id}" "${logical_key}"
  done
  # Assert the ref run ADDED gguf-created lines, and the safetensors lines from
  # the check run are unchanged (the ref run loaded no new safetensors).
  local gguf_ref; gguf_ref=$(grep 'ModelReader\[gguf\] created' "${LOG}")
  local st_ref; st_ref=$(grep 'ModelReader\[safetensors\] created' "${LOG}")
  [[ -n "${gguf_ref}" && "${st_ref}" == "${ST_CHECK}" ]] || {
    echo "FAIL: gguf-st ref run should have loaded models as gguf but loaded safetensors." 1>&2
    exit 1
  }
}

# =============================================================================
# Common phases
# =============================================================================

# interstitial: between the check and reference collections. Validate that the
# check collection produced rows and exit here on a hard failure, before the
# (slower) reference is collected.
function interstitial() {
  CHECK_ROWS=$(count_rows "${CHECK_BACKENDS}")
  [[ "${CHECK_ROWS}" -gt 0 ]] || {
    echo "FAIL: ${EQUIV} check collection produced no rows" 1>&2
    exit 1
  }
  echo ""
  echo "${EQUIV} check collection: ${CHECK_ROWS} row(s) [${CHECK_BACKENDS}]"
}

# finish: after the reference collection. Validate the reference produced rows,
# run the comparison over RESULTS_TSV (semantic + answer-token logprob), and
# emit the final verdict.
function finish() {
  REF_ROWS=$(count_rows "${REF_BACKENDS}")
  [[ "${REF_ROWS}" -gt 0 ]] || {
    echo "FAIL: ${EQUIV} reference collection produced no rows" 1>&2
    exit 1
  }
  echo ""
  echo "${EQUIV} reference collection: ${REF_ROWS} row(s) [${REF_BACKENDS}]"

  printf '\n========== EQUIV RESULTS ==========\n'
  compare_results "${RESULTS_TSV}" "${VISION_PASS_THRESH}" && PASS=true || PASS=false

  printf '\nrun logs: %s\n' "${run_dir}"
  ${PASS:-false} || exit 1
  echo "All checked models passed ${EQUIV} vision equivalence test"
}

# compare_results: render the EQUIV RESULTS table (semantic answer-match +
# answer-token logprob delta) from the results TSV ($1), gating on the logprob
# threshold ($2). Returns non-zero iff any row FAILs. Split out of finish() so
# the large Python comparison isn't inline in the driver phase — mirrors
# test_chat_equiv.sh's separate compare_logprobs().
compare_results() {
  "${PYTHON}" - "${1}" "${2}" <<'PY'
import sys
from collections import defaultdict

path = sys.argv[1]
pass_thresh = float(sys.argv[2])
rows = defaultdict(dict)   # (model, image, prompt) -> {backend: response}
logps = defaultdict(dict)  # (model, image, prompt) -> {backend: float logprob}
order = []
seen = set()
backends_seen: list[str] = []
with open(path) as f:
    for line in f:
        parts = line.rstrip("\n").split("\t", 5)
        if len(parts) < 5:
            continue
        backend, model, image, prompt, resp = parts[:5]
        lp = parts[5] if len(parts) >= 6 else ""
        key = (model, image, prompt)
        if key not in seen:
            seen.add(key); order.append(key)
        if backend not in backends_seen:
            backends_seen.append(backend)
        rows[key][backend] = resp
        if lp.strip():
            try:
                logps[key][backend] = float(lp)
            except ValueError:
                pass

if not order:
    print("(no results)")
    sys.exit(0)

try:
    import shutil
    term_w = shutil.get_terminal_size((220, 24)).columns
except Exception:
    term_w = 220

import re
# Multi-word responses are compared on a first-N-words prefix only — past
# the prefix, free-form prose divergence is expected and is not what we're
# testing. Prompts in this script request ≤10 words; 5 is roughly half,
# enough to catch real semantic disagreement (different subject, swapped
# attribute) without flagging incidental phrasing differences.
_EQUIV_WORD_CAP = 5

def _norm(s: str) -> str:
    """Light normalization for cross-backend equivalence: lowercase,
    collapse whitespace, strip terminator punctuation, then truncate to
    the first _EQUIV_WORD_CAP tokens. Equivalence is a textual
    "did the two backends produce the same answer" check — NOT a
    correctness judgment against ground truth."""
    s = s.strip().lower()
    s = re.sub(r"[.,;:!?\"'`()\[\]]+", " ", s)
    tokens = s.split()
    return " ".join(tokens[:_EQUIV_WORD_CAP])

def _row_result(per_backend: dict) -> str:
    """pass iff every present backend produced the same normalized answer.
    Sentinel responses (<NO-RESPONSE>, <EMPTY>) on either side count as
    a non-comparison and report '—'."""
    vals = [v for v in per_backend.values() if v]
    if len(vals) < 2:
        return "—"
    if any(v.startswith("<") and v.endswith(">") for v in vals):
        return "—"
    normed = {_norm(v) for v in vals}
    return "pass" if len(normed) == 1 else "FAIL"

# The two backends compared per row are whichever two this run produced
# (bench/llama in llama mode; bench-st/bench-gguf in gguf-st mode). Derive
# them from the first two distinct backends seen so the logprob diff is
# mode-agnostic.
_lp_backends = backends_seen[:2]

def _lp_diff(per_backend_lp: dict):
    """Return (a_lp, b_lp, diff) for a row over the two compared backends,
    with None where a backend is missing. diff is |a - b| only when both
    are present."""
    if len(_lp_backends) < 2:
        return None, None, None
    a = per_backend_lp.get(_lp_backends[0])
    b = per_backend_lp.get(_lp_backends[1])
    d = abs(a - b) if (a is not None and b is not None) else None
    return a, b, d

def _lp_result(diff) -> str:
    """pass/FAIL on the answer-token logprob delta vs pass_thresh; '—' when a
    diff couldn't be computed (a backend lacked a logprob fingerprint)."""
    if diff is None:
        return "—"
    return "pass" if diff <= pass_thresh else "FAIL"

w_result = max(len("result"), 4)  # "pass" / "FAIL" / "—"
w_model  = max(len("model"),  max(len(k[0]) for k in order))
w_img    = max(len("image"),  max(len(k[1]) for k in order))
w_prompt = max(len("prompt"), max(len(k[2]) for k in order))
fixed = w_result + 2 + w_model + 2 + w_img + 2 + w_prompt + 2
gutter = 2 * (len(backends_seen) - 1) if backends_seen else 0
remaining = max(40, term_w - fixed - gutter)
w_resp = max(20, remaining // max(1, len(backends_seen)))

def trunc(s: str, n: int) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"

header = (
    f"{'result':<{w_result}}  {'model':<{w_model}}  {'image':<{w_img}}  {'prompt':<{w_prompt}}  "
    + "  ".join(f"{b:<{w_resp}}" for b in backends_seen)
)
print(header)
print("-" * len(header))
last_model = None
n_pass = n_fail = n_skip = 0
for k in order:
    model, img, prompt = k
    if last_model is not None and model != last_model:
        print("-" * len(header))
    last_model = model
    result = _row_result(rows[k])
    if   result == "pass": n_pass += 1
    elif result == "FAIL": n_fail += 1
    else:                  n_skip += 1
    cells = [trunc(rows[k].get(b, "—"), w_resp) for b in backends_seen]
    print(
        f"{result:<{w_result}}  {model:<{w_model}}  {img:<{w_img}}  {prompt:<{w_prompt}}  "
        + "  ".join(f"{c:<{w_resp}}" for c in cells)
    )
print("-" * len(header))
total = n_pass + n_fail + n_skip
print(f"semantic  pass: {n_pass}/{total}   FAIL: {n_fail}/{total}   skip: {n_skip}/{total}")

# ---- answer-token logprob comparison (the two compared backends) -----------
# Informational AND gating: a row whose |a_lp - b_lp| exceeds pass_thresh is a
# logprob FAIL; a row where either backend lacked a logprob fingerprint reports
# '—' and does not gate.
_lp_a = _lp_backends[0] if len(_lp_backends) > 0 else "a"
_lp_b = _lp_backends[1] if len(_lp_backends) > 1 else "b"
print(f"\n----- answer-token logprob (|{_lp_a} - {_lp_b}|, thresh={pass_thresh:g}) -----")
lp_header = (
    f"{'logprob':<{w_result}}  {'model':<{w_model}}  {'prompt':<{w_prompt}}  "
    f"{f'{_lp_a}_lp':>13}  {f'{_lp_b}_lp':>13}  {'diff':>13}"
)
print(lp_header)
print("-" * len(lp_header))
last_model = None
lp_pass = lp_fail = lp_skip = 0
def _fmt(x):
    return f"{x:>13.5f}" if x is not None else f"{'—':>13}"
for k in order:
    model, img, prompt = k
    if last_model is not None and model != last_model:
        print("-" * len(lp_header))
    last_model = model
    b, l, d = _lp_diff(logps[k])
    res = _lp_result(d)
    if   res == "pass": lp_pass += 1
    elif res == "FAIL": lp_fail += 1
    else:               lp_skip += 1
    print(
        f"{res:<{w_result}}  {model:<{w_model}}  {prompt:<{w_prompt}}  "
        f"{_fmt(b)}  {_fmt(l)}  {_fmt(d)}"
    )
print("-" * len(lp_header))
lp_total = lp_pass + lp_fail + lp_skip
print(f"logprob   pass: {lp_pass}/{lp_total}   FAIL: {lp_fail}/{lp_total}   skip: {lp_skip}/{lp_total}")

# Exit non-zero if any row FAILs semantically OR exceeds the logprob threshold.
exit(0 if (n_fail == 0 and lp_fail == 0) else 1)
PY
}

# =============================================================================
# Driver — one place sequences every phase; per-mode logic lives in the
# functions above, dispatched by name ('-' → '_').
# =============================================================================

case "${EQUIV}" in
  llama|gguf-st)
    setup_equiv_${EQUIV_FN}
    echo "Collecting ${EQUIV} check results..."
    gather_check_${EQUIV_FN}
    interstitial
    echo ""
    echo "Collecting ${EQUIV} referent results..."
    gather_ref_${EQUIV_FN}
    finish
    ;;
  *)
    echo "Usage: [MODEL=<model>] ${THIS_SCRIPT} [llama|gguf-st]   (default: llama)" 1>&2
    echo "unknown arg: ${EQUIV}" 1>&2
    exit 1
    ;;
esac

exit 0
