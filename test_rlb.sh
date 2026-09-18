#!/usr/bin/env bash
THIS_SCRIPT=$(basename "$0")
THIS_DIR=$(dirname "$0")

cd "${THIS_DIR}"

export MODEL=Qwen3.5-9B

TS=$(date +%Y%m%d_%H%M%S)

DRY_RUN=${DRY_RUN:-false}

# THINK=true first: Qwen 3.5 was trained to refine with thinking enabled, so
# the RLB recurrence signal is expected to be strongest in that mode. Run it
# first so that even if the sweep is aborted early, the richest data lands.
TEST_RLB_THINK_SWEEP=${TEST_RLB_THINK_SWEEP:-true}
TEST_RLB_HALT_RULE_SWEEP=${TEST_RLB_HALT_RULE_SWEEP:-dH_threshold,fixed_3,fixed_5}
TEST_RLB_ALPHA_SWEEP=${TEST_RLB_ALPHA_SWEEP:-auto,0.3,0.5}
TEST_RLB_PREFILL_SWEEP=${TEST_RLB_PREFILL_SWEEP:-false}

TEST_RLB_RUN_NON_RLB=${TEST_RLB_RUN_NON_RLB:-true}

PROMPT_COUNT_BITS_FLOAT64="Write a C function that counts the set bits in a float64 and returns the value as a uint8_t."
PROMPT_COUNT_BITS_DOUBLE="Write a C function that counts the set bits in a double and returns the value as a uint8_t."
PROMPT_COUNT_BITS_DOUBLE_SCRATCH=${PROMPT_COUNT_BITS_DOUBLE/C function/C function from scratch}
PROMPT_SIN_OF_DEGREES="Write a C function that computes sin(x) where x is in degrees."
PROMPT_PARSE_INT="Write a C function that parses a signed integer from a null-terminated string and returns an error code if the value would overflow int32_t or if the string contains invalid characters."
PROMPT_PARSE_INT_SCRATCH=${PROMPT_PARSE_INT/C function/C function from scratch}
PROMPT_READ_UINT32="Write a C function that reads a uint32_t from a byte buffer that takes a bool param indicating whether to read in big-ending or little-endian byte order."
PROMPT_FREE_LINKED_LIST="Write a C function that frees a singly-linked list where each node contains 'int value; struct node *next;'."

PROMPT=${PROMPT:-"${*}"}
PROMPT_LIST=${PROMPT_LIST:-}
[[ -z "${PROMPT}" && -z "${PROMPT_LIST}" ]] && { echo "prompt required (argv or PROMPT= or PROMPT_LIST=<prompt_key1>,[prompt_key2],...)" 1>&2; exit 1; }
[[ -n "${PROMPT}" && -n "${PROMPT_LIST}" ]] && { echo "PROMPT (or argv) and PROMPT_LIST are mutually exclusive" 1>&2; exit 1; }

[[ "${PROMPT_LIST}" == all ]] && {
  PROMPT_LIST="count_bits_float64,count_bits_double,count_bits_double_scratch,sin_of_degrees,parse_int,parse_int_scratch,read_uint32,free_linked_list"
}

function resolve_prompt() {
  local prompt="${*}"
  case ${prompt} in
    *" "*|Hello|hello|Hi|hi)
      # this is a manual prompt - common greeting or more than one word.
      prompt=${prompt}
      ;;
    # keyed prompts
    count_bits_float64)
      prompt=${PROMPT_COUNT_BITS_FLOAT64}
      ;;
    count_bits_double)
      prompt=${PROMPT_COUNT_BITS_DOUBLE}
      ;;
    count_bits_double_scratch)
      prompt=${PROMPT_COUNT_BITS_DOUBLE_SCRATCH}
      ;;
    sin_of_degrees)
      prompt=${PROMPT_SIN_OF_DEGREES}
      ;;
    parse_int)
      prompt=${PROMPT_PARSE_INT}
      ;;
    parse_int_scratch)
      prompt=${PROMPT_PARSE_INT_SCRATCH}
      ;;
    read_uint32)
      prompt=${PROMPT_READ_UINT32}
      ;;
    free_linked_list)
      prompt=${PROMPT_FREE_LINKED_LIST}
      ;;
    *)
      echo "invalid or unknown prompt key \"${prompt}\"" 1>&2
      return 1
      ;;
  esac
  echo "${prompt}"
  return 0
}

[[ -n "${PROMPT}" ]] && {
  PROMPT=$(resolve_prompt "${PROMPT}") || exit 1
  PROMPT_LIST=("${PROMPT}")
  unset PROMPT
} || {
  PL=${PROMPT_LIST//,/ }
  PROMPT_LIST=()
  for P in ${PL}; do
    [[ ${P} != *" "* ]] || { echo "PROMPT_LIST must only contain prompt keys. (\"${P}\" invalid)" 1>&2; exit 1; }
    resolve_prompt "${P}" > /dev/null || exit 1;
    PROMPT_LIST+=("${P}")
  done
  unset PL P
}

function run_inference() {
  local params=()
  params+=("RLB_ALPHA=${RLB_ALPHA}")
  params+=("RLB_HALT_RULE=${RLB_HALT_RULE}")
  params+=("RLB_PREFILL=${RLB_PREFILL}")
  params+=("THINK=${THINK}")
  params+=("USE_RLB_GEN=${USE_RLB_GEN}")

  echo "${params[*]} ./test_inference.sh [prompt elided]"
  ${DRY_RUN} && echo "DRY_RUN" && return 0
  ./test_inference.sh "${PROMPT}"
}

function rlb_sweeps() {
  echo ""
  echo "running RLB sweeps"
  echo "------------------"
  local T P H A
  for T in ${TEST_RLB_THINK_SWEEP//,/ }; do
    for P in ${TEST_RLB_PREFILL_SWEEP//,/ }; do
      for H in ${TEST_RLB_HALT_RULE_SWEEP//,/ }; do
        for A in ${TEST_RLB_ALPHA_SWEEP//,/ }; do
          RLB_ALPHA=${A} \
          RLB_HALT_RULE=${H} \
          RLB_PREFILL=${P} \
          THINK=${T} \
          USE_RLB_GEN=true run_inference
        done
      done
    done
  done
}

function non_rlb_sweeps() {
  echo ""
  echo "running NON-RLB sweeps"
  echo "----------------------"
  local T
  for T in ${TEST_RLB_THINK_SWEEP//,/ }; do
    THINK=${T} USE_RLB_GEN=false run_inference
  done
}

function run_suite() {
  echo ""
  echo "running suite with prompt \"${PROMPT}\""
  echo "==============================="

  rlb_sweeps

  if ${TEST_RLB_RUN_NON_RLB}; then
    non_rlb_sweeps
  else
    echo "TEST_RLB_RUN_NON_RLB=false, non_rlb_think sweep skipped."
  fi
}

function run_suites() {
  local P
  echo ""
  echo "Running test suites for prompts ${PROMPT_LIST[*]}"
  echo "##########################"

  for P in "${PROMPT_LIST[@]}"; do
    PROMPT=$(resolve_prompt "${P}") run_suite
  done
}

BENCH_PORT=$(awk '/^[ \t]*port[ \t]*=/ { gsub("=", " "); print $2; exit(0); }' config/api_config.toml)
BENCH_BASE_URL=http://localhost:${BENCH_PORT}
BENCH_API_BASE_URL=${BENCH_BASE_URL}/api/v1
BENCH_CTL_URL=${BENCH_BASE_URL}/ctl

function wait_for_starting_server() {
  # Wait for server to be ready (up to 30s)
  for _i in $(seq 1 60); do
    curl -sf "${BENCH_API_BASE_URL}/models" 2>&1 > /dev/null && return 0
    sleep 0.5
  done
  echo "failed getting models from ${API_BASE_URL}/models" 1>&2
  return 1
}

function quit_api_server() {
  (curl -s "${BENCH_CTL_URL}/?quit&now" > /dev/null && sleep 1) || {
    ps aux | awk '/awk/ { next; } /bin\/bench/{ print $2; }' | xargs kill
  }
  echo "bench api server shut down."
}

DATA_DIR=./memory/rlb_test/run_${TS}
mkdir -p "${DATA_DIR}"

# managing the server life cycle ourselves saves a server startup/shutdown/model_load cycle per test run.
# over a total sweep set of 30+ tests that adds up.
function start_api_server() {
  [[ -x "./bin/bench" ]] || { echo "./bin/bench binary missing. 'make all' first." 1>&2; exit 1; }
  ./bin/bench serve-api serve-api --log "${DATA_DIR}/test_rlb_server.log" --log-level NONE &
  SERVER_PID=$!
  # test_inference.sh pulls the model list from bench api server before running which would
  # make this redundant, but checking here lets us early terminate.
  wait_for_starting_server || { echo "bench api server failed to start." 1>&2 && exit 1; }
  echo "bench api server started."
}

quit_api_server
trap quit_api_server EXIT
start_api_server

run_suites 2>&1 | tee "${DATA_DIR}/test_rlb.log"

! [[ -d bin/diag/rlb ]] || mv bin/diag/rlb "${DATA_DIR}"/
