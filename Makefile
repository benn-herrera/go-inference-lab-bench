DIAGRAM_TARGETS  := $(patsubst models/arch/%.arch.toml,models/arch/%.arch.svg,$(wildcard models/arch/*.arch.toml))
ST_TOK_TARGETS  := $(patsubst models/%.st/tokenizer.json,models/%.st/tokenizer.gguf,$(wildcard models/*.st/tokenizer.json))
ST_GGUF_TARGETS := $(patsubst models/%.st,models/%-f16.gguf,$(wildcard models/*.st))
GH_ROOT := $(shell dirname $$(git remote -v | awk '{print $$2; exit 0;}'))

.PHONY: all arch-diagrams arch-diagram-targets build test integration-test equiv-test update-dependencies udpate-agents-dependency serve model-diagrams clean agents st-tok-ggufs st-ggufs

all: build

AGENTS_REPO := https://github.com/benn-herrera/adjagent.git
AGENTS_DIR := .claude/$(basename $(notdir $(AGENTS_REPO)))
agents:
	@mkdir -p $(dir $(AGENTS_DIR))
	@[[ -d "$(AGENTS_DIR)" ]] && git -C "$(AGENTS_DIR)" pull || git -C "$(dir $(AGENTS_DIR))" clone $(AGENTS_REPO)
	just --justfile $(AGENTS_DIR)/justfile install "$(CURDIR)"

build:
	$(MAKE) -C src all
	@[[ -L bin/config ]] || (cd bin && ln -s ../config .)
	@[[ -L bin/models ]] || (cd bin && ln -s ../models .)

LOG_LEVEL ?= INFO
LOG ?= bin/bench.log
serve: build st-tok-ggufs
	./bin/bench serve-api --log $(LOG) --log-level $(LOG_LEVEL)

equiv-test: build
	./test_chat_equiv.sh stateless
	./test_chat_equiv.sh llama
	./test_chat_equiv.sh standard-attention
	./test_chat_equiv.sh gguf-st
	./test_vision_equiv.sh

PROMPT ?= explain pi in one short sentence.
integration-test: build
	@(ls models/*.gguf 2>&1) > /dev/null || (echo "$@: No model/.gguf files." 1>&2 && exit 1)
	@(rm -f ./bin/itest*.log 2>&1) > /dev/null || true
	LOG=./bin/itest.log FORCE_NEW_SERVER=$${FORCE_NEW_SERVER:-true} ALL_MODELS=$${ALL_MODELS:-true} THINK=$${THINK:-true} MAX_TOKENS=$${MAX_TOKENS:-1000} ./test_inference.sh "$(PROMPT)" 2>&1 | tee ./bin/itest_stdout.log
	@grep '<NO-MODEL-RESPONSE>' ./bin/itest_stdout.log && exit 1 || true
	@grep '\[ERROR\]' ./bin/itest.log && exit 1 || true

st-tok-ggufs: $(ST_TOK_TARGETS)
st-ggufs: $(ST_GGUF_TARGETS)

models/arch/%.arch.svg: models/arch/%.arch.toml
	@[[ -x ./bin/bench ]] || (echo "make build first." 1>&2 && exit 1)
	./bin/bench gen-arch-diagram $(<) $(@)

arch-diagram-targets: $(DIAGRAM_TARGETS)

arch-diagrams: build
	@$(MAKE) -B arch-diagram-targets

update-attributions:
	claude -p "read CONVENTIONS.md and ARCHITECTURE.md, then read the direct imports section of src/go.mod and update 'Third Party Acknowledgements' at the end of README.md" \
	      --allowedTools "Read,Edit,Write,Glob,Grep" \
				--model sonnet

update-dependencies:
	@$(MAKE) -C src $@

test:
	@$(MAKE) -C src $@

clean:
	@$(MAKE) -C src $@

# used only for reference - llama.cpp is not built or used for inference
llama-cpp:
	@$(MAKE) -C tools $@

# leave this rule at the end - SECONDEXPANSION applies to everything after it
# handles on-demand generation of f16 ggufs from safetensor directories.
# DO NOT make st-ggufs a dependency of anything else. this is not a 1-second process
.SECONDEXPANSION:

#	Handles creation of tokenizer-only thing gguf sidecar in [model].st/ safetensor directories.
# 1. it's quick and cheap
# 2. the sidecar is required for loading a safetensor model directly in the server.
models/%.st/tokenizer.gguf: models/%.st/tokenizer.json models/%.st/tokenizer_config.json $$(wildcard models/%.st/*.jinja)
	./tools/hf_to_gguf.sh --bench-tokenizer $(dir $(<))

# Decoder F16 GGUF. For multimodal models (config.json has vision_config or
# audio_config), --bench-convert also produces an mmproj-<name>.gguf sidecar
# alongside the decoder, or prints a prominent warning if the converter's
# --mmproj support doesn't cover the architecture. The sidecar is not listed
# in Make's dependency graph by design — it's a side-effect of producing the
# decoder GGUF, regenerated whenever the decoder is.
models/%-f16.gguf: $$(wildcard models/%.st/*.json) $$(wildcard models/%.st/*.safetensors) $$(wildcard models/%.st/*.jinja) $$(wildcard models/%.st/*.py)
	OUT_TYPE=f16 ./tools/hf_to_gguf.sh --bench-convert $(dir $(<))
