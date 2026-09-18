# Inference Lab Bench

## Read the docs before ad-hoc grepping
Before discovering a capability by grepping source — a harness/API/tooling
feature, where a subsystem lives, how something is wired — check **CONVENTIONS.md**
(API extensions, `test_inference.sh`/`.py` env knobs, `test_*_equiv.sh`, tooling)
and **ARCHITECTURE.md** (subsystem/source map) first. They summarize these; grep
source only to confirm wiring once the docs point you at the right place. Example
miss: hunting apiserver/engine code for how `"stateless"` is passed when it's
listed in CONVENTIONS.md's API extensions — and the documented "stateless is the only
`ForwardCaptures` mode" fact answers *why* the capture path is stateless.
