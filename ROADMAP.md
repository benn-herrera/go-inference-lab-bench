# ROADMAP – Inference Lab Bench

## Critical Issues
* None open

## Serious Issues
* None open

## Tech Debt
* Code review following vision support implementation - a LOT has changed.

## Features
* More safetensors architectures (.arch.stmap.toml)

## Deferred Work
1. vision support (done!)
  1. vision tower diagrams (not done)
2. **Chat client streaming + acontextual mode** — `--no-history` flag, real-time SSE output; refactor client to separate impl from CLI entry point.
3. Batch inference - (pad_token handling will be needed)
4. Multiple concurrent models
5. Linux/CUDA support (rename GPU init, add CUDA backend)
6. implement microbatching support for diffusion generation
