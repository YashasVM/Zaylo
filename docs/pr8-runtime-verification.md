# PR 8 runtime verification

Verified on Linux x86_64 on 2026-11-16 using one CPU inference thread. No production services, paid APIs, or large model downloads were used. The available RTX 4050 was not used: the reusable Python environment has CPU-only PyTorch.

## Regressions found and fixed

- A symlinked llama-server could not find libllama.so. Discovery now resolves the executable before setting library paths.
- b4800 started with --embeddings reported healthy but returned 501 for completions. The managed server now stays in chat mode, reports no embedding capability, and returns 503 EMBEDDINGS_UNAVAILABLE. Document retrieval uses keyword fallback.
- Forge discarded a requested adapter path when it did not exist. Comparison now rejects that request before starting Python.

## Actual model execution

- Reused SmolLM2-135M-Instruct-Q8_0.gguf and llama.cpp b4800. Real HTTP chat through ApiServer returned 200 and "The capital of France is Paris." Usage was 22 prompt tokens and 8 completion tokens. Streaming returned content and [DONE].
- The same API returned 503 for embeddings and for image generation without an engine, and 404 for a nonexistent image. LAN metrics returned 401 without a key and 200 with a valid key.
- Ran the real diffusion worker with diffusers/tiny-stable-diffusion-torch at revision c05d78754bd90a3ffea8fd3a0823ac17329cd1a6, seed 123, CPU, two requested steps, 64x64. Pillow verified the PNG dimensions and nonconstant pixels. File size was 11,449 bytes; SHA-256 was 795e01dded5add0177f25586a52287bae927266e2e84e091f76cd050d17756a8. This tiny testing model proves diffusion execution, not visual quality. The Python worker accepts 64x64 directly; the public API deliberately requires at least 256x256.
- An incompatible Flax diffusion fixture produced an actionable error and no image. The compatible PyTorch fixture above succeeded without changing worker code or synthesizing replacement output.
- Ran train_lora.py with hf-internal-testing/tiny-random-LlamaForCausalLM at revision 9fb191250dd56d0ba7ec9785a025ed29c03d5998. Two distinct examples, 50% validation split, rank 2, one epoch, one optimizer step. Loss was 10.2929 and held-out loss was 10.3180. Saved held-out JSONL, checkpoint, and adapter. Asserted that LoRA B tensors contain a nonzero update.
- Ran evaluate_compare.py on the exact held-out prompt with the real base and adapter, then merge_lora.py. Reloaded both PEFT and merged Safetensors models and compared logits: maximum absolute difference was 5.960464477539063e-08. Random test weights produce meaningless text; no accuracy or quality claim follows from this run.
- Exported with the actual b4800 convert_hf_to_gguf.py and matching gguf-py source. The upstream tiny model has invalid pad_token_id=-1, which the converter rejected safely without publishing a target. In a separate scratch copy, changed that fixture metadata to pad_token_id=0, then produced a 2,790,016-byte F16 GGUF. The exported GGUF then loaded in b4800 and generated text through the committed smoke check. No application code silently repairs model metadata.

## Reproduce chat verification

Build the workspace and supply an installed runtime directory containing llama/llama-server or its nested upstream layout, plus a local GGUF:

```sh
pnpm build --concurrency=1
node tests/smoke_runtime.mjs /path/to/runtime /path/to/model.gguf
```

The smoke check loads real weights, checks health, generates text, asserts the embedding error, and stops its own process. It downloads nothing and uses one CPU thread. Use an external timeout if running unattended.

## Local checks

- pnpm test --maxWorkers=1 --no-file-parallelism: 138 passed across 19 files.
- python3 -m unittest discover -s tests -p 'test_forge_workers.py' -v: 7 passed.
- pnpm build --concurrency=1 --force: 16 tasks passed without cache. Rebuilt affected packages after the final runtime fix.
- pnpm typecheck --concurrency=1 --force: 30 tasks passed without cache.
- The committed real-runtime smoke check passed with the cached SmolLM2 model.

An initial suite run hit the shared /tmp disk quota: 14 suites failed to load and 72 tests passed. The successful rerun used a worktree-local TMPDIR. Initial converter setup also hit that quota; converter dependencies were installed in worktree-local scratch space. These environmental failures were not counted as passing checks.

## Remaining limits

GitHub Validate Zaylo requires maintainer approval before it can execute. Local results do not substitute for that CI run or a maintainer review. CUDA, Metal/MPS, Vulkan, Windows/macOS installers, full-size diffusion quality, and production-model fine-tuning remain unverified. Image generation succeeded at the worker boundary; real image API delivery remains covered by stub-based regression tests. Semantic embeddings are explicitly unavailable with the managed pinned chat runtime.
