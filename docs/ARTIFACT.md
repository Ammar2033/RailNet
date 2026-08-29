# RailNet Artifact

## Compiled directory (the working format)

`railnet compile model.safetensors --out compiled/` produces:

```
compiled/
  manifest.json                      # config, source_model, tokenizer, constants, per-tensor index, verdict, sha256
  layers/layer_00/q_proj.json        # rail table + routing table + shape + sha256
  layers/layer_00/q_proj.routeids.npy  # (out, in) uint16 map: each weight's BF16 bit pattern
  ...
```

`RailNetModel.load("compiled/")` runs from this. Norm vectors and the tied
embedding are **not** in the artifact — they are streamed from `source_model`
(the original safetensors) at runtime (compression not claimed for those,
spec 50). So the compiled directory is only runnable next to its safetensors.

`verify_checksum` (manifest + each tensor JSON) and `compiler.model.verify_compiled`
(structure, shapes, rail-id bounds) validate a directory.

## Single-file `.rnmodel` — OPEN DESIGN ITEM

`artifacts.write_rnmodel` / `verify_rnmodel` implement the container framing
(magic `RNET`, version, JSON header, canonical SHA-256) and are tested at that
level. `read_rnmodel` currently raises `NotImplementedError`.

Unresolved before a runnable single file:

1. **Norm + embedding.** A `.rnmodel` that does not reference an external
   safetensors must carry the ~1152-wide norm vectors (tiny) and decide what to
   do about the embedding (~600 MB for Gemma3 1B, compression not claimed).
   Options: (a) pack norms, keep an external-safetensors reference for the
   embedding; (b) pack norms + embedding raw; (c) a separate embedding sidecar.
2. **Route-map storage.** Inline base64 blows the JSON header up; a binary blob
   section after the header (offset table) is the likely answer — ties into the
   route-map compression work (docs/MEMORY.md).
3. **Config + tokenizer.** Embed the HF config; reference or embed the tokenizer.

Status: `PLANNED`. The directory format is the contract until this is decided.
