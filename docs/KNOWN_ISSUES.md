# Known Issues

## Missing Official Environment File

The repository README references `env/spark_infer.yaml`, but the branch does
not contain an `env/` directory. E000 reconstructs an environment from the
server base env and exports `env/requirements-e000.txt` plus
`env/environment-e000.yml` after the sample succeeds.

## Transformers and PyTorch Compatibility

The local VeOmni Qwen3 patch targets Transformers 4.57.3, while the server base
environment uses PyTorch 2.1.2. A scoped `sitecustomize.py` shim in the E000
conda env bridges the `_pytree` API mismatch. This does not modify repository
source code.

## wav2vec2 PyTorch Bin Safety Gate

Transformers 4.57 refuses to load `pytorch_model.bin` with PyTorch below 2.6 due
to a torch-load safety gate. The Spark-TTS wav2vec2 `pytorch_model.bin` was
converted once to `model.safetensors` in the external model directory.

## Parameter Budget Failure

The current loaded/invoked module audit reports:

```text
cot_tts_text_model_lora:          619,708,416
spark_bicodec:                    156,316,291
spark_wav2vec2_feature_extractor: 315,438,720
total:                          1,091,463,427
```

Under the challenge interpretation recorded in `AGENTS.md`, every loaded or
invoked inference module counts, so this E000 sample run is not yet proven
eligible for the `<1B` parameter-constrained category.

## RTF Scope

The recorded E000 RTF is subprocess cold-start style and includes process
startup plus model loading. A true in-process steady-state RTF still needs a
warm-model runner.

## Vendor Outputs

Tracked files under `infer/results/sample_case.*` are imported vendor
artifacts. They are not evidence for this E000 reproduction.
