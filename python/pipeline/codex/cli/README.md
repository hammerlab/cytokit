# CODEX Processor CLI

The cli application ```processor.py``` can be used to execute processing of CODEX data given raw TIF stacks.

Usage and configuration for this application can be viewed using ```python processor.py localhost -- --help```.

## Example CLI Invocations

#### Minimal Windows Example

```bash
# On Windows, these operations must be forced into cpu-only mode as some CUDA instructions
# used by tensorflow are not valid
set CODEX_CPU_ONLY_OPS=CodexFocalPlaneSelector,TranslationApplier

# Run the processor
python processor.py localhost --data-dir="F:\7-7-17-multicycle" --output-dir="F:\7-7-17-multicycle-out"
```

#### Advanced Windows Example

```bash
set CODEX_CPU_ONLY_OPS=CodexFocalPlaneSelector,TranslationApplier

# Run the process to process tiles 1 through 8 for region 1 only,
# use the first two GPUs, do NOT run best focal plane selection,
# and use log levels to debug codex application but not TensorFlow
python processor.py localhost ^
--region-indexes=1 --tile-indexes=(1,9) ^
--data-dir="F:\7-7-17-multicycle" ^
--output-dir="F:\7-7-17-multicycle-out-pipeline\1-Processor" ^
--run-best-focus=False --gpus=[0,1] ^
--codex-py-log-level=debug --tf-py-log-level=error --tf-cpp-log-level=error
```
