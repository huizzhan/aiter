---
name: sglang-debug-log
description: >-
  Temporarily instrument SGLang Python source with debug prints, start the
  server, drive traffic with a client/benchmark, extract the printed values from
  logs, then revert the code and stop the server. Use when the user wants to
  print/inspect runtime values (tensor shapes, cu_seqlens, states, args) of an
  SGLang kernel or function under a real request, or asks to "开启 debug log
  测试后关闭" an sglang service.
disable-model-invocation: true
---

# SGLang Debug-Log Workflow

Instrument SGLang source with temporary `print(...)` statements, run a real request to capture the values, then clean up (revert code + stop server). SGLang loads modules at process start, so **any source edit requires a server restart** to take effect.

## Environment defaults (this setup)

- Scripts: `/opt/scripts/qwen3.5_ptpc/server.sh` (launch) and `client.sh` (bench).
- SGLang source root: `/opt/sglang/python/sglang/`.
- Server port: **8081** (port 8080 is taken by an AMD gateway; do not use 8080).
- `server.sh` writes a timestamped log to `logs/server_YYYYmmdd_HHMMSS.log` via `tee`; find the newest with `ls -t logs/*.log | head -1`.
- A proxy hijacks localhost. Always bypass it: `curl --noproxy '*' ...` and `export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost` before running the client.

Confirm these by reading the actual scripts first; adjust if the user changed them (port, tp-size, model path).

## Workflow checklist

```
- [ ] 1. Locate target function(s) + read signature
- [ ] 2. Add temporary debug prints (gated, see below)
- [ ] 3. Restart server, wait for "ready to roll"
- [ ] 4. Run client to drive traffic
- [ ] 5. Extract prints from the log
- [ ] 6. Revert ALL debug edits
- [ ] 7. Stop the server (if the user asked)
```

## Step 2: Add debug prints (avoid the pitfalls)

Two pitfalls make naive prints useless:

1. **Warmup floods the budget.** Server startup runs a tiny warmup request (short seq, e.g. T=80). A plain counter limit is fully consumed by warmup before the real client runs. **Gate on sequence length** so only the real request prints, e.g. `T >= 500`.
2. **Per-rank / per-layer duplication.** Each TP rank is a separate process with its own module state, and every model layer calls the function. Expect the same line repeated many times; dedup with `sort -u` when reading.

Print pattern (module-level counter + shape/value helpers):

```python
_DBG_LIMIT = 4
_dbg_n = 0

def _dbg_shape(x):
    return None if x is None else (tuple(x.shape) if hasattr(x, "shape") else x)

def _dbg_val(x):
    return None if x is None else (x.tolist() if hasattr(x, "tolist") else x)
```

At the top of the target function (after seq length `T` is known):

```python
    global _dbg_n
    if T >= 500 and _dbg_n < _DBG_LIMIT:
        _dbg_n += 1
        print(
            f"[DBG fn] arg1={_dbg_shape(arg1)} arg2={_dbg_shape(arg2)} "
            f"cu_seqlens={_dbg_val(cu_seqlens)} "
            f"initial_state_indices={_dbg_val(initial_state_indices)}",
            flush=True,
        )
```

- Use `flush=True` so lines appear promptly.
- Use `_dbg_shape` for tensors (shape only) and `_dbg_val` for small index/offset tensors you want values of (`.tolist()` syncs GPU→CPU; fine for debug, avoid on huge tensors).
- When instrumenting **two functions in different files**, give each its own counter/helpers (e.g. `_dbg_n_h`, `_dbg_shape_h`) to avoid name clashes.
- Use a distinct grep tag like `[DBG fn]` per function.

## Step 3: Restart server

```bash
cd /opt/scripts/qwen3.5_ptpc
ps -ef | grep -iE "sglang|launch_server" | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 3
nohup bash server.sh > /tmp/sgl_boot.txt 2>&1 &
```

Then wait for readiness (weights load + AITER JIT compile take minutes; tp=8 ~2-4 min):

```bash
LOG=$(ls -t logs/*.log | head -1)
for i in $(seq 1 80); do
  grep -qiE "ready to roll|The server is fired up" "$LOG" && { echo READY; break; }
  grep -qiE "address already in use|Traceback|CUDA error|out of memory|HIP error|Initialization failed" "$LOG" && { echo ERROR; break; }
  sleep 5
done; tail -n 3 "$LOG"
```

Poll with a long `block_until_ms` (e.g. 300000+). "address already in use" means something else holds the port — change `--port` or stop the other process.

## Step 4: Run client

```bash
cd /opt/scripts/qwen3.5_ptpc
export no_proxy=127.0.0.1,localhost NO_PROXY=127.0.0.1,localhost
bash client.sh > bench.log 2>&1
```

## Step 5: Extract values

```bash
LOG=$(ls -t logs/*.log | head -1)
grep "DBG fn\]" "$LOG" | sort -u
```

Report the values. Note per-rank duplication (values identical across ranks unless per-rank state differs).

## Step 6: Revert (mandatory)

Remove every debug edit (the module-level counters/helpers AND the in-function print block) from all touched files. Verify with `ReadLints` and by re-reading the changed regions. Never leave debug prints in the source.

## Step 7: Stop server (when asked)

```bash
cd /opt/scripts/qwen3.5_ptpc
ps -ef | grep -iE "sglang|launch_server" | grep -v grep | awk '{print $2}' | xargs -r kill -9
sleep 3
ps -ef | grep -iE "sglang|launch_server" | grep -v grep || echo "no sglang process"
curl -s --noproxy '*' --max-time 3 http://127.0.0.1:8081/health >/dev/null 2>&1 && echo "still up" || echo "stopped"
```

## Notes

- The in-memory server keeps the debug code until restarted; after reverting source, the running server still logs debug lines (harmless). Restart to load clean code if needed.
- `server.sh` uses `> server.log`/`tee` — timestamped logs in `logs/` won't be overwritten across runs; back up a log if it must be preserved.
