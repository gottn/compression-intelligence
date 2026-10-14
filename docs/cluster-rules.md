# AAU AI Cloud: rules we follow

Sources, read 2026-10-14: login notice on `ai-fe02`, https://hpc.aau.dk/ai-cloud/fair-usage,
https://hpc.aau.dk/ai-cloud/resource-quotas, https://hpc.aau.dk/ai-cloud/terms-and-conditions,
https://hpc.aau.dk/ai-cloud/getting-started/run-jobs, and `scontrol`/`sacctmgr` on the cluster.

## Hard rules (terms and conditions; misuse means a permanent ban)

1. **No computation on the front end** (`ai-fe02`). It is for file management, git and job submission only. Tokenizing, scoring, downloading models and building environments all happen inside jobs.
2. **Jobs run only while they actively use the resources they hold.** No idle allocations, no interactive GPU shells, no "allocate and then think". Every job has a defined start and stop and releases on completion.
3. **No GPU unless the job needs one.** CPU-only work (corpus collection, analysis, plots) goes to `--partition=cpu`.
4. **Multiple GPUs only when the job really uses them.** One GPU per job for ≤3B models; the default quota is 12 simultaneous jobs and 12 GPUs.
5. **Only non-confidential data** (AAU data classification level 1). Our corpora are public text; model weights are public.
6. **Not a storage system.** Results are copied back to the Mac and committed; finished run directories and caches are deleted.
7. **GDPR**: nothing personal is uploaded.

## Strong recommendations we treat as rules

- `sbatch`, never `srun` for real work. `srun` only inside an sbatch script.
- Containers for GPU software: `singularity exec --nv` with an image from `/home/container` (or our own `.sif` in `~/containers`).
- Test GPU utilisation on a short job before a long one. Our smoke job runs one model on one corpus with a 20-minute limit.
- Prefer off-peak submission for long jobs: `--begin=` for nights and weekends.
- Interactive editing on the cluster is not done. Code is developed on the Mac, committed, pushed, and pulled on the front end by `infra/cluster/submit.sh`, which refuses to submit a dirty checkout.

## Limits from Slurm (queried 2026-10-14)

| item | value |
|---|---|
| partitions | `prioritized` (default; a40, t4, l4, l40s, a10, v100), `cpu`; `aicentre*` need Pioneer Centre affiliation |
| max and default walltime | 6 days; we always set `--time` explicitly and short |
| QoS `normal` | 12 jobs, 12 GPUs per user |
| preemption | `PreemptMode=REQUEUE` on `prioritized`: jobs can be requeued, so scoring is resumable and idempotent |
| best single-GPU nodes for us | `a768-l40s-0x` (L40S 48 GB, 128–192 cores, 735 GB RAM) |

## How this repo uses the cluster

- `infra/cluster/setup.sh`: one-time; deploy key, clone to `~/projects/compression-intelligence`, directories `~/runs/ci`, `~/hf_cache`.
- `infra/cluster/submit.sh <job>`: pulls fast-forward, refuses local edits, records the commit SHA, submits `infra/slurm/<job>.sbatch` with output under `~/runs/ci/<job>/<timestamp>-<sha>/`.
- Every job: explicit `--time`, `--mem`, `--cpus-per-task`, one GPU, `HF_HOME=~/hf_cache`, container from `/home/container`, `set -euo pipefail`, exits when done.
- Results come back with `rsync` to the Mac and are committed there; the cluster copy is deleted after.

## Validated (2026-10-14)

Smoke job v2 (job 1080233, commit c9242c3) completed in 1 min 57 s on an NVIDIA A10: container `pytorch_25.09.sif`, venv on top, GPT-2 scored enwik8_ctrl at 1.2426 bpb against the Mac's 1.243, verdict ok. v1 (job 1080110) timed out at 20 min because pip took 14 min and the sanity command also fetched a 3.5 GB model. The four "not a valid test operator" lines at container start come from the image's own startup scripts and are harmless.
