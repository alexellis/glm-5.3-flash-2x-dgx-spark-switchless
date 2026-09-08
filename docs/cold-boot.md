# Cold-boot behaviour

## The failure

The original service unit was added around an already-running deployment. It
proved steady-state recovery, but it did not prove a power-off boot.

On the first real cold boot, both ranks loaded the checkpoint and allocated the
KV pool correctly. FlashInfer then loaded 42 persisted autotune entries and ran
the full dummy autotune workload again. Both ranks remained alive, the GPU sat
busy, no Xid or OOM was recorded, and `/health` never opened. A second clean
attempt reproduced the same stop at the same point.

There was also a lifecycle defect: stopping a `Type=oneshot` unit while its
long `ExecStart` was still activating did not invoke the expected `ExecStop`,
leaving both containers behind.

## The correction

`patch-autotune-cache.py` adds a deliberately narrow behaviour behind
`GLM53_TRUST_FLASHINFER_CACHE=1`: once the cache has been broadcast, written,
barriered, and successfully loaded, return instead of running the redundant
dummy autotune. A missing cache still takes the ordinary tune-and-save path.

The systemd unit is now `Type=simple`. Its foreground supervisor traps stop and
exit, and `ExecStopPost` also calls the pair teardown. A failure or interruption
therefore removes worker and head containers together.

The launcher also requires `/etc/glm53-tp2.env` to be byte-for-byte identical
on both ranks. An asymmetric KV allocation is otherwise accepted by each
worker independently and silently limits the whole TP pool to the smaller
rank.

## Observed proof

With the correction mounted on both ranks:

- main rank weight load: about 11 minutes 32 seconds;
- total model load: about 12 minutes;
- persisted cache: 42 configurations loaded;
- CUDA graph capture: about 76 seconds;
- API ready: about 14 minutes after start; and
- `/health`, exact text output, and a real image-understanding request passed.

The first boot on a clean cache can take longer because it must generate the
cache. Preserve `/var/lib/glm53-tp2/cache/flashinfer` across restarts.
