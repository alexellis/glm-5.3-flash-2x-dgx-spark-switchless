# SPDX-License-Identifier: Apache-2.0
# Derived from the vLLM project (Apache-2.0): AdaptiveKScheduler subclasses
# vllm.v1.core.sched.async_scheduler.AsyncScheduler and re-implements its methods.
# Modifications (c) 2026 Jacopo Nardiello, MIT-licensed where separable.
# See CREDITS.md.
"""Adaptive per-request speculative draft length for vLLM V1 (DFlash2 lane).

Why
---
Draft-token acceptance varies substantially by request shape: structured output
often accepts a longer prefix than free-form prose, while every extra verified
position touches more experts. A single fixed length therefore trades throughput
between workloads. This scheduler makes the verification length a property of the
request and keeps the engine's fused draft pass unchanged. The public design and
current results are in ``docs/benchmarks.md``.

How (design summary in docs/production-recipe.md)
-------------------------------------------------
In this build (487ecf187, V2 GPU runner, DFlash2) the drafter always drafts
``num_speculative_tokens`` tokens per request in one fused pass, and the
scheduler decides how many of them are VERIFIED next step. Production runs
``AsyncScheduler`` (dflash is an Eagle-type method for the async check,
config/speculative.py:65-69 and config/vllm.py:1190-1239): after every
``schedule()`` its ``_update_after_schedule`` hands each running request a
read-only placeholder list ``[-1] * num_spec_tokens_to_schedule``
(async_scheduler.py:16, 23-25, 44); the next ``schedule()`` schedules
``len(request.spec_token_ids)`` drafts for that request (scheduler.py:660-671)
and the V2 runner verifies exactly that many (variable per request). So this
subclass of ``AsyncScheduler`` overrides ``_update_after_schedule``: after the
base has handed out its placeholders, it replaces them per request with
``[-1] * k_req`` (or ``[-1] * k_step`` in batch-uniform mode), never longer
than what the engine drafts (``self.num_spec_tokens``). The decision lives in
the single engine-core process, so it is identical on every TP rank.

Signal: a per-request EMA of the accepted-prefix length. The drafter still
produces seven tokens, while the verifier chooses 2, 4, or 7. A fully accepted
short prefix is recorded as seven so the policy can climb back up instead of
ratcheting down. New requests run four full-length observations before they
may be shortened. This policy was introduced by MiaAI-Lab; the surrounding
async observation and placeholder plumbing remains Jacopo Nardiello's more
conservative implementation. Two modes:

* ``per-request``: each request gets its own k. A step that mixes lengths is
  non-uniform, so the target model runs the PIECEWISE CUDA
  graph on that step (FULL decode graphs need a uniform per-request token
  count, gpu/cudagraph_utils.py:89-93, worker/utils.py:671-676).
* ``batch-uniform``: one k per step. An uncalibrated or structured-output
  request pins the step to seven; otherwise the minimum request choice wins.
  Every step is uniform.

CUDA graphs and the dynamic-SD table: by default only ``num_speculative_tokens
+ 1`` tokens per request is captured as a FULL decode graph
(gpu/cudagraph_utils.py:219, 242-258), so a shorter uniform step would run
PIECEWISE. The overlay therefore also passes ``num_speculative_tokens_per_batch_size``
(``[[1,1,7],[2,2,4],[3,6,2]]``): the runner captures one decode-graph family per
K it lists (``:196-216``; 8, 5, and 3 tokens per request) and
``config/vllm.py:952-960``
keeps FULL graphs on the V2 runner. On the ASYNC path that table is live:
``AsyncScheduler`` sizes its placeholders from ``num_spec_tokens_to_schedule``
(async_scheduler.py:23-25, from scheduler.py:1203-1205), i.e. the stock async
scheduler would verify 7 drafts at batch size 1, 4 at batch size 2, and 2 at
batch sizes 3-6. This subclass
replaces the placeholder length per request right after that, so the table's
k is neutralised while all three graph families stay captured. The table also
disables the uniformity padding of resumed requests (scheduler.py:893-905):
such a request schedules one token without drafts on its first step (no
``scheduled_spec_decode_tokens`` entry, nothing to observe).

Observation gating: a request is observed only when this scheduler handed it
placeholders for the step being reported. The engine core schedules one step
ahead (core.py:661 schedule(N) → :689 appendleft → :704 pop → :719
update_from_output(N-1), batch queue of 2), so the placeholders of the step
reported at the iteration of N were handed out at the iteration of N-2: the
last THREE hand-outs are kept (``DraftedRing``). On the async path with the
table every non-empty placeholder list is ours by construction; without the
table the base pads resumed requests with ``[-1] * k`` (scheduler.py:892-905)
and the ring is what keeps those out of the EMA. The other guards are
CONSERVATIVE, not exact mirrors of the base: the request must be countable
(``generated`` non-empty or ``num_sampled_tokens_per_step == 0``), any stale
output after a preemption is skipped (``num_stale_output_tokens > 0``, whether
or not ``drop_stale_output`` would deliver it), and a KV-load failure skips the
whole step (the base recomputes only the affected requests, :1753-1756). With
async scheduling OFF the class disables itself; the sync path
(``update_draft_token_ids``) is intentionally not supported.

Installation: bind-mount this file at /opt/tp2/adaptive_k_scheduler.py, set
PYTHONPATH=/opt/tp2 and pass ``--scheduler-cls
adaptive_k_scheduler.AdaptiveKScheduler``. The complete wiring lives in
``scripts/rank-launcher.sh``.
No image file is shadowed. Env knobs (read once, in the engine-core process):

  VLLM_ADAPTIVE_K_ENABLE   1 | 0 (0 = behave exactly like the base Scheduler)
  VLLM_ADAPTIVE_K_SET      2,4,7  candidate verified draft lengths
  VLLM_ADAPTIVE_K_ALPHA    0.25   EMA weight of the newest observation
  VLLM_ADAPTIVE_K_MARGIN   1.0    headroom added before selecting a candidate
  VLLM_ADAPTIVE_K_MIN_STEPS 4     full-length observations before adapting
  VLLM_ADAPTIVE_K_SATURATE max | n  let a fully accepted short prefix climb
  VLLM_ADAPTIVE_K_MODE     batch-uniform | per-request
  VLLM_ADAPTIVE_K_LOG_EVERY 200   log the counters every N verify steps (0 = never)

The policy is pure Python and importable without vLLM (see
test_adaptive_k_policy.py); the scheduler subclass is defined only when vLLM
is importable. Any exception in the policy path is logged once and the
scheduler falls back to the base behaviour for the rest of the process.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

__all__ = ["AdaptiveKConfig", "AdaptiveKPolicy", "AdaptiveKScheduler", "should_observe",
           "placeholder_len", "DraftedRing", "filter_candidates", "assign_lengths"]

_MODES = ("per-request", "batch-uniform")
_SATURATE = ("max", "n")


def _env_int(env, name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(env, name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


@dataclass(frozen=True)
class AdaptiveKConfig:
    enabled: bool = True
    k_set: tuple[int, ...] = (2, 4, 7)
    alpha: float = 0.25
    margin: float = 1.0
    min_steps: int = 4
    saturate: str = "max"
    mode: str = "batch-uniform"
    log_every: int = 200

    def __post_init__(self) -> None:
        if not self.k_set or tuple(sorted(set(self.k_set))) != self.k_set or self.k_set[0] < 1:
            raise ValueError(f"adaptive-k: k_set must be sorted, unique, and positive, got {self.k_set}")
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"adaptive-k: need 0 < alpha <= 1, got {self.alpha}")
        if self.margin < 0.0:
            raise ValueError("adaptive-k: margin must be >= 0")
        if self.min_steps < 0:
            raise ValueError("adaptive-k: min_steps must be >= 0")
        if self.saturate not in _SATURATE:
            raise ValueError(f"adaptive-k: saturate must be one of {_SATURATE}, got {self.saturate!r}")
        if self.mode not in _MODES:
            raise ValueError(f"adaptive-k: mode must be one of {_MODES}, got {self.mode!r}")
        if self.log_every < 0:
            raise ValueError("adaptive-k: log_every must be >= 0")

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "AdaptiveKConfig":
        env = os.environ if environ is None else environ
        raw_set = (env.get("VLLM_ADAPTIVE_K_SET") or "2,4,7").strip() or "2,4,7"
        return cls(
            enabled=_env_int(env, "VLLM_ADAPTIVE_K_ENABLE", 1) != 0,
            k_set=tuple(sorted({int(v.strip()) for v in raw_set.split(",") if v.strip()})),
            alpha=_env_float(env, "VLLM_ADAPTIVE_K_ALPHA", 0.25),
            margin=_env_float(env, "VLLM_ADAPTIVE_K_MARGIN", 1.0),
            min_steps=_env_int(env, "VLLM_ADAPTIVE_K_MIN_STEPS", 4),
            saturate=(env.get("VLLM_ADAPTIVE_K_SATURATE") or "max").strip() or "max",
            mode=(env.get("VLLM_ADAPTIVE_K_MODE") or "batch-uniform").strip() or "batch-uniform",
            log_every=_env_int(env, "VLLM_ADAPTIVE_K_LOG_EVERY", 200),
        )

    def describe(self) -> str:
        return (
            f"enabled={int(self.enabled)} k_set={','.join(map(str, self.k_set))} "
            f"alpha={self.alpha} margin={self.margin} min_steps={self.min_steps} "
            f"saturate={self.saturate} mode={self.mode} log_every={self.log_every}"
        )


@dataclass
class _ReqState:
    ema: float
    k: int
    steps: int = 0


@dataclass
class AdaptiveKPolicy:
    """Per-request accepted-prefix EMA -> one of the configured draft lengths."""

    cfg: AdaptiveKConfig
    _state: dict[str, _ReqState] = field(default_factory=dict)
    counters: dict[str, int] = field(
        default_factory=lambda: {
            "observations": 0,
            "switches": 0,
            "structured_pins": 0,
        }
    )

    # -- state -------------------------------------------------------------
    def _get(self, req_id: str) -> _ReqState:
        st = self._state.get(req_id)
        if st is None:
            st = _ReqState(ema=float(self.cfg.k_set[-1]), k=self.cfg.k_set[-1])
            self._state[req_id] = st
        return st

    def _k_for(self, st: _ReqState) -> int:
        if st.steps < self.cfg.min_steps:
            return self.cfg.k_set[-1]
        import math
        target = int(math.ceil(st.ema + self.cfg.margin))
        candidates = [k for k in self.cfg.k_set if k <= target]
        return max(candidates) if candidates else self.cfg.k_set[0]

    def ema(self, req_id: str) -> float | None:
        st = self._state.get(req_id)
        return None if st is None else st.ema

    def tracked(self) -> int:
        return len(self._state)

    def signal(self, num_accepted: int, num_draft: int) -> float:
        """Accepted-prefix observation, allowing a short full hit to climb."""
        accepted = min(max(num_accepted, 0), max(num_draft, 0))
        if accepted >= num_draft and self.cfg.saturate == "max":
            return float(self.cfg.k_set[-1])
        return float(accepted)

    def observe(self, req_id: str, num_accepted: int, num_draft: int) -> float:
        if num_draft <= 0:
            return self._get(req_id).ema
        st = self._get(req_id)
        a = self.cfg.alpha
        st.ema = a * self.signal(num_accepted, num_draft) + (1.0 - a) * st.ema
        st.steps += 1
        self.counters["observations"] += 1
        return st.ema

    # -- decisions ---------------------------------------------------------
    def decide(self, req_id: str, structured: bool = False) -> int:
        st = self._get(req_id)
        if structured:
            self.counters["structured_pins"] += 1
            k = self.cfg.k_set[-1]
        else:
            k = self._k_for(st)
        if k != st.k:
            self.counters["switches"] += 1
            st.k = k
        key = f"decisions_k{k}"
        self.counters[key] = self.counters.get(key, 0) + 1
        return k

    def decide_batch(self, req_ids, structured_ids=()) -> int:
        """One uniform k; uncalibrated or structured requests pin full length."""
        req_ids = list(req_ids)
        structured_ids = set(structured_ids)
        ks = [self.decide(r, r in structured_ids) for r in req_ids]
        pinned = any(r in structured_ids or self._get(r).steps < self.cfg.min_steps for r in req_ids)
        k = self.cfg.k_set[-1] if pinned or not ks else min(ks)
        key = f"steps_uniform_k{k}"
        self.counters[key] = self.counters.get(key, 0) + 1
        return k

    def evict(self, req_id: str) -> None:
        self._state.pop(req_id, None)

    def counters_line(self) -> str:
        c = self.counters
        decisions = "/".join(
            f"k{k}={c.get(f'decisions_k{k}', 0)}" for k in self.cfg.k_set
        )
        uniform = "/".join(
            f"k{k}={c.get(f'steps_uniform_k{k}', 0)}" for k in self.cfg.k_set
        )
        return (
            f"tracked={len(self._state)} obs={c['observations']} "
            f"decisions {decisions} switches={c['switches']} structured-pins={c['structured_pins']} "
            f"uniform-steps {uniform}"
        )


def should_observe(
    req_id: str,
    drafted: "set[str] | frozenset[str]",
    num_draft: int,
    has_generated: bool,
    num_sampled_per_step: int,
    is_stale: bool = False,
    drop_stale: bool = False,
    kv_load_failed: bool = False,
) -> bool:
    """Whether one request's step result may feed the EMA (pure, testable).

    Conservative guards (they skip at least what the base skips): the request
    must be in the drafted set (placeholders handed out by this scheduler, not
    the base's ``[-1]`` resume padding), have a countable output, no stale
    output from a preemption (whether or not it would be delivered), and the
    step must carry no KV-load failure (step-global, the base is per request).
    """
    if kv_load_failed or num_draft <= 0 or req_id not in drafted:
        return False
    if not has_generated and num_sampled_per_step != 0:
        return False
    if is_stale and drop_stale:
        return False
    if is_stale:
        return False
    return True


def placeholder_len(k_req: int, engine_k: int) -> int:
    """Draft length to hand a request: never more than the engine drafts."""
    return max(0, min(int(k_req), int(engine_k)))


class DraftedRing:
    """The req_ids this scheduler handed placeholders to, over the last N hand-outs.

    Three slots on the async path: the step reported by ``update_from_output`` at
    the iteration of N received its placeholders at the iteration of N-2.
    """

    def __init__(self, slots: int = 3) -> None:
        if slots < 1:
            raise ValueError("DraftedRing needs >= 1 slot")
        self.slots = slots
        self._ring: list[frozenset[str]] = []

    def push(self, req_ids) -> None:
        self._ring.append(frozenset(req_ids))
        if len(self._ring) > self.slots:
            del self._ring[0]

    def union(self) -> frozenset[str]:
        out: frozenset[str] = frozenset()
        for s in self._ring:
            out |= s
        return out

    def __contains__(self, req_id: str) -> bool:
        return any(req_id in s for s in self._ring)

    def __len__(self) -> int:
        return len(self._ring)


def filter_candidates(items) -> list[str]:
    """req_ids that get their placeholders re-sized this step.

    ``items``: iterable of ``(req_id, is_finished, is_prefill_chunk, has_placeholders)``
    for the requests in ``scheduler_output.num_scheduled_tokens``. A request that
    the base skipped (prefill chunk) or gave no placeholders keeps whatever it
    has; finished requests are ignored.
    """
    out = []
    for req_id, is_finished, is_prefill_chunk, has_placeholders in items:
        if is_finished or is_prefill_chunk or not has_placeholders:
            continue
        out.append(req_id)
    return out


def assign_lengths(
    policy: AdaptiveKPolicy,
    req_ids,
    mode: str,
    engine_k: int,
    structured_ids=(),
) -> dict[str, int]:
    """Placeholder length per candidate request for the next step (pure)."""
    req_ids = list(req_ids)
    if not req_ids:
        return {}
    if mode == "batch-uniform":
        k_step = placeholder_len(policy.decide_batch(req_ids, structured_ids), engine_k)
        return {r: k_step for r in req_ids}
    structured_ids = set(structured_ids)
    return {
        r: placeholder_len(policy.decide(r, r in structured_ids), engine_k)
        for r in req_ids
    }


# ---------------------------------------------------------------------------
# vLLM scheduler subclass (only when vLLM is importable)
# ---------------------------------------------------------------------------
try:  # pragma: no cover - exercised only inside the vLLM image
    from vllm.logger import init_logger as _init_logger
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler as _BaseScheduler

    logger = _init_logger("vllm.adaptive_k")  # under the vllm.* hierarchy so INFO lines reach the log
    _HAVE_VLLM = True
except Exception:  # noqa: BLE001 - any import problem means "policy-only module"
    logger = logging.getLogger(__name__)
    _BaseScheduler = object  # type: ignore[assignment,misc]
    _HAVE_VLLM = False


if _HAVE_VLLM:

    class AdaptiveKScheduler(_BaseScheduler):  # type: ignore[misc,valid-type]
        """AsyncScheduler that sizes each request's draft placeholders adaptively.

        Matches vLLM 487ecf187:
          async_scheduler.py:13-17  __init__ (self._spec_token_placeholders = [-1]*num_spec_tokens)
          async_scheduler.py:19-49  _update_after_schedule(self, scheduler_output)
                                    L23-25 placeholders from num_spec_tokens_to_schedule,
                                    L26-28 skip prefill chunks, L44 request.spec_token_ids = placeholders
          scheduler.py:1681         update_from_output(self, scheduler_output, model_runner_output)
          scheduler.py:1777-1787    num_accepted = max(len(generated) - num_sampled_tokens_per_step, 0)
          scheduler.py:2327         _free_request(self, request, delay_free_blocks=False)
        The sync path (scheduler.py:2173 update_draft_token_ids) is intentionally not
        supported: the class disables itself when async scheduling is off.
        """

        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self._ak_failed = False
            self._ak_steps = 0
            self._ak_ring = DraftedRing(3)
            cfg = AdaptiveKConfig(enabled=False)
            engine_k = 0
            try:
                cfg = AdaptiveKConfig.from_env()
                async_on = bool(getattr(self.scheduler_config, "async_scheduling", False))
                engine_k = int(getattr(self, "num_spec_tokens", 0) or 0)
                if cfg.enabled and not async_on:
                    logger.error(
                        "adaptive-k: async scheduling is OFF; this class sizes AsyncScheduler "
                        "placeholders and is not meaningful on the sync path. Disabling the policy."
                    )
                    cfg = AdaptiveKConfig(**{**cfg.__dict__, "enabled": False})
                if cfg.enabled and engine_k < 1:
                    logger.error("adaptive-k: engine drafts no tokens (num_spec_tokens=%s); disabling", engine_k)
                    cfg = AdaptiveKConfig(**{**cfg.__dict__, "enabled": False})
                if cfg.enabled and engine_k < cfg.k_set[-1]:
                    clamped = tuple(sorted({min(k, engine_k) for k in cfg.k_set}))
                    logger.warning(
                        "adaptive-k: engine drafts %s tokens but k_set=%s; clamping to %s",
                        engine_k, cfg.k_set, clamped,
                    )
                    cfg = AdaptiveKConfig(**{**cfg.__dict__, "k_set": clamped})
            except Exception:  # noqa: BLE001 - never let the policy kill the engine core
                logger.exception("adaptive-k: configuration failed, running as the base AsyncScheduler")
                cfg = AdaptiveKConfig(enabled=False)
            self._ak_cfg = cfg
            self._ak = AdaptiveKPolicy(cfg)
            self._ak_engine_k = max(engine_k, 0)
            # read-only placeholder lists per draft length, shared like the base's
            self._ak_placeholders: dict[int, list[int]] = {
                k: [-1] * k for k in range(0, self._ak_engine_k + 1)
            }
            table = getattr(getattr(self.vllm_config, "speculative_config", None),
                            "num_speculative_tokens_per_batch_size", None)
            logger.info(
                "adaptive-k: AdaptiveKScheduler active (%s) engine_k=%s async=%s dynamic_sd_table=%s "
                "(table sizes are overridden per request by this scheduler)",
                cfg.describe(), self._ak_engine_k,
                getattr(self.scheduler_config, "async_scheduling", None), table,
            )

        # -- size next step's placeholders per request ---------------------
        def _update_after_schedule(self, scheduler_output) -> None:
            super()._update_after_schedule(scheduler_output)
            if not self._ak_cfg.enabled or self._ak_failed:
                return
            try:
                self._ak_size_placeholders(scheduler_output)
            except Exception:  # noqa: BLE001
                self._ak_failed = True
                logger.exception("adaptive-k: placeholder sizing failed, falling back to AsyncScheduler")

        def _ak_size_placeholders(self, scheduler_output) -> None:
            reqs = self.requests
            items = []
            for req_id in scheduler_output.num_scheduled_tokens:
                request = reqs.get(req_id)
                if request is None:
                    continue
                items.append((
                    req_id,
                    request.is_finished(),
                    bool(getattr(request, "is_prefill_chunk", False)),
                    bool(request.spec_token_ids),
                ))
            candidates = filter_candidates(items)
            self._ak_ring.push(candidates)
            structured = {
                req_id for req_id in candidates
                if bool(getattr(reqs[req_id], "use_structured_output", False))
                or getattr(reqs[req_id], "structured_output_request", None) is not None
            }
            lengths = assign_lengths(
                self._ak,
                candidates,
                self._ak_cfg.mode,
                self._ak_engine_k,
                structured,
            )
            for req_id, k in lengths.items():
                reqs[req_id].spec_token_ids = self._ak_placeholders[k]

        # -- observe acceptance (before the base mutates/frees) ------------
        def update_from_output(self, scheduler_output, model_runner_output):
            if self._ak_cfg.enabled and not self._ak_failed:
                try:
                    self._ak_observe(scheduler_output, model_runner_output)
                except Exception:  # noqa: BLE001
                    self._ak_failed = True
                    logger.exception("adaptive-k: observe failed, falling back to AsyncScheduler")
            return super().update_from_output(scheduler_output, model_runner_output)

        def _ak_observe(self, scheduler_output, model_runner_output) -> None:
            sched_spec = scheduler_output.scheduled_spec_decode_tokens
            if not sched_spec:
                return
            kv_out = getattr(model_runner_output, "kv_connector_output", None)
            kv_load_failed = bool(kv_out is not None and getattr(kv_out, "invalid_block_ids", None))
            sampled_token_ids = model_runner_output.sampled_token_ids
            req_id_to_index = model_runner_output.req_id_to_index
            num_sampled = self.num_sampled_tokens_per_step
            drafted = self._ak_ring.union()  # last three hand-outs (async runs one step ahead)
            for req_id, spec_token_ids in sched_spec.items():
                request = self.requests.get(req_id)
                if request is None or request.is_finished():
                    continue
                req_index = req_id_to_index.get(req_id)
                if req_index is None:
                    continue
                generated = sampled_token_ids[req_index] if sampled_token_ids else []
                if not should_observe(
                    req_id,
                    drafted,
                    len(spec_token_ids),
                    bool(generated),
                    num_sampled,
                    is_stale=getattr(request, "num_stale_output_tokens", 0) > 0,
                    drop_stale=bool(getattr(request, "drop_stale_output", False)),
                    kv_load_failed=kv_load_failed,
                ):
                    continue
                num_accepted = max(len(generated) - num_sampled, 0)
                self._ak.observe(req_id, num_accepted, len(spec_token_ids))
            self._ak_steps += 1
            if self._ak_cfg.log_every and self._ak_steps % self._ak_cfg.log_every == 0:
                logger.info("adaptive-k: %s", self._ak.counters_line())

        # -- eviction --------------------------------------------------------
        def _free_request(self, request, delay_free_blocks: bool = False):
            try:
                self._ak.evict(request.request_id)
            except Exception:  # noqa: BLE001
                pass
            return super()._free_request(request, delay_free_blocks)

else:  # policy-only import (tests, tooling)

    class AdaptiveKScheduler:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("AdaptiveKScheduler needs vLLM; only AdaptiveKPolicy is available here")
