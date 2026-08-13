"""In-run application of the simple-scaling streamflow DA.

Wired into the NHF (-V5) run loop and the BMI driver. Before each ``nwm_route``
call, :meth:`ScalingDA.build_usgs_df` injects the accepted gage observations into
the Muskingum-Cunge nudging override, carrying the correction DOWNSTREAM through
routing. After routing, :meth:`ScalingDA.apply_in_kernel` reads the kernel-recorded
innovation (``nudge``), reconstructs the gage background, and spreads the
correction UPSTREAM (area-scaling along reaches, flow-ratio split at confluences).
Stale-obs decay within a window follows ``da_decay_coefficient``; decay state is
NOT carried across forcing windows.

Everything here is in ``up_node_id`` space, and the gage crosswalk comes from
``network.gages`` -- the same set the execution plan splits reaches at, so the
injection always lands on a reach boundary.

Observations come from the shared ``usgs_timeslices_folder`` through
``nhd_io.get_obs_from_timeslices`` (same files, QC gate, and interpolation as
nudging and reservoir persistence), or synthetically as ``synthetic_obs_factor``
times a frozen no-DA baseline.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from troute.routing.fast_reach.scaling_da import apply_scaling_da
from troute.scaling_da import build_scaling_da_setup
from troute.scaling_da import (
    # Deliberate re-export; tests and harnesses import the roster from here.
    timeslice_station_roster as timeslice_station_roster,  # noqa: PLC0414
)
from troute.scaling_da.preprocess import TIMESLICE_GLOB_SUFFIX as _TIMESLICE_GLOB_SUFFIX

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from numpy.typing import NDArray

    from troute.AbstractNetwork import AbstractNetwork
    from troute.scaling_da import GageTree

    # One kernel result block, read POSITIONALLY: r[0] segment ids, r[1] the
    # (n_seg, nts*4) q/v/d/ql array, r[3][0] gage segs, r[9] the nudge array.
    RunResult = Sequence[Any]
    RunResults = Sequence[RunResult]

LOG = logging.getLogger("TROUTE")


def merge_injected_obs(
    injected: pd.DataFrame, existing: pd.DataFrame | None
) -> pd.DataFrame:
    """Overlay the scaling DA's rows on the existing observation frame.

    The frame is shared: the diversion DA reads its gage's row from it, so the
    injection must overlay its own rows and leave the rest intact. Surviving rows
    are reindexed onto the injected frame's columns because the kernel reads the
    frame positionally -- a column union of two grids would shift observations.
    """
    if injected.empty:
        # Nothing injected this call; do not wipe the diversion/nudging rows.
        return existing if existing is not None else injected
    if existing is None or existing.empty:
        return injected
    kept = existing.drop(index=injected.index, errors="ignore")
    if kept.empty:
        return injected
    aligned = kept.reindex(columns=injected.columns)
    # A producer on a different time grid would lose every value in this reindex
    # and silently assimilate nothing; say so.
    emptied = kept.notna().any(axis=1) & ~aligned.notna().any(axis=1)
    if emptied.any():
        LOG.warning(
            "merge_injected_obs: %d observation row(s) lost every value when "
            "reindexed onto the injected frame's time grid (e.g. %s); those rows "
            "will assimilate nothing this window.",
            int(emptied.sum()),
            list(kept.index[emptied][:5]),
        )
    return pd.concat([injected, aligned]).sort_index()


def should_seed_state(
    scaling_da: ScalingDA | None, window_index: int, n_windows: int
) -> bool:
    """Should this window's upstream spread run BEFORE the warmstate snapshot?

    True only on the FINAL window. Muskingum-Cunge's C3*qdp recurrence carries only short previous-discharge
    memory, so a
    dQ written into q0 is a one-shot pulse that inflates the source gage's
    background and debits the next window's innovation (measured: held-out bias
    -23.8% with one seeding boundary vs -29.1% with eleven). Seeding only at the
    hand-off keeps the cycling analysis bit-identical to the diagnostic arm while
    a forecast still inherits the correction.
    """
    return scaling_da is not None and window_index == n_windows - 1


class ScalingDA:
    """Consume the network's prebuilt static setup; apply the correction per loop."""

    # Class defaults so a partially built instance (tests construct via __new__)
    # behaves: the lag is always on (the lag and its reach limit are part of the
    # method), and with no dx the horizon still means zero travel time -- every
    # segment at shift 0, numerically identical to an un-shifted spread.
    max_travel_time_h: float = 48.0
    celerity_source: str = "constant"
    celerity_mps: float = 0.8
    da_decay_min: float = 120.0
    _dx = None

    def __init__(
        self,
        network: AbstractNetwork,
        params: Mapping[str, Any],
        da_params: Mapping[str, Any] | None = None,
        cpu_pool: int | None = None,
    ) -> None:
        # Travel-time lag: an upstream segment's correction is shifted by its
        # channel travel time to the gage, and the same horizon limits how far
        # upstream a correction propagates at all. Always on -- the lag and its
        # reach limit are part of the method, not a switch.
        self.max_travel_time_h = float(params.get("max_travel_time_h", 48.0))
        if not (math.isfinite(self.max_travel_time_h) and self.max_travel_time_h > 0):
            raise ValueError(
                f"streamflow_scaling_parameters.max_travel_time_h must be a finite "
                f"positive number (the travel-time lag and its reach limit are part "
                f"of the method), got {self.max_travel_time_h!r}"
            )
        # Wave celerity for the travel-time lag. With max_travel_time_h it sets
        # how far upstream a correction reaches at all, so it is a modeling
        # decision, not a numerical guard.
        self.celerity_mps = float(params.get("celerity_mps", 0.8))
        # inf passes a bare > 0: dx/inf = 0 makes every shift zero, silently
        # resurrecting the removed un-lagged mode.
        if not (math.isfinite(self.celerity_mps) and self.celerity_mps > 0):
            raise ValueError(
                f"streamflow_scaling_parameters.celerity_mps must be a finite "
                f"positive number, got {self.celerity_mps!r}"
            )
        try:
            self._dx = network.dataframe["dx"].astype(float)
        except (AttributeError, KeyError) as e:
            raise ValueError(
                "the scaling DA needs reach lengths for the travel-time lag, and the "
                f"network dataframe has no usable 'dx' column ({e})."
            ) from e
        # The edge closure (past the window end) is innovation PERSISTENCE, so it
        # decays on the same clock as every other stale observation here rather than
        # holding a constant increment forward indefinitely.
        self.da_decay_min = float((da_params or {}).get("da_decay_coefficient", 120.0))
        if not self.da_decay_min > 0:
            raise ValueError(
                f"data_assimilation_parameters.da_decay_coefficient must be positive, "
                f"got {self.da_decay_min!r}"
            )
        self.min_flow = float(params.get("min_flow_cms", 1e-6))

        # Spread time-chunking: None = auto (only above the memory budget),
        # 0 = off, N = fixed. Bit-identical either way; see _resolve_spread_chunk.
        chunk = params.get("spread_chunk_timesteps")
        self.spread_chunk_timesteps = None if chunk is None else int(chunk)
        if self.spread_chunk_timesteps is not None and self.spread_chunk_timesteps < 0:
            raise ValueError(
                f"streamflow_scaling_parameters.spread_chunk_timesteps must be >= 0 (0 disables "
                f"chunking) or unset for auto; got {chunk!r}"
            )
        self.synthetic_factor = params.get("synthetic_obs_factor", None)
        # Must be a FROZEN no-DA output dir; factor*live_model compounds under feedback.
        self.synthetic_obs_baseline = params.get("synthetic_obs_baseline", None)
        da_params = da_params or {}
        self._qc_threshold = da_params.get("qc_threshold", 1)
        self._interpolation_limit = da_params.get("interpolation_limit_min", 59)
        self._cpu_pool = cpu_pool or 1
        self._obs_cache: tuple[tuple[Any, ...], pd.DataFrame] | None = None
        self._synth_base = None  # cached frozen synthetic baseline (site -> Series)

        # Static setup (crosswalk, source set, trees): built by NHF during network
        # construction when the DA is enabled; the fallback is the SAME function,
        # for callers constructing ScalingDA against a network that did not
        # pre-build (tests, harnesses, stubs). One implementation either way.
        setup = getattr(network, "scaling_da_setup", None)
        if setup is None:
            setup = build_scaling_da_setup(network, params, da_params, cpu_pool)
        self.trees = dict(setup.trees)
        self.gage_seg = dict(setup.gage_seg)
        self.gage_fp = dict(setup.gage_fp)
        # Full crosswalk (incl. held-out/co-located sites) for evaluation harnesses.
        self._all_gage_seg = dict(setup.all_gage_seg)
        self._da_sites = list(setup.da_sites)  # fixed injected set (plan determinism)
        self._obs_sites = set(setup.obs_sites)
        self._waterbody = setup.waterbody
        self._ts_folder: Path | None = setup.ts_folder
        self.theta_default = setup.theta_default
        self.theta_by_vpu = dict(setup.theta_by_vpu)
        LOG.info(
            "scaling DA: in-kernel downstream mode -- %d gages injected",
            len(self._da_sites),
        )

    # -- per-loop helpers --------------------------------------------------
    def _assemble_q_model(
        self, run_results: RunResults, nts: int, dt: float, t0: Any
    ) -> pd.DataFrame:
        # q lives at flat columns 0::4 (q,v,d,ql interleaved per timestep)
        ids = np.concatenate([np.asarray(r[0]) for r in run_results])
        q = np.concatenate([np.asarray(r[1])[:, 0::4] for r in run_results], axis=0)
        index = pd.date_range(
            pd.Timestamp(t0) + pd.Timedelta(seconds=dt),
            periods=nts,
            freq=pd.Timedelta(seconds=dt),
            name="time",
        )
        return pd.DataFrame(q.T.astype(np.float64), index=index, columns=[int(i) for i in ids])

    def _reach_dx(self, seg: int) -> float | None:
        """One reach's length in meters, or None when it is unusable.

        A length of 0 or less is as unusable as NaN: it would yield a NEGATIVE
        shift, which the NumPy spread reads as a wrap-around index rather than
        an error.
        """
        if self._dx is None:
            return None
        dx = self._dx.get(seg)
        if dx is None or not np.isfinite(dx) or dx <= 0:
            return None
        return float(dx)

    def _beyond_horizon_steps(self, dt: float) -> float:
        """A transit guaranteed to place a reach outside the propagation horizon."""
        return self.max_travel_time_h * 3600.0 / float(dt) + 1.0

    def _tree_tau(self, tree: GageTree, dt: float) -> tuple[NDArray[np.float64], int]:
        """Travel time to the gage, in TIMESTEPS, per segment in ``seg_order``.

        ``tau(s) = tau(parent) + dx(parent) / celerity``, one forward pass along
        the BFS parent chain since parents precede children. Returns
        ``(tau, n_unusable_reaches)``.

        The PARENT's reach is charged, not the segment's own: the correction sits
        on segment j's output, so the water it describes still has to traverse
        the parent, then the parent's parent, down to the gage. Charging each
        segment its own length is off by one reach and desynchronises the
        children of a junction, whose corrections must be evaluated at one
        instant for the flow-ratio split to be consistent.
        """
        segs = np.asarray(tree.seg_order).astype("int64").ravel()
        parent = np.asarray(tree.seg_parent_idx).astype("int64").ravel()
        tau = np.zeros(segs.shape[0], dtype=np.float64)
        unusable = 0
        if self._dx is None:
            # No reach lengths AT ALL: the lag is inert, every segment at shift 0,
            # numerically identical to an un-shifted spread. This is a partially
            # built instance, not a data defect.
            return tau, 0
        for j in range(1, segs.shape[0]):
            p = int(parent[j])
            dx = self._reach_dx(int(segs[p]))
            if dx is None:
                # A reach with no usable length in a network that HAS lengths is
                # a data defect, and fails CLOSED. A transit of 0 would put the
                # reach at the gage's own instant and keep its whole subtree
                # inside the horizon, applying corrections at wrong timesteps.
                unusable += 1
                tau[j] = self._beyond_horizon_steps(dt)
                continue
            tau[j] = tau[p] + dx / (self.celerity_mps * float(dt))
        return tau, unusable

    def _build_lag(
        self, trees: Mapping[str, GageTree], dt: float
    ) -> dict[str, tuple[NDArray[np.float64], NDArray[np.int64]]]:
        """site -> (in_window, tshift) per tree, in ``seg_order``.

        A segment tau upstream produced the water the gage sees tau later, so its
        correction is shifted to ``dQ_o(t + tau)``. This is a dominant-delay
        approximation, NOT a Muskingum-Cunge inverse: MC is diffusive, so the
        exact inverse spreads one gage increment over a RANGE of upstream times.

        ``max_travel_time_h`` doubles as a localization radius: a segment beyond
        it is not corrected. That is regularization rather than physics, since MC
        has no distance at which influence abruptly becomes zero.

        Depends only on topology, reach length and the celerity constant, so it
        is computed once per run and cached.
        """
        cache = getattr(self, "_lag_cache", None)
        if cache is None:
            cache = self._lag_cache = {}
        window_steps = self.max_travel_time_h * 3600.0 / float(dt)
        out, unusable, dropped, total = {}, 0, 0, 0
        # Integer-step histogram for the summary below, bounded by the horizon:
        # concatenating every tree's tau is itself a multi-GB transient at CONUS.
        nbin = int(np.ceil(window_steps)) + 2
        hist = np.zeros(nbin, dtype=np.int64)
        for site, tree in trees.items():
            hit = cache.get(site)
            if hit is None:
                tau, bad = self._tree_tau(tree, dt)
                hit = cache[site] = (
                    (tau <= window_steps).astype(np.float64),
                    np.rint(tau).astype(np.int64),
                    bad,
                )
            inside, shift, bad = hit
            unusable += bad
            dropped += int(inside.size - inside.sum())
            total += int(inside.size)
            hist += np.bincount(
                np.clip(shift[inside > 0], 0, nbin - 1), minlength=nbin
            )
            out[site] = (inside, shift)
        if unusable:
            LOG.warning(
                "scaling DA: %d tree reach(es) have no usable length; they are held "
                "outside the propagation horizon rather than treated as zero "
                "travel time.",
                unusable,
            )
        # Percentiles off the histogram: exact for the integer shifts the kernel
        # applies, and O(bins) rather than O(segments).
        n_in = int(hist.sum())
        cum = np.cumsum(hist)
        step_h = float(dt) / 3600.0

        def _pct(q: float) -> float:
            return float(np.searchsorted(cum, q * n_in) * step_h) if n_in else 0.0

        LOG.info(
            "scaling DA: travel-time lag at %.2f m/s, %.0f h horizon -- %d of %d "
            "segment(s) beyond it (not corrected); tau within the horizon: median "
            "%.1f h, p90 %.1f h, max %.1f h (%d shift by 0 steps)",
            self.celerity_mps, self.max_travel_time_h, dropped, total,
            _pct(0.5), _pct(0.9),
            float(np.flatnonzero(hist).max() * step_h) if n_in else 0.0,
            int(hist[0]),
        )
        return out

    def _scatter_back(self, run_results: RunResults, q_corr: pd.DataFrame) -> None:
        pos = {int(c): k for k, c in enumerate(q_corr.columns)}
        corr = q_corr.to_numpy()  # [time, seg]
        for r in run_results:
            ids_r = np.asarray(r[0])
            idx = [pos[int(i)] for i in ids_r]
            r[1][:, 0::4] = corr[:, idx].T  # write corrected q back (float32 cast)

    # -- Stage A: in-kernel downstream propagation -------------------------
    def build_usgs_df(
        self, t0: Any, dt: float, nts: int, da_run: Mapping[str, Any] | None = None
    ) -> pd.DataFrame:
        """The ``usgs_df`` injected into the MC nudging override (downstream leg).

        index = gage ``up_node_id``; columns = the kernel's positional grid
        ``[t0, t0+dt, ..., t0+nts*dt]`` (column 0 seeds the IC). The injected gage
        set is FIXED across loops for execution-plan determinism; NaN where a gage
        has no obs (kernel persists/decays).
        """
        if not self._da_sites:
            return pd.DataFrame()
        full_index = pd.date_range(
            pd.Timestamp(t0), periods=nts + 1, freq=pd.Timedelta(seconds=dt), name="time"
        )
        obs = self._loop_observations(full_index, da_run)
        if obs is None or obs.shape[1] == 0:
            return pd.DataFrame()
        rows, segs = [], []
        for site in obs.columns:
            seg = self.gage_seg.get(site)
            if seg is None:
                continue
            rows.append(obs[site].to_numpy())
            segs.append(int(seg))
        if not rows:
            return pd.DataFrame()
        usgs = pd.DataFrame(
            np.asarray(rows, dtype=np.float64),
            index=pd.Index(segs, name="link", dtype="int64"),
            columns=full_index,
        )
        return usgs[~usgs.index.duplicated(keep="first")].sort_index()

    def _loop_observations(
        self, index: pd.DatetimeIndex, da_run: Mapping[str, Any] | None = None
    ) -> pd.DataFrame | None:
        """Obs (time x site) over the fixed DA-site set, aligned to *index*."""
        sites = self._da_sites
        if self.synthetic_factor is not None:
            base = self._synthetic_baseline()
            if not base:
                return None
            cols = {
                s: base[s].reindex(index).to_numpy() * float(self.synthetic_factor)
                for s in sites
                if s in base
            }
            return pd.DataFrame(cols, index=index) if cols else None
        if self._ts_folder is not None and self._obs_sites:
            want = [s for s in sites if s in self._obs_sites]
            if not want:
                return None
            df = self._read_timeslices(want, index, da_run)
            return None if df is None else df.reindex(index)
        return None

    def _read_timeslices(
        self, want: list[str], index: pd.DatetimeIndex, da_run: Mapping[str, Any] | None
    ) -> pd.DataFrame | None:
        """Site-keyed observations for this window, via t-route's shared reader.

        ``get_obs_from_timeslices`` with an identity crosswalk gives site-keyed
        rows under the same QC and interpolation every other consumer uses. The
        file list is ``build_da_sets``'s per-window list; a driver that never
        built da_sets falls back to the whole directory.
        """
        from troute import nhd_io

        folder = self._ts_folder
        if folder is None:  # _loop_observations gates on this; keep the narrow local
            return None
        # A PRESENT key is authoritative even when empty (build_da_sets chose no
        # file); only an ABSENT key falls back to directory discovery.
        if da_run is not None and "usgs_timeslice_files" in da_run:
            paths = [folder / f for f in da_run["usgs_timeslice_files"]]
        else:
            paths = sorted(folder.glob(f"*.{_TIMESLICE_GLOB_SUFFIX}"))
        paths = [p for p in paths if p.exists()]
        if not paths:
            return None
        step = index[1] - index[0] if len(index) > 1 else pd.Timedelta(seconds=300)
        # The reader resamples in whole minutes; a dt it cannot express would drop
        # or misalign observations against the kernel grid, so refuse it.
        secs = int(step.total_seconds())
        if secs < 60 or secs % 60:
            raise ValueError(
                f"scaling_da cannot read TimeSlice observations at dt={secs}s: the "
                "shared reader resamples in whole minutes. Use a dt that is a "
                "multiple of 60 s."
            )
        # Cache: the BMI driver replays one da_run across every window of an
        # update_until call. The key covers the files, the step, the site list,
        # and the resample-grid PHASE (not t0 itself -- window boundaries share
        # the phase, and keying on t0 would defeat the cache for exactly that
        # replay; an off-phase t0 is a genuine miss).
        phase = int(pd.Timestamp(index[0]).value) % (secs * 1_000_000_000)
        key = (tuple(str(p) for p in paths), step, phase, tuple(want))
        if self._obs_cache is not None and self._obs_cache[0] == key:
            return self._obs_cache[1]
        obs = nhd_io.get_obs_from_timeslices(
            crosswalk_df=pd.DataFrame({"gages": want, "site": want}),
            crosswalk_gage_field="gages",
            crosswalk_dest_field="site",
            timeslice_files=paths,
            qc_threshold=self._qc_threshold,
            interpolation_limit=self._interpolation_limit,
            frequency_secs=secs,
            t0=index[0],
            cpu_pool=self._cpu_pool,
        )
        if obs.empty:
            return None
        df = obs.transpose()  # reader returns destination x time
        idx = pd.DatetimeIndex(df.index)
        df.index = idx.tz_localize(None) if idx.tz is not None else idx
        df = df[~df.index.duplicated(keep="first")].reindex(columns=want)
        # Cache what is RETURNED, so hit and miss hand back the same column set.
        self._obs_cache = (key, df)
        return df

    def _synthetic_baseline(self):
        """Frozen open-loop baseline for synthetic runs: ``site -> flow Series``
        read once from a no-DA output dir (feature_id = fp_id)."""
        if self._synth_base is not None:
            return self._synth_base
        if not self.synthetic_obs_baseline:
            LOG.warning(
                "scaling DA: in-kernel synthetic needs 'synthetic_obs_baseline' "
                "(a no-DA output dir); none set -> no obs injected."
            )
            self._synth_base = {}
            return self._synth_base
        import xarray as xr

        frames = []
        for f in sorted(Path(self.synthetic_obs_baseline).glob("troute_output_*.nc")):
            ds = xr.open_dataset(f)
            cols = [int(x) for x in ds["feature_id"].to_numpy()]
            frames.append(
                pd.DataFrame(
                    ds["flow"].transpose("time", "feature_id").to_numpy(),
                    index=pd.DatetimeIndex(ds["time"].to_numpy()),
                    columns=cols,
                )
            )
            ds.close()
        if not frames:
            self._synth_base = {}
            return self._synth_base
        q = pd.concat(frames).sort_index()
        q = q[~q.index.duplicated()]
        self._synth_base = {
            s: q[self.gage_fp[s]] for s in self._da_sites if self.gage_fp.get(s) in q.columns
        }
        return self._synth_base

    @staticmethod
    def gather_innovation(run_results: RunResults) -> dict[int, NDArray[np.float64]]:
        """``gage segment -> per-timestep applied delta`` from the kernel's nudge.

        Exposed on its own so the driver can read window k+1's innovation and hand it
        to window k as the halo its backward shift needs. Column 0 is the initial
        condition, so it is dropped to align with the output grid.
        """
        out: dict[int, NDArray[np.float64]] = {}
        for r in run_results:
            gage_ids = np.asarray(r[3][0]).astype("int64").ravel()
            if gage_ids.size == 0:
                continue
            nud = np.asarray(r[9])
            for k, seg in enumerate(gage_ids):
                out[int(seg)] = nud[k, 1:].astype(np.float64)
        return out

    def apply_in_kernel(
        self,
        run_results: RunResults,
        nts: int,
        dt: float,
        t0: Any,
        halo: Mapping[int, NDArray[np.float64]] | None = None,
    ) -> None:
        """Post-routing pass: spread the kernel-recorded innovation UPSTREAM.

        The gage and the reach below it are already corrected in-kernel. The
        kernel's ``nudge`` (r[9]) is the applied per-timestep delta; the gage
        background is reconstructed as ``Q_analyzed - nudge`` so the confluence
        flow-ratio split reads modeled flow at the tree root.

        ``halo`` maps gage segment to the NEXT window's innovation; the backward
        shift's tail reads it past this window's end.
        """
        if not self.trees:
            return
        import os as _os
        if _os.environ.get("SCALING_DA_LEANMEM"):
            # UNSUPPORTED DIAGNOSTIC (profiling only; the supported memory control
            # is the spread_chunk_timesteps config field): release the idle loky
            # routing pool before the memory-heavy spread. Bit-identical.
            try:
                from joblib.externals.loky import get_reusable_executor

                get_reusable_executor().shutdown(wait=True)
            except Exception as _e:  # noqa: BLE001 -- best-effort memory relief
                LOG.debug("scaling DA: worker-pool release skipped (%s)", _e)
        _t = time.perf_counter()
        # 1. map kernel nudge -> gage segment.
        nudge_by_seg = self.gather_innovation(run_results)
        if not nudge_by_seg:
            LOG.info("scaling DA: in-kernel -- no gage nudges this loop; skipping spread.")
            return
        # 2. assemble q_model and gather candidate gages. Work on the numpy array +
        # a position map: per-column pandas writes on a CONUS-wide frame copy it.
        q_model = self._assemble_q_model(run_results, nts, dt, t0)
        nt = len(q_model.index)
        arr = q_model.to_numpy(dtype=np.float64, copy=True)  # [nts, N_seg]
        colpos = {int(c): i for i, c in enumerate(q_model.columns)}
        cand = {}  # site -> (seg, nudge) for gages with a non-zero applied delta
        halo_by_site: dict[str, NDArray[np.float64]] = {}
        for site in self.trees:
            seg = self.gage_seg.get(site)
            if seg is None or seg not in nudge_by_seg or seg not in colpos:
                continue
            nud = nudge_by_seg[seg]
            if nud.shape[0] != nt:
                continue
            # NOTE: a site with an all-zero OWN innovation stays a candidate. Its
            # halo can still be nonzero, and the lag reads it at t inside THIS
            # window -- gating on the window's own values made inclusion depend
            # on max_loop_size. Whether anything spreads is decided below, from
            # the concatenated array.
            cand[site] = (seg, nud)
            # The halo is the NEXT window's innovation at the same gage, so the
            # tail of this window reads real observations rather than falling
            # back to persistence.
            if halo is not None and seg in halo:
                halo_by_site[site] = np.asarray(halo[seg], dtype=np.float64)
        if not cand:
            LOG.info("scaling DA: in-kernel -- no candidate gages this loop.")
            return
        # 3. per site: splice the halo onto this window's innovation and include
        # the site if the result is nonzero anywhere. Inclusion is read off the
        # concatenated array, not off this window's own values: a site can have a
        # zero innovation here and a nonzero halo that the lag reads at a
        # timestep inside this window.
        dq_o_by_site = {}
        _nhalo = 0
        for site, (seg, nud) in cand.items():
            h = halo_by_site.get(site)
            full = nud
            if h is not None and h.size:
                full = np.concatenate([nud, h])
                _nhalo = max(_nhalo, int(h.size))
            if not np.any(full):
                continue  # nothing to spread at any reachable timestep
            # Reconstruct the gage background so the confluence flow-ratio split
            # reads modeled flow at the tree root.
            arr[:, colpos[seg]] -= nud
            dq_o_by_site[site] = full
        if not dq_o_by_site:
            LOG.info(
                "scaling DA: in-kernel -- innovations all zero everywhere the "
                "spread can read; downstream only."
            )
            return
        trees = {s: self.trees[s] for s in dq_o_by_site}
        gmap = {s: self.gage_seg[s] for s in trees}
        lag = self._build_lag(trees, dt)
        _chunk = self._resolve_spread_chunk(nt, arr.shape[1])
        # A chunk's shifts read FORWARD, so each chunk's innovation slice is
        # extended by the largest shift in play; slicing to the chunk alone would
        # make t+shift cross the boundary and read the wrong end.
        _maxshift = max((int(s.max()) for _, s in lag.values()), default=0)
        if _nhalo:
            LOG.info(
                "scaling DA: halo of %d step(s) from the next window feeds the lag's "
                "tail; %d step(s) of it are actually reached.",
                _nhalo, min(_nhalo, _maxshift),
            )
        _edge = float(np.exp(-float(dt) / (60.0 * self.da_decay_min)))
        if _chunk > 0:
            # Chunk the spread so its [nt x N] transients stay [chunk x N].
            # Timesteps are independent on this path, so the result is
            # bit-identical to the single call.
            colpos_out = {int(c): k for k, c in enumerate(q_model.columns)}
            for c0 in range(0, nt, _chunk):
                c1 = min(c0 + _chunk, nt)
                qmb = pd.DataFrame(
                    arr[c0:c1], index=q_model.index[c0:c1], columns=q_model.columns
                )
                # +_maxshift: the overlap the lag needs to read into.
                dqo = {s: dq_o_by_site[s][c0:min(c1 + _maxshift, dq_o_by_site[s].shape[0])]
                       for s in dq_o_by_site}
                qc, _ = apply_scaling_da(
                    qmb, None, gmap, trees,
                    min_flow_cms=self.min_flow, dq_o_by_site=dqo,
                    lag_by_site=lag, edge_decay=_edge,
                )
                corr = qc.to_numpy()  # [chunk_time, seg]
                for r in run_results:
                    idx = [colpos_out[int(i)] for i in np.asarray(r[0])]
                    r[1][:, 4 * c0:4 * c1:4] = corr[:, idx].T
        else:
            q_model_bg = pd.DataFrame(arr, index=q_model.index, columns=q_model.columns)
            q_corr, _dq = apply_scaling_da(
                q_model_bg,
                None,
                gmap,
                trees,
                min_flow_cms=self.min_flow,
                dq_o_by_site=dq_o_by_site,
                lag_by_site=lag,
                edge_decay=_edge,
            )
            self._scatter_back(run_results, q_corr)
        # Mean |innovation|: a corrected state should feed back a SMALLER
        # innovation next window; growth means divergent feedback.
        per_site = {s: float(np.mean(np.abs(dq_o_by_site[s][:nt]))) for s in dq_o_by_site}
        LOG.info(
            "scaling DA: upstream spread on %d gage trees, "
            "mean |innovation| %.4f cms [%.2fs]",
            len(trees),
            np.mean(list(per_site.values())),
            time.perf_counter() - _t,
        )
        # Per-site breakdown, so arms can be A/B'd on a matched site set.
        # ponytail: capped at 50 sites -- readable on Ohio, CONUS has thousands.
        if len(per_site) <= 50:
            LOG.info(
                "scaling DA: per-site |innovation| %s",
                " ".join(f"{s}={v:.4f}" for s, v in sorted(per_site.items())),
            )

    # ~0.5 GB of float64 per spread transient before auto-chunking kicks in;
    # CONUS-sized windows exceed it (with a measured swap cliff), Ohio never does.
    _SPREAD_CHUNK_BUDGET_ELEMS = 64_000_000

    def _resolve_spread_chunk(self, nt: int, n_seg: int) -> int:
        """Chunk size (timesteps) for this window's spread; 0 = no chunking.

        Priority: config field (0 = off, N = fixed), then the SCALING_DA_CHUNK env
        var (kept for the benchmark harnesses), then auto -- chunk only above the
        budget, and never under 'aligned'.
        """
        import os

        cfg = getattr(self, "spread_chunk_timesteps", None)
        if cfg is not None:
            return int(cfg)
        env = int(os.environ.get("SCALING_DA_CHUNK", "0") or "0")
        if env > 0:
            return env
        if nt * n_seg <= self._SPREAD_CHUNK_BUDGET_ELEMS:
            return 0
        chunk = max(1, self._SPREAD_CHUNK_BUDGET_ELEMS // max(n_seg, 1))
        LOG.info(
            "scaling DA: auto-chunking the upstream spread to %d timestep(s) per "
            "pass (%d segments x %d steps ~ %.1f GB per transient); bit-identical. "
            "Set scaling_da.spread_chunk_timesteps to override.",
            chunk, n_seg, nt, nt * n_seg * 8 / 1e9,
        )
        return chunk

def validate_window_envelope(
    run_sets: Sequence[Mapping[str, Any]],
    scaling_da: ScalingDA,
    default_dt: float | None = None,
) -> None:
    """Reject a run partitioning whose scaling DA results would depend on it.

    The deferred-window halo is exactly one forcing window deep, so every
    NON-FINAL window must cover the travel-time horizon or the shift's tail
    falls back to persistence. ``max_loop_size`` counts forcing FILES, so the
    realized window length in hours is only known once the run sets exist
    (sub-hourly forcing shortens it; ``stream_output_time`` enlarges it). The
    final window is exempt: it clamps at the run end regardless.
    """
    for k, run in enumerate(run_sets[:-1]):
        dt = float(run.get("dt") or default_dt or 0.0)
        nts = run.get("nts")
        if not dt or not nts:
            msg = f"validate_window_envelope: run set {k} has no usable nts/dt."
            raise ValueError(msg)
        hours = float(nts) * dt / 3600.0
        if scaling_da.max_travel_time_h > 0 and hours < scaling_da.max_travel_time_h - 1e-9:
            msg = (
                f"scaling DA: forcing window {k} is {hours:g} h, shorter than "
                f"max_travel_time_h ({scaling_da.max_travel_time_h:g} h). The "
                "deferred-window halo is one window deep, so longer shifts would "
                "fall back to persistence and the result would depend on "
                "max_loop_size, a memory knob. Note max_loop_size counts forcing "
                "FILES: with sub-hourly forcing the realized window is shorter "
                "than the count suggests. Raise max_loop_size, lower "
                "max_travel_time_h, or set max_travel_time_h: 0."
            )
            raise ValueError(msg)


def build_scaling_da(
    network: AbstractNetwork,
    supernetwork_parameters: Mapping[str, Any] | None,
    data_assimilation_parameters: Mapping[str, Any] | None,
    cpu_pool: int | None = None,
) -> ScalingDA | None:
    """Return a configured ScalingDA if enabled in the config, else None.

    The whole ``data_assimilation_parameters`` block is passed through: the
    observation folder, qc_threshold, and interpolation limit live on the parent.
    """
    da_params = data_assimilation_parameters or {}
    sda = da_params.get("streamflow_da") or {}
    if not sda.get("streamflow_scaling"):
        return None
    params = sda.get("streamflow_scaling_parameters") or {}
    return ScalingDA(network, params, da_params, cpu_pool)


def network_gage_segments(network: AbstractNetwork) -> set[int]:
    """The static gage split set handed to ``nwm_route``.

    Both drivers derive it through this one function so the cached execution plan
    never depends on a window's observations and the -V5/BMI plans split
    identically. ``network.gages`` is ``{"gages": {up_node_id: site_no}}``.
    """
    gages = (getattr(network, "gages", None) or {}).get("gages", {}) or {}
    return set(gages)
