"""How much memory this process may actually use, under a scheduler or not.

``psutil.virtual_memory().available`` reads ``/proc/meminfo``, which is the HOST's
memory. Under Slurm, PBS, Kubernetes or Docker the job is confined to far less, and a
window sized from the host number is sized from memory the job will never be allowed to
touch. The kill arrives at the job's limit, not the host's.

Three places state that limit, and the smallest of them wins:

* the process's OWN cgroup and every ancestor, since a limit anywhere up the tree binds
  it. Reading only the cgroup MOUNT ROOT is what containers make look correct: a private
  cgroup namespace puts the container's limit right there. A Slurm step is nested under
  ``slurmstepd.scope/job_N/step_M`` with nothing at the root, so the root read finds no
  limit and silently hands back the host. PBS's cgroup hook nests the same way.
* the scheduler's own statement of the allocation, for setups whose cgroup this process
  cannot see. Slurm exports it; PBS generally does not, and relies on its cgroup hook.
* the host, which still binds when it is the tightest of the three.

Deliberately NOT consulted: ``RLIMIT_AS``. It caps address space, not resident memory,
and this workload's virtual size runs far above its RSS (allocator arenas, memory-mapped
files, one mapping set per worker). Subtracting VMS from it would refuse runs that fit.
Linux does not enforce ``RLIMIT_RSS`` at all.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

__all__ = ["MemoryBudget", "job_memory_headroom"]

# (limit, usage, stat, reclaimable-key) for cgroup v2 and v1.
_V2 = ("memory.max", "memory.current", "memory.stat", "inactive_file")
_V1 = ("memory.limit_in_bytes", "memory.usage_in_bytes", "memory.stat",
       "total_inactive_file")

# v1 writes a number near 2**63 where v2 writes the word "max".
_UNLIMITED = 2**62

MemoryBudget = tuple[int, str]


def _read_int(path: Path) -> int | None:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _reclaimable(stat_path: Path, key: str) -> int:
    """Page cache the kernel would hand back rather than kill for.

    Usage counts it, and this workload streams forcing and TimeSlice files through it,
    so a long-lived job drifts toward usage == limit while holding gigabytes that are
    free for the asking. Subtracting it is what the working-set readings do.
    """
    try:
        fields = stat_path.read_text().split()
        return int(fields[fields.index(key) + 1])
    except (OSError, ValueError, IndexError):
        return 0


def _own_cgroup(proc_cgroup: Path) -> tuple[str | None, str | None]:
    """This process's v2 path and v1 memory path, from /proc/self/cgroup.

    v2 lines look like ``0::/system.slice/foo.scope``; v1 memory lines like
    ``9:memory:/slurm/uid_0/job_42/step_0``, and the controller field may list several
    controllers together, as in ``9:cpu,memory:/...``.
    """
    v2 = v1 = None
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return None, None
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        _, controllers, path = parts
        if controllers == "":
            v2 = path
        elif "memory" in controllers.split(","):
            v1 = path
    return v2, v1


def _tree_headroom(base: Path, rel: str, names: tuple[str, str, str, str]) -> int | None:
    """Smallest remaining budget over this cgroup and its ancestors, or None.

    A limit at ANY level binds the process, so the effective headroom is the minimum,
    not the one at the leaf.
    """
    limit_name, usage_name, stat_name, cache_key = names
    parts = [p for p in rel.strip("/").split("/") if p]
    best: int | None = None
    for depth in range(len(parts), -1, -1):
        level = base.joinpath(*parts[:depth])
        limit = _read_int(level / limit_name)
        if limit is None or limit >= _UNLIMITED:
            continue                       # no limit here, or v2's literal "max"
        used = _read_int(level / usage_name) or 0
        used -= _reclaimable(level / stat_name, cache_key)
        room = max(0, limit - used)
        best = room if best is None else min(best, room)
    return best


def _scheduler_budget(environ: Mapping[str, str], resident: int) -> int | None:
    """The allocation the scheduler says it gave this job, minus what is already used.

    Slurm exports it in MB, per node or per CPU. PBS does not export a memory figure
    reliably and enforces through its cgroup hook, which the cgroup walk above covers.
    """
    total_mb: float | None = None
    per_node = environ.get("SLURM_MEM_PER_NODE")
    per_cpu = environ.get("SLURM_MEM_PER_CPU")
    try:
        if per_node:
            total_mb = float(per_node)
        elif per_cpu:
            cpus = float(environ.get("SLURM_CPUS_ON_NODE") or 1)
            total_mb = float(per_cpu) * cpus
    except ValueError:
        return None
    if not total_mb or total_mb <= 0:
        return None
    return max(0, int(total_mb * 1024 * 1024) - resident)


def job_memory_headroom(
    host_available: int,
    resident: int = 0,
    *,
    cgroup_root: Path = Path("/sys/fs/cgroup"),
    proc_cgroup: Path = Path("/proc/self/cgroup"),
    environ: Mapping[str, str] | None = None,
) -> MemoryBudget:
    """Bytes this process may use, and the name of whatever set that.

    The name is for the log: an operator who sees a window sized smaller than they
    expected needs to know whether it was the host, their cgroup, or their batch
    allocation that decided it.
    """
    env = os.environ if environ is None else environ
    budgets: list[tuple[int, str]] = [(max(0, host_available), "host")]

    v2_path, v1_path = _own_cgroup(proc_cgroup)
    # Fall back to the mount root when /proc/self/cgroup is unreadable, which keeps a
    # namespaced container working even without procfs.
    for rel, base, names in (
        (v2_path if v2_path is not None else "/", cgroup_root, _V2),
        (v1_path if v1_path is not None else "/", cgroup_root / "memory", _V1),
    ):
        room = _tree_headroom(base, rel, names)
        if room is not None:
            budgets.append((room, "cgroup"))

    scheduled = _scheduler_budget(env, resident)
    if scheduled is not None:
        budgets.append((scheduled, "scheduler"))

    return min(budgets, key=lambda pair: pair[0])
