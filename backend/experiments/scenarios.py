"""Stress-scenario library for the benchmark networks.

A single failed pipe on a looped, well-pressurised network barely hurts anything,
so every method (including doing nothing) scores ~100% and the comparison says
nothing. These builders produce failures that actually cut supply: several pipes
at once, failures that cascade outward, a drop in source head, demand surges, and
combinations. Each builder returns a *new* network; the input is never mutated.
"""
import copy

import networkx as nx
import numpy as np

from backend.wntr_engine import inject_multi_failure, scale_node_demand

MIN_DIAMETER_M = 0.05


def eligible_pipes(wn) -> list:
    """Pipes that may fail: not tiny, and not the pipe attached directly to a source."""
    sources = set(wn.reservoir_name_list + wn.tank_name_list)
    out = []
    for pid in wn.pipe_name_list:
        p = wn.get_link(pid)
        if p.diameter < MIN_DIAMETER_M:
            continue
        if p.start_node_name in sources or p.end_node_name in sources:
            continue
        out.append(pid)
    return out or list(wn.pipe_name_list)


def random_multi_failure(wn, k: int, rng: np.random.Generator) -> list:
    pipes = eligible_pipes(wn)
    return [str(p) for p in rng.choice(pipes, size=min(k, len(pipes)), replace=False)]


def cascading_failure(wn, k: int, rng: np.random.Generator) -> list:
    """Start at a random pipe, then repeatedly fail a pipe adjacent to one already failed
    (damage spreading outward, e.g. a pressure surge rupturing neighbours)."""
    pipes = eligible_pipes(wn)
    ends = {p: (wn.get_link(p).start_node_name, wn.get_link(p).end_node_name) for p in pipes}
    failed = [str(rng.choice(pipes))]
    while len(failed) < k:
        touched = {n for p in failed for n in ends[p]}
        frontier = [p for p in pipes if p not in failed and (ends[p][0] in touched or ends[p][1] in touched)]
        if not frontier:
            break
        failed.append(str(rng.choice(frontier)))
    return failed


def reduce_source_head(wn, drop_m: float):
    """A copy with every reservoir's head lowered (drought, pump trip, upstream loss)."""
    out = copy.deepcopy(wn)
    for r in out.reservoir_name_list:
        out.get_node(r).base_head = out.get_node(r).base_head - drop_m
    return out


def scale_all_demands(wn, factor: float):
    out = copy.deepcopy(wn)
    for nid in out.junction_name_list:
        scale_node_demand(out.get_node(nid), factor)
    return out


def build_stress_scenario(wn_base, spec: dict, rng: np.random.Generator):
    """spec keys (all optional): failures=k, cascade=k, head_drop=m, surge=factor.
    Returns (stressed network, failed pipe ids)."""
    failed = []
    if spec.get("failures"):
        failed += random_multi_failure(wn_base, spec["failures"], rng)
    if spec.get("cascade"):
        failed += [p for p in cascading_failure(wn_base, spec["cascade"], rng) if p not in failed]
    wn = inject_multi_failure(wn_base, failed) if failed else copy.deepcopy(wn_base)
    if spec.get("head_drop"):
        wn = reduce_source_head(wn, spec["head_drop"])
    if spec.get("surge"):
        wn = scale_all_demands(wn, spec["surge"])
    return wn, failed
