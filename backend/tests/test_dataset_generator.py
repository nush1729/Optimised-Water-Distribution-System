"""Scenario generation: deterministic per seed, correct train/val/test split."""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import backend.gat.dataset_generator as dg


def _touch_pickles(directory, net, ids):
    for i in ids:
        with open(os.path.join(directory, f"{net}_{i:04d}.pkl"), "wb") as f:
            pickle.dump({}, f)


def test_split_is_80_10_10_in_seed_order_and_only_covers_existing_files(tmp_path, monkeypatch):
    monkeypatch.setattr(dg, "GENERATED_DIR", str(tmp_path))
    _touch_pickles(tmp_path, "Netx", [i for i in range(100) if i % 10 != 3])   # 10 failed scenarios
    split = dg._write_split("Netx", 0, 100)
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (72, 9, 9)
    assert split["train"] + split["val"] + split["test"] == sorted(split["train"] + split["val"] + split["test"])
    assert not any(i % 10 == 3 for i in split["train"] + split["val"] + split["test"])
    assert json.load(open(tmp_path / "Netx_split.json")) == split


def test_tiny_dataset_goes_entirely_to_train(tmp_path, monkeypatch):
    monkeypatch.setattr(dg, "GENERATED_DIR", str(tmp_path))
    _touch_pickles(tmp_path, "Netx", range(5))
    split = dg._write_split("Netx", 0, 5)
    assert split["train"] == [0, 1, 2, 3, 4] and split["val"] == [] and split["test"] == []


def test_same_seed_gives_identical_scenarios(tmp_path, monkeypatch):
    runs = []
    for tag in ("a", "b"):
        d = tmp_path / tag
        d.mkdir()
        monkeypatch.setattr(dg, "GENERATED_DIR", str(d))
        dg._generate_chunk("Net3", 4, seed_start=0)
        runs.append(d)
    files = sorted(p.name for p in runs[0].glob("Net3_*.pkl"))
    assert files and files == sorted(p.name for p in runs[1].glob("Net3_*.pkl"))
    for name in files:
        a = pickle.load(open(runs[0] / name, "rb"))
        b = pickle.load(open(runs[1] / name, "rb"))
        assert a["failed_pipe_id"] == b["failed_pipe_id"]
        assert np.allclose(a["pressure_heads_m"], b["pressure_heads_m"])


def test_scenario_file_name_and_id_follow_the_global_seed(tmp_path, monkeypatch):
    monkeypatch.setattr(dg, "GENERATED_DIR", str(tmp_path))
    dg._generate_chunk("Net3", 3, seed_start=40)      # a chunk that does not start at 0
    names = sorted(p.name for p in tmp_path.glob("Net3_*.pkl"))
    assert names and all(40 <= int(n[5:9]) < 43 for n in names)
    sc = pickle.load(open(tmp_path / names[0], "rb"))
    assert sc["scenario_id"] == int(names[0][5:9])
