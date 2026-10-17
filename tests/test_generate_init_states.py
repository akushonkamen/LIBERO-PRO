# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression tests for notebooks/generate_init_states.py (RLinf/RPent#246).

The shipped generator used to swallow every per-attempt env failure and still
write a zero- or partial-state ``.pruned_init`` archive with exit status 0,
which is how three empty init-state files reached the LIBERO-PRO release.
These tests pin the fail-loud contract: a task's archive is written only when
every attempt succeeded, and any failed attempt aborts the run with an error
naming the broken task — without leaving an empty/partial archive behind.

The env class is stubbed via ``sys.modules`` so no simulator stack is needed.
"""

import importlib.util
import pickle
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent / "notebooks" / "generate_init_states.py"
)

STATE_DIM = 47  # arbitrary fixed dim for the stub; real sim-state dim is task-dependent


def _load_script():
    spec = importlib.util.spec_from_file_location("generate_init_states", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_fake_env(monkeypatch, failure_plan):
    """Stub ``liberopro.liberopro.envs.OffScreenRenderEnv``.

    ``failure_plan`` maps a task stem to the set of 1-based attempt numbers
    that must raise, so tests cover total and partial generation failures.
    Must be called BEFORE the script module is loaded: the unfixed script
    imports the env class at module level, so the stub has to be in
    ``sys.modules`` first for both the fixed and unfixed variants to run
    against the same fake.
    Returns ``(script_module, calls)`` with the record of ``(task, attempt)``
    constructor calls.
    """

    calls = {"attempts": []}
    counters = {}

    class FakeOffScreenRenderEnv:
        def __init__(self, bddl_file_name, camera_heights=None, camera_widths=None):
            self.task = Path(bddl_file_name).stem
            attempt = counters.get(self.task, 0) + 1
            counters[self.task] = attempt
            calls["attempts"].append((self.task, attempt))
            if attempt in failure_plan.get(self.task, ()):
                raise ValueError(f"sim boom in {self.task} attempt {attempt}")
            self._state = np.full(STATE_DIM, float(attempt))

        def get_sim_state(self):
            return self._state

        def close(self):
            pass

    envs_module = types.ModuleType("liberopro.liberopro.envs")
    envs_module.OffScreenRenderEnv = FakeOffScreenRenderEnv
    monkeypatch.setitem(sys.modules, "liberopro", types.ModuleType("liberopro"))
    monkeypatch.setitem(
        sys.modules, "liberopro.liberopro", types.ModuleType("liberopro.liberopro")
    )
    monkeypatch.setitem(sys.modules, "liberopro.liberopro.envs", envs_module)
    return _load_script(), calls


def _write_bddl(bddl_dir, name):
    bddl_dir.mkdir(parents=True, exist_ok=True)
    bddl_file = bddl_dir / f"{name}.bddl"
    bddl_file.write_text("(define (problem LIBERO_Tabletop_Manipulation))")
    return bddl_file


def test_total_failure_raises_and_writes_no_archive(tmp_path, monkeypatch):
    """All attempts fail -> RuntimeError, no (empty) .pruned_init shipped."""
    bddl_dir = tmp_path / "bddl"
    _write_bddl(bddl_dir, "broken_task")
    out_dir = tmp_path / "out"
    script, _ = _install_fake_env(monkeypatch, failure_plan={"broken_task": {1, 2, 3}})

    with pytest.raises(RuntimeError) as excinfo:
        script.generate_init_states(str(bddl_dir), str(out_dir), num_inits=3)

    message = str(excinfo.value)
    assert "broken_task" in message
    assert "sim boom" in message
    assert not out_dir.exists() or not list(out_dir.glob("*.pruned_init"))


def test_partial_failure_raises_and_writes_no_archive(tmp_path, monkeypatch):
    """A partial run (2/3) is also refused: partial archives shrink the
    evaluation denominator silently (the #246 symptom class)."""
    bddl_dir = tmp_path / "bddl"
    _write_bddl(bddl_dir, "partial_task")
    out_dir = tmp_path / "out"
    script, _ = _install_fake_env(monkeypatch, failure_plan={"partial_task": {2}})

    with pytest.raises(RuntimeError) as excinfo:
        script.generate_init_states(str(bddl_dir), str(out_dir), num_inits=3)

    assert "partial_task" in str(excinfo.value)
    assert not out_dir.exists() or not list(out_dir.glob("*.pruned_init"))


def test_success_writes_full_archive(tmp_path, monkeypatch):
    """No failures -> archive written with every state, old shape preserved."""
    bddl_dir = tmp_path / "bddl"
    _write_bddl(bddl_dir, "healthy_task")
    out_dir = tmp_path / "out"
    script, calls = _install_fake_env(monkeypatch, failure_plan={})

    script.generate_init_states(str(bddl_dir), str(out_dir), num_inits=3)

    assert sorted(attempt for _, attempt in calls["attempts"]) == [1, 2, 3]
    archive_path = out_dir / "healthy_task.pruned_init"
    assert archive_path.is_file()
    with zipfile.ZipFile(archive_path) as zipf:
        states = pickle.loads(zipf.read("archive/data.pkl"))
    assert isinstance(states, np.ndarray)
    assert states.shape == (3, STATE_DIM)
    # one state per distinct attempt, all attempts present
    assert sorted(states[:, 0].tolist()) == [1.0, 2.0, 3.0]


def test_failed_task_is_reported_while_healthy_tasks_are_written(tmp_path, monkeypatch):
    """Mixed suite: healthy archives are kept, the broken task is named, and
    the run still aborts nonzero so a release cannot ship the broken task."""
    bddl_dir = tmp_path / "bddl"
    _write_bddl(bddl_dir, "broken_task")
    _write_bddl(bddl_dir, "healthy_task")
    out_dir = tmp_path / "out"
    script, _ = _install_fake_env(monkeypatch, failure_plan={"broken_task": {1, 2, 3}})

    with pytest.raises(RuntimeError) as excinfo:
        script.generate_init_states(str(bddl_dir), str(out_dir), num_inits=3)

    assert "broken_task" in str(excinfo.value)
    assert (out_dir / "healthy_task.pruned_init").is_file()
    assert not (out_dir / "broken_task.pruned_init").exists()
