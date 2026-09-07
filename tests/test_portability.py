import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load_dump_main():
    path = ROOT / "Data Preprocessing" / "Dumps_to_Cnosolidated" / "main.py"
    module_name = "dump_main_test"
    old_path = list(sys.path)
    old_config = sys.modules.pop("config", None)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path[:] = old_path
        sys.modules.pop("config", None)
        if old_config is not None:
            sys.modules["config"] = old_config


def test_default_processes_every_sample(tmp_path, monkeypatch):
    main = load_dump_main()
    base = tmp_path / "dumps"
    for name in ("aaa", "bbb"):
        (base / "Family" / name / "Dumps").mkdir(parents=True)
        (base / "Family" / name / "Dumps" / "one.vmem").write_bytes(b"")
    seen = []
    monkeypatch.setattr(main, "extract_regions_dumps_folder", lambda folder, output: seen.append(folder) or "timeout")
    main.process_dataset(base, tmp_path / "out", tmp_path / "consolidated")
    assert len(seen) == 2


def test_missing_dump_input_is_actionable(tmp_path):
    main = load_dump_main()
    try:
        main.process_dataset(tmp_path / "missing", tmp_path / "out", tmp_path / "consolidated")
    except FileNotFoundError as exc:
        assert "VADVIT_DUMPS_DIR" in str(exc)
    else:
        raise AssertionError("missing input should fail")


def test_resume_hash_is_inclusive(tmp_path, monkeypatch):
    main = load_dump_main()
    base = tmp_path / "dumps"
    for name in ("aaa", "bbb"):
        (base / "Family" / name / "Dumps").mkdir(parents=True)
        (base / "Family" / name / "Dumps" / "one.vmem").write_bytes(b"")
    seen = []
    monkeypatch.setattr(main, "extract_regions_dumps_folder", lambda folder, output: seen.append(folder) or "timeout")
    main.process_dataset(base, tmp_path / "out", tmp_path / "consolidated", resume_from="bbb")
    assert [Path(p).parent.name for p in seen] == ["bbb"]


def test_preprocessing_defaults_are_repo_anchored_from_any_cwd(tmp_path):
    code = "import config; print(config.CONSOLIDATED_DIR)"
    dirs = [ROOT / "Data Preprocessing" / "Dumps_to_Cnosolidated", ROOT / "Data Preprocessing" / "Consolidated_to_Grid"]
    values = []
    for directory in dirs:
        result = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, env={**os.environ, "PYTHONPATH": str(directory)}, capture_output=True, text=True, check=True)
        values.append(result.stdout.strip())
    assert values[0] == values[1] == str(ROOT / "data" / "BCCC_Consolidated_Dataset")
