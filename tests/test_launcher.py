import json

import pytest

import launcher


def test_new_processing_uses_the_cli_subcommand():
    cmd = launcher.sketch_command(r"C:\Program Files\Processing\Processing.exe", "--run")
    assert cmd[:2] == [r"C:\Program Files\Processing\Processing.exe", "cli"]
    assert cmd[-1] == "--run"  # must be the last argument
    assert any(a.startswith("--sketch=") and a.endswith("robotic_arm") for a in cmd)


def test_old_processing_java_is_called_directly():
    cmd = launcher.sketch_command(r"C:\processing-4.3\processing-java.exe", "--build")
    assert cmd[0].endswith("processing-java.exe")
    assert "cli" not in cmd
    assert cmd[-1] == "--build"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """No Processing on PATH, no env var, no default install locations."""
    monkeypatch.setattr(launcher, "CONFIG", tmp_path / "launcher_config.json")
    monkeypatch.setattr(launcher.shutil, "which", lambda name: None)
    monkeypatch.setattr(launcher, "processing_candidates", lambda: [])
    monkeypatch.delenv("PROCESSING", raising=False)
    return tmp_path


def test_finds_processing_from_env_and_remembers_it(isolated, monkeypatch):
    exe = isolated / "Processing.exe"
    exe.write_text("")
    monkeypatch.setenv("PROCESSING", str(exe))
    assert launcher.find_processing() == str(exe)
    assert json.loads(launcher.CONFIG.read_text())["processing"] == str(exe)


def test_saved_path_that_no_longer_exists_is_skipped(isolated, monkeypatch):
    launcher.CONFIG.write_text(json.dumps({"processing": str(isolated / "gone.exe")}))
    exe = isolated / "processing-java"
    exe.write_text("")
    monkeypatch.setattr(launcher, "processing_candidates", lambda: [str(exe)])
    assert launcher.find_processing() == str(exe)


def test_asks_the_user_when_nothing_is_found(isolated, monkeypatch):
    picked = isolated / "picked" / "Processing.exe"
    monkeypatch.setattr(launcher, "pick_processing_dialog", lambda: str(picked))
    assert launcher.find_processing() == str(picked)
    assert json.loads(launcher.CONFIG.read_text())["processing"] == str(picked)


def test_gives_up_with_a_clear_message_if_the_user_cancels(isolated, monkeypatch, capsys):
    monkeypatch.setattr(launcher, "pick_processing_dialog", lambda: None)
    with pytest.raises(SystemExit):
        launcher.find_processing()
    assert "processing.org/download" in capsys.readouterr().out


def test_stage_sketch_copies_into_a_folder_named_after_the_pde(tmp_path, monkeypatch):
    src = tmp_path / "repo"
    src.mkdir()
    (src / "robotic_arm.pde").write_text("void setup() {}")
    (src / "cube.obj").write_text("v 0 0 0")
    monkeypatch.setattr(launcher, "ROOT", src)
    monkeypatch.setattr(launcher, "BUILD", tmp_path / ".build" / "robotic_arm")
    launcher.stage_sketch()
    assert sorted(p.name for p in launcher.BUILD.iterdir()) == ["cube.obj", "robotic_arm.pde"]
