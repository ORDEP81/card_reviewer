import subprocess

from card_reviewer.knowledge import doctor


def fake_runner(found: set[str]):
    def run(cmd, **kwargs):
        if cmd[0] not in found:
            raise FileNotFoundError(cmd[0])
        return subprocess.CompletedProcess(cmd, 0, stdout="2026.01.01\n", stderr="")

    return run


def test_reports_missing_binary_with_install_hint():
    checks = doctor.check_all(
        runner=fake_runner({"ffmpeg"}), has_module=lambda n: True
    )
    by_name = {c.name: c for c in checks}
    assert by_name["yt-dlp"].found is False
    assert "brew install yt-dlp" in by_name["yt-dlp"].install_hint
    assert by_name["ffmpeg"].found is True
    assert by_name["ffmpeg"].version == "2026.01.01"


def test_reports_missing_python_module():
    checks = doctor.check_all(
        runner=fake_runner({"yt-dlp", "ffmpeg"}), has_module=lambda n: False
    )
    by_name = {c.name: c for c in checks}
    assert by_name["mlx-whisper"].found is False
    assert "uv sync" in by_name["mlx-whisper"].install_hint
