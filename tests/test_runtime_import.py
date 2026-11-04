"""Native imports must not rely on optional PyTorch preloading CUDA libraries."""

import os
import subprocess
import sys


def test_native_import_without_torch_or_library_path(tmp_path):
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-I", "-c", "import style3d_sim; import sys; assert 'torch' not in sys.modules"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
