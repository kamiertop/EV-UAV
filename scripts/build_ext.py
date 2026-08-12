#!/usr/bin/env python
"""
Build the HAIS_OP CUDA extension.

Usage:
    uv run python scripts/build_ext.py                        # auto-detect
    uv run python scripts/build_ext.py --gcc 13               # find GCC 13
    uv run python scripts/build_ext.py --gcc /opt/gcc-13/bin  # explicit path
    uv run python scripts/build_ext.py --clean --gcc 13       # clean + rebuild
    CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 uv run python scripts/build_ext.py

This script compiles the CUDA extension using torch.utils.cpp_extension
and copies the resulting .so to the project root, where it can be
imported as ``import HAIS_OP``. No setuptools or pip install needed.

Requirements:
    - CUDA toolkit (nvcc on PATH)
    - ninja (for parallel builds)
    - PyTorch (for CUDA headers and build utilities)
    - GCC ≤ 13 or allow-unsupported-compiler
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "lib" / "hais_ops" / "src"
BUILD_DIR = PROJECT_ROOT / "build" / "hais_ops"

# Directories scanned when --gcc N is given (newest patch version wins)
_GCC_SEARCH_DIRS = [
    "/opt",
    "/usr/local",
    "/usr",
]
_CUDA_MAX_GCC = 13  # CUDA 12.x supports up to GCC 13


def _find_gcc_version_dir(major: int) -> Path | None:
    """Find a GCC installation directory for the given major version."""
    for base in _GCC_SEARCH_DIRS:
        # e.g. /opt/gcc-13/bin, /usr/local/gcc-13/bin
        for pattern in [f"gcc-{major}", f"gcc{major}"]:
            gcc_dir = Path(base) / pattern / "bin"
            gcc = gcc_dir / f"gcc-{major}"
            gxx = gcc_dir / f"g++-{major}"
            if gcc.exists() and gxx.exists():
                return gcc_dir
    return None


def detect_compiler(gcc_spec: str | None = None):
    """Return (CC, CXX) paths for a CUDA-compatible compiler, or (None, None).

    Args:
        gcc_spec: ``"13"`` to search for GCC 13, ``"/path/to/gcc/bin"`` to
                  use an explicit directory, or ``None`` for auto-detect.
    """
    # ── explicit path ──────────────────────────────────────────
    if gcc_spec is not None:
        gcc_dir = Path(gcc_spec)
        if not gcc_dir.is_absolute():
            # Version number like "13"
            gcc_dir = _find_gcc_version_dir(int(gcc_spec))
            if gcc_dir is None:
                print(f"[build_ext] ERROR: GCC {gcc_spec} not found."
                      f" Searched: {_GCC_SEARCH_DIRS}", file=sys.stderr)
                sys.exit(1)

        # Try gcc-N / g++-N first, then bare gcc / g++
        for pair in [(f"gcc-{_CUDA_MAX_GCC}", f"g++-{_CUDA_MAX_GCC}"),
                     ("gcc", "g++")]:
            gcc = gcc_dir / pair[0]
            gxx = gcc_dir / pair[1]
            if gcc.exists() and gxx.exists():
                print(f"[build_ext] Using compiler: {gxx}")
                return str(gcc), str(gxx)

        print(f"[build_ext] ERROR: No gcc/g++ found in {gcc_dir}", file=sys.stderr)
        sys.exit(1)

    # ── auto-detect ────────────────────────────────────────────
    try:
        out = subprocess.check_output(
            ["g++", "--version"], stderr=subprocess.STDOUT, text=True
        )
        match = re.search(r"g\+\+.*?(\d+)\.\d+\.\d+", out)
        if match and int(match.group(1)) <= _CUDA_MAX_GCC:
            return None, None  # system GCC is fine
    except Exception:
        pass

    # Try known GCC directories
    for major in range(_CUDA_MAX_GCC, 10, -1):  # 13, 12, 11
        gcc_dir = _find_gcc_version_dir(major)
        if gcc_dir is not None:
            gcc = gcc_dir / f"gcc-{major}"
            gxx = gcc_dir / f"g++-{major}"
            print(f"[build_ext] Using compiler: {gxx}")
            return str(gcc), str(gxx)

    print("[build_ext] WARNING: System GCC may be too new for CUDA.", file=sys.stderr)
    print(f"[build_ext] Use --gcc 13 or set CC/CXX env vars.", file=sys.stderr)
    return None, None


def clean():
    """Remove the previous build artifacts and compiled .so."""
    if BUILD_DIR.exists():
        shutil.rmtree(BUILD_DIR)
        print(f"[build_ext] Removed {BUILD_DIR}")

    # Remove old compiled .so from project root
    for so in PROJECT_ROOT.glob("HAIS_OP*.so"):
        so.unlink()
        print(f"[build_ext] Removed {so}")


def _get_nvcc_version() -> int | None:
    """Return nvcc major version, or None."""
    try:
        out = subprocess.check_output(["nvcc", "--version"], text=True)
        m = re.search(r"release (\d+)\.", out)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def _find_cuda_home(spec: str | None = None) -> str:
    """Find CUDA toolkit root.

    Args:
        spec: ``"12.6"``, ``"12"``, ``"/usr/local/cuda-12.6"``, or None.
    """
    if spec is not None:
        spec_path = Path(spec)
        if not spec_path.is_absolute():
            # Version number → /usr/local/cuda-X.X
            spec_path = Path("/usr/local") / f"cuda-{spec}"
        if spec_path.is_dir():
            return str(spec_path)
        print(f"[build_ext] ERROR: CUDA '{spec}' not found at {spec_path}", file=sys.stderr)
        sys.exit(1)

    for p in [os.environ.get("CUDA_HOME"), os.environ.get("CUDA_ROOT"),
              "/usr/local/cuda", "/usr/local/cuda-12.6", "/usr/local/cuda-12.4",
              "/usr/local/cuda-12.2", "/usr/local/cuda-12.1", "/usr/local/cuda-11.8"]:
        if p and Path(p).is_dir():
            return p
    nvcc = shutil.which("nvcc")
    if nvcc:
        return str(Path(nvcc).resolve().parent.parent)
    return "/usr/local/cuda"


def build(gcc_spec: str | None = None, cuda_spec: str | None = None):
    """Compile the CUDA extension and copy .so to project root."""
    import torch
    from torch.utils.cpp_extension import load

    cc, cxx = detect_compiler(gcc_spec)

    cuda_home = _find_cuda_home(cuda_spec)
    nvcc_ver = _get_nvcc_version()
    std_flag = "c++17" if (nvcc_ver and nvcc_ver < 12) else "c++20"
    print(f"[build_ext] CUDA: {cuda_home}, nvcc {nvcc_ver}, -std={std_flag}")

    # Compute RPATH so the .so finds torch's shared libs at runtime
    torch_lib = str(Path(torch.__file__).parent / "lib")

    # Apply compiler overrides
    saved_env = {}
    if cc:
        saved_env["CC"] = os.environ.get("CC")
        saved_env["CXX"] = os.environ.get("CXX")
        os.environ["CC"] = cc
        os.environ["CXX"] = cxx

    try:
        sources = [
            str(SRC_DIR / "hais_ops_api.cpp"),
            str(SRC_DIR / "hais_ops.cpp"),
            str(SRC_DIR / "cuda.cu"),
        ]
        include_dirs = [
            str(SRC_DIR),
            *(str(SRC_DIR / d) for d in [
                "bfs_cluster", "cal_iou_and_masklabel", "datatype",
                "get_iou", "hierarchical_aggregation",
                "roipool", "sec_mean", "voxelize",
            ]),
            # Include CUDA headers so GCC can find cuda.h / cuda_runtime_api.h
            str(Path(cuda_home) / "include"),
        ]

        print("[build_ext] Compiling HAIS_OP CUDA extension...")
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        load(
            name="HAIS_OP",
            sources=sources,
            extra_include_paths=include_dirs,
            extra_cflags=[
                f"-std={std_flag}", "-g",
                "-Wno-sign-compare", "-Wno-reorder", "-Wno-unused-parameter",
            ],
            extra_ldflags=[f"-Wl,-rpath,{torch_lib}"],
            extra_cuda_cflags=[
                f"-std={std_flag}", "-O2",
                "-allow-unsupported-compiler",
                "-Xcompiler", "-Wno-unused-parameter",
            ],
            build_directory=str(BUILD_DIR),
            verbose=True,
            with_cuda=True,
        )
        print("[build_ext] Compilation succeeded.")
    finally:
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # Find and copy the .so to project root
    so_files = list(BUILD_DIR.glob("HAIS_OP*.so"))
    if not so_files:
        print("[build_ext] ERROR: Could not find compiled .so file.", file=sys.stderr)
        print(f"[build_ext] Build directory: {BUILD_DIR}", file=sys.stderr)
        sys.exit(1)

    so_file = so_files[0]
    dest = PROJECT_ROOT / so_file.name
    shutil.copy2(so_file, dest)
    print(f"[build_ext] Copied {so_file.name} to project root")
    print(f"[build_ext] Ready: import HAIS_OP")
    return dest


def main():
    parser = argparse.ArgumentParser(description="Build the HAIS_OP CUDA extension")
    parser.add_argument("--gcc", default=None, type=str,
                        help="GCC version (e.g. 13) or path (e.g. /opt/gcc-13/bin)")
    parser.add_argument("--cuda", default=None, type=str,
                        help="CUDA version (e.g. 12.6) or path (e.g. /usr/local/cuda-12.6)")
    parser.add_argument("--clean", action="store_true",
                        help="Remove build artifacts before building")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if args.clean:
        clean()

    build(gcc_spec=args.gcc, cuda_spec=args.cuda)


if __name__ == "__main__":
    main()
