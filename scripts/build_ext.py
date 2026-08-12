#!/usr/bin/env python
"""
Build the HAIS_OP CUDA extension.

Usage:
    uv run python scripts/build_ext.py          # build only
    uv run python scripts/build_ext.py --clean  # clean + rebuild

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
ALT_GCC_PATHS = [
    Path("/opt/gcc-13/bin"),
    Path("/opt/gcc-12/bin"),
]
ALT_GCC_MAX_VERSION = 13  # CUDA 12.x supports up to GCC 13


def detect_compiler():
    """Return (CC, CXX) paths for a CUDA-compatible compiler, or (None, None)."""
    try:
        out = subprocess.check_output(
            ["g++", "--version"], stderr=subprocess.STDOUT, text=True
        )
        match = re.search(r"g\+\+.*?(\d+)\.\d+\.\d+", out)
        if match and int(match.group(1)) <= ALT_GCC_MAX_VERSION:
            return None, None  # system GCC is fine
    except Exception:
        pass

    for gcc_dir in ALT_GCC_PATHS:
        gcc = gcc_dir / "gcc-13"
        gxx = gcc_dir / "g++-13"
        if gcc.exists() and gxx.exists():
            print(f"[build_ext] Using compiler: {gxx}")
            return str(gcc), str(gxx)

    print("[build_ext] WARNING: System GCC may be too new for CUDA.", file=sys.stderr)
    print("[build_ext] Install GCC 13 or set CC/CXX environment variables.", file=sys.stderr)
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


def build():
    """Compile the CUDA extension and copy .so to project root."""
    import torch
    from torch.utils.cpp_extension import load

    cc, cxx = detect_compiler()

    # Compute RPATH so the .so finds torch's shared libs at runtime
    torch_lib = str(Path(torch.__file__).parent / "lib")

    # Apply compiler overrides via environment
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
        ]

        print("[build_ext] Compiling HAIS_OP CUDA extension...")
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        load(
            name="HAIS_OP",
            sources=sources,
            extra_include_paths=include_dirs,
            extra_cflags=["-g", "-Wno-sign-compare", "-Wno-reorder", "-Wno-unused-parameter"],
        extra_ldflags=[f"-Wl,-rpath,{torch_lib}"],
            extra_cuda_cflags=[
                "-O2",
                "-allow-unsupported-compiler",
                "-Xcompiler", "-Wno-unused-parameter",
            ],
            build_directory=str(BUILD_DIR),
            verbose=True,
            with_cuda=True,
        )
        print("[build_ext] Compilation succeeded.")
    finally:
        # Restore environment
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
    parser.add_argument("--clean", action="store_true", help="Remove build artifacts before building")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if args.clean:
        clean()

    build()


if __name__ == "__main__":
    main()
