"""RailNet C++ SIMD native extension auto-builder and ctypes loader."""

import ctypes
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Optional

CSRC_DIR = Path(__file__).parent.resolve()
CPP_SRC = CSRC_DIR / "rail_kernel.cpp"
H_SRC = CSRC_DIR / "rail_kernel.h"


def get_lib_filename() -> str:
    system = platform.system()
    if system == "Windows":
        return "rail_kernel.dll"
    elif system == "Darwin":
        return "librailnet_kernel.dylib"
    else:
        return "librailnet_kernel.so"


def get_lib_path() -> Path:
    return CSRC_DIR / get_lib_filename()


def _find_vswhere() -> Optional[str]:
    vswhere_candidates = [
        r"C:\Program Files (x86)\Microsoft Visual Studio\Installer\vswhere.exe",
        r"C:\Program Files\Microsoft Visual Studio\Installer\vswhere.exe",
    ]
    for p in vswhere_candidates:
        if os.path.exists(p):
            return p
    return None


def _find_msvc_cl() -> Optional[str]:
    vswhere = _find_vswhere()
    if not vswhere:
        return None
    try:
        cmd = [
            vswhere,
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ]
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
        if not out:
            return None
        vc_tools_dir = Path(out) / "VC" / "Tools" / "MSVC"
        if not vc_tools_dir.exists():
            return None
        versions = sorted(vc_tools_dir.iterdir(), reverse=True)
        for v in versions:
            cl = v / "bin" / "Hostx64" / "x64" / "cl.exe"
            if cl.exists():
                return str(cl)
    except Exception:
        pass
    return None


def compile_with_gcc_clang(compiler: str, out_path: Path) -> bool:
    cmd = [
        compiler,
        "-O3",
        "-shared",
        "-fPIC",
        "-std=c++17",
        "-mavx2",
        "-mfma",
        "-fopenmp",
        str(CPP_SRC),
        "-o",
        str(out_path),
    ]
    env = os.environ.copy()
    c_parent = str(Path(compiler).parent)
    if c_parent and c_parent not in env.get("PATH", ""):
        env["PATH"] = f"{c_parent};{env.get('PATH', '')}"

    try:
        res = subprocess.run(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode == 0 and out_path.exists()
    except Exception:
        return False


def compile_with_msvc(cl_path: str, out_path: Path) -> bool:
    cmd = [
        cl_path,
        "/O2",
        "/LD",
        "/EHsc",
        "/std:c++17",
        "/arch:AVX2",
        "/openmp",
        "/DRAILNET_EXPORTS",
        str(CPP_SRC),
        f"/Fe:{out_path}",
    ]
    try:
        res = subprocess.run(cmd, cwd=str(CSRC_DIR), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return res.returncode == 0 and out_path.exists()
    except Exception:
        return False


def build_native_kernel(force: bool = False) -> Optional[Path]:
    """Compiles rail_kernel.cpp to shared library if not present."""
    out_path = get_lib_path()
    if out_path.exists() and not force:
        return out_path

    candidates = ["g++", "clang++", "c++"]
    w64_gpp = Path(__file__).resolve().parents[2] / "w64devkit" / "bin" / "g++.exe"
    if w64_gpp.exists():
        candidates.insert(0, str(w64_gpp))

    # Try GCC / Clang first if available
    for cc in candidates:
        try:
            env = os.environ.copy()
            c_parent = str(Path(cc).parent)
            if c_parent and c_parent not in env.get("PATH", ""):
                env["PATH"] = f"{c_parent};{env.get('PATH', '')}"
            if subprocess.run([cc, "--version"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
                if compile_with_gcc_clang(cc, out_path):
                    return out_path
        except Exception:
            pass

    # Try MSVC on Windows
    if platform.system() == "Windows":
        cl = _find_msvc_cl()
        if cl and compile_with_msvc(cl, out_path):
            return out_path

    return None if not out_path.exists() else out_path


_LOADED_LIB = None


def get_native_kernel(force_rebuild: bool = False):
    """Loads and returns the ctypes CDLL handle to the compiled kernel."""
    global _LOADED_LIB
    if _LOADED_LIB is not None and not force_rebuild:
        return _LOADED_LIB

    lib_path = build_native_kernel(force=force_rebuild)
    if lib_path is None or not lib_path.exists():
        return None

    try:
        w64_bin = Path(__file__).resolve().parents[2] / "w64devkit" / "bin"
        if w64_bin.exists():
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(w64_bin))
                except Exception:
                    pass
            os.environ["PATH"] = f"{str(w64_bin)};{os.environ.get('PATH', '')}"

        dll = ctypes.CDLL(str(lib_path))

        # Setup signatures
        dll.railnet_has_avx2.argtypes = []
        dll.railnet_has_avx2.restype = ctypes.c_int

        dll.railnet_has_avx512.argtypes = []
        dll.railnet_has_avx512.restype = ctypes.c_int

        dll.railnet_get_num_threads.argtypes = []
        dll.railnet_get_num_threads.restype = ctypes.c_int

        dll.railnet_set_num_threads.argtypes = [ctypes.c_int]
        dll.railnet_set_num_threads.restype = None

        # railnet_linear_fp32
        dll.railnet_linear_fp32.argtypes = [
            ctypes.c_void_p,  # const float* x
            ctypes.c_void_p,  # const int32_t* route_ids
            ctypes.c_void_p,  # const int32_t* term_rail
            ctypes.c_void_p,  # const int8_t* term_sign
            ctypes.c_void_p,  # const float* rails
            ctypes.c_void_p,  # float* y
            ctypes.c_int64,   # out_features
            ctypes.c_int64,   # in_features
            ctypes.c_int32,   # rail_count
            ctypes.c_int32,   # max_terms
            ctypes.c_int32,   # num_threads
        ]
        dll.railnet_linear_fp32.restype = ctypes.c_int

        # railnet_linear_fp64
        dll.railnet_linear_fp64.argtypes = [
            ctypes.c_void_p,  # const double* x
            ctypes.c_void_p,  # const int32_t* route_ids
            ctypes.c_void_p,  # const int32_t* term_rail
            ctypes.c_void_p,  # const int8_t* term_sign
            ctypes.c_void_p,  # const double* rails
            ctypes.c_void_p,  # double* y
            ctypes.c_int64,   # out_features
            ctypes.c_int64,   # in_features
            ctypes.c_int32,   # rail_count
            ctypes.c_int32,   # max_terms
            ctypes.c_int32,   # num_threads
        ]
        dll.railnet_linear_fp64.restype = ctypes.c_int

        # railnet_linear_int8_w8a_float
        dll.railnet_linear_int8_w8a_float.argtypes = [
            ctypes.c_void_p,  # const float* x
            ctypes.c_void_p,  # const int32_t* route_ids
            ctypes.c_void_p,  # const int32_t* term_rail
            ctypes.c_void_p,  # const int8_t* term_sign
            ctypes.c_void_p,  # const int32_t* rails
            ctypes.c_void_p,  # float* y
            ctypes.c_float,   # float scale
            ctypes.c_int64,   # out_features
            ctypes.c_int64,   # in_features
            ctypes.c_int32,   # rail_count
            ctypes.c_int32,   # max_terms
            ctypes.c_int32,   # num_threads
        ]
        dll.railnet_linear_int8_w8a_float.restype = ctypes.c_int

        # railnet_linear_int8_w8a16
        dll.railnet_linear_int8_w8a16.argtypes = [
            ctypes.c_void_p,  # const int16_t* x
            ctypes.c_void_p,  # const int32_t* route_ids
            ctypes.c_void_p,  # const int32_t* term_rail
            ctypes.c_void_p,  # const int8_t* term_sign
            ctypes.c_void_p,  # const int32_t* rails
            ctypes.c_void_p,  # int32_t* y
            ctypes.c_int64,   # out_features
            ctypes.c_int64,   # in_features
            ctypes.c_int32,   # rail_count
            ctypes.c_int32,   # max_terms
            ctypes.c_int32,   # num_threads
        ]
        dll.railnet_linear_int8_w8a16.restype = ctypes.c_int

        _LOADED_LIB = dll
        return dll
    except Exception as e:
        return None
