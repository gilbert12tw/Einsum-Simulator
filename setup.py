"""
setup.py for cudaq-einsum

Builds the C++ shared library (libnvqir-einsum.so) via CMake during
pip install, and bundles it into the Python package.

Usage:
    pip install .            # build and install
    pip install -e .         # editable install (still builds C++)
    pip install .[torch]     # with PyTorch support
    pip install .[all]       # with all optional deps
"""

import os
import subprocess
import shutil
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    """Placeholder extension to trigger CMake build."""

    def __init__(self, name: str, source_dir: str = ""):
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir)


class CMakeBuild(build_ext):
    """Custom build step that runs CMake to compile the C++ library."""

    def run(self):
        # Check CMake is available
        try:
            subprocess.run(["cmake", "--version"], check=True, capture_output=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "CMake is required to build cudaq-einsum. "
                "Install it with: apt install cmake  or  conda install cmake"
            )

        for ext in self.extensions:
            self._build_cmake(ext)

    def _build_cmake(self, ext: CMakeExtension):
        cudaq_root = self._find_cudaq_root()
        if cudaq_root is None:
            raise RuntimeError(
                "Cannot find CUDA-Q installation.\n"
                "Options:\n"
                "  1. Set env var:      export CUDAQ_ROOT=/path/to/cudaq\n"
                "  2. Conda env:        conda activate cudaq-env\n"
                "  3. pip install:      pip install cuda-quantum"
            )

        build_dir = Path(self.build_temp) / ext.name
        build_dir.mkdir(parents=True, exist_ok=True)

        print(f"-- cudaq-einsum: CUDAQ_ROOT = {cudaq_root}")
        print(f"-- cudaq-einsum: build dir  = {build_dir}")

        # Run CMake configure
        cmake_args = [
            f"-DCUDAQ_ROOT={cudaq_root}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]
        subprocess.run(
            ["cmake", ext.source_dir] + cmake_args,
            cwd=str(build_dir),
            check=True,
        )

        # Build
        nproc = os.cpu_count() or 1
        subprocess.run(
            ["cmake", "--build", ".", "--", f"-j{nproc}"],
            cwd=str(build_dir),
            check=True,
        )

        # Copy compiled .so into the Python package directory so it gets
        # bundled into the wheel and is importable after installation.
        so_src = build_dir / "libnvqir-einsum.so"
        if not so_src.exists():
            raise RuntimeError(f"Build succeeded but .so not found: {so_src}")

        pkg_out_dir = Path(self.build_lib) / "cudaq_einsum"
        pkg_out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(so_src), str(pkg_out_dir / "libnvqir-einsum.so"))
        print(f"-- cudaq-einsum: bundled .so -> {pkg_out_dir / 'libnvqir-einsum.so'}")

        # Also copy einsum.yml (needed by install_cudaq_target())
        yml_src = Path(ext.source_dir) / "targets" / "einsum.yml"
        if yml_src.exists():
            shutil.copy2(str(yml_src), str(pkg_out_dir / "einsum.yml"))
            print(f"-- cudaq-einsum: bundled yml -> {pkg_out_dir / 'einsum.yml'}")

    @staticmethod
    def _find_cudaq_root() -> "str | None":
        """
        Auto-detect the CUDA-Q installation root.

        Search order:
          1. CUDAQ_ROOT environment variable
          2. Active conda environment ($CONDA_PREFIX)
          3. pip-installed cudaq (via import)
        """
        # 1. Explicit env var
        root = os.environ.get("CUDAQ_ROOT")
        if root and Path(root).is_dir():
            return root

        # 2. Conda prefix — look for libnvqir.so as a signal
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            for lib_name in ("libnvqir.so", "libcudaq.so"):
                if Path(conda_prefix, "lib", lib_name).exists():
                    return conda_prefix

        # 3. Direct site-packages inspection based on current sys.base_prefix.
        # This is needed because pip isolated builds create a new environment
        # where `import cudaq` fails since it's not installed in the isolation.
        import sys
        
        # Determine the base python installation (e.g. miniforge env)
        base_prefix = getattr(sys, "base_prefix", sys.prefix)
        # Check standard python versions (3.8-3.13)
        for minor_version in range(8, 14):
            site_pkgs = Path(base_prefix) / "lib" / f"python3.{minor_version}" / "site-packages"
            if site_pkgs.is_dir():
                for lib_dir in ["lib", "cudaq/lib", "../lib"]:
                    for lib_name in ("libnvqir.so", "libcudaq.so"):
                        if (site_pkgs / lib_dir / lib_name).exists():
                            return str((site_pkgs / lib_dir).parent.resolve())

        # Fallback trying to import directly (for non-isolated builds)
        try:
            import cudaq
            cudaq_file = Path(cudaq.__file__)
            site_packages_dir = cudaq_file.parent.parent
            if (site_packages_dir / "lib" / "libnvqir.so").exists():
                return str(site_packages_dir)
            for candidate in [cudaq_file.parent, site_packages_dir, site_packages_dir.parent]:
                for lib_dir in ["lib", "cudaq/lib", "../lib"]:
                    for lib_name in ("libnvqir.so", "libcudaq.so"):
                        if (candidate / lib_dir / lib_name).exists():
                            return str((candidate / lib_dir).parent.resolve())
        except Exception:
            pass

        return None


setup(
    # Use standard 'src' layout
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    package_data={
        "cudaq_einsum": ["*.so", "*.yml"],
    },
    # The CMakeExtension is a dummy that triggers our custom build step
    ext_modules=[CMakeExtension("cudaq_einsum", source_dir=".")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)
