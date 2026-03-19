from __future__ import annotations

import os
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import override

from setuptools import Extension, setup  # type: ignore[resolve]
from setuptools.command.build_ext import build_ext  # type: ignore[resolve]


class F2PyBuildExt(build_ext):  # noqa: D101
    @override
    def run(self) -> None:
        self.build_f2py_extension()
        super().run()

    def build_f2py_extension(self) -> None:
        """Build the Fortran extension using f2py."""
        project_dir = Path(__file__).resolve().parent
        fortran_dir = (project_dir / "fortran").resolve()
        build_temp = Path(self.build_temp)
        build_temp.mkdir(parents=True, exist_ok=True)

        ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
        if not ext_suffix:
            msg = "Could not determine Python extension suffix"
            raise RuntimeError(msg)

        module_name = "_multiscat_f2py"
        expected_output = build_temp / f"{module_name}{ext_suffix}"

        f77flags = f"-I{fortran_dir}"
        f90flags = f"-I{fortran_dir}"

        command = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            f"--f77flags={f77flags}",
            f"--f90flags={f90flags}",
            "fortran/multiscat_python_bindings.pyf",
            "fortran/multiscat_python_bindings.f90",
            "fortran/multiscat.f90",
            "fortran/scatsub.f90",
            "fortran/multiscat_gmres.f90",
            "fortran/diagsub.f",
            "-m",
            module_name,
        ]

        env = os.environ.copy()
        env["CFLAGS"] = (env.get("CFLAGS", "")).strip()
        env["FFLAGS"] = (env.get("FFLAGS", "")).strip()
        env["FCFLAGS"] = (env.get("FCFLAGS", "")).strip()
        env["LDFLAGS"] = (env.get("LDFLAGS", "")).strip()

        subprocess.run(command, cwd=project_dir, check=True, env=env)  # noqa: S603

        built_artifacts = sorted(project_dir.glob(f"{module_name}*.so"))
        if not built_artifacts:
            msg = "f2py build succeeded but no shared library was produced"
            raise RuntimeError(
                msg,
            )

        built_artifact = built_artifacts[0]
        shutil.copy2(built_artifact, expected_output)

        package_dir = Path(self.build_lib) / "multiscat_fortran"
        package_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(expected_output, package_dir / expected_output.name)


setup(
    ext_modules=[Extension("multiscat_fortran._multiscat_f2py", sources=[])],
    cmdclass={"build_ext": F2PyBuildExt},
)
