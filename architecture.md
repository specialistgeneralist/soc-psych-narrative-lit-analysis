# Architecture and dependency installation reference

This reference expands the installation guidance in [README.md](README.md). All commands assume the **analysis root**: the directory containing `src/`, `requirements.txt` and `requirements-variants/`, regardless of that directory's name.

The supplied requirement profiles describe compatibility constraints. They are **not independently validated end-to-end platform guarantees or exact reproduction locks**. The guidance below preserves the existing installation advice; extracting it into this document does not establish new platform validation.

## Python and environment setup

Use **Python 3.11** as the recommended compatibility baseline. The profiles do not promise support for every Python release.

On macOS or Linux:

`python3.11 -m venv venv`

`source venv/bin/activate`

On Windows, create the environment with `py -3.11 -m venv venv`, then activate it in PowerShell with `.\venv\Scripts\Activate.ps1`.

After activation:

`python -m pip install --upgrade pip setuptools wheel`

`python -m pip install -r requirements.txt`

Always use `python -m pip` in the activated environment. If installation appears to target another environment, check:

`python -c "import sys; print(sys.executable)"`

The portable `requirements.txt` uses environment markers to select Intel-macOS-compatible Numba, llvmlite, NumPy and PyTorch versions. Other platforms resolve dependencies within the shared constraints. It requires binary wheels for Numba, llvmlite, PyArrow and PyTorch to avoid accidental LLVM, Arrow C++ or PyTorch source builds. Other dependencies, such as `annoy`, may still need a C/C++ toolchain: Xcode Command Line Tools on macOS, compiler/build tools on Linux, or suitable Visual C++ Build Tools on Windows.

## Explicit architecture profiles

Instead of the final installation command above, choose **one** matching profile. Each includes `requirements.txt`; do not install it separately or combine a profile with `requirements-lock.txt`. Do not use `--no-deps`.

| Python 3.11 environment | Installation command |
| --- | --- |
| macOS Intel / x86-64 | `python -m pip install -r requirements-variants/macos-intel.txt` |
| macOS Apple Silicon / ARM64 | `python -m pip install -r requirements-variants/macos-arm64.txt` |
| Linux x86-64 (glibc-based) | `python -m pip install -r requirements-variants/linux-x86_64.txt` |
| Linux ARM64 / aarch64 (glibc-based) | `python -m pip install -r requirements-variants/linux-aarch64.txt` |
| Windows x86-64 | `python -m pip install -r requirements-variants/windows-x86_64.txt` |

A requirements filename does not change or emulate the platform. Match the **running Python interpreter**, not just the hardware: Apple Silicon running Intel Python under Rosetta needs the Intel profile. Alpine/musl Linux, Windows ARM64, 32-bit Python and other unlisted environments are not covered.

Inspect the interpreter before choosing:

`python -c "import platform, sys; print(sys.version); print(sys.platform, platform.machine())"`

### Version bounds and binary interfaces

The Intel Mac profile selects `numba==0.61.2`, `llvmlite==0.44.0`, `torch==2.2.2`, `numpy>=1.24,<2` and `pyarrow>=17.0.0,<26`. PyTorch 2.2.2 is the last official Intel macOS wheel and requires the NumPy 1.x binary interface; PyArrow 26 dropped support for that interface.

Other explicit profiles retain `numba==0.64.0` and `llvmlite==0.46.0`, matching the pair in the original lockfile, and select `numpy>=2,<2.5` for compatibility with current PyArrow wheels. They inherit `torch>=2.4.1` to exclude older NumPy-1-only wheels. TriMAP 1.2 imports PyTorch even when the pipeline uses its legacy NumPy/Numba estimator or only loads saved coordinates.

Every profile inherits `pyarrow>=17.0.0` and `pandas>=2.2.2`; Intel adds the `<26` PyArrow cap. PyArrow 17 includes NumPy 2 compatibility fixes while also supporting NumPy 1.x. PyArrow 26 requires NumPy 2 and pandas 2.2.2 or later.

Pip selects a wheel compatible with the interpreter, OS version and architecture, potentially choosing an older release on an older OS. These are compatibility bounds, not a claim that every allowed combination has been tested. Remaining dependencies are resolved rather than completely locked.

Install the complete profile rather than only replacing NumPy: editing a requirements file does not install missing packages. With the intended Intel environment activated, run from the analysis root:

`python -m pip install -r requirements-variants/macos-intel.txt`

## Local installation checks

The following checks do not call paid APIs. First check dependency consistency and core imports:

`python -m pip check`

`python -c "import numpy, pandas, numba, llvmlite, trimap, torch, hdbscan, pyarrow, pyarrow.parquet; print('Core imports OK')"`

Then exercise the binary interfaces. Imports and `pip check` can miss ABI problems; these checks use synthetic data in memory, not research data.

### NumPy/PyTorch bridge

`python -c "import numpy as np, torch; x = np.array([1., 2.], dtype=np.float32); assert np.array_equal(torch.from_numpy(x).numpy(), x); print('NumPy/PyTorch bridge OK')"`

### Parquet round-trip

`python -c "from io import BytesIO; import numpy as np; import pandas as pd; b = BytesIO(); df = pd.DataFrame({'id': ['a', 'b'], 'vector_unit': [[1., 0.], [0., 1.]]}); df.to_parquet(b, engine='pyarrow', index=False); b.seek(0); restored = pd.read_parquet(b, engine='pyarrow'); assert restored['id'].tolist() == ['a', 'b']; assert np.array_equal(np.stack(restored['vector_unit']), np.array([[1., 0.], [0., 1.]])); print('Parquet round-trip OK')"`

Run the actual import and round-trip checks after every environment update. `pip check` can pass with an unusable binary combination; PyArrow 26 with NumPy 1.26.4 is one example. Passing these local checks is not an end-to-end pipeline validation.

## Troubleshooting wheels and Parquet

If PyArrow is missing or incompatible, reinstall the complete matching profile. For a targeted repair, activate the intended Intel environment and run from the analysis root:

`python -m pip install --only-binary=:all: "numpy>=1.24,<2" "pyarrow>=17.0.0,<26"`

Do not install unbounded PyArrow into the Intel NumPy-1 environment. Fastparquet is an alternative engine, not an additional requirement when PyArrow works.

If an installed PyArrow still fails to import, expose the underlying error directly; pandas can wrap it in a generic missing-engine message:

`python -c "import pyarrow, pyarrow.parquet; print(pyarrow.__version__)"`

If no matching Numba, llvmlite, PyArrow or PyTorch wheel is found, first check Python 3.11, the intended interpreter architecture and a supported OS version. Do not remove the binary-only guard or install generic LLVM/Arrow build packages as the default workaround. Linux wheel tags also impose a minimum glibc version, which can increase in newer package releases.

## Hashed locks and reproducibility

The existing `requirements-lock.txt` is retained **unchanged as the original hashed dependency snapshot**. It pins newer Numba/llvmlite releases without Intel macOS wheels and is not a universal installation entry point. On a compatible environment it can be installed with:

`python -m pip install --only-binary=numba,llvmlite --require-hashes -r requirements-lock.txt`

Portable requirements and architecture profiles can resolve different versions over time. Numerical-library versions and architectures can also change numerical results despite fixed random seeds. Retain the exact environment details and the package-version metadata written by the pipeline to `outputs/results/<tag>_meta.json`; see [Reproducibility in the README](README.md#reproducibility).

### Generating a target-specific hashed snapshot

Use a dedicated Python 3.11 tooling environment on the target OS and architecture. For example, on Intel macOS:

`python -m pip install pip-tools`

`python -m piptools compile --generate-hashes --output-file=requirements-lock-macos-intel-py311.txt requirements-variants/macos-intel.txt`

Substitute the matching profile and output name elsewhere. Compile on the target platform: naming a file for another platform does not make `pip-compile` cross-platform. Record tooling versions if regenerating locks reproducibly.

Test installation from every generated lock in a fresh environment on its target platform before publishing it. New target-specific hashed locks are **not supplied** with the compatibility profiles. Do not overwrite the original hashed lock with `pip freeze`, which does not preserve the `pip-compile` hash-based locking workflow.
