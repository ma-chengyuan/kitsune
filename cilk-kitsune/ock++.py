#!/usr/bin/env python3

# A Frankenstein compiler: merge OpenCilk frontend with Kitsune backend

import atexit
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HELP = """
OpenCilk-Kitsune compiler wrapper.

Usage:
    ock++ [options] <input file>

Options:
    Those you would normally pass to clang++.

Environment:
    OPENCILK_PATH   Compatible OpenCilk build or install prefix
    KITSUNE_PATH    Kitsune build prefix
    BUILD_DIR       Default parent directory for the opencilk and kitsune prefixes
"""

ARTIFACT_DIR = Path(__file__).resolve().parent


def require_tool(prefix: Path, executable: str, variable: str) -> Path:
    path = prefix / "bin" / executable
    if not path.is_file() or not os.access(path, os.X_OK):
        raise SystemExit(
            f"Missing or non-executable tool at {path}. Set {variable} to the "
            "toolchain prefix or follow cilk-kitsune/ARTIFACT.md."
        )
    return path


# Define color codes for different log levels
class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors, timestamps and fixed width to log messages."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset to default
    }

    def format(self, record):
        # Save the original format
        orig_format = self._style._fmt
        levelname = record.levelname
        levelname_fixed = f"{levelname:<8}"
        if levelname in self.COLORS:
            colored_levelname = (
                f"{self.COLORS[levelname]}{levelname_fixed}{self.COLORS['RESET']}"
            )
            self._style._fmt = orig_format.replace("%(levelname)s", colored_levelname)
        result = super().format(record)
        self._style._fmt = orig_format
        return result


# Setup initial logging configuration with default level
handler = logging.StreamHandler(sys.stdout)
formatter = ColoredFormatter(
    fmt="[%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.WARNING)
logger.addHandler(handler)


class Invocation:
    def __init__(self) -> None:
        build = Path(os.environ.get("BUILD_DIR") or ARTIFACT_DIR.parent / "build")
        self.cilk_path = Path(
            os.environ.get("OPENCILK_PATH") or build / "opencilk"
        ).absolute()
        self.kitsune_path = Path(
            os.environ.get("KITSUNE_PATH") or build / "kitsune"
        ).absolute()
        self.kitsune_lib = self.kitsune_path / "lib/clang/19/lib"
        self.kitsune_cilk_lib = self.kitsune_lib / "x86_64-unknown-linux-gnu"

    clang_args: list[str] = []
    opt_args: list[str] = []
    llc_args: list[str] = []
    linker_args: list[str] = []

    input_file: Path | None = None
    object_files: list[Path] = []
    temp_dir: Path | None = None
    temp_dir_cleanup = lambda: None
    verbosity: int = 0
    debug: bool = False
    opt_level: int = 1

    cc: str = "clang++"
    replace_malloc: bool = True

    @staticmethod
    def parse_from_args() -> "Invocation":
        ret = Invocation()

        args = sys.argv[1:]
        if len(args) and (args[0].endswith("clang") or args[0].endswith("clang++")):
            ret.cc = Path(args.pop(0)).name

        while args:
            arg = args.pop(0)
            # Frontend options that are standalone
            if any(
                arg.startswith(p)
                for p in [
                    "-std=",
                    "-fopencilk",
                    "-fdiagnostics-",
                    "-ffile-prefix-map",
                    "-Rpass-analysis",
                    "-fvisibility",
                ]
            ) or arg in [
                "-E",
                "-C",
                "-dM",
                "-dD",
                "-fno-exceptions",
                "-fno-rtti",
                "-MD",
                "-fPIE",
                "-fPIC",
            ]:
                ret.clang_args.append(arg)
            elif arg.startswith("-Wl") or arg in ["-shared", "-static"]:
                ret.linker_args.append(arg)
            elif arg == "-Xlinker":
                ret.linker_args.append(arg)
                ret.linker_args.append(args.pop(0))
            elif arg == "-g":
                ret.clang_args.append(arg)
                ret.linker_args.append(arg)
            elif any(arg.startswith(p) for p in ["-I", "-D", "-U", "-W", "-x"]):
                ret.clang_args.append(arg)
                if len(arg) == 2:
                    ret.clang_args.append(args.pop(0))
            elif any(arg.startswith(p) for p in ["-MF", "-MT"]):
                ret.clang_args.append(arg)
                if len(arg) == 3:
                    ret.clang_args.append(args.pop(0))
            elif any(
                arg.startswith(p) for p in ["-include", "-isystem", "-cxx-isystem"]
            ):
                ret.clang_args.append(arg)
                ret.clang_args.append(args.pop(0))
            elif any(arg.startswith(p) for p in ["-march", "-mcpu"]):
                ret.llc_args.append(arg)
                if "=" not in arg:
                    ret.llc_args.append(args.pop(0))
            elif arg == "-fomit-frame-pointer":
                ret.llc_args.append("--frame-pointer=none")
            elif arg == "-fno-omit-frame-pointer":
                ret.llc_args.append("--frame-pointer=all")
            elif any(arg.startswith(p) for p in ["-L", "-l"]):
                ret.linker_args.append(arg)
                if len(arg) == 2:
                    ret.linker_args.append(args.pop(0))
            elif arg == "-c":
                ret.linker_args.append(arg)
            elif arg.startswith("-o"):
                output_path = Path(args.pop(0) if len(arg) == 2 else arg[2:]).resolve()
                ret.linker_args += ["-o", output_path]
            elif arg.startswith("-save-temps"):
                temp_dir = args.pop(0) if "=" not in arg else arg.split("=")[1]
                ret.temp_dir = Path(temp_dir).resolve()
            elif arg.startswith("-v") and set(arg[1:]) == {"v"}:
                ret.verbosity = len(arg) - 1
            elif arg.startswith("-O"):
                ret.opt_level = int(args.pop(0) if len(arg) == 2 else arg[2:])
                if ret.opt_level == 0:
                    logging.warning("Warning: -O0 is not supported")
                elif ret.opt_level < 0 or ret.opt_level > 3:
                    raise ValueError(f"Invalid optimization level: {ret.opt_level}")
            elif arg.startswith("--cuabi"):
                ret.opt_args.append(arg)
            elif arg == "--debug":
                ret.debug = True
            elif not arg.startswith("-"):
                if (
                    arg.endswith(".o")
                    or arg.endswith(".a")
                    or re.match(r".*\.so(\.\d+)*$", arg)
                ):
                    ret.object_files.append(Path(arg))
                else:
                    assert ret.input_file is None, "Input file already specified"
                    ret.input_file = Path(arg)
            elif arg == "--help":
                print(HELP)
                sys.exit(0)
            elif arg == "--version":
                # Just run clang++ --version
                ret._run_command(
                    [require_tool(ret.cilk_path, "clang++", "OPENCILK_PATH"), "--version"],
                    "OpenCilk compiler version",
                )
                sys.exit(0)
            else:
                raise ValueError(f"Unknown option: {arg}")

        if ret.input_file is None and not ret.object_files:
            print(HELP)
            raise ValueError("Input file not specified")

        if "-fPIC" not in ret.clang_args and "-fPIE" not in ret.clang_args:
            ret.clang_args.append("-fPIC")

        return ret

    def setup_logging(self) -> None:
        """Set up logging based on command line arguments."""
        # Determine log level
        if self.debug:
            log_level = logging.DEBUG
        elif self.verbosity >= 2:
            log_level = logging.INFO
        elif self.verbosity == 1:
            log_level = logging.WARNING
        else:
            log_level = logging.ERROR

        # Apply the new log level
        logger = logging.getLogger()
        logger.setLevel(log_level)

        # Log the selected level
        logging.debug(f"Logging level set to {logging.getLevelName(log_level)}")

    def setup_temp_dir(self) -> None:
        """Create and return the intermediate directory path."""
        prefix = f"{self.input_file.stem}_" if self.input_file else "ock_"
        if self.temp_dir is None:
            self.temp_dir = Path(tempfile.mkdtemp(prefix=prefix))

            def cleanup_temp_dir():
                if not self.temp_dir.exists():
                    return
                try:
                    shutil.rmtree(self.temp_dir)
                    logging.info(f"Cleaned up temporary directory at {self.temp_dir}")
                except Exception as e:
                    logging.warning(f"Failed to clean up temporary directory: {e}")

            self.temp_dir_cleanup = atexit.register(cleanup_temp_dir)
            logging.debug(f"Created intermediate directory at {self.temp_dir}")

        # Create the directory if it doesn't exist
        self.temp_dir.mkdir(exist_ok=True, parents=True)
        # Clean up the directory
        to_remove = sum(
            [
                list(self.temp_dir.glob(glob))
                for glob in ["*.ll", "*.s", "*.kitsune", "*.ptx", "*.o"]
            ],
            [],
        )
        for file in to_remove:
            if file.is_dir():
                shutil.rmtree(file)
            else:
                file.unlink()
        logging.info(f"Cleaned up intermediate directory at {self.temp_dir}")

    def check_input_file(self) -> None:
        if not self.input_file.is_file():
            logging.error(f"Input file '{self.input_file}' not found")
            sys.exit(1)
        # Resolve the input file path because the compilers will be invoked with
        # a different working directory.
        self.input_file = self.input_file.resolve()

    @property
    def llvm_ir_output_file(self) -> Path:
        assert self.input_file is not None, "Input file not specified"
        return self.temp_dir / f"{self.input_file.stem}.ll"

    def generate_llvm_ir(self) -> None:
        """Generate LLVM IR using OpenCilk's clang++."""

        self._run_command(
            [
                require_tool(self.cilk_path, self.cc, "OPENCILK_PATH"),
                self.input_file,
                # Disable some aggressive optimizations --- they should be done after
                # code is flipped to device side
                "-fno-unroll-loops",
                "-fno-vectorize",
                "-fno-slp-vectorize",
                # Tell OpenCilk frontend to emit reducer intrinsics
                "-fopencilk-kitsune",
                f"-O{self.opt_level}",
                "-S",
                "-emit-llvm",
                "-o",
                self.llvm_ir_output_file,
                *self.clang_args,
                # Keep detach/sync and reducer lookups for Kitsune's lowering pass.
                "-ftapir=none",
            ],
            f"OpenCilk compiler dumped IR to {self.llvm_ir_output_file}",
            # cwd=self.temp_dir,
        )

    @property
    def lowered_llvm_ir_output_file(self) -> Path:
        assert self.input_file is not None, "Input file not specified"
        return self.temp_dir / f"{self.input_file.stem}.kitsune.ll"

    @property
    def lowered_llvm_ir_output_file_aux_1(self) -> Path:
        assert self.input_file is not None, "Input file not specified"
        return self.temp_dir / f"{self.input_file.stem}.kitsune.aux.1.ll"

    @property
    def lowered_llvm_ir_output_file_aux_2(self) -> Path:
        assert self.input_file is not None, "Input file not specified"
        return self.temp_dir / f"{self.input_file.stem}.kitsune.aux.2.ll"

    def run_kitsune_lowering(self) -> None:
        """Run Kitsune's opt for IR lowering."""

        opt = require_tool(self.kitsune_path, "opt", "KITSUNE_PATH")
        opencilk_bitcode = self.kitsune_cilk_lib / "libopencilk-abi.bc"
        kitcuda_bitcode = self.kitsune_lib / "kitcuda.bc"
        for bitcode in (opencilk_bitcode, kitcuda_bitcode):
            if not bitcode.is_file():
                raise SystemExit(
                    f"Missing Kitsune runtime bitcode at {bitcode}. Check KITSUNE_PATH "
                    "or run cilk-kitsune/build-artifact.sh."
                )

        # TODO: re-introduce vectorization after tapir lowering

        debug_flags = [
            "--debug-only=cuabi",
            "--verify-each",
            "--cuabi-keep-files",
            "--pass-remarks-analysis=loop-spawning",
        ]

        self._run_command(
            [
                opt,
                f"-passes=tapir-lowering<O{self.opt_level}>,tapir-memops-replacement",
                # These flags are for lowering to OpenCilk
                "--tapir-target=opencilk",
                "--use-opencilk-runtime-bc",
                f"--opencilk-runtime-bc-path={opencilk_bitcode}",
                "--use-kitcuda-runtime-bc",
                f"--kitcuda-runtime-bc-path={kitcuda_bitcode}",
                "--cuabi-embed-ptx",
                # *debug_flags,
                "-S",
                self.llvm_ir_output_file,
                "-o",
                self.lowered_llvm_ir_output_file,
                *self.opt_args,
            ],
            f"Kitsune compiler lowered IR to {self.lowered_llvm_ir_output_file}",
            cwd=self.temp_dir,
        )

        return
        # For precise diffing between Kitsune and OpenCilk lowering, to catch
        # bug fixes not yet backported to Kitsune.
        for p, out in [
            (self.cilk_path, self.lowered_llvm_ir_output_file_aux_1),
            (self.kitsune_path, self.lowered_llvm_ir_output_file_aux_2),
        ]:
            _run_command(
                [
                    f"{p}/bin/opt",
                    f"-passes=tapir-lowering<O{self.opt_level}>",
                    # These flags are for lowering to OpenCilk
                    "--tapir-target=opencilk",
                    "--use-opencilk-runtime-bc",
                    f"--opencilk-runtime-bc-path={self.kitsune_path}/lib/clang/19/lib/x86_64-unknown-linux-gnu/libopencilk-abi.bc",
                    "--debug-abi-calls",
                    "-S",
                    self.llvm_ir_output_file,
                    "-o",
                    out,
                    *self.opt_args,
                ],
                f"Kitsune compiler lowered IR to {out}",
                cwd=self.temp_dir,
            )
        # Compare the two files
        if (
            self.lowered_llvm_ir_output_file_aux_2.read_text().strip()
            != self.lowered_llvm_ir_output_file_aux_1.read_text().strip()
        ):
            print("IRs differ")
            print(self.lowered_llvm_ir_output_file_aux_1)
            print(self.lowered_llvm_ir_output_file_aux_2)
            atexit.unregister(self.temp_dir_cleanup)

    @property
    def kitmalloc_path(self) -> Path:
        return ARTIFACT_DIR / "kitcuda_malloc.cpp"

    def compile_executable(self) -> None:
        """Compile the assembly to an executable."""

        # The compilation follows a 3-step process.
        # 1. Compile all the .o files (or source files) into a single .o file,
        # where malloc, free and C++ new/delete are wrapped to use kitcuda mallocs.
        # 2. The .o file is then edited to localize the overridden symbols.
        # 3. Link the .o file with the CUDA runtime, the OpenCilk runtime, and
        # the Kitsune runtime.
        # The three-step process, especially step 2, is necessary to avoid
        # accidentally overriding mallocs in the runtimes (which would cause
        # infinite recursion).

        input_files = (
            ([self.lowered_llvm_ir_output_file] if self.input_file is not None else [])
            + list(self.object_files)
        )

        cuda_home = Path(os.environ.get("CUDA_HOME") or "/usr/local/cuda").absolute()

        self._run_command(
            [
                require_tool(self.kitsune_path, "kit++", "KITSUNE_PATH"),
                *input_files,
                *([self.kitmalloc_path] if self.replace_malloc else []),
                f"-O{self.opt_level}",
                *self.linker_args,
                # Link in CUDA runtime
                f"-L{cuda_home}/lib64",
                f"-Wl,-rpath,{cuda_home}/lib64",
                "-lcudart",
                # Link in OpenCilk runtime
                f"-L{self.kitsune_cilk_lib}",
                f"-Wl,-rpath,{self.kitsune_cilk_lib}",
                "-lopencilk",
                "-lopencilk-personality-cpp",
                # Link in Kitsune runtime
                f"-L{self.kitsune_lib}",
                f"-Wl,-rpath,{self.kitsune_lib}",
                "-lkitrt",
            ],
            f"Kitsune compiler compiled to binary",
            print_cmd=True,
        )

    def compile_object(self) -> None:
        """Compile the assembly to an object file."""
        output_file = None
        idx = 0
        while idx < len(self.linker_args):
            arg = self.linker_args[idx]
            if arg == "-o" and idx + 1 < len(self.linker_args):
                output_file = Path(self.linker_args[idx + 1])
                break
            idx += 1
        assert output_file is not None

        self._run_command(
            [
                require_tool(self.kitsune_path, "kitcc", "KITSUNE_PATH"),
                f"-O{self.opt_level}",
                self.lowered_llvm_ir_output_file,
                *self.linker_args,
            ],
            f"Kitsune compiler compiled to {output_file}",
            print_cmd=True,
        )

    def run(self) -> None:
        self.setup_logging()
        logging.info(f"clang args: {self.clang_args}")
        logging.info(f"opt args: {self.opt_args}")
        logging.info(f"llc args: {self.llc_args}")
        logging.info(f"linker args: {self.linker_args}")
        self.setup_temp_dir()
        if self.input_file is not None:
            if "-c" in self.linker_args and self.object_files:
                raise ValueError("Object files cannot be linked with -c")
            self.check_input_file()
            self.generate_llvm_ir()
            self.run_kitsune_lowering()
            # self.generate_assembly()
            if "-c" in self.linker_args:
                self.compile_object()
            else:
                self.compile_executable()
        else:
            assert self.object_files, "No input files specified"
            self.compile_executable()

    def _run_command(
        self,
        command: list[str | Path],
        description: str,
        cwd: Path | None = None,
        print_cmd: bool = False,
    ) -> subprocess.CompletedProcess:
        """Run a shell command and handle errors."""
        command_str = " ".join(str(c) for c in command)
        if print_cmd:
            pass
            # print(command_str)
            # print()
        logging.debug(f"Running command for: {command_str}")
        if cwd:
            logging.debug(f"Working directory: {cwd}")

        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=cwd,
            )
            skip_line = 0
            for line in result.stdout.decode("utf-8").split("\n"):
                skip_line = max(0, skip_line - 1)
                line = line.rstrip()
                if "warning: inconsistent use of MD5 checksums" in line:
                    # This is a spurious warning from clang which we can safely ignore.
                    skip_line = 3  # Skip next three lines
                if skip_line > 0:
                    continue
                if line:
                    print(line)
            logging.info(description)
            return result
        except subprocess.CalledProcessError as e:
            logging.error(f"{description} failed")
            logging.error(f"Command: {command_str}")
            logging.error(f"Return code: {e.returncode}")
            logging.error(f"Output: {e.stdout.decode('utf-8')}")
            atexit.unregister(self.temp_dir_cleanup)
            sys.exit(1)


if __name__ == "__main__":
    Invocation.parse_from_args().run()
