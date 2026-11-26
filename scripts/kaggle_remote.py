"""Push private SiClink runs to Kaggle and retrieve their logs or files."""

import argparse
import base64
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "siclink-mouse-visual-reconstruction-bci-challenge"
OWNER = "ccjjllt"
TOKEN_FILE = Path.home() / ".kaggle" / "access_token_ccjjllt_check"
PROXY = "http://127.0.0.1:7897"


def run_cli(args, command):
    token = args.token_file.expanduser().read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"Empty Kaggle token file: {args.token_file}")
    environment = os.environ.copy()
    environment["KAGGLE_API_TOKEN"] = token
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        environment[variable] = args.proxy
    invocation = ["kaggle", "kernels", *command]
    if command[0] != "push":
        return subprocess.run(invocation, env=environment, check=False).returncode

    result = subprocess.run(invocation, env=environment, check=False, capture_output=True, text=True)
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    # Kaggle CLI can report a rejected push while returning exit status zero.
    rejected = any(line.lstrip().startswith("Kernel push error:")
                   for line in (result.stdout + "\n" + result.stderr).splitlines())
    return result.returncode or int(rejected)


def stage_entry(entry, includes, source_root, stage, entry_args=()):
    entry = entry.resolve(strict=True)
    destination = stage / entry.name
    if not includes and not entry_args:
        shutil.copyfile(entry, destination)
        return destination.name

    source_root = source_root.resolve(strict=True)
    sources = {path.resolve(strict=True) for path in [entry, *includes]}
    if any(path.suffix != ".py" or not path.is_file() for path in sources - {entry}):
        raise ValueError("--include accepts Python source files")
    # Embed the sources in the entry because Kaggle CLI uploads only code_file.
    files = io.BytesIO()
    with zipfile.ZipFile(files, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(sources - ({entry} if entry.suffix == ".ipynb" else set())):
            archive.write(path, path.relative_to(source_root).as_posix())
    encoded = base64.b85encode(files.getvalue()).decode("ascii")
    lines = "\n".join(f"    {line!r}" for line in textwrap.wrap(encoded, width=100))
    relative_entry = entry.relative_to(source_root).as_posix()
    bootstrap = f'''import base64
import io
import sys
import tempfile
import zipfile
from pathlib import Path

_siclink_bundle = (
{lines}
)
_siclink_sources = tempfile.TemporaryDirectory(prefix="siclink-")
with zipfile.ZipFile(io.BytesIO(base64.b85decode(_siclink_bundle))) as _archive:
    _archive.extractall(_siclink_sources.name)
_siclink_entry = Path(_siclink_sources.name) / {relative_entry!r}
sys.path[:0] = [str(_siclink_entry.parent), _siclink_sources.name]
'''
    if entry.suffix == ".ipynb":
        if entry_args:
            raise ValueError("--entry-args applies to Python script entries")
        notebook = json.loads(entry.read_text(encoding="utf-8"))
        notebook["cells"].insert(0, {
            "cell_type": "code",
            "execution_count": None,
            "id": "siclink-source-bundle",
            "metadata": {},
            "outputs": [],
            "source": bootstrap,
        })
        destination.write_text(json.dumps(notebook), encoding="utf-8")
    else:
        destination.write_text(
            bootstrap + f"\nimport runpy\nsys.argv = [str(_siclink_entry), *{list(entry_args)!r}]\n"
            "runpy.run_path(str(_siclink_entry), run_name='__main__')\n",
            encoding="utf-8",
        )
    return destination.name


def push(args, kernel):
    entry_type = {".py": "script", ".ipynb": "notebook"}.get(args.entry.suffix)
    if entry_type is None:
        raise ValueError("Entry must be a .py script or .ipynb notebook")
    with tempfile.TemporaryDirectory(prefix="siclink-kaggle-") as folder:
        stage = Path(folder)
        code_file = stage_entry(args.entry, args.include, args.source_root, stage, args.entry_args)
        gpu = args.accelerator != "cpu"
        metadata = {
            "id": kernel,
            "title": args.slug.replace("-", " "),
            "code_file": code_file,
            "language": "python",
            "kernel_type": entry_type,
            "is_private": True,
            "enable_gpu": gpu,
            "enable_internet": args.internet,
            "competition_sources": [COMPETITION],
        }
        if args.kernel_source:
            metadata["kernel_sources"] = args.kernel_source
        (stage / "kernel-metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        command = ["push", "--path", folder]
        if gpu:
            command += ["--accelerator", args.accelerator]
        if args.timeout is not None:
            command += ["--timeout", str(args.timeout)]
        return run_cli(args, command)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", default=PROXY)
    parser.add_argument("--token-file", type=Path, default=TOKEN_FILE)
    actions = parser.add_subparsers(dest="action", required=True)
    uploader = actions.add_parser("push", help="save and run a private notebook version")
    uploader.add_argument("slug", help=f"kernel slug without {OWNER}/")
    uploader.add_argument("entry", type=Path)
    uploader.add_argument("--include", action="append", type=Path, default=[], help="Python source to bundle; repeat as needed")
    uploader.add_argument("--kernel-source", action="append", default=[], help="Kaggle input notebook as owner/slug")
    uploader.add_argument("--source-root", type=Path, default=ROOT, help="root whose import paths are preserved")
    uploader.add_argument("--accelerator", default="NvidiaTeslaT4", help="Kaggle accelerator ID or cpu")
    uploader.add_argument("--internet", action="store_true", help="allow remote downloads of public research assets")
    uploader.add_argument("--timeout", type=int, help="execution timeout in seconds")
    uploader.add_argument("--entry-args", nargs=argparse.REMAINDER, default=[],
                          help="arguments for the remote script; place this option last")

    for action in ("status", "logs", "files", "output"):
        reader = actions.add_parser(action)
        reader.add_argument("slug", help=f"kernel slug without {OWNER}/")
        reader.add_argument("--version", type=int)
        if action == "logs":
            reader.add_argument("--follow", action="store_true")
        if action == "output":
            reader.add_argument("--path", type=Path)
            reader.add_argument("--file-pattern")

    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", args.slug):
        parser.error("Use a lowercase Kaggle kernel slug without the owner")
    return args


def main():
    args = parse_args()
    kernel = f"{OWNER}/{args.slug}"
    if args.action == "push":
        return push(args, kernel)
    if args.version is not None:
        kernel += f"/{args.version}"
    command = [args.action, kernel]
    if args.action == "logs" and args.follow:
        command.append("--follow")
    if args.action == "output":
        folder = args.path or (ROOT / "outputs" / args.slug)
        folder.mkdir(parents=True, exist_ok=True)
        command += ["--path", str(folder)]
        if args.file_pattern:
            command += ["--file-pattern", args.file_pattern]
    return run_cli(args, command)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
