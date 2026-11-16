#!/usr/bin/env python3
"""Archive captured HEAD Git blobs and verify every ZIP entry byte-for-byte.

Python 3.7+; standard library only. This does not call git archive, copy working
tree content, run checkout filters, publish files, or overwrite existing output.
For output NAME.zip, sidecars are NAME.zip.sha256 and
NAME.zip.verification.json. A failed run can leave a new unverified ZIP for
inspection; rerun with a new output name rather than overwriting that evidence.
"""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import subprocess
import sys
import zipfile


SCHEMA_VERSION = "head_blob_source_archive_v1"
IMAGE_SUFFIXES = {
    ".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff",
    ".webp", ".ico", ".svg", ".eps", ".heic", ".avif", ".dng",
    ".pcx", ".ppm", ".pgm", ".pbm", ".pnm", ".jp2", ".j2k",
}
GENERATED_SUFFIXES = {
    ".mat", ".fig", ".asv", ".mexa64", ".mexw64", ".pyc",
    ".onnx", ".pt", ".pth", ".h5", ".hdf5", ".tflite", ".kmodel",
    ".safetensors", ".caffemodel", ".npy", ".npz", ".bin", ".weights",
    ".ckpt", ".pkl", ".pickle", ".mp4", ".avi", ".mov", ".webm", ".mkv",
    ".zip", ".7z", ".rar", ".tar", ".gz", ".bz2", ".xz",
}


def git(project_root, *args, **kwargs):
    result = subprocess.run(
        ["git", "-C", str(project_root)] + list(args),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False,
    )
    if result.returncode != 0 and not kwargs.get("allow_failure", False):
        raise RuntimeError("Git command failed: {}\n{}".format(
            " ".join(args), result.stderr.decode("utf-8", "replace").strip()))
    return result


def text_git(project_root, *args):
    return git(project_root, *args).stdout.decode("utf-8", "strict").strip()


def exclusion_reason(path):
    p = PurePosixPath(path)
    lowered_parts = [part.lower() for part in p.parts]
    if "runs" in lowered_parts:
        return "run output directory"
    if any(part in {".git", "__pycache__"} for part in lowered_parts):
        return "Git internals or generated Python cache"
    if len(lowered_parts) >= 2 and lowered_parts[0] == "data" and \
            lowered_parts[1] in {"raw", "extracted", "cache"}:
        return "local raw/extracted/cached data"
    if path.lower() == "local_config.json":
        return "local machine configuration"
    suffix = p.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image content excluded from source archive"
    if suffix in GENERATED_SUFFIXES:
        return "model, generated binary, checkpoint, figure, or nested archive"
    return None


def head_entries(project_root, commit):
    raw = git(project_root, "ls-tree", "-r", "-z", "--full-tree", commit).stdout
    entries, excluded = [], []
    names = set()
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        mode, kind, object_id = metadata.decode("ascii").split()
        path = path_bytes.decode("utf-8", "strict")
        p = PurePosixPath(path)
        if p.is_absolute() or not p.parts or "\\" in path or \
                any(part in {"", ".", ".."} or ":" in part for part in p.parts):
            raise RuntimeError("Unsafe archive member name: {!r}".format(path))
        key = path.casefold()
        if key in names:
            raise RuntimeError("Case-insensitive duplicate archive path: " + path)
        names.add(key)
        entry = {"path": path, "git_mode": mode, "git_blob": object_id}
        reason = exclusion_reason(path)
        if reason:
            entry["reason"] = reason
            excluded.append(entry)
            continue
        if kind != "blob" or mode not in {"100644", "100755", "120000"}:
            raise RuntimeError("Unsupported Git entry; refusing incomplete source: " + path)
        if not re.match(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$", object_id):
            raise RuntimeError("Invalid Git blob ID for " + path)
        entries.append(entry)
    if not entries:
        raise RuntimeError("No allowed source files in captured HEAD")
    return entries, excluded


class GitBlobReader(object):
    """Read exact blob bytes through one binary cat-file process."""

    def __init__(self, project_root):
        self.process = subprocess.Popen(
            ["git", "-C", str(project_root), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

    def __enter__(self):
        return self

    def read(self, object_id):
        self.process.stdin.write(object_id.encode("ascii") + b"\n")
        self.process.stdin.flush()
        header = self.process.stdout.readline().rstrip(b"\n").split()
        if len(header) != 3 or header[0].decode("ascii") != object_id or header[1] != b"blob":
            raise RuntimeError("Unexpected cat-file response for " + object_id)
        size = int(header[2])
        chunks, remaining = [], size
        while remaining:
            chunk = self.process.stdout.read(remaining)
            if not chunk:
                raise RuntimeError("Truncated Git blob " + object_id)
            chunks.append(chunk)
            remaining -= len(chunk)
        if self.process.stdout.read(1) != b"\n":
            raise RuntimeError("Invalid Git blob terminator for " + object_id)
        return b"".join(chunks)

    def __exit__(self, exc_type, exc_value, traceback):
        self.process.stdin.close()
        if exc_type is not None:
            self.process.terminate()
        self.process.stdout.close()
        error = self.process.stderr.read()
        self.process.stderr.close()
        code = self.process.wait()
        if exc_type is None and code != 0:
            raise RuntimeError("cat-file failed: " + error.decode("utf-8", "replace"))
        return False


def zip_timestamp(commit_unix_time):
    stamp = datetime.datetime.fromtimestamp(commit_unix_time, datetime.timezone.utc)
    year = min(2107, max(1980, stamp.year))
    if year != stamp.year:
        return (year, 1, 1, 0, 0, 0)
    return (stamp.year, stamp.month, stamp.day, stamp.hour, stamp.minute, stamp.second)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_archive(project_root, output):
    project_root = project_root.expanduser().resolve()
    requested_output = output.expanduser().absolute()
    # Check the requested names before resolution, so even a broken symlink
    # at one of the output names is treated as existing evidence.
    for path in (requested_output, Path(str(requested_output) + ".sha256"),
                 Path(str(requested_output) + ".verification.json")):
        if os.path.lexists(str(path)):
            raise RuntimeError("Refusing to overwrite existing output: " + str(path))
    output = requested_output.resolve()
    if not project_root.is_dir():
        raise RuntimeError("Project root is not a directory")
    top_level = Path(text_git(project_root, "rev-parse", "--show-toplevel")).resolve()
    if os.path.normcase(str(top_level)) != os.path.normcase(str(project_root)):
        raise RuntimeError("--project-root must be the Git top-level directory")
    if output.suffix.lower() != ".zip":
        raise RuntimeError("--output must name a new .zip file")
    hash_path = Path(str(output) + ".sha256")
    verification_path = Path(str(output) + ".verification.json")
    for path in (output, hash_path, verification_path):
        if os.path.lexists(str(path)):
            raise RuntimeError("Refusing to overwrite existing output: " + str(path))

    commit = text_git(project_root, "rev-parse", "--verify", "HEAD^{commit}")
    tree = text_git(project_root, "rev-parse", commit + "^{tree}")
    commit_time = int(text_git(project_root, "show", "-s", "--format=%ct", commit))
    tags = text_git(project_root, "tag", "--points-at", commit).splitlines()
    entries, excluded = head_entries(project_root, commit)
    dirty = bool(git(project_root, "status", "--porcelain", "-z").stdout)
    timestamp = zip_timestamp(commit_time)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation is also the race-safe final no-overwrite gate.
    with output.open("xb") as target:
        with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=9, allowZip64=True) as archive:
            with GitBlobReader(project_root) as reader:
                for entry in entries:
                    payload = reader.read(entry["git_blob"])
                    info = zipfile.ZipInfo(entry["path"], date_time=timestamp)
                    info.create_system = 3
                    info.compress_type = zipfile.ZIP_DEFLATED
                    mode = int(entry["git_mode"], 8)
                    info.external_attr = mode << 16
                    archive.writestr(info, payload, compress_type=zipfile.ZIP_DEFLATED,
                                     compresslevel=9)

    verified = []
    with zipfile.ZipFile(str(output), "r") as archive:
        infos = archive.infolist()
        if [item.filename for item in infos] != [item["path"] for item in entries]:
            raise RuntimeError("ZIP entry names/order differ from the captured allowed HEAD tree")
        if len(set(item.filename for item in infos)) != len(entries):
            raise RuntimeError("ZIP contains duplicate names")
        with GitBlobReader(project_root) as reader:
            for entry, info in zip(entries, infos):
                expected = reader.read(entry["git_blob"])
                actual = archive.read(info)
                if actual != expected:
                    raise RuntimeError("Byte verification failed for " + entry["path"])
                if info.external_attr >> 16 != int(entry["git_mode"], 8):
                    raise RuntimeError("Git file mode verification failed for " + entry["path"])
                result = dict(entry)
                result.update({
                    "size_bytes": len(actual),
                    "sha256": hashlib.sha256(actual).hexdigest(),
                    "zip_crc32": "{:08x}".format(info.CRC),
                    "byte_equal_to_git_blob": True,
                })
                verified.append(result)

    archive_hash = sha256_file(output)
    head_after = text_git(project_root, "rev-parse", "--verify", "HEAD^{commit}")
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "project_root": str(project_root),
        "archive_path": str(output),
        "archive_size_bytes": output.stat().st_size,
        "archive_sha256": archive_hash,
        "source_ref": "HEAD captured once at start",
        "commit": commit,
        "tree": tree,
        "tags_pointing_at_commit": tags,
        "head_after_packaging": head_after,
        "head_unchanged_during_packaging": head_after == commit,
        "working_tree_dirty_at_start": dirty,
        "source_bytes": "git cat-file blob by captured HEAD tree object IDs; no worktree or checkout filters",
        "archive_prefix": "",
        "timestamp_policy": "Captured commit UTC timestamp, ZIP DOS range clamped if necessary",
        "entry_count": len(verified),
        "verified_entry_count": len(verified),
        "excluded_entry_count": len(excluded),
        "all_entries_byte_equal_to_git_blobs": True,
        "exclusion_policy": {
            "directories": ["runs (any path component)", ".git", "__pycache__",
                            "data/raw", "data/extracted", "data/cache"],
            "files": ["local_config.json"],
            "image_suffixes": sorted(IMAGE_SUFFIXES),
            "generated_binary_suffixes": sorted(GENERATED_SUFFIXES),
        },
        "python_version": platform.python_version(),
        "git_version": text_git(project_root, "--version"),
        "tool_sha256": sha256_file(Path(__file__).resolve()),
        "verification_method": "Each ZIP member decompressed and compared byte-for-byte to a fresh read of its HEAD Git blob; SHA-256 and modes recorded",
        "entries": verified,
        "excluded_entries": excluded,
    }
    # Sidecars are exclusive too; an existing sidecar is never replaced.
    with hash_path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("{}  {}\n".format(archive_hash, output.name))
    with verification_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return {
        "status": "PASS", "commit": commit, "tags": tags,
        "entry_count": len(verified), "excluded_entry_count": len(excluded),
        "sha256": archive_hash, "output": str(output),
        "sha256_file": str(hash_path), "verification_file": str(verification_path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", required=True, type=Path,
                        help="Git top-level repository directory")
    parser.add_argument("--output", required=True, type=Path,
                        help="New .zip; neither ZIP nor either sidecar may exist")
    args = parser.parse_args()
    try:
        result = create_archive(args.project_root, args.output)
    except (OSError, RuntimeError, ValueError, UnicodeError, zipfile.BadZipFile) as error:
        print(json.dumps({"status": "FAILED", "error": str(error)}, ensure_ascii=False),
              file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
