"""Bounded, lossless experiment archives with per-file verification."""
from contextlib import contextmanager
from decimal import Decimal
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
from tempfile import NamedTemporaryFile, TemporaryDirectory


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def home_quota_guard(output, additional_bytes, reserve_gb=Decimal("1")):
    """Parse CCV checkquota's explicit home row; unknown formats fail closed."""
    if "Used(G)" not in output or "SLIMIT(G)" not in output:
        raise ValueError("Unrecognized quota units")
    matches = re.findall(r"(?m)^\S+\s+/oscar/home\s+([0-9.]+)\s+[0-9.]+\s+([0-9.]+)\s+", output)
    if len(matches) != 1:
        raise ValueError("Expected exactly one personal home quota row")
    used, limit = map(Decimal, matches[0])
    # Decimal GB overestimates a byte count if the site's G unit is GiB.
    projected = used + Decimal(additional_bytes) / Decimal(10**9)
    if projected + reserve_gb > limit:
        raise RuntimeError(f"Home storage gate: {projected} G projected, {reserve_gb} G reserve, {limit} G soft limit")
    return dict(used_g=str(used), soft_limit_g=str(limit), added_bytes=additional_bytes, reserve_g=str(reserve_gb))


def safe_relative(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError(f"Unsafe archive member: {name}")
    return path


def artifact_paths(directory, name):
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name) is None or name in (".", ".."):
        raise ValueError("Invalid artifact name")
    directory = Path(directory)
    return directory / (name + ".tar.gz"), directory / (name + ".manifest.json")


def verify_archive(directory, name):
    archive, manifest_path = artifact_paths(directory, name)
    manifest = json.loads(manifest_path.read_text())
    if manifest["name"] != name or manifest["archive_bytes"] != archive.stat().st_size:
        raise ValueError("Archive identity or size differs")
    if manifest["archive_sha256"] != sha256(archive):
        raise ValueError("Archive hash differs")
    return manifest


@contextmanager
def publish_lock(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".publish.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def publish_tree(source, includes, directory, name, *, budget_bytes, quota_check):
    """Compress locally, then atomically publish under a shared storage cap."""
    source, directory = Path(source), Path(directory)
    files = {}
    for include in includes:
        safe_relative(include)
        path = source / include
        if not path.exists():
            raise FileNotFoundError(path)
        candidates = [path, *path.rglob("*")] if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_symlink():
                raise ValueError("Result archives cannot contain symbolic links")
            if candidate.is_file():
                name_in_archive = candidate.relative_to(source).as_posix()
                files[name_in_archive] = dict(bytes=candidate.stat().st_size, sha256=sha256(candidate))
    if not files:
        raise ValueError("Cannot publish an empty result archive")
    archive, manifest_path = artifact_paths(directory, name)
    with TemporaryDirectory(prefix="amor-pack-") as temporary:
        local = Path(temporary) / "result.tar.gz"
        with tarfile.open(local, "w:gz", compresslevel=6, dereference=True) as stream:
            for relative in sorted(files):
                stream.add(source / relative, arcname=relative, recursive=False)
        manifest = dict(name=name, archive_bytes=local.stat().st_size, archive_sha256=sha256(local), files=files)
        encoded = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        with publish_lock(directory):
            if archive.exists() or manifest_path.exists():
                previous = verify_archive(directory, name)
                if previous["files"] != files:
                    raise ValueError("Refusing to overwrite a different completed artifact")
                return previous
            used = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
            required = local.stat().st_size + len(encoded)
            if used + required > budget_bytes:
                raise RuntimeError(f"Archive budget exceeded: {used}+{required}>{budget_bytes}")
            quota_check(required)
            with NamedTemporaryFile(prefix=".publish-", dir=directory, delete=False) as stream:
                partial = Path(stream.name)
                try:
                    with local.open("rb") as incoming:
                        shutil.copyfileobj(incoming, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                    if sha256(partial) != manifest["archive_sha256"]:
                        raise ValueError("Archive changed during copy")
                    os.replace(partial, archive)
                finally:
                    partial.unlink(missing_ok=True)
            with NamedTemporaryFile(prefix=".manifest-", dir=directory, delete=False) as stream:
                partial = Path(stream.name)
                try:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                    os.replace(partial, manifest_path)
                finally:
                    partial.unlink(missing_ok=True)
        return verify_archive(directory, name)


def restore_tree(directory, name, destination):
    """Restore regular files only and verify every file, including overlaps."""
    manifest = verify_archive(directory, name)
    archive, _ = artifact_paths(directory, name)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    seen = set()
    with tarfile.open(archive, "r:gz") as stream:
        for member in stream:
            relative = safe_relative(member.name)
            if not member.isfile() or member.name in seen or member.name not in manifest["files"]:
                raise ValueError("Unexpected, repeated or non-regular archive member")
            seen.add(member.name)
            expected = manifest["files"][member.name]
            if member.size != expected["bytes"]:
                raise ValueError("Archive member size differs")
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(root) or target.is_symlink():
                raise ValueError("Extraction would escape its destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if target.stat().st_size != expected["bytes"] or sha256(target) != expected["sha256"]:
                    raise ValueError("Conflicting files across stage archives")
            else:
                with stream.extractfile(member) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                if sha256(target) != expected["sha256"]:
                    raise ValueError("Restored file hash differs")
    if seen != set(manifest["files"]):
        raise ValueError("Missing archive members")
    return manifest
