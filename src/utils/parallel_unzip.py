"""Parallel ZIP extraction for large multi-file archives.

For STORED or low-compression JPEG-heavy archives (like FFHQ), the
per-file work is dominated by I/O + small CRC checks, so Python threads
can parallelize despite the GIL: zlib's C extension releases the GIL
during DEFLATE inflate, and STORED entries are pure I/O.

Two design decisions matter for the 50k-file FFHQ case:

1. **One ``ZipFile`` handle per worker, not per file.** Each ``ZipFile``
   constructor re-parses the central directory (~50k ``ZipInfo`` objects
   for FFHQ-1024). Doing that per file would dominate runtime; we shard
   members across ``num_workers * CHUNK_OVERSUBSCRIPTION`` round-robin
   chunks so the CD is parsed at most ``num_workers * 8`` times instead
   of 50,000.
2. **``--stage-local`` copies the source zip off Drive FUSE first.**
   Google Drive's FUSE driver serializes reads of a single backing file,
   so thread parallelism is wasted when ``src`` lives on
   ``/content/drive/...``. A single sequential copy (~60 s for 18 GB at
   ~300 MB/s) then extracts in parallel from local overlay FS.

Wall-time on Colab Pro+ (A100 instance, FFHQ-1024 50k files, ~18 GB):

    unzip -q -j -o                  (single-threaded)   ~25-35 min
    parallel_unzip (Drive src)                          ~10-20 min
    parallel_unzip --stage-local                        ~3-5 min

Notes:
    * ``strip_dirs=True`` mirrors ``unzip -j`` (flatten directory layout).
    * Overwriting is implicit (open ``wb``), no need for an ``-o``-style flag.

CLI:

    uv run python -m src.utils.parallel_unzip \\
        --src train_50k_1024.zip \\
        --dst /content/ffhq/train_1024 \\
        --stage-local \\
        --stage-dir /content/ffhq \\
        --strip-dirs
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import tempfile
import time
import zipfile
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
from typing import Final

logger = logging.getLogger(__name__)

# Number of shards per worker. Round-robin sharding with 8x oversubscription
# keeps workers fed under variable file sizes while keeping the per-shard
# ``ZipFile`` open count tiny (num_workers * 8 vs the 50k file count).
CHUNK_OVERSUBSCRIPTION: Final[int] = 8

# Mount root considered "remote/slow" for the purpose of picking a stage
# directory. Google Drive on Colab mounts here via FUSE; reads of the same
# file from multiple threads serialize at the FUSE layer.
_DRIVE_FUSE_PREFIX: Final[str] = "/content/drive"


def _default_workers() -> int:
    """Auto-detect a sensible thread-pool size for I/O-bound work.

    Returns the logical CPU count (``os.cpu_count()``) since the pool is
    thread-based and over-subscription is safe for I/O-dominant work
    (zlib inflate + filesystem syscalls). Falls back to 8 if detection
    fails (e.g., restricted containers without /proc).
    """
    return os.cpu_count() or 8


def _is_under_drive_fuse(path: Path) -> bool:
    """True iff ``path`` is on the Drive FUSE mount.

    Uses ``Path.relative_to`` rather than ``str.startswith`` so unrelated
    directories like ``/content/drive_cache`` or ``/content/driven`` (which
    share the textual prefix but are NOT under the mount root) are not
    misclassified as Drive FUSE.
    """
    fuse_root = Path(_DRIVE_FUSE_PREFIX)
    try:
        path.resolve().relative_to(fuse_root.resolve())
        return True
    except (ValueError, OSError):
        return False


def _pick_stage_dir(dst: Path) -> Path:
    """Pick a local-FS directory for the staged zip copy.

    Prefers ``dst.parent`` when it is on local overlay FS (so the staged
    file and extracted tree live on the same filesystem). Falls back to
    the system temp dir when ``dst.parent`` is itself on Drive FUSE.
    """
    parent = dst.parent.resolve()
    if _is_under_drive_fuse(parent):
        return Path(tempfile.gettempdir())
    return parent


def _estimate_extracted_bytes(src: Path) -> int:
    """Sum uncompressed sizes from the zip central directory.

    Reads ``ZipInfo.file_size`` (the original byte length recorded per
    entry). Robust to DEFLATE-compressed archives where ``src.stat().size``
    can dramatically underestimate the extracted footprint (and a hostile
    zip-bomb input — though our FFHQ archives are STORED, the accurate
    estimate is cheap so we use it unconditionally).
    """
    with zipfile.ZipFile(src) as zf:
        return sum(m.file_size for m in zf.infolist() if not m.is_dir())


@contextmanager
def _maybe_stage(
    src: Path, *, stage_local: bool, stage_dir: Path | None, dst: Path
) -> Iterator[Path]:
    """Yield a path to extract from, optionally staging ``src`` locally first.

    When ``stage_local`` is False, yields ``src`` unchanged. When True,
    copies ``src`` to a temporary file under ``stage_dir`` (or an
    auto-picked local dir), yields that path, and unlinks the temp on
    exit (including on exception). The temp dir itself is left in place
    even if it was just created — its lifecycle is owned by the caller.
    """
    if not stage_local:
        yield src
        return

    target_dir = stage_dir if stage_dir is not None else _pick_stage_dir(dst)
    target_dir = Path(target_dir)

    # Foot-gun guard: stage_dir on Drive FUSE defeats the purpose
    # of staging (the staged zip would still be on FUSE and reads would
    # still serialize at the FUSE layer). Raise early with the explicit
    # fix instead of silently giving the user the slow path. Validate
    # BEFORE mkdir so a wrong stage_dir doesn't leave a half-created
    # directory on the user's Drive. Uses ``_is_under_drive_fuse`` so
    # sibling directories like ``/content/drive_cache`` are not
    # misclassified.
    if _is_under_drive_fuse(target_dir):
        raise ValueError(
            f"stage_dir={target_dir} is on Drive FUSE; staging there "
            "defeats stage_local's purpose (FUSE serializes single-file "
            "reads). Pass a local-FS directory or leave stage_dir=None "
            "to auto-pick."
        )

    # Create the directory only after the FUSE check passes -- otherwise a
    # wrong stage_dir would leak an empty directory onto the user's Drive.
    target_dir.mkdir(parents=True, exist_ok=True)

    # Disk-budget guard. Two free-space checks:
    #   * ``target_dir`` must hold the staged zip (= ``src_bytes``);
    #   * ``dst.parent`` must hold the extracted tree (= sum of
    #     ``ZipInfo.file_size`` over members, accurate even for
    #     DEFLATE archives).
    # When ``target_dir`` and ``dst.parent`` are on the same filesystem
    # we charge both to a single free-space pool so we don't double-spend.
    src_bytes = src.stat().st_size
    extracted_bytes = _estimate_extracted_bytes(src)
    stage_need = int(src_bytes * 1.05)              # zip + tiny FS slack
    dst_need = int(extracted_bytes * 1.05)          # extracted + tiny FS slack
    dst_parent = dst.parent.resolve()
    try:
        same_fs = target_dir.stat().st_dev == dst_parent.stat().st_dev
    except OSError:
        same_fs = False
    if same_fs:
        combined_need = stage_need + dst_need
        free = shutil.disk_usage(target_dir).free
        if free < combined_need:
            raise OSError(
                f"insufficient free space in {target_dir} "
                f"(shared FS for stage + dst): have {free / 1e9:.1f} GB, "
                f"need ~{combined_need / 1e9:.1f} GB "
                f"(staged zip {src_bytes / 1e9:.1f} GB + extracted "
                f"{extracted_bytes / 1e9:.1f} GB)"
            )
    else:
        stage_free = shutil.disk_usage(target_dir).free
        if stage_free < stage_need:
            raise OSError(
                f"insufficient free space in stage_dir {target_dir}: "
                f"have {stage_free / 1e9:.1f} GB, "
                f"need ~{stage_need / 1e9:.1f} GB "
                f"(staged zip {src_bytes / 1e9:.1f} GB)"
            )
        dst_free = shutil.disk_usage(dst_parent).free
        if dst_free < dst_need:
            raise OSError(
                f"insufficient free space in dst {dst_parent}: "
                f"have {dst_free / 1e9:.1f} GB, "
                f"need ~{dst_need / 1e9:.1f} GB "
                f"(extracted tree {extracted_bytes / 1e9:.1f} GB)"
            )

    # SIM115: we want delete=False + manual unlink in the finally below so
    # the file outlives this scope (workers read it) and the cleanup runs
    # exactly once even on KeyboardInterrupt.
    tmp = tempfile.NamedTemporaryFile(  # noqa: SIM115
        dir=target_dir, suffix=".zip", delete=False
    )
    tmp_path = Path(tmp.name)
    tmp.close()
    try:
        t0 = time.time()
        logger.info(
            "staging %s -> %s (%.2f GB)",
            src,
            tmp_path,
            src_bytes / 1e9,
        )
        shutil.copyfile(src, tmp_path)
        # Cache `elapsed` so the duration and bandwidth columns are
        # computed from the same instant. Two `time.time()` calls in the
        # `logger.info` argument list would otherwise diverge by the
        # microseconds of formatting between them.
        elapsed = time.time() - t0
        logger.info(
            "stage copy done in %.1fs (%.0f MB/s)",
            elapsed,
            src_bytes / 1e6 / max(elapsed, 1e-6),
        )
        yield tmp_path
    finally:
        tmp_path.unlink(missing_ok=True)


def _safe_member_name(member: zipfile.ZipInfo, *, strip_dirs: bool) -> str:
    """Return the on-disk name for ``member``, rejecting zip-slip vectors.

    Zip-slip defence-in-depth:

    * **Absolute paths** (``/etc/passwd``) — rejected here in both modes,
      because joining as ``dst / name`` would override ``dst`` entirely
      with the absolute path semantics of ``pathlib.PurePath``.
    * **Parent-traversal in non-flattened mode** (``../foo``) — rejected
      here. The post-resolve containment check in ``_extract_member`` is
      a backstop, but failing early gives a clearer error message tied to
      the member name rather than to ``out_path``.
    * **Parent-traversal in flattened mode** (``strip_dirs=True``) — NOT
      caught here. ``Path('../foo.bin').name == 'foo.bin'`` (safe), but
      ``Path('..').name == '..'`` (UNSAFE on its own). The literal-``..``
      case relies on ``_extract_member``'s post-resolve check to refuse
      ``dst / '..'`` (which resolves outside ``dst``). The check is
      placed there, not here, so the strip-dirs path stays a single
      basename call rather than re-walking ``Path(raw).parts``.
    """
    raw = member.filename
    # Cross-platform absolute-path detection. ``Path(raw)`` instantiates the
    # host's concrete ``Path`` class (Posix vs Windows), so on Linux/macOS
    # ``Path('C:\\foo').is_absolute()`` returns False even though that name is
    # a Windows-absolute path. ``PureWindowsPath`` is OS-independent and
    # always uses Windows path semantics, so it catches drive-letter
    # absolutes from a crafted zip regardless of host OS. The two checks
    # together cover both POSIX and Windows absolutes.
    if Path(raw).is_absolute() or PureWindowsPath(raw).is_absolute():
        raise ValueError(
            f"zip member name {raw!r} is absolute; refusing to write "
            "outside dst (zip-slip protection)."
        )
    if strip_dirs:
        return Path(raw).name
    # Non-flattened mode: reject any `..` component up-front. The caller
    # additionally re-checks ``out_path.resolve()`` against dst, which
    # catches symlink-traversal and any pathological encoding this string
    # check might miss.
    if ".." in Path(raw).parts:
        raise ValueError(
            f"zip member name {raw!r} contains a parent-traversal "
            "component; refusing to extract outside dst (zip-slip "
            "protection)."
        )
    return raw


def _extract_member(
    zfp: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    dst: Path,
    dst_resolved: Path,
    *,
    strip_dirs: bool,
    chunk_size: int,
) -> None:
    """Write one archive member to disk through ``zfp``.

    Args:
        dst_resolved: ``dst.resolve()`` pre-computed by the caller. Used
            for the final ``out_path.resolve()`` containment check that
            backs up :func:`_safe_member_name` against symlink shenanigans
            (e.g., a pre-existing ``dst/foo`` symlink pointing outside).
    """
    name = _safe_member_name(member, strip_dirs=strip_dirs)
    if not name:  # archive entry pointing to a directory itself
        return
    out_path = dst / name
    if not strip_dirs:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    # Final containment check post-resolve. Catches symlink-based escape
    # vectors that the pre-resolve string check on member.filename can't
    # see (e.g., dst/foo already exists as a symlink to /tmp/elsewhere).
    try:
        out_path.resolve().relative_to(dst_resolved)
    except ValueError as exc:
        raise ValueError(
            f"resolved write path {out_path.resolve()} escapes dst "
            f"{dst_resolved} for member {member.filename!r} "
            "(zip-slip protection)."
        ) from exc
    with zfp.open(member) as srcf, out_path.open("wb") as dstf:
        while True:
            chunk = srcf.read(chunk_size)
            if not chunk:
                break
            dstf.write(chunk)


def _extract_shard(
    members: Sequence[zipfile.ZipInfo],
    src: Path,
    dst: Path,
    *,
    strip_dirs: bool,
    chunk_size: int,
) -> None:
    """Extract one shard via a single shared ``ZipFile`` handle.

    The handle is opened once per shard (not once per member), amortising
    the central-directory parse over all files the worker handles.
    """
    # Cache ``dst.resolve()`` once per shard so the per-member containment
    # check doesn't have to re-stat the path 50k times.
    dst_resolved = dst.resolve()
    with zipfile.ZipFile(src) as zfp:
        for member in members:
            _extract_member(
                zfp,
                member,
                dst,
                dst_resolved,
                strip_dirs=strip_dirs,
                chunk_size=chunk_size,
            )


def parallel_unzip(
    src: Path,
    dst: Path,
    *,
    num_workers: int | None = None,
    strip_dirs: bool = False,
    chunk_size: int = 64 * 1024,
    stage_local: bool = False,
    stage_dir: Path | None = None,
) -> int:
    """Extract ``src`` zip to ``dst`` using ``num_workers`` threads.

    Args:
        src: Path to ``.zip`` archive.
        dst: Destination directory (created if missing).
        num_workers: Thread count for the ``ThreadPoolExecutor``.
            ``None`` (default) auto-detects via :func:`_default_workers`
            (= ``os.cpu_count()``, fallback 8).
        strip_dirs: If ``True``, drop leading directories from each
            archive entry (equivalent to ``unzip -j``). Useful for FFHQ
            zips that wrap a single top-level directory.
        chunk_size: Per-read chunk size in bytes (default 64 KiB).
        stage_local: If ``True``, copy ``src`` to a local-FS temp file
            before extraction (kills Drive FUSE serialization). Cleaned
            up on exit, including on exception.
        stage_dir: Override for the staging directory. Only consulted
            when ``stage_local`` is ``True``; defaults to ``dst.parent``
            (or system temp if that is itself on Drive FUSE).

    Returns:
        Number of files extracted (directory entries skipped).

    Raises:
        FileNotFoundError: If ``src`` does not exist.
        ValueError: If ``num_workers < 1``.
        OSError: If ``stage_local`` is requested but the staging dir has
            insufficient free space for the staged zip + extracted tree.
    """
    src = Path(src)
    dst = Path(dst)
    # Normalize stage_dir up-front so callers can pass a raw string (matches
    # the existing behavior for src/dst). Without this, _maybe_stage's
    # `.mkdir()` would AttributeError on a string input.
    stage_dir = Path(stage_dir) if stage_dir is not None else None
    if not src.exists():
        raise FileNotFoundError(f"zip not found: {src}")
    if num_workers is None:
        num_workers = _default_workers()
    if num_workers < 1:
        raise ValueError(f"num_workers must be >= 1, got {num_workers}")
    dst.mkdir(parents=True, exist_ok=True)

    # Surface the most common foot-gun: the source zip is on Drive FUSE
    # but the caller forgot ``stage_local=True``. FUSE serializes reads of
    # a single backing file across threads, so the chunked extraction
    # below regresses to roughly single-worker throughput. We don't auto-
    # enable staging because that would surprise callers with extra disk
    # use; a log line is enough to point to the right knob. Boundary-aware
    # check (``_is_under_drive_fuse``) so /content/drive_cache etc. don't
    # trigger a false positive.
    if not stage_local and _is_under_drive_fuse(src):
        logger.warning(
            "src=%s is on Drive FUSE but stage_local=False; thread "
            "parallelism is bottlenecked by FUSE's single-file read "
            "serialization. Pass stage_local=True (or --stage-local on "
            "the CLI) for a 5-10x speedup on FFHQ-scale archives.",
            src,
        )

    with _maybe_stage(
        src, stage_local=stage_local, stage_dir=stage_dir, dst=dst
    ) as effective_src:
        # Enumerate members once on the main thread; workers each open
        # their own ZipFile per shard.
        with zipfile.ZipFile(effective_src) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]

        n_shards = max(1, num_workers * CHUNK_OVERSUBSCRIPTION)
        # Round-robin sharding interleaves member positions, so
        # variable per-file sizes balance naturally across shards.
        shards = [members[i::n_shards] for i in range(n_shards)]
        shards = [s for s in shards if s]  # drop empties when fewer members than shards

        def _run(shard: Sequence[zipfile.ZipInfo]) -> None:
            _extract_shard(
                shard,
                effective_src,
                dst,
                strip_dirs=strip_dirs,
                chunk_size=chunk_size,
            )

        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            # Drain so any exception in a worker propagates here.
            for _ in ex.map(_run, shards):
                pass
    return len(members)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help=".zip path")
    parser.add_argument(
        "--dst", type=Path, required=True, help="destination directory"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help=(
            "thread count (default: auto-detect via os.cpu_count(), "
            "fallback 8). Over-subscription is safe for I/O-bound work."
        ),
    )
    parser.add_argument(
        "--strip-dirs",
        action="store_true",
        help="flatten directory layout (equivalent to `unzip -j`)",
    )
    parser.add_argument(
        "--stage-local",
        action="store_true",
        help=(
            "copy --src to a local-FS temp file before extraction. "
            "Required for Drive-FUSE-mounted zips to bypass FUSE's "
            "single-file read serialization."
        ),
    )
    parser.add_argument(
        "--stage-dir",
        type=Path,
        default=None,
        help=(
            "directory for the staged zip copy (default: --dst's parent, "
            "or system temp if that is itself on Drive FUSE). Ignored "
            "unless --stage-local is set."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    t0 = time.time()
    resolved_workers = args.num_workers if args.num_workers is not None else _default_workers()
    logger.info(
        "parallel_unzip: src=%s dst=%s workers=%d (logical_cpus=%d) "
        "strip_dirs=%s stage_local=%s stage_dir=%s",
        args.src,
        args.dst,
        resolved_workers,
        os.cpu_count() or -1,
        args.strip_dirs,
        args.stage_local,
        args.stage_dir,
    )
    n = parallel_unzip(
        args.src,
        args.dst,
        num_workers=args.num_workers,
        strip_dirs=args.strip_dirs,
        stage_local=args.stage_local,
        stage_dir=args.stage_dir,
    )
    elapsed = time.time() - t0
    logger.info(
        "extracted %d files in %.1fs (%.0f files/s) from %s to %s",
        n,
        elapsed,
        n / max(elapsed, 1e-6),
        args.src,
        args.dst,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
