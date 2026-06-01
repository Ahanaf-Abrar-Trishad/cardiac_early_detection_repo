#!/usr/bin/env python3
"""
Download CAMUS and ACDC datasets from the Human Heart Project
(https://humanheart-project.creatis.insa-lyon.fr/database/)

The platform is built on Girder — a REST-based data management system.
This script walks the collection hierarchy via the REST API and downloads
all files, preserving the folder structure.

Speed optimisations vs the original script
-------------------------------------------
1. No HEAD requests — the old --resume used a HEAD per file to get its size.
   Now we send the GET immediately with a Range header; if the server returns
   416 (Range Not Satisfiable) the file is already complete and we skip it.
   If Content-Length comes back on the GET we use that.  Zero extra round-trips.

2. Parallel patient downloads — top-level patient folders are downloaded with
   a ThreadPoolExecutor (default 4 workers, tunable via --workers).
   Each worker has its own requests.Session so connections don't conflict.

3. Larger chunk size — bumped from 1 MB → 4 MB so fewer Python loop iterations
   per file and better utilisation of a fast connection.

Usage
-----
    # Download both datasets (default, 4 parallel workers)
    python download_cardiac_datasets.py

    # Download only CAMUS with 8 parallel workers
    python download_cardiac_datasets.py --datasets camus --workers 8

    # Resume (skips already-complete files without any HEAD requests)
    python download_cardiac_datasets.py --resume

    # Jump straight to patient_051 and download from there onward
    python download_cardiac_datasets.py --skip-until patient_051

    # Download only one specific patient folder
    python download_cardiac_datasets.py --patient patient_051

    # Combine: start at patient_051 and resume partial files inside it
    python download_cardiac_datasets.py --skip-until patient_051 --resume

    # Limit download speed (bytes/sec), e.g. 5 MB/s
    python download_cardiac_datasets.py --max-speed 5242880

Requirements
------------
    pip install requests tqdm
"""

import argparse
import sys
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_API = "https://humanheart-project.creatis.insa-lyon.fr/database/api/v1"

COLLECTIONS = {
    "camus": {
        "id": "6373703d73e9f0047faa1bc8",
        "name": "CAMUS",
        "description": "Cardiac Acquisitions for Multi-structure Ultrasound Segmentation "
                       "(500 patients, 2D echocardiography)",
    },
    "acdc": {
        "id": "637218c173e9f0047faa00fb",
        "name": "ACDC",
        "description": "Automated Cardiac Diagnosis Challenge "
                       "(150 patients, 3D cine-MRI)",
    },
}

CHUNK_SIZE = 4 * 1024 * 1024   # 4 MB — fewer loop iterations, better throughput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Thread-local storage so every worker thread gets its own requests.Session
_local = threading.local()


def get_session() -> requests.Session:
    """Return (or lazily create) a per-thread requests.Session."""
    if not hasattr(_local, "session"):
        s = requests.Session()
        s.headers.update({"User-Agent": "cardiac-dataset-downloader/2.0 (research use)"})
        _local.session = s
    return _local.session


# ---------------------------------------------------------------------------
# Girder REST helpers
# ---------------------------------------------------------------------------

def api_get(endpoint: str, params: dict = None) -> dict | list:
    url = f"{BASE_API}/{endpoint.lstrip('/')}"
    resp = get_session().get(url, params=params or {}, timeout=60)
    if resp.status_code == 401:
        raise PermissionError(
            f"Authentication required for {url}. "
            "Pass --username/--password or --api-key."
        )
    if resp.status_code == 404:
        raise FileNotFoundError(f"Resource not found: {url}")
    resp.raise_for_status()
    return resp.json()


def paginate(endpoint: str, params: dict = None):
    params = dict(params or {})
    params.setdefault("limit", 500)   # fewer round-trips for large collections
    params.setdefault("offset", 0)
    while True:
        page = api_get(endpoint, params)
        if not page:
            break
        yield from page
        if len(page) < params["limit"]:
            break
        params["offset"] += params["limit"]


def list_folders(parent_id: str, parent_type: str):
    yield from paginate("folder", {"parentType": parent_type, "parentId": parent_id})


def list_items(folder_id: str):
    yield from paginate("item", {"folderId": folder_id})


def list_files(item_id: str):
    yield from paginate(f"item/{item_id}/files")


# ---------------------------------------------------------------------------
# Download a single file  (no HEAD request)
# ---------------------------------------------------------------------------

def download_file(
    file_id: str,
    dest_path: Path,
    known_size: int = 0,
    resume: bool = False,
    max_speed: int = 0,
) -> str:
    """
    Stream-download one Girder file.

    known_size  – size reported by the Girder item listing (bytes). Used to
                  decide whether the file is already complete without a HEAD
                  request.  0 means unknown.

    Returns one of: "downloaded", "resumed", "skipped", "error:<msg>"
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE_API}/file/{file_id}/download"
    session = get_session()

    existing = dest_path.stat().st_size if dest_path.exists() else 0

    # ── fast skip: we already know the target size from the listing ──────────
    if resume and known_size > 0 and existing >= known_size:
        return "skipped"

    # ── decide whether to resume a partial file ───────────────────────────
    headers = {}
    mode = "wb"
    start_byte = 0
    if resume and existing > 0:
        headers["Range"] = f"bytes={existing}-"
        mode = "ab"
        start_byte = existing

    resp = session.get(url, headers=headers, stream=True, timeout=120)

    # 416 = server says the range is beyond EOF → file is already complete
    if resp.status_code == 416:
        return "skipped"
    resp.raise_for_status()

    # Content-Length here is the remaining bytes (not total when using Range)
    content_len = int(resp.headers.get("Content-Length", 0))
    total = content_len + start_byte   # 0 if server doesn't tell us

    # Second-chance skip: server confirmed size via Content-Length on the GET
    if resume and total > 0 and existing >= total:
        resp.close()
        return "skipped"

    try:
        with open(dest_path, mode) as fh, tqdm(
            total=total or None,
            initial=start_byte,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dest_path.name[-48:],
            leave=False,
            position=None,   # tqdm picks a free row in multi-thread mode
        ) as bar:
            last_tick = time.monotonic()
            bytes_this_sec = 0

            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                fh.write(chunk)
                n = len(chunk)
                bar.update(n)

                if max_speed > 0:
                    bytes_this_sec += n
                    elapsed = time.monotonic() - last_tick
                    if elapsed < 1.0:
                        if bytes_this_sec >= max_speed:
                            time.sleep(1.0 - elapsed)
                            last_tick = time.monotonic()
                            bytes_this_sec = 0
                    else:
                        last_tick = time.monotonic()
                        bytes_this_sec = 0
    except Exception as exc:
        return f"error:{exc}"

    return "resumed" if start_byte > 0 else "downloaded"


# ---------------------------------------------------------------------------
# Stream folder tree — yield (gfile, dest_path) as they are discovered,
# printing each folder name immediately so the user sees API progress.
# ---------------------------------------------------------------------------

def iter_files(folder: dict, dest_root: Path):
    """
    Generator: walk a Girder folder tree and yield (gfile_dict, dest_path)
    one file at a time.  Each folder name is logged the moment it is entered,
    so the user sees steady output even before any download starts.
    """
    folder_dest = dest_root / folder["name"]
    log.info("  📁 %s", folder["name"])

    for item in list_items(folder["_id"]):
        for gfile in list_files(item["_id"]):
            fname = gfile.get("name", gfile["_id"])
            log.info("    📄 %-40s %s", fname, _human_size(gfile.get("size", 0)))
            yield gfile, folder_dest / fname

    for sub in list_folders(folder["_id"], "folder"):
        yield from iter_files(sub, folder_dest)


# ---------------------------------------------------------------------------
# Download one patient folder  (runs inside a thread-pool worker)
# ---------------------------------------------------------------------------

def download_patient(
    folder: dict,
    dest_root: Path,
    resume: bool,
    max_speed: int,
) -> dict:
    """
    Download every file in one patient folder.

    Files are downloaded as soon as they are discovered — there is no
    silent pre-scan phase.  Each folder/file is logged immediately on
    discovery so the terminal always shows what the worker is doing.
    """
    stats = {"downloaded": 0, "resumed": 0, "skipped": 0, "errors": 0, "bytes": 0}

    for gfile, fpath in iter_files(folder, dest_root):
        known_size = gfile.get("size", 0)
        result = download_file(
            gfile["_id"], fpath,
            known_size=known_size,
            resume=resume,
            max_speed=max_speed,
        )
        if result == "downloaded":
            stats["downloaded"] += 1
            stats["bytes"] += known_size
        elif result == "resumed":
            stats["resumed"] += 1
            stats["bytes"] += known_size
        elif result == "skipped":
            stats["skipped"] += 1
        else:
            log.error("    ❌ %s — %s", fpath.name, result)
            stats["errors"] += 1

    return stats


# ---------------------------------------------------------------------------
# Top-level collection downloader
# ---------------------------------------------------------------------------

def download_collection(
    collection_key: str,
    output_dir: Path,
    resume: bool,
    max_speed: int,
    workers: int,
    skip_until: str | None = None,
    patient_filter: str | None = None,
):
    info = COLLECTIONS[collection_key]
    coll_id = info["id"]
    coll_name = info["name"]
    dest = output_dir / coll_name

    log.info("=" * 60)
    log.info("Collection : %s", coll_name)
    log.info("Description: %s", info["description"])
    log.info("Destination: %s", dest)
    log.info("Workers    : %d parallel patient downloads", workers)
    log.info("=" * 60)

    try:
        meta = api_get(f"collection/{coll_id}")
        log.info("✅ Collection found: %s  (size: %s)",
                 meta.get("name", coll_name), _human_size(meta.get("size", 0)))
    except FileNotFoundError:
        log.error("❌ Collection '%s' (id=%s) not found or not yet released.", coll_name, coll_id)
        return
    except PermissionError as exc:
        log.error("❌ %s", exc)
        return

    # ── Collect ALL folders at every depth, then filter ────────────────────
    #
    # CAMUS layout on this platform:
    #   collection
    #     database_nifti/   ← wrapper folder (not a patient)
    #       patient0001/
    #       patient0002/
    #       ...
    #     database_split/   ← metadata only
    #     jupyter/          ← notebooks only
    #
    # --skip-until and --patient must match against the full flattened list,
    # not just the 3 top-level wrappers.

    log.info("⏳ Fetching full folder tree from API …")

    def collect_all_folders(parent_id: str, parent_type: str) -> list:
        """Return every folder in the tree, depth-first."""
        result = []
        for f in list_folders(parent_id, parent_type):
            result.append(f)
            result.extend(collect_all_folders(f["_id"], "folder"))
        return result

    all_folders = collect_all_folders(coll_id, "collection")
    log.info("✅ Found %d total folders (all depths).", len(all_folders))

    if not all_folders:
        log.error("❌ No folders found. "
                  "Browse https://humanheart-project.creatis.insa-lyon.fr/database/#collection/%s", coll_id)
        return

    # ── apply --patient filter ───────────────────────────────────────────────
    if patient_filter:
        matched = [f for f in all_folders if f["name"].lower() == patient_filter.lower()]
        if not matched:
            sample = sorted(set(f["name"] for f in all_folders))[:20]
            log.error("❌ Patient '%s' not found. Sample folder names:\n   %s",
                      patient_filter, "\n   ".join(sample))
            return
        log.info("🔍 Patient filter — downloading only: %s", matched[0]["name"])
        top_folders = matched

    # ── apply --skip-until ───────────────────────────────────────────────────
    elif skip_until:
        names_lower = [f["name"].lower() for f in all_folders]
        target = skip_until.lower()
        if target not in names_lower:
            # show folders whose name starts with the same prefix to help typo-fixing
            sample = sorted(set(f["name"] for f in all_folders
                                if f["name"].lower().startswith(target[:4])))[:10]
            hint = (("  Closest matches: " + ", ".join(sample)) if sample
                    else "  Run with --list-folders to see all available names.")
            log.warning("⚠️  --skip-until '%s' not matched.\n%s", skip_until, hint)
            top_folders = all_folders   # fall back: download everything
        else:
            idx = names_lower.index(target)
            top_folders = all_folders[idx:]
            log.info("⏩ Skipping %d folders, starting from '%s' (%d of %d total).",
                     idx, all_folders[idx]["name"], idx + 1, len(all_folders))

    else:
        top_folders = all_folders

    # ── parallel download ────────────────────────────────────────────────────
    totals = {"downloaded": 0, "resumed": 0, "skipped": 0, "errors": 0, "bytes": 0}
    n = len(top_folders)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(download_patient, folder, dest, resume, max_speed): folder["name"]
            for folder in top_folders
        }
        with tqdm(total=n, unit="patient", desc=coll_name, position=0) as pbar:
            for future in as_completed(futures):
                name = futures[future]
                try:
                    s = future.result()
                    for k in totals:
                        totals[k] += s.get(k, 0)
                    pbar.set_postfix_str(f"last: {name}")
                except Exception as exc:
                    log.error("❌ Patient %s failed: %s", name, exc)
                    totals["errors"] += 1
                finally:
                    pbar.update(1)

    log.info("")
    log.info("── %s summary ─────────────────────────────────", coll_name)
    log.info("   Downloaded : %d files  (%s)", totals["downloaded"], _human_size(totals["bytes"]))
    log.info("   Resumed    : %d files (partial → completed)", totals["resumed"])
    log.info("   Skipped    : %d files (already complete)", totals["skipped"])
    log.info("   Errors     : %d", totals["errors"])


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def authenticate(username: str, password: str, api_key: str):
    """Set auth token on the calling thread's session (and the thread-local default)."""
    s = get_session()
    if api_key:
        log.info("Authenticating with API key …")
        resp = s.post(f"{BASE_API}/api_key/token",
                      params={"key": api_key, "duration": 1440}, timeout=30)
        resp.raise_for_status()
        token = resp.json()["authToken"]["token"]
    elif username and password:
        log.info("Authenticating as %s …", username)
        resp = s.get(f"{BASE_API}/user/authentication",
                     auth=(username, password), timeout=30)
        resp.raise_for_status()
        token = resp.json()["authToken"]["token"]
    else:
        log.info("No credentials — proceeding as anonymous user.")
        return

    # Store for use by worker threads via a shared token variable
    # Workers call get_session() which creates a fresh Session; we patch the
    # default headers by keeping the token in a module-level variable so each
    # new thread session can inherit it.
    global _auth_token
    _auth_token = token
    s.headers.update({"Girder-Token": token})
    log.info("✅ Authenticated.")


_auth_token: str = ""


# Monkey-patch get_session so worker threads also pick up the auth token
_orig_get_session = get_session


def get_session() -> requests.Session:  # noqa: F811
    s = _orig_get_session()
    if _auth_token and "Girder-Token" not in s.headers:
        s.headers.update({"Girder-Token": _auth_token})
    return s


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download CAMUS / ACDC datasets from the Human Heart Project.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Requirements")[0],
    )
    p.add_argument(
        "--datasets", nargs="+", choices=["camus", "acdc"],
        default=["camus", "acdc"],
        help="Which dataset(s) to download (default: both).",
    )
    p.add_argument(
        "--output", "-o", default="./cardiac_datasets",
        help="Root directory for downloaded files (default: ./cardiac_datasets).",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Skip files already fully downloaded. No HEAD requests — uses "
             "local file size vs Girder-reported size.",
    )
    p.add_argument(
        "--workers", type=int, default=4, metavar="N",
        help="Number of parallel patient-folder downloads (default: 4). "
             "Try 8 on a fast connection; lower to 2 if you get rate-limited.",
    )
    p.add_argument(
        "--skip-until", metavar="FOLDER_NAME", default=None,
        help="Skip all patient folders before this name, then download from "
             "here onward.  Example: --skip-until patient_051",
    )
    p.add_argument(
        "--patient", metavar="FOLDER_NAME", default=None,
        help="Download only this one patient folder. "
             "Example: --patient patient_051",
    )
    p.add_argument(
        "--max-speed", type=int, default=0, metavar="BYTES_PER_SEC",
        help="Throttle total download speed in bytes/sec (0 = unlimited).",
    )
    p.add_argument("--username", default="", help="Girder username.")
    p.add_argument("--password", default="", help="Girder password.")
    p.add_argument("--api-key", default="", help="Girder API key.")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        authenticate(args.username, args.password, args.api_key)
    except Exception as exc:
        log.error("Authentication failed: %s", exc)
        sys.exit(1)

    log.info("Output directory : %s", output_dir)
    log.info("Datasets         : %s", ", ".join(args.datasets))
    log.info("Workers          : %d", args.workers)
    log.info("Resume mode      : %s", "ON" if args.resume else "OFF")
    if args.skip_until:
        log.info("Skip-until       : %s", args.skip_until)
    if args.patient:
        log.info("Patient filter   : %s", args.patient)
    if args.max_speed:
        log.info("Speed limit      : %s/s", _human_size(args.max_speed))

    for key in args.datasets:
        try:
            download_collection(
                collection_key=key,
                output_dir=output_dir,
                resume=args.resume,
                max_speed=args.max_speed,
                workers=args.workers,
                skip_until=args.skip_until,
                patient_filter=args.patient,
            )
        except KeyboardInterrupt:
            log.warning("Interrupted. Run with --resume to continue later.")
            sys.exit(130)
        except Exception as exc:
            log.exception("Unexpected error on %s: %s", key.upper(), exc)

    log.info("Done.")


if __name__ == "__main__":
    main()