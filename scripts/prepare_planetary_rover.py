#!/usr/bin/env python3
"""Build an evaluation-only Mars + Moon rover dataset from official archives.

The output layout is deliberately identical to the navigation datasets read by
``BaseDataset``::

    <output>/<trajectory>/{0.jpg, 1.jpg, ..., traj_data.pkl}

``traj_data.pkl`` contains only ``position`` and ``yaw`` numpy arrays.  Rich
provenance is stored beside it in JSON/JSONL files and is therefore never read
or cast by the training data loader.

No visual odometry, SLAM, learned pose estimator, simulated image, or
interpolated image is used.  Mars image-time telemetry comes from each PDS4
image label and is tied to the officially localized PLACES ``best_interp``
path.  Lunar position and attitude come directly from CE4 PCAM 2B labels.
"""

from __future__ import annotations

import argparse
import bisect
import concurrent.futures
import csv
import datetime as dt
import hashlib
import http.client
import io
import json
import math
import os
import pickle
import re
import shutil
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from collections import defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ATLAS_SEARCH_URL = "https://pds-imaging.jpl.nasa.gov/api/search/atlas/_search"
ATLAS_CDN_ROOT = "https://d1ejlg980osaur.cloudfront.net/m20"
MARS_PLACES_ROOT = (
    "https://pds-geosciences.wustl.edu/m2020/urn-nasa-pds-mars2020_rover_places"
)
MOON_ROOT = "https://moon.bao.ac.cn"
MOON_API_ROOT = f"{MOON_ROOT}/moon-admin/client/science"

PLACES_FILES = {
    "best_interp.csv": (
        f"{MARS_PLACES_ROOT}/data_localizations/best_interp.csv",
        "58fbc62f2676195a00b40b2da53a797a70284a8d2344b288c999d325b7ccb4cc",
    ),
    "best_interp.csv.xml": (
        f"{MARS_PLACES_ROOT}/data_localizations/best_interp.csv.xml",
        "03a382042c73d7a7eaced1c32f64108ed42c07f2139bbcc815cb4ab882eac70b",
    ),
    "best_tactical.csv": (
        f"{MARS_PLACES_ROOT}/data_localizations/best_tactical.csv",
        "9da19e4f70287fe0d33ba716ff2169c10447c30c7a048583f730e4155226cde9",
    ),
    "best_tactical.csv.xml": (
        f"{MARS_PLACES_ROOT}/data_localizations/best_tactical.csv.xml",
        "777074bc03a3f06be8241b85cd4448cbdac5bff4dfecc443de6757152d22ac79",
    ),
    "telemetry.csv": (
        f"{MARS_PLACES_ROOT}/data_localizations/telemetry.csv",
        "8ac01f021f584273a6ce5025b53e79b53c8e9ea128523eb58a2576dececf84ca",
    ),
    "telemetry.csv.xml": (
        f"{MARS_PLACES_ROOT}/data_localizations/telemetry.csv.xml",
        "f73ed4f2782613a6a3ffba2b6f3140c6897f371caa49264a26b4864c0e96b527",
    ),
    "Mars2020_Rover_PLACES_PDS_SIS.pdf": (
        f"{MARS_PLACES_ROOT}/document/Mars2020_Rover_PLACES_PDS_SIS.pdf",
        "ca3eb9d2a0d992cedf96860f30537ac508e15fb0e2417fb74c5b80cb31418d1a",
    ),
}

MARS_BUNDLE = "mars2020_navcam_ops_raw"
MARS_ACTIVITY_RE = re.compile(
    r"^[A-Z]{3}_\d{4}_\d{10}_\d{3}[A-Z]{3}_"
    r"[A-Z]\d{7}([A-Z0-9_]{4})"
)
MOON_SEQUENCE_RE = re.compile(r"_(\d{4})_B\.2BL$")
MIN_TRAJECTORY_FRAMES = 68


@dataclass(frozen=True)
class Similarity2D:
    origin_in: np.ndarray
    origin_out: np.ndarray
    matrix: np.ndarray
    lower_key: tuple[int, int]
    upper_key: tuple[int, int]

    def apply(self, point: np.ndarray) -> np.ndarray:
        return self.origin_out + self.matrix @ (point - self.origin_in)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_write_text(path: Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode("utf-8"))


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    atomic_write_text(
        path,
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
    )


def wrap_radians(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def wrap_degrees(angle: float) -> float:
    return (angle + 180.0) % 360.0 - 180.0


def circular_mean(values: Sequence[float]) -> float:
    vector = np.mean(np.exp(1j * np.asarray(values, dtype=np.float64)))
    if abs(vector) < 1e-12:
        raise ValueError("Circular mean is undefined")
    return float(np.angle(vector))


def circular_std_degrees(values: Sequence[float]) -> float:
    vector = abs(np.mean(np.exp(1j * np.asarray(values, dtype=np.float64))))
    vector = min(1.0, max(float(vector), np.finfo(np.float64).tiny))
    return math.degrees(math.sqrt(max(0.0, -2.0 * math.log(vector))))


def force_ipv4() -> None:
    """Work around hosts whose resolver returns unreachable IPv6 CDN records."""

    original = socket.getaddrinfo
    if getattr(original, "_planetary_ipv4", False):
        return

    def ipv4_getaddrinfo(
        host: str | bytes | None,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[Any]:
        # Asking explicitly for AF_INET is more reliable than making an
        # AF_UNSPEC query and filtering it afterwards: the latter intermittently
        # returns only CloudFront AAAA records in this IPv4-only environment.
        requested_family = socket.AF_INET if family in {0, socket.AF_UNSPEC} else family
        return original(host, port, requested_family, type, proto, flags)

    ipv4_getaddrinfo._planetary_ipv4 = True
    socket.getaddrinfo = ipv4_getaddrinfo  # type: ignore[assignment]


def make_opener(*, insecure: bool = False) -> urllib.request.OpenerDirector:
    handlers: list[Any] = [urllib.request.ProxyHandler({})]
    if insecure:
        # The official CLEP host currently presents an expired certificate.
        # Payload hashes are recorded and PDS labels remain self-describing.
        context = ssl._create_unverified_context()
        handlers.append(urllib.request.HTTPSHandler(context=context))
    opener = urllib.request.build_opener(*handlers)
    opener.addheaders = [("User-Agent", "CompACT-planetary-rover-dataset/1.0")]
    return opener


def fetch_bytes(
    url: str,
    *,
    data: bytes | None = None,
    headers: dict[str, str] | None = None,
    insecure: bool = False,
    retries: int = 10,
    timeout: float = 90.0,
) -> bytes:
    request_headers = dict(headers or {})
    if data is not None:
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, headers=request_headers)
    opener = make_opener(insecure=insecure)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.read()
        except (
            OSError,
            http.client.IncompleteRead,
            urllib.error.URLError,
            urllib.error.HTTPError,
        ) as error:
            last_error = error
            if isinstance(error, urllib.error.HTTPError) and error.code < 500:
                break
            time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"Unable to fetch {url}: {last_error}") from last_error


def fetch_json(url: str, **kwargs: Any) -> dict[str, Any]:
    return json.loads(fetch_bytes(url, **kwargs))


def ensure_download(
    path: Path,
    url: str,
    *,
    expected_sha256: str | None = None,
    insecure: bool = False,
) -> str:
    if path.is_file():
        actual = sha256_file(path)
        if expected_sha256 is None or actual == expected_sha256:
            return actual
        raise ValueError(f"Existing file has wrong SHA-256: {path}: {actual}")
    payload = fetch_bytes(url, insecure=insecure)
    actual = sha256_bytes(payload)
    if expected_sha256 is not None and actual != expected_sha256:
        raise ValueError(
            f"Downloaded file has wrong SHA-256: {url}: {actual} != {expected_sha256}"
        )
    atomic_write_bytes(path, payload)
    return actual


def atlas_to_cdn_url(uri: str, release: int) -> str:
    prefix = "atlas:pds4:mars_2020:perseverance:/"
    if not uri.startswith(prefix):
        raise ValueError(f"Unexpected Atlas URI: {uri}")
    return f"{ATLAS_CDN_ROOT}/r{release}/{uri[len(prefix) :]}"


def mars_release(record: dict[str, Any]) -> int:
    source = record["source"]
    for candidate in (
        source.get("release_id"),
        source.get("release_id_num"),
        source.get("archive", {}).get("release_id"),
        source.get("archive", {}).get("release_id_num"),
        source.get("gather", {}).get("pds_archive", {}).get("release_id"),
        source.get("gather", {}).get("pds_archive", {}).get("release_id_num"),
    ):
        if candidate is not None:
            return int(float(candidate))
    match = re.search(r"::(\d+)(?:\.0)?$", record.get("atlas_id", ""))
    if match:
        return int(match.group(1))
    raise ValueError(f"Missing release identifier: {record.get('atlas_id')}")


def query_mars_navcam_index(destination: Path) -> list[dict[str, Any]]:
    if destination.is_file():
        return read_jsonl(destination)
    filters = [
        {"term": {"archive.bundle_id": MARS_BUNDLE}},
        {"term": {"gather.common.product_type": "EDR"}},
        {"term": {"gather.common.kind": "regular"}},
        {"term": {"gather.landed_missions.land_msn_eye": "left"}},
    ]
    source_fields = [
        "release_id",
        "release_id_num",
        "archive.name",
        "archive.md5",
        "archive.size",
        "archive.release_id",
        "archive.release_id_num",
        "gather.landed_missions.rmc_site",
        "gather.landed_missions.rmc_drive",
        "gather.landed_missions.rmc_pose",
        "gather.landed_missions.planet_day_number",
        "gather.landed_missions.frame_type",
        "gather.landed_missions.site_instrument_azimuth",
        "gather.landed_missions.site_instrument_elevation",
        "gather.time.start_time",
        "gather.time.spacecraft_clock_start_count",
        "gather.pds_archive.release_id",
        "gather.pds_archive.release_id_num",
        "gather.pds_archive.product_id",
        "gather.pds_archive.related.label",
        "gather.pds_archive.related.browse",
        "pds4_label.pds:Axis_Array/pds:elements",
    ]
    after: dict[str, int] | None = None
    rows: list[dict[str, Any]] = []
    while True:
        composite: dict[str, Any] = {
            # Ten-thousand-result aggregation pages intermittently time out at
            # the public Atlas gateway.  Smaller pages are slower but stable and
            # retain exactly the same deterministic composite ordering.
            "size": 1_000,
            "sources": [
                {"site": {"terms": {"field": "gather.landed_missions.rmc_site"}}},
                {"drive": {"terms": {"field": "gather.landed_missions.rmc_drive"}}},
            ],
        }
        if after is not None:
            composite["after"] = after
        query = {
            "size": 0,
            "query": {"bool": {"filter": filters}},
            "aggs": {
                "rmc": {
                    "composite": composite,
                    "aggs": {
                        "product": {
                            "top_hits": {
                                "size": 1,
                                "_source": source_fields,
                                "sort": [
                                    {
                                        "gather.time.spacecraft_clock_start_count": {
                                            "order": "asc"
                                        }
                                    },
                                    {"_id": {"order": "asc"}},
                                ],
                            }
                        }
                    },
                }
            },
        }
        response = fetch_json(ATLAS_SEARCH_URL, data=json.dumps(query).encode("utf-8"))
        aggregation = response["aggregations"]["rmc"]
        buckets = aggregation["buckets"]
        for bucket in buckets:
            hit = bucket["product"]["hits"]["hits"][0]
            rows.append(
                {
                    "site": int(bucket["key"]["site"]),
                    "drive": int(bucket["key"]["drive"]),
                    "candidate_count": int(bucket["doc_count"]),
                    "atlas_id": hit["_id"],
                    "source": hit["_source"],
                }
            )
        print(f"Atlas RMC index: {len(rows)} rows", flush=True)
        after = aggregation.get("after_key")
        if not buckets or after is None:
            break
    write_jsonl(destination, rows)
    return rows


def query_moon_pcam_index(destination: Path) -> list[dict[str, Any]]:
    if destination.is_file():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        return payload["rows"] if isinstance(payload, dict) else payload
    params = {
        "category": 1,
        "task": 7,
        "load": 372,
        "dataLevel": 373,
        "dataCatalogueId": 672,
        "pageSize": 5000,
    }
    rows: list[dict[str, Any]] = []
    total: int | None = None
    page = 1
    while total is None or len(rows) < total:
        params["pageNum"] = page
        url = f"{MOON_API_ROOT}/dataInfoList?{urllib.parse.urlencode(params)}"
        payload = fetch_json(url, insecure=True)
        if payload.get("code") != 200:
            raise RuntimeError(f"CLEP index request failed: {payload}")
        total = int(payload["total"])
        page_rows = payload["rows"]
        if not page_rows:
            break
        rows.extend(page_rows)
        page += 1
        print(f"CLEP PCAM index: {len(rows)}/{total} rows", flush=True)
    if total is None or len(rows) != total:
        raise ValueError(f"Incomplete CLEP index: {len(rows)} / {total}")
    labels = [row for row in rows if str(row.get("name", "")).endswith(".2BL")]
    write_json(
        destination,
        {
            "source": f"{MOON_API_ROOT}/dataInfoList",
            "query": params,
            "total_products": total,
            "total": len(labels),
            "rows": labels,
        },
    )
    return labels


def ensure_source_indexes(
    cache: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sources = cache / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, dict[str, str]] = {}
    for name, (url, expected) in PLACES_FILES.items():
        actual = ensure_download(sources / name, url, expected_sha256=expected)
        hashes[name] = {"url": url, "sha256": actual}
    mars = query_mars_navcam_index(sources / "mars_navcam_rmc_index.jsonl")
    moon = query_moon_pcam_index(sources / "ce4_pcam_2b_label_index.json")
    hashes["mars_navcam_rmc_index.jsonl"] = {
        "url": ATLAS_SEARCH_URL,
        "sha256": sha256_file(sources / "mars_navcam_rmc_index.jsonl"),
    }
    hashes["ce4_pcam_2b_label_index.json"] = {
        "url": f"{MOON_API_ROOT}/dataInfoList",
        "sha256": sha256_file(sources / "ce4_pcam_2b_label_index.json"),
    }
    manifest_path = sources / "source_manifest.json"
    manifest = {
        "mars_places_bundle": "urn:nasa:pds:mars2020_rover_places::16.0",
        "mars_places_doi": "10.17189/btz6-5a82",
        "mars_navcam_bundle": MARS_BUNDLE,
        "mars_navcam_doi": "10.17189/d3nm-pp09",
        "moon_provider": "中国探月工程科学数据发布系统 / CLEP",
        "moon_pcam_catalogue_id": 672,
        "moon_pcam_catalogue_url": f"{MOON_API_ROOT}/catalogue/672",
        "moon_pcam_dois": [
            "10.12350/CLPDS.GRAS.CE4.PCAM-2B-2019.vB",
            "10.12350/CLPDS.GRAS.CE4.PCAM-2B-2020.vB",
            "10.12350/CLPDS.GRAS.CE4.PCAM-2B-2021.vB",
        ],
        "moon_tls_verification_disabled": True,
        "files": hashes,
    }
    created_utc = dt.datetime.now(dt.UTC).isoformat()
    if manifest_path.is_file():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing_created_utc = existing_manifest.pop("created_utc", None)
        if existing_manifest == manifest and existing_created_utc:
            created_utc = existing_created_utc
    write_json(manifest_path, {"created_utc": created_utc, **manifest})
    return mars, moon


def mars_activity(name: str) -> str | None:
    match = MARS_ACTIVITY_RE.match(name)
    return match.group(1) if match else None


def atlas_rover_azimuth(record: dict[str, Any]) -> float | None:
    landed = record["source"].get("gather", {}).get("landed_missions", {})
    value = landed.get("site_instrument_azimuth")
    return None if value is None else wrap_degrees(float(value))


def select_mars_index(
    records: Sequence[dict[str, Any]], *, require_release: bool = True
) -> list[dict[str, Any]]:
    selected = []
    for record in records:
        source = record["source"]
        name = source["archive"]["name"]
        if mars_activity(name) not in {"VCE_", "TRAV"}:
            continue
        azimuth = atlas_rover_azimuth(record)
        if azimuth is None or abs(azimuth) > 5.0:
            continue
        related = source["gather"]["pds_archive"]["related"]
        if "label" not in related or "browse" not in related:
            continue
        if require_release:
            try:
                mars_release(record)
            except (KeyError, TypeError, ValueError):
                # Atlas can expose products that have not been assigned to a
                # published release.  They have no immutable official CDN
                # object (verified by the absent release ID), so they are not
                # usable.
                continue
        selected.append(record)
    return selected


def resolve_moon_label_urls(
    rows: Sequence[dict[str, Any]], destination: Path, workers: int
) -> dict[str, str]:
    existing: dict[str, str] = {}
    if destination.is_file():
        existing = {row["name"]: row["url"] for row in read_jsonl(destination)}
    wanted = [row for row in rows if "PCAML-" in row["name"]]

    def resolve(row: dict[str, Any]) -> tuple[str, str]:
        name = row["name"]
        if name in existing:
            return name, existing[name]
        endpoint = f"{MOON_API_ROOT}/dataInfo/getAnnexZip/{row['dataInfoId']}"
        response = fetch_json(endpoint, insecure=True)
        if response.get("code") != 200 or not response.get("data"):
            raise RuntimeError(f"Could not resolve CLEP product {name}: {response}")
        return name, str(response["data"])

    pending = [row for row in wanted if row["name"] not in existing]
    failures: list[tuple[str, Exception]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(resolve, row): row["name"] for row in pending}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            name = futures[future]
            try:
                resolved_name, url = future.result()
                existing[resolved_name] = url
            except Exception as error:  # noqa: BLE001 - aggregate worker failures
                failures.append((name, error))
            total_done = len(wanted) - len(pending) + completed
            if total_done % 250 == 0 or completed == len(pending):
                checkpoint = [
                    {"name": row["name"], "url": existing[row["name"]]}
                    for row in wanted
                    if row["name"] in existing
                ]
                write_jsonl(destination, checkpoint)
                print(
                    f"Resolved lunar labels: {total_done}/{len(wanted)} "
                    f"({len(failures)} failures)",
                    flush=True,
                )
    if failures:
        name, error = failures[0]
        raise RuntimeError(
            f"Failed to resolve {len(failures)} lunar label URLs; first={name}: {error}"
        ) from error
    ordered = [{"name": row["name"], "url": existing[row["name"]]} for row in wanted]
    write_jsonl(destination, ordered)
    return existing


def download_label_set(
    jobs: Sequence[tuple[Path, str, bool]], workers: int, description: str
) -> None:
    def fetch(job: tuple[Path, str, bool]) -> Path:
        path, url, insecure = job
        ensure_download(path, url, insecure=insecure)
        return path

    failures: list[tuple[Path, Exception]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(fetch, job): job[0] for job in jobs}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - aggregate worker failures
                failures.append((futures[future], error))
            if index % 250 == 0 or index == len(jobs):
                print(
                    f"{description}: {index}/{len(jobs)} ({len(failures)} failures)",
                    flush=True,
                )
    if failures:
        path, error = failures[0]
        raise RuntimeError(
            f"{description} had {len(failures)} failures; first={path}: {error}"
        ) from error


def fetch_labels(
    cache: Path,
    mars_index: Sequence[dict[str, Any]],
    moon_index: Sequence[dict[str, Any]],
    workers: int,
) -> None:
    mars_selected = select_mars_index(mars_index)
    mars_jobs: list[tuple[Path, str, bool]] = []
    for record in mars_selected:
        source = record["source"]
        release = mars_release(record)
        uri = source["gather"]["pds_archive"]["related"]["label"]["uri"]
        name = Path(urllib.parse.urlparse(atlas_to_cdn_url(uri, release)).path).name
        mars_jobs.append(
            (cache / "labels" / "mars" / name, atlas_to_cdn_url(uri, release), False)
        )
    download_label_set(mars_jobs, workers, "Mars PDS4 labels")

    url_map = resolve_moon_label_urls(
        moon_index, cache / "sources" / "ce4_pcam_left_label_urls.jsonl", workers
    )
    moon_jobs = [
        (cache / "labels" / "moon" / row["name"], url_map[row["name"]], True)
        for row in moon_index
        if "PCAML-" in row["name"]
    ]
    download_label_set(moon_jobs, workers, "Moon PCAM labels")


def local_name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def child(element: ET.Element, name: str) -> ET.Element:
    for candidate in element:
        if local_name(candidate) == name:
            return candidate
    raise KeyError(f"Missing child {name} below {local_name(element)}")


def child_text(element: ET.Element, name: str) -> str:
    value = child(element, name).text
    if value is None:
        raise ValueError(f"Empty element {name}")
    return value.strip()


def descendants(element: ET.Element, name: str) -> Iterator[ET.Element]:
    for candidate in element.iter():
        if local_name(candidate) == name:
            yield candidate


def vector3(element: ET.Element, names: tuple[str, str, str]) -> np.ndarray:
    return np.asarray(
        [float(child_text(element, name)) for name in names], dtype=np.float64
    )


def quaternion_yaw(quaternion: np.ndarray) -> float:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion).all() or norm < 1e-12:
        raise ValueError(f"Invalid quaternion: {quaternion.tolist()}")
    w, x, y, z = quaternion / norm
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def parse_mars_label(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    rover_definition: ET.Element | None = None
    for definition in descendants(root, "Coordinate_Space_Definition"):
        identifiers = [
            (item.text or "").strip()
            for item in definition
            if local_name(item) == "local_identifier"
        ]
        if not any(
            identifier.startswith("ROVER_NAV_FRAME_") for identifier in identifiers
        ):
            continue
        solution_ids = [
            (item.text or "").strip() for item in descendants(definition, "solution_id")
        ]
        if "TELEMETRY" in solution_ids or any(
            identifier.endswith("_TELEMETRY") for identifier in identifiers
        ):
            rover_definition = definition
            break
    if rover_definition is None:
        raise ValueError(f"No telemetry ROVER_NAV_FRAME in {path}")
    indexed = next(descendants(rover_definition, "Coordinate_Space_Indexed"))
    counters: dict[str, int] = {}
    for index in descendants(indexed, "Coordinate_Space_Index"):
        counters[child_text(index, "index_id")] = int(
            child_text(index, "index_value_number")
        )
    origin = vector3(
        next(descendants(rover_definition, "Vector_Origin_Offset")),
        ("x_position", "y_position", "z_position"),
    )
    quaternion = vector3(
        next(descendants(rover_definition, "Quaternion_Plus_Direction")),
        ("qsin1", "qsin2", "qsin3"),
    )
    quaternion = np.concatenate(
        [
            [
                float(
                    child_text(
                        next(
                            descendants(rover_definition, "Quaternion_Plus_Direction")
                        ),
                        "qcos",
                    )
                )
            ],
            quaternion,
        ]
    )
    rover_identifier = next(
        identifier
        for identifier in [
            (item.text or "").strip()
            for item in rover_definition
            if local_name(item) == "local_identifier"
        ]
        if identifier.startswith("ROVER_NAV_FRAME_")
        and not identifier.endswith("_TELEMETRY")
    )
    instrument_azimuth = None
    instrument_elevation = None
    for geometry in descendants(root, "Derived_Geometry"):
        references = [
            (item.text or "").strip()
            for item in descendants(geometry, "local_identifier_reference")
        ]
        if rover_identifier not in references:
            continue
        try:
            instrument_azimuth = float(child_text(geometry, "instrument_azimuth"))
            instrument_elevation = float(child_text(geometry, "instrument_elevation"))
            break
        except KeyError:
            continue
    if instrument_azimuth is None:
        raise ValueError(f"Missing rover-frame instrument azimuth in {path}")
    clocks = [
        float((item.text or "").strip())
        for item in descendants(root, "spacecraft_clock_start")
    ]
    if not clocks:
        raise ValueError(f"Missing spacecraft clock in {path}")
    return {
        "site": counters["SITE"],
        "drive": counters["DRIVE"],
        "pose": counters["POSE"],
        "origin_site": origin.tolist(),
        "quaternion_wxyz": quaternion.tolist(),
        "yaw_rad": quaternion_yaw(quaternion),
        "instrument_azimuth_deg": wrap_degrees(instrument_azimuth),
        "instrument_elevation_deg": instrument_elevation,
        "sclk": clocks[0],
        "label_sha256": sha256_file(path),
    }


def parse_moon_label(path: Path, source_url: str) -> dict[str, Any]:
    root = ET.parse(path).getroot()
    mission = next(descendants(root, "Mission_Area"))
    location_xyz = next(descendants(mission, "Rover_LocationXYZ"))
    location = next(descendants(mission, "Rover_Location"))
    pointing = next(descendants(mission, "Vector_Cartesian_3_Pointing"))
    center = next(descendants(pointing, "center_point_observe_vector"))
    exterior = next(descendants(mission, "Exterior_Orientation_Elements"))
    rotation = next(descendants(mission, "Rotation_Angle"))
    axes: dict[str, int] = {}
    for axis in descendants(root, "Axis_Array"):
        axes[child_text(axis, "axis_name")] = int(child_text(axis, "elements"))
    data_type = child_text(next(descendants(root, "Element_Array")), "data_type")
    angles = {
        "roll_deg": float(child_text(exterior, "camera_rotation_angle_roll")),
        "pitch_deg": float(child_text(exterior, "camera_rotation_angle_pitch")),
        "yaw_deg": float(child_text(exterior, "camera_rotation_angle_yaw")),
    }
    center_vector = vector3(center, ("x", "y", "z"))
    mast_pitch_deg = float(child_text(rotation, "pitch"))
    mast_yaw_deg = float(child_text(rotation, "yawing"))
    camera_yaw = ce4_camera_yaw(
        angles["roll_deg"], angles["pitch_deg"], angles["yaw_deg"]
    )
    rover_yaw = ce4_rover_yaw(center_vector, mast_yaw_deg)
    start_time = child_text(
        next(descendants(root, "Time_Coordinates")), "start_date_time"
    )
    sequence_id = child_text(mission, "sequence_id")
    filename_sequence = MOON_SEQUENCE_RE.search(path.name)
    if filename_sequence is None or filename_sequence.group(1) != sequence_id:
        raise ValueError(
            f"CE4 filename/label sequence mismatch: {path.name} / {sequence_id}"
        )
    return {
        "name": path.name,
        "product_id": child_text(mission, "product_id"),
        "sequence_id": sequence_id,
        "start_time": start_time,
        "rover_xyz": vector3(location_xyz, ("x", "y", "z")).tolist(),
        "longitude_deg": float(child_text(location, "longitude")),
        "latitude_deg": float(child_text(location, "latitude")),
        "center_rover": center_vector.tolist(),
        **angles,
        "mast_pitch_deg": mast_pitch_deg,
        "mast_yaw_deg": mast_yaw_deg,
        "camera_yaw_rad": camera_yaw,
        "rover_yaw_rad": rover_yaw,
        "lines": axes["Line"],
        "samples": axes["Sample"],
        "data_type": data_type,
        "label_url": source_url,
        "image_url": source_url.removesuffix("L"),
        "label_sha256": sha256_file(path),
    }


def rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def ce4_camera_yaw(roll_deg: float, pitch_deg: float, yaw_deg: float) -> float:
    """Project the official CE4 exterior orientation's +Z optical axis."""

    pitch, roll, yaw = np.radians([pitch_deg, roll_deg, yaw_deg])
    matrix = rotation_x(pitch) @ rotation_y(roll) @ rotation_z(yaw)
    optical = matrix @ np.asarray([0.0, 0.0, 1.0])
    return math.atan2(float(optical[1]), float(optical[0]))


def ce4_rover_yaw(center_rover: np.ndarray, mast_yaw_deg: float) -> float:
    """Recover body yaw from two official PCAM pointing quantities.

    The label provides the optical-center pointing vector in
    ``ROVER_COORDINATE_SYSTEM`` and the mast yaw in ``Rotation_Angle``. Their
    wrapped azimuth difference removes the camera pan and leaves rover body
    heading. Independent images in one panorama must recover the same value.
    """

    center_rover = np.asarray(center_rover, dtype=np.float64)
    if center_rover.shape != (3,) or not np.isfinite(center_rover).all():
        raise ValueError(f"Invalid CE4 center vector: {center_rover}")
    pointing_yaw = math.atan2(float(center_rover[1]), float(center_rover[0]))
    return float(wrap_radians(pointing_yaw - math.radians(mast_yaw_deg)))


def load_places_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def highest_pose_by_rmc(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[int, int], dict[str, str]]:
    result: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        if row["frame"] != "ROVER":
            continue
        key = (int(row["site"]), int(row["drive"]))
        if key not in result or int(row["pose"]) > int(result[key]["pose"]):
            result[key] = row
    return result


def rover_by_rmc(
    rows: Iterable[dict[str, str]],
) -> dict[tuple[int, int], dict[str, str]]:
    return {
        (int(row["site"]), int(row["drive"])): row
        for row in rows
        if row["frame"] == "ROVER"
    }


def site_origins(rows: Iterable[dict[str, str]]) -> dict[int, np.ndarray]:
    return {
        int(row["site"]): np.asarray(
            [float(row["northing"]), float(row["easting"])], dtype=np.float64
        )
        for row in rows
        if row["frame"] == "SITE"
    }


def row_xy(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row["northing"]), float(row["easting"])], dtype=np.float64)


class MarsLocalizer:
    """Tie image-time telemetry to the official PLACES best_interp path."""

    def __init__(
        self, telemetry_rows: list[dict[str, str]], interp_rows: list[dict[str, str]]
    ):
        self.telemetry = highest_pose_by_rmc(telemetry_rows)
        self.interp = rover_by_rmc(interp_rows)
        self.site_origin = site_origins(telemetry_rows)
        self.keys_by_site: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for key in sorted(set(self.telemetry) & set(self.interp)):
            self.keys_by_site[key[0]].append(key)

    def telemetry_global_xy(self, label: dict[str, Any]) -> np.ndarray:
        site = int(label["site"])
        if site not in self.site_origin:
            raise KeyError(f"No telemetry SITE frame {site}")
        return self.site_origin[site] + np.asarray(
            label["origin_site"][:2], dtype=np.float64
        )

    def transform_for(
        self, site: int, drive: int, minimum_baseline: float = 0.1
    ) -> Similarity2D:
        keys = self.keys_by_site.get(site, [])
        insertion = bisect.bisect_left(keys, (site, drive))
        lower = insertion - 1
        upper = insertion
        while lower >= 0 and upper < len(keys):
            lower_key, upper_key = keys[lower], keys[upper]
            delta_in = row_xy(self.telemetry[upper_key]) - row_xy(
                self.telemetry[lower_key]
            )
            if float(np.linalg.norm(delta_in)) >= minimum_baseline:
                delta_out = row_xy(self.interp[upper_key]) - row_xy(
                    self.interp[lower_key]
                )
                if float(np.linalg.norm(delta_out)) < 1e-12:
                    matrix = np.zeros((2, 2), dtype=np.float64)
                else:
                    angle = math.atan2(delta_out[1], delta_out[0]) - math.atan2(
                        delta_in[1], delta_in[0]
                    )
                    scale = float(np.linalg.norm(delta_out) / np.linalg.norm(delta_in))
                    c, s = math.cos(angle), math.sin(angle)
                    matrix = scale * np.asarray([[c, -s], [s, c]], dtype=np.float64)
                return Similarity2D(
                    origin_in=row_xy(self.telemetry[lower_key]),
                    origin_out=row_xy(self.interp[lower_key]),
                    matrix=matrix,
                    lower_key=lower_key,
                    upper_key=upper_key,
                )
            lower -= 1
            upper += 1
        raise ValueError(f"No local PLACES bracket for site={site} drive={drive}")

    def localize(self, label: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        key = (int(label["site"]), int(label["drive"]))
        if key in self.interp:
            return row_xy(self.interp[key]), {
                "method": "exact_best_interp_rmc",
                "best_interp_rmc": list(key),
            }
        transform = self.transform_for(*key)
        localized = transform.apply(self.telemetry_global_xy(label))
        return localized, {
            "method": "image_telemetry_plus_local_best_interp_similarity",
            "lower_best_interp_rmc": list(transform.lower_key),
            "upper_best_interp_rmc": list(transform.upper_key),
        }


def mars_source_fields(record: dict[str, Any]) -> tuple[str, str, str]:
    source = record["source"]
    release = mars_release(record)
    related = source["gather"]["pds_archive"]["related"]
    return (
        source["archive"]["name"],
        atlas_to_cdn_url(related["label"]["uri"], release),
        atlas_to_cdn_url(related["browse"]["uri"], release),
    )


def mars_pose_rejection_category(error: Exception) -> str:
    message = str(error)
    for prefix, category in (
        ("No local PLACES bracket", "localization_no_places_bracket"),
        ("No telemetry SITE frame", "localization_no_telemetry_site"),
        ("Atlas/label RMC mismatch", "atlas_label_rmc_mismatch"),
        ("Atlas/label SCLK mismatch", "atlas_label_sclk_mismatch"),
        ("No telemetry ROVER_NAV_FRAME", "label_no_telemetry_rover_frame"),
        ("Missing rover-frame instrument azimuth", "label_no_rover_camera_azimuth"),
    ):
        if message.startswith(prefix):
            return category
    return f"pose_error:{type(error).__name__}"


def build_mars_frames(
    cache: Path, records: Sequence[dict[str, Any]], workers: int
) -> list[dict[str, Any]]:
    sources = cache / "sources"
    telemetry_rows = load_places_csv(sources / "telemetry.csv")
    interp_rows = load_places_csv(sources / "best_interp.csv")
    localizer = MarsLocalizer(telemetry_rows, interp_rows)
    frames: list[dict[str, Any]] = []
    rejection: defaultdict[str, int] = defaultdict(int)
    rejection_examples: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    selected_records = select_mars_index(records)
    unversioned_candidates = len(
        select_mars_index(records, require_release=False)
    ) - len(selected_records)

    def build_one(
        record: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, str | None, str | None]:
        name, label_url, image_url = mars_source_fields(record)
        label_path = (
            cache / "labels" / "mars" / Path(urllib.parse.urlparse(label_url).path).name
        )
        try:
            label = parse_mars_label(label_path)
            expected_rmc = (int(record["site"]), int(record["drive"]))
            label_rmc = (int(label["site"]), int(label["drive"]))
            if label_rmc != expected_rmc:
                raise ValueError(
                    f"Atlas/label RMC mismatch: {expected_rmc} != {label_rmc}"
                )
            index_sclk = float(
                record["source"]["gather"]["time"]["spacecraft_clock_start_count"]
            )
            if not math.isclose(
                float(label["sclk"]), index_sclk, rel_tol=0.0, abs_tol=1e-3
            ):
                raise ValueError(
                    f"Atlas/label SCLK mismatch: {index_sclk} != {label['sclk']}"
                )
            if abs(float(label["instrument_azimuth_deg"])) > 5.0:
                return (
                    None,
                    "label_not_forward",
                    (f"instrument_azimuth_deg={label['instrument_azimuth_deg']}"),
                )
            position, localization = localizer.localize(label)
        except (KeyError, ValueError, ET.ParseError) as error:
            return None, mars_pose_rejection_category(error), str(error)
        landed = record["source"]["gather"]["landed_missions"]
        frame = {
            "source": "mars2020_perseverance_navcam",
            "name": name,
            "product_id": record["source"]["gather"]["pds_archive"]["product_id"],
            "atlas_id": record["atlas_id"],
            "release": mars_release(record),
            "site": int(label["site"]),
            "drive": int(label["drive"]),
            "pose": int(label["pose"]),
            "sol": int(landed["planet_day_number"]),
            "sclk": float(label["sclk"]),
            "position": position.tolist(),
            "yaw": float(label["yaw_rad"]),
            "telemetry_origin_site": label["origin_site"],
            "telemetry_quaternion_wxyz": label["quaternion_wxyz"],
            "camera_azimuth_rover_deg": label["instrument_azimuth_deg"],
            "camera_elevation_rover_deg": label["instrument_elevation_deg"],
            "activity": mars_activity(name),
            "label_url": label_url,
            "image_url": image_url,
            "label_sha256": label["label_sha256"],
            "raw_img_md5": record["source"]["archive"].get("md5"),
            "localization": localization,
        }
        return frame, None, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(build_one, record): record for record in selected_records
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = futures[future]
            frame, reason, detail = future.result()
            if frame is not None:
                frames.append(frame)
            else:
                assert reason is not None and detail is not None
                rejection[reason] += 1
                if len(rejection_examples[reason]) < 10:
                    rejection_examples[reason].append(
                        {
                            "name": record["source"]["archive"]["name"],
                            "detail": detail,
                        }
                    )
            if completed % 1000 == 0 or completed == len(selected_records):
                print(
                    f"Parsed Mars labels: {completed}/{len(selected_records)}",
                    flush=True,
                )
    frames.sort(key=lambda row: (row["sclk"], row["name"]))
    write_json(
        cache / "manifests" / "mars_frame_selection.json",
        {
            "input_rmc_records": len(records),
            "selected_frames": len(frames),
            "rejections": dict(rejection),
            "rejection_examples": dict(rejection_examples),
            "camera_filter": "VCE_/TRAV; absolute rover-frame azimuth <= 5 degrees",
            "one_frame_per_site_drive": True,
            "published_release_required": True,
            "unversioned_atlas_candidates_excluded": unversioned_candidates,
        },
    )
    return frames


def build_moon_frames(
    cache: Path, rows: Sequence[dict[str, Any]], workers: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    url_map = {
        row["name"]: row["url"]
        for row in read_jsonl(cache / "sources" / "ce4_pcam_left_label_urls.jsonl")
    }
    parsed: list[dict[str, Any]] = []
    left_rows = [row for row in rows if "PCAML-" in row["name"]]

    def parse_one(row: dict[str, Any]) -> dict[str, Any]:
        name = row["name"]
        return parse_moon_label(cache / "labels" / "moon" / name, url_map[name])

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(parse_one, row) for row in left_rows]
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            parsed.append(future.result())
            if completed % 500 == 0 or completed == len(left_rows):
                print(
                    f"Parsed Moon labels: {completed}/{len(left_rows)}",
                    flush=True,
                )
    by_sequence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for frame in parsed:
        by_sequence[frame["sequence_id"]].append(frame)
    selected: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    sequence_diagnostics: list[dict[str, Any]] = []
    for sequence_id, candidates in sorted(by_sequence.items()):
        positions = np.asarray([frame["rover_xyz"] for frame in candidates])
        position_span = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
        headings = [frame["rover_yaw_rad"] for frame in candidates]
        heading_std = circular_std_degrees(headings)
        if position_span > 0.05:
            rejected[sequence_id] = f"position_span={position_span:.6f}m"
            continue
        # The 3-D optical projection changes slightly between mast pitch rings.
        # A two-degree bound rejects inconsistent labels while retaining valid
        # panoramas; the median official-sequence circular std is about 0.30°.
        if heading_std > 2.0:
            rejected[sequence_id] = f"heading_circular_std={heading_std:.6f}deg"
            continue

        def forward_score(frame: dict[str, Any]) -> float:
            mast_yaw = float(wrap_radians(math.radians(frame["mast_yaw_deg"])))
            mast_pitch = math.radians(frame["mast_pitch_deg"])
            return math.hypot(mast_yaw, mast_pitch)

        best = min(
            candidates, key=lambda frame: (forward_score(frame), frame["start_time"])
        )
        if math.degrees(forward_score(best)) > 30.0:
            rejected[sequence_id] = (
                f"best_forward_score={math.degrees(forward_score(best)):.6f}deg"
            )
            continue
        chosen = {
            **best,
            "source": "ce4_yutu2_pcam",
            "position": best["rover_xyz"][:2],
            "yaw": circular_mean(headings),
            "sequence_heading_std_deg": heading_std,
            "sequence_position_span_m": position_span,
            "forward_score_deg": math.degrees(forward_score(best)),
        }
        selected.append(chosen)
        sequence_diagnostics.append(
            {
                "sequence_id": sequence_id,
                "label_count": len(candidates),
                "position_span_m": position_span,
                "heading_circular_std_deg": heading_std,
                "selected_name": best["name"],
                "forward_score_deg": chosen["forward_score_deg"],
            }
        )
    selected.sort(key=lambda frame: (frame["start_time"], frame["sequence_id"]))
    diagnostics = {
        "left_labels": len(parsed),
        "sequences": len(by_sequence),
        "selected_sequences": len(selected),
        "rejected_sequences": rejected,
        "sequence_diagnostics": sequence_diagnostics,
        "selection": (
            "one left image per observation sequence minimizing official wrapped "
            "mast yaw and mast pitch magnitude"
        ),
        "heading_policy": (
            "circular mean of wrap(azimuth(center_point_observe_vector in "
            "ROVER_COORDINATE_SYSTEM) - Rotation_Angle.yawing)"
        ),
        "max_heading_circular_std_deg": 2.0,
        "max_forward_score_deg": 30.0,
    }
    write_json(cache / "manifests" / "moon_frame_selection.json", diagnostics)
    return selected, diagnostics


def segment_mars_frames(
    frames: Sequence[dict[str, Any]],
    *,
    max_step_m: float = 5.0,
    max_gap_s: float = 900_000.0,
    max_yaw_step_deg: float = 120.0,
) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for frame in frames:
        continuous = False
        if current:
            previous = current[-1]
            step = float(
                np.linalg.norm(
                    np.asarray(frame["position"]) - np.asarray(previous["position"])
                )
            )
            gap = float(frame["sclk"] - previous["sclk"])
            yaw_step = abs(
                math.degrees(float(wrap_radians(frame["yaw"] - previous["yaw"])))
            )
            continuous = (
                frame["site"] == previous["site"]
                and 0.0 < gap <= max_gap_s
                and step <= max_step_m
                and yaw_step <= max_yaw_step_deg
            )
        if continuous:
            current.append(frame)
        else:
            if current:
                segments.append(current)
            current = [frame]
    if current:
        segments.append(current)
    return segments


def trajectory_record(name: str, frames: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "name": name,
        "source": frames[0]["source"],
        "frames": list(frames),
    }


def scale_moon_evaluation_positions(
    frames: Sequence[dict[str, Any]], target_nonzero_step: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Map sparse lunar metres to the shared evaluation displacement scale.

    CE4 labels remain the authority: every raw rover XY value is retained in
    frame provenance.  The translated and uniformly scaled value is the one
    written to ``traj_data.pkl`` so the single combined dataset can use the
    same waypoint spacing as its Mars trajectories.  No image or pose is
    interpolated.
    """

    if len(frames) < 2:
        raise ValueError("Need at least two lunar frames to estimate a scale")
    if not math.isfinite(target_nonzero_step) or target_nonzero_step <= 0.0:
        raise ValueError(f"Invalid target lunar step: {target_nonzero_step}")
    raw_positions = np.asarray(
        [frame["position"] for frame in frames], dtype=np.float64
    )
    if raw_positions.shape != (len(frames), 2) or not np.isfinite(raw_positions).all():
        raise ValueError("Invalid raw lunar XY positions")
    raw_steps = np.linalg.norm(np.diff(raw_positions, axis=0), axis=1)
    moving_steps = raw_steps[raw_steps > 1e-6]
    if not len(moving_steps):
        raise ValueError("Lunar trajectory has no nonzero displacement")
    raw_median = float(np.median(moving_steps))
    scale = float(target_nonzero_step / raw_median)
    origin = raw_positions[0].copy()
    evaluation_positions = (raw_positions - origin) * scale
    evaluation_steps = np.linalg.norm(np.diff(evaluation_positions, axis=0), axis=1)

    transform = {
        "type": "translate_then_uniform_scale",
        "input_units": "metre",
        "output_units": "mars_equivalent_metre",
        "origin_xy_m": origin.tolist(),
        "scale": scale,
        "forward_formula": "evaluation_xy=(raw_rover_xy_m-origin_xy_m)*scale",
        "inverse_formula": "raw_rover_xy_m=evaluation_xy/scale+origin_xy_m",
        "raw_nonzero_median_step_m": raw_median,
        "target_nonzero_median_step": float(target_nonzero_step),
        "raw_step_quantiles_m": np.quantile(
            raw_steps, [0, 0.05, 0.5, 0.95, 0.99, 1]
        ).tolist(),
        "evaluation_step_quantiles": np.quantile(
            evaluation_steps, [0, 0.05, 0.5, 0.95, 0.99, 1]
        ).tolist(),
        "images_or_poses_interpolated": False,
    }
    scaled_frames: list[dict[str, Any]] = []
    for frame, raw_position, evaluation_position in zip(
        frames, raw_positions, evaluation_positions
    ):
        scaled_frames.append(
            {
                **frame,
                "raw_position_xy_m": raw_position.tolist(),
                "position_transform": transform,
                "position": evaluation_position.tolist(),
            }
        )
    return scaled_frames, transform


def build_plan(
    cache: Path,
    mars_index: Sequence[dict[str, Any]],
    moon_index: Sequence[dict[str, Any]],
    workers: int,
) -> dict[str, Any]:
    manifests = cache / "manifests"
    manifests.mkdir(parents=True, exist_ok=True)
    mars_frames = build_mars_frames(cache, mars_index, workers)
    moon_frames, moon_diagnostics = build_moon_frames(cache, moon_index, workers)
    raw_mars_segments = segment_mars_frames(mars_frames)
    mars_segments = [
        segment
        for segment in raw_mars_segments
        if len(segment) >= MIN_TRAJECTORY_FRAMES
    ]
    mars_steps = []
    for segment in mars_segments:
        position = np.asarray([frame["position"] for frame in segment])
        mars_steps.extend(np.linalg.norm(np.diff(position, axis=0), axis=1).tolist())
    moving_steps = [step for step in mars_steps if step > 1e-6]
    if not moving_steps:
        raise ValueError("Eligible Mars trajectories have no nonzero displacement")
    spacing = float(np.median(moving_steps))
    trajectories = [
        trajectory_record(f"mars_perseverance_navcam_{index:04d}", segment)
        for index, segment in enumerate(mars_segments)
    ]
    if len(moon_frames) >= MIN_TRAJECTORY_FRAMES:
        moon_frames, moon_transform = scale_moon_evaluation_positions(
            moon_frames, spacing
        )
        moon_diagnostics["position_transform"] = moon_transform
        moon_diagnostics["temporal_semantics"] = (
            "sparse official rover observation sequence; timestamps are irregular; "
            "no temporal or visual interpolation"
        )
        trajectories.append(trajectory_record("moon_yutu2_pcam_0000", moon_frames))
    else:
        moon_diagnostics["evaluation_exclusion"] = (
            f"only {len(moon_frames)} valid sequences; need {MIN_TRAJECTORY_FRAMES}"
        )
    write_json(manifests / "moon_frame_selection.json", moon_diagnostics)
    plan = {
        "schema_version": 2,
        "created_utc": dt.datetime.now(dt.UTC).isoformat(),
        "dataset": "planetary_rover",
        "test_only": True,
        "context_size": 4,
        "len_traj_pred": 64,
        "minimum_trajectory_frames": MIN_TRAJECTORY_FRAMES,
        "image_policy": (
            "official image; CE4 UnsignedLSB2 DN validated in [0,1023] and "
            "converted to uint8 by retaining the most-significant 8 bits; center "
            "square crop; resize 224x224; RGB JPEG quality 95"
        ),
        "mars_selection": {
            "products": "left regular EDR, one representative per (site, drive)",
            "activities": ["VCE_", "TRAV"],
            "max_abs_rover_camera_azimuth_deg": 5.0,
            "max_step_m": 5.0,
            "max_sclk_gap_s": 900_000.0,
            "max_yaw_step_deg": 120.0,
            "input_frames": len(mars_frames),
            "raw_segments": len(raw_mars_segments),
            "eligible_segments": len(mars_segments),
            "eligible_frames": sum(map(len, mars_segments)),
        },
        "moon_selection": moon_diagnostics,
        "metric_waypoint_spacing": spacing,
        "spacing_policy": "median nonzero Mars displacement in eligible test trajectories",
        "position_policy": (
            "Mars position is official metric PLACES XY. Moon traj_data position is a "
            "documented reversible scale transform of official metric CE4 rover XY; "
            "raw values remain in provenance."
        ),
        "trajectories": trajectories,
    }
    write_json(manifests / "build_plan.json", plan)
    return plan


def center_crop_224(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((224, 224), Image.Resampling.LANCZOS)


def encode_jpeg(image: Image.Image) -> bytes:
    stream = io.BytesIO()
    image.save(stream, format="JPEG", quality=95, optimize=True, subsampling=0)
    return stream.getvalue()


def process_source_image(
    frame: dict[str, Any], payload: bytes
) -> tuple[bytes, dict[str, Any]]:
    if frame["source"] == "mars2020_perseverance_navcam":
        with Image.open(io.BytesIO(payload)) as image:
            processing = {
                "source_mode": image.mode,
                "source_size": list(image.size),
                "display_conversion": "PIL RGB conversion",
            }
            return encode_jpeg(center_crop_224(image)), processing
    if frame["source"] == "ce4_yutu2_pcam":
        if frame["data_type"] != "UnsignedLSB2":
            raise ValueError(f"Unsupported CE4 type: {frame['data_type']}")
        values = np.frombuffer(payload, dtype="<u2")
        expected = int(frame["lines"]) * int(frame["samples"])
        if values.size != expected:
            raise ValueError(f"CE4 image size mismatch: {values.size} != {expected}")
        minimum = int(values.min())
        maximum = int(values.max())
        if minimum < 0 or maximum > 1023:
            raise ValueError(
                f"CE4 2B DN is outside the validated 10-bit range: "
                f"[{minimum}, {maximum}]"
            )
        # PCAM DN is stored in the official two-byte little-endian container.
        # Retaining the most-significant eight of its ten populated bits is a
        # fixed, scene-independent display mapping; it avoids per-image
        # percentile stretching that would destroy photometric comparability.
        array = (
            (values >> 2)
            .astype(np.uint8)
            .reshape(int(frame["lines"]), int(frame["samples"]))
        )
        processing = {
            "source_data_type": "UnsignedLSB2",
            "source_dn_min": minimum,
            "source_dn_max": maximum,
            "validated_populated_bits": 10,
            "display_conversion": "uint8 = uint16_dn >> 2",
            "scene_dependent_stretch": False,
        }
        return (
            encode_jpeg(center_crop_224(Image.fromarray(array, mode="L"))),
            processing,
        )
    raise ValueError(f"Unknown source: {frame['source']}")


def sanitized_frame_metadata(frame: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in frame.items() if key not in {"position", "yaw"}
    }


def build_trajectory_images(
    trajectory: dict[str, Any], output: Path, workers: int
) -> dict[str, Any]:
    name = trajectory["name"]
    target = output / name
    frames = trajectory["frames"]
    metadata_path = target / "frame_metadata.jsonl"
    pickle_path = target / "traj_data.pkl"
    metadata_parts = target / ".frame_metadata_parts"
    if (
        metadata_path.is_file()
        and pickle_path.is_file()
        and all((target / f"{index}.jpg").is_file() for index in range(len(frames)))
    ):
        if metadata_parts.is_dir():
            shutil.rmtree(metadata_parts)
        return {"name": name, "frames": len(frames), "resumed": True}
    target.mkdir(parents=True, exist_ok=True)
    metadata_parts.mkdir(parents=True, exist_ok=True)

    def build_one(item: tuple[int, dict[str, Any]]) -> tuple[int, dict[str, Any]]:
        index, frame = item
        path = target / f"{index}.jpg"
        part_path = metadata_parts / f"{index}.json"
        if path.is_file() and part_path.is_file():
            metadata = json.loads(part_path.read_text(encoding="utf-8"))
            if metadata.get("frame") == index and metadata.get(
                "processed_jpeg_sha256"
            ) == sha256_file(path):
                with Image.open(path) as check:
                    if check.size == (224, 224) and check.mode == "RGB":
                        return index, metadata
        payload = fetch_bytes(
            frame["image_url"], insecure=frame["source"] == "ce4_yutu2_pcam"
        )
        jpeg, image_processing = process_source_image(frame, payload)
        atomic_write_bytes(path, jpeg)
        with Image.open(path) as check:
            if check.size != (224, 224) or check.mode != "RGB":
                raise ValueError(
                    f"Invalid processed image: {path}: {check.size}/{check.mode}"
                )
        metadata = {
            "frame": index,
            **sanitized_frame_metadata(frame),
            "source_image_sha256": sha256_bytes(payload),
            "processed_jpeg_sha256": sha256_bytes(jpeg),
            "image_processing": image_processing,
            "position": frame["position"],
            "yaw": frame["yaw"],
        }
        write_json(part_path, metadata)
        return index, metadata

    metadata_rows: list[dict[str, Any] | None] = [None] * len(frames)
    failures: list[tuple[int, Exception]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(build_one, item): item[0] for item in enumerate(frames)
        }
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            try:
                index, metadata = future.result()
                metadata_rows[index] = metadata
            except Exception as error:  # noqa: BLE001 - aggregate worker failures
                failures.append((futures[future], error))
            if completed % 100 == 0 or completed == len(frames):
                print(
                    f"{name}: images {completed}/{len(frames)} "
                    f"({len(failures)} failures)",
                    flush=True,
                )
    if failures:
        index, error = failures[0]
        raise RuntimeError(
            f"{name} had {len(failures)} image failures; first frame={index}: {error}"
        ) from error
    positions = np.asarray([frame["position"] for frame in frames], dtype=np.float64)
    yaws = np.asarray([frame["yaw"] for frame in frames], dtype=np.float64)
    if positions.shape != (len(frames), 2) or yaws.shape != (len(frames),):
        raise ValueError(f"Invalid trajectory arrays for {name}")
    pickle_payload = pickle.dumps({"position": positions, "yaw": yaws}, protocol=4)
    atomic_write_bytes(pickle_path, pickle_payload)
    write_jsonl(metadata_path, (row for row in metadata_rows if row is not None))
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    source = trajectory["source"]
    trajectory_metadata = {
        "trajectory": name,
        "source": source,
        "frames": len(frames),
        "anchors_context4_horizon64": max(0, len(frames) - 67),
        "position_units": (
            "metre"
            if source == "mars2020_perseverance_navcam"
            else "mars_equivalent_metre"
        ),
        "path_length": float(np.sum(steps)),
        "step_quantiles": np.quantile(steps, [0, 0.05, 0.5, 0.95, 1]).tolist(),
        "traj_data_sha256": sha256_bytes(pickle_payload),
    }
    if source == "ce4_yutu2_pcam":
        raw_positions = np.asarray(
            [frame["raw_position_xy_m"] for frame in frames], dtype=np.float64
        )
        raw_steps = np.linalg.norm(np.diff(raw_positions, axis=0), axis=1)
        trajectory_metadata.update(
            {
                "raw_physical_path_length_m": float(np.sum(raw_steps)),
                "raw_physical_step_quantiles_m": np.quantile(
                    raw_steps, [0, 0.05, 0.5, 0.95, 1]
                ).tolist(),
                "position_transform": frames[0]["position_transform"],
            }
        )
    write_json(target / "metadata.json", trajectory_metadata)
    shutil.rmtree(metadata_parts)
    return {"name": name, "frames": len(frames), "resumed": False}


def write_split(path: Path, names: Sequence[str]) -> None:
    atomic_write_text(path, "".join(f"{name}\n" for name in names))


def validate_dataset(
    output: Path, split_path: Path | None = None, checksum_workers: int = 16
) -> dict[str, Any]:
    if checksum_workers < 1:
        raise ValueError("checksum_workers must be positive")
    if split_path is not None and split_path.is_file():
        names = [
            line.strip() for line in split_path.read_text().splitlines() if line.strip()
        ]
    else:
        names = sorted(
            path.name for path in output.iterdir() if (path / "traj_data.pkl").is_file()
        )
    if not names:
        raise ValueError("Dataset has no trajectories")
    if len(names) != len(set(names)):
        raise ValueError("Dataset split contains duplicate trajectories")
    sources: defaultdict[str, int] = defaultdict(int)
    source_steps: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    source_raw_physical_steps: defaultdict[str, list[np.ndarray]] = defaultdict(list)
    moon_dn_minima: list[int] = []
    moon_dn_maxima: list[int] = []
    lengths: list[int] = []
    all_steps: list[np.ndarray] = []
    all_yaw_steps: list[np.ndarray] = []
    checksum_jobs: list[tuple[Path, str, str, int]] = []
    for name in names:
        trajectory = output / name
        with (trajectory / "traj_data.pkl").open("rb") as stream:
            data = pickle.load(stream)
        if set(data) != {"position", "yaw"}:
            raise ValueError(f"Unexpected traj_data keys: {name}: {sorted(data)}")
        position = np.asarray(data["position"])
        yaw = np.asarray(data["yaw"])
        if (
            position.ndim != 2
            or position.shape[1] != 2
            or yaw.shape != (len(position),)
        ):
            raise ValueError(
                f"Bad trajectory shapes: {name}: {position.shape}/{yaw.shape}"
            )
        if len(position) < MIN_TRAJECTORY_FRAMES:
            raise ValueError(f"Trajectory too short: {name}: {len(position)}")
        if not np.isfinite(position).all() or not np.isfinite(yaw).all():
            raise ValueError(f"Nonfinite pose: {name}")
        image_names = sorted(
            (path.name for path in trajectory.glob("*.jpg")),
            key=lambda value: int(Path(value).stem),
        )
        if image_names != [f"{index}.jpg" for index in range(len(position))]:
            raise ValueError(f"Non-contiguous image sequence: {name}")
        metadata = read_jsonl(trajectory / "frame_metadata.jsonl")
        if len(metadata) != len(position):
            raise ValueError(f"Frame metadata mismatch: {name}")
        if [row.get("frame") for row in metadata] != list(range(len(position))):
            raise ValueError(f"Non-contiguous frame metadata: {name}")
        metadata_position = np.asarray(
            [row.get("position") for row in metadata], dtype=np.float64
        )
        metadata_yaw = np.asarray(
            [row.get("yaw") for row in metadata], dtype=np.float64
        )
        if metadata_position.shape != position.shape or not np.allclose(
            metadata_position, position, rtol=0.0, atol=1e-9
        ):
            raise ValueError(f"Pose/metadata position mismatch: {name}")
        if metadata_yaw.shape != yaw.shape or not np.allclose(
            metadata_yaw, yaw, rtol=0.0, atol=1e-9
        ):
            raise ValueError(f"Pose/metadata yaw mismatch: {name}")
        provenance_fields = {
            "label_url",
            "label_sha256",
            "image_url",
            "source_image_sha256",
            "processed_jpeg_sha256",
        }
        if any(not provenance_fields.issubset(row) for row in metadata):
            raise ValueError(f"Incomplete frame provenance: {name}")
        source = metadata[0]["source"]
        if any(row["source"] != source for row in metadata):
            raise ValueError(f"Mixed source in trajectory: {name}")
        if source == "ce4_yutu2_pcam":
            for index, row in enumerate(metadata):
                processing = row.get("image_processing")
                if (
                    not isinstance(processing, dict)
                    or processing.get("display_conversion") != "uint8 = uint16_dn >> 2"
                ):
                    raise ValueError(
                        f"Missing fixed lunar display conversion: {name}/{index}"
                    )
                source_minimum = int(processing.get("source_dn_min"))
                source_maximum = int(processing.get("source_dn_max"))
                if not 0 <= source_minimum <= source_maximum <= 1023:
                    raise ValueError(
                        f"Invalid lunar source DN range: {name}/{index}: "
                        f"[{source_minimum}, {source_maximum}]"
                    )
                moon_dn_minima.append(source_minimum)
                moon_dn_maxima.append(source_maximum)
            transform = metadata[0].get("position_transform")
            if not isinstance(transform, dict) or any(
                row.get("position_transform") != transform for row in metadata
            ):
                raise ValueError(f"Missing or inconsistent lunar transform: {name}")
            raw_position = np.asarray(
                [row.get("raw_position_xy_m") for row in metadata], dtype=np.float64
            )
            rover_position = np.asarray(
                [row.get("rover_xyz", [None, None])[:2] for row in metadata],
                dtype=np.float64,
            )
            if raw_position.shape != position.shape or not np.allclose(
                raw_position, rover_position, rtol=0.0, atol=1e-9
            ):
                raise ValueError(f"Lunar raw position provenance mismatch: {name}")
            origin = np.asarray(transform.get("origin_xy_m"), dtype=np.float64)
            scale = float(transform.get("scale"))
            expected_position = (raw_position - origin) * scale
            if (
                origin.shape != (2,)
                or scale <= 0.0
                or not np.allclose(expected_position, position, rtol=0.0, atol=1e-9)
            ):
                raise ValueError(f"Lunar position transform mismatch: {name}")
            source_raw_physical_steps[source].append(
                np.linalg.norm(np.diff(raw_position, axis=0), axis=1)
            )
        sources[source] += len(position)
        lengths.append(len(position))
        trajectory_steps = np.linalg.norm(np.diff(position, axis=0), axis=1)
        all_steps.append(trajectory_steps)
        source_steps[source].append(trajectory_steps)
        all_yaw_steps.append(np.abs(wrap_radians(np.diff(yaw))))
        for index, row in enumerate(metadata):
            image_path = trajectory / f"{index}.jpg"
            checksum_jobs.append(
                (image_path, row["processed_jpeg_sha256"], name, index)
            )
            if index not in {0, len(position) // 2, len(position) - 1}:
                continue
            with Image.open(image_path) as image:
                if image.size != (224, 224) or image.mode != "RGB":
                    raise ValueError(f"Bad image {name}/{index}.jpg")

    def verify_checksum(job: tuple[Path, str, str, int]) -> None:
        image_path, expected, name, index = job
        if sha256_file(image_path) != expected:
            raise ValueError(f"Processed image checksum mismatch: {name}/{index}.jpg")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=checksum_workers
    ) as executor:
        list(executor.map(verify_checksum, checksum_jobs))
    steps = np.concatenate(all_steps)
    yaw_steps = np.concatenate(all_yaw_steps)
    report = {
        "dataset": "planetary_rover",
        "test_only": True,
        "trajectories": len(names),
        "frames": int(sum(lengths)),
        "eligible_frames": int(sum(lengths)),
        "anchors_context4_horizon64": int(sum(length - 67 for length in lengths)),
        "trajectory_length_quantiles": np.quantile(
            lengths, [0, 0.05, 0.5, 0.95, 1]
        ).tolist(),
        "source_frames": dict(sources),
        "translation_step_quantiles_evaluation_units": np.quantile(
            steps, [0, 0.05, 0.5, 0.95, 0.99, 1]
        ).tolist(),
        "source_translation_step_quantiles": {
            source: np.quantile(
                np.concatenate(values), [0, 0.05, 0.5, 0.95, 0.99, 1]
            ).tolist()
            for source, values in sorted(source_steps.items())
        },
        "source_raw_physical_step_quantiles_m": {
            source: np.quantile(
                np.concatenate(values), [0, 0.05, 0.5, 0.95, 0.99, 1]
            ).tolist()
            for source, values in sorted(source_raw_physical_steps.items())
        },
        "moon_source_dn_range": (
            [min(moon_dn_minima), max(moon_dn_maxima)] if moon_dn_minima else None
        ),
        "absolute_yaw_step_quantiles_rad": np.quantile(
            yaw_steps, [0, 0.05, 0.5, 0.95, 0.99, 1]
        ).tolist(),
        "split_sha256": (
            sha256_file(split_path) if split_path and split_path.is_file() else None
        ),
        "validated_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    return report


def build_dataset(
    cache: Path,
    output: Path,
    split_output: Path,
    plan: dict[str, Any],
    workers: int,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for index, trajectory in enumerate(plan["trajectories"], 1):
        print(
            f"Building trajectory {index}/{len(plan['trajectories'])}: {trajectory['name']}",
            flush=True,
        )
        results.append(build_trajectory_images(trajectory, output, workers))
    names = [trajectory["name"] for trajectory in plan["trajectories"]]
    write_split(split_output, names)
    report = validate_dataset(output, split_output, checksum_workers=workers)
    build_plan_path = cache / "manifests" / "build_plan.json"
    source_manifest_path = cache / "sources" / "source_manifest.json"
    report.update(
        {
            "metric_waypoint_spacing": plan["metric_waypoint_spacing"],
            "build_plan_sha256": sha256_file(build_plan_path),
            "source_manifest_sha256": sha256_file(source_manifest_path),
            "builder_sha256": sha256_file(Path(__file__)),
            "position_policy": plan["position_policy"],
            "moon_position_transform": plan["moon_selection"].get("position_transform"),
            "dyaw_policy": "shortest signed angular difference in [-pi, pi)",
            "authoritative_sources": {
                "mars_places_doi": "10.17189/btz6-5a82",
                "mars_navcam_doi": "10.17189/d3nm-pp09",
                "moon_provider": "中国探月工程科学数据发布系统 / CLEP",
                "moon_catalogue_id": 672,
            },
            "build_results": results,
        }
    )
    atomic_write_bytes(output / "build_plan.json", build_plan_path.read_bytes())
    atomic_write_bytes(
        output / "source_manifest.json", source_manifest_path.read_bytes()
    )
    write_json(output / "dataset_report.json", report)
    write_json(output / "_SUCCESS", report)
    return report


def copy_seed_sources(cache: Path, seed_dir: Path) -> None:
    """Import previously downloaded authoritative source indexes by checksum."""

    sources = cache / "sources"
    sources.mkdir(parents=True, exist_ok=True)
    mapping = {
        "m20_best_interp.csv": "best_interp.csv",
        "m20_best_interp.csv.xml": "best_interp.csv.xml",
        "m20_best_tactical.csv": "best_tactical.csv",
        "m20_best_tactical.csv.xml": "best_tactical.csv.xml",
        "m20_telemetry.csv": "telemetry.csv",
        "m20_telemetry.csv.xml": "telemetry.csv.xml",
        "Mars2020_Rover_PLACES_PDS_SIS.pdf": "Mars2020_Rover_PLACES_PDS_SIS.pdf",
        "m20_navcam_rmc_top_hits.jsonl": "mars_navcam_rmc_index.jsonl",
        "ce4_pcam_labels_index.complete.json": "ce4_pcam_2b_label_index.json",
    }
    for source_name, target_name in mapping.items():
        source = seed_dir / source_name
        target = sources / target_name
        if source.is_file() and not target.exists():
            atomic_write_bytes(target, source.read_bytes())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("index", "labels", "plan", "build", "all", "validate")
    )
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--split-output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--seed-source-dir",
        type=Path,
        help="Optional directory containing already downloaded official source files",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.command in {"build", "all", "validate"} and args.output_root is None:
        parser.error(f"{args.command} requires --output-root")
    if args.command in {"build", "all"} and args.split_output is None:
        parser.error(f"{args.command} requires --split-output")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    force_ipv4()
    cache = args.cache_root.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    if args.seed_source_dir:
        copy_seed_sources(cache, args.seed_source_dir.resolve())
    if args.command == "validate":
        report = validate_dataset(
            args.output_root.resolve(), args.split_output, checksum_workers=args.workers
        )
        print(json.dumps(report, indent=2))
        return 0
    mars_index, moon_index = ensure_source_indexes(cache)
    if args.command == "index":
        return 0
    if args.command in {"labels", "all"}:
        fetch_labels(cache, mars_index, moon_index, args.workers)
        if args.command == "labels":
            return 0
    if args.command in {"plan", "all"}:
        plan = build_plan(cache, mars_index, moon_index, args.workers)
        if args.command == "plan":
            print(
                json.dumps(
                    {
                        key: value
                        for key, value in plan.items()
                        if key != "trajectories"
                    },
                    indent=2,
                )
            )
            return 0
    else:
        plan = json.loads((cache / "manifests" / "build_plan.json").read_text())
    report = build_dataset(
        cache,
        args.output_root.resolve(),
        args.split_output.resolve(),
        plan,
        args.workers,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
