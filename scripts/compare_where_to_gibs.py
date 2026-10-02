# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "matplotlib",
#     "numpy",
# ]
# ///
"""Compare the dask and dask-less where= branches against NASA GIBS.

Adapted from tempo-virtual-zarr-pipeline's scripts/compare_to_gibs.py, which
reads the store directly. This one reads the same TEMPO scan through two
running titiler-multidim servers, one on feat/where-parameter (dask) and one
on feat/where-mask (no dask), applying GIBS's quality screen with where=
instead of numpy. Both renders go through GIBS's colormap and are checked
against the GIBS image of that scan on the same pixel grid, and against each
other: the two branches should agree exactly, and each should match GIBS to
the ~1% compare_to_gibs.py reports for the store itself.

The default store is the public Source Coop mirror, read through its S3
bucket. Its chunks are virtual references into ASDC's bucket, so each server
needs Earthdata access for s3://asdc-prod-protected/ (see the README) and
must run in us-west-2. Point the script at a deployment of each branch, or
run each branch from a worktree on an in-region host:

    git worktree add ../where-dask feat/where-parameter
    git worktree add ../where-nodask feat/where-mask
    (cd ../where-dask && uv run uvicorn titiler.multidim.main:app --port 8001)
    (cd ../where-nodask && uv run uvicorn titiler.multidim.main:app --port 8002)

    uv run scripts/compare_where_to_gibs.py --collection no2 \\
        --dask http://localhost:8001 --nodask http://localhost:8002
    uv run scripts/compare_where_to_gibs.py --collection hcho \\
        --time 2026-10-01T18:00 --bbox -125 24 -66 50 \\
        --dask http://localhost:8001 --nodask http://localhost:8002
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time as clock
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any

import numpy as np

GIBS = "https://gibs.earthdata.nasa.gov"
STORE = "s3://us-west-2.opendata.source.coop/pangeo/tempo-virtual-icechunk/tempo/{collection}/v04"
# collection -> (store variable, GIBS layer, GIBS colormap)
LAYERS = {
    "no2": (
        "vertical_column_troposphere",
        "TEMPO_L3_NO2_Vertical_Column_Troposphere",
        "TEMPO_NO2_Vertical_Column_Troposphere",
    ),
    "hcho": (
        "vertical_column",
        "TEMPO_L3_Formaldehyde_Vertical_Column",
        "TEMPO_HCHO_Vertical_Column",
    ),
}
# compare_to_gibs.py's screen(), as where= conditions
# ponytail: thresholds fitted to GIBS output there; keep the two in step.
WHERE = ["main_data_quality_flag<=1", "eff_cloud_fraction<0.5", "solar_zenith_angle<80"]
# The store's time axis is GPS seconds, which xarray decodes as if they were
# UTC; GIBS is keyed by UTC scan start.
# ponytail: leap seconds fixed at 18 (true since 2017); add one if IERS ever does.
GPS_MINUS_UTC = np.timedelta64(18, "s")
NO_DATA, OFF_PALETTE = -1, -2
MAX_PIXELS = 500_000  # per /bbox request; see titiler_values
BRANCHES = ("nodask", "dask")


def fetch(url: str) -> bytes:
    """GET ``url``."""
    with urllib.request.urlopen(url, timeout=900) as response:
        return bytes(response.read())


def get(api: str, path: str, **params: Any) -> bytes:
    """GET ``path`` on ``api`` with query ``params``."""
    return fetch(
        f"{api.rstrip('/')}{path}?{urllib.parse.urlencode(params, doseq=True)}"
    )


def parse_colormap(xml: bytes) -> tuple[np.ndarray, np.ndarray]:
    """Upper bin edges and their RGB colors from a GIBS v1.3 colormap."""
    edges, colors = [], []
    for entry in ET.fromstring(xml).iter("ColorMapEntry"):
        if entry.get("nodata") == "true":
            continue
        edges.append(float(entry.get("value", "").strip("[]()").split(",")[-1]))
        colors.append([int(c) for c in entry.get("rgb", "").split(",")])
    return np.array(edges), np.array(colors, dtype=np.uint8)


def value_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Colormap bin of each value, NO_DATA where it is NaN."""
    bins = np.searchsorted(edges, values, side="right").clip(0, len(edges) - 1)
    return np.where(np.isnan(values), NO_DATA, bins)


def color_bins(rgba: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Colormap bin of each GIBS pixel: NO_DATA where transparent,
    OFF_PALETTE where its color is not in the colormap."""
    lookup = {tuple(c): i for i, c in enumerate(colors.tolist())}
    rgb = rgba[..., :3].reshape(-1, 3)
    bins = np.array([lookup.get(tuple(c), OFF_PALETTE) for c in rgb.tolist()])
    return np.where(rgba[..., 3].ravel() == 0, NO_DATA, bins).reshape(rgba.shape[:2])


def gibs_rgba(
    layer: str, utc: np.datetime64, bbox: list[float], shape: tuple[int, int]
) -> np.ndarray:
    """GIBS's image of ``layer`` at ``utc``, south row first."""
    import matplotlib.image

    query = urllib.parse.urlencode(
        {
            "SERVICE": "WMS",
            "VERSION": "1.1.1",  # lon,lat axis order for EPSG:4326
            "REQUEST": "GetMap",
            "LAYERS": layer,
            "STYLES": "",
            "SRS": "EPSG:4326",
            "BBOX": ",".join(f"{v:.6f}" for v in bbox),
            "WIDTH": shape[1],
            "HEIGHT": shape[0],
            "FORMAT": "image/png",
            "TRANSPARENT": "TRUE",
            "TIME": f"{np.datetime_as_string(utc, unit='s')}Z",
        }
    )
    png = fetch(f"{GIBS}/wms/epsg4326/best/wms.cgi?{query}")
    image = matplotlib.image.imread(io.BytesIO(png), format="png")
    return np.flipud((image * 255).round().astype(np.uint8))


def titiler_values(
    api: str, params: dict[str, Any], bbox: list[float], shape: tuple[int, int]
) -> tuple[np.ndarray, float]:
    """The screened scan as titiler renders it, NaN where masked, south row
    first, and the seconds the requests took.

    Read in strips of rows, north first: a Lambda response tops out at
    6 MB and titiler's npy costs 8 bytes a pixel (float32 value and mask).
    """
    w, s, e, n = bbox
    res = (n - s) / shape[0]
    rows = max(1, MAX_PIXELS // shape[1])
    strips, seconds = [], 0.0
    # ponytail: strips fetched one after another so the time is comparable;
    # thread them if a full-domain run is too slow
    for top in range(0, shape[0], rows):
        height = min(rows, shape[0] - top)
        corners = f"{w:.6f},{n - (top + height) * res:.6f},{e:.6f},{n - top * res:.6f}"
        start = clock.perf_counter()
        raw = get(api, f"/bbox/{corners}/{shape[1]}x{height}.npy", **params)
        seconds += clock.perf_counter() - start
        strips.append(np.load(io.BytesIO(raw)))
    # titiler's npy is the bands followed by the mask (0 = masked)
    data = np.concatenate(strips, axis=1)
    values = np.where(data[-1] == 0, np.nan, data[0].astype("float64"))
    return np.flipud(values), seconds


def agreement(label: str, a: np.ndarray, b: np.ndarray, names: tuple[str, str]) -> int:
    """Print how bins ``a`` and ``b`` agree; return the pixels both draw."""
    both = (a >= 0) & (b >= 0)
    diff = np.abs(a - b)[both]
    counts = {
        "both": int(both.sum()),
        f"{names[0]} only": int(((a >= 0) & (b == NO_DATA)).sum()),
        f"{names[1]} only": int(((a == NO_DATA) & (b != NO_DATA)).sum()),
    }
    if (b == OFF_PALETTE).any():
        counts[f"{names[1]} off-palette"] = int((b == OFF_PALETTE).sum())
    print(f"  {label}: " + ", ".join(f"{k} {v:,}" for k, v in counts.items()))
    if diff.size:
        print(
            f"    same bin {np.mean(diff == 0):.1%}, within 1 bin {np.mean(diff <= 1):.1%}"
        )
    return counts["both"]


def plot(
    path: str,
    title: str,
    gibs: np.ndarray,
    gibs_bin: np.ndarray,
    bins: dict[str, np.ndarray],
    colors: np.ndarray,
    extent: list[float],
) -> None:
    """Write GIBS, both branches, and their differences to ``path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    show: dict[str, Any] = {
        "origin": "lower",
        "extent": extent,
        "interpolation": "nearest",
    }

    def difference(ax: Any, a: np.ndarray, b: np.ndarray, label: str) -> Any:
        both = (a >= 0) & (b >= 0)
        coverage = np.zeros(a.shape + (4,))
        coverage[(a >= 0) & (b == NO_DATA)] = (0.95, 0.55, 0.1, 1)
        coverage[(a == NO_DATA) & (b != NO_DATA)] = (0, 0, 0, 1)
        image = ax.imshow(
            np.ma.masked_where(~both, a - b), cmap="RdBu_r", vmin=-5, vmax=5, **show
        )
        ax.imshow(coverage, **show)
        ax.set_title(label)
        return image

    fig, axes = plt.subplots(3, 2, figsize=(24, 17), constrained_layout=True)
    axes[0, 0].imshow(gibs, **show)
    axes[0, 0].set_title("GIBS (what Worldview shows)")
    difference(
        axes[0, 1],
        bins["dask"],
        bins["nodask"],
        "dask bin − dask-less bin   (orange: dask only, black: dask-less only)",
    )
    palette = ListedColormap(colors / 255)
    for row, branch in enumerate(BRANCHES, start=1):
        axes[row, 0].imshow(
            np.ma.masked_less(bins[branch], 0),
            cmap=palette,
            vmin=0,
            vmax=len(colors) - 1,
            **show,
        )
        axes[row, 0].set_title(f"titiler {branch}, where= screen, GIBS colormap")
        image = difference(
            axes[row, 1],
            bins[branch],
            gibs_bin,
            f"{branch} bin − GIBS bin   (orange: {branch} only, black: GIBS only)",
        )
    fig.colorbar(
        image, ax=axes[:, 1], location="bottom", shrink=0.4, label="colormap bins"
    )
    for ax in axes.flat:
        ax.set_facecolor("#dddddd")
        ax.set_aspect(1 / np.cos(np.deg2rad((extent[2] + extent[3]) / 2)))
    fig.suptitle(title)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    """Compare both branches with GIBS for one scan."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dask", required=True, help="API on feat/where-parameter")
    parser.add_argument("--nodask", required=True, help="API on feat/where-mask")
    parser.add_argument("--collection", choices=sorted(LAYERS), default="no2")
    parser.add_argument("--url", help=f"store (default: {STORE})")
    parser.add_argument(
        "--time",
        help="UTC; the scan in progress then, as Worldview picks it (default: latest)",
    )
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("W", "S", "E", "N"))
    parser.add_argument(
        "--out", help="PNG path (default: where-<collection>-<scan>.png)"
    )
    args = parser.parse_args()

    variable, layer, colormap = LAYERS[args.collection]
    url = args.url or STORE.format(collection=args.collection)
    info = json.loads(
        get(args.nodask, "/info", url=url, variable=variable, show_times="true")
    )

    # Worldview shows the scan whose start precedes the time, not the nearest;
    # compare whole seconds, as GIBS keys scans (the axis has fractions).
    times = info["times"]
    starts = (np.array(times, dtype="datetime64[ns]") - GPS_MINUS_UTC).astype(
        "datetime64[s]"
    )
    index = len(times)
    if args.time:
        index = int(
            np.searchsorted(starts, np.datetime64(args.time, "s"), side="right")
        )
        if index == 0:
            parser.error(f"--time {args.time} is before the first scan")
    utc = starts[index - 1]

    # snap the box outward onto the store's pixel grid; the tolerance absorbs
    # float noise in the reported bounds
    w0, s0, e0, n0 = info["bounds"]
    res = (e0 - w0) / info["width"]
    w, s, e, n = args.bbox or (w0, s0, e0, n0)
    w = w0 + np.floor((max(w, w0) - w0) / res + 1e-3) * res
    s = s0 + np.floor((max(s, s0) - s0) / res + 1e-3) * res
    e = w0 + np.ceil((min(e, e0) - w0) / res - 1e-3) * res
    n = s0 + np.ceil((min(n, n0) - s0) / res - 1e-3) * res
    bbox = [float(w), float(s), float(e), float(n)]
    shape = (round((n - s) / res), round((e - w) / res))

    edges, colors = parse_colormap(fetch(f"{GIBS}/colormaps/v1.3/{colormap}.xml"))
    gibs = gibs_rgba(layer, utc, bbox, shape)
    gibs_bin = color_bins(gibs, colors)
    params = {
        "url": url,
        "variable": variable,
        "sel": f"time={times[index - 1]}",
        "where": WHERE,
    }
    print(f"{args.collection} scan {utc}Z  {layer}  {shape[1]}x{shape[0]} px")
    bins, common = {}, 0
    for branch in BRANCHES:
        values, seconds = titiler_values(getattr(args, branch), params, bbox, shape)
        bins[branch] = value_bins(values, edges)
        print(f"  {branch}: {seconds:.1f}s")
        common += agreement(
            f"{branch} vs GIBS", bins[branch], gibs_bin, (branch, "GIBS")
        )
    agreement("dask vs dask-less", bins["dask"], bins["nodask"], ("dask", "dask-less"))
    identical = np.array_equal(bins["dask"], bins["nodask"])
    print(f"  dask and dask-less bins {'identical' if identical else 'DIFFER'}")

    stamp = np.datetime_as_string(utc, unit="s").replace(":", "")
    out = args.out or f"where-{args.collection}-{stamp}.png"
    title = f"{args.collection.upper()} {variable}  scan {utc}Z  where " + " & ".join(
        WHERE
    )
    plot(out, title, gibs, gibs_bin, bins, colors, [bbox[0], bbox[2], bbox[1], bbox[3]])
    print(f"  wrote {out}")
    if not common:
        print("  no pixels in common with GIBS for this scan or box", file=sys.stderr)
        return 1
    return 0 if identical else 1


if __name__ == "__main__":
    sys.exit(main())
