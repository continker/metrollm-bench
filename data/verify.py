"""Transit network verification for MetroLLM-Bench.

Checks coordinate coverage, bounding boxes, edge distances vs haversine,
graph connectivity, line completeness, and canonical routes.

Usage:
    uv run python data/verify.py                # Run all checks
    uv run python data/verify.py --export-map   # Also generate dashboard/verify_data.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

BASE = Path(__file__).parent
SYSTEMS_DIR = BASE / "systems"
CASES_DIR = BASE.parent / "cases"
WIKI_DIR = BASE / "gtfs"

SYSTEMS = ["marta", "bart", "cta", "taipei", "doha", "beijing"]

# Bounding boxes: (lat_min, lat_max, lon_min, lon_max)
BOUNDING_BOXES: dict[str, tuple[float, float, float, float]] = {
    "marta": (33.5, 34.0, -84.6, -84.2),
    "bart": (37.3, 38.1, -122.5, -121.7),
    "cta": (41.7, 42.1, -87.95, -87.6),
    "taipei": (24.95, 25.2, 121.4, 121.62),
    "doha": (25.19, 25.5, 51.3, 51.62),
    "beijing": (39.4, 40.3, 115.9, 116.8),
}

SYSTEM_CENTERS: dict[str, tuple[float, float]] = {
    "marta": (33.75, -84.39),
    "bart": (37.78, -122.27),
    "cta": (41.88, -87.67),
    "taipei": (25.04, 121.54),
    "doha": (25.30, 51.50),
    "beijing": (39.90, 116.40),
}

HARDCODED_ROUTES: dict[str, list[tuple[str, str]]] = {
    "marta": [
        ("MARTA-NS", "MARTA-AP"),
        ("MARTA-IC", "MARTA-DO"),
        ("MARTA-BK", "MARTA-NS"),
        ("MARTA-FP", "MARTA-DW"),
        ("MARTA-AP", "MARTA-IC"),
    ],
    "bart": [
        ("BART-RICH", "BART-MLBR"),
        ("BART-DUBL", "BART-DALY"),
        ("BART-ANTC", "BART-BERY"),
        ("BART-SFO", "BART-NBRK"),
        ("BART-EMBR", "BART-FRMT"),
    ],
    "cta": [
        ("CTA-HOW", "CTA-95D"),
        ("CTA-ORD", "CTA-FPK"),
        ("CTA-KIM", "CTA-CLK"),
        ("CTA-MID", "CTA-HLK"),
        ("CTA-LIN", "CTA-DMP"),
    ],
    "taipei": [
        ("TRTC-TAM", "TRTC-XSH"),
        ("TRTC-TPZ", "TRTC-NEC"),
        ("TRTC-LUZ", "TRTC-NSJ"),
        ("TRTC-DIN", "TRTC-NKG"),
        ("TRTC-SSN", "TRTC-XDN"),
    ],
    "doha": [
        ("DOHA-LUS", "DOHA-WKR"),
        ("DOHA-RIF", "DOHA-MAN"),
        ("DOHA-AZZ", "DOHA-RBA"),
        ("DOHA-HIA", "DOHA-EDC"),
        ("DOHA-LUS", "DOHA-AZZ"),
    ],
    "beijing": [
        ("BJM-PIN", "BJM-SIH"),
        ("BJM-XIZ", "BJM-DON2"),
        ("BJM-SON", "BJM-GUO"),
        ("BJM-BWR", "BJM-DAJ3"),
        ("BJM-FUX", "BJM-CAO2"),
    ],
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 3958.8
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def _load_system_data(system: str) -> dict:
    """Load stations, graph, and lines for a system."""
    d = SYSTEMS_DIR / system
    stations_list = json.loads((d / "stations.json").read_text())
    graph_data = json.loads((d / "graph.json").read_text())
    lines_list = json.loads((d / "lines.json").read_text())
    return {
        "stations": {s["id"]: s for s in stations_list},
        "stations_list": stations_list,
        "edges": graph_data["edges"],
        "lines": lines_list,
    }


def _load_wiki_coords(system: str) -> dict[str, dict]:
    """Load coordinate data from wiki JSON. Returns {station_name: {lat, lon}}."""
    wiki_path = WIKI_DIR / f"{system}_wiki.json"
    if not wiki_path.exists():
        return {}
    data = json.loads(wiki_path.read_text())
    coords = data.get("stations", {})
    return {k: v for k, v in coords.items() if isinstance(v, dict) and "lat" in v}


def _get_station_coords(
    system: str, stations: dict[str, dict]
) -> dict[str, tuple[float, float]]:
    """Build station_id -> (lat, lon) mapping from wiki coords + station names."""
    wiki_coords = _load_wiki_coords(system)
    # Build name->id mapping
    name_to_id = {s["name"]: sid for sid, s in stations.items()}
    result: dict[str, tuple[float, float]] = {}
    for name, coord in wiki_coords.items():
        sid = name_to_id.get(name)
        if sid and coord.get("lat") is not None and coord.get("lon") is not None:
            result[sid] = (coord["lat"], coord["lon"])
    return result


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    passed: bool
    summary: str
    issues: list[str] = field(default_factory=list)
    report: str = ""


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def check_coordinates(
    systems_data: dict[str, dict],
    coords_by_system: dict[str, dict[str, tuple[float, float]]],
) -> tuple[list[str], list[str]]:
    """Check 1: Every station must have coordinates."""
    issues: list[str] = []
    lines: list[str] = []
    lines.append("=" * 60)
    lines.append("CHECK 1: Coordinates exist")
    lines.append("=" * 60)

    for system in SYSTEMS:
        data = systems_data[system]
        coords = coords_by_system[system]
        missing = [
            sid for sid in data["stations"] if sid not in coords
        ]
        if missing:
            issues.append(
                f"{system.upper()}: {len(missing)}/{len(data['stations'])} "
                f"stations missing coordinates"
            )
            lines.append(
                f"  FAIL {system.upper()}: {len(missing)} stations missing coords:"
            )
            for sid in missing[:10]:
                name = data["stations"][sid]["name"]
                lines.append(f"    - {sid} ({name})")
            if len(missing) > 10:
                lines.append(f"    ... and {len(missing) - 10} more")
        else:
            lines.append(
                f"  OK   {system.upper()}: all {len(data['stations'])} stations "
                f"have coordinates"
            )

    return issues, lines


def check_bounding_box(
    systems_data: dict[str, dict],
    coords_by_system: dict[str, dict[str, tuple[float, float]]],
) -> tuple[list[str], list[str]]:
    """Check 2: Stations within expected bounding box."""
    issues: list[str] = []
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("CHECK 2: Bounding box")
    lines.append("=" * 60)

    for system in SYSTEMS:
        coords = coords_by_system[system]
        if not coords:
            lines.append(f"  SKIP {system.upper()}: no coordinates available")
            continue

        bbox = BOUNDING_BOXES[system]
        lat_min, lat_max, lon_min, lon_max = bbox
        out_of_bounds = []
        for sid, (lat, lon) in coords.items():
            if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
                name = systems_data[system]["stations"][sid]["name"]
                out_of_bounds.append((sid, name, lat, lon))

        if out_of_bounds:
            issues.append(
                f"{system.upper()}: {len(out_of_bounds)} stations outside bounding box"
            )
            lines.append(
                f"  FAIL {system.upper()}: {len(out_of_bounds)} stations outside "
                f"bbox ({lat_min}-{lat_max}, {lon_min}-{lon_max}):"
            )
            for sid, name, lat, lon in out_of_bounds:
                lines.append(f"    - {sid} ({name}): ({lat:.4f}, {lon:.4f})")
        else:
            lines.append(
                f"  OK   {system.upper()}: all {len(coords)} stations within "
                f"bounding box"
            )

    return issues, lines


def check_edge_distances(
    systems_data: dict[str, dict],
    coords_by_system: dict[str, dict[str, tuple[float, float]]],
) -> tuple[list[str], list[str]]:
    """Check 3: Edge distance vs haversine."""
    issues: list[str] = []
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("CHECK 3: Edge distance vs haversine")
    lines.append("=" * 60)

    for system in SYSTEMS:
        coords = coords_by_system[system]
        edges = systems_data[system]["edges"]
        if not coords:
            lines.append(f"  SKIP {system.upper()}: no coordinates available")
            continue

        discrepancies = []
        checked = 0
        for edge in edges:
            from_id, to_id = edge["from"], edge["to"]
            if from_id not in coords or to_id not in coords:
                continue
            checked += 1
            lat1, lon1 = coords[from_id]
            lat2, lon2 = coords[to_id]
            h_dist = haversine_miles(lat1, lon1, lat2, lon2)
            g_dist = edge["distance_miles"]
            if h_dist < 0.01 and g_dist < 0.01:
                continue  # both essentially zero
            if h_dist > 0:
                pct_diff = abs(g_dist - h_dist) / h_dist * 100
            else:
                pct_diff = 999.0
            if pct_diff > 20:
                from_name = systems_data[system]["stations"].get(from_id, {}).get(
                    "name", from_id
                )
                to_name = systems_data[system]["stations"].get(to_id, {}).get(
                    "name", to_id
                )
                discrepancies.append(
                    (
                        from_id,
                        from_name,
                        to_id,
                        to_name,
                        edge["line"],
                        g_dist,
                        round(h_dist, 2),
                        round(pct_diff, 1),
                    )
                )

        # Only count as issue if graph distance is SHORTER than haversine
        # (indicates wrong coordinates). Longer is expected for thinned edges.
        real_errors = [d for d in discrepancies if d[5] < d[6] * 0.8]
        if real_errors:
            issues.append(
                f"{system.upper()}: {len(real_errors)}/{checked} edges shorter "
                f"than haversine (possible coordinate error)"
            )
        if discrepancies:
            lines.append(
                f"  FLAG {system.upper()}: {len(discrepancies)} edges with >20% "
                f"discrepancy (of {checked} checked):"
            )
            for (
                fid,
                fname,
                tid,
                tname,
                line,
                g_dist,
                h_dist,
                pct,
            ) in discrepancies:
                lines.append(
                    f"    {fid} ({fname}) -> {tid} ({tname}) [{line}]: "
                    f"graph={g_dist}mi, haversine={h_dist}mi, diff={pct}%"
                )
        else:
            lines.append(
                f"  OK   {system.upper()}: all {checked} edges within 20% of haversine"
            )

    return issues, lines


def check_connectivity(
    systems_data: dict[str, dict],
) -> tuple[list[str], list[str]]:
    """Check 4: Graph connectivity."""
    import networkx as nx

    issues: list[str] = []
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("CHECK 4: Graph connectivity")
    lines.append("=" * 60)

    for system in SYSTEMS:
        data = systems_data[system]
        G = nx.Graph()
        for sid in data["stations"]:
            G.add_node(sid)
        for edge in data["edges"]:
            G.add_edge(edge["from"], edge["to"])

        if nx.is_connected(G):
            lines.append(
                f"  OK   {system.upper()}: graph is connected "
                f"({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)"
            )
        else:
            components = list(nx.connected_components(G))
            issues.append(
                f"{system.upper()}: graph has {len(components)} disconnected components"
            )
            lines.append(
                f"  FAIL {system.upper()}: {len(components)} components:"
            )
            for i, comp in enumerate(sorted(components, key=len, reverse=True)):
                sample = sorted(comp)[:5]
                lines.append(
                    f"    Component {i + 1}: {len(comp)} stations "
                    f"(e.g. {', '.join(sample)})"
                )

    return issues, lines


def check_line_completeness(
    systems_data: dict[str, dict],
) -> tuple[list[str], list[str]]:
    """Check 5: Every consecutive station pair in a line has an edge."""
    issues: list[str] = []
    lines_out: list[str] = []
    lines_out.append("")
    lines_out.append("=" * 60)
    lines_out.append("CHECK 5: Line edge completeness")
    lines_out.append("=" * 60)

    for system in SYSTEMS:
        data = systems_data[system]
        # Build edge set
        edge_set: set[tuple[str, str, str]] = set()
        for edge in data["edges"]:
            edge_set.add((edge["from"], edge["to"], edge["line"]))
            edge_set.add((edge["to"], edge["from"], edge["line"]))

        missing_edges = []
        for line in data["lines"]:
            line_id = line["id"]
            stations = line["stations"]
            for i in range(len(stations) - 1):
                a, b = stations[i], stations[i + 1]
                if (a, b, line_id) not in edge_set:
                    missing_edges.append((line_id, a, b))

        if missing_edges:
            issues.append(
                f"{system.upper()}: {len(missing_edges)} missing line edges"
            )
            lines_out.append(
                f"  FAIL {system.upper()}: {len(missing_edges)} missing edges:"
            )
            for line_id, a, b in missing_edges:
                lines_out.append(f"    {line_id}: {a} -> {b}")
        else:
            total_line_edges = sum(
                max(0, len(l["stations"]) - 1) for l in data["lines"]
            )
            lines_out.append(
                f"  OK   {system.upper()}: all {total_line_edges} consecutive "
                f"pairs have edges"
            )

    return issues, lines_out


def check_canonical_routes(
    systems_data: dict[str, dict],
) -> tuple[list[str], list[str]]:
    """Check 6: Canonical route sanity checks."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from harness.graph import MetroGraph

    issues: list[str] = []
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 60)
    lines.append("CHECK 6: Canonical routes")
    lines.append("=" * 60)

    for system in SYSTEMS:
        lines.append(f"\n  --- {system.upper()} ---")
        system_dir = SYSTEMS_DIR / system
        graph = MetroGraph(system_dir)

        # Get 5 Cat A case routes
        case_routes: list[tuple[str, str]] = []
        cases_path = CASES_DIR / f"{system}_cases.json"
        if cases_path.exists():
            cases = json.loads(cases_path.read_text())
            cat_a = [c for c in cases if c["category"] == "A"]
            for case in cat_a[:5]:
                origin = None
                dest = None
                for event in case["events"]:
                    if event.get("field") == "origin" and "station_id" in event:
                        origin = event["station_id"]
                    elif event.get("field") == "destination" and "station_id" in event:
                        dest = event["station_id"]
                if origin and dest:
                    case_routes.append((origin, dest))

        # Hardcoded routes
        hardcoded = HARDCODED_ROUTES.get(system, [])

        all_routes = [("case", o, d) for o, d in case_routes] + [
            ("hardcoded", o, d) for o, d in hardcoded
        ]

        for source, origin, dest in all_routes:
            try:
                result = graph.shortest_path(origin, dest)
                flag = " *** HIGH TRANSFERS" if result.transfers > 3 else ""
                if result.transfers > 3:
                    issues.append(
                        f"{system.upper()}: {origin}->{dest} has "
                        f"{result.transfers} transfers"
                    )
                lines.append(
                    f"  [{source:9s}] {origin} -> {dest}: "
                    f"{result.transfers} transfers, "
                    f"{result.distance_miles:.1f}mi, "
                    f"{result.estimated_minutes:.0f}min, "
                    f"lines={result.line_sequence}{flag}"
                )
            except Exception as e:
                issues.append(
                    f"{system.upper()}: route {origin}->{dest} failed: {e}"
                )
                lines.append(
                    f"  [{source:9s}] {origin} -> {dest}: ERROR - {e}"
                )

    return issues, lines


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------


def run_verification() -> VerificationResult:
    """Run all verification checks. Returns VerificationResult."""
    # Load all system data
    systems_data: dict[str, dict] = {}
    coords_by_system: dict[str, dict[str, tuple[float, float]]] = {}

    for system in SYSTEMS:
        systems_data[system] = _load_system_data(system)
        coords_by_system[system] = _get_station_coords(
            system, systems_data[system]["stations"]
        )

    all_issues: list[str] = []
    all_lines: list[str] = []
    all_lines.append("MetroLLM-Bench Network Verification Report")
    all_lines.append("=" * 60)

    # Run checks
    for check_fn in [
        lambda: check_coordinates(systems_data, coords_by_system),
        lambda: check_bounding_box(systems_data, coords_by_system),
        lambda: check_edge_distances(systems_data, coords_by_system),
        lambda: check_connectivity(systems_data),
        lambda: check_line_completeness(systems_data),
        lambda: check_canonical_routes(systems_data),
    ]:
        issues, lines = check_fn()
        all_issues.extend(issues)
        all_lines.extend(lines)

    # Summary
    all_lines.append("")
    all_lines.append("=" * 60)
    if all_issues:
        all_lines.append(f"FAIL: {len(all_issues)} issue(s) found")
        for issue in all_issues:
            all_lines.append(f"  - {issue}")
    else:
        all_lines.append("PASS: all checks passed")
    all_lines.append("=" * 60)

    report = "\n".join(all_lines)
    summary = (
        f"{len(all_issues)} issue(s): " + "; ".join(all_issues[:5])
        if all_issues
        else "all checks passed"
    )

    return VerificationResult(
        passed=len(all_issues) == 0,
        summary=summary,
        issues=all_issues,
        report=report,
    )


# ---------------------------------------------------------------------------
# Map data export
# ---------------------------------------------------------------------------


def export_map_data() -> Path:
    """Generate dashboard/verify_data.json with all system data for Leaflet map."""
    output: dict[str, dict] = {}

    for system in SYSTEMS:
        data = _load_system_data(system)
        coords = _get_station_coords(system, data["stations"])

        # Build station list with coords
        stations_out = []
        for sid, sdata in data["stations"].items():
            coord = coords.get(sid)
            stations_out.append(
                {
                    "id": sid,
                    "name": sdata["name"],
                    "lines": sdata.get("lines", []),
                    "lat": coord[0] if coord else None,
                    "lon": coord[1] if coord else None,
                }
            )

        # Build edges with haversine
        edges_out = []
        for edge in data["edges"]:
            from_id, to_id = edge["from"], edge["to"]
            h_dist = None
            if from_id in coords and to_id in coords:
                lat1, lon1 = coords[from_id]
                lat2, lon2 = coords[to_id]
                h_dist = round(haversine_miles(lat1, lon1, lat2, lon2), 2)
            edges_out.append(
                {
                    "from": from_id,
                    "to": to_id,
                    "line": edge["line"],
                    "distance_miles": edge["distance_miles"],
                    "haversine_miles": h_dist,
                }
            )

        # Lines with colors
        lines_out = [
            {"id": l["id"], "name": l["name"], "color": l["color"]}
            for l in data["lines"]
        ]

        # Load all wiki stations (including thinned-out) for full topology view
        wiki_path = BASE / "gtfs" / f"{system}_wiki.json"
        all_wiki_stations = []
        if wiki_path.exists():
            wiki = json.loads(wiki_path.read_text())
            wiki_coords = wiki.get("stations", {})
            # Collect all station names from all lines + branches
            wiki_names: set[str] = set()
            for ld in wiki.get("lines", []):
                wiki_names.update(ld.get("stations", []))
                for branch in ld.get("branches", []):
                    wiki_names.update(branch.get("stations", []))
            # Build list of wiki stations not in ingested data
            ingested_names = {s["name"] for s in stations_out}
            for name in sorted(wiki_names):
                wc = wiki_coords.get(name, {})
                all_wiki_stations.append({
                    "name": name,
                    "lat": wc.get("lat"),
                    "lon": wc.get("lon"),
                    "thinned": name not in ingested_names,
                })

        # Build wiki line sequences (full, unthinned) for drawing connections
        wiki_lines_full = []
        if wiki_path.exists():
            wiki_coords_map = wiki.get("stations", {})
            for ld in wiki.get("lines", []):
                seq = []
                for name in ld.get("stations", []):
                    wc = wiki_coords_map.get(name, {})
                    if wc.get("lat") and wc.get("lon"):
                        seq.append({"name": name, "lat": wc["lat"], "lon": wc["lon"]})
                # Close loop visually by appending first station
                if ld.get("loop_closure") and seq:
                    seq.append({"name": seq[0]["name"], "lat": seq[0]["lat"], "lon": seq[0]["lon"]})

                wiki_lines_full.append({
                    "id": ld["id"],
                    "color": ld.get("color", "#999"),
                    "stations": seq,
                })
                # Add branch sequences
                for branch in ld.get("branches", []):
                    bseq = []
                    for name in branch.get("stations", []):
                        wc = wiki_coords_map.get(name, {})
                        if wc.get("lat") and wc.get("lon"):
                            bseq.append({"name": name, "lat": wc["lat"], "lon": wc["lon"]})
                    if bseq:
                        wiki_lines_full.append({
                            "id": ld["id"],
                            "color": ld.get("color", "#999"),
                            "stations": bseq,
                            "branch": True,
                        })

        output[system] = {
            "stations": stations_out,
            "edges": edges_out,
            "lines": lines_out,
            "center": list(SYSTEM_CENTERS[system]),
            "wiki_stations": all_wiki_stations,
            "wiki_lines": wiki_lines_full,
        }

    out_path = BASE.parent / "dashboard" / "verify_data.json"
    out_path.write_text(json.dumps(output, indent=2) + "\n")
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Verify MetroLLM-Bench transit network data"
    )
    parser.add_argument(
        "--export-map",
        action="store_true",
        help="Also generate dashboard/verify_data.json for Leaflet map",
    )
    args = parser.parse_args()

    result = run_verification()
    print(result.report)

    if args.export_map:
        path = export_map_data()
        print(f"\nMap data exported to {path}")

    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
