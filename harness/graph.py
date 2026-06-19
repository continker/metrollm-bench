"""Station graph operations using NetworkX — expanded line-graph routing."""

import json
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field

import networkx as nx

TRANSFER_PENALTY_MIN = 5.0


@dataclass
class RouteResult:
    path: list[str]           # station IDs in order
    stations: list[dict]      # full station info per stop (name, line, is_transfer, etc.)
    distance_miles: float
    estimated_minutes: float
    transfers: int
    line_sequence: list[str]  # e.g. ["red", "blue"] if transferring


class MetroGraph:
    def __init__(self, system_dir: Path):
        """Load graph.json, stations.json, lines.json from a system directory."""
        self.system_dir = system_dir

        with open(system_dir / "stations.json") as f:
            stations_list = json.load(f)
        self.stations: dict[str, dict] = {s["id"]: s for s in stations_list}

        with open(system_dir / "lines.json") as f:
            self.lines: dict[str, dict] = {l["id"]: l for l in json.load(f)}

        with open(system_dir / "graph.json") as f:
            graph_data = json.load(f)

        self._edges_raw: list[dict] = graph_data["edges"]

        # station_id -> set of line_ids serving that station (derived from edges)
        self.station_lines: dict[str, set[str]] = defaultdict(set)
        for edge in self._edges_raw:
            self.station_lines[edge["from"]].add(edge["line"])
            self.station_lines[edge["to"]].add(edge["line"])

        # Simple graph for connectivity checks (is_valid_path, adjacent_stations)
        self.G: nx.Graph = nx.Graph()
        for sid, sdata in self.stations.items():
            self.G.add_node(sid, **sdata)
        for edge in self._edges_raw:
            self.G.add_edge(
                edge["from"],
                edge["to"],
                distance_miles=edge["distance_miles"],
                travel_time_min=edge["travel_time_min"],
                line=edge["line"],
                type=edge["type"],
            )

        # Expanded directed graph for routing
        self._expanded = self._build_expanded(self._edges_raw, set(self.stations))

    def _build_expanded(
        self,
        edges: list[dict],
        station_ids: set[str],
    ) -> nx.DiGraph:
        """Build the expanded line graph for transfer-aware Dijkstra.

        Nodes:
          ("enter", station_id)  — virtual entry point
          (station_id, line_id)  — station on a specific line
          ("exit", station_id)   — virtual exit point

        Edges:
          entry:    ("enter", s) → (s, line)        weight=0, distance=0
          exit:     (s, line)    → ("exit", s)       weight=0, distance=0
          travel:   (sA, line)   → (sB, line)        weight=travel_time, distance=d
          transfer: (s, lineA)   → (s, lineB)        weight=TRANSFER_PENALTY_MIN, distance=0
        """
        G = nx.DiGraph()

        # Collect which lines serve each station
        station_lines: dict[str, set[str]] = defaultdict(set)

        for edge in edges:
            s_from, s_to = edge["from"], edge["to"]
            line = edge["line"]
            dist = edge["distance_miles"]
            time = edge["travel_time_min"]

            station_lines[s_from].add(line)
            station_lines[s_to].add(line)

            # Travel edges (both directions since graph is undirected)
            G.add_edge(
                (s_from, line), (s_to, line),
                weight=time, distance_miles=dist, line=line,
                edge_type="travel",
            )
            G.add_edge(
                (s_to, line), (s_from, line),
                weight=time, distance_miles=dist, line=line,
                edge_type="travel",
            )

        # Entry, exit, and transfer edges
        for sid in station_ids:
            lines = station_lines.get(sid, set())
            for line in lines:
                # Entry
                G.add_edge(
                    ("enter", sid), (sid, line),
                    weight=0, distance_miles=0, edge_type="entry",
                )
                # Exit
                G.add_edge(
                    (sid, line), ("exit", sid),
                    weight=0, distance_miles=0, edge_type="exit",
                )

            # Transfer edges between all line pairs at this station
            lines_list = sorted(lines)
            for i, lineA in enumerate(lines_list):
                for lineB in lines_list[i + 1:]:
                    G.add_edge(
                        (sid, lineA), (sid, lineB),
                        weight=TRANSFER_PENALTY_MIN, distance_miles=0,
                        edge_type="transfer",
                    )
                    G.add_edge(
                        (sid, lineB), (sid, lineA),
                        weight=TRANSFER_PENALTY_MIN, distance_miles=0,
                        edge_type="transfer",
                    )

        return G

    def lines_for_station(self, station_id: str) -> set[str]:
        """Return the set of line ids that serve a station."""
        sid = self._resolve_station(station_id)
        return set(self.station_lines.get(sid, set()))

    def _line_subgraph(self, line_id: str) -> nx.Graph:
        if line_id not in self.lines:
            raise ValueError(f"Unknown line: {line_id}")
        sub = nx.Graph()
        for edge in self._edges_raw:
            if edge["line"] == line_id:
                sub.add_edge(edge["from"], edge["to"])
        return sub

    def is_loop_line(self, line_id: str) -> bool:
        """True if the line has no terminals (every station has degree >= 2 on its own line)."""
        sub = self._line_subgraph(line_id)
        if sub.number_of_nodes() == 0:
            return False
        return all(deg >= 2 for _, deg in sub.degree())

    def line_terminals(self, line_id: str) -> list[str]:
        """Stations with degree 1 on the line subgraph. Empty list for loop lines."""
        sub = self._line_subgraph(line_id)
        return [n for n, deg in sub.degree() if deg == 1]

    def expand_line_closures(
        self,
        closures: list[dict],
    ) -> list[tuple[str, str]]:
        """Expand line-level closures into segment_closures.

        Each closure dict: {"line": str, "from_station"?: str, "to_station"?: str}.
        Omitting both endpoints closes the entire line. Partial closure requires
        both endpoints and raises ValueError on a loop line (ambiguous).
        """
        segments: list[tuple[str, str]] = []
        for c in closures:
            line_id = c.get("line")
            if not line_id or line_id not in self.lines:
                raise ValueError(f"Unknown line: {line_id}")
            from_s = c.get("from_station")
            to_s = c.get("to_station")
            ordered = list(self.lines[line_id].get("stations", []))
            if not ordered:
                raise ValueError(f"Line '{line_id}' has no stations defined")
            if from_s is None and to_s is None:
                keep = set(ordered)
            elif from_s is None or to_s is None:
                raise ValueError(
                    f"Partial closure on line '{line_id}' requires both from_station and to_station"
                )
            else:
                if self.is_loop_line(line_id):
                    raise ValueError(
                        f"Partial closure on loop line '{line_id}' is ambiguous — use whole-line closure or specify segments"
                    )
                a = self._resolve_station(from_s)
                b = self._resolve_station(to_s)
                if a not in ordered or b not in ordered:
                    raise ValueError(
                        f"Endpoints '{from_s}'/'{to_s}' are not on line '{line_id}'"
                    )
                i, j = ordered.index(a), ordered.index(b)
                lo, hi = min(i, j), max(i, j)
                keep = set(ordered[lo:hi + 1])
            for edge in self._edges_raw:
                if (
                    edge["line"] == line_id
                    and edge["from"] in keep
                    and edge["to"] in keep
                ):
                    segments.append((edge["from"], edge["to"]))
        return segments

    def shortest_path(self, origin: str, destination: str) -> RouteResult:
        """Find shortest path by time (with transfer penalty). Returns RouteResult.

        Raises ValueError if either station cannot be resolved.
        Raises nx.NetworkXNoPath if no path exists between the two stations.
        Raises nx.NodeNotFound if a resolved ID is not present in the graph.
        """
        origin_id = self._resolve_station(origin)
        dest_id = self._resolve_station(destination)

        if origin_id == dest_id:
            station = self.stations[origin_id]
            stop = {
                "station_id": origin_id,
                "station_name": station["name"],
                "line": None,
                "is_transfer": False,
                "transfer_to": None,
            }
            return RouteResult(
                path=[origin_id],
                stations=[stop],
                distance_miles=0.0,
                estimated_minutes=0.0,
                transfers=0,
                line_sequence=[],
            )

        return self._route_on_expanded(origin_id, dest_id, self._expanded)

    def shortest_path_avoiding(
        self,
        origin: str,
        destination: str,
        blocked_edges: list[tuple[str, str]] | None = None,
        blocked_stations: list[str] | None = None,
    ) -> RouteResult:
        """Compute shortest path avoiding specified edges and stations.

        Used by case generator for computing post-disruption alternative routes.
        Rebuilds the expanded graph with disrupted edges/stations removed.

        Raises ValueError if origin or destination is blocked or cannot be resolved.
        Raises nx.NetworkXNoPath if no alternative path exists.
        """
        origin_id = self._resolve_station(origin)
        dest_id = self._resolve_station(destination)

        blocked_station_set = set(blocked_stations) if blocked_stations else set()
        blocked_edge_set = set()
        if blocked_edges:
            for u, v in blocked_edges:
                blocked_edge_set.add((u, v))
                blocked_edge_set.add((v, u))

        if origin_id in blocked_station_set:
            raise ValueError(
                f"Origin station '{origin}' is blocked by disruption"
            )
        if dest_id in blocked_station_set:
            raise ValueError(
                f"Destination station '{destination}' is blocked by disruption"
            )

        # Filter edges and stations
        remaining_stations = set(self.stations) - blocked_station_set
        remaining_edges = [
            e for e in self._edges_raw
            if e["from"] not in blocked_station_set
            and e["to"] not in blocked_station_set
            and (e["from"], e["to"]) not in blocked_edge_set
        ]

        expanded = self._build_expanded(remaining_edges, remaining_stations)

        try:
            return self._route_on_expanded(origin_id, dest_id, expanded)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise nx.NetworkXNoPath(
                f"No alternative path between '{origin}' and '{destination}' "
                "with current disruption"
            )

    def shortest_path_with_restrictions(
        self,
        origin: str,
        destination: str,
        station_restrictions: list[dict] | None = None,
        segment_closures: list[tuple[str, str]] | None = None,
    ) -> RouteResult:
        """Compute shortest path with typed station restrictions.

        station_restrictions: list of {"station": name_or_id, "restriction": type}
            - "closed": no entry, exit, transfer, or pass-through
            - "skip": trains pass through but don't stop (no entry/exit/transfer)
            - "no_transfer": can board/alight but cannot change lines

        segment_closures: list of (stationA, stationB) pairs where track is closed.

        Raises ValueError if origin/destination is closed or skip.
        Raises nx.NetworkXNoPath if no path exists with restrictions.
        """
        if not station_restrictions and not segment_closures:
            return self.shortest_path(origin, destination)

        origin_id = self._resolve_station(origin)
        dest_id = self._resolve_station(destination)

        # Build restrictions map: station_id → restriction type
        restrictions_map: dict[str, str] = {}
        for r in (station_restrictions or []):
            sid = self._resolve_station(r["station"])
            restrictions_map[sid] = r["restriction"]

        # Validate origin/destination
        for label, sid, name in [("Origin", origin_id, origin),
                                  ("Destination", dest_id, destination)]:
            restriction = restrictions_map.get(sid)
            if restriction in ("closed", "skip"):
                raise ValueError(
                    f"{label} station '{name}' is {restriction} by disruption"
                )

        # Build segment closure set (both directions)
        closed_segments: set[tuple[str, str]] = set()
        for seg in (segment_closures or []):
            u = self._resolve_station(seg[0])
            v = self._resolve_station(seg[1])
            closed_segments.add((u, v))
            closed_segments.add((v, u))

        # Build expanded graph with restrictions
        closed_stations = {s for s, r in restrictions_map.items() if r == "closed"}
        skip_stations = {s for s, r in restrictions_map.items() if r == "skip"}
        no_transfer_stations = {s for s, r in restrictions_map.items()
                                if r == "no_transfer"}

        G = nx.DiGraph()
        station_lines: dict[str, set[str]] = defaultdict(set)

        # Phase 1: travel edges
        for edge in self._edges_raw:
            s_from, s_to = edge["from"], edge["to"]
            line = edge["line"]
            dist = edge["distance_miles"]
            time = edge["travel_time_min"]

            # Skip segment closures
            if (s_from, s_to) in closed_segments:
                continue
            # Skip travel edges touching closed stations
            if s_from in closed_stations or s_to in closed_stations:
                continue

            station_lines[s_from].add(line)
            station_lines[s_to].add(line)

            G.add_edge(
                (s_from, line), (s_to, line),
                weight=time, distance_miles=dist, line=line,
                edge_type="travel",
            )
            G.add_edge(
                (s_to, line), (s_from, line),
                weight=time, distance_miles=dist, line=line,
                edge_type="travel",
            )

        # Phase 2: entry, exit, transfer edges
        no_entry_exit = closed_stations | skip_stations
        no_transfer = closed_stations | skip_stations | no_transfer_stations

        for sid in set(self.stations) - closed_stations:
            lines = station_lines.get(sid, set())

            if sid not in no_entry_exit:
                for line in lines:
                    G.add_edge(
                        ("enter", sid), (sid, line),
                        weight=0, distance_miles=0, edge_type="entry",
                    )
                    G.add_edge(
                        (sid, line), ("exit", sid),
                        weight=0, distance_miles=0, edge_type="exit",
                    )

            if sid not in no_transfer:
                lines_list = sorted(lines)
                for i, lineA in enumerate(lines_list):
                    for lineB in lines_list[i + 1:]:
                        G.add_edge(
                            (sid, lineA), (sid, lineB),
                            weight=TRANSFER_PENALTY_MIN, distance_miles=0,
                            edge_type="transfer",
                        )
                        G.add_edge(
                            (sid, lineB), (sid, lineA),
                            weight=TRANSFER_PENALTY_MIN, distance_miles=0,
                            edge_type="transfer",
                        )

        try:
            return self._route_on_expanded(origin_id, dest_id, G)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            raise nx.NetworkXNoPath(
                f"No path between '{origin}' and '{destination}' "
                "with current restrictions"
            )

    def _route_on_expanded(
        self, origin_id: str, dest_id: str, expanded: nx.DiGraph
    ) -> RouteResult:
        """Run Dijkstra on the expanded graph and convert to RouteResult."""
        enter_node = ("enter", origin_id)
        exit_node = ("exit", dest_id)

        if enter_node not in expanded:
            raise nx.NodeNotFound(
                f"Node '{origin_id}' is not in the expanded graph"
            )
        if exit_node not in expanded:
            raise nx.NodeNotFound(
                f"Node '{dest_id}' is not in the expanded graph"
            )

        try:
            exp_path = nx.shortest_path(
                expanded, enter_node, exit_node, weight="weight"
            )
        except nx.NetworkXNoPath:
            raise nx.NetworkXNoPath(
                f"No path found between '{origin_id}' and '{dest_id}'"
            )

        # Convert expanded path to station-level RouteResult
        path: list[str] = []
        stops: list[dict] = []
        line_sequence: list[str] = []
        total_distance = 0.0
        total_time = 0.0
        transfers = 0
        current_line: str | None = None

        for i in range(len(exp_path) - 1):
            node = exp_path[i]
            next_node = exp_path[i + 1]
            edge_data = expanded[node][next_node]
            edge_type = edge_data["edge_type"]

            if edge_type == "entry":
                # (enter, station) -> (station, line): add origin station
                station_id = node[1]
                line = next_node[1]
                current_line = line
                if line not in line_sequence:
                    line_sequence.append(line)
                station = self.stations[station_id]
                path.append(station_id)
                stops.append({
                    "station_id": station_id,
                    "station_name": station["name"],
                    "line": current_line,
                    "is_transfer": False,
                    "transfer_to": None,
                })

            elif edge_type == "travel":
                # (stationA, line) -> (stationB, line): add stationB
                station_id = next_node[0]
                total_distance += edge_data["distance_miles"]
                total_time += edge_data["weight"]
                station = self.stations[station_id]
                path.append(station_id)
                stops.append({
                    "station_id": station_id,
                    "station_name": station["name"],
                    "line": current_line,
                    "is_transfer": False,
                    "transfer_to": None,
                })

            elif edge_type == "transfer":
                # (station, lineA) -> (station, lineB): transfer at station
                new_line = next_node[1]
                transfers += 1
                total_time += edge_data["weight"]
                if new_line not in line_sequence:
                    line_sequence.append(new_line)
                # Mark the last stop as a transfer point
                if stops:
                    stops[-1]["is_transfer"] = True
                    stops[-1]["transfer_to"] = new_line
                current_line = new_line

            # exit edges: no action needed

        return RouteResult(
            path=path,
            stations=stops,
            distance_miles=round(total_distance, 2),
            estimated_minutes=round(total_time, 1),
            transfers=transfers,
            line_sequence=line_sequence,
        )

    def is_valid_path(self, path: list[str]) -> bool:
        """Check if all consecutive stations in path are adjacent in the graph."""
        if len(path) == 0:
            return False
        for i in range(len(path) - 1):
            if not self.G.has_edge(path[i], path[i + 1]):
                return False
        return True

    def adjacent_stations(self, station_id: str) -> list[str]:
        """Return neighbor station IDs for a given station.

        Raises ValueError if the station cannot be resolved.
        """
        sid = self._resolve_station(station_id)
        return list(self.G.neighbors(sid))

    def station_info(self, station_id: str) -> dict | None:
        """Return full station data, or None if the station does not exist."""
        try:
            sid = self._resolve_station(station_id)
        except ValueError:
            return None
        return self.stations.get(sid)

    def _resolve_station(self, name_or_id: str) -> str:
        """Resolve a station name or ID to its canonical ID.

        Accepts an exact station ID or a station name (case-insensitive).
        Also matches the base name without parenthetical suffixes, e.g.
        "Olympic Park" matches "Aolinpike Gongyuan (Olympic Park)".
        Raises ValueError if no match is found.
        """
        if name_or_id in self.stations:
            return name_or_id

        name_lower = name_or_id.lower().strip()
        for sid, sdata in self.stations.items():
            full = sdata["name"].lower()
            # Exact match
            if full == name_lower:
                return sid
            # Match base name (before parenthetical)
            if "(" in full:
                base = full.split("(")[0].strip()
                paren = full.split("(")[1].rstrip(")").strip()
                if name_lower == base or name_lower == paren:
                    return sid

        raise ValueError(f"Unknown station: '{name_or_id}'")
