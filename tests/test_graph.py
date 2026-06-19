"""Tests for harness.graph — MetroGraph routing, transfers, station resolution."""

import pytest
import networkx as nx

from harness.graph import MetroGraph


# ===== Station resolution =====

class TestResolveStation:
    def test_resolve_by_id(self, marta_graph):
        assert marta_graph._resolve_station("MARTA-FP") == "MARTA-FP"

    def test_resolve_by_name_case_insensitive(self, marta_graph):
        assert marta_graph._resolve_station("five points") == "MARTA-FP"

    def test_resolve_by_name_doha(self, doha_graph):
        assert doha_graph._resolve_station("Msheireb") == "DOHA-MSH"

    def test_resolve_unknown_raises(self, marta_graph):
        with pytest.raises(ValueError, match="Unknown station"):
            marta_graph._resolve_station("Nonexistent Station XYZ")


# ===== Shortest path =====

class TestShortestPath:
    def test_same_station(self, marta_graph):
        result = marta_graph.shortest_path("MARTA-FP", "MARTA-FP")
        assert result.distance_miles == 0.0
        assert result.transfers == 0
        assert result.path == ["MARTA-FP"]
        assert result.line_sequence == []

    def test_adjacent_marta(self, marta_graph):
        result = marta_graph.shortest_path("MARTA-NS", "MARTA-SS")
        assert result.path == ["MARTA-NS", "MARTA-SS"]
        assert result.transfers == 0
        assert result.distance_miles == 0.95

    def test_cross_line_marta(self, marta_graph):
        """Indian Creek (Blue) -> Airport (Red south) must go through Five Points."""
        result = marta_graph.shortest_path("MARTA-IC", "MARTA-AP")
        assert "MARTA-FP" in result.path
        assert result.transfers >= 1
        assert len(result.line_sequence) >= 2

    def test_cross_line_doha(self, doha_graph):
        """Ras Bu Abboud (Gold) -> Lusail (Red) must go through Msheireb."""
        result = doha_graph.shortest_path("DOHA-RBA", "DOHA-LUS")
        assert "DOHA-MSH" in result.path
        assert result.transfers >= 1

    def test_by_name(self, marta_graph):
        result = marta_graph.shortest_path("Airport", "Five Points")
        assert result.path[0] == "MARTA-AP"
        assert result.path[-1] == "MARTA-FP"

    def test_distance_positive(self, marta_graph):
        result = marta_graph.shortest_path("MARTA-NS", "MARTA-AP")
        assert result.distance_miles > 0
        assert result.estimated_minutes > 0

    def test_unknown_station_raises(self, marta_graph):
        with pytest.raises(ValueError):
            marta_graph.shortest_path("MARTA-FP", "NONEXISTENT")


# ===== Shortest path avoiding =====

class TestShortestPathAvoiding:
    def test_blocked_edges_unaffected(self, marta_graph):
        """Blocking an irrelevant edge doesn't change the path."""
        normal = marta_graph.shortest_path("MARTA-NS", "MARTA-SS")
        alt = marta_graph.shortest_path_avoiding(
            "MARTA-NS", "MARTA-SS",
            blocked_edges=[("MARTA-FP", "MARTA-GA")],
        )
        assert alt.path == normal.path
        assert alt.distance_miles == normal.distance_miles

    def test_blocked_edge_cuts_path(self, marta_graph):
        """Blocking the only edge on a terminus isolates it."""
        with pytest.raises(nx.NetworkXNoPath):
            marta_graph.shortest_path_avoiding(
                "MARTA-NS", "MARTA-SS",
                blocked_edges=[("MARTA-NS", "MARTA-SS")],
            )

    def test_origin_blocked_raises(self, marta_graph):
        with pytest.raises(ValueError, match="blocked"):
            marta_graph.shortest_path_avoiding(
                "MARTA-FP", "MARTA-AP",
                blocked_stations=["MARTA-FP"],
            )

    def test_dest_blocked_raises(self, marta_graph):
        with pytest.raises(ValueError, match="blocked"):
            marta_graph.shortest_path_avoiding(
                "MARTA-FP", "MARTA-AP",
                blocked_stations=["MARTA-AP"],
            )

    def test_no_path_raises(self, marta_graph):
        """Block the only neighbor of a terminus -> no path out."""
        with pytest.raises(nx.NetworkXNoPath):
            marta_graph.shortest_path_avoiding(
                "MARTA-BK", "MARTA-AP",
                blocked_stations=["MARTA-AS"],
            )


# ===== Build route result =====

class TestBuildRouteResult:
    def test_single_line_no_transfer(self, marta_graph):
        result = marta_graph.shortest_path("MARTA-NS", "MARTA-SS")
        assert result.transfers == 0
        assert len(result.line_sequence) == 1
        assert result.line_sequence[0] == "red"

    def test_implicit_transfer(self, marta_graph):
        """Cross-line trip detects line change as transfer."""
        result = marta_graph.shortest_path("MARTA-IC", "MARTA-AP")
        assert result.transfers >= 1
        assert len(result.line_sequence) >= 2

    def test_distance_accumulation(self, marta_graph):
        """RouteResult distance equals sum of edge weights."""
        result = marta_graph.shortest_path("MARTA-NS", "MARTA-AP")
        total = 0.0
        for i in range(len(result.path) - 1):
            edge = marta_graph.G[result.path[i]][result.path[i + 1]]
            total += edge["distance_miles"]
        assert abs(result.distance_miles - round(total, 2)) < 0.01


# ===== is_valid_path =====

class TestIsValidPath:
    def test_adjacent_true(self, marta_graph):
        assert marta_graph.is_valid_path(["MARTA-NS", "MARTA-SS"]) is True

    def test_gap_false(self, marta_graph):
        assert marta_graph.is_valid_path(["MARTA-NS", "MARTA-AP"]) is False

    def test_empty_false(self, marta_graph):
        assert marta_graph.is_valid_path([]) is False

    def test_single_station_true(self, marta_graph):
        assert marta_graph.is_valid_path(["MARTA-FP"]) is True


# ===== Adjacent stations =====

class TestAdjacentStations:
    def test_hub_has_neighbors(self, marta_graph):
        """Five Points connects to at least 4 lines/directions."""
        neighbors = marta_graph.adjacent_stations("MARTA-FP")
        assert len(neighbors) >= 3

    def test_terminus_has_one_neighbor(self, marta_graph):
        neighbors = marta_graph.adjacent_stations("MARTA-NS")
        assert len(neighbors) == 1

    def test_doha_hub_neighbors(self, doha_graph):
        """Msheireb connects to Red, Green, and Gold."""
        neighbors = doha_graph.adjacent_stations("DOHA-MSH")
        assert len(neighbors) >= 3


# ===== Station info =====

class TestStationInfo:
    def test_exists(self, marta_graph):
        info = marta_graph.station_info("MARTA-FP")
        assert info is not None
        assert "name" in info
        assert info["name"] == "Five Points"

    def test_nonexistent(self, marta_graph):
        info = marta_graph.station_info("NONEXISTENT")
        assert info is None

    def test_by_name(self, marta_graph):
        info = marta_graph.station_info("Airport")
        assert info is not None
        assert info["id"] == "MARTA-AP"


# ===== Taipei-specific: graph cycles =====

class TestTaipeiCycles:
    def test_has_cycles(self, taipei_graph):
        """Taipei downtown should have graph cycles (unlike MARTA/Doha)."""
        import networkx as nx
        cycles = nx.cycle_basis(taipei_graph.G)
        assert len(cycles) >= 2, "Taipei should have at least 2 cycles"

    def test_cross_line_taipei(self, taipei_graph):
        """Tamsui (Red) -> Nangang (Blue) requires at least one transfer."""
        result = taipei_graph.shortest_path("TRTC-TAM", "TRTC-NKG")
        assert result.transfers >= 1
        assert len(result.line_sequence) >= 2

    def test_alternative_route_exists(self, taipei_graph):
        """Blocking Taipei Main should still allow Red→Blue via downtown loop."""
        import networkx as nx
        # Daan (Red+Brown) to Zhongxiao Fuxing (Blue+Brown) via Brown line
        result = taipei_graph.shortest_path("TRTC-DAA", "TRTC-ZXF")
        assert result.distance_miles > 0


# ===== CTA-specific: Loop cycle =====

class TestCtaLoop:
    def test_loop_has_cycles(self, cta_graph):
        """CTA Loop should have graph cycles."""
        import networkx as nx
        cycles = nx.cycle_basis(cta_graph.G)
        assert len(cycles) >= 1, "CTA should have at least 1 cycle (the Loop)"

    def test_cross_line_via_loop(self, cta_graph):
        """Kimball (Brown) -> Midway (Orange) requires Loop transfer."""
        result = cta_graph.shortest_path("CTA-KIM", "CTA-MID")
        assert result.transfers >= 1
        assert result.distance_miles > 0

    def test_ohare_to_midway(self, cta_graph):
        """O'Hare (Blue) -> Midway (Orange) — cross-system trip."""
        result = cta_graph.shortest_path("CTA-ORD", "CTA-MID")
        assert result.transfers >= 1
        assert len(result.line_sequence) >= 2


# ===== Shortest path with restrictions =====

class TestShortestPathWithRestrictions:
    def test_no_restrictions_same_as_normal(self, taipei_graph):
        """No restrictions → identical to shortest_path()."""
        normal = taipei_graph.shortest_path("TRTC-XDN", "TRTC-LUZ")
        restricted = taipei_graph.shortest_path_with_restrictions(
            "TRTC-XDN", "TRTC-LUZ",
        )
        assert restricted.path == normal.path
        assert restricted.distance_miles == normal.distance_miles

    def test_skip_allows_passthrough(self, taipei_graph):
        """Guting skip: trains pass through, route via SJN transfer."""
        result = taipei_graph.shortest_path_with_restrictions(
            "TRTC-XDN", "TRTC-LUZ",
            station_restrictions=[{"station": "TRTC-GUT", "restriction": "skip"}],
        )
        assert result.path is not None
        assert len(result.path) > 1
        assert result.transfers >= 1
        # Guting is on the path (trains pass through) but transfer is elsewhere
        assert "TRTC-GUT" in result.path

    def test_skip_origin_raises(self, taipei_graph):
        """Cannot start from a skip station."""
        with pytest.raises(ValueError, match="skip"):
            taipei_graph.shortest_path_with_restrictions(
                "TRTC-GUT", "TRTC-LUZ",
                station_restrictions=[{"station": "TRTC-GUT", "restriction": "skip"}],
            )

    def test_skip_destination_raises(self, taipei_graph):
        """Cannot end at a skip station."""
        with pytest.raises(ValueError, match="skip"):
            taipei_graph.shortest_path_with_restrictions(
                "TRTC-XDN", "TRTC-GUT",
                station_restrictions=[{"station": "TRTC-GUT", "restriction": "skip"}],
            )

    def test_closed_blocks_all(self, taipei_graph):
        """Guting closed severs Green↔Orange link → no path Xindian→Luzhou."""
        with pytest.raises(nx.NetworkXNoPath):
            taipei_graph.shortest_path_with_restrictions(
                "TRTC-XDN", "TRTC-LUZ",
                station_restrictions=[{"station": "TRTC-GUT", "restriction": "closed"}],
            )

    def test_no_transfer_allows_entry_exit(self, taipei_graph):
        """Guting no_transfer: can still ride a single line through it."""
        result = taipei_graph.shortest_path_with_restrictions(
            "TRTC-GUT", "TRTC-TAB",
            station_restrictions=[{"station": "TRTC-GUT", "restriction": "no_transfer"}],
        )
        assert result.path == ["TRTC-GUT", "TRTC-TAB"]
        assert result.transfers == 0

    def test_segment_closure_raises_no_path(self, taipei_graph):
        """Closing the only segment between adjacent stations blocks the path."""
        with pytest.raises(nx.NetworkXNoPath):
            taipei_graph.shortest_path_with_restrictions(
                "TRTC-TAB", "TRTC-GGN",
                segment_closures=[("TRTC-TAB", "TRTC-GGN")],
            )

    def test_segment_closure_finds_alternative(self, taipei_graph):
        """Closing CKS↔Guting segment on Green, route via Red→Orange."""
        result = taipei_graph.shortest_path_with_restrictions(
            "TRTC-CKS", "TRTC-DXI",
            segment_closures=[("TRTC-CKS", "TRTC-GUT")],
        )
        assert result.path is not None
        assert result.transfers >= 1
        assert result.distance_miles > 0


# ===== Line topology helpers =====

@pytest.fixture(scope="module")
def beijing_graph():
    from pathlib import Path
    return MetroGraph(Path(__file__).resolve().parent.parent / "data" / "systems" / "beijing")


class TestLineTopology:
    def test_lines_for_station_single_line(self, marta_graph):
        assert marta_graph.lines_for_station("MARTA-NS") == {"red"}

    def test_lines_for_station_hub(self, marta_graph):
        # Five Points is served by all MARTA lines
        assert marta_graph.lines_for_station("MARTA-FP") >= {"red", "gold", "blue", "green"}

    def test_is_loop_line_beijing_2(self, beijing_graph):
        assert beijing_graph.is_loop_line("2") is True

    def test_is_loop_line_beijing_10(self, beijing_graph):
        assert beijing_graph.is_loop_line("10") is True

    def test_is_loop_line_marta_red_false(self, marta_graph):
        assert marta_graph.is_loop_line("red") is False

    def test_line_terminals_marta_red(self, marta_graph):
        terms = set(marta_graph.line_terminals("red"))
        # MARTA Red runs between North Springs and Airport
        assert terms == {"MARTA-NS", "MARTA-AP"}

    def test_line_terminals_empty_on_loop(self, beijing_graph):
        assert beijing_graph.line_terminals("10") == []


class TestExpandLineClosures:
    def test_whole_line_closure_beijing_10(self, beijing_graph):
        segs = beijing_graph.expand_line_closures([{"line": "10"}])
        # Line 10 has 45 stations; loop topology ⇒ each station has 2 neighbours ⇒ 45 edges
        assert len(segs) == 45

    def test_partial_closure_inclusive(self, beijing_graph):
        # Yanfang line: 9 stations linear. Close between first two stations → 1 edge.
        stations = beijing_graph.lines["yanfang"]["stations"]
        segs = beijing_graph.expand_line_closures([
            {"line": "yanfang", "from_station": stations[0], "to_station": stations[1]}
        ])
        assert len(segs) == 1

    def test_partial_closure_rejects_loop(self, beijing_graph):
        stations = beijing_graph.lines["10"]["stations"]
        with pytest.raises(ValueError, match="loop"):
            beijing_graph.expand_line_closures([
                {"line": "10", "from_station": stations[0], "to_station": stations[5]}
            ])

    def test_unknown_line_raises(self, marta_graph):
        with pytest.raises(ValueError, match="Unknown line"):
            marta_graph.expand_line_closures([{"line": "nonexistent"}])

    def test_partial_closure_missing_endpoint_raises(self, beijing_graph):
        stations = beijing_graph.lines["yanfang"]["stations"]
        with pytest.raises(ValueError, match="requires both"):
            beijing_graph.expand_line_closures([
                {"line": "yanfang", "from_station": stations[0]}
            ])

    def test_whole_line_closure_reroutes(self, beijing_graph):
        """Closing Line 10 entirely should still yield an alternative for Yanshan→Dongbabei."""
        segs = beijing_graph.expand_line_closures([{"line": "10"}])
        result = beijing_graph.shortest_path_with_restrictions(
            "Yanshan", "Dongbabei", segment_closures=segs
        )
        assert "10" not in result.line_sequence
        assert result.transfers >= 1
