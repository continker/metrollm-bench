"""Per-system fare calculators."""

import json
from pathlib import Path
from dataclasses import dataclass


@dataclass
class FareResult:
    items: list[dict]      # [{label, amount, currency}]
    subtotal: float
    discounts: list[dict]  # [{label, amount, currency}]
    total: float
    currency: str


class FareCalculator:
    def __init__(self, system_dir: Path):
        with open(system_dir / "fares.json") as f:
            self.rules: dict = json.load(f)
        self.system: str = self.rules["system"]
        self.model: str = self.rules["model"]
        self.currency: str = self.rules.get("currency", "USD")

    def calculate(
        self,
        passengers: dict,  # {adults: int, children: int, seniors: int, disabled: int}
        ticket_type: str = "single",
        payment_method: str = "smartcard",
        route_distance_miles: float | None = None,
        origin_id: str | None = None,
        destination_id: str | None = None,
    ) -> FareResult:
        """Calculate fare based on system rules.

        Supports 'flat' and 'distance' fare models.
        Raises NotImplementedError for unrecognised fare models.
        Raises ValueError if passenger counts are negative or the passenger
        dict contains no recognised keys with a positive value.
        """
        self._validate_passengers(passengers)

        if self.model == "flat":
            return self._flat_fare(passengers, ticket_type, payment_method)

        if self.model == "flat_with_exceptions":
            return self._flat_with_exceptions(
                passengers, ticket_type, payment_method, origin_id, destination_id,
            )

        if self.model == "distance":
            return self._distance_fare(
                passengers, ticket_type, payment_method,
                route_distance_miles, origin_id, destination_id,
            )

        raise NotImplementedError(f"Fare model '{self.model}' not yet implemented")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_passengers(self, passengers: dict) -> None:
        """Raise ValueError if any passenger count is negative."""
        for key in ("adults", "children", "seniors", "disabled"):
            value = passengers.get(key, 0)
            if not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"Passenger count for '{key}' must be a non-negative integer, "
                    f"got {value!r}"
                )

    def _is_gold_class(self, payment_method: str) -> bool:
        """Check if the payment method indicates gold class."""
        pm_info = self.rules.get("payment_methods", {}).get(payment_method, {})
        if pm_info.get("class") == "gold":
            return True
        # Also match by name convention
        return "gold" in payment_method.lower()

    def _flat_fare(
        self,
        passengers: dict,
        ticket_type: str,
        payment_method: str,
    ) -> FareResult:
        """Flat-rate fare calculation."""
        # Determine base fare: use gold_fare if payment method is gold class
        # and the system supports it, otherwise standard base_fare
        if "gold_fare" in self.rules and self._is_gold_class(payment_method):
            base: float = self.rules["gold_fare"]
        else:
            base = self.rules["base_fare"]

        currency = self.currency
        items: list[dict] = []
        discounts_list: list[dict] = []

        adults = passengers.get("adults", 0)
        children = passengers.get("children", 0)
        seniors = passengers.get("seniors", 0)
        disabled = passengers.get("disabled", 0)

        # Adults at full base fare
        if adults > 0:
            items.append(
                {
                    "label": f"Adult x{adults}",
                    "amount": round(base * adults, 2),
                    "currency": currency,
                }
            )

        # Seniors (reduced fare) — only if the system offers a senior discount
        discounts = self.rules.get("discounts", {})
        senior_discount = discounts.get("senior_65_plus")
        if seniors > 0:
            if senior_discount:
                senior_fare: float = senior_discount["fare"]
                items.append(
                    {
                        "label": f"Senior x{seniors}",
                        "amount": round(senior_fare * seniors, 2),
                        "currency": currency,
                    }
                )
            else:
                # No senior discount — charge full fare
                items.append(
                    {
                        "label": f"Senior x{seniors}",
                        "amount": round(base * seniors, 2),
                        "currency": currency,
                    }
                )

        # Disabled riders (reduced fare) — only if the system offers it
        disabled_discount = discounts.get("disabled")
        if disabled > 0:
            if disabled_discount:
                disabled_fare: float = disabled_discount["fare"]
                items.append(
                    {
                        "label": f"Disabled x{disabled}",
                        "amount": round(disabled_fare * disabled, 2),
                        "currency": currency,
                    }
                )
            else:
                # No disabled discount — charge full fare
                items.append(
                    {
                        "label": f"Disabled x{disabled}",
                        "amount": round(base * disabled, 2),
                        "currency": currency,
                    }
                )

        # Children: free up to max_per_adult per paying adult.
        # Any additional children beyond the free allowance pay full base fare.
        children_cfg = discounts.get("children", {})
        child_qualifier = children_cfg.get("qualifier", "free")
        max_free_per_adult: int = children_cfg.get("max_per_adult", 2)
        paying_adults_total = adults + seniors + disabled
        free_children = (
            min(children, paying_adults_total * max_free_per_adult)
            if paying_adults_total > 0
            else 0
        )
        paid_children = children - free_children

        if free_children > 0:
            discounts_list.append(
                {
                    "label": f"Child ({child_qualifier}, free) x{free_children}",
                    "amount": 0.0,
                    "currency": currency,
                }
            )
        if paid_children > 0:
            items.append(
                {
                    "label": f"Child (fare required) x{paid_children}",
                    "amount": round(base * paid_children, 2),
                    "currency": currency,
                }
            )

        subtotal = round(sum(i["amount"] for i in items), 2)
        total_discounts = round(sum(d["amount"] for d in discounts_list), 2)

        return FareResult(
            items=items,
            subtotal=subtotal,
            discounts=discounts_list,
            total=round(subtotal - total_discounts, 2),
            currency=currency,
        )

    def _flat_with_exceptions(
        self,
        passengers: dict,
        ticket_type: str,
        payment_method: str,
        origin_id: str | None = None,
        destination_id: str | None = None,
    ) -> FareResult:
        """Flat fare with payment-method adjustments and station overrides."""
        # Check station overrides (e.g. O'Hare $5.00 flat)
        overrides = self.rules.get("station_overrides", {})
        override_fare = None
        ignores_adjustment = False
        for station_id in (origin_id, destination_id):
            if station_id and station_id in overrides:
                override_fare = overrides[station_id]["fare"]
                ignores_adjustment = overrides[station_id].get(
                    "ignores_payment_adjustment", False
                )
                break

        # Determine per-ride fare
        if override_fare is not None:
            if ignores_adjustment:
                per_ride = override_fare
            else:
                pm_info = self.rules.get("payment_methods", {}).get(payment_method, {})
                per_ride = override_fare + pm_info.get("fare_adjustment", 0.0)
        else:
            pm_info = self.rules.get("payment_methods", {}).get(payment_method, {})
            per_ride = self.rules["base_fare"] + pm_info.get("fare_adjustment", 0.0)

        per_ride = round(per_ride, 2)
        currency = self.currency
        items: list[dict] = []
        discounts_list: list[dict] = []

        adults = passengers.get("adults", 0)
        children = passengers.get("children", 0)
        seniors = passengers.get("seniors", 0)
        disabled = passengers.get("disabled", 0)

        # Adults at per-ride fare
        if adults > 0:
            items.append({
                "label": f"Adult x{adults}",
                "amount": round(per_ride * adults, 2),
                "currency": currency,
            })

        # Seniors — flat reduced fare from discounts config
        discounts = self.rules.get("discounts", {})
        senior_cfg = discounts.get("senior_65_plus")
        if seniors > 0:
            if senior_cfg and "fare" in senior_cfg:
                senior_fare = senior_cfg["fare"]
            else:
                senior_fare = per_ride
            items.append({
                "label": f"Senior x{seniors}",
                "amount": round(senior_fare * seniors, 2),
                "currency": currency,
            })

        # Disabled — flat reduced fare
        disabled_cfg = discounts.get("disabled")
        if disabled > 0:
            if disabled_cfg and "fare" in disabled_cfg:
                disabled_fare = disabled_cfg["fare"]
            else:
                disabled_fare = per_ride
            items.append({
                "label": f"Disabled x{disabled}",
                "amount": round(disabled_fare * disabled, 2),
                "currency": currency,
            })

        # Children: free up to max_per_adult per paying adult
        children_cfg = discounts.get("children", {})
        child_qualifier = children_cfg.get("qualifier", "free")
        max_free: int = children_cfg.get("max_per_adult", 2)
        paying_total = adults + seniors + disabled
        free_children = (
            min(children, paying_total * max_free)
            if paying_total > 0
            else 0
        )
        paid_children = children - free_children

        if free_children > 0:
            discounts_list.append({
                "label": f"Child ({child_qualifier}, free) x{free_children}",
                "amount": 0.0,
                "currency": currency,
            })
        if paid_children > 0:
            items.append({
                "label": f"Child (fare required) x{paid_children}",
                "amount": round(per_ride * paid_children, 2),
                "currency": currency,
            })

        subtotal = round(sum(i["amount"] for i in items), 2)
        total_discounts = round(sum(d["amount"] for d in discounts_list), 2)

        return FareResult(
            items=items,
            subtotal=subtotal,
            discounts=discounts_list,
            total=round(subtotal - total_discounts, 2),
            currency=currency,
        )

    # ------------------------------------------------------------------
    # Distance-based fare model
    # ------------------------------------------------------------------

    def _get_bracket_fare(self, distance_miles: float) -> float:
        """Look up fare from distance brackets."""
        for bracket in self.rules["fare_brackets"]:
            if distance_miles <= bracket["max_miles"]:
                return bracket["fare"]
        # Fallback: last bracket covers everything
        return self.rules["fare_brackets"][-1]["fare"]

    def _compute_surcharges(
        self, origin_id: str | None, destination_id: str | None,
    ) -> list[dict]:
        """Return list of applicable surcharges as {label, amount, replaces_base} dicts.

        Supports three surcharge formats in fares.json:
        - Transbay-style: {sf_side, east_bay_side, amount} — triggers when crossing
        - Single-station: {station, amount} — triggers when origin or dest matches
        - Multi-station: {stations, amount} — triggers when origin or dest in list
        - replaces_base: if true, surcharge replaces bracket fare (e.g. airport express)
        """
        surcharges_config = self.rules.get("surcharges", {})
        result: list[dict] = []

        for key, cfg in surcharges_config.items():
            if not isinstance(cfg, dict):
                continue

            # Transbay-style: cross-bay check
            if "sf_side" in cfg and "east_bay_side" in cfg:
                if origin_id and destination_id:
                    sf = set(cfg.get("sf_side", []))
                    eb = set(cfg.get("east_bay_side", []))
                    crosses = (
                        (origin_id in sf and destination_id in eb)
                        or (origin_id in eb and destination_id in sf)
                    )
                    if crosses:
                        result.append({
                            "label": cfg.get("description", f"{key} surcharge"),
                            "amount": cfg["amount"],
                            "replaces_base": cfg.get("replaces_base", False),
                        })
                continue

            # Station-based surcharges
            matched = False
            if "station" in cfg:
                # Single station format (BART sfo_airport, oakl_airport)
                matched = origin_id == cfg["station"] or destination_id == cfg["station"]
            elif "stations" in cfg:
                # Multi-station format (Beijing airport express)
                station_set = set(cfg["stations"])
                matched = (origin_id in station_set) or (destination_id in station_set)

            if matched:
                result.append({
                    "label": cfg.get("description", f"{key} surcharge"),
                    "amount": cfg["amount"],
                    "replaces_base": cfg.get("replaces_base", False),
                })

        return result

    def _distance_fare(
        self,
        passengers: dict,
        ticket_type: str,
        payment_method: str,
        route_distance_miles: float | None,
        origin_id: str | None,
        destination_id: str | None,
    ) -> FareResult:
        """Distance-based fare with bracket lookup + surcharges."""
        if route_distance_miles is None:
            raise ValueError("route_distance_miles required for distance fare model")

        base = self._get_bracket_fare(route_distance_miles)
        surcharges = self._compute_surcharges(origin_id, destination_id)

        # Check if any surcharge replaces the base fare (e.g. airport express flat fare)
        replacing = [s for s in surcharges if s.get("replaces_base")]
        if replacing:
            # Use the highest replacing surcharge as the flat fare
            per_ride = max(s["amount"] for s in replacing)
        else:
            surcharge_total = sum(s["amount"] for s in surcharges)
            per_ride = round(base + surcharge_total, 2)

        currency = self.currency
        discounts = self.rules.get("discounts", {})
        items: list[dict] = []
        discounts_list: list[dict] = []

        adults = passengers.get("adults", 0)
        children = passengers.get("children", 0)
        seniors = passengers.get("seniors", 0)
        disabled = passengers.get("disabled", 0)

        # Adults pay full per-ride fare
        if adults > 0:
            items.append({
                "label": f"Adult x{adults}",
                "amount": round(per_ride * adults, 2),
                "currency": currency,
            })

        # Seniors — multiplier-based discount
        senior_cfg = discounts.get("senior_65_plus")
        if seniors > 0:
            if senior_cfg and "multiplier" in senior_cfg:
                senior_fare = round(per_ride * senior_cfg["multiplier"], 2)
            elif senior_cfg and "fare" in senior_cfg:
                senior_fare = senior_cfg["fare"]
            else:
                senior_fare = per_ride
            items.append({
                "label": f"Senior x{seniors}",
                "amount": round(senior_fare * seniors, 2),
                "currency": currency,
            })

        # Disabled — multiplier-based discount
        disabled_cfg = discounts.get("disabled")
        if disabled > 0:
            if disabled_cfg and "multiplier" in disabled_cfg:
                disabled_fare = round(per_ride * disabled_cfg["multiplier"], 2)
            elif disabled_cfg and "fare" in disabled_cfg:
                disabled_fare = disabled_cfg["fare"]
            else:
                disabled_fare = per_ride
            items.append({
                "label": f"Disabled x{disabled}",
                "amount": round(disabled_fare * disabled, 2),
                "currency": currency,
            })

        # Children: free up to max_per_adult per paying adult
        children_cfg = discounts.get("children", {})
        child_qualifier = children_cfg.get("qualifier", "free")
        max_free_per_adult: int = children_cfg.get("max_per_adult", 2)
        paying_adults_total = adults + seniors + disabled
        free_children = (
            min(children, paying_adults_total * max_free_per_adult)
            if paying_adults_total > 0
            else 0
        )
        paid_children = children - free_children

        if free_children > 0:
            discounts_list.append({
                "label": f"Child ({child_qualifier}, free) x{free_children}",
                "amount": 0.0,
                "currency": currency,
            })
        if paid_children > 0:
            items.append({
                "label": f"Child (fare required) x{paid_children}",
                "amount": round(per_ride * paid_children, 2),
                "currency": currency,
            })

        # Add surcharge line items for transparency
        for s in surcharges:
            items.append({
                "label": s["label"],
                "amount": 0.0,  # already included in per-ride
                "currency": currency,
            })

        subtotal = round(sum(i["amount"] for i in items), 2)
        total_discounts = round(sum(d["amount"] for d in discounts_list), 2)

        return FareResult(
            items=items,
            subtotal=subtotal,
            discounts=discounts_list,
            total=round(subtotal - total_discounts, 2),
            currency=currency,
        )
