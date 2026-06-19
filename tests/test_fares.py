"""Tests for harness.fares — FareCalculator arithmetic for all systems."""

import pytest

from harness.fares import FareCalculator


# ===== MARTA fares ($2.50 flat, $1.25 senior/disabled, children free max 2/adult) =====

class TestMartaFares:
    def test_single_adult(self, marta_fare_calc):
        result = marta_fare_calc.calculate({"adults": 1})
        assert result.total == 2.50
        assert result.currency == "USD"

    def test_two_adults(self, marta_fare_calc):
        result = marta_fare_calc.calculate({"adults": 2})
        assert result.total == 5.00

    def test_senior_discount(self, marta_fare_calc):
        result = marta_fare_calc.calculate({"seniors": 1})
        assert result.total == 1.25

    def test_disabled_discount(self, marta_fare_calc):
        result = marta_fare_calc.calculate({"disabled": 1})
        assert result.total == 1.25

    def test_children_free_within_limit(self, marta_fare_calc):
        """1 adult + 2 children = $2.50 (2 children ride free)."""
        result = marta_fare_calc.calculate({"adults": 1, "children": 2})
        assert result.total == 2.50

    def test_children_exceed_free_limit(self, marta_fare_calc):
        """1 adult + 3 children = $5.00 (2 free + 1 pays $2.50)."""
        result = marta_fare_calc.calculate({"adults": 1, "children": 3})
        assert result.total == 5.00

    def test_children_no_adult(self, marta_fare_calc):
        """0 adults + 2 children = $5.00 (no free allowance without paying adult)."""
        result = marta_fare_calc.calculate({"children": 2})
        assert result.total == 5.00

    def test_children_exceed_free_two_adults(self, marta_fare_calc):
        """2 adults + 4 children = $5.00 (4 free: 2 per adult)."""
        result = marta_fare_calc.calculate({"adults": 2, "children": 4})
        assert result.total == 5.00

    def test_children_exceed_free_two_adults_five(self, marta_fare_calc):
        """2 adults + 5 children = $7.50 (4 free + 1 pays $2.50)."""
        result = marta_fare_calc.calculate({"adults": 2, "children": 5})
        assert result.total == 7.50

    def test_mixed_passengers(self, marta_fare_calc):
        """1A + 1S + 1D + 2C = $2.50 + $1.25 + $1.25 + 0 = $5.00."""
        result = marta_fare_calc.calculate({
            "adults": 1, "seniors": 1, "disabled": 1, "children": 2,
        })
        assert result.total == 5.00


# ===== Doha fares (QR 2 flat, QR 10 gold, no senior discount, children free max 2/adult) =====

class TestDohaFares:
    def test_single_adult_standard(self, doha_fare_calc):
        result = doha_fare_calc.calculate({"adults": 1})
        assert result.total == 2
        assert result.currency == "QAR"

    def test_single_adult_gold(self, doha_fare_calc):
        result = doha_fare_calc.calculate(
            {"adults": 1}, payment_method="gold_travel_card",
        )
        assert result.total == 10

    def test_two_adults_gold(self, doha_fare_calc):
        result = doha_fare_calc.calculate(
            {"adults": 2}, payment_method="gold_travel_card",
        )
        assert result.total == 20

    def test_no_senior_discount(self, doha_fare_calc):
        """Doha has no senior discount — seniors pay full fare."""
        result = doha_fare_calc.calculate({"seniors": 1})
        assert result.total == 2

    def test_children_free_standard(self, doha_fare_calc):
        result = doha_fare_calc.calculate({"adults": 1, "children": 2})
        assert result.total == 2

    def test_children_exceed_free_limit(self, doha_fare_calc):
        """1 adult + 3 children = QR 4 (2 free + 1 pays QR 2)."""
        result = doha_fare_calc.calculate({"adults": 1, "children": 3})
        assert result.total == 4

    def test_children_gold_class(self, doha_fare_calc):
        """1 adult + 2 children gold = QR 10 (gold fare, children free)."""
        result = doha_fare_calc.calculate(
            {"adults": 1, "children": 2}, payment_method="gold_travel_card",
        )
        assert result.total == 10


# ===== BART fares (distance-based, brackets + surcharges, 37.5% senior/disabled) =====

class TestBartFares:
    def test_single_adult_short(self, bart_fare_calc):
        """5 miles = bracket ≤6 = $2.15."""
        result = bart_fare_calc.calculate(
            {"adults": 1}, route_distance_miles=5.0,
            origin_id="BART-EMBR", destination_id="BART-DALY",
        )
        assert result.total == 2.15
        assert result.currency == "USD"

    def test_single_adult_medium(self, bart_fare_calc):
        """10 miles = bracket ≤12 = $3.50."""
        result = bart_fare_calc.calculate(
            {"adults": 1}, route_distance_miles=10.0,
            origin_id="BART-EMBR", destination_id="BART-DALY",
        )
        assert result.total == 3.50

    def test_single_adult_long(self, bart_fare_calc):
        """35 miles = bracket ≤40 = $8.50."""
        result = bart_fare_calc.calculate(
            {"adults": 1}, route_distance_miles=35.0,
            origin_id="BART-EMBR", destination_id="BART-DALY",
        )
        assert result.total == 8.50

    def test_transbay_surcharge(self, bart_fare_calc):
        """5 miles + Transbay = $2.15 + $1.40 = $3.55."""
        result = bart_fare_calc.calculate(
            {"adults": 1}, route_distance_miles=5.0,
            origin_id="BART-12TH", destination_id="BART-EMBR",
        )
        assert result.total == 3.55

    def test_sfo_surcharge(self, bart_fare_calc):
        """15 miles + SFO = $5.00 + $4.95 = $9.95."""
        result = bart_fare_calc.calculate(
            {"adults": 1}, route_distance_miles=15.0,
            origin_id="BART-EMBR", destination_id="BART-SFO",
        )
        assert result.total == 9.95

    def test_oakl_surcharge(self, bart_fare_calc):
        """5 miles + OAK = $2.15 + $6.70 = $8.85."""
        result = bart_fare_calc.calculate(
            {"adults": 1}, route_distance_miles=5.0,
            origin_id="BART-COLS", destination_id="BART-OAKL",
        )
        assert result.total == 8.85

    def test_transbay_plus_sfo(self, bart_fare_calc):
        """20 miles + Transbay + SFO = $5.00 + $1.40 + $4.95 = $11.35."""
        result = bart_fare_calc.calculate(
            {"adults": 1}, route_distance_miles=20.0,
            origin_id="BART-12TH", destination_id="BART-SFO",
        )
        assert result.total == 11.35

    def test_senior_multiplier(self, bart_fare_calc):
        """Senior pays 37.5%: $2.15 * 0.375 = $0.81 (rounded)."""
        result = bart_fare_calc.calculate(
            {"seniors": 1}, route_distance_miles=5.0,
            origin_id="BART-EMBR", destination_id="BART-DALY",
        )
        assert result.total == 0.81

    def test_disabled_multiplier(self, bart_fare_calc):
        """Disabled pays 37.5%: $2.15 * 0.375 = $0.81 (rounded)."""
        result = bart_fare_calc.calculate(
            {"disabled": 1}, route_distance_miles=5.0,
            origin_id="BART-EMBR", destination_id="BART-DALY",
        )
        assert result.total == 0.81

    def test_senior_with_transbay(self, bart_fare_calc):
        """Senior Transbay: ($2.15 + $1.40) * 0.375 = $3.55 * 0.375 = $1.33."""
        result = bart_fare_calc.calculate(
            {"seniors": 1}, route_distance_miles=5.0,
            origin_id="BART-12TH", destination_id="BART-EMBR",
        )
        assert result.total == 1.33

    def test_children_free_within_limit(self, bart_fare_calc):
        """1 adult + 2 children = $2.15."""
        result = bart_fare_calc.calculate(
            {"adults": 1, "children": 2}, route_distance_miles=5.0,
            origin_id="BART-EMBR", destination_id="BART-DALY",
        )
        assert result.total == 2.15

    def test_children_exceed_free(self, bart_fare_calc):
        """1 adult + 3 children = $2.15 + $2.15 = $4.30."""
        result = bart_fare_calc.calculate(
            {"adults": 1, "children": 3}, route_distance_miles=5.0,
            origin_id="BART-EMBR", destination_id="BART-DALY",
        )
        assert result.total == 4.30

    def test_no_distance_raises(self, bart_fare_calc):
        """Distance fare model requires route_distance_miles."""
        with pytest.raises(ValueError, match="route_distance_miles"):
            bart_fare_calc.calculate({"adults": 1})

    def test_no_surcharge_same_side(self, bart_fare_calc):
        """Both stations on same side = no Transbay surcharge."""
        result = bart_fare_calc.calculate(
            {"adults": 1}, route_distance_miles=5.0,
            origin_id="BART-MCAR", destination_id="BART-12TH",
        )
        assert result.total == 2.15  # No surcharge


# ===== Taipei fares (distance-based, TWD brackets, 50% senior/disabled, children free max 2/adult) =====

class TestTaipeiFares:
    def test_single_adult_short(self, taipei_fare_calc):
        """3.20 miles = bracket ≤5.0 = NT$25."""
        result = taipei_fare_calc.calculate(
            {"adults": 1}, route_distance_miles=3.20,
            origin_id="TRTC-TPM", destination_id="TRTC-T101",
        )
        assert result.total == 25
        assert result.currency == "TWD"

    def test_single_adult_medium(self, taipei_fare_calc):
        """7.00 miles = bracket ≤7.5 = NT$30."""
        result = taipei_fare_calc.calculate(
            {"adults": 1}, route_distance_miles=7.0,
            origin_id="TRTC-TPM", destination_id="TRTC-NKG",
        )
        assert result.total == 30

    def test_single_adult_long(self, taipei_fare_calc):
        """14.10 miles = bracket ≤14.9 = NT$45."""
        result = taipei_fare_calc.calculate(
            {"adults": 1}, route_distance_miles=14.10,
            origin_id="TRTC-TAM", destination_id="TRTC-XSH",
        )
        assert result.total == 45

    def test_senior_half_price(self, taipei_fare_calc):
        """Senior pays 50%: NT$25 * 0.5 = NT$12.5."""
        result = taipei_fare_calc.calculate(
            {"seniors": 1}, route_distance_miles=3.20,
            origin_id="TRTC-TPM", destination_id="TRTC-T101",
        )
        assert result.total == 12.5

    def test_children_free_within_limit(self, taipei_fare_calc):
        """1 adult + 2 children = NT$25 (2 children ride free)."""
        result = taipei_fare_calc.calculate(
            {"adults": 1, "children": 2}, route_distance_miles=3.20,
            origin_id="TRTC-TPM", destination_id="TRTC-T101",
        )
        assert result.total == 25

    def test_no_distance_raises(self, taipei_fare_calc):
        """Distance fare model requires route_distance_miles."""
        with pytest.raises(ValueError, match="route_distance_miles"):
            taipei_fare_calc.calculate({"adults": 1})


# ===== CTA fares (flat_with_exceptions: $2.50 Ventra, $3.00 contactless, $3.50 disposable, $5.00 O'Hare) =====

class TestCtaFares:
    def test_single_adult_ventra(self, cta_fare_calc):
        result = cta_fare_calc.calculate({"adults": 1}, payment_method="ventra")
        assert result.total == 2.50
        assert result.currency == "USD"

    def test_single_adult_contactless(self, cta_fare_calc):
        """Contactless adds $0.50 adjustment: $2.50 + $0.50 = $3.00."""
        result = cta_fare_calc.calculate({"adults": 1}, payment_method="contactless")
        assert result.total == 3.00

    def test_single_adult_disposable(self, cta_fare_calc):
        """Disposable adds $1.00 adjustment: $2.50 + $1.00 = $3.50."""
        result = cta_fare_calc.calculate({"adults": 1}, payment_method="disposable_ticket")
        assert result.total == 3.50

    def test_ohare_override_ventra(self, cta_fare_calc):
        """O'Hare flat $5.00 regardless of payment method."""
        result = cta_fare_calc.calculate(
            {"adults": 1}, payment_method="ventra",
            origin_id="CTA-ORD", destination_id="CTA-CLK",
        )
        assert result.total == 5.00

    def test_ohare_override_contactless(self, cta_fare_calc):
        """O'Hare ignores payment adjustment: still $5.00."""
        result = cta_fare_calc.calculate(
            {"adults": 1}, payment_method="contactless",
            origin_id="CTA-ORD", destination_id="CTA-CLK",
        )
        assert result.total == 5.00

    def test_ohare_override_disposable(self, cta_fare_calc):
        """O'Hare ignores payment adjustment: still $5.00."""
        result = cta_fare_calc.calculate(
            {"adults": 1}, payment_method="disposable_ticket",
            origin_id="CTA-CLK", destination_id="CTA-ORD",
        )
        assert result.total == 5.00

    def test_senior_flat(self, cta_fare_calc):
        """Senior flat $1.25 regardless of route."""
        result = cta_fare_calc.calculate({"seniors": 1}, payment_method="ventra")
        assert result.total == 1.25

    def test_disabled_flat(self, cta_fare_calc):
        result = cta_fare_calc.calculate({"disabled": 1}, payment_method="ventra")
        assert result.total == 1.25

    def test_children_under_7_free(self, cta_fare_calc):
        """1 adult + 2 children = $2.50 (children free, max 2 per adult)."""
        result = cta_fare_calc.calculate(
            {"adults": 1, "children": 2}, payment_method="ventra",
        )
        assert result.total == 2.50

    def test_children_exceed_free(self, cta_fare_calc):
        """1 adult + 3 children = $2.50 + $2.50 = $5.00."""
        result = cta_fare_calc.calculate(
            {"adults": 1, "children": 3}, payment_method="ventra",
        )
        assert result.total == 5.00

    def test_no_distance_needed(self, cta_fare_calc):
        """Flat model works without route_distance_miles."""
        result = cta_fare_calc.calculate({"adults": 1}, payment_method="ventra")
        assert result.total == 2.50


# ===== Gold class detection =====

class TestGoldClassDetection:
    def test_by_config(self, doha_fare_calc):
        assert doha_fare_calc._is_gold_class("gold_travel_card") is True

    def test_by_name(self, doha_fare_calc):
        assert doha_fare_calc._is_gold_class("gold") is True

    def test_not_gold_travel_card(self, doha_fare_calc):
        assert doha_fare_calc._is_gold_class("travel_card") is False

    def test_not_gold_contactless(self, doha_fare_calc):
        assert doha_fare_calc._is_gold_class("contactless") is False


# ===== Validation =====

class TestValidatePassengers:
    def test_negative_raises(self, marta_fare_calc):
        with pytest.raises(ValueError):
            marta_fare_calc.calculate({"adults": -1})

    def test_string_raises(self, marta_fare_calc):
        with pytest.raises(ValueError):
            marta_fare_calc.calculate({"adults": "two"})
