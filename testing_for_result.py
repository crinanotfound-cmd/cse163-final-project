"""
Tests for result.py
"""
import pandas as pd
from result import (
    county_case_peaks,
    county_death_peaks,
    major_peaks_timing_comparison,
    county_case_death_correlation,
    lagged_correlation
)


def test_county_case_peaks() -> None:
    """
    Tests that county_case_peaks correctly identifies prominent
    weekly case peaks from a small synthetic dataset.
    """
    data = pd.DataFrame({
        "county": ["Test County"] * 7,
        "date": pd.date_range(
            "2020-01-01",
            periods=7,
            freq="W"
        ),
        "New cases": [1, 5, 1, 4, 1, 3, 1],
        "New deaths": [0, 0, 0, 0, 0, 0, 0]
    })

    result = county_case_peaks(
        data,
        "Test County"
    )

    assert len(result) == 3
    assert result["New cases"].tolist() == [5, 4, 3]


def test_county_death_peaks() -> None:
    """
    Tests that county_death_peaks correctly identifies prominent
    weekly death peaks from a small synthetic dataset.
    """
    data = pd.DataFrame({
        "county": ["Test County"] * 7,
        "date": pd.date_range(
            "2020-01-01",
            periods=7,
            freq="W"
        ),
        "New cases": [0, 0, 0, 0, 0, 0, 0],
        "New deaths": [1, 5, 1, 4, 1, 3, 1]
    })

    result = county_death_peaks(
        data,
        "Test County"
    )

    assert len(result) == 3
    assert result["New deaths"].tolist() == [5, 4, 3]


def test_major_peaks_timing_comparison() -> None:
    """
    Tests that major_peaks_timing_comparison correctly identifies
    the month of the maximum case and death peaks.
    """
    data = pd.DataFrame({
        "county": [
            "Test County",
            "Test County",
            "Test County"
        ],
        "date": pd.to_datetime([
            "2020-01-01",
            "2020-02-01",
            "2020-03-01"
        ]),
        "New cases": [1, 10, 2],
        "New deaths": [1, 2, 8]
    })

    result = major_peaks_timing_comparison(data)

    case_peak_month = result[
        result["Type"] == "cases"
    ]["Peak month"].iloc[0]

    death_peak_month = result[
        result["Type"] == "deaths"
    ]["Peak month"].iloc[0]

    assert case_peak_month == pd.Period("2020-02")
    assert death_peak_month == pd.Period("2020-03")


def test_county_case_death_correlation() -> None:
    """
    Tests that county_case_death_correlation correctly calculates
    Pearson correlations for counties with known relationships.
    """
    data = pd.DataFrame({
        "county": [
            "Positive County", "Positive County",
            "Positive County", "Positive County",
            "Negative County", "Negative County",
            "Negative County", "Negative County"
        ],
        "New cases": [
            1, 2, 3, 4,
            1, 2, 3, 4
        ],
        "New deaths": [
            2, 4, 6, 8,
            8, 6, 4, 2
        ]
    })
    result = county_case_death_correlation(data)

    positive_correlation = result.iloc[0]["Correlation"]
    negative_correlation = result.iloc[-1]["Correlation"]

    assert abs(positive_correlation - 1) < 0.0001
    assert abs(negative_correlation + 1) < 0.0001


def test_lagged_correlation() -> None:
    """
    Tests that lagged_correlation correctly matches previous-week
    case counts with current-week deaths and calculates the correlation.
    """
    data = pd.DataFrame({
        "county": ["Test County"] * 6,
        "date": pd.date_range(
            "2020-01-01",
            periods=6,
            freq="W"
        ),
        "New cases": [1, 4, 2, 8, 3, 7],
        "New deaths": [0, 1, 4, 2, 8, 3]
    })

    result = lagged_correlation(
        data,
        "Test County"
    )

    lag_one_correlation = result[
        result["Lag"] == 1
    ]["Correlation"].iloc[0]

    assert len(result) == 4
    assert abs(lag_one_correlation - 1.0) < 0.0001


def main() -> None:
    test_county_case_peaks()
    test_county_death_peaks()
    test_major_peaks_timing_comparison()
    test_county_case_death_correlation()
    test_lagged_correlation()
    print("All tests passed!")


if __name__ == "__main__":
    main()
