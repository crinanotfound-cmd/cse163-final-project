"""
Tests for analysis.py.
"""
import pandas as pd

from analysis import (
    clean_data,
    summarize_counties,
    summarize_missing_data,
    summarize_quantitative
)


def test_clean_data() -> None:
    """
    Tests filtering and data conversion in clean_data.
    """
    test_data = pd.DataFrame({
        "county": [
            "King County",
            "Pierce County",
            "Unallocated Washington",
            "Multnomah County"
        ],
        "state": ["WA", "WA", "WA", "OR"],
        "date": [
            "01/01/2022",
            "01/08/2022",
            "01/15/2022",
            "01/22/2022"
        ],
        "New cases": ["1,200", "50", "25", "100"],
        "New deaths": ["4", "2", "1", "3"]
    })

    result = clean_data(test_data)

    assert len(result) == 2
    assert result["county"].tolist() == [
        "King County",
        "Pierce County"
    ]
    assert result["New cases"].tolist() == [1200, 50]
    assert result["New deaths"].tolist() == [4, 2]
    assert pd.api.types.is_datetime64_any_dtype(result["date"])


def test_summarize_counties() -> None:
    """
    Tests the number of records for each county.
    """
    test_data = pd.DataFrame({
        "county": [
            "King County",
            "King County",
            "Pierce County",
            "Multnomah County"
        ]
    })

    result = summarize_counties(test_data)

    assert result["King County"] == 2
    assert result["Pierce County"] == 1
    assert result["Multnomah County"] == 1


def test_summarize_missing_data() -> None:
    """
    Tests the number of missing values in each column.
    """
    test_data = pd.DataFrame({
        "New cases": [10, None, 30],
        "New deaths": [None, 2, 3]
    })

    result = summarize_missing_data(test_data)

    assert result["New cases"] == 1
    assert result["New deaths"] == 1


def test_summarize_quantitative() -> None:
    """
    Tests summary statistics for quantitative columns.
    """
    test_data = pd.DataFrame({
        "New cases": [10, 20, 30],
        "New deaths": [1, 2, 3]
    })

    result = summarize_quantitative(test_data)

    assert result.loc["count", "New cases"] == 3
    assert result.loc["mean", "New cases"] == 20
    assert result.loc["min", "New cases"] == 10
    assert result.loc["max", "New cases"] == 30

    assert result.loc["count", "New deaths"] == 3
    assert result.loc["mean", "New deaths"] == 2
    assert result.loc["min", "New deaths"] == 1
    assert result.loc["max", "New deaths"] == 3


def main() -> None:
    test_clean_data()
    test_summarize_counties()
    test_summarize_missing_data()
    test_summarize_quantitative()
    print("All tests passed!")


if __name__ == "__main__":
    main()
