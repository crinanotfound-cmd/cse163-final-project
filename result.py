"""
Project Part 3
Caroline Shi
CSE 163
Summer 2026

This file analyzes weekly COVID-19 case
and death trends across Washington counties.
It compares statewide trends, county peak timing
and magnitude, case-death correlations,
lagged relationships, and Random Forest death predictions.
"""
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from analysis import load_data, clean_data, DATA_PATH
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def statewide_trend(data: pd.DataFrame) -> None:
    """
    Prints statewide weekly case and death changes
    and yearly totals and averages.
    """
    weekly = data.groupby('date')[['New cases', 'New deaths']].sum()
    weekly['Case change'] = weekly['New cases'].diff()
    weekly['Death change'] = weekly['New deaths'].diff()

    max_weekly_change_case = weekly['Case change'].max()
    max_weekly_change_case_week = weekly['Case change'].idxmax()

    print(
        f'The week of max weekly case increase: {max_weekly_change_case_week}'
    )
    print(
        f'The max weekly case increase value: {max_weekly_change_case}'
    )
    print()

    max_weekly_change_death = weekly['Death change'].max()
    max_weekly_change_death_week = weekly['Death change'].idxmax()

    print(
        "The week of max weekly death increase: "
        f"{max_weekly_change_death_week}"
    )
    print(
        "The max weekly death increase value: "
        f"{max_weekly_change_death}"
    )
    print()

    weekly['year'] = weekly.index.year
    yearly_average = weekly.groupby('year')[['New cases', 'New deaths']].mean()
    yearly_total = weekly.groupby('year')[['New cases', 'New deaths']].sum()

    print(f'The yearly total of weekly cases and deaths: {yearly_total}')
    print(f'The yearly average of weekly cases and deaths: {yearly_average}')


def county_case_peaks(data: pd.DataFrame, county: str) -> pd.DataFrame:
    """
    Returns the three most prominent weekly case peaks for the
    specified county, ordered by date. If fewer than three peaks
    are found, returns all available peaks.
    """
    county_data = data[data["county"] == county].sort_values("date")
    peaks, properties = find_peaks(
        county_data["New cases"],
        prominence=0
    )
    peak_rows = county_data.iloc[peaks].copy()
    peak_rows["Prominence"] = properties["prominences"]
    major_peaks = peak_rows.sort_values(
        "Prominence",
        ascending=False
    ).head(3)
    major_peaks = major_peaks.sort_values("date")
    return major_peaks[["county", "date", "New cases"]]


def county_death_peaks(data: pd.DataFrame, county: str) -> pd.DataFrame:
    """
    Returns the three most prominent weekly death peaks for the
    specified county, ordered by date. If fewer than three peaks
    are found, returns all available peaks.
    """
    county_data = data[data["county"] == county].sort_values("date")
    peaks, properties = find_peaks(
        county_data["New deaths"],
        prominence=0
    )
    peak_rows = county_data.iloc[peaks].copy()
    peak_rows["Prominence"] = properties["prominences"]
    major_peaks = peak_rows.sort_values(
        "Prominence",
        ascending=False
    ).head(3)
    major_peaks = major_peaks.sort_values("date")
    return major_peaks[["county", "date", "New deaths"]]


def collect_peaks(data: pd.DataFrame) -> pd.DataFrame:
    """
    Collects prominent case and death peaks for all counties,
    saves them to a CSV file, and returns the combined data.
    """
    all_peaks = []

    for county in data["county"].unique():
        case_peaks = county_case_peaks(data, county)
        death_peaks = county_death_peaks(data, county)

        case_peaks = case_peaks.rename(
            columns={"New cases": "Magnitude"}
        )
        case_peaks["Type"] = "cases"

        death_peaks = death_peaks.rename(
            columns={"New deaths": "Magnitude"}
        )
        death_peaks["Type"] = "deaths"

        all_peaks.append(case_peaks)
        all_peaks.append(death_peaks)

    result = pd.concat(all_peaks, ignore_index=True)

    result.to_csv("county_prominent_peaks.csv", index=False)

    return result


def major_peaks_timing_comparison(
        data: pd.DataFrame) -> pd.DataFrame:
    """
    Compares the timing of maximum weekly case and death peaks
    by counting how many counties reached each peak in each month.
    If multiple weeks share the same maximum value, the first
    occurrence will be used.
    """
    major_peaks = []
    for county in data["county"].unique():
        county_data = data[data["county"] == county]

        case_peak_index = county_data["New cases"].idxmax()
        death_peak_index = county_data["New deaths"].idxmax()

        major_peaks.append({
            "county": county,
            "date": county_data.loc[case_peak_index, "date"],
            "Type": "cases"
        })

        major_peaks.append({
            "county": county,
            "date": county_data.loc[death_peak_index, "date"],
            "Type": "deaths"
        })

    major_peaks = pd.DataFrame(major_peaks)

    major_peaks["Peak month"] = (
        major_peaks["date"].dt.to_period("M")
    )

    timing = major_peaks.groupby(
        ["Type", "Peak month"]
    ).size().reset_index(
        name="Number of counties"
    )
    timing.to_csv(
        "major_peak_timing_comparison.csv",
        index=False
    )
    return timing


def plot_major_peak_timing(timing: pd.DataFrame) -> None:
    """
    Creates and saves a bar plot showing the months when counties
    reached their maximum weekly case and death peaks.
    """
    plot_data = timing.pivot(
        index="Peak month",
        columns="Type",
        values="Number of counties"
    ).fillna(0)
    ax = plot_data.plot(
        kind="bar",
        figsize=(12, 7)
    )
    plt.title(
        "Timing of Major COVID-19 Peaks Across Washington Counties"
    )
    plt.xlabel("Peak Month")
    plt.ylabel("Number of Counties")
    plt.legend(title="Peak Type")

    for bars in ax.containers:
        ax.bar_label(bars)

    fig = ax.get_figure()
    fig.text(
        0.5,
        0.01,
        (
            "Figure: Number of Washington counties whose maximum weekly "
            "case or death count occurred in each month."
        ),
        ha="center"
    )

    plt.xticks(rotation=45)

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    plt.savefig(
        "major_peak_timing.png",
        bbox_inches="tight"
    )
    plt.close()


def major_peak_magnitude_comparison(data: pd.DataFrame) -> None:
    """
    Compares maximum weekly case and death across counties
    and saves the five largest and five smallest values.
    If counties tie for an extreme value,
    the printed county name reflects one of the tied counties.
    """
    case_magnitudes = []
    death_magnitudes = []

    for county in data["county"].unique():
        county_data = data[data["county"] == county]

        case_magnitudes.append({
            "county": county,
            "Magnitude": county_data["New cases"].max()
        })

        death_magnitudes.append({
            "county": county,
            "Magnitude": county_data["New deaths"].max()
        })

    case_magnitudes = pd.DataFrame(case_magnitudes)
    death_magnitudes = pd.DataFrame(death_magnitudes)

    case_magnitudes = case_magnitudes.sort_values(
        "Magnitude",
        ascending=False,
        ignore_index=True
    )

    death_magnitudes = death_magnitudes.sort_values(
        "Magnitude",
        ascending=False,
        ignore_index=True
    )

    case_comparison = pd.concat([
        case_magnitudes.head(5),
        case_magnitudes.tail(5)
    ])

    death_comparison = pd.concat([
        death_magnitudes.head(5),
        death_magnitudes.tail(5)
    ])

    case_difference = (
        case_magnitudes["Magnitude"].max()
        - case_magnitudes["Magnitude"].min()
    )

    death_difference = (
        death_magnitudes["Magnitude"].max()
        - death_magnitudes["Magnitude"].min()
    )

    largest_case_county = case_magnitudes.iloc[0]["county"]
    smallest_case_county = case_magnitudes.iloc[-1]["county"]

    largest_death_county = death_magnitudes.iloc[0]["county"]
    smallest_death_county = death_magnitudes.iloc[-1]["county"]

    print("Top 5 and bottom 5 major case peaks:")
    print(case_comparison)
    print()
    print("Top 5 and bottom 5 major death peaks:")
    print(death_comparison)

    print(f"Largest case peak county: {largest_case_county}")
    print(f"Smallest case peak county: {smallest_case_county}")
    print(f"Difference between largest and smallest case peak: "
          f"{case_difference}")
    print()
    print(f"Largest death peak county: {largest_death_county}")
    print(f"Smallest death peak county: {smallest_death_county}")
    print(f"Difference between largest and smallest death peak: "
          f"{death_difference}")

    case_comparison.to_csv(
        "major_case_peak_magnitudes.csv",
        index=False
    )

    death_comparison.to_csv(
        "major_death_peak_magnitudes.csv",
        index=False
    )


def county_case_death_correlation(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates same-week Pearson correlations between weekly cases
    and deaths for each county and returns the ordered correlations,
    from highest to lowest.

    The printed comparison includes the five highest and five lowest
    correlations. If there are fewer than five counties, the printed
    groups may contain repeated counties.
    """
    correlations = []

    for county in data["county"].unique():
        county_data = data[data["county"] == county]

        correlation, _ = pearsonr(
            county_data["New cases"],
            county_data["New deaths"]
        )

        correlations.append({
            "county": county,
            "Correlation": correlation
        })

    correlations = pd.DataFrame(correlations)

    correlations = correlations.sort_values(
        "Correlation",
        ascending=False,
        ignore_index=True
    )

    correlation_comparison = pd.concat([
        correlations.head(5),
        correlations.tail(5)
    ])

    print("Top 5 and bottom 5 same week case-death correlations:")
    print(correlation_comparison)

    return correlations


def correlation_summary(correlations: pd.DataFrame) -> None:
    """
    Prints the maximum, minimum, mean, and median correlations
    and the counties with the highest and lowest correlations.

    The input correlations are expected to be
    ordered from highest to lowest correlation.

    If some counties tie for an extreme correlation,
    one of the tied counties is reported.
    """
    max_correlation = correlations["Correlation"].max()
    min_correlation = correlations["Correlation"].min()
    mean_correlation = correlations["Correlation"].mean()
    median_correlation = correlations["Correlation"].median()

    highest_county = correlations.iloc[0]["county"]
    lowest_county = correlations.iloc[-1]["county"]

    print(f"Maximum correlation: {max_correlation}")
    print()
    print(f"Minimum correlation: {min_correlation}")
    print()
    print(f"Mean correlation: {mean_correlation}")
    print()
    print(f"Median correlation: {median_correlation}")
    print()

    print(f"County with highest same week correlation: {highest_county}")
    print()
    print(f"County with lowest same week correlation: {lowest_county}")


def lagged_correlation(
        data: pd.DataFrame, county: str) -> pd.DataFrame:
    """
    Calculates correlations between weekly deaths and case counts
    from the same week and the previous three weeks for a county.
    Weeks that do not have enough prior case data for a lag are excluded
    from that correlation.
    """
    county_data = data[
        data["county"] == county
    ].sort_values("date").copy()

    results = []

    for lag in range(4):
        county_data["Lagged cases"] = (
            county_data["New cases"].shift(lag)
        )

        lag_data = county_data[
            ["Lagged cases", "New deaths"]
        ].dropna()

        correlation, _ = pearsonr(
            lag_data["Lagged cases"],
            lag_data["New deaths"]
        )

        results.append({
            "county": county,
            "Lag": lag,
            "Correlation": correlation
        })

    return pd.DataFrame(results)


def random_forest_death_prediction(
        data: pd.DataFrame, county: str) -> None:
    """
    Trains a Random Forest model to predict weekly deaths from
    current and recent case counts. Prints prediction error,
    model fit, and feature importance for the specified county.
    Rows without complete lagged case data are excluded.
    """
    county_data = data[
        data["county"] == county
    ].sort_values("date").copy()

    county_data["Cases lag 1"] = (
        county_data["New cases"].shift(1)
    )
    county_data["Cases lag 2"] = (
        county_data["New cases"].shift(2)
    )
    county_data["Cases lag 3"] = (
        county_data["New cases"].shift(3)
    )

    county_data = county_data.dropna(
        subset=[
            "New cases",
            "Cases lag 1",
            "Cases lag 2",
            "Cases lag 3",
            "New deaths"
        ]
    )

    features = county_data[[
        "New cases",
        "Cases lag 1",
        "Cases lag 2",
        "Cases lag 3"
    ]]

    labels = county_data["New deaths"]

    features_train, features_test, labels_train, labels_test = (
        train_test_split(
            features,
            labels,
            test_size=0.2,
            shuffle=False
        )
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=1
    )

    model.fit(features_train, labels_train)
    predictions = model.predict(features_test)

    mae = mean_absolute_error(
        labels_test,
        predictions
    )
    r2 = r2_score(
        labels_test,
        predictions
    )

    print(f"Random Forest results for {county}:")
    print(f"Mean Absolute Error: {mae}")
    print(f"R^2: {r2}")
    print()

    feature_importance = pd.DataFrame({
        "Feature": features.columns,
        "Importance": model.feature_importances_
    })
    feature_importance = feature_importance.sort_values(
        "Importance",
        ascending=False,
        ignore_index=True
    )
    print("Feature importance:")
    print(feature_importance)
    print()


def main() -> None:
    # data preparation
    data = load_data(DATA_PATH)
    wa = clean_data(data)
    # Q1
    statewide_trend(wa)
    print()
    # Q2
    collect_peaks(wa)
    print()
    timing = major_peaks_timing_comparison(wa)
    print()
    plot_major_peak_timing(timing)
    print()
    major_peak_magnitude_comparison(wa)
    print()
    # Q3
    correlations = county_case_death_correlation(wa)
    correlation_summary(correlations)
    print()

    clark_lag = lagged_correlation(wa, "Clark County")
    king_lag = lagged_correlation(wa, "King County")
    wahkiakum_lag = lagged_correlation(wa, "Wahkiakum County")

    print(clark_lag)
    print()
    print(king_lag)
    print()
    print(wahkiakum_lag)
    print()

    print("Random Forest Prediction Results")
    print()

    random_forest_death_prediction(
        wa, "Clark County"
    )

    random_forest_death_prediction(
        wa, "King County"
    )

    random_forest_death_prediction(
        wa, "Wahkiakum County"
    )


if __name__ == "__main__":
    main()
