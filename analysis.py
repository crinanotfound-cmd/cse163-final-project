"""
EDA Project
Caroline Shi
CSE 163
Summer 2026

This file contains functions for data cleaning, summarization,
and 2 visulizations creating.
"""
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = (
    "Weekly_United_States_COVID-19_Cases_and_Deaths_"
    "by_County_-_ARCHIVED_20260719.csv"
)


def load_data(filename: str) -> pd.DataFrame:
    """
    Reads and returns the COVID-19 dataset.
    """
    return pd.read_csv(filename, low_memory=False)


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the data to Washington counties and converts relevant
    columns to appropriate data types.
    """
    wa = data[data["state"] == "WA"].copy()
    wa = wa[
        wa["county"] != "Unallocated Washington"
    ].copy()

    wa["date"] = pd.to_datetime(wa["date"])

    wa["New deaths"] = pd.to_numeric(
        wa["New deaths"],
        errors="coerce"
    )

    wa["New cases"] = pd.to_numeric(
        wa["New cases"].str.replace(",", "", regex=False),
        errors="coerce"
    )

    return wa


def summarize_dataset(data: pd.DataFrame) -> None:
    """
    Prints basic information about the Washington dataset.
    """
    print("Washington data shape:")
    print(data.shape)
    print()

    print("Date range:")
    print(data["date"].min())
    print(data["date"].max())
    print()

    print("Counties:")
    print(sorted(data["county"].unique()))
    print()

    print("Number of counties:")
    print(data["county"].nunique())


def summarize_counties(data: pd.DataFrame) -> pd.Series:
    """
    Returns the number of records for each county.
    """
    return data["county"].value_counts().sort_index()


def summarize_missing_data(data: pd.DataFrame) -> pd.Series:
    """
    Returns the number of missing values in each column.
    """
    return data.isna().sum()


def summarize_quantitative(data: pd.DataFrame) -> pd.DataFrame:
    """
    Returns summary statistics for weekly new cases and deaths.
    """
    return data[["New cases", "New deaths"]].describe()


def plot_cases_and_deaths(data: pd.DataFrame) -> None:
    """
    Saves a figure with two subplots showing weekly new cases
    and weekly new deaths in Washington State.
    """
    weekly = data.groupby("date")[["New cases", "New deaths"]].sum()
    fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(12, 8))

    weekly["New cases"].plot(ax=ax1)
    ax1.set_title("Weekly New COVID-19 Cases in Washington State")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("New Cases")

    weekly["New deaths"].plot(ax=ax2)
    ax2.set_title("Weekly New COVID-19 Deaths in Washington State")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("New Deaths")

    plt.tight_layout()
    plt.savefig("weekly_cases_and_deaths.png")
    plt.close()


def plot_random_counties(data: pd.DataFrame) -> None:
    """
    Saves six lineplots comparing comparing weekly cases and deaths
    for three randomly selected Washington counties.
    """
    counties = data["county"].drop_duplicates()
    selected_counties = counties.sample(n=3, random_state=1)
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(20, 18)
    )
    county1 = selected_counties.iloc[0]
    county2 = selected_counties.iloc[1]
    county3 = selected_counties.iloc[2]

    data1 = data[data["county"] == county1]
    data2 = data[data["county"] == county2]
    data3 = data[data["county"] == county3]

    data1.plot(
        ax=axes[0, 0],
        x="date",
        y="New cases",
        legend=True
    )
    peak_cases1 = data1["New cases"].max()
    axes[0, 0].set_title(
        f"{county1}: Weekly New Cases, peak =  {peak_cases1:.0f}"
    )

    data1.plot(
        ax=axes[0, 1],
        x="date",
        y="New deaths",
        legend=True
    )
    peak_deaths1 = data1["New deaths"].max()
    axes[0, 1].set_title(
        f"{county1}: Weekly New Deaths, peak = {peak_deaths1:.0f}"
    )

    data2.plot(
        ax=axes[1, 0],
        x="date",
        y="New cases",
        legend=True
    )
    peak_cases2 = data2["New cases"].max()
    axes[1, 0].set_title(
        f"{county2}: Weekly New Cases, peak = {peak_cases2:.0f}"
    )

    data2.plot(
        ax=axes[1, 1],
        x="date",
        y="New deaths",
        legend=True
    )
    peak_deaths2 = data2["New deaths"].max()
    axes[1, 1].set_title(
        f"{county2}: Weekly New Deaths, peak = {peak_deaths2:.0f}"
    )

    data3.plot(
        ax=axes[2, 0],
        x="date",
        y="New cases",
        legend=True
    )
    peak_cases3 = data3["New cases"].max()
    axes[2, 0].set_title(
        f"{county3}: Weekly New Cases, peak = {peak_cases3:.0f}"
    )

    data3.plot(
        ax=axes[2, 1],
        x="date",
        y="New deaths",
        legend=True
    )
    peak_deaths3 = data3["New deaths"].max()
    axes[2, 1].set_title(
        f"{county3}: Weekly New Deaths, peak = {peak_deaths3:.0f}"
    )

    plt.tight_layout()
    plt.savefig("Deaths and Cases.png")
    plt.close()


def main() -> None:
    df = load_data(DATA_PATH)
    wa = clean_data(df)
    summarize_dataset(wa)

    print()
    print("County counts:")
    print(summarize_counties(wa))
    print()
    print("Missing values:")
    print(summarize_missing_data(wa))
    print()
    print("Quantitative summary:")
    print(summarize_quantitative(wa))

    plot_cases_and_deaths(wa)
    plot_random_counties(wa)


if __name__ == "__main__":
    main()
