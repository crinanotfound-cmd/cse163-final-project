# COVID-19 Trends and County Differences in Washington State

## Required Libraries and Data

This project requires the following Python libraries:

- pandas
- matplotlib
- scipy
- scikit-learn

The project uses：
`Weekly_United_States_COVID-19_Cases_and_Deaths_by_County_-_ARCHIVED_20260719.csv`

The dataset is included in this repository. It should remain in the
same location relative to the Python files so that the file path in
`analysis.py` works correctly.


## Files

- `analysis.py`: Loads and cleans the COVID-19 dataset and performs
  exploratory data analysis and creates two visualizations.

- `result.py`: Performs the main analysis for the all research questions,
  including statewide trends, county peak timing and magnitude,
  same-week and lagged case-death correlations and Random Forest
  death prediction.

- `testing_for_analysis.py`: Contains tests for functions in
  `analysis.py`.

- `testing_for_result.py`: Contains tests for functions in
  `result.py`.


## How to Run the Project

1. Download or clone this repository.

2. Make sure to install the required Python libraries.

3. Make sure the COVID-19 dataset is in the location expected by
   `DATA_PATH` in `analysis.py`.

4. Run the exploratory analysis:

   `python analysis.py`

5. Run the main analysis:

   `python result.py`

6. Run the tests:

   `python testing_for_analysis.py`

   `python testing_for_result.py`


## Generated Files

Running `analysis.py` generates:

- `weekly_cases_and_deaths.png`
- `Deaths and Cases.png`

Running `result.py` generates:

- `county_prominent_peaks.csv`
- `major_peak_timing_comparison.csv`
- `major_case_peak_magnitudes.csv`
- `major_death_peak_magnitudes.csv`
- `major_peak_timing.png`