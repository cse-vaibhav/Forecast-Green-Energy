# Forecast Green Energy

My submission for [Analytics Vidhya JOB-A-THON November 2022](https://datahack.analyticsvidhya.com/contest/job-a-thon-november-2022)

# Run Code

- Install dependencies in the `pyproject.toml` file using poetry

```bash
$ poetry install
$ poetry shell
$ jupyter notebook Notebook.ipynb
```

# Approach

- Split the `datetime` feature into

  - year
  - month
  - day
  - hour
  - minutes
  - second

- Used the Decision Tree Regression for the dataset because it didn't get affected by outliers
