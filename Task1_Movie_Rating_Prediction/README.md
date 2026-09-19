# Movie Rating Prediction

**Status:** In Progress

Selected project from the CodSoft Data Science internship (listed as Task 2
in the CodSoft task document).

## Objective

Build a regression model that predicts the IMDb rating of Indian movies from
features such as genre, director and actors.

## Dataset

- **File:** `dataset/IMDb Movies India.csv`
- **Size:** 15,509 rows × 10 columns
- **Columns:** `Name`, `Year`, `Duration`, `Genre`, `Rating`, `Votes`,
  `Director`, `Actor 1`, `Actor 2`, `Actor 3`
- **Encoding:** the file is not valid UTF-8; load it with `encoding="latin-1"`.

```python
import pandas as pd

df = pd.read_csv("dataset/IMDb Movies India.csv", encoding="latin-1")
```

The raw CSV is kept unmodified. Any cleaned or derived data will be produced
by code rather than by editing the original file.

## Project Structure

```
Task1_Movie_Rating_Prediction/
├── dataset/          # Raw dataset (unmodified)
├── notebooks/        # Jupyter notebooks for exploration and modelling
├── src/              # Reusable Python modules
├── visualizations/   # Saved plots
├── models/           # Saved trained models
├── requirements.txt
└── README.md
```

## Setup

From the repository root:

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r Task1_Movie_Rating_Prediction/requirements.txt
```

## Results

Not available yet. The project is in progress.
