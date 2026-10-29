# Movie Industry Analysis and Prediction

An exploratory machine-learning study of the factors associated with movie ratings and commercial performance. The notebook combines data cleaning, visualization, classical models, time-series summaries, and two neural-network baselines.

> [!NOTE]
> This is an academic analysis, not a production forecasting system. Reported metrics describe held-out samples from the included dataset and should not be generalized to future releases without additional temporal validation.

## Research questions

- How are critic scores, audience scores, duration, release date, and financial variables related?
- Which features are associated with commercial success?
- Can baseline models classify rating tiers or estimate IMDb scores?
- How did monthly release and box-office patterns change over time?

## Analysis workflow

1. Load the raw CSV directly from `IMDb Movies.zip`.
2. Normalize identifiers, dates, numeric columns, and list-valued text fields.
3. Retain financial values explicitly denominated in US dollars.
4. Explore leading movies, actors, genres, and numeric correlations.
5. Fit linear regression, logistic regression, and linear SVC baselines.
6. Summarize monthly release, rating, and box-office trends.
7. Train neural networks for success classification and score regression.

The commercial-success target is defined as profit above USD 50 million and a profit margin of at least `1.0`. Rating classes are defined in the notebook before the SVC model.

## Repository data

| File | Purpose |
| --- | --- |
| `IMDb Movies.zip` | Raw source archive containing `IMDb Movies.csv` |
| `IMDb_Movies_Updated.csv` | Processed checkpoint from the original analysis |
| `movie-analysis.ipynb` | Reproducible cleaning, analysis, and modeling workflow |

> [!IMPORTANT]
> The original repository did not document the external dataset source or license. Confirm the provenance and redistribution terms before reusing or publishing the data outside this academic project.

The notebook reads the ZIP archive directly; manual extraction is not required. The processed checkpoint is provided for inspection, but rerunning the cleaning section is the authoritative way to reproduce the current transformation logic.

## Environment setup

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start JupyterLab:

```bash
jupyter lab movie-analysis.ipynb
```

Run the cells from top to bottom. TensorFlow training is the most resource-intensive stage and may take several minutes, depending on the machine.

> [!TIP]
> The notebook sets NumPy and TensorFlow seeds for repeatable splits and initialization. Hardware, library versions, and parallel numerical operations can still produce small differences.

## Models and metrics

| Task | Model | Main metrics |
| --- | --- | --- |
| IMDb score estimation | Linear regression | RMSE, R² |
| Commercial-success classification | Logistic regression | Precision, recall, F1, ROC AUC |
| Rating-tier classification | Linear SVC | Precision, recall, F1, confusion matrix |
| Commercial-success classification | Dense neural network | Accuracy, ROC AUC |
| IMDb score estimation | Dense neural network | MSE, RMSE, MAE |

## Validation

Run the lightweight repository validation without installing the scientific stack:

```bash
python3 scripts/validate_notebook.py
```

This checks the notebook schema, Python syntax, cleared outputs, required section headings, and both data-file schemas. It does not train the models.

## Project structure

```text
.
├── IMDb Movies.zip
├── IMDb_Movies_Updated.csv
├── movie-analysis.ipynb
├── requirements.txt
└── scripts/
    └── validate_notebook.py
```

## Methodological limitations

- The dataset may contain selection bias and incomplete financial observations.
- USD-only filtering reduces currency ambiguity but also removes many international financial records.
- Random train/test splits do not simulate prediction of genuinely future releases.
- Preprocessing in the exploratory classification sections is fitted before the split, so their reported metrics may be mildly optimistic.
- Correlations and the 2019–2020 comparison are descriptive, not causal.
