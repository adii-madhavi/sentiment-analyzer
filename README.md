# AI-Powered Sentiment Analysis

Classifies customer reviews as Positive, Negative or Neutral using a scikit-learn pipeline
(TF-IDF + Multinomial Naive Bayes), served through a Streamlit app and a browser app on GitHub Pages.

**Live app:** https://adii-madhavi.github.io/sentiment-analyzer/

## Project structure

```
sentiment-analyzer/
├── web/                              # browser app (GitHub Pages): index.html, model.js, model.json
├── scripts/export_web_model.py       # exports the saved pipeline to web/model.json
├── tests/check_web_model.mjs         # checks the browser model matches the Python pipeline
├── .github/workflows/                # pages.yml (deploy web/), release.yml (release on v* tags)
├── data/reviews.csv                  # 10,000 reviews (provided dataset)
├── notebooks/sentiment_analysis.ipynb  # cleaning, EDA, features, training, evaluation, saving
├── src/text_utils.py                 # clean_text(), shared by notebook and app
├── models/sentiment_pipeline.pkl     # saved trained pipeline
├── reports/
│   ├── evaluation_report.md          # model comparison and justification
│   ├── metrics.json                  # test metrics
│   └── figures/                      # EDA, confusion matrix, ROC plots
├── app.py                            # Streamlit app
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run the notebook (optional, retrains the model)

```bash
cd notebooks
jupyter nbconvert --to notebook --execute --inplace sentiment_analysis.ipynb
```

## Run the app

```bash
streamlit run app.py
```

Enter a review and click **Analyze** to see the predicted sentiment and class probabilities.
The app loads `models/sentiment_pipeline.pkl`, so it does not retrain at start-up.

## Browser app (GitHub Pages)

`web/` is a static version of the same model. `scripts/export_web_model.py` writes the TF-IDF vocabulary, IDF weights
and Naive Bayes parameters to `web/model.json`, and `web/model.js` reproduces the pipeline's predictions in JavaScript
(checked against Python to within 1e-9 by `tests/check_web_model.mjs`). Nothing leaves the browser.

After retraining, re-export and check:

```bash
python scripts/export_web_model.py
node tests/check_web_model.mjs
```

Pushing changes under `web/` to `main` redeploys the site through `.github/workflows/pages.yml`.
To preview locally: `python -m http.server --directory web`, then open http://localhost:8000.

## Releases

Pushing a tag such as `v1.0.0` runs `.github/workflows/release.yml`, which creates a GitHub Release with the
trained pipeline, a zip of the web app, and the evaluation report attached.

## Workflow

1. **Data understanding and cleaning:** check missing values, duplicates and label consistency.
2. **EDA:** class balance, review length, category and source breakdowns, top words per class.
3. **Preprocessing:** lowercase, expand negations ("n't" to "not"), remove digits and punctuation. Stopwords are kept.
4. **Features:** Bag of Words and TF-IDF, unigrams and uni+bigrams.
5. **Models:** Logistic Regression and Multinomial Naive Bayes.
6. **Evaluation:** accuracy, precision, recall, F1, confusion matrix, ROC AUC on a held-out test set.
7. **Validation and tuning:** 5-fold grouped cross-validation with GridSearchCV.
8. **Saving:** the full pipeline (vectorizer + classifier) is serialised with joblib.

Splits are grouped by review text, since the dataset contains only 300 unique texts.

## Results

Held-out test: accuracy 1.0, macro F1 1.0. These scores are at the ceiling because the dataset is highly
repetitive. See [reports/evaluation_report.md](reports/evaluation_report.md) for details and caveats.

## Requirements

See `requirements.txt`. Tested with Python 3.14, scikit-learn 1.9, pandas 3.0, Streamlit 1.66.
