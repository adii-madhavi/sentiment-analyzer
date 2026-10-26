# Evaluation Report: Sentiment Classification

Dataset: `data/reviews.csv` (10,000 reviews; labels Positive, Negative, Neutral). No other data was used.

## 1. Data checks

- No missing values, no duplicate rows, no duplicate `review_id`, no empty reviews.
- Classes are balanced: Positive 3,334; Negative 3,333; Neutral 3,333.
- **Only 300 unique review texts.** Each text appears 21 to 52 times (median 33) and always has the same label.

Because of this, a plain random split would place copies of the same text in both train and test. The project uses a
`StratifiedGroupKFold` split grouped by review text: 8,000 train rows and 2,000 test rows, with 0 texts shared.

## 2. Features and preprocessing

Cleaning (`src/text_utils.py`): lowercase, expand "n't" to "not", remove digits and punctuation.
Stopwords are kept (negations such as "not" carry sentiment). No stemming or lemmatisation.

Features compared: Bag of Words (`CountVectorizer`) and TF-IDF (`TfidfVectorizer`), each with unigrams or uni+bigrams.

## 3. Model comparison (5-fold grouped cross-validation, training portion)

Best mean macro F1 for each combination (search over C for Logistic Regression and alpha for Naive Bayes, and both feature sets):

| Classifier          | Features | Best CV macro F1 |
|---------------------|----------|------------------|
| Logistic Regression | BoW      | 0.9912           |
| Logistic Regression | TF-IDF   | 0.9922           |
| Multinomial NB      | BoW      | 0.9959           |
| Multinomial NB      | TF-IDF   | 0.9959           |

Multinomial NB tied at the top across all settings, so the choice of TF-IDF, uni+bigrams, alpha 0.1 is a tie-break,
not a clear winner. Naive Bayes was selected because it has the best CV score and gives probabilities for the app.

## 4. Final model and held-out test results

Selected pipeline: `TfidfVectorizer(ngram_range=(1, 2))` + `MultinomialNB(alpha=0.1)`, saved to `models/sentiment_pipeline.pkl`.

| Class    | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Negative | 1.0000    | 1.0000 | 1.0000   | 666     |
| Neutral  | 1.0000    | 1.0000 | 1.0000   | 668     |
| Positive | 1.0000    | 1.0000 | 1.0000   | 666     |

- Accuracy: 1.0000; macro F1: 1.0000
- Confusion matrix: no off-diagonal errors (`reports/figures/confusion_matrix.png`)
- One-vs-rest ROC AUC: 1.0 for all classes (`reports/figures/roc_curves.png`)
- A plain random split with TF-IDF and Logistic Regression also scores 1.0 accuracy, so grouping did not change the number here.

## 5. Interpreting these results

The scores are at the ceiling. The reason is likely the dataset itself: 300 unique texts, built from repeated phrase
patterns (one label per text, with Neutral reviews written as factual statements). The held-out test texts are new
sentences, but they are made from the same phrase pool, so the model only needs to recognise those phrases.
**These numbers are not evidence of how the model will perform on real customer reviews.** Testing on real
reviews, with more varied wording, is the important next step. (Only the provided dataset was used, as requested.)

Confidence scores are also saturated: Naive Bayes returns probabilities close to 0 or 1 on most inputs. They are
useful for ranking but overstate certainty. Logistic Regression gives softer probabilities and would be the better
choice for calibrated confidence on messier text.

## 6. Limitations

- Results come from one dataset with heavy text repetition; no external validation.
- Tuning was limited to a small grid (about 24 configurations); the best configurations tie.
- The app shows probabilities from the Naive Bayes model, which are not well calibrated (see above).
- Reproducibility: `RANDOM_STATE = 42` throughout.
