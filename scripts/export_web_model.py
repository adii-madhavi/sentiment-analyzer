"""Export the saved pipeline to JSON so the browser app (web/) can run it without Python.

Run from the project root: python scripts/export_web_model.py
Also writes tests/web_expected.json, used by tests/check_web_model.mjs to confirm the JS port matches.
"""
import json
import os
import sys

import joblib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

pipe = joblib.load(os.path.join(ROOT, "models", "sentiment_pipeline.pkl"))
vec, clf = pipe.named_steps["vec"], pipe.named_steps["clf"]

# The JS port implements exactly this configuration; fail loudly if the saved model changes shape
assert type(vec).__name__ == "TfidfVectorizer" and type(clf).__name__ == "MultinomialNB"
assert vec.token_pattern == r"(?u)\b\w\w+\b" and vec.norm == "l2" and not vec.sublinear_tf and vec.use_idf

model = {
    "classes": clf.classes_.tolist(),
    "ngram_range": list(vec.ngram_range),
    "vocabulary": {term: int(i) for term, i in vec.vocabulary_.items()},
    "idf": vec.idf_.tolist(),
    "class_log_prior": clf.class_log_prior_.tolist(),
    "feature_log_prob": clf.feature_log_prob_.tolist(),
}
with open(os.path.join(ROOT, "web", "model.json"), "w") as f:
    json.dump(model, f, separators=(",", ":"))

texts = pd.read_csv(os.path.join(ROOT, "data", "reviews.csv"))["review_text"].unique().tolist()
texts += ["The product doesn't work, I won't buy again!", "", "!!! 123 ???", "Great GREAT great.",
          "I would not recommend it. Customer support was slow.", "Ünïcode café — naïve review"]
os.makedirs(os.path.join(ROOT, "tests"), exist_ok=True)
with open(os.path.join(ROOT, "tests", "web_expected.json"), "w") as f:
    json.dump({"texts": texts, "proba": pipe.predict_proba(texts).tolist()}, f)

print(f"vocabulary: {len(model['vocabulary'])} terms; expected cases: {len(texts)}")
