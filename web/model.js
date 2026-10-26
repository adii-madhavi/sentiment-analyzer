// Browser port of the saved scikit-learn pipeline:
// clean_text -> TfidfVectorizer(ngram_range=(1, 2), l2 norm) -> MultinomialNB.predict_proba
// Parameters come from model.json (scripts/export_web_model.py). Must stay in step with src/text_utils.py.

export function cleanText(text) {
  return String(text)
    .toLowerCase()
    .replace(/n't\b/g, " not")
    .replace(/[^a-z\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

// After cleaning only a-z and spaces remain, so sklearn's token_pattern \b\w\w+\b is "words of 2+ letters"
function ngrams(text, [minN, maxN]) {
  const tokens = cleanText(text).split(" ").filter((t) => t.length >= 2);
  const out = [];
  for (let n = minN; n <= maxN; n++) {
    for (let i = 0; i + n <= tokens.length; i++) out.push(tokens.slice(i, i + n).join(" "));
  }
  return out;
}

export function predictProba(model, text) {
  const counts = new Map();
  for (const g of ngrams(text, model.ngram_range)) {
    const idx = model.vocabulary[g];
    if (idx !== undefined) counts.set(idx, (counts.get(idx) || 0) + 1);
  }
  const weights = new Map();
  let norm = 0;
  for (const [idx, c] of counts) {
    const w = c * model.idf[idx];
    weights.set(idx, w);
    norm += w * w;
  }
  norm = Math.sqrt(norm);

  const jll = model.classes.map((_, k) => {
    let s = model.class_log_prior[k];
    if (norm > 0) for (const [idx, w] of weights) s += (w / norm) * model.feature_log_prob[k][idx];
    return s;
  });
  const max = Math.max(...jll);
  const exps = jll.map((v) => Math.exp(v - max));
  const total = exps.reduce((a, b) => a + b, 0);
  return { proba: exps.map((e) => e / total), knownTerms: counts.size };
}
