import re

NEGATION = re.compile(r"n't\b")
NON_LETTERS = re.compile(r"[^a-z\s]")
SPACES = re.compile(r"\s+")


def clean_text(text):
    """Normalise one review. Shared by the notebook, the saved pipeline and the app."""
    text = str(text).lower()
    # "don't" -> "do not": keep negation as a real token, it flips sentiment
    text = NEGATION.sub(" not", text)
    # drop digits and punctuation: they add noise, not sentiment, in this data
    text = NON_LETTERS.sub(" ", text)
    return SPACES.sub(" ", text).strip()
