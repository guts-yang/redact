import nltk

_punkt_tokenizer = None


def split_sentences(text):
    """Split text into sentences using NLTK PunktSentenceTokenizer."""
    global _punkt_tokenizer
    if _punkt_tokenizer is None:
        try:
            _punkt_tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
        except LookupError:
            nltk.download("punkt_tab", quiet=False)
            _punkt_tokenizer = nltk.data.load("tokenizers/punkt_tab/english.pickle")
    text = text.strip()
    if not text:
        return []
    return _punkt_tokenizer.tokenize(text)
