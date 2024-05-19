import spacy
import pandas as pd
# Load SpaCy model
nlp = spacy.load('en_core_web_sm')


def preprocess_text(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return ' '.join(tokens)

def preprocess_bios(bios):
    # Ensure bios is a pandas Series
    if not isinstance(bios, pd.Series):
        bios = pd.Series(bios)
    return bios.apply(preprocess_text)

