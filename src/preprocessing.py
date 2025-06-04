import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer # Removing stemmer for simplicity

# --- NLTK Data Download --- 
def download_nltk_data():
    """Downloads necessary NLTK data (stopwords) if not found."""
    try:
        nltk.data.find("corpora/stopwords")
        print("NLTK stopwords found.")
    except LookupError:
        print("NLTK stopwords not found. Downloading...")
        nltk.download("stopwords", quiet=True)
        print("NLTK stopwords downloaded.")
        # Verify after download attempt
        try:
            nltk.data.find("corpora/stopwords")
            print("NLTK stopwords now available.")
        except LookupError:
            print("Warning: NLTK stopwords still not found after download attempt.")

download_nltk_data()
# --- End NLTK Data Download ---

stop_words = set(stopwords.words("english"))
# stemmer = PorterStemmer() # Removing stemmer

def preprocess_text(text: str) -> str:
    """Cleans and preprocesses text data (simplified).

    - Converts to lowercase
    - Removes punctuation
    - Removes numbers
    - Splits into words (simple split)
    - Removes stopwords
    # - Stems words (Removed for simplicity)

    Args:
        text: The input text string.

    Returns:
        The cleaned text string.
    """
    if not isinstance(text, str):
        return "" # Return empty string for non-string input

    text = text.lower() # Lowercase
    text = text.translate(str.maketrans("", "", string.punctuation)) # Remove punctuation
    text = re.sub(r"\d+", "", text) # Remove numbers
    # Simple split instead of nltk.word_tokenize to avoid punkt_tab dependency
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words] # Remove stopwords
    # tokens = [stemmer.stem(word) for word in tokens] # Stemming removed
    return " ".join(tokens)

def create_tfidf_vectorizer(texts: list[str], max_features: int = 5000) -> TfidfVectorizer:
    """Creates and fits a TF-IDF vectorizer.

    Args:
        texts: A list of preprocessed text documents.
        max_features: The maximum number of features (terms) to keep.

    Returns:
        A fitted TfidfVectorizer object.
    """
    vectorizer = TfidfVectorizer(max_features=max_features)
    vectorizer.fit(texts)
    print(f"TF-IDF Vectorizer fitted with {len(vectorizer.get_feature_names_out())} features.")
    return vectorizer

if __name__ == "__main__":
    # Example Usage
    sample_texts = [
        "This is the first document, with numbers 123 and punctuation!",
        "This document is the second document.",
        "And this is the third one.",
        "Is this the first document? Yes, it is."
    ]

    print("Original Texts:")
    for text in sample_texts:
        print(text)

    processed_texts = [preprocess_text(text) for text in sample_texts]
    print("\nProcessed Texts (Simplified):")
    for text in processed_texts:
        print(text)

    vectorizer = create_tfidf_vectorizer(processed_texts, max_features=10)
    tfidf_matrix = vectorizer.transform(processed_texts)

    print("\nTF-IDF Matrix Shape:", tfidf_matrix.shape)
    print("\nFeature Names:", vectorizer.get_feature_names_out())

