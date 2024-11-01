import json
import math
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def load_index(file_path):
    """
    Loads an index (inverted index or term frequency) from a JSON file.
    :param file_path: Path to the JSON file containing the index
    :return: Loaded index as a dictionary
    """
    try:
        with open(file_path, 'r') as f:
            index = json.load(f)
        return index
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return {}
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the input file.")
        return {}

def preprocess_query(query):
    """
    Preprocesses the search query by tokenizing, removing stop words, and stemming.
    :param query: The search query as a string
    :return: List of processed query tokens
    """
    tokens = word_tokenize(query.lower())
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

def tf_idf_score(term, doc_id, term_frequency, document_frequency, total_docs):
    """
    Calculates the TF-IDF score for a given term and document.
    :param term: The term to calculate the score for
    :param doc_id: The document ID
    :param term_frequency: Term frequency dictionary
    :param document_frequency: Document frequency dictionary
    :param total_docs: Total number of documents in the corpus
    :return: TF-IDF score
    """
    tf = term_frequency[doc_id].get(term, 0)
    df = document_frequency.get(term, 0)
    if tf == 0 or df == 0:
        return 0
    return tf * math.log(total_docs / df)

def search(query, inverted_index, term_frequency, document_frequency, total_docs):
    """
    Searches for documents relevant to the query using the inverted index and TF-IDF scoring.
    :param query: The search query as a string
    :param inverted_index: The inverted index dictionary
    :param term_frequency: Term frequency dictionary
    :param document_frequency: Document frequency dictionary
    :param total_docs: Total number of documents in the corpus
    :return: List of document IDs ranked by TF-IDF score
    """
    query_tokens = preprocess_query(query)
    document_scores = defaultdict(float)

    for token in query_tokens:
        if token in inverted_index:
            for doc_id, _ in inverted_index[token]:
                document_scores[doc_id] += tf_idf_score(token, doc_id, term_frequency, document_frequency, total_docs)

    ranked_docs = sorted(document_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_docs

def display_results(ranked_docs):
    """
    Displays the search results in a user-friendly format.
    :param ranked_docs: List of tuples (document ID, score)
    """
    if not ranked_docs:
        print("No results found.")
    else:
        print("\nSearch Results:")
        for doc_id, score in ranked_docs:
            print(f"Document: {doc_id} | Score: {score}")

if __name__ == "__main__":
    inverted_index_file = "data/inverted_index.json"
    term_frequency_file = "data/term_frequency.json"
    document_frequency_file = "data/document_frequency.json"

    inverted_index = load_index(inverted_index_file)
    term_frequency = load_index(term_frequency_file)
    document_frequency = load_index(document_frequency_file)
    total_docs = len(term_frequency)

    query = input("\nEnter your search query: ")
    ranked_docs = search(query, inverted_index, term_frequency, document_frequency, total_docs)
    display_results(ranked_docs)