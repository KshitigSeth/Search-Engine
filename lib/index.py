import json
from collections import defaultdict

def load_processed_docs(input_file):
    """
    Loads the processed documents from a JSON file.
    :param input_file: Path to the JSON file with processed document data
    :return: Dictionary where keys are document IDs and values are lists of tokens
    """
    try:
        with open(input_file, 'r') as f:
            processed_docs = json.load(f)
        return processed_docs
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return {}
    except json.JSONDecodeError:
        print("Error: Failed to decode JSON from the input file.")
        return {}

def build_inverted_index(docs):
    """
    Builds an inverted index and a term frequency dictionary from the processed documents.
    :param docs: Dictionary of processed documents, where keys are document IDs and values are lists of tokens
    :return: Inverted index, term frequency dictionary, and document frequency dictionary
    """
    inverted_index = defaultdict(list)
    term_frequency = defaultdict(lambda: defaultdict(int))
    document_frequency = defaultdict(int)

    for doc_id, tokens in docs.items():
        seen_terms = set()
        total_terms = len(tokens)
        for position, token in enumerate(tokens):
            inverted_index[token].append((doc_id, position))
            term_frequency[doc_id][token] += 1

            if token not in seen_terms:
                document_frequency[token] += 1
                seen_terms.add(token)
        
        for term in term_frequency[doc_id]:
            term_frequency[doc_id][term] /= total_terms

    return inverted_index, term_frequency, document_frequency

def save_index(data, output_file):
    """
    Saves any dictionary data to a JSON file.
    :param data: Data to save
    :param output_file: Path to the output JSON file
    """
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"\nIndex saved to {output_file}\n")
    except IOError:
        print(f"Error: Could not write to file {output_file}")

if __name__ == "__main__":
    input_file = "data/processed_docs.json"
    inverted_index_file = "data/inverted_index.json"
    term_frequency_file = "data/term_frequency.json"
    document_frequency_file = "data/document_frequency.json"
    
    docs = load_processed_docs(input_file)
    inverted_index, term_frequency, document_frequency = build_inverted_index(docs)
    
    save_index(inverted_index, inverted_index_file)
    save_index(term_frequency, term_frequency_file)
    save_index(document_frequency, document_frequency_file)