import os
from lib.preprocess import process_and_save_all_pdfs
from lib.index import load_processed_docs, build_inverted_index, save_index
from lib.search import load_index, search, display_results

def main():
    pdf_directory = "data/PDFs"
    processed_docs_file = "data/processed_docs.json"
    inverted_index_file = "data/inverted_index.json"
    term_frequency_file = "data/term_frequency.json"
    document_frequency_file = "data/document_frequency.json"

    # Step 1: Preprocess if not already done
    if not os.path.exists(processed_docs_file):
        print("Preprocessing documents...")
        process_and_save_all_pdfs(pdf_directory, processed_docs_file)
        print("Document preprocessing complete.\n")
    else:
        print("Processed documents file found. Skipping preprocessing.\n")

    # Step 2: Indexing if not already done
    if not (os.path.exists(inverted_index_file) and os.path.exists(term_frequency_file) and os.path.exists(document_frequency_file)):
        print("Building index...")
        docs = load_processed_docs(processed_docs_file)
        inverted_index, term_frequency, document_frequency = build_inverted_index(docs)
        save_index(inverted_index, inverted_index_file)
        save_index(term_frequency, term_frequency_file)
        save_index(document_frequency, document_frequency_file)
        print("Indexing complete.\n")
    else:
        print("Index files found. Skipping indexing.\n")

    # Step 3: Query interface
    print("Welcome to the Search Engine!")
    print("Type your search query below, or type '$exit$' to quit.\n")

    while True:
        query = input("Enter your search query: ")
        if query.lower() == '$exit$':
            print("Exiting search engine. Goodbye!")
            break
        
        # Load indexes for search
        inverted_index = load_index(inverted_index_file)
        term_frequency = load_index(term_frequency_file)
        document_frequency = load_index(document_frequency_file)
        total_docs = len(term_frequency)

        # Perform search and display results
        ranked_docs = search(query, inverted_index, term_frequency, document_frequency, total_docs)
        display_results(ranked_docs)

if __name__ == "__main__":
    main()