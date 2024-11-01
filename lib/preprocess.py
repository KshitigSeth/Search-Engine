import fitz  # PyMuPDF for PDF text extraction
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
import json

nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def extract_text_from_pdf(pdf_path):
    """
    Extracts text from a PDF file.
    :param pdf_path: Path to the PDF file
    :return: Extracted text as a single string
    """
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

def preprocess_text(text):
    """
    Tokenizes, removes stop words, and stems the input text.
    :param text: Raw text to preprocess
    :return: List of processed tokens
    """
    tokens = word_tokenize(text.lower())
    tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

def process_and_save_all_pdfs(pdf_directory, output_file):
    """
    Processes all PDFs in the given directory and saves the processed data.
    :param pdf_directory: Directory containing PDF files
    :param output_file: Path to save the processed text data
    """
    processed_docs = {}

    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            text = extract_text_from_pdf(pdf_path)
            tokens = preprocess_text(text)

            doc_id = filename[:-4] # Remove file extension
            processed_docs[doc_id] = tokens

    with open(output_file, 'w') as f:
        json.dump(processed_docs, f, indent=4)

    print(f"\nProcessed data saved to {output_file}\n")

if __name__ == "__main__":
    pdf_directory = "data/PDFs"
    output_file = "data/processed_docs.json"
    process_and_save_all_pdfs(pdf_directory, output_file)