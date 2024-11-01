import os
import requests
import time
from xml.etree import ElementTree as ET

# List of search queries (ArXiv categories or keywords)
search_queries = [
    "cs.AI",                  # Artificial Intelligence
    "cs.LG",                  # Machine Learning
    "cs.CL",                  # Natural Language Processing
    "cs.CV",                  # Computer Vision
    "cs.DB",                  # Databases
    "cs.OS",                  # Operating Systems
    "cs.CR",                  # Cryptography
    "cs.DS",                  # Data Structures
    "math.PR",                # Probability
    "stat.ML",                # Statistics in Machine Learning
    "math.CA",                # Calculus
    "math.LA",                # Linear Algebra
    "math.FA",                # Modern Analysis (Functional Analysis)
    "math.DS",                # Dynamical Systems (Differential Equations)
    "q-fin.TR",               # Trading
    "q-fin.RM",               # Risk Management
    "q-fin.PM",               # Portfolio Management
    "q-fin.ST",               # Quantitative Finance (Statistics)
    "q-fin.PR"                # Pricing of Securities (Stock Markets)
]

max_results_per_query = 10   # number of papers per query
output_dir = "data/PDFs"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def download_pdf(url, file_path):
    """
    Downloads a PDF file from a URL to the specified file path.
    :param url: URL of the PDF file
    :param file_path: Local path to save the PDF
    """
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded: {file_path}")
    else:
        print(f"Failed to download: {url}")

def fetch_arxiv_papers(query, max_results):
    """
    Fetches paper metadata from ArXiv using the ArXiv API and downloads PDFs.
    :param query: Search query or category (e.g., "cs.AI" for AI papers in computer science)
    :param max_results: Number of papers to fetch
    """
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print("Error fetching data from ArXiv API")
        return
    
    root = ET.fromstring(response.text)
    namespace = {"ns": "http://www.w3.org/2005/Atom"}

    for entry in root.findall("ns:entry", namespace):
        paper_id = entry.find("ns:id", namespace).text.split("/")[-1]
        title = entry.find("ns:title", namespace).text.strip()
        pdf_url = entry.find("ns:link[@title='pdf']", namespace).attrib['href']

        file_name = f"{paper_id}.pdf"
        file_path = os.path.join(output_dir, file_name)

        if not os.path.exists(file_path):
            print(f"Downloading {title}...")
            download_pdf(pdf_url, file_path)
            time.sleep(1)  # Delay to avoid overloading the server

if __name__ == "__main__":
    for query in search_queries:
        print(f"\nFetching papers for query: {query}")
        fetch_arxiv_papers(query, max_results_per_query)
