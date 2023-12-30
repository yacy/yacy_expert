# NOCDEX, a Python module that provides functions for indexing and searching JSON documents.
# Its small, in-memory, very fast and supports only OR search by default.
# But it can boost fields and can do BM25 scoring!
# The code is purpose-built for YaCy Expert which requires not a complex full search enginne but something
# that can be embedded in a small Python application. Here it is used for RAG functions in YaCy Expert.
# (C) 2023 by Michael Christen
# License: Apache License 2.0

import os
import gzip
import json
import math
import logging
from collections import OrderedDict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Data structures for the inverted index
documents = {}  # Key: id (a counter), Value: full Document data (title, content, etc.)
search_index = {} # Key: field name, Value: inverted index for that field and avgdl in a dict
total_docs = 0 # Counter for assigning unique ids to documents

# metrics for BM25
k1 = 1.2
b = 0.75

def clean_text(text):
    text = text.replace("\n", " ").replace("\r", " ").replace("\t, ", " ").replace("-", " ").replace("#", " ").replace("[", " ").replace("]", " ")
    text = text.lower().strip().split()
    return text

# Parse a single JSON line and return the parsed document.
def parse_json_line(line):
    line = line.strip()
    if not line:
        return None  # Skip empty lines
    if line.startswith('{"index"'):
        return None  # Skip index lines
    try:
        doc = json.loads(line)
        if "index" in doc:
            return None  # Skip documents with "index" field
        return doc
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse JSON line: {line}. Error: {e}")
        return None

# Load and parse JSON documents from a file in parallel.
def load_document_parse_json(filepath):
    json_data = []
    
    # Check if the file exists
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return json_data  # Return an empty list

    # Read all lines from the file
    try:
        if filepath.endswith(".gz"):
            with gzip.open(filepath, "rt") as f:
                lines = f.readlines()
        else:
            with open(filepath, "r") as f:
                lines = f.readlines()
    except Exception as e:
        logging.error(f"Failed to read file: {filepath}. Error: {e}")
        return json_data

    if not lines:
        logging.warning(f"File is empty: {filepath}")
        return json_data

    # Use ThreadPoolExecutor to parse JSON lines in parallel
    with ThreadPoolExecutor() as executor:
        # Submit tasks to parse each line
        futures = [executor.submit(parse_json_line, line) for line in lines]
        
        # Collect results from completed tasks
        for future in futures:
            try:
                doc = future.result()
                if doc is not None:
                    json_data.append(doc)
            except Exception as e:
                logging.warning(f"Failed to process a line. Error: {e}")

    logging.info(f"Parsed {len(json_data)} documents from {filepath}.")
    return json_data

def define_index(fieldname):
    indexdefinition = {}
    indexdefinition["index"] = {}
    indexdefinition["avgdl"] = 0.0
    search_index[fieldname] = indexdefinition


# add a text to the index and return the number of words in the text
def reverse_index(index, text, id):
    words = clean_text(text)
    tf = {}
    
    # term frequency: count the number of appearances of each word in the text
    for word in words:
        if word not in tf:
            tf[word] = 1
        else:
            tf[word] += 1

    # Add (URL, term frequency) tuples to the index for each word
    for word, count in tf.items():
        if word not in index:
            index[word] = []
        index[word].append((id, count))

    # Return the number of words in the text
    return len(words)

def add_to_index(doc):
    global search_index, total_docs, avgdl_title, avgdl_text

    # store the document
    id = total_docs
    documents[id] = doc
    total_docs += 1

    # iterate over all index fields
    for fieldname in search_index:
        indexentry = search_index[fieldname]
        index = indexentry["index"]
        avgdl = indexentry["avgdl"]
        dlen = reverse_index(index, doc.get(fieldname, ""), id)
        doc[fieldname + "_dlen"] = dlen
        avgdl += dlen
        indexentry["avgdl"] = avgdl

def finish_indexing():
    for fieldname in search_index:
        indexentry = search_index[fieldname]
        indexentry["avgdl"] = indexentry["avgdl"] / total_docs
        logging.info(f"Index field '{fieldname}' has {len(indexentry.get("index", {}))} entries with average doc length {indexentry.get("avgdl", 0)}.")
        #logging.info(f"Sample entries for '{fieldname}': {list(indexentry['index'].items())[:1]}")

# Load JSON documents from the "knowledge" folder into the index
def load_documents_into_index(knowledge_folder, allowed_keys):
    global total_docs, avgdl_title, avgdl_text
    for filename in os.listdir(knowledge_folder):
        if filename.endswith(".jsonl") or filename.endswith(".flatjson") or filename.endswith(".jsonl.gz") or filename.endswith(".flatjson.gz"):
            filepath = os.path.join(knowledge_folder, filename)
            logging.info(f"Reading index dump from {filepath}.")
            json_data = load_document_parse_json(filepath)
            # loop over all documents in the file
            for doc in json_data:
                # data cleaning
                url = doc.get("url", doc.get("url_s", doc.get("sku", "")))
                if not url:
                    continue  # Skip documents without a URL
                doc["url"] = url

                # fix the title if it is a list
                title = doc.get("title", "")
                if isinstance(title, list): title = title[0]
                doc["title"] = title

                # remove unwanted keys
                for key in list(doc.keys()):
                    if key not in allowed_keys:
                        del doc[key]
                
                # index and store the document
                add_to_index(doc)

            logging.info(f"Finished indexing of {filepath}.")
    
    # finish indexing: compute average document length
    finish_indexing()

def retrieve4index(query_keys, index, index_dlen_name, avgdl):
    # collect matching ids and count how often they appear for each word
    # first iterate over all words in the query
    matching_ids = {} # this holds the term frequency for each id of that key
    for key in query_keys:
        if key in index:
            index4key = index[key]
            for id, count in index4key:
                if id in documents:
                    doc = documents[id]
                    dlen = doc.get(index_dlen_name, 0)
                    # now we have everything to compute the BM25 TF part
                    tf = count * (k1 + 1) / (count + k1 * (1 - b + b * (dlen / avgdl)))
                    # compute the IDF part
                    len_index4key = len(index4key)
                    idf = math.log((total_docs - len_index4key + 0.5) / (len_index4key + 0.5) + 1.0)
                    if id in matching_ids:
                        matching_ids[id] += idf * tf
                    else:
                        matching_ids[id] = idf * tf

    # return the dictionary with the matching ids and their score
    # logging.info(f"BM25 scores for query keys {query_keys} in index {index}: {matching_ids}")
    return matching_ids

def retrieve(query_keys, boostdict):
    logging.info(f"Search started for query keys {query_keys}")

    # loop over index fields
    matching_ids = {}
    for fieldname in search_index:
        boost = boostdict.get(fieldname, 0.0)
        if (boost <= 0.0): continue
        indexentry = search_index[fieldname]
        index = indexentry["index"]
        avgdl = indexentry["avgdl"]
        this_matching_ids = retrieve4index(query_keys, index, fieldname + "_dlen", avgdl)
        logging.info(f"Field '{fieldname}' has {len(this_matching_ids)} matches")

        for id in this_matching_ids:
            if id in matching_ids:
                matching_ids[id] += boost * this_matching_ids[id]
            else:
                matching_ids[id] = boost * this_matching_ids[id]

    logging.info(f"All matches for '{query_keys}' are {len(matching_ids)} matches")
    # Sort the dictionary by values in descending order
    sorted_ids_with_scores = sorted(matching_ids.items(), key=lambda item: item[1], reverse=True)

    return sorted_ids_with_scores

# Function to compute cosine similarity
def compute_similarity(query, documents_text):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([query] + documents_text)
    similarity_scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return similarity_scores

if __name__ == '__main__':
    # test
    knowledge_folder = "knowledge"  # Folder containing JSON documents
    allowed_keys = ["url", "title", "keywords", "text_t"]
    load_documents_into_index(knowledge_folder, allowed_keys)
    boost = {"title": 5, "text_t": 1}
    sorted_ids = retrieve(clean_text("hilfe"), boost)
    logging.info(f"Search results: {sorted_ids}")
