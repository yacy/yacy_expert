# NOCDEX, a Python module that provides functions for indexing and searching JSON documents.
# Its small, in-memory, very fast and supports only OR search by default.
# But it can boost fields and can do BM25 scoring!
# The code is purpose-built for YaCy Expert which requires not a complex full search enginne but something
# that can be embedded in a small Python application. Here it is used for RAG functions in YaCy Expert.
# (C) 2023 by Michael Christen
# License: Apache License 2.0

import os
import math
import logging
import jsonloader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Data structures for the inverted index
__documents = {}  # Key: id (a counter), Value: full Document data (title, content, etc.)
__search_index = {} # Key: field name, Value: inverted index for that field and avgdl in a dict
__total_docs = 0 # Counter for assigning unique ids to documents

# metrics for BM25
K1 = 1.2
B = 0.75

DELCHARS = "'()-+*/\n\r\t-#.,?![\\]^_{|}~"
TRANSLATION = str.maketrans(DELCHARS, " " * len(DELCHARS))

MAXHITS = 100

def tokenizer(text):
    text = text.translate(TRANSLATION)
    query_keys = text.lower().strip().split()
    return query_keys

def define_index(fieldname):
    indexdefinition = {}
    indexdefinition["index"] = {}
    indexdefinition["avgdl"] = 0.0
    __search_index[fieldname] = indexdefinition

# add a text to the index and return the number of words in the text
def __reverse_index(index, text, id):
    words = tokenizer(text)
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
    global __search_index, __total_docs

    # store the document
    id = __total_docs
    __documents[id] = doc
    __total_docs += 1

    # iterate over all index fields
    for fieldname in __search_index:
        indexentry = __search_index[fieldname]
        index = indexentry["index"]
        avgdl = indexentry["avgdl"]
        dlen = __reverse_index(index, doc.get(fieldname, ""), id)
        doc[fieldname + "_dlen"] = dlen
        avgdl += dlen
        indexentry["avgdl"] = avgdl

def finish_indexing():
    for fieldname in __search_index:
        fieldindex = __search_index[fieldname]
        fieldindex["avgdl"] = fieldindex["avgdl"] / __total_docs
        logging.info(f"Index field '{fieldname}' has {len(fieldindex.get("index", {}))} entries with average doc length {fieldindex.get("avgdl", 0)}.")
        #logging.info(f"Sample entries for '{fieldname}': {list(indexentry['index'].items())[:1]}")

        # iterate over all words in the index and sort the list of (id, count) tuples
        index = fieldindex["index"]
        index_size = {}
        fieldindex["index_size"] = index_size
        for word in index:
            # sort the list of (id, count) tuples by count in descending order
            index[word].sort(key=lambda x: x[1], reverse=True)
            
            # we must preserve the original length of the list for the IDF scoring
            index_size[word] = len(index[word])

            # now as that lists is sorted and the length was remembered, we can truncate the list to a maximum length of MAXHITS
            index[word] = index[word][:MAXHITS]

            #print(f"Word '{word}': {index[word]}")

# Load JSON documents from the "knowledge" folder into the index
def load_documents_into_index(knowledge_folder, allowed_keys):
    global __total_docs
    for filename in os.listdir(knowledge_folder):
        if filename.endswith(".jsonl") or filename.endswith(".flatjson") or filename.endswith(".jsonl.gz") or filename.endswith(".flatjson.gz"):
            filepath = os.path.join(knowledge_folder, filename)
            logging.info(f"Reading index dump from {filepath}.")
            json_data = jsonloader.load(filepath)
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

def __retrieve4index(query_keys, fieldname, maxcount):
    # collect matching ids and count how often they appear for each word
    # first iterate over all words in the query
    matching_ids = {} # this holds the term frequency for each id of that key
    fieldindex = __search_index[fieldname]
    avgdl = fieldindex["avgdl"]
    index = fieldindex["index"]
    index_size = fieldindex["index_size"]
    for key in query_keys:
        if key in index:
            index4key = index[key]

            # compute the IDF part
            len_index4key = index_size[key]
            logging.info(f"Field '{fieldname}' has {len_index4key} matches for keys {key}.")
            idf_bm25 = math.log((__total_docs - len_index4key + 0.5) / (len_index4key + 0.5) + 1.0)

            # iterate over all (id, count) tuples for that key to compute the TF part
            for id, count in index4key:
                if id in __documents:
                    doc = __documents[id]
                    dlen = doc.get(fieldname + "_dlen", 0)
                    # compute the BM25 TF part
                    tf_bm25 = count * (K1 + 1) / (count + K1 * (1 - B + B * (dlen / avgdl)))
                    if id in matching_ids:
                        matching_ids[id] += idf_bm25 * tf_bm25
                    else:
                        matching_ids[id] = idf_bm25 * tf_bm25
        if len(matching_ids) >= maxcount:
            break

    # return the dictionary with the matching ids and their score
    logging.info(f"BM25 scores for query keys {query_keys} in index {fieldname}: {matching_ids}")
    return matching_ids

def retrieve(query_keys, boostdict, maxcount):
    logging.info(f"Search started for query keys {query_keys}")

    # loop over index fields
    matching_ids = {}
    for fieldname in __search_index:
        boost = boostdict.get(fieldname, 0.0)
        if (boost <= 0.0): continue
        this_matching_ids = __retrieve4index(query_keys, fieldname, maxcount)
        for id in this_matching_ids:
            if id in matching_ids:
                matching_ids[id] += boost * this_matching_ids[id]
            else:
                matching_ids[id] = boost * this_matching_ids[id]

    logging.info(f"All matches for '{query_keys}' are {len(matching_ids)} matches")
    # Sort the dictionary by values in descending order
    sorted_ids_with_scores = sorted(matching_ids.items(), key=lambda item: item[1], reverse=True)

    # limit the number of results to maxcount
    if len(sorted_ids_with_scores) > maxcount:
        sorted_ids_with_scores = sorted_ids_with_scores[:maxcount]
    return sorted_ids_with_scores

def get_document(id):
    return __documents.get(id, {})

# Function to compute cosine similarity
def compute_similarity(query, documents_text):
    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform([query] + documents_text)
    similarity_scores = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
    return similarity_scores

if __name__ == '__main__':
    # test

    # define the index
    define_index("title")
    define_index("text_t")

    # start indexing
    knowledge_folder = "knowledge"  # Folder containing JSON documents
    allowed_keys = ["url", "title", "keywords", "text_t"]
    load_documents_into_index(knowledge_folder, allowed_keys)

    # start search
    query="frankfurt"
    query_keys = tokenizer(query)
    boost = {"title": 5, "text_t": 1}
    count = 50
    sorted_ids = retrieve(query_keys, boost, count)
    logging.info(f"Search results: {len(sorted_ids)}")
    for id, score in sorted_ids:
        # get the document
        doc = get_document(id)
        logging.info(f"Document {id} has score {score} and title {doc.get('title', '')}")

