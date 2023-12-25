import os
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from whoosh.fields import Schema, TEXT, ID, STORED
from whoosh.index import Index
from whoosh.qparser import QueryParser
from whoosh.filedb.filestore import RamStorage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Define the schema for the Whoosh index
schema = Schema(
    title=TEXT(phrase=False, stored=True),  # Document title
    content=TEXT(phrase=False, stored=True),  # Document content
    url=STORED,  # URL of the document
    pubDate=STORED  # Publication date (if any)
)

# Create an in-memory index using RamStorage
storage = RamStorage()
ix = storage.create_index(schema) # Create an index with RamStorage

# Load JSON documents from the "knowledge" folder into the Whoosh index
def load_documents_into_index(knowledge_folder):
    writer = ix.writer()
    for filename in os.listdir(knowledge_folder):
        if filename.endswith(".jsonl") or filename.endswith(".flatjson"):
            filepath = os.path.join(knowledge_folder, filename)
            print("reading index dump from " + filepath)
            n = 0
            with open(filepath, "r") as f:
                for line in f:
                    doc = json.loads(line.strip())
                    if "index" in doc:
                        continue # skip this line
                    writer.add_document(
                        title=doc.get("title", ""),
                        content=doc.get("text_t", ""),
                        url=doc.get("url", doc.get("url_s", doc.get("sku", ""))),
                        pubDate=doc.get("pubDate", "")
                    )
                    n += 1
                    if n > 1000:
                        break
    writer.commit()
    
# Load documents into the index
knowledge_folder = "knowledge"  # Folder containing JSON documents
load_documents_into_index(knowledge_folder)

# Function to compute TF-IDF vectors and cosine similarity
def compute_similarity(query, documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([query] + documents)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarity_scores

# Search endpoint
@app.route('/yacysearch.json', methods=['GET', 'POST'])
def yacysearch():
    # Parse query and count from the request
    if request.method == 'GET':
        query = request.args.get('query', '')
        count = int(request.args.get('count', '3'))
    elif request.method == 'POST':
        data = request.get_json()
        query = data.get('query', '')
        count = int(data.get('count', '3'))

    # Perform the search using Whoosh
    results = []
    with ix.searcher() as searcher:
        query_parser = QueryParser("content", ix.schema)
        parsed_query = query_parser.parse(query)
        # Perform the search using or logic
        whoosh_results = searcher.search(parsed_query, limit=count, terms=True)

        # Extract document content for similarity computation
        documents = [hit["content"] for hit in whoosh_results]
        if documents:
            # Compute similarity scores
            similarity_scores = compute_similarity(query, documents)

            # Combine results with similarity scores
            for i, hit in enumerate(whoosh_results):
                result = {
                    "title": hit.get("title", ""),
                    "link": hit.get("url", ""),
                    "description": hit.get("content", ""),
                    "similarity": float(similarity_scores[i])
                }
                results.append(result)

    # Sort results by similarity (descending order)
    results.sort(key=lambda x: x["similarity"], reverse=True)

    # Format the response in YaCy API format
    yacy_results = {
        "channels": [
            {
                "title": "YaCy Expert Search",
                "description": "Items from YaCy Search Engine Dumps as Search Results",
                "startIndex": "0",
                "itemsPerPage": str(count),
                "searchTerms": query,
                "items": results[:count]  # Limit results to the requested count
            }
        ]
    }

    # Return the response as JSON
    return jsonify(yacy_results)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False, port=8094, host='0.0.0.0')
