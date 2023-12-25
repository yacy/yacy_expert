import os
import json
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import lucene
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.document import Document, Field, FieldType, StringField, TextField
from org.apache.lucene.index import IndexWriter, IndexWriterConfig, DirectoryReader
from org.apache.lucene.store import RAMDirectory
from org.apache.lucene.search import IndexSearcher, QueryParser
from org.apache.lucene.queryparser.classic import QueryParser
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# install pylucene:
# wget https://dlcdn.apache.org/lucene/pylucene/pylucene-10.0.0-src.tar.gz
# tar -xf pylucene-10.0.0-src.tar.gz
# cd pylucene-10.0.0
# make
# make install


# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize Lucene
lucene.initVM()

# Define the schema for the Lucene index
analyzer = StandardAnalyzer()
index_dir = RAMDirectory()
config = IndexWriterConfig(analyzer)
writer = IndexWriter(index_dir, config)

# Load JSON documents from the "knowledge" folder into the Lucene index
def load_documents_into_index(knowledge_folder):
    for filename in os.listdir(knowledge_folder):
        if filename.endswith(".jsonl") or filename.endswith(".flatjson"):
            filepath = os.path.join(knowledge_folder, filename)
            print("Reading index dump from " + filepath)
            n = 0
            with open(filepath, "r") as f:
                for line in f:
                    doc = json.loads(line.strip())
                    if "index" in doc:
                        continue  # Skip this line
                    lucene_doc = Document()
                    lucene_doc.add(StringField("title", doc.get("title", ""), Field.Store.YES))
                    lucene_doc.add(TextField("content", doc.get("text_t", ""), Field.Store.YES))
                    lucene_doc.add(StringField("url", doc.get("url", doc.get("url_s", doc.get("sku", ""))), Field.Store.YES))
                    lucene_doc.add(StringField("pubDate", doc.get("pubDate", ""), Field.Store.YES))
                    writer.addDocument(lucene_doc)
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

    # Perform the search using Lucene
    results = []
    reader = DirectoryReader.open(index_dir)
    searcher = IndexSearcher(reader)
    query_parser = QueryParser("content", analyzer)
    parsed_query = query_parser.parse(query)
    lucene_results = searcher.search(parsed_query, count)

    # Extract document content for similarity computation
    documents = []
    for score_doc in lucene_results.scoreDocs:
        doc = searcher.doc(score_doc.doc)
        documents.append(doc.get("content"))

    if documents:
        # Compute similarity scores
        similarity_scores = compute_similarity(query, documents)

        # Combine results with similarity scores
        for i, score_doc in enumerate(lucene_results.scoreDocs):
            doc = searcher.doc(score_doc.doc)
            result = {
                "title": doc.get("title", ""),
                "link": doc.get("url", ""),
                "description": doc.get("content", ""),
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