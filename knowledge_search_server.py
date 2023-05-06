import os
import json
import faiss
import knowledge_indexing
import argparse
from flask import Flask, request, jsonify, Response, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load pre-trained tokenizer and model
tokenizer = knowledge_indexing.tokenizer
model = knowledge_indexing.model

# Function to load FAISS indexes and corresponding data
def load_faiss_indexes(knowledge_path):
    index = {}
    datas = {}
    jsons = {}
    for faiss_name in os.listdir(knowledge_path):
        if faiss_name.endswith(".faiss"):
            jsonl_name = faiss_name[:-6]
            index_name = jsonl_name[:-6]
            index_file = os.path.join(knowledge_path, faiss_name)
            jsonl_file = os.path.join(knowledge_path, jsonl_name)

            index[index_name] = faiss.read_index(index_file)
            datas[index_name] = knowledge_indexing.read_text_list(jsonl_file) # these are just text lines
            jsons[index_name] = [json.loads(line) for line in datas[index_name]] # these are json objects
    
            # validate the loaded indexes, both must have the same size
            print(f"Loaded {index[index_name].ntotal} indexes   from {index_file}")
            print(f"Loaded {len(jsons[index_name])} documents from {jsonl_file}")

    return index, jsons

# Load all FAISS indexes and data from the data path
faiss_indexes, index_data = load_faiss_indexes(knowledge_indexing.knowledge_path())

# Function to search across all indexes
def search_across_indexes(query_vector, k):
    combined_results = []
    for index_name, faiss_index in faiss_indexes.items():
        distances, indices = faiss_index.search(query_vector, k)
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Ignore invalid indices
                result = index_data[index_name][idx]
                result['distance'] = float(distances[0][i])
                combined_results.append(result)
    combined_results.sort(key=lambda x: x['distance'])
    return combined_results[:k]

# Endpoint for search
@app.route('/yacysearch.json', methods=['GET', 'POST'])
def search():
    
    if request.method == 'GET':
        query = request.args.get('query', '')
        count = int(request.args.get('count', '1'))
    elif request.method == 'POST':
        data = request.get_json()
        query = data.get('query', '')
        count = int(data.get('count', '1'))

    if query:
        # Embed the query
        vector = knowledge_indexing.embedding(query)
        vector = vector.reshape(1, -1).astype('float32')

        # Search across all indexes
        results = search_across_indexes(vector, count)

        # Translate the results to the yacysearch.json format
        yacy_results = {
            "channels": [
                {
                    "title": "YaCy Expert Vector Search",
                    "description": "Items from YaCy Search Engine Dumps as Vector Search Results",
                    "startIndex": "0",
                    "itemsPerPage": str(count),
                    "searchTerms": query,
                    "items": []
                }
            ]
        }
        for result in results:
            text_t = result.get('text_t', '')
            if (len(text_t) > 0):
                item = {
                    "title": result.get('title', ''),
                    "link": result.get('url', ''),
                    "description": text_t,
                    "pubDate": "",
                    "image": result.get('image', ''),
                    "ranking": str(result.get('distance', ''))
                }
                yacy_results['channels'][0]['items'].append(item)

        # Pretty-print the result
        pretty_json = json.dumps(yacy_results, indent=4)
        response = Response(pretty_json, content_type="application/json; charset=utf-8")
        return response
    else:
        error_message = json.dumps({"error": "Invalid query or index name"}, indent=4)
        return Response(error_message, status=400, content_type="application/json; charset=utf-8")

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Server for YaCy Expert Vector Search from Faiss Indexes.')
    parser.add_argument('--port', type=int, default=8094, help='Port to run the Flask app on.')
    args = parser.parse_args()
    app.run(debug=False, port=args.port)

#curl -X POST "http://localhost:8094/yacysearch.json" -H "Content-Type: application/json" -d '{"query": "one two three", "count": "1"}'