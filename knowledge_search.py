import os
import json
import faiss
import argparse
import expert_common
import knowledge_indexing
from flask import Flask, request, jsonify, Response, make_response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Function to load FAISS indexes and corresponding data
def load_faiss_indexes(knowledge_path):
    index = {}
    datas = {}
    ini_names = {}  # Add a dictionary to store the ini names
    for faiss_name in os.listdir(knowledge_path):
        if faiss_name.endswith(".faiss"):
            jsonl_name = faiss_name[:-6] # ok, this is the jsonl name without possible .gz suffix
            index_name = jsonl_name[:jsonl_name.rfind('.')]
            ini_name = jsonl_name + '.ini' #ok, this is the ini name
            index_file = os.path.join(knowledge_path, faiss_name)
            print(f"Loading index file: {index_file}")

            # the jsonl file might either be one with or without .gz suffix
            if os.path.exists(os.path.join(knowledge_path, jsonl_name + '.gz')):
                jsonl_name = jsonl_name + '.gz'
            jsonl_file = os.path.join(knowledge_path, jsonl_name)

            print(f"Loading jsonl file: {jsonl_file}")
            index[index_name] = faiss.read_index(index_file)
            print(f"Size of faiss index file {index_name}: {index[index_name].ntotal}")
            datas[index_name] = expert_common.read_text_list(jsonl_file) # these are just text lines
            print(f"Size of index  data file {jsonl_file}: {len(datas[index_name])}")
            ini_names[index_name] = os.path.join(knowledge_path, ini_name) # Store the ini name for each index
    
    return index, datas, ini_names  # Return the ini_names dictionary

# Load all FAISS indexes and data from the data path
faiss_indexes, jsonl_text, ini_names = load_faiss_indexes(expert_common.knowledge_path())

# load ini file if it exists
knowledge = expert_common.knowledge_path()
model_name = knowledge_indexing.load_ini(os.path.join(knowledge, 'knowledge.ini'))
tokenizer, model = knowledge_indexing.tokenizer_model_from_name(model_name)

# Function to search across all indexes
def search_across_indexes(query, k):
    combined_results = []
    for index_name, faiss_index in faiss_indexes.items():

        # Embed the query
        max_sequence_length = model.config.max_position_embeddings
        query_vector = knowledge_indexing.embedding(query, model_name, tokenizer, model, max_sequence_length)
        query_vector = query_vector.reshape(1, -1).astype('float32')

        distances, indices = faiss_index.search(query_vector, k)
        for i, idx in enumerate(indices[0]):
            print(f"Index: {index_name}, Distance: {distances[0][i]}, Index: {idx}")
            if idx != -1:  # Ignore invalid indices
                text_line = jsonl_text[index_name][idx]
                result = json.loads(text_line)
                distance = distances[0][i]
                result['distance'] = float(distance)
                result['index'] = index_name
                combined_results.append(result)
    combined_results.sort(key=lambda x: x['distance'])
    return combined_results[:k]

# Endpoint for search
@app.route('/yacysearch.json', methods=['GET', 'POST'])
def yacysearch():
    #print(f"Request: {request}")
    if request.method == 'GET':
        query = request.args.get('query', '')
        count = int(request.args.get('count', '3'))
    elif request.method == 'POST':
        data = request.get_json()
        #print(f"Data: {data}")
        query = data.get('query', '')
        count = int(data.get('count', '3'))

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

    print(f"Searching for '{query}' with count {count}")
    if query:
        # Search across all indexes
        results = search_across_indexes(query=query, k=count)

        for result in results:
            text_t = result.get('text_t', '')
            if (len(text_t) > 0):
                item = {
                    "title": result.get('title', ''),
                    "link": result.get('url', result.get('url_s', result.get('sku', ''))),
                    "description": text_t,
                    "pubDate": "",
                    "image": result.get('image', ''),
                    "distance": result.get('distance', ''),
                    "index": result.get('index', '')
                }
            yacy_results['channels'][0]['items'].append(item)

    # Pretty-print the result
    pretty_json = json.dumps(yacy_results, indent=4)
    
    response = Response(pretty_json, content_type="application/json; charset=utf-8")
    return response

if __name__ == '__main__':
    # Set up the argument parser
    parser = argparse.ArgumentParser(description='Server for YaCy Expert Vector Search from Faiss Indexes.')
    parser.add_argument('--port', type=int, default=8094, help='Port to run the Flask app on.')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the Flask app on.')
    args = parser.parse_args()
    app.run(debug=False, port=args.port, host=args.host)

#curl -X POST "http://localhost:8094/yacysearch.json" -H "Content-Type: application/json" -d '{"query": "one two three", "count": "1"}'
# or http://localhost:8094/yacysearch.json?q=hello%20my%friend&count=1