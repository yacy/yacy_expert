import json
import logging
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import nocdex

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Search endpoint
@app.route('/yacysearch.json', methods=['GET', 'POST'])
def yacysearch():
    # Parse query and count from the request
    if request.method == 'GET':
        query = request.args.get('query', '')
        count = int(request.args.get('count', '20'))
    elif request.method == 'POST':
        data = request.get_json()
        query = data.get('query', '')
        count = int(data.get('count', '20'))

    query_keys = nocdex.tokenizer(query)
    boost = {"title": 5, "text_t": 1}
    sorted_ids_with_scores = nocdex.retrieve(query_keys, boost)
    logging.info(f"Search results: {len(sorted_ids_with_scores)}")

    # Extract document content for similarity computation
    results = []
    for id, score in sorted_ids_with_scores:

        # get the document
        doc = nocdex.get_document(id)
        if doc:
            result = {
                "title": doc.get("title", ""),
                "link": doc.get("url", ""),
                "description": doc.get("text_t", ""),
                "ranking": score
            }
            results.append(result)

        if len(results) >= count:
            break

    # Sort results by similarity (descending order)
    #results.sort(key=lambda x: x["similarity"], reverse=True)

    # Format the response in YaCy API format
    yacy_results = {
        "channels": [
            {
                "title": "YaCy Expert Search",
                "description": "Items from YaCy Search Engine Dumps as Search Results",
                "startIndex": "0",
                "itemsPerPage": str(count),
                "searchTerms": query,
                "items": results
            }
        ]
    }

    # Return the response as JSON with correct MIME type
    return Response(json.dumps(yacy_results), mimetype='application/json')


# Run the Flask app
if __name__ == '__main__':


    # define the index
    nocdex.define_index("title")
    nocdex.define_index("text_t")

    # Load documents into the index
    knowledge_folder = "knowledge"  # Folder containing JSON documents
    allowed_keys = ["url", "title", "keywords", "text_t"]
    nocdex.load_documents_into_index(knowledge_folder, allowed_keys)

    # Run the app
    app.run(debug=False, port=8094, host='0.0.0.0')
