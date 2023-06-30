import os
import json
import requests
import argparse
from flask import Flask, request, jsonify, make_response, Response, stream_with_context
from flask_cors import CORS
from urllib.parse import urlparse
import http.client

app = Flask(__name__)
CORS(app)

# default settings
# run this i.e. with:
# OPENAI_API_KEY="sk-.." python3 expert_server.py
YACYSEARCH_HOST = "http://localhost:8094"
OPENAI_API_HOST = "https://api.openai.com"
OPENAI_API_KEY  = ""
RAG = False
RAG_CONTEXT_PREFIX = "If necessary, use the following context to answer the question. If this text does not match the question, ignore it."
#RAG_CONTEXT_PREFIX = "Wenn nötig, verwende den folgenden Kontext, um die Frage zu beantworten. Wenn dieser Text nicht zur Frage passt, ignorieren Sie ihn."
SYSTEM = False
SYSTEM_CONTEXT_PREFIX = "If necessary, use the following context to answer the question. If this text does not match the question, ignore it."
SYSTEM_CONTEXT_PATH = "knowledge"
    
# overload settings with environment variables in case they are set
if 'YACYSEARCH_HOST' in os.environ: YACYSEARCH_HOST = os.environ['YACYSEARCH_HOST']
if 'OPENAI_API_HOST' in os.environ: OPENAI_API_HOST = os.environ['OPENAI_API_HOST']
if 'OPENAI_API_KEY'  in os.environ: OPENAI_API_KEY  = os.environ['OPENAI_API_KEY']
if 'RAG_CONTEXT_PREFIX' in os.environ: RAG_CONTEXT_PREFIX = os.environ['RAG_CONTEXT_PREFIX']

# Implement a protocol-terminating proxy for the OpenAI API:
# This enables us to add a context to the prompt before it is sent to the OpenAI API using RAG
@app.route('/v1/chat/completions', methods=['GET', 'POST', 'OPTIONS'])
def proxy():
    contextlog = ""
    if request.method == 'OPTIONS':
        response = make_response()
        return response
    elif request.method in ['GET', 'POST']:

        # read the content from the users request that was sent out with chat.html
        body = request.get_json()
        messages = body['messages']
            
        if SYSTEM and os.path.isdir(SYSTEM_CONTEXT_PATH):
            # read the content from the users request that was sent out with chat.html
            systemmessage = messages[0]['content']
            # read the large context
            largetext = ""
            for filename in os.listdir(SYSTEM_CONTEXT_PATH):
                if filename.endswith(".txt") or filename.endswith(".md"):
                    with open(os.path.join(SYSTEM_CONTEXT_PATH, filename), 'r') as f: largetext += f.read() + '\n\n'
            # patch the system message
            messages[0]['content'] = systemmessage + '\n\n' + SYSTEM_CONTEXT_PREFIX + '\n\n' + largetext
            print(systemmessage)
            # todo: set n_keep

        if RAG:
            # get the content object from the latest message: this is the current prompt
            prompt = messages[-1]['content']

            # search for a RAG document that matches the prompt
            try:
                searchresult = requests.post(YACYSEARCH_HOST + '/yacysearch.json', json={'query': prompt, 'count': 6})
                context = ""
                if searchresult.status_code == 200:
                    try:
                        # parse the result to json
                        searchresult_json = json.loads(searchresult.text)
                        items = searchresult_json['channels'][0]['items']
                        print(items)
                        # number of results
                        hitcount = len(items)
                        minimum_distance = 70

                        for i in range(0, min(hitcount, 5)):
                            if items[i]['distance'] < minimum_distance - (i * 10):
                                description = items[i]['description']
                                title = items[0]['title']
                                title = title[0] if isinstance(title, list) and len(title) > 0 else str(title)
                                if len(description) > 600: description = description[:600] + "..."
                                context = context + '\n\n' + description
                                contextlog = contextlog + "&diams; added RAG document: " + title + "\n"
                    
                        print(context)
                        # replace the content object with the enriched content in the body object
                    except json.decoder.JSONDecodeError as e:
                        # Handle invalid JSON response from search API
                        print(f"Error: Search API returned invalid JSON: {e}")

                    # add the new context to the prompt if the context length is > 0
                    if len(context) > 0:
                        prompt = prompt + '\n\n' + RAG_CONTEXT_PREFIX + context

                # write back the enriched prompt to the body object
                messages[-1]['content'] = prompt
            except requests.exceptions.ConnectionError as e:
                # Handle connection error to search API
                print(f"Error: Search API connection error: {e}")

            # get the body object back into a string
            body['model'] = "gpt-3.5-turbo"
            #body['type'] = "json_object"
            body['stream'] = True # this return format is stream only
        
        encoded_body = json.dumps(body).encode('utf-8') 
        print(encoded_body)

        # Create a connection and send the request
        api_url = urlparse(OPENAI_API_HOST + "/v1/chat/completions")

        #depending on the api_url, make a http or https connection
        if api_url.scheme == "https":
            conn = http.client.HTTPSConnection(api_url.netloc)
        else:
            conn = http.client.HTTPConnection(api_url.netloc)
        headers = {key: value for (key, value) in request.headers if key != 'Host'}
        headers['Content-Type'] = 'application/json'
        headers['Content-Length'] = str(len(encoded_body)) # content length must be re-computed because it might have changed
        if OPENAI_API_KEY != "": headers['Authorization'] = 'Bearer ' + OPENAI_API_KEY
        conn.request("POST", api_url.path, encoded_body, headers)
        #print("sent request to OpenAI API")
        
        # Thread function for reading from the server
        def read_from_upstream():
            resp = conn.getresponse()
            while True:
                line = resp.readline()
                if line:
                    t = line.decode('utf-8')
                    #print(t)
                    yield line
                    if t.find("data: [DONE]") != -1: break
                else:
                    break

        return app.response_class(stream_with_context(read_from_upstream()))

    else:
        # Unsupported HTTP method
        return jsonify({'message': 'Method not supported'}), 405

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text using a specific model.")
    parser.add_argument("--model", default="gptneo_oetker_trained", type=str, help="Model name of remote LLM")
    parser.add_argument("--port", default=5001, type=int, help="Port for the reverse proxy server")
    parser.add_argument("--host", default="0.0.0.0", type=str, help="Local host address")
    parser.add_argument("--yacy_search_host", default=YACYSEARCH_HOST, type=str, help="http address of search host")
    parser.add_argument("--openai_api_host", default=OPENAI_API_HOST, type=str, help="http address of ApenAI API")
    parser.add_argument("--openai_api_key", default=OPENAI_API_KEY, type=str, help="OpenAI API key")
    parser.add_argument("--rag", default=False, action='store_true', help="Enable RAG (Retrieval Augmented Generation) feature")
    parser.add_argument("--rag_context_prefix", default=RAG_CONTEXT_PREFIX, type=str, help="Prefix string for RAG context")
    parser.add_argument("--system", default=False, action='store_true', help="Enable large system context prompts")
    parser.add_argument("--system_context_prefix", default=SYSTEM_CONTEXT_PREFIX, type=str, help="Prefix string for large system context")
    parser.add_argument("--system_context_path", default=SYSTEM_CONTEXT_PATH, type=str, help="Path to files to be put into system context")
    args = parser.parse_args()
    YACYSEARCH_HOST = args.yacy_search_host
    OPENAI_API_HOST = args.openai_api_host
    OPENAI_API_KEY = args.openai_api_key
    RAG = args.rag
    RAG_CONTEXT_PREFIX = args.rag_context_prefix
    SYSTEM = args.system
    SYSTEM_CONTEXT_PREFIX = args.system_context_prefix
    SYSTEM_CONTEXT_PATH = args.system_context_path
    app.run(host=args.host, port=args.port, debug=True)
