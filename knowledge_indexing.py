import os
import gzip
import time
import json
import faiss
import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from concurrent.futures import ThreadPoolExecutor
import gzip
import configparser

def knowledge_path():
    # Load all FAISS indexes and index files from the knowledge path
    path = 'knowledge'
    # if the directory_path is empty, try to use the local/parallel yacy export path
    # if the knowledge path is empty or contains one single file '.gitignore', use the local/parallel yacy export path
    if not path or (len(os.listdir(path)) == 1 and os.listdir(path)[0] == '.gitignore'):
        path = '../yacy_search_server/DATA/EXPORT/'
    return path

# Function to embed a text using BERT
# An embedding is a vector of size 768
def embedding(text, tokenizer, model):
    # make downcase of given text; the model is trained on lowercased text
    text = text.lower()
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    # Extract the embeddings
    with torch.no_grad(): outputs = model(**inputs) # hidden states
    # Use the average of the last hidden states as the embedding vector
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    #embeddings_with_index = np.insert(embeddings, 0, index)
    return embeddings


def read_text_list(jsonl_file):
    # This reads a YaCy jsonl/flatjson file that was exported for a elasticsearch bulk import
    # Because a elasticsearch bulk file has a header line with {"index":{}} for each record
    # we need to skip those lines.
    # This function returns only the lines that are valid json.
    # We expect that all json objects have a 'text_t' field that contains the text to be indexed.
    lines = []
    print(f"Starting to read {jsonl_file}")

    def read(file):
        line_count = 0
        start_time = time.time()
        for line in file:
            if line.startswith('{"index":'): continue # if line starts with {"index":{}} skip it
            if 'text_t' not in line: continue # if line does not contain 'text_t', skip it
            lines.append(line)
            line_count += 1

            # Logging progress at regular intervals, e.g., every 100,000 lines
            if line_count % 100000 == 0:
                elapsed_time = time.time() - start_time
                print(f"Read {line_count} lines in {elapsed_time:.2f} seconds")
        print(f"Finished reading {line_count} valid document lines from {jsonl_file}")

    if jsonl_file.endswith('.gz'):
        with gzip.open(jsonl_file, 'rt', encoding='utf-8') as file: read(file)
    else:
        with open(jsonl_file, 'r', encoding='utf-8') as file: read(file)

    return lines

def parse_json_lines(lines, batch_size=1000):
    def parse_batch(batch):
        json_records = []
        for line in batch:
            try:
                record = json.loads(line)
                # the json object should have a 'text_t' field that contains the text to be indexed
                json_records.append(record)
            except json.JSONDecodeError:
                pass  # Optionally log parse errors or invalid lines
        return json_records

    print(f"Parsing {len(lines)} lines in batches of {batch_size}")
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(parse_batch, lines[i:i + batch_size]) for i in range(0, len(lines), batch_size)]
        # this concurrency is needed for very large files; it preserves the order of the lines (important!)
        return [record for future in futures for record in future.result()]

def load_ini(ini_file):
    print(f"Loading ini file: {ini_file}")
    if os.path.exists(ini_file):
        with open(ini_file, 'r', encoding='utf-8') as file:
            config = configparser.ConfigParser()
            config.read(ini_file)
            print(f"Loaded ini file: {ini_file}")

            if 'DEFAULT' in config:
                ini = config['DEFAULT']
            else:
                ini = {}
            if 'dimension' in ini:
                dimension = ini['dimension']
            else:
                dimension = 768
            print(f"dimension: {dimension}")
            if 'model_name' in ini:
                model_name = ini['model_name']
            else:
                model_name = "bert-base-multilingual-uncased"
            print(f"model_name: {model_name}")  
    else:
        # model_name = "dbmdz/bert-base-german-uncased"
        model_name = "bert-base-multilingual-uncased"
        dimension = 768

    return model_name, dimension

def process_file(jsonl_file):
    # this function reads a YaCy export file and creates a FAISS index file.
    if jsonl_file.endswith('.gz'):
        faiss_index_file = jsonl_file[:-3] + '.faiss'
        faiss_ini_file = jsonl_file[:-3] + '.ini'
    else:
        faiss_index_file = jsonl_file + '.faiss'
        faiss_ini_file = jsonl_file + '.ini'

    if os.path.exists(faiss_index_file):
        print(f"FAISS index for {jsonl_file} already exists. Skipping.")
        return
    
    # load the ini file if it exists
    model_name, dimension = load_ini(faiss_ini_file)

    # Load a pre-trained model tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # read jsonl file and parse it into a list of json objects
    text_list = read_text_list(jsonl_file)
    print(f"Read {len(text_list)} lines from {jsonl_file}")
    json_records = parse_json_lines(text_list)
    print(f"Parsed {len(json_records)} json objects from {jsonl_file}")

    # concurrent embedding computation
    start_time = time.time()
    print(f"Starting to compute embeddings for {len(json_records)} records")
    with ThreadPoolExecutor() as executor:
        futures = []
        for i in range(0, len(json_records)):
            record = json_records[i]
            record_text = record['text_t']
            future = executor.submit(embedding, record_text, tokenizer, model)
            futures.append(future)

            # Log progress every 100 lines
            if (i+1) % 10000 == 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed / (i+1) * len(json_records)
                remaining = estimated_total - elapsed
                print(f"Submitted {i+1}/{len(json_records)} records to concurrent executor. Estimated time remaining: {remaining/60:.2f} minutes.")

        # wait for all futures to finish
        vectors = []
        start_time = time.time()
        print(f"Waiting for {len(futures)} futures to finish")
        for i in range(0, len(futures)):
            future = futures[i]
            vector = future.result()
            vectors.append(vector)

            # Log progress every 100 lines
            if (i+1) % 100 == 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed / (i+1) * len(json_records)
                remaining = estimated_total - elapsed
                print(f"Computed {i+1}/{len(json_records)} embeddings. Estimated time remaining: {remaining/60:.2f} minutes.")

    print(f"Finished computing embeddings for {len(json_records)} records, computing FAISS index")

    # Convert list of vectors to a FAISS compatible format
    vectors = np.array(vectors).astype('float32')
    # Create a FAISS index
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(vectors)  # Add vectors to the index

    # Save the index to a file
    faiss.write_index(faiss_index, jsonl_file + '.faiss')
    print(f"Finished and saved FAISS index to {jsonl_file + '.faiss'}")

# Process all .jsonl/.flatjson files
if __name__ == "__main__":
    knowledge = knowledge_path()

    print(f"Processing directory: {knowledge}")
    for file in os.listdir(knowledge):
        if file.endswith('.jsonl') or file.endswith('.flatjson') or file.endswith('.jsonl.gz') or file.endswith('.flatjson.gz'): # .flatjson is the yacy export format
            print(f"Processing file: {file}")
            path = os.path.join(knowledge, file)

            # run the indexing process
            process_file(path)
