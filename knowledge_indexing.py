import os
import gzip
import time
import json
import faiss
import torch
import numpy as np
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
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
# An embedding is a vector of size 768 (mostly)
# we do not compute several batches at once because we use CPU using concurrency which creates better throughput
def embeddingBERT(text, tokenizer, model, max_length):
    # Tokenize the text
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    #print(f"tokens.input_ids.shape: {tokens.input_ids.shape}") # torch.Size([1, 438])  .. up to 512

    # compute the hidden states from the BERT model without computing the gradients
    with torch.no_grad(): outputs = model(**tokens) # **tokens is a dictionary of all tokens
    #print(f"outputs.last_hidden_state.shape: {outputs.last_hidden_state.shape}") # torch.Size([1, 438, 768]) .. up to 512
    #print(f"outputs.last_hidden_state.mean(dim=1).shape: {outputs.last_hidden_state.mean(dim=1).shape}") # torch.Size([1, 768])
    #print(f"outputs.last_hidden_state.mean(dim=1).squeeze().shape: {outputs.last_hidden_state.mean(dim=1).squeeze().shape}") # torch.Size([768])

    # Use the average of the last hidden states as the embedding vector

    # The last hidden state of the tensor is a tensor of size [1, 438, 768] (or 512 depending on the max_length).
    # To get a single vector of size 768, we need to compute the mean of the last hidden states,
    # which is independent from the length of the text; this reduces the dimension of the tensor to [1, 768].
    # squeeze() removes the first dimension of the tensor, creating a vector of size 768.
    embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    # return the embedding vector
    return embeddings

# Function to embed a text using GPT2
def embeddingGPT2(text, tokenizer, model, max_length):
    # Tokenize text
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    # print(f"tokens.input_ids.shape: {tokens.input_ids.shape}") # torch.Size([1, 876])

    # Ensure the model returns hidden states
    model.config.output_hidden_states = True

    # Compute the hidden states without computing the gradients
    with torch.no_grad(): outputs = model(**tokens)

    # Extract the last layer's hidden states
    hidden_states = outputs.hidden_states[-1]
    # print(f"hidden_states.shape: {hidden_states.shape}") # hidden_states.shape: torch.Size([1, 876, 768])

    # In GPT-2, since the model is unidirectional, averaging token embeddings doesn't always provide a meaningful
    # representation of the sequence. Instead, the representation of the first token (especially in the last layer)
    # is used, as it is the most "informed" token, having seen all other tokens
    # in the sequence during the forward pass of the model.
    embeddings = hidden_states[:, 0, :].squeeze().numpy()
    # print(f"embeddings.shape: {embeddings.shape}") # embeddings.shape: (768,)

    return embeddings

def embedding(text, model_name, tokenizer, model, max_sequence_length):
    # if model_name contains "uncased", then we need to downcase the text
    if "uncased" in model_name: text = text.lower()

    # get specific embedding for the model
    if model_name.startswith('gpt2'):
        return embeddingGPT2(text, tokenizer, model, max_sequence_length)
    else:
        return embeddingBERT(text, tokenizer, model, max_sequence_length)

def read_text_list(jsonl_file):
    # This reads a YaCy jsonl/flatjson file that was exported for a elasticsearch bulk import
    # Because a elasticsearch bulk file has a header line with {"index":{}} for each record
    # we need to skip those lines.
    # This function returns only the lines that are valid json.
    # We expect that all json objects have a 'text_t' field that contains the text to be indexed.
    lines = []

    def read(file):
        line_count = 0
        start_time = time.time()
        for line in file:
            # alternating every second line the json object must either:
            # - start with {"index":{}} or
            # - contain a 'text_t' field
            if line.startswith('{"index":'): continue # if line starts with {"index":{}} skip it
            if 'text_t' not in line: continue # if line does not contain 'text_t', skip

            lines.append(line)
            line_count += 1

            # Logging progress at regular intervals, e.g., every 100,000 lines
            if line_count % 100000 == 0:
                elapsed_time = time.time() - start_time
                print(f"Read {line_count} lines in {elapsed_time:.2f} seconds")

    if os.path.exists(jsonl_file):
        if jsonl_file.endswith('.gz'):
            with gzip.open(jsonl_file, 'rt', encoding='utf-8') as file: read(file)
        else:
            with open(jsonl_file, 'r', encoding='utf-8') as file: read(file)

    return lines

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
        # choose one from https://huggingface.co/transformers/v4.12.0/pretrained_models.html
        #model_name = "bert-base-german-dbmdz-cased" # do not uncomment, write the name into a ini file instead
        model_name = "bert-base-multilingual-cased"
        #model_name = "gpt2"
        dimension = 768

    return model_name, dimension

def get_faiss_and_ini_file(jsonl_file):
    # this function returns the faiss index file and the ini file for a given jsonl file
    if jsonl_file.endswith('.gz'):
        faiss_index_file = jsonl_file[:-3] + '.faiss'
        faiss_ini_file = jsonl_file[:-3] + '.ini'
    else:
        faiss_index_file = jsonl_file + '.faiss'
        faiss_ini_file = jsonl_file + '.ini'
    return faiss_index_file, faiss_ini_file

def tokenizer_model_from_name(model_name):
    # Load a pre-trained model tokenizer and model
    # see full list at https://huggingface.co/transformers/v4.12.0/pretrained_models.html
    if model_name.startswith('gpt2'): # i.e. gpt2
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    return tokenizer, model

def index_file(jsonl_file):
    # this function reads a YaCy export file and creates a FAISS index file.
    faiss_index_file, faiss_ini_file = get_faiss_and_ini_file(jsonl_file)

    if os.path.exists(faiss_index_file):
        print(f"FAISS index for {jsonl_file} already exists. Skipping.")
        return
    
    # load the ini file if it exists
    model_name, dimension = load_ini(faiss_ini_file)
    dimension = int(dimension)
    print(f"Creating FAISS index for {jsonl_file} with model {model_name} and dimension {dimension}")
    tokenizer, model = tokenizer_model_from_name(model_name)

    # compute the dimension of the model
    if model_name.startswith('gpt'): # i.e. gpt2
        dimensionc = model.config.n_embd
    else:
        dimensionc = model.config.hidden_size

    # compare the dimension of the model with the dimension from the ini file
    assert dimension == dimensionc, f"Error: dimension {dimension} from ini file {faiss_ini_file} does not match dimension {dimensionc} of model {model_name}"

    # get the maximum token length for the model
    max_sequence_length = model.config.max_position_embeddings
    print(f"max_sequence_length: {max_sequence_length}")

    # read jsonl file and parse it into a list of json objects
    text_list = read_text_list(jsonl_file)

    # in case that the text_list is empty, we just skip this file
    if len(text_list) == 0:
        print(f"Skipping empty file {jsonl_file}")
        return

    print(f"Read {len(text_list)} lines from {jsonl_file}")

    # concurrent embedding computation
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for i in range(0, len(text_list)):
            text_line = text_list[i]
            # parse the json line
            try:
                record = json.loads(text_line)
            except json.JSONDecodeError:
                 # this makes the file unusable for the FAISS index becuase the FAISS index would not match the line number any more
                print(f"Error parsing json line: {text_line}")
                continue # we just continue here to make identification of more errors possible. it would be correct to fail and exit here.
            record_text = record['text_t']
            future = executor.submit(embedding, record_text, model_name, tokenizer, model, max_sequence_length)
            futures.append(future)

            # Log progress every 100 lines
            if (i+1) % 10000 == 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed / (i+1) * len(text_list)
                remaining = estimated_total - elapsed
                print(f"Submitted {i+1}/{len(text_list)} records to concurrent executor. Estimated time remaining: {remaining/60:.2f} minutes.")

        # wait for all futures to finish
        vectors = []
        start_time = time.time()
        print(f"Waiting for {len(futures)} futures to finish, please be patient.")
        for i in range(0, len(futures)):
            future = futures[i]
            vector = future.result()
            vectors.append(vector)

            # Log progress every 100 lines
            if (i+1) % 100 == 0:
                elapsed = time.time() - start_time
                estimated_total = elapsed / (i+1) * len(futures)
                remaining = estimated_total - elapsed
                print(f"Computed {i+1}/{len(futures)} embeddings. Estimated time remaining: {remaining/60:.2f} minutes;  {(i+1)/elapsed*60:.2f} embeddings per minute.")

    print(f"Finished computing embeddings for {len(futures)} records, computing FAISS index")

    # Convert list of vectors to a FAISS compatible format
    vectors = np.array(vectors).astype('float32')

    # Check the dimension of the model's output vector
    vector_example = vectors[0]
    print(f"Dimension of the model's output vector: {vector_example.shape[0]}")
    print(f"Dimension expected by FAISS index: {dimension}")
    #Dimension of the model's output vector: 50257
    #Dimension expected by FAISS index: 768
    # Ensure they match
    assert vector_example.shape[0] == dimension, "Model output dimension does not match FAISS index dimension"

    # Add vectors to the index
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(vectors)  

    # Save the index to a file
    faiss.write_index(faiss_index, faiss_index_file)
    print(f"Finished and saved FAISS index to {faiss_index_file}")

# Process all .jsonl/.flatjson files
if __name__ == "__main__":
    knowledge = knowledge_path()

    print(f"Processing directory for indexing: {knowledge}")
    for file in os.listdir(knowledge):
        if  file.endswith('.jsonl') or file.endswith('.jsonl.gz') or \
            file.endswith('.flatjson') or file.endswith('.flatjson.gz'):  # .flatjson is the yacy export format
            print(f"Indexing file: {file}")
            path = os.path.join(knowledge, file)

            # run the indexing process
            index_file(path)
