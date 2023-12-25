import os
import gzip
import time
import json
import faiss
import torch
import numpy as np
from transformers import BertModel, BertTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
import gzip
import configparser
import expert_common
import ssl

# predefine dictionary for each language model one number which holds the number of characters per token
chars_per_token = {
    "bert-base-multilingual-cased": 3.63, # with max_sequence_length = 512; measured on the german wikipedia
    "bert-base-german-dbmdz-cased": 5.5, # guessed by copilot
    "gpt2": 1.0, # guessed by copilot
}

stat_token_count = 0
stat_text_length = 0

# Function to embed a text using BERT
# An embedding is a vector of size 768 (mostly)
# we do not compute several batches at once because we use CPU using concurrency which creates better throughput
def embeddingBERT(text, tokenizer, model, max_length):
    # Tokenize the text
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    #print(f"tokens.input_ids.shape: {tokens.input_ids.shape}") # torch.Size([1, 438])  .. up to 512

    # collect statistics about the token count and text length in case the max_length is not reached
    token_count = len(tokens.input_ids[0])
    if token_count > 0 and token_count < max_length :
        global stat_token_count
        stat_token_count += len(tokens.input_ids[0])
        global stat_text_length
        stat_text_length += len(text)

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

def embeddingSBERT(text, tokenizer, model, max_length): 
    model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
    embedding = model.encode([text]).astype('float32')
    return embedding

# Function to embed a text using GPT2
def embeddingGPT2(text, tokenizer, model, max_length):
    # Tokenize text
    tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=max_length)
    # print(f"tokens.input_ids.shape: {tokens.input_ids.shape}") # torch.Size([1, 876])

    # collect statistics about the token count and text length in case the max_length is not reached
    token_count = len(tokens.input_ids[0])
    if token_count > 0 and token_count < max_length :
        global stat_token_count
        stat_token_count += len(tokens.input_ids[0])
        global stat_text_length
        stat_text_length += len(text)

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
    elif model_name == 'distiluse-base-multilingual-cased-v1':
        return embeddingSBERT(text, tokenizer, model, max_sequence_length)
    else:
        return embeddingBERT(text, tokenizer, model, max_sequence_length)

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
            if 'model_name' in ini:
                model_name = ini['model_name']
            else:
                model_name = "bert-base-multilingual-cased"
            print(f"model_name: {model_name}")  
    else:
        # choose one from https://huggingface.co/transformers/v4.12.0/pretrained_models.html
        #model_name = "bert-base-german-dbmdz-cased" # do not uncomment, write the name into a ini file instead
        #model_name = "bert-base-multilingual-cased"
        #model_name = "distiluse-base-multilingual-cased-v1"
        #model_name = "gpt2"
        model_name = "bert-base-multilingual-cased"

    return model_name

def get_faiss(jsonl_file):
    # this function returns the faiss index file and the ini file for a given jsonl file
    if jsonl_file.endswith('.gz'):
        faiss_index_file = jsonl_file[:-3] + '.faiss'
    else:
        faiss_index_file = jsonl_file + '.faiss'
    return faiss_index_file

def tokenizer_model_from_name(model_name):
    # Load a pre-trained model tokenizer and model
    # see full list at https://huggingface.co/transformers/v4.12.0/pretrained_models.html
    if model_name.startswith('gpt2'): # i.e. gpt2
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        model = GPT2LMHeadModel.from_pretrained(model_name)
    elif model_name == 'distiluse-base-multilingual-cased-v1':
        model = SentenceTransformer('distiluse-base-multilingual-cased-v1')
        tokenizer = model.tokenizer
    else:
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name)
    return tokenizer, model

def index_file(jsonl_file):
    # this function reads a YaCy export file and creates a FAISS index file.
    faiss_index_file = get_faiss(jsonl_file)

    if os.path.exists(faiss_index_file):
        print(f"FAISS index for {jsonl_file} already exists. Skipping.")
        return

    # get the maximum token length for the model
    if model_name == 'distiluse-base-multilingual-cased-v1':
        max_sequence_length = 512
    else:
        max_sequence_length = model.config.max_position_embeddings
    print(f"max_sequence_length: {max_sequence_length}")

    # read jsonl file and parse it into a list of json objects
    text_list = expert_common.read_text_list(jsonl_file)

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
            if not "text_t" in text_line: continue # Skip if text_t is not in the line
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
                global stat_token_count
                global stat_text_length
                if stat_token_count > 0:
                    chars_per_token = stat_text_length/stat_token_count
                else:
                    chars_per_token = 0
                print(f"Computed {i+1}/{len(futures)} embeddings. Time remaining: {remaining/60:.2f} minutes; {(i+1)/elapsed*60:.2f} embeddings per minute, chars per token: {chars_per_token:.2f}")

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
    # this is needed to avoid an SSL error when downloading the model; the problem usually only occurs behind ssl-terminating proxies
    ssl._create_default_https_context = ssl._create_unverified_context

    # get the knowledge path
    knowledge = expert_common.knowledge_path()

    # load ini file if it exists
    model_name = load_ini(os.path.join(knowledge, 'knowledge.ini'))

    # load tokenizer and model
    global tokenizer
    global model
    tokenizer, model = tokenizer_model_from_name(model_name)
    global dimension
    if model_name.startswith('gpt'): # i.e. gpt2
        dimension = model.config.n_embd
    elif model_name == 'distiluse-base-multilingual-cased-v1':
        dimension = 512
    else:
        dimension = model.config.hidden_size

    print(f"Processing directory for indexing: {knowledge}")
    orderedfilelist = expert_common.list_files_by_size(knowledge)
    for file in orderedfilelist:
        if  file.endswith('.jsonl') or file.endswith('.jsonl.gz') or \
            file.endswith('.flatjson') or file.endswith('.flatjson.gz'):  # .flatjson is the yacy export format
            print(f"Indexing file: {file}")
            path = os.path.join(knowledge, file)

            # run the indexing process
            index_file(path)
