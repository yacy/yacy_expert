import os
import time
import gzip
import json

# Common functions for the expert system

def knowledge_path():
    # Load all FAISS indexes and index files from the knowledge path
    path = 'knowledge'
    # if the directory_path is empty, try to use the local/parallel yacy export path
    # if the knowledge path is empty or contains one single file '.gitignore', use the local/parallel yacy export path
    if not path or (len(os.listdir(path)) == 1 and os.listdir(path)[0] == '.gitignore'):
        path = '../yacy_search_server/DATA/EXPORT/'
    return path

# given a knowledge path, list all files in increasing size
def list_files_by_size(path):
    files = []
    count = 0

    # create a list of tuples (size, path)
    # to prevent that two files with the same size cause that we miss one file
    for file in os.listdir(path):
        fpath = os.path.join(path, file)
        size = os.path.getsize(fpath)
        files.append((size * 1000 + count, file))
        count += 1

    # sort the list of tuples by size
    files.sort(key=lambda x: x[0])

    # return only the file names
    return [file[1] for file in files]


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

def write_jsonlist(json_objects_out, jsonl_file_out):
    print(f"Writing {len(json_objects_out)} lines to {jsonl_file_out}")
    with open(jsonl_file_out, 'w', encoding='utf-8') as file:
        for j in json_objects_out:
            file.write(json.dumps(j, ensure_ascii=False) + '\n')
