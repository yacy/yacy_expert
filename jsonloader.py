# jsonloader, a fast, parallel JSON loader for jsonl files (lists of JSON objects)
# (C) 2023 by Michael Christen
# License: Apache License 2.0

import os
import gzip
import json
import logging
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Parse a single JSON line and return the parsed document.
def parse_json_line(line):
    line = line.strip()
    if not line:
        return None  # Skip empty lines
    if line.startswith('{"index"'):
        return None  # Skip index lines
    try:
        doc = json.loads(line)
        if "index" in doc:
            return None  # Skip documents with "index" field
        return doc
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse JSON line: {line}. Error: {e}")
        return None

# Load and parse JSON documents from a file in parallel.
def load(filepath):
    json_data = []
    
    # Check if the file exists
    if not os.path.exists(filepath):
        logging.error(f"File not found: {filepath}")
        return json_data  # Return an empty list

    # Read all lines from the file
    try:
        if filepath.endswith(".gz"):
            with gzip.open(filepath, "rt") as f:
                lines = f.readlines()
        else:
            with open(filepath, "r") as f:
                lines = f.readlines()
    except Exception as e:
        logging.error(f"Failed to read file: {filepath}. Error: {e}")
        return json_data

    if not lines:
        logging.warning(f"File is empty: {filepath}")
        return json_data

    # Use ThreadPoolExecutor to parse JSON lines in parallel
    with ThreadPoolExecutor() as executor:
        # Submit tasks to parse each line
        futures = [executor.submit(parse_json_line, line) for line in lines]
        
        # Collect results from completed tasks
        for future in futures:
            try:
                doc = future.result()
                if doc is not None:
                    json_data.append(doc)
            except Exception as e:
                logging.warning(f"Failed to process a line. Error: {e}")

    logging.info(f"Parsed {len(json_data)} documents from {filepath}.")
    return json_data

if __name__ == '__main__':
    # test
    knowledge_folder = "knowledge"  # Folder containing JSON documents
    json_objects = []  # List of parsed JSON objects
    for filename in os.listdir(knowledge_folder):
        if filename.endswith(".jsonl") or filename.endswith(".flatjson") or filename.endswith(".jsonl.gz") or filename.endswith(".flatjson.gz"):
            filepath = os.path.join(knowledge_folder, filename)
            logging.info(f"Reading index dump from {filepath}.")
            json_data = load(filepath)
            json_objects.extend(json_data)

    logging.info(f"loaded files: {len(json_objects)} objects")