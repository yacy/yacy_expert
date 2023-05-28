import os
import json
import time
import gzip


def knowledge_path():
    # Load all FAISS indexes and index files from the knowledge path
    path = 'knowledge'
    # if the directory_path is empty, try to use the local/parallel yacy export path
    # if the knowledge path is empty or contains one single file '.gitignore', use the local/parallel yacy export path
    if not path or (len(os.listdir(path)) == 1 and os.listdir(path)[0] == '.gitignore'):
        path = '../yacy_search_server/DATA/EXPORT/'
    return path

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

# split the text into chunks of max_chars; return a list of chunks
# the maximum number of characters per chunk in source text
# this can be computed by the average chars per token
# times the maximum number of tokens for embedding computation
# i.e. gpt2: 2.81 * 1024 = 2876.16
# i.e. bert: 3.63 * 512 = 1859.56
def split_text(text, max_chars):
    
    # compute number of chunks
    num_chunks = len(text) // max_chars + 1

    # now that the number of chunks is known, we can compute an optimal chunk size which is smaller than max_chars
    max_chars_here = len(text) // num_chunks + 10 # add 10 to get space for space searching method
    
    # split the text into chunks of max_chars_here maximum length
    chunks = []
    # if the text length is smaller than max_chars_here, then we just return the text in one vector element
    if len(text) <= max_chars_here:
        return [text]

    # otherwise we split the text into chunks of max_chars_here maximum length
    while len(text) > max_chars_here:
        split_at = text.rfind(' ', 0, max_chars_here) # find the last space before max_chars_here
        if split_at == -1: split_at = max_chars_here
        chunks.append(text[:split_at])
        text = text[split_at:]
    chunks.append(text)

    # if the number of chunks is now larger than num_chunks, then we need to merge the last two chunks
    if len(chunks) > num_chunks:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks = chunks[:-1]

    return chunks

def split_file(jsonl_file, jsonl_file_out, chunklen=800):

    # read jsonl file and parse it into a list of json objects
    texts_in = read_text_list(jsonl_file)

    # in case that the text_list is empty, we just skip this file
    if len(texts_in) == 0:
        print(f"Skipping empty file {jsonl_file}")
        return

    print(f"Read {len(texts_in)} lines from {jsonl_file}")

    # for each of the text lines, we split the text into chunks of max_chars
    # and store the chunks in a new list
    texts_out = []
    for text in texts_in:
        # parse the json object
        json_object = json.loads(text)

        # check if text_t field exists
        if 'text_t' not in json_object: continue

        # get the text_t field
        text = json_object['text_t']

        # get the title field
        title = json_object['title']

        # in case that title is a list, we just take the first element
        if isinstance(title, list): title = title[0]

        # split the text into chunks
        chunks = split_text(text, chunklen // 2)

        # We combine two successive chunk each as one longchunk; that means we produce a 50% overlap.
        # The resulting number of longchunks is len(chunks) - 1
        longchunks = []
        for i in range(len(chunks) - 1):
            longchunks.append(chunks[i] + " " + chunks[i+1])

        # get original url from the json_object
        # the url field is either named "sku" or "url_s"
        if 'sku' in json_object:
            url = json_object['sku']
        else:
            url = json_object['url_s']

        # remove the original text_t field from the json_object
        del json_object['text_t']
        if 'sku'   in json_object: del json_object['sku']   # old name for url field
        if 'url_s' in json_object: del json_object['url_s'] # new name for url field

        # we use the json_object as template for new json objects
        # which are constructed in the following way:
        # - we create a clone of the json_object for each longchunk without the text_t field
        # - the text_t field is replaced with one of the longchunks, longchunk[i]
        # - the url field is replaced with the original url + '#' + str(i)
        new_json_objects = []
        for i in range(len(longchunks)):
            new_json_object = json_object.copy()
            newtext = longchunks[i]
            if title not in newtext: newtext = title + " " + newtext
            new_json_object['text_t'] = newtext.strip()
            new_json_object['url_s'] = url + '#' + str(i)
            new_json_objects.append(new_json_object)

        # store the new_json_objects in the new list
        texts_out.extend(new_json_objects)
    
    print(f"Writing {len(texts_out)} lines to {jsonl_file_out}")
    with open(jsonl_file_out, 'w', encoding='utf-8') as file:
        for text in texts_out:
            file.write(json.dumps(text, ensure_ascii=False) + '\n')

    # return the number of lines written
    return len(texts_out)

def new_filename(old_filename):
    # the new jsonl file name must have the string '.split' right before the suffix
    # with the exception of .gz suffixes, where it is inserted before the suffix before the .gz suffix
    jsonl_file_out = old_filename
    if old_filename.endswith('.jsonl.gz'):
        jsonl_file_out = old_filename[:-9] + '.split.jsonl'
    elif old_filename.endswith('.flatjson.gz'):
        jsonl_file_out = old_filename[:-12] + '.split.flatjson'
    elif old_filename.endswith('.jsonl'):
        jsonl_file_out = old_filename[:-6] + '.split.jsonl'
    elif old_filename.endswith('.flatjson'):
        jsonl_file_out = old_filename[:-9] + '.split.flatjson'
    else:
        return jsonl_file_out + ".split.jsonl"
    return jsonl_file_out

# Process all .jsonl/.flatjson files
if __name__ == "__main__":
    knowledge = knowledge_path()

    print(f"Processing directory for indexing: {knowledge}")
    for file in os.listdir(knowledge):
        if  file.endswith('.jsonl') or file.endswith('.jsonl.gz') or \
            file.endswith('.flatjson') or file.endswith('.flatjson.gz'):  # .flatjson is the yacy export format

            # in case that the file name contains "split", skip it
            if "split" in file: continue

            print(f"Splitting file: {file}")
            path = os.path.join(knowledge, file)

            # run the indexing process
            new_path = os.path.join(knowledge, new_filename(file))
            split_file(path, new_path, 800)

            # we rename the original file by appending '.original' to the file name
            os.rename(path, path + '.original')
            
            # gzip the new file new_filename
            os.system(f"gzip -9 {new_path}")

