import os
import re
import json
import expert_common

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
        text = text[split_at:].strip()
    chunks.append(text)

    # if the number of chunks is now larger than num_chunks, then we need to merge the last two chunks
    if len(chunks) > num_chunks:
        chunks[-2] = chunks[-2] + " " + chunks[-1]
        chunks = chunks[:-1]

    return chunks

def split_file(jsonl_file, jsonl_file_out, chunklen=800):

    # read jsonl file and parse it into a list of json objects
    texts_in = expert_common.read_text_list(jsonl_file)

    # in case that the text_list is empty, we just skip this file
    if len(texts_in) == 0:
        print(f"Skipping empty file {jsonl_file}")
        return

    print(f"Read {len(texts_in)} lines from {jsonl_file}")

    # for each of the text lines, we split the text into chunks of max_chars
    # and store the chunks in a new list
    json_objects_out = []
    for text in texts_in:

        # Skip if text_t is not in the line
        if not "text_t" in text: continue

        # parse the json object
        json_object = json.loads(text)

        # get the text_t field
        text = json_object['text_t']

        # get the title field
        title = json_object['title']

        # get the description field
        description = json_object['description']
        if not description or len(description) == 0: description = title

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
            if description not in newtext: newtext = description + "\n" + newtext
            new_json_object['text_t'] = newtext.strip()
            if '#' in url: 
                new_json_object['url_s'] = url + '_' + str(i)
            else:
                new_json_object['url_s'] = url + '#' + str(i)
            new_json_objects.append(new_json_object)

        # store the new_json_objects in the new list
        json_objects_out.extend(new_json_objects)
    
    expert_common.write_jsonlist(json_objects_out, jsonl_file_out)

    # return the number of lines written
    return len(json_objects_out)

# Parses the metadata at the beginning of the markdown file.
def parse_md_metadata(lines):
    metadata = {}
    metadata_lines = []
    inside_metadata = False
    for line in lines:
        if line.strip() == '---':
            if inside_metadata:
                # End of metadata section
                break
            else:
                # Start of metadata section
                inside_metadata = True
        elif inside_metadata:
            metadata_lines.append(line)
    for line in metadata_lines:
        key_value_match = re.match(r"(\w+):\s*(.*)", line)
        if key_value_match:
            key, value = key_value_match.groups()
            metadata[key.lower()] = value.strip()
    return metadata

# parse a markdown file and return a list of chunks for each headline
def parse_md(file_path, jsonl_file_out):
    json_objects_out = []  # List to hold all chunks as dictionaries
    current_hierarchy = []  # To keep track of the hierarchy leading to the current headline
    titles = []
    file_name = os.path.basename(file_path)
    content = expert_common.read_text_list(file_path)

    # Parse metadata if it exists
    metadata = {}
    if content and content[0].strip() == '---':
        metadata = parse_md_metadata(content)
        # Skip past the metadata section to the rest of the content
        content = content[content.index('---', 1) + 1:] if '---' in content else content

    # get the title from the metadata
    title = metadata.get('Title', '').strip()
    if len(title) > 0: titles.append(title)

    linecount = 0
    for line in content:
        if line.startswith('#'):
            line = line.strip()
            level = line.count('#')
            subtitle = line.lstrip('#').strip()
            if level == 1:
                titles.append(subtitle)

            # Adjust the current_hierarchy to match the new level
            if len(current_hierarchy) >= level:
                current_hierarchy = current_hierarchy[:level-1]
            current_hierarchy.append(subtitle)

            # Define the chunk with its description and paragraph number
            description = current_hierarchy[:-1]  # The description is the hierarchy excluding the current title
            descriptions = '\n'.join(description).strip()

            chunk = {
                'url_s': 'file://' + file_name + '#l' + str(linecount),
                'title': titles,
                'description': (descriptions + '\n' + subtitle).strip(),
                'text_t': descriptions + '\n' + subtitle + '\n\n',
                'level': level
            }
            json_objects_out.append(chunk)
        else:
            if json_objects_out:  # Append text to the last chunk
                json_objects_out[-1]['text_t'] += line + '\n'
        linecount += 1
    # write the file
    expert_common.write_jsonlist(json_objects_out, jsonl_file_out)

    # return the number of lines written
    return len(json_objects_out)

def jsonl_filename(md_filename):
    jsonl_file_out = md_filename
    if md_filename.endswith('.md'): return md_filename[:-3] + '.jsonl'
    elif md_filename.endswith('.md.gz'): return md_filename[:-6] + '.jsonl'
    else: return jsonl_file_out

def split_filename(jsonl_in_filename):
    # the new jsonl file name must have the string '.split' right before the suffix
    # with the exception of .gz suffixes, where it is inserted before the suffix before the .gz suffix
    split_file_out = jsonl_in_filename
    if jsonl_in_filename.endswith('.jsonl.gz'):      return jsonl_in_filename[:-9] + '.split.jsonl'
    elif jsonl_in_filename.endswith('.flatjson.gz'): return jsonl_in_filename[:-12] + '.split.flatjson'
    elif jsonl_in_filename.endswith('.jsonl'):       return jsonl_in_filename[:-6] + '.split.jsonl'
    elif jsonl_in_filename.endswith('.flatjson'):    return jsonl_in_filename[:-9] + '.split.flatjson'
    else: return split_file_out + ".split.jsonl"

# Process all .jsonl/.flatjson files
if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    knowledge = expert_common.knowledge_path()

    print(f"Processing directory for indexing: {knowledge}")
    orderedfilelist = expert_common.list_files_by_size(knowledge)
    for file in orderedfilelist:
        print(f"reading: {file}")

        if file.endswith('.md') or file.endswith('.md.gz'):
            # hack to easily test the semantic search with markdown files so we don't need
            # to produce jsonl files from YaCy exports. Also usable for mass-data production from
            # other sources.
            print(f"Converting markdown file to jsonl: {file}")
            
            path_in = os.path.join(knowledge, file)
            print(f"Reading md from: {path_in}")
            path_out = os.path.join(knowledge, jsonl_filename(file))
            print(f"Writing parsed md to jsonl: {path_out}")
            chunks = parse_md(path_in, path_out)
            os.system(f"gzip -9 {path_out}")

            # from here on we may continue to split the jsonl file
            file = path_out + '.gz'
            print(f"New splitting file, resulting from markdown parsing: {file}")

        if  file.endswith('.jsonl') or file.endswith('.jsonl.gz') or \
            file.endswith('.flatjson') or file.endswith('.flatjson.gz'):  # .flatjson is the yacy export format

            # in case that the file name contains "split", skip it
            if "split" in file: continue

            print(f"Splitting file: {file}")
            path_out = split_filename(file)
            print(f"Writing to: {path_out}")
            split_file(file, path_out, 800)

            # we rename the original file by appending '.original' to the file name
            origfile = file + '.original'
            if os.path.exists(origfile): os.remove(origfile)
            os.rename(file, origfile)
            
            # gzip the new file new_filename
            gzipfile = path_out + '.gz'
            if os.path.exists(gzipfile): os.remove(gzipfile)
            os.system(f"gzip -9 {path_out}")
            print(f"Splitting file: {file} done")

            