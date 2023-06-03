import os
import expert_common

# run this if you want to reset the knowledge directory
# to the state before the knowledge_splitter and knowledge_indexer were run
if __name__ == "__main__":
    knowledge = expert_common.knowledge_path()

    print(f"Resetting all processing steps in: {knowledge}")

    # delete all .faiss files in the knowledge directory
    count = 0
    for file in os.listdir(knowledge):
        if file.endswith('.faiss'):
            os.remove(os.path.join(knowledge, file))
            count += 1
    print(f"Deleted {count} .faiss files")

    # delete all files containing 'split' in the name
    count = 0
    for file in os.listdir(knowledge):
        if 'split' in file:
            os.remove(os.path.join(knowledge, file))
            count += 1
    print(f"Deleted {count} split files")

    # rename all .original files to their original name
    count = 0
    for file in os.listdir(knowledge):
        if file.endswith('.original'):
            os.rename(os.path.join(knowledge, file), os.path.join(knowledge, file[:-9]))
            count += 1
    print(f"Renamed {count} .original files")

    # for all files ending with .md or .md.gz remove the corresponding .jsonl or jsonl.gz file
    count = 0
    for file in os.listdir(knowledge):
        if file.endswith('.md'):
            f   = os.path.join(knowledge, file[:-3] + '.jsonl')
            fgz = os.path.join(knowledge, file[:-3] + '.jsonl.gz')
            if os.path.exists(f):
                os.remove(f)
                count += 1
            if os.path.exists(fgz):
                os.remove(fgz)
                count += 1
        if file.endswith('.md.gz'):
            f   = os.path.join(knowledge, file[:-6] + '.jsonl')
            fgz = os.path.join(knowledge, file[:-6] + '.jsonl.gz')
            if os.path.exists(f):
                os.remove(f)
                count += 1
            if os.path.exists(fgz):
                os.remove(fgz)
                count += 1
    print(f"Deleted {count} .jsonl files for markdown sources")

    print("To run all processing steps to make a search index, do:")
    print("- run knowledge_dedup.py to remove double index entries")
    print("- run knowledge_splitter.py to separate large chunks of documents into smaller ones")
    print("- run knowledge_indexing.py to generate .faiss index files")
