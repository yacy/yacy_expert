import os
import knowledge_splitter

# run this if you want to reset the knowledge directory
# to the state before the knowledge_splitter and knowledge_indexer were run
if __name__ == "__main__":
    knowledge = knowledge_splitter.knowledge_path()

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
