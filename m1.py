import os
import json
from pathlib import Path
from bs4 import BeautifulSoup
import nltk.tokenize
from nltk.stem import PorterStemmer
import re
from collections import defaultdict
nltk.download('punkt_tab')
## NOTE: NEED TO PIP INSALL LXML


ind_size = 5000
partial_index_directory = os.path.join(os.getcwd(), "partial_index")
complete_index_directory = os.path.join(os.getcwd(), "complete_index")
stemmer = PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

def json_files(path):
    files = []
    path = Path(path)
    for json_file in path.rglob("*.json"):
        print(json_file)
        with json_file.open("r") as f:
            files.append(json.load(f))

    return files

def tokenize(text):
    soup = BeautifulSoup(text, 'lxlm')
    word_dict = defaultdict(lambda : defaultdict(int))
    
    # Categorizing text based on their importance level
    important_text = {
        'title' : soup.title.string if soup.title else '',
        'h1' : ' '.join([h.get_text() for h in soup.find_all('h1')]),
        'h2' : ' '.join([h.get_text() for h in soup.find_all('h1')]),
        'h3' : ' '.join([h.get_text() for h in soup.find_all('h1')]),
        'bold' : ' '.join([b.get_text() for b in soup.find_all(['b', 'string'])]),
        'normal' : soup.get_text()
    }

    for level, content in important_text.items():
        tokens = tokenizer.tokenize(content)
        for token in tokens:
            token = stemmer.stem(token) # Porter Stemming
            word_dict[token][level] += 1
    return word_dict

def index(files):
    index = defaultdict(list)
    counter = 0
    running_count = 0
    part = 1
    offload_count = 0
    # Prompt specifies: The indexer must off load the inverted index hash map from main memory to a partial index on disk at least 3 times during index construction

    for doc in files:
        print(doc['url'])
        content = doc['content']
        tokens = tokenize(content)
        print()
        for word, freq in tokens.items():
            # index_list = [running_count, doc['url'], freq] ## document ID, document URL, frequency of token
            # index_list = [running_count, freq] ## document ID, frequency of token , need to accomadcate by adding a map func for docs to IDs
            index[word].append([running_count, freq])

        counter += 1
        running_count += 1
        print(counter)

        if counter >= ind_size: ## gets called every 10k pages, could lower i think theres like
            index_partial(index, part) ## 50k total ?
            index.clear()
            part += 1
            counter = 0
            offload_count += 1

    if len(index.keys()) != 0:
        index_partial(index, part) ## catches the final indexes

def index_partial(index, part):
    if not os.path.exists(partial_index_directory): ## wait is this supposed to be ran on lab or local? does os.path work for lab
        os.makedirs(partial_index_directory) ## even so need to upload all the files to the repo which is hmmmm
    filename = f"partial_index_part{part}.json"
    file = os.path.join(partial_index_directory, filename)
    with open(file, "w") as f:
        json.dump(index, f)

def index_complete():
    complete_index = defaultdict(list)
    for f in os.listdir(partial_index_directory):
        if f.endswith(".json"):
            with open(os.path.join(partial_index_directory, f), 'r') as f:
                partial_index = json.load(f)
                for word, info in partial_index.items():
                    complete_index[word].extend(info)
    
    return write_complete_index(complete_index)
    # return complete_index
def write_complete_index(complete_index):
    file_splits = {"af": defaultdict(list), "gk": defaultdict(list), "lp": defaultdict(list), "qu": defaultdict(list), "vz": defaultdict(list), "nonalpha": defaultdict(list)}

    for word, info in complete_index.items():
        if word[0] in "abcedf":
            file_splits["af"][word].extend(info)
        elif word[0] in "ghijk":
            file_splits["gk"][word].extend(info)
        elif word[0] in "lmnop":
            file_splits["lp"][word].extend(info)
        elif word[0] in "qrstu":
            file_splits["qu"][word].extend(info)
        elif word[0] in "vwxyz":
            file_splits["vz"][word].extend(info)
        else:
            file_splits["nonalpha"][word].extend(info)
    
    if not os.path.exists(complete_index_directory):
        os.makedirs(complete_index_directory)
    
    for key, info in file_splits.items():
        file = os.path.join(complete_index_directory, f"complete_index_{key}.json")
        with open(file, "w") as f:
            json.dump(info, f)


    print(len(file_splits["af"]), len(file_splits["gk"]), len(file_splits["lp"]), len(file_splits["qu"]), len(file_splits["vz"]), len(file_splits["nonalpha"]))
    total_tokens = len(file_splits["af"]) + len(file_splits["gk"]) + len(file_splits["lp"]) + len(file_splits["qu"]) + len(file_splits["vz"]) + len(file_splits["nonalpha"])
    return total_tokens

def write_files(file, part_complete_index):
    if not os.path.exists(complete_index_directory): ## wait is this supposed to be ran on lab or local? does os.path work for lab
        os.makedirs(complete_index_directory)
    with open(file, "w") as f:
        json.dump(part_complete_index, f)

def calculate_index_size(directory):
    total_size = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            total_size += os.path.getsize(filepath)
    return total_size / 1024


def write_report(total_tokens, total_files, index_size_kb):
    report_path = os.path.join(os.getcwd(), "report.txt")
    with open(report_path, "w") as f:
        msg = f"Total Number of Tokens: {total_tokens}\nTotal Number of Files: {total_files}\nTotal Size of Index (KB): {index_size_kb}"
        f.write(msg)


# def write_single_index(complete_index):
#     file = os.path.join(complete_index_directory, "complete_index_SINGLE.json")
#     write_files(file, complete_index)


def main(path):
    files = json_files(path)
    index(files)
    total_tokens = index_complete()
    # total_tokens = write_single_index(complete_index)
    # total_tokens = write_complete_index(complete_index)
    index_size_kb = calculate_index_size(complete_index_directory)
    write_report(total_tokens, len(files), index_size_kb)


if __name__ == "__main__":
    path = "c:/users/16264/desktop/developer/DEV"
    # path = "c:/users/16264/desktop/developer/ANALYST"
    main(path)
