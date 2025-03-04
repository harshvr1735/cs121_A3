import os
import json
from pathlib import Path
from bs4 import BeautifulSoup
import nltk.tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
import heapq
import re
import math
import time

# nltk.download('punkt_tab')
# NOTE: NEED TO PIP INSTALL LXML

ind_size = 5000
partial_index_directory = os.path.join(os.getcwd(), "partial_index")
complete_index_directory = os.path.join(os.getcwd(), "complete_index")
stemmer = PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9]+')
positions = defaultdict(dict)

def json_files(path):
    files = []
    path = Path(path)
    for json_file in path.rglob("*.json"):
        print(json_file)
        with json_file.open("r") as f:
            files.append(json.load(f))

    return files


def tokenize(text):
    soup = BeautifulSoup(text, 'lxml')
    word_dict = defaultdict(lambda: defaultdict(int))

    # Categorizing text based on their importance level
    important_text = {
        't': soup.title.string if soup.title and soup.title.string else '',
        'h1': ' '.join([h.get_text() for h in soup.find_all('h1')]),
        'h2': ' '.join([h.get_text() for h in soup.find_all('h2')]),
        'h3': ' '.join([h.get_text() for h in soup.find_all('h3')]),
        'b': ' '.join([b.get_text() for b in soup.find_all(['b', 'strong'])]),
        'n': soup.get_text() if soup.get_text() else ''
    }

    for level, content in important_text.items():
        tokens = tokenizer.tokenize(content)
        # print(tokens)
        # time.sleep(10)
        for token in tokens:
            token = token.lower()
            # print(tokens)
            # time.sleep(2)
            token = stemmer.stem(token.lower())  # porter stemming
            word_dict[token][level] += 1
    return word_dict

def compute_tf(freq):
    return math.log(1 + freq)

def index(files):
    index = defaultdict(list)
    counter = 0
    running_count = 0
    part = 1
    docID_url = {}

    for doc in files:
        print(doc['url'])
        docID_url[running_count] = doc['url']
        
        content = doc['content']
        tokens = tokenize(content)
        print()
        for word, freq_by_importance in tokens.items():
            for imp_level, freq in freq_by_importance.items():
                log_normalized_tf = compute_tf(freq)
                index[word].append([running_count, log_normalized_tf, imp_level])

        counter += 1
        running_count += 1
        print(counter)

        if counter >= ind_size:  # gets called every 10k pages, could lower i think theres like
            index_partial(index, part)  # 50k total ?
            index.clear()
            part += 1
            counter = 0

    if len(index.keys()) != 0:
        index_partial(index, part)  # catches the final indexes

    return docID_url, running_count

def index_partial(index, part):
    if not os.path.exists(
            partial_index_directory):  # wait is this supposed to be ran on lab or local? does os.path work for lab
        os.makedirs(partial_index_directory)  # even so need to upload all the files to the repo which is hmmmm
    filename = f"partial_index_part{part}.json"
    index = dict(sorted(index.items()))
    file = os.path.join(partial_index_directory, filename)
    with open(file, "w") as f:
        json.dump(index, f)

def write_docID_url(docID_url): ## writes the document IDs and URLs to a file to return the results
    file = os.path.join(os.getcwd(), "docID_url_map.json")
    with open(file, "w") as f:
        json.dump(docID_url, f)

def compute_idf(current_word_count, total_documents):
    idf = math.log((total_documents + 1) / (current_word_count + 1)) + 1
    return idf

def compute_tf_idf(tf, idf):
    return round(tf * idf, 5)

def index_complete(running_count):
    partial_paths = []
    for f in os.listdir(partial_index_directory):
        if f.endswith(".json"):
            partial_paths.append(os.path.join(partial_index_directory, f))

    iterators = []
    iterators2 = []
    for path in partial_paths:
        iterators.append(iterator_partial(path))
        iterators2.append(iterator_partial(path))


    merged = heapq.merge(*iterators,
                         key=lambda x: x[0])  # sorts the iterators based on the first letter, makes an iterator

    # iterators2 = iterators.copy()
    copymerged = heapq.merge(*iterators2,
                         key=lambda x: x[0])  # sorts the iterators based on the first letter, makes an iterator

    current_word = None
    current_word_count = 0
    idf_dict = defaultdict(int)
    for word, info in copymerged:
        print("word",word)
        # time.sleep(1)
        if current_word is None: ## sets the initial currenet word to be compared
            # print("hit")
            # time.sleep(1)
            current_word = word
            current_word_count = len(info)

        elif current_word != word:
            idf = compute_idf(current_word_count, running_count)
            idf_dict[current_word] = idf
            print("idf, current_word, current_count: ",idf, current_word, current_word_count)
            # time.sleep(0.1)
            current_word = word
            current_word_count = len(info)

        else:
            # time.sleep(1)
            current_word_count += len(info)
    if current_word:
        idf = compute_idf(current_word_count, running_count)
        idf_dict[current_word] = idf

    # print(idf_dict, current_word_count)
    # THE IDEA IS:
    # everything is stored inside partial indexes, so there are iterators for each partial index
    # once you hit the next range: example: you hit "a" with your iterator, you switch from the
    # number files, and just to the "af" files. then you continue
    
    current_prefix = None
    current_data = defaultdict(list)
    utoken = set()

    for word, info in merged:
        utoken.add(word)  # the token counter
        prefix = get_prefix(word)  # finds the first letter

        if current_prefix and prefix != current_prefix:  # checks if the new prefix == our old prefix
            save_partial_file(current_prefix, current_data)  #if not, writes the data
            update_positions(current_prefix, current_data)

            current_data.clear()  # clears the dictionary so we dont hold leftovers

        current_prefix = prefix  # sets the new prefix
        new_info = [[inf[0], compute_tf_idf(inf[1], idf_dict[word]), inf[2]] for inf in info] ## tf-idf
                            ## tf = log(1 + tf), so we dont have as much weight on frequently appearing terms
                            ## idf = log((total_docs / number of docs that contain that word) + 1) also for normalization
        current_data[word].extend(new_info)  # adds the word/info


    if current_data:  # sends off the last of the data
        save_partial_file(current_prefix, current_data)
            
            
    update_positions(current_prefix, current_data)

    save_positions_to_file()  # Save positions after processing all prefixes
    return len(utoken)


def get_prefix(word):  # names the files and checks prefixes
    if re.match(r'^[0-9]+$', word[0]):
        return word[0]
    if word[0] in "abcdefghijklmnopqrstuvwxyz":
        return word[0]
    else:
        return "nonalpha"


def save_partial_file(prefix, data):
    if not os.path.exists(complete_index_directory):
        os.makedirs(complete_index_directory)
    file_path = os.path.join(complete_index_directory, f"complete_index_{prefix}.json")
    with open(file_path, "w") as f:
        f.write("[\n")
        
        first = True
        for word, postings in data.items():
            sorted_postings = sorted(postings, key=lambda x: x[1], reverse=True)
            if not first:
                f.write(",\n")
            json.dump({"word": word, "postings": sorted_postings}, f)
            first = False
        f.write("\n]")



def update_positions(prefix, data):
    file_path = os.path.join(complete_index_directory, f"complete_index_{prefix}.json")
    temp_positions = {}
    with open(file_path, "rb") as f:
        line = f.readline()
        byte_offset = f.tell()
        for word in data.keys():
            # print(byte_offset)

            temp_positions[word] = byte_offset
            line = f.readline() ## just moves the counter to the next line
            byte_offset = f.tell()
    positions[prefix] = temp_positions

def save_positions_to_file():
    position_file_path = os.path.join(complete_index_directory, "positions.json")
    with open(position_file_path, "w") as f:
        # json.dump(positions, f)
        f.write("[\n")
        
        first = True
        for prefix, position in positions.items():
            if not first:
                f.write(",\n")
            json.dump({"prefix": prefix, "position": position}, f)
            first = False
        f.write("\n]")


def iterator_partial(file_path):  # makes the iterators
    with open(file_path, 'r') as f:
        partial_index = json.load(f)
        for word, postings in sorted(partial_index.items()):
            yield word, postings


def calculate_index_size(directory):
    total_size = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            total_size += os.path.getsize(filepath)
    return total_size / 1024


def write_report(total_tokens, total_files):
    report_path = os.path.join(os.getcwd(), "report.txt")
    index_size_kb = calculate_index_size(complete_index_directory)
    with open(report_path, "w") as f:
        msg = (f"Total Number of Tokens: {total_tokens}\nTotal Number of Files: {total_files}\n"
               f"Total Size of Index (KB): {index_size_kb}")
        f.write(msg)


def main(path):
    files = json_files(path)
    docID_url, running_count = index(files)
    write_docID_url(docID_url)
    total_tokens = index_complete(running_count)
    write_report(total_tokens, len(files))


if __name__ == "__main__":
    path = "C:/users/16264/desktop/developer/DEV/"
    # path = "C:/users/16264/desktop/developer/ANALYST/www-db_ics_uci_edu"

    main(path)
