import os
import json
from pathlib import Path
from bs4 import BeautifulSoup
import nltk.tokenize
from nltk.stem import PorterStemmer
from collections import defaultdict
import heapq
nltk.download('punkt_tab')
## NOTE: NEED TO PIP INSALL LXML

ind_size = 500
partial_index_directory = os.path.join(os.getcwd(), "partial_index")
complete_index_directory = os.path.join(os.getcwd(), "complete_index")
stemmer = PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

def json_files(path):
    files = []
    path = Path(path)
    for json_file in path.rglob("*.json"):
        # print(json_file)
        with json_file.open("r") as f:
            files.append(json.load(f))

    return files

def tokenize(text):
    soup = BeautifulSoup(text, 'lxml')
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
            token = stemmer.stem(token) # porter stemming
            word_dict[token][level] += 1
    return word_dict

def index(files):
    index = defaultdict(list)
    unitokens = set()
    counter = 0
    running_count = 0
    part = 1

    for doc in files:
        print(doc['url'])
        content = doc['content']
        tokens = tokenize(content)
        print()
        for word, freq_by_importance in tokens.items():
            for imp_level, freq in freq_by_importance.items():
                index[word].append([running_count, freq, imp_level])
                unitokens.add(word)

        counter += 1
        running_count += 1
        print(counter)

        if counter >= ind_size: ## gets called every 10k pages, could lower i think theres like
            index_partial(index, part) ## 50k total ?
            index.clear()
            part += 1
            counter = 0

    if len(index.keys()) != 0:
        index_partial(index, part) ## catches the final indexes
    return len(unitokens)

def index_partial(index, part):
    if not os.path.exists(partial_index_directory): ## wait is this supposed to be ran on lab or local? does os.path work for lab
        os.makedirs(partial_index_directory) ## even so need to upload all the files to the repo which is hmmmm
    filename = f"partial_index_part{part}.json"
    file = os.path.join(partial_index_directory, filename)
    with open(file, "w") as f:
        json.dump(index, f)

def index_complete():
    partial_paths = []
    for f in os.listdir(partial_index_directory):
        if f.endswith(".json"):
            partial_paths.append(os.path.join(partial_index_directory, f))
    
    iterators = []
    for path in partial_paths:
        iterators.append(iterator_partial(path))
    
    merged = heapq.merge(*iterators, key=lambda x: x[0]) ## sorts the iterators based on the first letter, makes an iterator
    
    current_prefix = None
    current_data = defaultdict(list)
    utoken = set()
    
## THE IDEA IS:
## everything is stored inside partial indexes, so there are iterators for each partial index
## once you hit the next range: example: you hit "a" with your iterator, you switch from the 
## number files, and just to the "af" files. then you continue

    for word, info in merged:
        utoken.add(word) ## the token counter
        prefix = get_prefix(word) ## finds the first letter
        
        if current_prefix and prefix != current_prefix: ## checks if the new prefix == our old prefix
            save_partial_file(current_prefix, current_data) ##if not, writes the data
            current_data.clear() ## clears the dictionary so we dont hold it

        current_prefix = prefix ## sets the new prefix
        current_data[word].extend(info) ## adds the word/info

    if current_data: ## sends off the last of the data
        save_partial_file(current_prefix, current_data)

    return len(utoken)

def get_prefix(word): ## names the files and checks prefixes
    if word[0].isdigit():
        return "numbers"
    if word[0] in "abcdef":
        return "af"
    elif word[0] in "ghijk":
        return "gk"
    elif word[0] in "lmnop":
        return "lp"
    elif word[0] in "qrstu":
        return "qu"
    elif word[0] in "vwxyz":
        return "vz"
    else:
        return "nonalpha"

def save_partial_file(prefix, data):
    if not os.path.exists(complete_index_directory):
        os.makedirs(complete_index_directory)
    file_path = os.path.join(complete_index_directory, f"complete_index_{prefix}.json")
    with open(file_path, "w") as f:
        json.dump(data, f)

def iterator_partial(file_path): ## makes the iterators
    with open(file_path, 'r') as f:
        partial_index = json.load(f)
        for word, postings in sorted(partial_index.items()):
            yield (word, postings)

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
        msg = f"Total Number of Tokens: {total_tokens}\nTotal Number of Files: {total_files}\nTotal Size of Index (KB): {index_size_kb}"
        f.write(msg)


def main(path):
    files = json_files(path)
    index(files)
    total_tokens = index_complete()
    write_report(total_tokens, len(files))


if __name__ == "__main__":
    path = "DEV"
    path = "c:/users/16264/desktop/developer/ANALYST"
    main(path)
