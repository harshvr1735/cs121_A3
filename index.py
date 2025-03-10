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
import hashlib

# nltk.download('punkt_tab')
# NOTE: NEED TO PIP INSTALL LXML

ind_size = 5000
partial_index_directory = os.path.join(os.getcwd(), "partial_index")
complete_index_directory = os.path.join(os.getcwd(), "complete_index")
ngram_index_directory = os.path.join(os.getcwd(), "ngram_index")
position_index_directory = os.path.join(os.getcwd(), "position_index")
stemmer = PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9]+')
positions = defaultdict(dict)
ngram_positions = defaultdict(dict)
word_position_map = defaultdict(dict)

def json_files(path):
    files = []
    path = Path(path)
    for json_file in path.rglob("*.json"):
        print(json_file)
        with json_file.open("r") as f:
            files.append(json.load(f))

    return files

def create_ngrams(tokens, n=2):
    """Create n-grams from a list of tokens"""
    return ['_'.join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

def tokenize(text):
    soup = BeautifulSoup(text, 'lxml')
    word_dict = defaultdict(lambda: defaultdict(int))
    word_positions = defaultdict(list)
    ngrams_dict = defaultdict(lambda: defaultdict(int))
    
    important_text = {
        't': soup.title.string if soup.title and soup.title.string else '',
        'a': ' '.join([h.get_text() for h in soup.find_all('h1')]),
        'h1': ' '.join([h.get_text() for h in soup.find_all('h1')]),
        'h2': ' '.join([h.get_text() for h in soup.find_all('h2')]),
        'h3': ' '.join([h.get_text() for h in soup.find_all('h3')]),
        'b': ' '.join([b.get_text() for b in soup.find_all(['b', 'strong'])]),
        'n': soup.get_text() if soup.get_text() else ''
    }

    position = 0
    all_tokens = []
    
    for level, content in important_text.items():
        tokens = tokenizer.tokenize(content)
        for token_pos, token in enumerate(tokens):
            token = token.lower()
            stemmed_token = stemmer.stem(token)
            
            global_position = position + token_pos
            all_tokens.append(stemmed_token)
            
            word_dict[stemmed_token][level] += 1
            word_positions[stemmed_token].append((level, global_position))
        
        position += len(tokens)

    bigrams = create_ngrams(all_tokens, 2)
    for bigram in bigrams:
        ngrams_dict[bigram]['2gram'] += 1
    
    trigrams = create_ngrams(all_tokens, 3)
    for trigram in trigrams:
        ngrams_dict[trigram]['3gram'] += 1
    
    return word_dict, word_positions, ngrams_dict

def compute_tf(freq):
    return math.log(1 + freq)

def compute_hash(doc_content):
    return hashlib.md5(doc_content.encode('utf-8')).hexdigest()

def create_shingles(content):
    words = content.split()
    shingle_size = min(5, len(words))
    shingles = set()
    for i in range(len(words) - shingle_size + 1):
        shingle = " ".join(words [i:i + shingle_size])
        shingles.add(compute_hash(shingle))
    return shingles

def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

def index(files):
    index = defaultdict(list)
    ngram_index = defaultdict(list)
    position_index = defaultdict(list)
    counter = 0
    running_count = 0
    part = 1
    docID_url = {}
    content_hashes = set()
    duplicate_count = 0
    shingle_sets = {}
    near_duplicate_count = 0

    for doc in files:
        content = doc['content']
        content_hash = compute_hash(content)
        if content_hash in content_hashes:
            print(f"Duplicate page detected: Skipping {doc['url']}")
            duplicate_count += 1
            continue

        shingles = create_shingles(content)
        is_near_duplicate = False
        for existing_shingles in shingle_sets.values():
            if jaccard_similarity(shingles, existing_shingles) > 0.8:
                print(f"Near-duplicate page detected: Skipping {doc['url']}")
                is_near_duplicate = True
                near_duplicate_count += 1
                break
        
        if is_near_duplicate:
            continue

        shingle_sets[running_count] = shingles
        content_hashes.add(content_hash)
        print(doc['url'])
        docID_url[running_count] = doc['url']
        
        tokens, token_positions, ngrams = tokenize(content)
        
        for word, freq_by_importance in tokens.items():
            for imp_level, freq in freq_by_importance.items():
                log_normalized_tf = compute_tf(freq)
                index[word].append([running_count, log_normalized_tf, imp_level])
        
        for ngram, freq_by_type in ngrams.items():
            for gram_type, freq in freq_by_type.items():
                log_normalized_tf = compute_tf(freq)
                ngram_index[ngram].append([running_count, log_normalized_tf, gram_type])
        
        for word, positions_list in token_positions.items():
            position_index[word].append([running_count, positions_list])

        counter += 1
        running_count += 1
        print(counter)

        if counter >= ind_size:
            index_partial(index, part)
            if ngram_index:
                index_ngram_partial(ngram_index, part)
            if position_index:
                index_position_partial(position_index, part)
            index.clear()
            ngram_index.clear()
            position_index.clear()
            part += 1
            counter = 0

    if len(index.keys()) != 0:
        index_partial(index, part)
    if len(ngram_index.keys()) != 0:
        index_ngram_partial(ngram_index, part)
    if len(position_index.keys()) != 0:
        index_position_partial(position_index, part)

    return docID_url, running_count, duplicate_count, near_duplicate_count

def index_partial(index, part):
    if not os.path.exists(partial_index_directory):
        os.makedirs(partial_index_directory)
    filename = f"partial_index_part{part}.txt"
    index = dict(sorted(index.items()))
    file = os.path.join(partial_index_directory, filename)
    with open(file, "w") as f:
        json.dump(index, f)

def index_ngram_partial(ngram_index, part):
    if not os.path.exists(partial_index_directory + "_ngram"):
        os.makedirs(partial_index_directory + "_ngram")
    filename = f"partial_ngram_index_part{part}.txt"
    ngram_index = dict(sorted(ngram_index.items()))
    file = os.path.join(partial_index_directory + "_ngram", filename)
    with open(file, "w") as f:
        json.dump(ngram_index, f)

def index_position_partial(position_index, part):
    if not os.path.exists(partial_index_directory + "_position"):
        os.makedirs(partial_index_directory + "_position")
    filename = f"partial_position_index_part{part}.txt"
    position_index = dict(sorted(position_index.items()))
    file = os.path.join(partial_index_directory + "_position", filename)
    with open(file, "w") as f:
        json.dump(position_index, f)

def write_docID_url(docID_url):
    file = os.path.join(os.getcwd(), "docID_url_map.txt")
    with open(file, "w") as f:
        json.dump(docID_url, f)

def compute_idf(current_word_count, total_documents):
    idf = math.log((total_documents + 1) / (current_word_count + 1)) + 1
    return idf

def compute_tf_idf(tf, idf):
    return round(tf * idf, 5)

def index_complete(running_count):
    process_and_merge_index(partial_index_directory, complete_index_directory, running_count)
    
    if os.path.exists(partial_index_directory + "_ngram"):
        process_and_merge_index(partial_index_directory + "_ngram", ngram_index_directory, running_count)
    
    if os.path.exists(partial_index_directory + "_position"):
        process_and_merge_position_index(partial_index_directory + "_position", position_index_directory)
    
    unique_tokens = count_unique_tokens(complete_index_directory)
    
    return unique_tokens

def process_and_merge_index(input_dir, output_dir, total_documents):
    """Process and merge index files (works for both regular and n-gram indices)"""
    if not os.path.exists(input_dir):
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    partial_paths = []
    for f in os.listdir(input_dir):
        if f.endswith(".txt"):
            partial_paths.append(os.path.join(input_dir, f))

    iterators = []
    iterators2 = []
    for path in partial_paths:
        iterators.append(iterator_partial(path))
        iterators2.append(iterator_partial(path))

    merged = heapq.merge(*iterators, key=lambda x: x[0])
    copymerged = heapq.merge(*iterators2, key=lambda x: x[0])

    current_word = None
    current_word_docs = set()
    idf_dict = defaultdict(int)
    
    for word, info in copymerged:
        if current_word is None:
            current_word = word
            current_word_docs.add(info[0][0])
        elif current_word != word:
            idf = compute_idf(len(current_word_docs), total_documents)
            idf_dict[current_word] = idf
            current_word = word
            current_word_docs = {info[0][0]}
        else:
            current_word_docs.add(info[0][0])
            
    if current_word:
        idf = compute_idf(len(current_word_docs), total_documents)
        idf_dict[current_word] = idf
    
    current_prefix = None
    current_data = defaultdict(list)
    
    for word, info in merged:
        prefix = get_prefix(word)

        if current_prefix and prefix != current_prefix:
            save_partial_file(output_dir, current_prefix, current_data)
            update_positions(output_dir, current_prefix, current_data)
            current_data.clear()

        current_prefix = prefix

        if len(info[0]) >= 3:
            new_info = [[inf[0], compute_tf_idf(inf[1], idf_dict[word]), inf[2]] for inf in info]
        else:
            new_info = info
            
        current_data[word].extend(new_info)

    if current_data:
        save_partial_file(output_dir, current_prefix, current_data)
        update_positions(output_dir, current_prefix, current_data)

    if output_dir == complete_index_directory:
        save_positions_to_file(output_dir, positions)
    elif output_dir == ngram_index_directory:
        save_positions_to_file(output_dir, ngram_positions)

def process_and_merge_position_index(input_dir, output_dir):
    """Process and merge position index files"""
    if not os.path.exists(input_dir):
        return
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    partial_paths = []
    for f in os.listdir(input_dir):
        if f.endswith(".txt"):
            partial_paths.append(os.path.join(input_dir, f))

    iterators = []
    for path in partial_paths:
        iterators.append(iterator_partial(path))

    merged = heapq.merge(*iterators, key=lambda x: x[0])
    
    current_prefix = None
    current_data = defaultdict(list)
    
    for word, info in merged:
        prefix = get_prefix(word)

        if current_prefix and prefix != current_prefix:
            save_position_file(output_dir, current_prefix, current_data)
            update_word_position_map(output_dir, current_prefix, current_data)
            current_data.clear()

        current_prefix = prefix
        current_data[word].extend(info)

    if current_data:
        save_position_file(output_dir, current_prefix, current_data)
        update_word_position_map(output_dir, current_prefix, current_data)


    save_word_position_map_to_file(output_dir)

def get_prefix(word):
    if re.match(r'^[0-9]+$', word[0]):
        return word[0]
    if word[0] in "abcdefghijklmnopqrstuvwxyz":
        return word[0]
    else:
        return "nonalpha"

def save_partial_file(directory, prefix, data):
    file_path = os.path.join(directory, f"complete_index_{prefix}.txt")
    with open(file_path, "w") as f:
        f.write("[\n")
        
        first = True
        for word, postings in data.items():
            sorted_postings = sorted(postings, key=lambda x: x[1] if len(x) > 1 else 0, reverse=True)
            if not first:
                f.write(",\n")
            json.dump({"word": word, "postings": sorted_postings}, f)
            first = False
        f.write("\n]")

def save_position_file(directory, prefix, data):
    file_path = os.path.join(directory, f"position_index_{prefix}.txt")
    with open(file_path, "w") as f:
        f.write("[\n")
        
        first = True
        for word, postings in data.items():
            if not first:
                f.write(",\n")
            json.dump({"word": word, "postings": postings}, f)
            first = False
        f.write("\n]")

def update_positions(directory, prefix, data):
    file_path = os.path.join(directory, f"complete_index_{prefix}.txt")
    temp_positions = {}
    with open(file_path, "rb") as f:
        line = f.readline()
        byte_offset = f.tell()
        for word in data.keys():
            temp_positions[word] = byte_offset
            line = f.readline()
            byte_offset = f.tell()
    
    if directory == complete_index_directory:
        positions[prefix] = temp_positions
    elif directory == ngram_index_directory:
        ngram_positions[prefix] = temp_positions

def update_word_position_map(directory, prefix, data):
    file_path = os.path.join(directory, f"position_index_{prefix}.txt")
    temp_positions = {}
    with open(file_path, "rb") as f:
        line = f.readline()
        byte_offset = f.tell()
        for word in data.keys():
            temp_positions[word] = byte_offset
            line = f.readline()
            byte_offset = f.tell()
    
    word_position_map[prefix] = temp_positions

def save_positions_to_file(directory, pos_map):
    position_file_path = os.path.join(directory, "positions.txt")
    with open(position_file_path, "w") as f:
        f.write("[\n")
        
        first = True
        for prefix, position in pos_map.items():
            if not first:
                f.write(",\n")
            json.dump({"prefix": prefix, "position": position}, f)
            first = False
        f.write("\n]")

def save_word_position_map_to_file(directory):
    position_file_path = os.path.join(directory, "word_positions.txt")
    with open(position_file_path, "w") as f:
        f.write("[\n")
        
        first = True
        for prefix, position in word_position_map.items():
            if not first:
                f.write(",\n")
            json.dump({"prefix": prefix, "position": position}, f)
            first = False
        f.write("\n]")

def iterator_partial(file_path):
    with open(file_path, 'r') as f:
        partial_index = json.load(f)
        for word, postings in sorted(partial_index.items()):
            yield word, postings

def count_unique_tokens(directory):
    """Count unique tokens in the index"""
    unique_tokens = set()
    for filename in os.listdir(directory):
        if filename.startswith("complete_index_") and filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r') as f:
                content = f.read()
                word_matches = re.findall(r'"word"\s*:\s*"([^"]+)"', content)
                unique_tokens.update(word_matches)
    
    return len(unique_tokens)

def calculate_index_size(directory):
    total_size = 0
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            total_size += os.path.getsize(filepath)
    return total_size / 1024

def write_report(total_tokens, total_files, exact_duplicates, near_duplicates):
    report_path = os.path.join(os.getcwd(), "report.txt")
    index_size_kb = calculate_index_size(complete_index_directory)
    ngram_size_kb = calculate_index_size(ngram_index_directory) if os.path.exists(ngram_index_directory) else 0
    position_size_kb = calculate_index_size(position_index_directory) if os.path.exists(position_index_directory) else 0
    
    with open(report_path, "w") as f:
        msg = (f"Total Number of Tokens: {total_tokens}\n"
               f"Total Number of Files: {total_files}\n"
               f"Total Size of Index (KB): {index_size_kb}\n"
               f"Total Size of N-gram Index (KB): {ngram_size_kb}\n"
               f"Total Size of Position Index (KB): {position_size_kb}\n"
               f"Number of exact pages skipped: {exact_duplicates}\n"
               f"Number of near-duplicate pages skipped: {near_duplicates}")
        f.write(msg)

def main(path):
    files = json_files(path)
    docID_url, running_count, exact_duplicates, near_duplicates = index(files)
    write_docID_url(docID_url)
    total_tokens = index_complete(running_count)
    write_report(total_tokens, len(files), exact_duplicates, near_duplicates)

if __name__ == "__main__":
    path = "C:/users/16264/desktop/developer/DEV/"
    # path = "C:/users/16264/desktop/developer/ANALYST/www-db_ics_uci_edu"

    main(path)
