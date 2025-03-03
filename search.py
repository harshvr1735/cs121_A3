import os
import json
import re
import time
import nltk.tokenize
from nltk.stem import PorterStemmer
import numpy as np
import math

complete_index_directory = os.path.join(os.getcwd(), "complete_index")
stemmer = PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9]+')

def read_json(file_path, position, key):
    file_path.seek(position)
    chunk = file_path.read(1000) ## may need to up the threshold for v v common words ?

    try:
        start = chunk.find('{"word":')
        end = chunk.find('}', start) + 1
        word_postings = chunk[start:end]

        print("ACTUAL CHUNK: ", word_postings)
        print()
        data = json.loads(word_postings)
        if data.get("word") == key:
            return data.get("postings", [])
        
    except json.JSONDecodeError:
        pass

    print("NOT FOUND")
    return []

def load_positions():
    positions_file = os.path.join(complete_index_directory, "positions.json")
    with open(positions_file, "r") as f:
        positions = json.load(f)
    
    position_map = {}
    for entry in positions:
        prefix = entry["prefix"]
        position_map[prefix] = entry["position"]
    
    return position_map

def load_inverted_index():
    index = {}
    for file in os.listdir(complete_index_directory):
        if file.endswith(".json"):
            name_split = file.split("_")[-1].split(".")[0]
            # print(name_split)
            path = os.path.join(complete_index_directory, file)

            index[name_split] = (open(path, "r"))
    return index





####
def dot_product(vector1, vector2):
    return sum(v1 * v2 for v1, v2 in zip(vector1, vector2))

def vector_magnitude(vector):
    return math.sqrt(sum(v ** 2 for v in vector))

def cosine_similarity(vector1, vector2):
    vectors_product = dot_product(vector1, vector2)
    vector1_magnitude = vector_magnitude(vector1)
    vector2_magnitude = vector_magnitude(vector2)
    print("vector magnitudes", vector1_magnitude, vector2_magnitude)
    print("vectors product",vectors_product)
    if vector1_magnitude == 0 or vector2_magnitude == 0:
        return 0
    similarity = vectors_product / (vector1_magnitude * vector2_magnitude)
    print("SIMILARTIY: ",similarity)
    print()
    return similarity

def get_query_vector(query_terms, index):
    query_vector = []
    for term in query_terms:
        term = stemmer.stem(term)
        try:
            prefix = prefix_getter(term)

            file = index[prefix]
            file.seek(0)
            a = json.load(file)
            # print("query vector: ", a)
            for entry in a:
                # print("entry: ", entry)
                if term == entry["word"]:
                    print("IF TERM IN A")
                    # print(entry['postings'][0][1])
                    query_vector.extend([posting[1] for posting in entry["postings"]])
                    break
            else:
                query_vector.append(0)
        except KeyError:
            query_vector.append(0)
    print(query_vector)
    return query_vector

def get_document_vector(doc_id, query_terms, index):
    doc_vector = []
    for term in query_terms:
        term = stemmer.stem(term)
        prefix = prefix_getter(term)
        try:
            file = index[prefix]
            file.seek(0)
            a = json.load(file)
            for entry in a:
                # print("entry: ", entry)
                if term == entry["word"]:
            # if term in a:
            
                    tf_idf_score = next((posting[1] for posting in entry["postings"] if posting[0] == doc_id), 0)
                    print("DOCUMENT VECTOR TDIDF: ", tf_idf_score)
                    doc_vector.append(tf_idf_score)
            else:
                doc_vector.append(0)
        except KeyError:
            doc_vector.append(0)
    return doc_vector
#####






def index_getter(index, positions, input, docID_url):
    start = time.time()
    # split = tokenizer.tokenize(input)
    # split = [stemmer.stem(token.lower()) for token in tokenizer.tokenize(input)]
    query_terms = tokenizer.tokenize(input)
    query_terms = [stemmer.stem(term.lower()) for term in query_terms]
    query_vector = get_query_vector(query_terms, index)
    # print(stems)
    # split = [condition.strip() for condition in input.split("AND")]

    results = []
    for term in query_terms:
        try:
            prefix = prefix_getter(term)
            if prefix == "invalid":
                run(index, docID_url)
            if not prefix or prefix not in index or prefix not in positions:
                continue
            lookupstart = time.time()

            term_position = positions[prefix]
            if term not in term_position:
                print(f"{term} not found in index.")
                continue
            position = term_position[term]
            # positions = os.path.join(complete_index_directory, "positions.json")


            # path = os.path.join(complete_index_directory, f"complete_index_{prefix}.json")
            postings = set(doc_id for doc_id, _, _ in read_json(index[prefix], position, term))

            results.append(postings)
            print("lookup time: ", time.time() -  lookupstart)
        except KeyError:
            print("Not found in index.")

    if results:

        results.sort(key=len)
        smallest = results[0] ## more efficient to AND smallest -> mid -> largest
        for res in results:
            smallest &= res
            # smallest = smallest.intersection(res) ## actually TA says & doesnt work that well and returns bad results from a1
            ## need to do something else

        if smallest:
            with open("docID_url_map.json", "r") as f:
                docID_url = json.load(f)
                doc_scores = []
                for doc_id in smallest:
                    doc_vector = get_document_vector(doc_id, query_terms, index)
                    similarity = cosine_similarity(query_vector, doc_vector)
                    doc_scores.append((doc_id, similarity))
                doc_scores.sort(key=lambda x: x[1], reverse=True)
                for doc_id, score in doc_scores[:10]:
                    print(f"{doc_id}: {docID_url[str(doc_id)]} (Score: {score:.4f})")


    end = time.time()
    print("Query took: ", end - start)
    run(index, positions, docID_url)

def prefix_getter(word):  # names the files and checks prefixes
    if re.match(r'^[0-9]+$', word[0]):
        return "numbers"
    elif word[0] in "abcdefghijklmnopqrstuvwxyz":
        return word[0]
    else:
        return "invalid"

def take_input():
    input1 = input("Input or q to quit: ")
    if input1.lower() == 'q':
        exit()
    return input1

def run(index, positions, docID_url):
    input = take_input()
    index_getter(index, positions, input, docID_url)

def main():
    index = load_inverted_index()
    positions = load_positions()
    with open("docID_url_map.json", "r") as f:
        docID_url = json.load(f)
    run(index, positions, docID_url)


if __name__ == "__main__":
    main()
