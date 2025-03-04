import os
import json
import re
import time
import nltk.tokenize
from nltk.stem import PorterStemmer
import numpy as np
import math

import bisect

complete_index_directory = os.path.join(os.getcwd(), "complete_index")
stemmer = PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9]+')

def read_json(file_path, position, key):
    # print(key)
    # time.sleep(2)
    file_path.seek(position)
    # print(position)
    # chunk = file_path.read(190000) ## may need to up the threshold for v v common words ?
    # print(chunk)
    chunk = file_path.readline()
    # time.sleep(5)
    # print(chunk)
    try:
        start = chunk.find('{"word":')
        # end = chunk.find('}', start) + 1
        # end = chunk.find('}', start)
        # print(chunk.find('}', start))
        end = chunk.find('}', start)
        # print("END: ", end)
        # time.sleep(2)
        if end == -1:
            last_comma = chunk.rfind(',', start)
            if last_comma != -1:
                latest_closed_bracket = chunk.rfind(']', start)
                if last_comma + 2 != latest_closed_bracket:
                    # print("KYS")
                    # time.sleep(3)
                    truncated_chunk = chunk[:latest_closed_bracket] + "]]}"
                else:
                    print("CONTAINS COMMA")
                    # time.sleep(2)
                    truncated_chunk = chunk[:last_comma] + "]]}"
            else:
                print("SOMETHING ELSE")
                # time.sleep(2)
                truncated_chunk = chunk[:start] + "]}"
        else:
            truncated_chunk = chunk[:end + 1]
        # print()
        # print(truncated_chunk)
        # time.sleep(5)
        # print()
        word_postings = truncated_chunk

        # print("ACTUAL CHUNK: ", word_postings)
        # print()
        data = json.loads(word_postings)
        # print("HI!")
        # time.sleep(2)
        if data.get("word") == key:
            return data.get("postings", [])
        
    except json.JSONDecodeError:
        print("JSON ERROR")
        time.sleep(1)
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
    # print("vector magnitudes", vector1_magnitude, vector2_magnitude)
    # print("vectors product",vectors_product)
    if vector1_magnitude == 0 or vector2_magnitude == 0:
        return 0
    similarity = vectors_product / (vector1_magnitude * vector2_magnitude)
    # print("SIMILARTIY: ",similarity)
    # print()
    return similarity

def get_query_vector(query_terms, index, positions):
    term_postings = {}
    query_vector = []
    for term in query_terms:
        term = stemmer.stem(term)
        try:
            prefix = prefix_getter(term)

            # file = index[prefix]
            # file.seek(0)
            # a = json.load(file)
            term_position = positions[prefix]
            if term not in term_position:
                print(f"{term} not found in index.")
                continue
            position = term_position[term]

            postings = set((doc_id, tfidf) for doc_id, tfidf, _ in read_json(index[prefix], position, term))
            term_postings[term] = postings
            # print("query vector: ", a)
            # for entry in a:
                # print("entry: ", entry)
                # if term == entry["word"]:
                    # print("IF TERM IN A")
                    # print(entry['postings'][0][1])
            # print("postings: ", postings)
            postings = [tfidf for _, tfidf in postings]
            for posting in postings:
                query_vector.append(posting)
                    # break
            # else:
            #     query_vector.append(0)
        except KeyError:
            query_vector.append(0)
    # print(query_vector)
    # time.sleep(3)
    return query_vector, term_postings

def get_document_vector(doc_id, query_terms, index, positions, term_postings):
    # doc_time = time.time()
    doc_vector = []
    for term in query_terms:
        term = stemmer.stem(term)
        prefix = prefix_getter(term)
        try:
            term_position = positions[prefix]
            if term not in term_position:
                print(f"{term} not found in index.")
                continue
            # position = term_position[term]
            doc_time = time.time()
            # postings = set((doc_id, tfidf) for doc_id, tfidf, _ in read_json(index[prefix], position, term))
            # postings = read_json(index[prefix], position, term)
            postings = term_postings[term]
            print("postings generate:", time.time() - doc_time)
            # print(postings)

            print("number of postings:", len(postings))
            # print(postings)
            # time.sleep(1)
            tf_idf_score = bin_search(list(postings), doc_id)
            # print(tf_idf_score)
            # time.sleep(5)
            # postings = list(postings)
            # postings.sort(key=lambda x: x[0])

            # tf_idf_score = next((posting[1] for posting in postings if posting[0] == doc_id), 0)
            # print("DOCUMENT VECTOR TDIDF: ", tf_idf_score)
            doc_vector.append(tf_idf_score)

            print("inner doc time:", time.time() - doc_time)

        except KeyError:
            doc_vector.append(0)
    # print("doc time:", time.time() - doc_time)
    # print(doc_vector)
    return doc_vector
#####

def bin_search(postings, doc_id):
    postings.sort(key=lambda x: x[0])
    doc_ids = np.array([posting[0] for posting in postings])
    index = np.searchsorted(doc_ids, doc_id)
    if index < len(postings) and doc_ids[index] == doc_id:
        # print("binsearch time: ",time.time() - bin_search)
        return postings[index][1]
    else:
        return 0 ## not found




def index_getter(index, positions, input, docID_url):
    start = time.time()

    query_terms = tokenizer.tokenize(input)
    query_terms = [stemmer.stem(term.lower()) for term in query_terms]
    query_vector, term_postings = get_query_vector(query_terms, index, positions)

    results = []
    for term in query_terms:
        print(term)
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
            postings = set(doc_id for doc_id, _, _ in read_json(index[prefix], position, term))

            results.append(postings)
            print("lookup time: ", time.time() -  lookupstart)
        except KeyError:
            print("Not found in index.")

    if results:
        result_time = time.time()
        results.sort(key=len)
        smallest = results[0] ## more efficient to AND smallest -> mid -> largest
        for res in results:
            smallest &= res
            # smallest = smallest.intersection(res) ## actually TA says & doesnt work that well and returns bad results from a1
            ## need to do something else
        print(len(smallest))
        # time.sleep(4)
        if smallest:
            cosine_time = time.time()
            
            doc_scores = []
            if len(query_terms) == 1: ## we dont need to find cosine similarity in single token words
                term = query_terms[0]
                prefix = prefix_getter(term)
                position = positions[prefix][term]

                for id, tfidf, _ in read_json(index[prefix], position, term):
                    if id in smallest:
                        doc_scores.append((id, tfidf))

            else:
                smallest = list(smallest)
                for doc_id in smallest[:50]:
                    doc_time = time.time()
                    doc_vector = get_document_vector(doc_id, query_terms, index, positions, term_postings) ## takes around 0.02
                    print("DOC TIME:", time.time() - doc_time)

                    similarity = cosine_similarity(query_vector, doc_vector) ## takes 0.0

                    doc_scores.append((doc_id, similarity))

            doc_scores.sort(key=lambda x: x[1], reverse=True)
            for doc_id, score in doc_scores[:5]:
                print(f"{doc_id}: {docID_url[str(doc_id)]} (Score: {score:.4f})")

            print("cosine time: ", time.time() - cosine_time)

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
