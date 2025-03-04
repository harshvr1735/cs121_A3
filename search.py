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

    file_path.seek(position)
    # chunk = file_path.read(200000) ## this would be better if implemented sorting by tfidf
    ## for ex, we prioritize looking at everyyy posting. this limits it to like 15000 bytes of characters

    chunk = file_path.readline() ## this is if we want every posting

    try:## this is for if we read by bytes. readline() doesnt need this block
        start = chunk.find('{"word":')
        end = chunk.find('}', start)
        if end == -1:
            last_comma = chunk.rfind(',', start)
            if last_comma != -1:
                latest_closed_bracket = chunk.rfind(']', start)
                if last_comma + 2 != latest_closed_bracket:
                    truncated_chunk = chunk[:latest_closed_bracket] + "]]}"
                else:
                    print("CONTAINS COMMA")
                    truncated_chunk = chunk[:last_comma] + "]]}"
            else:
                print("SOMETHING ELSE")
                truncated_chunk = chunk[:start] + "]}"
        else:
            truncated_chunk = chunk[:end + 1]
        word_postings = truncated_chunk

        data = json.loads(word_postings)
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
            path = os.path.join(complete_index_directory, file)

            index[name_split] = (open(path, "r"))
    return index





####
def dot_product(vector1, vector2):
    return np.dot(vector1, vector2)
    # return sum(v1 * v2 for v1, v2 in zip(vector1, vector2))

def vector_magnitude(vector):
    return np.linalg.norm(vector)
    # return math.sqrt(sum(v ** 2 for v in vector))
    

def cosine_similarity(vector1, vector2):
    vectors_product = dot_product(vector1, vector2)
    vector1_magnitude = vector_magnitude(vector1)
    vector2_magnitude = vector_magnitude(vector2)
    if vector1_magnitude == 0 or vector2_magnitude == 0:
        return 0
    similarity = vectors_product / (vector1_magnitude * vector2_magnitude)
    return similarity

def get_query_vector(query_terms, index, positions):
    term_postings = {}
    query_vector = []
    for term in query_terms:
        term = stemmer.stem(term)
        try:
            prefix = prefix_getter(term)

            term_position = positions[prefix]
            if term not in term_position:
                print(f"{term} not found in index.")
                continue
            position = term_position[term]

            postings = set((doc_id, tfidf, text_type) for doc_id, tfidf, text_type in read_json(index[prefix], position, term))
            term_postings[term] = postings

            tf_idf_sum = 0 ## finds the average of allll the tfidf scores for a query term
            for _, tfidf, text_type in postings:
                if text_type in ['b', 'title', 'h1', 'h2', 'h3']: ##weights on text styling
                    tfidf *= 1.5
                tf_idf_sum += tfidf

            average_tf_idf = tf_idf_sum / len(postings)
            query_vector.append(average_tf_idf)

        except KeyError:
            query_vector.append(0)
    return query_vector, term_postings

def get_document_vector(doc_id, query_terms, index, positions, term_postings):
    doc_vector = []
    for term in query_terms:
        # term = stemmer.stem(term)
        prefix = prefix_getter(term)
        try:
            term_position = positions[prefix]
            if term not in term_position:
                print(f"{term} not found in index.")
                continue
            doc_time = time.time()
            postings = term_postings[term]

            # print("number of postings:", len(postings))

            # tf_idf_score = bin_search(list(postings), doc_id) ## MAKES IT 3 SECONDS LONGER THAN NEXT(())???
            ## its taking so long bc this requires a list, but next(()) only needs a set. ok.

            tf_idf_score = next((posting[1] for posting in postings if posting[0] == doc_id), 0)
            if any(posting[0] == doc_id and posting[2] in ['b', 'title', 'h1', 'h2', 'h3'] for posting in postings):
                tf_idf_score *= 1.5

            doc_vector.append(tf_idf_score)

        except KeyError:
            doc_vector.append(0)
    return doc_vector

def get_documents_containing_all_terms(query_terms, term_postings):
    ## init a set of documents that contains the first term
    # print(term_postings.keys())
    dict(sorted(term_postings.items(), key=lambda item: item[1]))
    # time.sleep(2)
    first_term = query_terms[0]
    term_postings_for_first_term = term_postings.get(first_term, [])
    
    if not term_postings_for_first_term:
        return set()

    #start with the set of documents for the first term
    result_docs = set(doc_id for doc_id, _, _ in term_postings_for_first_term)
    print("TERM: ", first_term)
    # intersect with documents containing all other terms
    for term in query_terms[1:]:
        print("TERM: ", term)

        term_postings_for_current_term = term_postings.get(term, [])
        if not term_postings_for_current_term:
            return set()  # if any term isnt in the index return an empty set

        term_docs = set(doc_id for doc_id, _, _ in term_postings_for_current_term)
        result_docs &= term_docs  # intersect

    return result_docs



def index_getter(index, positions, input, docID_url):
    start = time.time()

    query_terms = tokenizer.tokenize(input)
    query_terms = [stemmer.stem(term.lower()) for term in query_terms]
    query_vector, term_postings = get_query_vector(query_terms, index, positions)

    print("query_vector time: ", time.time() - start)
    result_docs = get_documents_containing_all_terms(query_terms, term_postings) ##
    print("doc intersection:", time.time() - start)

    if result_docs:
        doc_scores = []
        cosine_time = time.time()
        print("Number of results: ",len(result_docs))


        if len(query_terms) == 1: ## we dont need to find cosine similarity in single token words
            term = query_terms[0]
            prefix = prefix_getter(term)
            position = positions[prefix][term]

            for id, tfidf, _ in read_json(index[prefix], position, term):
                if id in result_docs:
                    doc_scores.append((id, tfidf))
        else:
            result_docs = list(result_docs)[:25]
            result_docs = set(result_docs)
            for doc_id in result_docs:
                doc_time = time.time()
                doc_vector = get_document_vector(doc_id, query_terms, index, positions, term_postings)
                print("DOC TIME: ", time.time() - doc_time)
                # cos_simtime = time.time()
                similarity = cosine_similarity(query_vector, doc_vector)
                # print("cos sim time:", time.time() - cos_simtime)
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
        return word[0]
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
