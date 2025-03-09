import os
import json
import re
import time
import nltk.tokenize
from nltk.stem import PorterStemmer
import numpy as np
from nltk.corpus import stopwords
## nltk.download('stopwords') uncomment for first time running program

complete_index_directory = os.path.join(os.getcwd(), "complete_index")
stemmer = PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9]+')
stopWords = set(stopwords.words('english')) 
weights = {
    't': 1.5,
    'a' : 1.4,
    'h1': 1.3,
    'h2': 1.2,
    'b': 1.1,
    'n': 1.0
}

def read_json(file_path, position, key):
    file_path.seek(position)
    if key in stopWords:
        chunk = file_path.read(200000)
    else:
        chunk = file_path.readline() 

    try:
        start = chunk.find('{"word":')
        end = chunk.find('}', start)
        if end == -1:
            last_comma = chunk.rfind(',', start)
            if last_comma != -1:
                latest_closed_bracket = chunk.rfind(']', start)
                if last_comma + 2 != latest_closed_bracket:
                    truncated_chunk = chunk[:latest_closed_bracket] + "]]}"
                else:
                    truncated_chunk = chunk[:last_comma] + "]]}"
            else:
                truncated_chunk = chunk[:start] + "]}"
        else:
            truncated_chunk = chunk[:end + 1]
        word_postings = truncated_chunk

        data = json.loads(word_postings)

        if data.get("word") == key:
            return data.get("postings", [])
        
    except json.JSONDecodeError:
        print("JSON ERROR")
        pass

    return []

def load_positions():
    positions_file = os.path.join(complete_index_directory, "positions.txt")
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
        if file.endswith(".txt"):
            name_split = file.split("_")[-1].split(".")[0]
            path = os.path.join(complete_index_directory, file)

            index[name_split] = (open(path, "r"))
    return index


#### COSINE SIMILARITY
def dot_product(vector1, vector2):
    return np.dot(vector1, vector2)

def vector_magnitude(vector):
    return np.linalg.norm(vector)    

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
    cached_postings = {}
    for term in query_terms:
        # qtime = time.time()
        prefix = prefix_getter(term)

        try:
            term_position = positions[prefix]
            if term not in term_position:
                print(f"{term} not found in index.")
                continue
            position = term_position[term]

            postings = set((doc_id, tfidf, text_type) for doc_id, tfidf, text_type in read_json(index[prefix], position, term))
            term_postings[term] = postings

            tf_idf_sum = 0 
            for _, tfidf, text_type in postings:
                tf_idf_sum += tfidf

            average_tf_idf = tf_idf_sum / len(postings)
            query_vector.append(average_tf_idf)
            cached_postings[term] = postings
            # print("qtime:", time.time() - qtime)

        except KeyError:
            query_vector.append(0)
    return query_vector, term_postings, cached_postings

def raw_tfidf_ranking(query_terms, term_postings, result_docs, query_vector):
    doc_scores = {}
    aaaaa = []
    for term in query_terms:
        postings = term_postings.get(term, [])
        for doc_id, tfidf, _ in postings:
            if doc_id not in result_docs:
                continue
                
            tfidf *= weights.get(style, 1.0)
            
            if doc_id not in doc_scores:
                doc_scores[doc_id] = [tfidf]
            else:
                doc_scores[doc_id].append(tfidf)

    for doc_id, doc_vector in doc_scores.items():
        padded_query_vector, padded_doc_vector = pad_vectors(query_vector, doc_vector)
        
        sim = cosine_similarity(padded_query_vector, padded_doc_vector)
        aaaaa.append((doc_id, sim))

    aaaaa.sort(key=lambda x: x[1], reverse=True)
    return aaaaa


def get_documents_containing_all_terms(query_terms, term_postings):
    sorted_terms = sorted(query_terms, key=lambda term: len(term_postings.get(term, [])))
    result_docs = set(doc_id for doc_id, _, _ in term_postings.get(sorted_terms[0], []))
    for term in sorted_terms[1:]:
        term_docs = set(doc_id for doc_id, _, _ in term_postings.get(term, []))
        result_docs &= term_docs
        if not result_docs:
            break

    return result_docs

def pad_vectors(query_vector, doc_vector): ## allows dot product by just throwing on extra 0s
    max_len = max(len(query_vector), len(doc_vector))
    
    if len(query_vector) < max_len:
        query_vector = query_vector + [0] * (max_len - len(query_vector))
    if len(doc_vector) < max_len:
        doc_vector = doc_vector + [0] * (max_len - len(doc_vector))
    
    return query_vector, doc_vector


def index_getter(index, positions, input, docID_url):
    start = time.time()

    query_terms = tokenizer.tokenize(input)
    query_terms = [stemmer.stem(term.lower()) for term in query_terms]

    query_vector, term_postings, cached_postings = get_query_vector(query_terms, index, positions)

    print("query_vector time: ", time.time() - start) ## .2 seconds
    result_docs = get_documents_containing_all_terms(query_terms, term_postings) ##
    end = time.time()
    if result_docs:
        doc_scores = []
        cosine_time = time.time()
        print("Number of results: ",len(result_docs))

        if len(query_terms) == 1:
            term = query_terms[0]
            prefix = prefix_getter(term)
            position = positions[prefix][term]

            for id, tfidf, _ in read_json(index[prefix], position, term):
                if id in result_docs:
                    doc_scores.append((id, tfidf))

        else:
            doc_scores = raw_tfidf_ranking(query_terms, term_postings, result_docs, query_vector)

        end = time.time()
        for doc_id, score in doc_scores[:5]:
            print(f"{doc_id}: {docID_url[str(doc_id)]} (Score: {score:.4f})")
        print("cosine time: ", time.time() - cosine_time)
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
    with open("docID_url_map.txt", "r") as f:
        docID_url = json.load(f)
    run(index, positions, docID_url)


if __name__ == "__main__":
    main()
