import os
import json
import re
import time
import nltk.tokenize
import math
from nltk.stem import PorterStemmer
import numpy as np

complete_index_directory = os.path.join(os.getcwd(), "complete_index")
stemmer = PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9]+')

def load_inverted_index():
    index = {}
    for file in os.listdir(complete_index_directory):
        if file.endswith(".json"):
            name_split = file.split("_")[-1].split(".")[0]
            # print(name_split)
            path = os.path.join(complete_index_directory, file)

            index[name_split] = (open(path, "r"))
    return index

#Cosine Similarity Calculation

#Computing the dot product of two vectors
def dot_product(vector1, vector2):
    return sum(v1 * v2 for v1, v2 in zip(vector1, vector2))

#Computing the magnitude of a vector
def vector_magnitude(vector):
    return math.sqrt(sum(v ** 2 for v in vector))

#Computing the cosine similarity between two vectors
def cosine_similarity(vector1, vector2):
    vectors_product = dot_product(vector1, vector2)
    vector1_magnitude = vector_magnitude(vector1)
    vector2_magnitude = vector_magnitude(vector2)
    if vector1_magnitude == 0 or vector2_magnitude == 0:
        return 0
    return vectors_product / (vector1_magnitude * vector2_magnitude)

#Converting query into a vector of TF-IDF scores
def get_query_vector(query_terms, index):
    query_vector = []
    for term in query_terms:
        term = stemmer.stem(term)
        try:
            file = index[prefix]
            file.seek(0)
            a = json.load(file)
            if term on a:
                query_vector.append(a[term][0][1])
            else:
                query_vector.append(0)
        except KeyError:
            query_vector.append(0)
    return query_vector

def index_getter(index, input):
    start = time.time()
    split = tokenizer.tokenize(input)
    # print(stems)
    # split = [condition.strip() for condition in input.split("AND")]
    print(split)

    results = []
    for spl in split:
        if spl:
            spl = stemmer.stem(spl)

            print(spl)
            spl = spl.lower()
            prefix = prefix_getter(index, spl)
            try:
                file = index[prefix]
                file.seek(0) ## resets to beginning of file
                a = json.load(file) ## done so we can run the same word multiple times

                with open("docID_url_map.json", "r") as f:
                    set1 = set([item[0] for item in a.get(spl, [])])
                    print(set1)
                    results.append(set1)
            except KeyError:
                print("Not found in index.")
    
    if results:
        results.sort(key=len)
        smallest = results[0] ## more efficient to AND smallest -> mid -> largest
        for res in results:
            smallest &= res ## actually TA says & doesnt work that well and returns bad results from a1
            ## need to do something else

        if smallest:
            with open("docID_url_map.json", "r") as f:
                docID_url = json.load(f)
                for doc in list(smallest)[:10]:
                    print(f"{doc}: {docID_url[str(doc)]}")

    end = time.time()
    print(end - start)
    run(index)

def prefix_getter(index, word):
    if re.match(r'^[0-9]+$', word[0]):
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
        print("invalid input") ## maybe change ?
        run(index)

def take_input():
    input1 = input("Input or q to quit: ")
    if input1.lower() == 'q':
        exit()
    return input1

def run(index):
    input = take_input()
    index_getter(index, input)

def main():
    index = load_inverted_index()
    run(index)


if __name__ == "__main__":
    main()
