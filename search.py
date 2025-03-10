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
ngram_index_directory = os.path.join(os.getcwd(), "ngram_index")
position_index_directory = os.path.join(os.getcwd(), "position_index")
stemmer = PorterStemmer()
tokenizer = nltk.tokenize.RegexpTokenizer(r'[a-zA-Z0-9]+')
stopWords = set(stopwords.words('english')) 
weights = {
    't': 1.5,
    'a': 1.4,
    'h1': 1.3,
    'h2': 1.2,
    'b': 1.1,
    'n': 1.0,
    '2gram': 1.2,
    '3gram': 1.3
}
term_cache = {}
MAX_CACHE_SIZE = 100

def read_json(file_path, position, key):
    """Optimized JSON reader for index files"""
    file_path.seek(position)
    
    chunk_size = 10000 if key not in stopWords else 100000
    chunk = file_path.read(chunk_size)
    
    try:
        start = chunk.find('{"word":"' + key + '"')
        if start == -1:
            return []
        
        brace_level = 0
        in_string = False
        escape_next = False
        end = start
        
        for i in range(start, len(chunk)):
            char = chunk[i]
            
            if escape_next:
                escape_next = False
                continue
                
            if char == '\\':
                escape_next = True
            elif char == '"' and not escape_next:
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    brace_level += 1
                elif char == '}':
                    brace_level -= 1
                    if brace_level == 0:
                        end = i + 1
                        break
        
        if brace_level != 0:
            posting_start = chunk.find('"postings":[[', start)
            if posting_start != -1:
                posting_end = chunk.find(']]', posting_start)
                if posting_end != -1:
                    word_json = f'{{"word":"{key}","postings":{chunk[posting_start+11:posting_end+2]}}}'
                    data = json.loads(word_json)
                    return data.get("postings", [])
            return []
            
        word_json = chunk[start:end]
        data = json.loads(word_json)
        return data.get("postings", [])
        
    except json.JSONDecodeError:
        try:

            posting_start = chunk.find('"postings":[[', start)
            if posting_start != -1:
                posting_end = chunk.find(']]', posting_start)
                if posting_end != -1:
                    posting_json = f'{{"postings":{chunk[posting_start+11:posting_end+2]}}}'
                    data = json.loads(posting_json)
                    return data.get("postings", [])
        except:
            pass
            
    return []

def load_positions(directory=complete_index_directory, filename="positions.txt"):
    """Load position mapping file"""
    positions_file = os.path.join(directory, filename)
    if not os.path.exists(positions_file):
        return {}
        
    with open(positions_file, "r") as f:
        positions = json.load(f)
    
    position_map = {}
    for entry in positions:
        prefix = entry["prefix"]
        position_map[prefix] = entry["position"]
    
    return position_map

def load_inverted_index(directory=complete_index_directory, file_prefix="complete_index"):
    """Load inverted index files"""
    index = {}
    if not os.path.exists(directory):
        return index
        
    for file in os.listdir(directory):
        if file.startswith(file_prefix) and file.endswith(".txt"):
            name_split = file.split("_")[-1].split(".")[0]
            path = os.path.join(directory, file)
            index[name_split] = (open(path, "r"))
    return index

def get_cached_postings(index, positions, term, prefix_getter_func, index_type="standard"):
    """Get postings with caching for better performance"""
    cache_key = f"{index_type}_{term}"
    if cache_key in term_cache:
        return term_cache[cache_key]
        
    prefix = prefix_getter_func(term)
    try:
        term_position = positions[prefix]
        if term not in term_position:
            return []
            
        position = term_position[term]
        postings = read_json(index[prefix], position, term)
        
        if len(term_cache) < MAX_CACHE_SIZE:
            term_cache[cache_key] = postings
            
        return postings
    except KeyError:
        return []

def get_word_positions(word):
    """Get word position information from the position index"""
    positions = load_positions(position_index_directory, "word_positions.txt")
    position_index = load_inverted_index(position_index_directory, "position_index")
    
    prefix = prefix_getter(word)
    try:
        term_position = positions[prefix]
        if word not in term_position:
            return {}
            
        position = term_position[word]
        postings = read_json(position_index[prefix], position, word)
        
        result = {}
        for posting in postings:
            doc_id = posting[0]
            positions_list = posting[1]
            result[doc_id] = positions_list
            
        return result
    except KeyError:
        return {}

def proximity_search(term1, term2, max_distance=5):
    """Find documents where term1 and term2 appear within max_distance words of each other"""
    positions1 = get_word_positions(term1)
    positions2 = get_word_positions(term2)
    
    result_docs = set()
    
    common_docs = set(positions1.keys()) & set(positions2.keys())
    
    for doc_id in common_docs:
        pos1 = [p[1] for p in positions1[doc_id]]
        pos2 = [p[1] for p in positions2[doc_id]]
        
        for p1 in pos1:
            for p2 in pos2:
                if abs(p1 - p2) <= max_distance:
                    result_docs.add(doc_id)
                    break
            if doc_id in result_docs:
                break
    
    return result_docs

def cosine_similarity(vector1, vector2):
    """Calculate cosine similarity between two vectors"""
    v1 = np.array(vector1, dtype=np.float32)
    v2 = np.array(vector2, dtype=np.float32)
    
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
        
    return dot / (norm1 * norm2)

def get_query_vector(query_terms, index, positions):
    """Get vector representation of query and term postings"""
    term_postings = {}
    query_vector = []
    cached_postings = {}
    
    for term in query_terms:
        postings = get_cached_postings(index, positions, term, prefix_getter)
        if not postings:
            query_vector.append(0)
            continue
            
        term_postings[term] = set((doc_id, tfidf, text_type) for doc_id, tfidf, text_type in postings)
        
        tf_idf_sum = sum(tfidf for _, tfidf, _ in postings)
        average_tf_idf = tf_idf_sum / len(postings) if postings else 0
        query_vector.append(average_tf_idf)
        cached_postings[term] = postings
            
    return query_vector, term_postings, cached_postings

def get_ngram_postings(query_terms, ngram_index, ngram_positions):
    """Get n-gram postings for the query terms"""
    ngram_postings = {}
    
    if len(query_terms) >= 2:
        for i in range(len(query_terms) - 1):
            bigram = f"{query_terms[i]}_{query_terms[i+1]}"
            postings = get_cached_postings(ngram_index, ngram_positions, bigram, prefix_getter, "ngram")
            if postings:
                ngram_postings[bigram] = set((doc_id, tfidf, gram_type) for doc_id, tfidf, gram_type in postings)
    
    if len(query_terms) >= 3:
        for i in range(len(query_terms) - 2):
            trigram = f"{query_terms[i]}_{query_terms[i+1]}_{query_terms[i+2]}"
            postings = get_cached_postings(ngram_index, ngram_positions, trigram, prefix_getter, "ngram")
            if postings:
                ngram_postings[trigram] = set((doc_id, tfidf, gram_type) for doc_id, tfidf, gram_type in postings)
    
    return ngram_postings

def get_documents_containing_all_terms(query_terms, term_postings):
    """Find documents containing all query terms"""
    if not query_terms or not any(term in term_postings for term in query_terms):
        return set()
        
    valid_terms = [term for term in query_terms if term in term_postings]
    if not valid_terms:
        return set()
        
    sorted_terms = sorted(valid_terms, key=lambda term: len(term_postings.get(term, [])))
    
    result_docs = set(doc_id for doc_id, _, _ in term_postings.get(sorted_terms[0], []))
    
    if not result_docs:
        return set()
        
    for term in sorted_terms[1:]:
        term_docs = set(doc_id for doc_id, _, _ in term_postings.get(term, []))
        result_docs &= term_docs
        if not result_docs:
            break
            
    return result_docs

def pad_vectors(query_vector, doc_vector):
    """Ensure vectors are of the same length for similarity calculation"""
    max_len = max(len(query_vector), len(doc_vector))
    
    if len(query_vector) < max_len:
        query_vector = query_vector + [0] * (max_len - len(query_vector))
    if len(doc_vector) < max_len:
        doc_vector = doc_vector + [0] * (max_len - len(doc_vector))
    
    return query_vector, doc_vector

def raw_tfidf_ranking(query_terms, term_postings, ngram_postings, result_docs, query_vector):
    """Rank documents based on TF-IDF and n-grams"""
    doc_scores = {}
    results = []
    
    batch_size = 500
    result_docs_list = list(result_docs)
    
    for i in range(0, len(result_docs_list), batch_size):
        batch_docs = set(result_docs_list[i:i+batch_size])
        batch_scores = {}
        
        for term in query_terms:
            postings = term_postings.get(term, [])
            for doc_id, tfidf, style in postings:
                if doc_id not in batch_docs:
                    continue
                    
                weighted_tfidf = tfidf * weights.get(style, 1.0)
                
                if doc_id not in batch_scores:
                    batch_scores[doc_id] = [weighted_tfidf]
                else:
                    batch_scores[doc_id].append(weighted_tfidf)
        
        for ngram, postings in ngram_postings.items():
            for doc_id, tfidf, gram_type in postings:
                if doc_id not in batch_docs:
                    continue
                
                weighted_tfidf = tfidf * weights.get(gram_type, 1.2)
                
                if doc_id not in batch_scores:
                    batch_scores[doc_id] = [weighted_tfidf]
                else:
                    batch_scores[doc_id].append(weighted_tfidf)
        
        for doc_id, doc_vector in batch_scores.items():
            padded_query_vector, padded_doc_vector = pad_vectors(query_vector, doc_vector)
            
            sim = cosine_similarity(padded_query_vector, padded_doc_vector)
            results.append((doc_id, sim))
    
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def index_getter(index, positions, input_query, docID_url):
    """Process user query and return ranked results"""
    start = time.time()

    ngram_index = load_inverted_index(ngram_index_directory, "complete_index")
    ngram_positions = load_positions(ngram_index_directory)

    query_terms = tokenizer.tokenize(input_query)
    query_terms = [stemmer.stem(term.lower()) for term in query_terms]
    
    if not query_terms:
        print("Empty query")
        run(index, positions, docID_url)
        return []

    query_vector, term_postings, cached_postings = get_query_vector(query_terms, index, positions)
    
    ngram_postings = {}
    if ngram_index:
        ngram_postings = get_ngram_postings(query_terms, ngram_index, ngram_positions)
    
    proximity_docs = set()
    if len(query_terms) > 1:
        for i in range(len(query_terms) - 1):
            proxim_results = proximity_search(query_terms[i], query_terms[i+1])
            if proxim_results:
                if not proximity_docs:
                    proximity_docs = proxim_results
                else:
                    proximity_docs = proximity_docs.union(proxim_results)
    
    result_docs = get_documents_containing_all_terms(query_terms, term_postings)
    
    if not result_docs and proximity_docs:
        result_docs = proximity_docs
        print("No exact matches found. Using proximity search results.")
    
    query_time = time.time() - start
    
    if not result_docs:
        print("No results found")
        run(index, positions, docID_url)
        return []
    
    max_docs = 2000
    if len(result_docs) > max_docs:
        print(f"Limiting ranking to top {max_docs} documents")

        result_docs = set(sorted(list(result_docs))[:max_docs])
    
    print(f"Number of results: {len(result_docs)}")
    
    if len(query_terms) == 1 and not ngram_postings:
        term = query_terms[0]
        postings = get_cached_postings(index, positions, term, prefix_getter)
        doc_scores = []
        
        for id, tfidf, style in postings:
            if id in result_docs:
                weighted_tfidf = tfidf * weights.get(style, 1.0)
                doc_scores.append((id, weighted_tfidf))
        
        doc_scores.sort(key=lambda x: x[1], reverse=True)
    else:
        doc_scores = raw_tfidf_ranking(query_terms, term_postings, ngram_postings, result_docs, query_vector)
    
    end = time.time()
    total_time = end - start
    
    print(f"\nTop results for query: '{input_query}'")
    for doc_id, score in doc_scores[:5]:
        print(f"{doc_id}: {docID_url[str(doc_id)]} (Score: {score:.4f})")
    
    print(f"\nQuery took: {total_time:.4f} seconds")
    
    run(index, positions, docID_url)
    return doc_scores

def prefix_getter(word):
    """Determine the prefix for a token"""
    if not word:
        return "nonalpha"
        
    if re.match(r'^[0-9]+$', word[0]):
        return word[0]
    elif word[0] in "abcdefghijklmnopqrstuvwxyz":
        return word[0]
    else:
        return "nonalpha"

def take_input():
    """Get user input"""
    input1 = input("\nEnter search query (or 'q' to quit): ")
    if input1.lower() == 'q':
        exit()
    return input1

def run(index, positions, docID_url):
    """Run the search interface"""
    input_query = take_input()
    index_getter(index, positions, input_query, docID_url)

def main():
    """Main function to start the search engine"""
    print("Loading search engine...")
    index = load_inverted_index()
    positions = load_positions()
    
    with open("docID_url_map.txt", "r") as f:
        docID_url = json.load(f)
    
    print("\n===== UCI ICS Search Engine =====")
    print("Features enabled:")
    print("- TF-IDF ranking with importance weighting")
    print("- Word position indexing for proximity search")
    
    if os.path.exists(ngram_index_directory):
        print("- N-gram indexing for phrase queries")
    
    print("\nType 'q' to quit at any time.")
    run(index, positions, docID_url)

if __name__ == "__main__":
    main()
