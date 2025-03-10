# cs121_A3

## 1. How to Run the Indexer?
1. Open the index.py file
2. Edit the path to the corpus at the end of the file
3. Run the index.py file

## 2. How to Run the GUI?
To start the search engine interface with the graphical user interface (GUI), run:
```python gui.py```

### A. Performing a Search
  Enter a Query: Type your search query in the input box.
  
  Click "Search": The search engine will process the query and display the top results.
  
  Pagination: Use the "Next" and "Previous" buttons to navigate through results (50 per page).
  
  Query Time: The time taken for the search will be displayed at the bottom.


### B. Search Functionality
  The search function uses an inverted index and positional index to retrieve relevant documents.
  
  Results are ranked using TF-IDF scoring.
  
  Each result shows the document ID, corresponding URL, and relevance score.
