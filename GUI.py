import tkinter as tk
from tkinter import messagebox
import json
from searcher2 import load_inverted_index, load_positions, search_query

class GUI:
    def __init__(self, root, index, positions, docID_url):
        self.root = root
        self.index = index
        self.positions = positions
        self.docID_url = docID_url
        self.current_page = 0
        self.results_per_page = 50
        self.all_results = []
        
        self.create_widgets()

    def create_widgets(self):
        self.search_label = tk.Label(self.root, text="Enter your search query:")
        self.search_label.pack()

        self.search_entry = tk.Entry(self.root, width=50)
        self.search_entry.pack()
        self.search_button = tk.Button(self.root, text="Search", command=self.on_search)
        self.search_button.pack()

        self.results = tk.Listbox(self.root, width=100, height=20)
        self.results.pack()

        self.prev = tk.Button(self.root, text="Previous", command=self.prev_page)
        self.prev.pack(side=tk.LEFT)
        self.next = tk.Button(self.root, text="Next", command=self.next_page)
        self.next.pack(side=tk.LEFT, padx=10)

        self.quit = tk.Button(self.root, text="Quit", command=self.root.quit)
        self.quit.pack(side=tk.RIGHT, pady=10)

        self.query_time_label = tk.Label(self.root, text="Query Time: N/A")
        self.query_time_label.pack(side=tk.BOTTOM, padx=10, pady=10)

    def on_search(self):
        query = self.search_entry.get()
        if query:
            self.current_page = 0
            self.results.delete(0, tk.END)
            self.search(query)
        else:
            messagebox.showwarning("Input Error", "Please enter a query to search.")

    def search(self, query):
        doc_scores, query_time = search_query(query, self.index, self.positions, self.docID_url)
        self.query_time_label.config(text=f"Query Time: {query_time:.4f} seconds")

        if doc_scores:
            self.all_results = [(doc_id, score) for doc_id, score in doc_scores]
            self.display_results()
        else:
            messagebox.showinfo("No Results", "No documents matched your query.")

    def display_results(self):
        self.results.delete(0, tk.END)
        start_index = self.current_page * self.results_per_page
        end_index = start_index + self.results_per_page

        results_to_display = self.all_results[start_index:end_index]

        for doc_id, score in results_to_display:
            self.results.insert(tk.END, f"{doc_id}: {self.docID_url[str(doc_id)]} (Score: {score:.4f})")

        if self.current_page == 0:
            self.prev.config(state=tk.DISABLED)
        else:
            self.prev.config(state=tk.NORMAL)

        if end_index >= len(self.all_results):
            self.next.config(state=tk.DISABLED)
        else:
            self.next.config(state=tk.NORMAL)

    def next_page(self):
        if (self.current_page + 1) * self.results_per_page < len(self.all_results):
            self.current_page += 1
            self.display_results()

    def prev_page(self):
        if self.current_page > 0:
            self.current_page -= 1
            self.display_results()

def run_gui():
    index = load_inverted_index()
    positions = load_positions()
    with open("docID_url_map.txt", "r") as f:
        docID_url = json.load(f)

    root = tk.Tk()
    root.title("Search Results")
    gui = GUI(root, index, positions, docID_url)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
