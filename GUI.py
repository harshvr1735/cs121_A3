import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
import os
import threading
import time
from searcher2 import load_inverted_index, load_positions, search_query

class EnhancedGUI:
    def __init__(self, root, index, positions, docID_url):
        self.root = root
        self.index = index
        self.positions = positions
        self.docID_url = docID_url
        self.current_page = 0
        self.results_per_page = 10
        self.all_results = []
        self.last_query = ""
        self.search_running = False
        
        self.has_ngrams = os.path.exists("ngram_index")
        self.has_positions = os.path.exists("position_index")
        
        self.create_widgets()

    def create_widgets(self):
        self.root.title("UCI ICS Search Engine")
        self.root.geometry("900x700")
        self.root.minsize(800, 600)
        
        style = ttk.Style()
        style.configure("TFrame", background="#f5f5f5")
        style.configure("TLabel", background="#f5f5f5", font=("Arial", 11))
        style.configure("Header.TLabel", font=("Arial", 14, "bold"))
        style.configure("Result.TLabel", font=("Arial", 12))
        style.configure("URL.TLabel", font=("Arial", 10), foreground="blue")
        style.configure("TButton", font=("Arial", 11))
        
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        title_label = ttk.Label(main_frame, text="UCI ICS Search Engine", style="Header.TLabel")
        title_label.pack(pady=(10, 5))
        
        features_text = "Features: TF-IDF Ranking"
        if self.has_ngrams:
            features_text += " • N-gram Indexing"
        if self.has_positions:
            features_text += " • Word Position Indexing"
            
        features_label = ttk.Label(main_frame, text=features_text)
        features_label.pack(pady=(0, 10))
        
        search_frame = ttk.Frame(main_frame)
        search_frame.pack(fill=tk.X, pady=10)
        
        search_label = ttk.Label(search_frame, text="Search query:")
        search_label.pack(side=tk.LEFT, padx=(0, 5))
        
        self.search_entry = ttk.Entry(search_frame, width=70, font=("Arial", 11))
        self.search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        self.search_entry.bind("<Return>", lambda e: self.on_search())
        
        search_button = ttk.Button(search_frame, text="Search", command=self.on_search)
        search_button.pack(side=tk.LEFT)
        
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X, pady=(5, 10))
        
        self.result_count_label = ttk.Label(status_frame, text="Enter a search query to begin")
        self.result_count_label.pack(side=tk.LEFT)
        
        self.query_time_label = ttk.Label(status_frame, text="")
        self.query_time_label.pack(side=tk.RIGHT)
        
        self.results_text = tk.Text(results_frame, wrap=tk.WORD, width=80, height=20, 
                                  font=("Arial", 11), padx=10, pady=10)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.results_text.config(state=tk.DISABLED)
        
        scrollbar = ttk.Scrollbar(results_frame, command=self.results_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        
        nav_frame = ttk.Frame(main_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.prev_button = ttk.Button(nav_frame, text="Previous", command=self.prev_page)
        self.prev_button.pack(side=tk.LEFT)
        self.prev_button.config(state=tk.DISABLED)
        
        self.page_label = ttk.Label(nav_frame, text="Page 1")
        self.page_label.pack(side=tk.LEFT, padx=10)
        
        self.next_button = ttk.Button(nav_frame, text="Next", command=self.next_page)
        self.next_button.pack(side=tk.LEFT)
        self.next_button.config(state=tk.DISABLED)
        
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.progress = ttk.Progressbar(bottom_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.LEFT, padx=(0, 10))
        
        export_button = ttk.Button(bottom_frame, text="Export Results", command=self.export_results)
        export_button.pack(side=tk.LEFT, padx=5)
        
        about_button = ttk.Button(bottom_frame, text="About", command=self.show_about)
        about_button.pack(side=tk.LEFT, padx=5)
        
        quit_button = ttk.Button(bottom_frame, text="Quit", command=self.root.quit)
        quit_button.pack(side=tk.RIGHT)
        
        self.results_text.tag_configure("url", foreground="blue", underline=1)
        self.results_text.tag_bind("url", "<Button-1>", self.open_url)
        self.results_text.tag_bind("url", "<Enter>", lambda e: self.results_text.config(cursor="hand2"))
        self.results_text.tag_bind("url", "<Leave>", lambda e: self.results_text.config(cursor=""))

    def on_search(self):
        """Handle search button click"""
        query = self.search_entry.get().strip()
        if not query:
            messagebox.showwarning("Input Error", "Please enter a query to search.")
            return
            
        if self.search_running:
            messagebox.showinfo("Search in Progress", "A search is already running. Please wait.")
            return
            
        self.last_query = query
        self.current_page = 0
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "Searching...\n\n")
        self.results_text.config(state=tk.DISABLED)
        
        self.search_running = True
        self.progress.start()
        
        threading.Thread(target=self.perform_search, args=(query,), daemon=True).start()

    def perform_search(self, query):
        """Perform the search in a background thread"""
        try:
            doc_scores, query_time = search_query(query, self.index, self.positions, self.docID_url)
            
            self.root.after(0, lambda: self.update_results(doc_scores, query_time))
        except Exception as e:
            self.root.after(0, lambda: self.show_error(f"Search error: {str(e)}"))
        finally:
            self.root.after(0, self.search_complete)

    def update_results(self, doc_scores, query_time):
        """Update UI with search results"""
        self.all_results = [(doc_id, score) for doc_id, score in doc_scores]
        self.query_time_label.config(text=f"Query time: {query_time:.4f} seconds")
        
        if self.all_results:
            self.result_count_label.config(text=f"Found {len(self.all_results)} results for '{self.last_query}'")
            self.display_results()
        else:
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.insert(tk.END, f"No results found for '{self.last_query}'.\n\n")
            self.results_text.insert(tk.END, "Tips for better results:\n")
            self.results_text.insert(tk.END, "• Try using more general terms\n")
            self.results_text.insert(tk.END, "• Check for spelling mistakes\n")
            self.results_text.insert(tk.END, "• Try related keywords\n")
            self.results_text.config(state=tk.DISABLED)
            self.result_count_label.config(text=f"No results for '{self.last_query}'")

    def search_complete(self):
        """Clean up after search completes"""
        self.progress.stop()
        self.search_running = False

    def show_error(self, message):
        """Display error message"""
        messagebox.showerror("Error", message)
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Error: {message}\n")
        self.results_text.config(state=tk.DISABLED)

    def display_results(self):
        """Display current page of results"""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        start_index = self.current_page * self.results_per_page
        end_index = min(start_index + self.results_per_page, len(self.all_results))
        
        results_to_display = self.all_results[start_index:end_index]
        
        for i, (doc_id, score) in enumerate(results_to_display, 1):
            result_num = start_index + i
            doc_id_str = str(doc_id)
            url = self.docID_url[doc_id_str]
            
            self.results_text.insert(tk.END, f"{result_num}. ", "result_num")
            self.results_text.insert(tk.END, f"(Score: {score:.4f})\n", "score")
            
            self.results_text.insert(tk.END, f"{url}\n", "url")
            end_pos = self.results_text.index(tk.END)
            line = int(float(end_pos)) - 1
            self.results_text.tag_add("url", f"{line}.0", f"{line}.end")
            
            if i < len(results_to_display):
                self.results_text.insert(tk.END, "\n" + "-"*80 + "\n\n")
            
        self.results_text.config(state=tk.DISABLED)
        
        self.page_label.config(text=f"Page {self.current_page + 1} of {max(1, (len(self.all_results) + self.results_per_page - 1) // self.results_per_page)}")
        
        if self.current_page == 0:
            self.prev_button.config(state=tk.DISABLED)
        else:
            self.prev_button.config(state=tk.NORMAL)
            
        if end_index >= len(self.all_results):
            self.next_button.config(state=tk.DISABLED)
        else:
            self.next_button.config(state=tk.NORMAL)

    def next_page(self):
        """Go to next page of results"""
        if (self.current_page + 1) * self.results_per_page < len(self.all_results):
            self.current_page += 1
            self.display_results()

    def prev_page(self):
        """Go to previous page of results"""
        if self.current_page > 0:
            self.current_page -= 1
            self.display_results()

    def open_url(self, event):
        """Open URL in external browser (placeholder - would need a web browser module)"""
        index = self.results_text.index(f"@{event.x},{event.y}")
        line = int(float(index))
        
        line_content = self.results_text.get(f"{line}.0", f"{line}.end")
        
        messagebox.showinfo("Open URL", f"URL clicked: {line_content}\n\nIn a real browser, this would open the URL.")
        # To actually open a URL, you would use:
        # import webbrowser
        # webbrowser.open(line_content)

    def export_results(self):
        """Export search results to a file"""
        if not self.all_results:
            messagebox.showinfo("No Results", "There are no results to export.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Export Search Results"
        )
        
        if not file_path:
            return
            
        try:
            with open(file_path, 'w') as file:
                file.write(f"Search Query: {self.last_query}\n")
                file.write(f"Found {len(self.all_results)} results\n\n")
                
                for i, (doc_id, score) in enumerate(self.all_results, 1):
                    doc_id_str = str(doc_id)
                    file.write(f"{i}. Score: {score:.4f}\n")
                    file.write(f"   URL: {self.docID_url[doc_id_str]}\n\n")
                    
            messagebox.showinfo("Export Successful", f"Results exported to {file_path}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting results: {str(e)}")

    def show_about(self):
        """Show about dialog"""
        about_text = "UCI ICS Search Engine\n\n"
        about_text += "A search engine developed for INF 141 Assignment 3\n\n"
        about_text += "Features:\n"
        about_text += "• TF-IDF ranking with importance weighting\n"
        if self.has_ngrams:
            about_text += "• N-gram indexing for phrase queries\n"
        if self.has_positions:
            about_text += "• Word position indexing for proximity search\n"
        
        messagebox.showinfo("About", about_text)

def run_gui():
    """Main function to start the GUI"""
    index = load_inverted_index()
    positions = load_positions()
    
    try:
        with open("docID_url_map.txt", "r") as f:
            docID_url = json.load(f)
    except FileNotFoundError:
        messagebox.showerror("Error", "Could not find docID_url_map.txt. Please run the indexer first.")
        return
    
    # Create and run GUI
    root = tk.Tk()
    root.title("UCI ICS Search Engine")
    gui = EnhancedGUI(root, index, positions, docID_url)
    root.mainloop()

if __name__ == "__main__":
    run_gui()
