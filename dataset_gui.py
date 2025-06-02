import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = None

# Load dataset
def load_csv():
    global data
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            data = pd.read_csv(file_path)
            preprocess_data()
            messagebox.showinfo("Success", "Dataset loaded and preprocessed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")

# Preprocess
def preprocess_data():
    global data
    data['Release_Date'] = pd.to_datetime(data['Release_Date'], errors='coerce')
    data['Director'].fillna('No Director', inplace=True)
    data['Cast'].fillna('No Cast', inplace=True)
    data['Country'].fillna('Unknown Country', inplace=True)
    data['Release_Date'].fillna(method='ffill', inplace=True)
    data['Rating'].fillna('Unknown', inplace=True)
    data['Duration'].fillna('Unknown', inplace=True)
    data['Year'] = data['Release_Date'].dt.year

def show_data():
    if data is not None:
        top = tk.Toplevel()
        top.title("First 10 Rows of Dataset")
        text = scrolledtext.ScrolledText(top, width=130, height=20)
        text.pack()
        text.insert(tk.END, str(data.head(10)))
    else:
        messagebox.showwarning("Warning", "Please load a dataset first.")

def show_columns_info():
    if data is not None:
        top = tk.Toplevel()
        top.title("Columns and Dataset Info")
        text = scrolledtext.ScrolledText(top, width=130, height=20)
        text.pack()
        info = "Shape of dataset: " + str(data.shape) + "\n\n"
        info += "Column Names:\n" + str(data.columns.tolist()) + "\n\n"
        info += "Data Types:\n" + str(data.dtypes)
        text.insert(tk.END, info)
    else:
        messagebox.showwarning("Warning", "Please load a dataset first.")

# Visualization functions
def plot_category_distribution():
    data['Category'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
    plt.title("Distribution of Category (Movie vs TV Show)")
    plt.xlabel("Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def plot_titles_over_years():
    data['Year'].value_counts().sort_index().plot(kind='line', marker='o', color='teal')
    plt.title("Number of Netflix Titles Over the Years")
    plt.xlabel("Year")
    plt.ylabel("Number of Releases")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_top_countries():
    top_countries = data['Country'].value_counts().head(10)
    plt.barh(top_countries.index, top_countries.values, color='forestgreen')
    plt.title("Top 10 Countries by Number of Titles")
    plt.tight_layout()
    plt.show()

def plot_rating_distribution():
    data['Rating'].value_counts().plot(kind='barh', color='purple')
    plt.title("Distribution of Ratings")
    plt.tight_layout()
    plt.show()

def plot_top_directors():
    top_directors = data[data['Director'] != 'No Director']['Director'].value_counts().head(10)
    top_directors.plot(kind='barh', color='darkblue')
    plt.title("Top 10 Directors by Number of Titles")
    plt.tight_layout()
    plt.show()

def plot_movies_vs_shows():
    cat_by_year = data.groupby(['Year', 'Category']).size().unstack()
    cat_by_year.plot(kind='area', stacked=False, alpha=0.7)
    plt.title("Movies vs TV Shows Over the Years")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_movie_duration():
    movies = data[data['Category'] == 'Movie'].copy()
    movies['Minutes'] = movies['Duration'].str.replace(' min', '').replace('Unknown', np.nan).astype(float)
    plt.hist(movies['Minutes'].dropna(), bins=30, color='coral', edgecolor='black')
    plt.title("Distribution of Movie Durations")
    plt.tight_layout()
    plt.show()

def plot_top_cast():
    cast_series = data['Cast'].dropna().str.split(', ')
    cast_flat = [person for sublist in cast_series for person in sublist]
    top_cast = pd.Series(cast_flat).value_counts().head(10)
    top_cast.plot(kind='barh', color='orange')
    plt.title("Top 10 Most Frequent Cast Members")
    plt.tight_layout()
    plt.show()

def plot_unknown_data():
    unknown_counts = (data == 'Unknown').sum()
    unknown_perc = 100 * unknown_counts / len(data)
    unknown_perc[unknown_perc > 0].sort_values().plot(kind='barh', color='grey')
    plt.title("Percentage of 'Unknown' Values per Column")
    plt.tight_layout()
    plt.show()

def plot_correlation_heatmap():
    num_cols = data.select_dtypes(include=['float64', 'int64'])
    plt.imshow(num_cols.corr(), cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_growth():
    growth = data.groupby(['Year', 'Category']).size().unstack()
    growth.fillna(0, inplace=True)
    growth.plot(kind='bar', stacked=True, figsize=(14,6), color=['teal', 'orange'])
    plt.title("Netflix's Growth Over Years")
    plt.tight_layout()
    plt.show()

def plot_top_genres():
    genres = data['Listed_In'].dropna().str.split(', ')
    flat_genres = [item for sublist in genres for item in sublist]
    top_genres = pd.Series(flat_genres).value_counts().head(10)
    top_genres.plot(kind='barh', color='purple')
    plt.title("Top 10 Netflix Genres")
    plt.tight_layout()
    plt.show()

# GUI setup
root = tk.Tk()
root.title("Netflix Dataset Visualizer")
root.geometry("800x1000")

# Add background image
bg_img = Image.open(r"E:\Netflix-dataset\netflix_bg.jpg")
bg_img = bg_img.resize((800, 1000))
bg_photo = ImageTk.PhotoImage(bg_img)

bg_label = tk.Label(root, image=bg_photo)
bg_label.place(x=0, y=0, relwidth=1, relheight=1)

frame = tk.Frame(root, bg="white", padx=10, pady=10)
frame.place(x=50, y=50)

# Buttons
buttons = [
    ("1. Load Netflix CSV", load_csv),
    ("2. Show First 10 Rows", show_data),
    ("3. Show Column Info", show_columns_info),
    ("4. Category Distribution", plot_category_distribution),
    ("5. Titles Over Years", plot_titles_over_years),
    ("6. Top Countries", plot_top_countries),
    ("7. Rating Distribution", plot_rating_distribution),
    ("8. Top Directors", plot_top_directors),
    ("9. Movies vs TV Shows", plot_movies_vs_shows),
    ("10. Movie Duration", plot_movie_duration),
    ("11. Top Cast Members", plot_top_cast),
    ("12. Unknown Data %", plot_unknown_data),
    ("13. Correlation Heatmap", plot_correlation_heatmap),
    ("14. Netflix Growth", plot_growth),
    ("15. Top Genres", plot_top_genres),
]

for text, command in buttons:
    tk.Button(frame, text=text, command=command, width=40, height=2, bg="black", fg="white").pack(pady=4)

root.mainloop()