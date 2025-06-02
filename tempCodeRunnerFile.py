# Importing required libraries
import pandas as pd
import numpy as np
import matplotlib as plt

# Load dataset
data = pd.read_csv("netflixdataset.csv")

print("Data loaded successfully.")

# Display shape and first few rows
print(f"\nShape of dataset: {data.shape}")
print(data.head())

# Convert 'Release_Date' to datetime
data['Release_Date'] = pd.to_datetime(data['Release_Date'], errors='coerce')

# Basic info and null values
print("\nDataset Info:")
print(data.info())

print("\nNull values:")
print(data.isnull().sum())

# Fill missing values with placeholders (for visualization clarity)
data['Director'].fillna('No Director', inplace=True)
data['Cast'].fillna('No Cast', inplace=True)
data['Country'].fillna('Unknown Country', inplace=True)
data['Release_Date'].fillna(method='ffill', inplace=True)
data['Rating'].fillna('Unknown', inplace=True)
data['Duration'].fillna('Unknown', inplace=True)

# -----------------------------
# 1. Distribution of Category
# -----------------------------
plt.figure(figsize=(6,4))
data['Category'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Distribution of Category (Movie vs TV Show)")
plt.xlabel("Category")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# -----------------------------
# 2. Number of Titles over Years
# -----------------------------
data['Year'] = data['Release_Date'].dt.year

plt.figure(figsize=(12,6))
data['Year'].value_counts().sort_index().plot(kind='line', marker='o', color='teal')
plt.title("Number of Netflix Titles Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Releases")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 3. Top 10 Countries Producing Titles
# -----------------------------
top_countries = data['Country'].value_counts().head(10)

plt.figure(figsize=(10,5))
plt.barh(top_countries.index, top_countries.values, color='forestgreen')
plt.title("Top 10 Countries by Number of Titles")
plt.xlabel("Number of Titles")
plt.ylabel("Country")
plt.tight_layout()
plt.show()

# -----------------------------
# 4. Rating Distribution
# -----------------------------
plt.figure(figsize=(12,5))
data['Rating'].value_counts().plot(kind='barh', color='purple')
plt.title("Distribution of Ratings")
plt.xlabel("Count")
plt.ylabel("Rating")
plt.tight_layout()
plt.show()

# -----------------------------
# 5. Top 10 Directors with Most Titles
# -----------------------------
top_directors = data[data['Director'] != 'No Director']['Director'].value_counts().head(10)

plt.figure(figsize=(10,5))
top_directors.plot(kind='barh', color='darkblue')
plt.title("Top 10 Directors by Number of Titles")
plt.xlabel("Number of Titles")
plt.tight_layout()
plt.show()

# -----------------------------
# 6. Movies vs TV Shows Over Time
# -----------------------------
cat_by_year = data.groupby(['Year', 'Category']).size().unstack()

plt.figure(figsize=(12,6))
cat_by_year.plot(kind='area', stacked=False, alpha=0.7)
plt.title("Movies vs TV Shows Over the Years")
plt.xlabel("Year")
plt.ylabel("Number of Releases")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 7. Content Duration Analysis (Movies only)
# -----------------------------
movies = data[data['Category'] == 'Movie']
movies['Minutes'] = movies['Duration'].str.replace(' min', '').replace('Unknown', np.nan).astype(float)

plt.figure(figsize=(10,5))
plt.hist(movies['Minutes'].dropna(), bins=30, color='coral', edgecolor='black')
plt.title("Distribution of Movie Durations")
plt.xlabel("Duration (minutes)")
plt.tight_layout()
plt.show()

# -----------------------------
# 8. Most Frequent Cast Members
# -----------------------------
cast_series = data['Cast'].dropna().str.split(', ')
cast_flat = [person for sublist in cast_series for person in sublist]
top_cast = pd.Series(cast_flat).value_counts().head(10)

plt.figure(figsize=(10,6))
top_cast.plot(kind='barh', color='orange')
plt.title("Top 10 Most Frequent Cast Members")
plt.xlabel("Appearances")
plt.ylabel("Actor/Actress")
plt.tight_layout()
plt.show()

# -----------------------------
# 9. Percentage of Unknown Data in Columns
# -----------------------------
unknown_counts = (data == 'Unknown').sum()
unknown_perc = 100 * unknown_counts / len(data)

plt.figure(figsize=(10,5))
unknown_perc[unknown_perc > 0].sort_values().plot(kind='barh', color='grey')
plt.title("Percentage of 'Unknown' Values per Column")
plt.xlabel("Percentage")
plt.tight_layout()
plt.show()

# -----------------------------
# 10. Correlation Heatmap (Numerical)
# -----------------------------
num_cols = data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=(6,4))
plt.imshow(num_cols.corr(), cmap='Blues', interpolation='nearest')
plt.colorbar()
plt.title("Correlation Heatmap (Numerical Data)")
plt.tight_layout()
plt.show()

# -----------------------------
# 11. Netflix's Growth: Year-wise stacked bar
# -----------------------------
growth = data.groupby(['Year', 'Category']).size().unstack()
growth.fillna(0, inplace=True)

growth.plot(kind='bar', stacked=True, figsize=(14,6), color=['teal', 'orange'])
plt.title("Netflix's Growth: Movies and TV Shows by Year")
plt.xlabel("Year")
plt.ylabel("Number of Releases")
plt.tight_layout()
plt.show()

# -----------------------------
# 12. Unique Genre/Type Count (from 'Listed_In')
# -----------------------------
genres = data['Listed_In'].dropna().str.split(', ')
flat_genres = [item for sublist in genres for item in sublist]
top_genres = pd.Series(flat_genres).value_counts().head(10)

plt.figure(figsize=(10,5))
top_genres.plot(kind='barh', color='purple')
plt.title("Top 10 Content Genres on Netflix")
plt.xlabel("Count")
plt.ylabel("Genre")
plt.tight_layout()
plt.show()

# END
print("\n Analysis complete. All charts generated.")
