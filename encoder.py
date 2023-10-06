import h5py  
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import os
os.chdir('C:\DiscordBot_VB')
# Load the Universal Sentence Encoder model
module_path = "C:\\DiscordBot_VB\\universal-sentence-encoder_4"
model = hub.load(module_path)

# Function to compute the embedding for a given keyword
def compute_embedding(keyword):
    # Compute the embedding using the USE model
    embedding = np.array(model([keyword])[0])
    return embedding

# Function to compute the similarity score between two embeddings
def compute_similarity(embedding1, embedding2):
    # Compute the cosine similarity between the embeddings
    similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
    return similarity

# Create a DataFrame with 'Restaurant Name' and 'Keyword' columns
df = pd.read_csv('C:\\DiscordBot_VB\\embeddings.csv')
df = df.drop('Embedding', axis=1)

# Create an empty column 'Embedding' to store the numpy arrays
df['Embedding'] = None
# Compute and store the embeddings in the 'Embedding' column
for i, keyword in enumerate(df['Keyword']):
    # Compute the embedding for the keyword
    embedding = compute_embedding(keyword)
    df.at[i, 'Embedding'] = embedding

# Query
query = "microbrewery"
query_embedding = compute_embedding(query)

# Compute the similarity score between the query embedding and the embeddings in the DataFrame
df['Similarity'] = df['Embedding'].apply(lambda emb: compute_similarity(query_embedding, emb))

# Print the DataFrame with similarity scores
# Sort the DataFrame based on similarity score in descending order
df = df.sort_values(by='Similarity', ascending=False)

# Drop rows with the same restaurant name and keyword
df.drop_duplicates(subset=['Restaurant Name', 'Keyword'], inplace=True)

# Drop the 'Embedding' column
df = df.drop('Embedding', axis=1)
total_sim=df.groupby('Restaurant Name')["Similarity"].sum()
print(total_sim)
# Save the DataFrame as a CSV file
df.to_csv('output.csv', index=False)
