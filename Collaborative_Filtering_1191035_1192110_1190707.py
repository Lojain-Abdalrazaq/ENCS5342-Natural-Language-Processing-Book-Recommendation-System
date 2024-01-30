# Books Recommendation System Using Collaborative Filtering
# Nour Rabee' 1191035
# Najwa Bsharat 1192110
# Lojain Abdalrazaq 1190707
# NOTE:The project was run using JUPYTER NOTEBOOK 
# NOTE:This script represents BUILDING THE COLLABRITIVE FILTERING SYSTEM

# In[1]:
import pandas as pd
my_books = pd.read_csv("liked_books.csv", index_col=0)
my_books["book_id"] = my_books["book_id"].astype(str)
my_books


book_set = set(my_books["book_id"])
book_set


# In[3]:

csv_book_mapping = {}

with open("book_id_map.csv", "r") as f:
    while True:
        line = f.readline()
        if not line:
            break
        csv_id, book_id = line.strip().split(",")
        csv_book_mapping[csv_id] = book_id
csv_book_mapping


# In[4]:

overlap_users = {}

with open("goodreads_interactions.csv", 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        user_id, csv_id, _, rating, _ = line.split(",")
        
        book_id = csv_book_mapping.get(csv_id)
        
        if book_id in book_set:
            if user_id not in overlap_users:
                overlap_users[user_id] = 1
            else:
                overlap_users[user_id] += 1


# In[7]:

len(overlap_users)


# In[8]:

filtered_overlap_users = set([k for k in overlap_users if overlap_users[k] > my_books.shape[0]/5])
len(filtered_overlap_users)


# In[9]:

len(filtered_overlap_users)


# In[10]:

interactions_list = []

with open("goodreads_interactions.csv", 'r') as f:
    while True:
        line = f.readline()
        if not line:
            break
        user_id, csv_id, _, rating, _ = line.split(",")
        
        if user_id in filtered_overlap_users:
            book_id = csv_book_mapping[csv_id]
            interactions_list.append([user_id, book_id, rating])
len(interactions_list)


# In[11]:

interactions_list[55]


# In[12]:

interactions = pd.DataFrame(interactions_list, columns=["user_id", "book_id", "rating"])
interactions = pd.concat([my_books[["user_id", "book_id", "rating"]], interactions])
interactions


# In[13]:

interactions["book_id"] = interactions["book_id"].astype(str)
interactions["user_id"] = interactions["user_id"].astype(str)
interactions["rating"] = pd.to_numeric(interactions["rating"])
interactions["user_index"] = interactions["user_id"].astype("category").cat.codes
interactions["book_index"] = interactions["book_id"].astype("category").cat.codes


# In[14]:

len(interactions["user_index"])


# In[15]:

len(interactions["book_index"])


# In[16]:

len(interactions["user_index"])*len(interactions["book_index"])


# In[17]:

from scipy.sparse import coo_matrix

ratings_mat_coo = coo_matrix((interactions["rating"], (interactions["user_index"], interactions["book_index"])))
ratings_mat_coo


# In[18]:

ratings_mat = ratings_mat_coo.tocsr()


# In[19]:
interactions[interactions["user_id"] == "-1"]
interactions


# In[20]:

my_index = 0
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(ratings_mat[my_index,:], ratings_mat).flatten()


# In[21]:

similarity[0]


# In[22]:

similarity[4]


# In[23]:

import numpy as np

indices = np.argpartition(similarity, -15)[-15:]
indices


# In[75]:

similar_users = interactions[interactions["user_index"].isin(indices)].copy()
similar_users = similar_users[similar_users["user_id"]!="-1"]
similar_users


# In[76]:

book_recs = similar_users.groupby("book_id").rating.agg(['count', 'mean'])
book_recs


# In[77]:

books_titles = pd.read_json("books_titles.json")
books_titles["book_id"] = books_titles["book_id"].astype(str)
book_recs = book_recs.merge(books_titles, how="inner", on="book_id")
book_recs
# take frst 15 books
# [] ground_truth = [1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]????
# ? relevant? /11 --> []


# In[136]:

book_recs.head(1)


# In[79]:

def make_clickable(val):
    return '<a target="_blank" href="{}">Goodreads</a>'.format(val, val)

def show_image(val):
    return '<a href="{}"><img src="{}" width=50></img></a>'.format(val, val)

book_recs[0:10].style.format({'url': make_clickable, 'cover_image': show_image})


# In[89]:

book_recs["adjusted_count"] = book_recs["count"] * (book_recs["count"] / book_recs["ratings"])
book_recs["score"] = book_recs["mean"] * book_recs["adjusted_count"]
book_recs = book_recs[~book_recs["book_id"].isin(my_books["book_id"])]
my_books["mod_title"] = my_books["title"].str.replace("[^a-zA-Z0-9 ]", "", regex=True).str.lower()
my_books["mod_title"] = my_books["mod_title"].str.replace("\s+", " ", regex=True)
book_recs = book_recs[~book_recs["mod_title"].isin(my_books["mod_title"])]
book_recs = book_recs[book_recs["count"]>2]
book_recs = book_recs[book_recs["mean"] >=4]
top_recs = book_recs.sort_values("mean", ascending=False)


# In[90]:

def make_clickable(val):
    return '<a target="_blank" href="{}">Goodreads</a>'.format(val, val)

def show_image(val):
    return '<a href="{}"><img src="{}" width=50></img></a>'.format(val, val)

top_recs.head(10).style.format({'url': make_clickable, 'cover_image': show_image})







