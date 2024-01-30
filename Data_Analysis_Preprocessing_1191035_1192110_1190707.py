# Books Recommendation System Using Collaborative Filtering
# Nour Rabee' 1191035
# Najwa Bsharat 1192110
# Lojain Abdalrazaq 1190707
# NOTE:The project was run using JUPYTER NOTEBOOK 
# NOTE:This script represents DATA ANALYSIS AND PREPROCESSING PART

import gzip
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# In[1]:
book_metadata_file = 'goodreads_books.json.gz' # the book file name
NumOfBooks = 0   # the num of books will stored in

with gzip.open(book_metadata_file, 'rt') as file: # opening the file
    for line in file:                   # looping throughout the file to count the number of file 
        NumOfBooks += 1
        
print("The number of books in the dataset: " + str(NumOfBooks) )


# In[2]:
#printing the content book data (attributes) for book Number 50
with gzip.open(book_metadata_file, 'rt') as file:
    for i in range(50): 
        file.readline()

    attributes= file.readline()
print(attributes)

# In[3]:
# the list that we will save the parsed data of each book (each line)
parsed_data_list = []
# looping through the content of the original books metadata
with gzip.open(book_metadata_file) as f:
    for i in f:
        try:
            line_data = json.loads(i)
            ratings = int(line_data["ratings_count"])
        except (ValueError, KeyError):
            continue
        
        # if the book has more that 10 ratings (count_ratings)
        # we add the this book data to our list
        if ratings > 10:
            target = {
                "BOOK_ID": line_data["book_id"],
                "NAME": line_data["title_without_series"],
                "RATINGS_NUM": line_data["ratings_count"],
                "URL": line_data["url"],
                "BOOK_IMAGE": line_data["image_url"]
            }
            # appending the book to our list
            parsed_data_list.append(target)

# In[4]:
# printing the parsed data list
parsed_data_list

# In[5]:
processed_data_list = pd.DataFrame.from_dict(parsed_data_list)
processed_data_list

# In[6]:
# turning the RATINGS_NUM column into numerical field
processed_data_list["RATINGS_NUM"] = pd.to_numeric(processed_data_list["RATINGS_NUM"])
# removing any character that is not "a-zA-Z0-9", such as stop words, etc.
processed_data_list["ADJUSTED_NAME"] = processed_data_list["NAME"].str.replace("[^a-zA-Z0-9 ]", "", regex=True)
# converting to lower case
processed_data_list["ADJUSTED_NAME"] = processed_data_list["ADJUSTED_NAME"].str.lower()
# removing any duplicated spaces into one single space
processed_data_list["dcfADJUSTED_NAME"] = processed_data_list["ADJUSTED_NAME"].str.replace("\s+", " ", regex=True)
# checking the titles, and only get the titles with length more than 1.
processed_data_list = processed_data_list[processed_data_list["ADJUSTED_NAME"].str.len() > 0]
# and finally, turning the data in the list into json file
processed_data_list.to_json("book_titiles.json")
processed_data_list

# In[7]:

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_simihularity
import numpy as np
import re
# then we need to implement the search function to find the book name--> inputs as query (from the ADJUSTED_NAME)
V = TfidfVectorizer() 
# using the TfidfVectorizer instance, it takes a list of strings and turns it to tf-idf matrix
TFIDF_MATRIX = V.fit_transform(processed_data_list["ADJUSTED_NAME"])
# then we create the function that takes the string, and turns it to vector, and then finds the most match book title
# this will be done using consine similarity library to get the most match title.
#-------------------------------------------------------------------------------
# function that excute html command to formate the UR
def Title_Formate(val):
    return '<a target="_blank" href="{}">BOOKLINK</a>'.format(val, val)
# function to show the book image in the output books
def Show_Book(val):
    return '<a href="{}"><img src="{}" width=50></img></a>'.format(val, val)
# function that turns the input string into vector
def search(q,V):
    # we will make the same prcessing done with the ADJUSTED_NAME
    # converting to lower
    processed = re.sub("[^a-zA-Z0-9 ]", "", q.lower())
    # transform into a vector
    query_vec = V.transform([q])
    # calculate the cosine similarity between the input vector query, and our tfidf matrix
    sim = cosine_similarity(query_vec, TFIDF_MATRIX).flatten()
    # finding the top 10 partition values indices to get values
    target_indices = np.argpartition(sim, -10)[-10:]
    # getting the book titles from the indeces
    out_res = processed_data_list.iloc[target_indices]
    # sort the result according to it popularity -> refers as the number of ratings submitted on te book
    out_res = out_res.sort_values("RATINGS_NUM", ascending=False)
    
    # print the ranked matched books results 
    return out_res.head(5).style.format({'URL': Title_Formate, 'BOOK_IMAGE': Show_Book})

# In[8]:
# an example of searching for a book, and getting the output of the input query "India After Gandhi"
search("India After Gandhi",V)

