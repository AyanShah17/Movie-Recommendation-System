import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# LOAD THE DATA

movies1 = pd.read_csv('dataset.csv')

# CLEANING THE DATASET BY TARGETING THE USABLE DATA

movies1 = movies1[['id', 'title', 'overview', 'genre']]

# COMBINING THE NECESSARY COLUMNS

movies1['tags'] = movies1['overview']+movies1['genre']

# DROPPING THE UNNECESSARY COLUMNS

movies2 = movies1.drop(columns = ['overview', 'genre'])

# CREATING THE COUNTVECTORIZER  OBJECT AND FITTING IT TO THE movies1

# max_features = 10000 BECAUSE WE HAVE 10000 ROWS OF movies1

# stop_words = 'english' BECAUSE WE ARE DEALING WITH TEXT IN ENGLISH

cv = CountVectorizer(max_features=10000, stop_words='english')

# CREATING A VECTOR TO TRANSFORM AND FIT THE movies1

# HERE THE 'U' IS FOR 'UTF' FORMAT

vector = cv.fit_transform(movies2['tags'].values.astype('U')).toarray()

# NOW LET'S CREATE A SIMILARITY VARIALBLE WITH VECTOR IN CONSIDERATION

sim = cosine_similarity(vector)

# NOW LET'S CREATE A FUNCTION TO RECOMMEND THE MOVIES

def recommend(movies1):
    
    index = movies2[movies2['title']==movies1].index[0]
    
# NOW WE CALCULATE THE DISTANCE BETWEEN THE VECTORS
    
    distance = sorted(list(enumerate(sim[index])), reverse=True, key=lambda vector:vector[1])
    
# NOW LET US CREATE THE LOOP TO GET THE FIRST
# 5 TOP RECOMMENDATIONS

    for i in distance [0:10]:
        print(movies2.iloc[i[0]].title)

user_input = input("Enter the Movie Title : ")
recommend(user_input)

pickle.dump(movies2, open('movies_list.pkl','wb'))

pickle.dump(sim, open('sim.pkl', 'wb'))

pickle.load(open('movies_list.pkl', 'rb'))

