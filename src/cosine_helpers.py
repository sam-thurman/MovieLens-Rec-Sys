import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def _get_similarities(userId, user_frame):
    new_user = user_frame.loc[userId]
    new_frame = user_frame.copy().drop(userId)

    sims = []
    for row in new_frame.iterrows():
        current_userId = row[0]
        row = row[1]
        sims.append((current_userId, cosine_similarity([row], [new_user])[0][0]))

    sims.sort(key = lambda x: x[1], reverse = True)
    return sims, new_user



def _get_top_five_from_one_user(user_info, user_info_other, movies):
    m = (user_info_other * (user_info == 0).astype(int))
    top_five = m[m >= 4].sort_values(ascending = False)[0: 5]

    for movie_id in top_five.index:
        top_five.loc[movie_id] = movies.set_index('movieId').loc[movie_id]['title']
    return top_five

def get_top_five(user, user_frame, movies):
    similarity_array, user_info = _get_similarities(user, user_frame)
    top_five = []
    index = 0
    while len(top_five) < 5 and index < len(similarity_array):
        closest_user = int(similarity_array[index][0])
        new_movies = list(_get_top_five_from_one_user(user_info, user_frame.iloc[closest_user], movies))
        for movie in new_movies:
            if movie not in top_five:
                top_five.append(movie)
        index += 1

    return top_five[:5]