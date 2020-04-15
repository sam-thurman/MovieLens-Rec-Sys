from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def compare_movie(movie_1, movie_2, encoded_genres = None):
    '''
    TODO: COMMENT MORE!!!!
    
    movies must either be their id number or their genre vector
    '''
    if isinstance(movie_1, int):
        movie_1 = encoded_genres[encoded_genres['movieId'] == movie_1].drop(['movieId', 'title'], axis = 1)
    if isinstance(movie_2, int):    
        movie_2 = encoded_genres[encoded_genres['movieId'] == movie_2].drop(['movieId', 'title'], axis = 1)
    return round(cosine_similarity(movie_1, movie_2)[0][0], 5)

def compare_all_movies(movieId, encoded_genres):
    '''
    TODO: COMMENT MORE!!!!
    
    encoded_genres: Encoded genres based on the table encoder. Has its encoded genres and movie ids and titles
    '''
    movie = encoded_genres[encoded_genres['movieId'] == movieId]
    index = movie.index
    movie_vector = movie.drop(['movieId', 'title'], axis = 1)
    lst = []
    for _, movie in encoded_genres.drop(index).iterrows():
        lst.append((encoded_genres.loc[_]['movieId'], compare_movie(movie_vector, [movie.drop(['movieId', 'title'])])))
    lst.sort(reverse = True, key = lambda x: x[1])
    return pd.DataFrame(lst, columns = ['movieId', 'similarity'])