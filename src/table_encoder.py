import pandas as pd

def tag_encoder(tags):
    user_temp = []
    movie_temp = []
    tags_temp = []
    unique_tags = set()
    for user, i in tags.groupby('userId'):
        for movie, j in i.groupby('movieId'):
            user_temp.append(user)
            movie_temp.append(movie)
            tags_temp.append(list(j['tag']))
            unique_tags.update(list(j['tag']))

    tag_frame = pd.DataFrame(columns = unique_tags)
    encoded_tags = pd.DataFrame(zip(user_temp, movie_temp), columns = ['userId', 'movieId'])
    encoded_tags = pd.merge(encoded_tags, tag_frame, how = 'left', left_index = True, right_index = True).fillna(0)

    for index in range(len(tags_temp)):
        for tag in tags_temp[index]:
            encoded_tags.at[index, tag] = 1
            
    return encoded_tags


def genre_encoder(movies):
    genres = []
    unique_genres = set()
    for _, row in movies.iterrows():
        current_genres = row['genres'].split('|')
        genres.append(current_genres)
        unique_genres.update(current_genres)
        
    genre_frame = pd.DataFrame(columns = unique_genres)
    encoded_genres = pd.merge(movies[['movieId', 'title']], genre_frame, how = 'left', left_index = True, right_index = True).fillna(0)

    for index in range(len(genres)):
        for genre in genres[index]:
            encoded_genres.at[index, genre] = 1
            
    return encoded_genres

def rating_vectorizer(ratings):
    users_set = ratings['userId'].unique()
    users_set.sort()

    movie_set = ratings['movieId'].unique()
    movie_set.sort()

    user_frame = pd.DataFrame(index = users_set, columns = movie_set)
    user_frame = user_frame.fillna(0)

    counter = 0
    quarter = int(ratings.shape[0] / 4)
    for row in ratings.iterrows():
        counter += 1
        if counter % (quarter) == 0:
            print(counter // quarter * 25, '% of the way done')
        row = row[1]
        _userId = row['userId']
        _movieId = row['movieId']
        user_frame.loc[_userId][_movieId] = row['rating']
    return user_frame