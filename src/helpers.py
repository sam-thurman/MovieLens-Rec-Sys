import pandas as pd
import numpy as np
import pickle
import os

from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.sql

from pyspark.ml.recommendation import ALS
from pyspark.ml.recommendation import ALSModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
sc = SparkContext()
spark = SparkSession(sc)

# VARIOUS HELPERS


def get_ids_from_titles(title_list, movies_df):
    movie_ids = []
    for title in title_list:
        movie_ids.append(
            int(movies_df[movies_df['title'] == title]['movieId']))
    return movie_ids

# DATA HELPERS


def load_format_data(data_path):
    '''
    Reads all necessary csv files to pandas dataframes
    Data_path is the path to a folder containing csv files 
    loaded below. If running from our jupyter notebook, this 
    will be "../data/csv"
    '''
    ratings_df = pd.DataFrame(pd.read_csv(
        os.path.join(data_path, 'ratings.csv')))
    movies_df = pd.DataFrame(pd.read_csv(
        os.path.join(data_path, 'movies.csv')))
    encoded_movies_df = pd.DataFrame(pd.read_csv(os.path.join(
        data_path, 'encoded_movies.csv'))).drop('Unnamed: 0', axis=1)
    tags_df = pd.DataFrame(pd.read_csv(os.path.join(data_path, 'tags.csv')))
    enoded_tags_df = pd.DataFrame(pd.read_csv(
        os.path.join(data_path, 'encoded_tags.csv')))

    return ratings_df, movies_df, encoded_movies_df, tags_df, enoded_tags_df


def format_ratings(ratings_df):
    if 'timestamp' in ratings_df.columns:
        ratings_df.drop('timestamp', inplace=True, axis=1)
    
    ratings_df = spark.createDataFrame(ratings_df)
    return ratings_df


def load_format_data_for_model(data_path):
    '''
    Call load_format_data, drop irrelevant columns and return train/test
    data as spark dataframes
    '''
    ratings_df, movies_df, encoded_movies_df, tags_df, enoded_tags_df = load_format_data(
        data_path)
    ratings_df = format_ratings(ratings_df)
    (train, test) = ratings_df.randomSplit([0.8, 0.2])
    return train, test


# MODEL HELPERS
def fit_als(training_data):
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="rating",
              coldStartStrategy="drop")
    model = als.fit(training_data)
    return model


def predict_for_one_user(model, user_id, test_data, movie_df):
    sample = test_data[test_data['userId'] == user_id]
    sample = format_ratings(sample)
    predictions = model.transform(sample)
    prediction_ids = spark_to_pandas(predictions).sort_values(
        'prediction', ascending=False)[:5]['movieId']
    prediction_titles = []
    for id in prediction_ids:
        title = movie_df[movie_df['movieId'] == id]['title']
        prediction_titles.append(title.values[0])
    return prediction_titles


def predict_for_all_users(model, test_data):
    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("RMSE = " + str(rmse))

    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(5)
    # Generate top 10 user recommendations for each movie
    movieRecs = model.recommendForAllItems(5)
    return userRecs, movieRecs


def evaluate_model(model, test):
    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("RMSE = " + str(rmse))


def save_model(model, file_path):
    model.save(file_path)
    print('model saved to file path')


def load_model(file_path):
    model = ALSModel().load(file_path)
    return model

# SPARK HELPERS


def spark_to_pandas(sparkDataFrame):
    return sparkDataFrame.select("*").toPandas()

# USER HELPERS


def get_single_user_top_ratings(user_recommendations_df, user_id, movies_df):
    '''
    input:
    - an output of an ALS recommendForAllUsers() method (list of users and top
    predictions for those users)
    - a single user ID number
    output: 
    - top number of recommendations for given userId (number of recs specified in 
    recommendForAllUsers(n) method)
    '''
    rec_rows = user_recommendations_df[user_recommendations_df['userId']
                                       == user_id]['recommendations']
    ratings = []
    for row in list(rec_rows)[0]:
        ratings.append(row['rating'])
    return ratings


def get_single_user_top_movies(user_recommendations_df, user_id, movies_df):
    rec_movie_rows = list(
        user_recommendations_df[user_recommendations_df['userId'] == user_id]['recommendations'])[0]
    rec_movie_ids = []
    for movie in rec_movie_rows:
        rec_movie_ids.append(movie['movieId'])
    titles = []
    for movie_id in rec_movie_ids:
        titles.append(movies_df[movies_df['movieId']
                                == movie_id]['title'].values[0])
    return titles


def get_top_movies_and_ratings(user_recommendations_df, user_id, movies_df):
    movies = get_single_user_top_movies(
        user_recommendations_df, user_id, movies_df)
    ratings = get_single_user_top_ratings(
        user_recommendations_df, user_id, movies_df)
    return dict(zip(movies, ratings))


def get_user_liked_movie_ids(user_id, ratings_df):
    user_rates = ratings_df[ratings_df['userId'] == user_id]
    good_user_rates = user_rates[user_rates['rating'] > 3]
    liked_movies = list(good_user_rates['movieId'])
    return liked_movies


def get_user_liked_movie_titles(user_id, ratings_df, movies_df):
    ids = get_user_liked_movie_ids(user_id, ratings_df)
    movie_titles = []
    for id in ids:
        movie_title = movies_df[movies_df['movieId'] == id]['title']
        movie_titles.append(movie_title.values[0])
    return movie_titles


def user_liked_compared_recommended(ratings_df, movies_df, user_recommendation_df, user_id):
    top_movies_and_rankings = get_top_movies_and_ratings(user_recommendation_df,
                                                         user_id, movies_df)
    top_movie_recs = list(top_movies_and_rankings.keys())
    user_rates = ratings_df[ratings_df['userId'] == user_id]
    good_user_rates = user_rates[user_rates['rating'] > 3]
    user_movies = []
    for movieId in list(good_user_rates['movieId']):
        user_movies.append(
            movies_df[movies_df['movieId'] == movieId]['title'].values[0])
    print('users liked movies:\n', user_movies)
    print('users recommended movies:\n', top_movie_recs)


# NEW USER HELPERS
def convert_input_to_spark(movieIds, ratings, movies_df):
    '''
    format for input is a dictionary where the keys are full titles of movies as they appear\
    in the MovieLens dataset, values are explicit ratings of movies (5 for 'liked', 0.5 for 'did
    not like', and 0s indicating a lack of feedback for that movie)
    '''
    us = [500]*len(movieIds)

    movieIds = pd.Series(movieIds)
    ratings = pd.Series(ratings)
    us = pd.Series(us)

    df = pd.DataFrame()
    df['userId'] = us
    df['movieId'] = movieIds
    df['rating'] = ratings
    sample = format_ratings(df)
    
    return sample


def predict_for_new_user(model, sample, movie_df):

    predictions = model.transform(sample)
    prediction_ids = predictions.select("*").toPandas().sort_values('prediction',
                                                              ascending=False).head()['movieId']
    prediction_titles = []
    for id in prediction_ids:
        title = movie_df[movie_df['movieId'] == id]['title']
        prediction_titles.append(title.values[0])

    return prediction_titles
