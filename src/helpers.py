import pandas as pd
import numpy as np
import os

from pyspark.sql import SparkSession
from pyspark import SparkContext
import pyspark.sql

from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import RegressionEvaluator

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

# spark helpers
def spark_to_pandas(sparkDataFrame):
    return sparkDataFrame.select("*").toPandas()

# helpers for getting movies and ratings from ALS model
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
    rec_rows = user_recommendations_df[user_recommendations_df['userId']==user_id]['recommendations']
    ratings = []
    for row in list(rec_rows)[0]:
        ratings.append(row['rating'])
    return ratings

def get_single_user_top_movies(user_recommendations_df, user_id, movies_df):
    rec_movie_rows = list(user_recommendations_df[user_recommendations_df['userId']==user_id]['recommendations'])[0]
    rec_movie_ids = []
    for movie in rec_movie_rows:
        rec_movie_ids.append(movie['movieId'])
    titles = []
    for movie_id in rec_movie_ids:
        titles.append(movies_df[movies_df['movieId']==movie_id]['title'].values[0])
    return titles

def get_top_movies_and_ratings(user_recommendations_df, user_id, movies_df):
    movies = get_single_user_top_movies(user_recommendations_df, user_id, movies_df)
    ratings = get_single_user_top_ratings(user_recommendations_df, user_id, movies_df)
    return dict(zip(movies, ratings))

# ALS model helpers
def fit_new_model(training_data):
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", 
              itemCol="movieId", ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(training_data)
    return model

def evaluate_model():
    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("RMSE = " + str(rmse))
    
def user_liked_compared_recommended(ratings_df, movies_df, user_recommendation_df, user_id):
    top_movies_and_rankings = get_top_movies_and_ratings(user_recommendation_df, 
                                                                 user_id, movies_df)
    top_movie_recs = list(top_movies_and_rankings.keys())
    user_rates = ratings_df[ratings_df['userId']==user_id]
    good_user_rates = user_rates[user_rates['rating']>3]
    user_movies = []
    for movieId in list(good_user_rates['movieId']):
        user_movies.append(movies_df[movies_df['movieId']==movieId]['title'].values[0])
    print('users liked movies:\n', user_movies)
    print('users recommended movies:\n', top_movie_recs)