import os
import sys
module_path = os.path.abspath(os.path.join(os.pardir))
if module_path not in sys.path:
    sys.path.append(module_path)
    
# scripts
import src.rank_metrics as rank_metrics
import src.helpers as helpers
import src.table_encoder as table_encoder
import src.metrics as metrics

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)

ratings_df, movies_df, encoded_movies_df, tags_df, enoded_tags_df = helpers.load_format_data(
    '../data/csv')
model = helpers.load_model('../notebooks/als.model')


@app.route('/')
def home():
    return render_template('index_2.html')


# @app.route('/ratings', methods=['POST'])
# def ratings():
#     ratings = [int(x) for x in request.form.values()]
#     movieIds = [int(x) for x in request.form.keys()]
    
#     sample = helpers.convert_input_to_spark(movieIds, ratings, movies_df)
#     prediction = helpers.predict_for_new_user(model, sample, movies_df)
#     return render_template('index.html', recommendation_text=f'{prediction}')

# if __name__ == "__main__":
#     app.run(debug=True)

@app.route('/ratings', methods=['POST'])
def ratings():
    user_id = [int(x) for x in request.form.values()][0]
    user_liked = helpers.get_user_liked_movie_titles(user_id, ratings_df, movies_df)
    prediction = helpers.predict_for_one_user(model, user_id, ratings_df, movies_df)
    return render_template('index_2.html', recommendation_text=f'{prediction}', user_profile_text=f'{user_liked}')
    
if __name__ == "__main__":
    app.run(debug=True)