from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movie_app = Flask(__name__)

movies = pd.read_pickle("movies.pkl")
similarity_data = pd.read_pickle("similarity.pkl")  


cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(similarity_data["tags"])
similarity = cosine_similarity(vectors)


@movie_app.route("/", methods=["GET"])
def form_show():
    return render_template("movie.html", movies=movies["title"].tolist(), rec=[])


@movie_app.route("/movie_show", methods=["POST"])  
def show():
    movie = request.form.get("movies")
    try:
        movie_index = movies[movies["title"] == movie].index[0]
        sim_scores = similarity[movie_index]
        movies_list = sorted(list(enumerate(sim_scores)), key=lambda x: x[1], reverse=True)[1:6]
        rec = [movies.iloc[i[0]]["title"] for i in movies_list]

        return render_template("movie.html", movies=movies["title"].tolist(), rec=rec)
    except Exception as e:
        return f"Error occurred: {e}"

if __name__ == "__main__":
    movie_app.run(debug=True)


    



