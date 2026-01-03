from flask import Flask, render_template, request
from model.recommender import (
    get_popular_movies,
    hybrid_recommend,
    get_movie_details
)

app = Flask(__name__)

# ================= HOME =================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        movie = request.form.get("movie")
        recommendations = hybrid_recommend(movie)

        return render_template(
            "index.html",
            popular=None,
            recommendations=recommendations.to_dict(orient="records"),
            searched_movie=movie
        )

    popular = get_popular_movies(50)

    return render_template(
        "index.html",
        popular=popular.to_dict(orient="records"),
        recommendations=None,
        searched_movie=None
    )

# ================= MOVIE DETAILS =================
@app.route("/movie/<title>")
def movie_page(title):
    movie = get_movie_details(title)
    return render_template("movie.html", movie=movie)

if __name__ == "__main__":
    app.run(debug=True)
