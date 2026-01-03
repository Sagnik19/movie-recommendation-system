# Movie Recommendation System (Hybrid)

## 1. Project Overview

This project is a **Hybrid Movie Recommendation System** built using **Python and Flask**.
It recommends movies by combining **popularity-based ranking** and **content-based similarity**.

The goal of this project is to demonstrate:

* Practical machine learning usage
* Recommendation system design
* Clean backend architecture with Flask

---

## 2. What the Application Does

### Home Page

* Displays **Top 50 popular movies**
* Shows **rating (rounded)** and **vote count**
* Movies are ranked using a popularity score

### Search

* User searches for a movie title
* The system finds similar movies
* Displays **hybrid recommendations** below the search result

### Movie Details Page

When a movie is clicked, it shows:

* Movie title
* Rating
* Vote count
* Director
* Cast
* Overview (description)

---

## 3. Recommendation Logic

### 3.1 Popularity-Based Recommendation

Movies are ranked using a weighted rating formula that considers:

* Average rating
* Number of votes

This avoids recommending movies with very few ratings.

---

### 3.2 Content-Based Filtering

* Uses **TF-IDF vectorization** on movie overviews
* Measures similarity using **cosine similarity**
* Similarity is computed **only for the searched movie** to save memory

---

### 3.3 Hybrid Recommendation

The final recommendation score is calculated as:

Hybrid Score =
0.6 × Content Similarity + 0.4 × Popularity Score

This balances:

* Relevance (similar content)
* Reliability (ratings and votes)

---

## 4. Technology Used

* Programming Language: **Python 3.11**
* Web Framework: **Flask**
* Libraries:

  * Pandas
  * NumPy
  * Scikit-learn
  * SciPy
* Frontend: HTML, Bootstrap
* IDE: PyCharm

---

## 5. Project Structure

```
movie-recommendation-system/
│
├── app.py
├── requirements.txt
│
├── model/
│   └── recommender.py
│
├── templates/
│   ├── index.html
│   └── movie.html
│
├── data/
│   └── (datasets not included)
│
└── .gitignore
```

---

## 6. Dataset Information

This project uses the **MovieLens and TMDB datasets**.

Due to GitHub file size limits, the datasets are **not included** in the repository.

### Dataset Source:

[https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset)

### Required files:

* movies_metadata.csv
* ratings.csv
* credits.csv

Place them inside the `data/` folder before running the project.

---

## 7. How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/Sagnik19/movie-recommendation-system.git
cd movie-recommendation-system
```

### Step 2: Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
python app.py
```

### Step 5: Open browser

```
http://127.0.0.1:5000
```

---

## 8. Why This Project Is Important

* Demonstrates **real-world recommendation systems**
* Shows **memory-efficient ML design**
* Uses **hybrid logic**, not simple similarity
* Suitable for **portfolio and interviews**

---

## 9. Future Improvements

* Pagination for movie lists
* User login and personalized recommendations
* REST API version
* Cloud deployment

---

## 10. Author

Sagnik Das
Aspiring Data Scientist / Machine Learning Engineer


