# 🎬 CineMatch — Movie Recommendation System

## Project Description (As per H N Verma's requirement)
A movie recommendation system using ML algorithms to suggest films based on user preferences.
Uses **Collaborative Filtering** + **Content-Based Filtering** with Flask web app.

## Features
- ✅ User Login / Logout
- ✅ Rate Movies (1-5 stars)
- ✅ Content-Based Filtering (similar movies by genre + overview)
- ✅ Collaborative Filtering (personalized based on ratings)
- ✅ Movie Search
- ✅ Clean Dark UI

## Tech Stack
- Python, Flask
- Pandas, Scikit-learn (TF-IDF, Cosine Similarity)
- HTML, CSS, JavaScript

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the App
```bash
python app.py
```

### 3. Open in Browser
```
http://localhost:5000
```

### Login Credentials
| Username | Password |
|----------|----------|
| demo     | demo123  |
| abhay    | abhay123 |

## How It Works

### Content-Based Filtering
- Combines movie `genres` + `overview` into tags
- Applies **TF-IDF Vectorizer** to convert text to numbers
- Calculates **Cosine Similarity** between movies
- Returns top 5 most similar movies

### Collaborative Filtering
- Builds a **User-Movie Rating Matrix**
- Finds users with similar taste using **Cosine Similarity**
- Recommends movies liked by similar users that you haven't seen yet

## Project Structure
```
movie-recommender/
├── app.py              # Flask routes
├── model.py            # ML logic (both algorithms)
├── requirements.txt    # Dependencies
├── templates/
│   ├── index.html      # Main page
│   └── login.html      # Login page
└── README.md
```

## To Use Real Dataset
1. Download TMDB 5000 Movie Dataset from Kaggle
2. Replace `movies_data` in `model.py` with:
```python
df_movies = pd.read_csv('tmdb_5000_movies.csv')
```
3. Adjust column names accordingly
