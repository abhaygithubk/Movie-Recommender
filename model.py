import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests

OMDB_API_KEY = "af82ea72"
OMDB_BASE = "http://www.omdbapi.com/"

def get_poster(title, year=None):
    try:
        params = {"t": title, "apikey": OMDB_API_KEY}
        if year:
            params["y"] = year
        res = requests.get(OMDB_BASE, params=params, timeout=5)
        data = res.json()
        poster = data.get("Poster", "")
        if poster and poster != "N/A":
            return poster
    except:
        pass
    return ""

movies_data = {
    'movie_id': list(range(1, 51)),
    'title': [
        # Hollywood Classics
        'The Dark Knight', 'Inception', 'Interstellar', 'The Matrix',
        'The Shawshank Redemption', 'The Godfather', 'Pulp Fiction', 'Fight Club',
        'Forrest Gump', 'Avengers: Endgame',
        # Hollywood Latest 2022-2024
        'Oppenheimer', 'Barbie', 'Dune: Part Two', 'Top Gun: Maverick',
        'Everything Everywhere All at Once', 'The Batman', 'Avatar: The Way of Water',
        'Saltburn', 'Poor Things', 'Killers of the Flower Moon',
        # Bollywood Classics
        '3 Idiots', 'Dangal', 'PK', 'Lagaan',
        'Dil Chahta Hai', 'Taare Zameen Par', 'Sholay', 'Kabhi Khushi Kabhie Gham',
        'Dilwale Dulhania Le Jayenge', 'Kal Ho Naa Ho',
        # Bollywood Latest 2022-2024
        'Pathaan', 'Jawan', 'Animal', 'Dunki',
        'Tiger 3', 'Sam Bahadur', 'Rocky Aur Rani Kii Prem Kahaani',
        'Crew', 'Fighter', 'Article 370',
        'Stree 2', 'Singham Returns', 'Kalki 2898 AD', 'Pushpa: The Rule',
        'Munjya', 'Do Aur Do Pyaar', 'Srikanth', 'Bade Miyan Chote Miyan',
        'Yodha', 'Chandu Champion'
    ],
    'genres': [
        'Action Crime Drama', 'Action Sci-Fi Thriller', 'Adventure Drama Sci-Fi', 'Action Sci-Fi',
        'Drama', 'Crime Drama', 'Crime Drama Thriller', 'Drama Thriller',
        'Drama Romance', 'Action Adventure Sci-Fi',
        'Biography Drama History', 'Comedy Fantasy', 'Adventure Drama Sci-Fi', 'Action Drama',
        'Action Adventure Comedy', 'Action Crime Drama', 'Action Adventure Sci-Fi',
        'Drama Thriller', 'Comedy Drama Fantasy', 'Crime Drama History',
        'Comedy Drama', 'Biography Drama Sport', 'Comedy Drama Sci-Fi', 'Drama History Sport',
        'Comedy Drama', 'Drama Family', 'Action Adventure Drama', 'Drama Family Romance',
        'Drama Romance', 'Drama Romance',
        'Action Thriller', 'Action Thriller', 'Action Crime Drama', 'Comedy Drama',
        'Action Thriller', 'Biography Drama War', 'Comedy Drama Romance',
        'Comedy Crime Thriller', 'Action Drama', 'Drama Thriller',
        'Comedy Horror', 'Action Comedy Drama', 'Action Sci-Fi Drama', 'Action Crime Drama',
        'Comedy Horror', 'Comedy Drama Romance', 'Biography Drama', 'Action Comedy',
        'Action Thriller', 'Biography Drama Sport'
    ],
    'overview': [
        'Batman faces Joker a criminal mastermind who plunges Gotham into chaos',
        'A thief who steals corporate secrets through dream-sharing technology',
        'A team of explorers travel through a wormhole in space to save humanity',
        'A hacker discovers reality is a simulation and joins rebels against machines',
        'Two imprisoned men bond over years finding redemption through decency',
        'The aging patriarch of a crime dynasty transfers control to his son',
        'The lives of two criminals a boxer and others intertwine in crime and violence',
        'An insomniac office worker forms an underground fight club with a soap maker',
        'The story of a man with low IQ who achieves great things through kindness',
        'Avengers assemble to undo the actions of Thanos and restore order to the universe',
        'The story of American scientist J Robert Oppenheimer and the atomic bomb',
        'Barbie and Ken go on a journey of self-discovery in the real world',
        'Paul Atreides unites with Chani and the Fremen to seek revenge on conspirators',
        'Maverick trains a group of Top Gun graduates for a specialized dangerous mission',
        'A middle-aged Chinese immigrant is swept up in an adventure across the multiverse',
        'Batman uncovers corruption in Gotham City that connects to his own family',
        'Jake Sully lives with his newfound family formed on the planet Pandora',
        'A young man manipulates his way into a wealthy English family over summer',
        'A woman sold into marriage uses newfound freedom to challenge Victorian society',
        'FBI agents investigate the murder of members of the Osage Nation for oil wealth',
        'Three engineering students chase their dreams and find the true meaning of education',
        'A father trains his daughters to become world champion wrestlers in India',
        'An alien on Earth helps a simple man question religious beliefs and superstitions',
        'A village in colonial India challenges British rule through a cricket match',
        'Three friends from college navigate their relationships and career aspirations',
        'A child is diagnosed with dyslexia and finds help from an unconventional teacher',
        'The story of two bandits who terrorize a village and the police chasing them',
        'A family comes together during Diwali amid personal secrets and love stories',
        'A young man follows his love to London against his fathers wishes',
        'A man learns to live life to the fullest before losing someone he loves',
        'A RAW agent goes on a mission to combat a sinister organization threatening India',
        'A man takes on a criminal empire to protect the common people of India',
        'A man becomes dangerously obsessed with protecting his family from enemies',
        'A man tries to secretly bring back illegal immigrants from UK to India',
        'Tiger is back and teams up with his wife Zoya for a new dangerous mission',
        'The life and times of Field Marshal Sam Manekshaw and the 1971 Bangladesh war',
        'A love story between two people from very different families and backgrounds',
        'Three air hostesses plan a heist together to secure their futures',
        'Indian Air Force pilots go on a dangerous mission to protect national interests',
        'A politician works tirelessly to revoke Article 370 and integrate Kashmir with India',
        'A small town man teams up with a ghost to fight evil forces once again',
        'Singham returns to fight corruption and a powerful criminal enemy',
        'In the distant future a man is prophesied to be the chosen one against evil god',
        'Pushpa Raj expands his smuggling empire and fights against the police system',
        'A young boy gets possessed by the spirit of a mischievous creature called Munjya',
        'A married woman and a divorced man discover unexpected feelings for each other',
        'The inspiring story of Srikanth Bolla a visually impaired entrepreneur',
        'Two soldiers of different sizes go on a comical mission together',
        'An air force officer on a mission to rescue hostages from a hijacked plane',
        'The incredible true story of Murlikant Petkar Indias first Paralympic gold medalist'
    ],
    'rating': [
        9.0, 8.8, 8.6, 8.7, 9.3, 9.2, 8.9, 8.8, 8.8, 8.4,
        8.9, 6.9, 8.5, 8.3, 7.8, 7.9, 7.6, 7.2, 7.9, 7.7,
        8.4, 8.4, 8.1, 8.1, 8.1, 8.5, 8.2, 7.4, 8.1, 7.8,
        5.9, 6.3, 6.3, 6.2, 5.6, 7.5, 7.1, 6.5, 5.7, 7.0,
        8.5, 5.5, 6.5, 7.6, 6.8, 6.5, 7.8, 4.5, 6.2, 8.1
    ],
    'year': [
        2008, 2010, 2014, 1999, 1994, 1972, 1994, 1999, 1994, 2019,
        2023, 2023, 2024, 2022, 2022, 2022, 2022, 2023, 2023, 2023,
        2009, 2016, 2014, 2001, 2001, 2007, 1975, 2001, 1995, 2003,
        2023, 2023, 2023, 2023, 2023, 2023, 2023, 2024, 2024, 2024,
        2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024, 2024
    ]
}

df_movies = pd.DataFrame(movies_data)

print("Fetching posters from OMDB API...")
posters = []
for _, row in df_movies.iterrows():
    p = get_poster(row['title'], row['year'])
    posters.append(p)
    print(f"  {'OK' if p else 'NA'} - {row['title']}")
df_movies['poster'] = posters
print("Done!")

df_movies['tags'] = df_movies['genres'] + ' ' + df_movies['overview']
tfidf = TfidfVectorizer(stop_words='english', max_features=500)
tfidf_matrix = tfidf.fit_transform(df_movies['tags'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def get_content_recommendations(movie_title, n=5):
    try:
        idx = df_movies[df_movies['title'] == movie_title].index[0]
        scores = list(enumerate(cosine_sim[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
        return df_movies.iloc[[i[0] for i in scores]][['movie_id','title','genres','rating','year','poster']].to_dict('records')
    except:
        return []

ratings_data = {
    'user_id': [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5],
    'movie_id': [1,2,3,4, 1,5,6,7, 8,9,10,11, 12,13,14,15, 3,4,16,17],
    'rating':   [5,5,4,5, 4,5,4,5,  5,4,5,4,   5,5,4,3,    5,4,4,5]
}
df_ratings = pd.DataFrame(ratings_data)

def get_collaborative_recommendations(user_ratings_dict, n=5):
    if not user_ratings_dict:
        return df_movies.nlargest(n, 'rating')[['movie_id','title','genres','rating','year','poster']].to_dict('records')
    all_ratings = df_ratings.copy()
    new_user_id = 999
    for mid, rat in user_ratings_dict.items():
        all_ratings = pd.concat([all_ratings, pd.DataFrame({'user_id':[new_user_id], 'movie_id':[mid], 'rating':[rat]})], ignore_index=True)
    matrix = all_ratings.pivot_table(index='user_id', columns='movie_id', values='rating').fillna(0)
    if new_user_id not in matrix.index:
        return df_movies.nlargest(n, 'rating')[['movie_id','title','genres','rating','year','poster']].to_dict('records')
    sim = cosine_similarity(matrix)
    user_idx = list(matrix.index).index(new_user_id)
    sim_scores = list(enumerate(sim[user_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:4]
    rated_ids = set(user_ratings_dict.keys())
    rec_movies = []
    for idx, score in sim_scores:
        similar_user_id = matrix.index[idx]
        user_top = all_ratings[all_ratings['user_id'] == similar_user_id].nlargest(5, 'rating')
        for _, row in user_top.iterrows():
            if row['movie_id'] not in rated_ids:
                movie = df_movies[df_movies['movie_id'] == row['movie_id']]
                if not movie.empty and row['movie_id'] not in [m['movie_id'] for m in rec_movies]:
                    rec_movies.append(movie[['movie_id','title','genres','rating','year','poster']].to_dict('records')[0])
    return rec_movies[:n] if rec_movies else df_movies.nlargest(n, 'rating')[['movie_id','title','genres','rating','year','poster']].to_dict('records')

def get_all_movies():
    return df_movies[['movie_id','title','genres','rating','year','poster']].to_dict('records')

def search_movies(query):
    mask = df_movies['title'].str.contains(query, case=False, na=False)
    return df_movies[mask][['movie_id','title','genres','rating','year','poster']].to_dict('records')

def get_latest_movies(n=12):
    return df_movies[df_movies['year'] >= 2022].nlargest(n, 'year')[['movie_id','title','genres','rating','year','poster']].to_dict('records')

def get_bollywood_movies(n=12):
    bollywood = [
        'Pathaan','Jawan','Animal','Dunki','Tiger 3','Sam Bahadur',
        'Rocky Aur Rani Kii Prem Kahaani','Crew','Fighter','Article 370',
        'Stree 2','Singham Returns','Kalki 2898 AD','Pushpa: The Rule',
        'Munjya','Do Aur Do Pyaar','Srikanth','Bade Miyan Chote Miyan',
        'Yodha','Chandu Champion','3 Idiots','Dangal','PK','Lagaan',
        'Dil Chahta Hai','Taare Zameen Par','Sholay','Kabhi Khushi Kabhie Gham',
        'Dilwale Dulhania Le Jayenge','Kal Ho Naa Ho'
    ]
    mask = df_movies['title'].isin(bollywood)
    return df_movies[mask][['movie_id','title','genres','rating','year','poster']].to_dict('records')
