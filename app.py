from flask import Flask, render_template, request, session, jsonify, redirect, url_for
from model import get_content_recommendations, get_collaborative_recommendations, get_all_movies, search_movies, get_latest_movies, get_bollywood_movies

app = Flask(__name__)
app.secret_key = 'movierecsecret2024'

users = {'demo': 'demo123', 'abhay': 'abhay123'}

@app.route('/')
def index():
    if 'user' not in session:
        return redirect(url_for('login'))
    movies = get_all_movies()
    latest = get_latest_movies(12)
    bollywood = get_bollywood_movies(12)
    user_ratings = session.get('ratings', {})
    collab_recs = get_collaborative_recommendations({int(k): v for k, v in user_ratings.items()})
    return render_template('index.html', movies=movies, latest=latest, bollywood=bollywood,
                           collab_recs=collab_recs, user=session['user'], user_ratings=user_ratings)

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in users and users[username] == password:
            session['user'] = username
            session['ratings'] = {}
            return redirect(url_for('index'))
        error = 'Invalid username or password'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    recs = get_content_recommendations(data.get('title', ''))
    return jsonify({'recommendations': recs})

@app.route('/rate', methods=['POST'])
def rate():
    if 'user' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    data = request.get_json()
    if 'ratings' not in session:
        session['ratings'] = {}
    session['ratings'][str(data.get('movie_id'))] = int(data.get('rating'))
    session.modified = True
    return jsonify({'success': True})

@app.route('/search')
def search():
    results = search_movies(request.args.get('q', ''))
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
