import pandas as pd
import urllib.request
import re
from flask import Flask, Response, render_template, url_for, redirect, request, flash, session, escape, jsonify
import cv2
import numpy as np
from keras.models import load_model
from http.server import SimpleHTTPRequestHandler, HTTPServer
from flask_sqlalchemy import SQLAlchemy
from argon2 import PasswordHasher
import os
from sqlalchemy import update
from tqdm import tqdm
import random

# disable chained assignments for recommender functions
pd.options.mode.chained_assignment = None

MODEL_PATH = 'resources/model'
emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: "Neutral"}

app = Flask(__name__)
video = cv2.VideoCapture(0)
model = load_model(MODEL_PATH)
app.config['DATABASE_FILE'] = './resources/licenta_db.db'
app.config['SECRET_KEY'] = 'superSecret'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + app.config['DATABASE_FILE']
db = SQLAlchemy(app)


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,
                         directory=r'C:\Users\User\Videos',
                         **kwargs)


class User(db.Model):
    __tablename__ = 'users'
    user_id = db.Column(db.Integer, primary_key=True, unique=True, nullable=False)
    user_name = db.Column(db.String(100), unique=True, nullable=False)
    user_pass = db.Column(db.String(), nullable=False)
    pref_song = db.Column(db.String(), nullable=False)
    pref_artist = db.Column(db.String(), nullable=False)
    pref_genre = db.Column(db.String(), nullable=False)
    min_year = db.Column(db.Integer, nullable=False)
    first_login = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return f"Name : {self.user_name}, password:{self.user_pass}"

    def get_data(self):
        return [self.user_id, self.user_name, self.user_pass, self.pref_song, self.pref_artist, self.pref_genre,
                self.min_year, self.first_login]


class Appreciation(db.Model):
    __tablename__ = 'appreciation'
    id = db.Column(db.Integer, primary_key=True, unique=True, nullable=False)
    user_id = db.Column(db.Integer, nullable=False)
    song_id = db.Column(db.Integer, nullable=False)
    liked = db.Column(db.Integer,
                      nullable=False)  # 0 if dislike, 1 if like, delete if it was liked then disliked or vice-versa

    def __repr__(self):
        return f'app_id : {self.id}, user_id:{self.user_id}, song_id:{self.song_id}, liked:{self.liked}'

    def get_data(self):
        return [self.id, self.user_id, self.song_id, self.liked]


def run(server_class=HTTPServer, handler_class=Handler):
    server_address = ('localhost', 8000)
    httpd = server_class(server_address, handler_class)
    httpd.serve_forever()


@app.route('/', methods=['GET', 'POST'])
def index():
    if 'username' in session:
        if session['username']:
            return render_template('emotion_page//emotion.html')
    if request.method == 'POST':
        ph = PasswordHasher()
        username = escape(request.form.get("username"))
        password = escape(request.form.get("password"))
        if username is not None and password is not None:
            check_user = User.query.filter_by(user_name=username).first()
            if check_user is None:
                flash('User doesn\'t exist')
                return redirect('/register')
            else:
                try:
                    check_pass = ph.verify(check_user.user_pass, password)
                    check_pass += 1
                except:
                    flash('Incorrect password')
                    return redirect('/')
                else:
                    session['username'] = username
                    first_login = User.query.filter_by(user_name=username).first()
                    user_data = first_login.get_data()
                    session['user_id'] = user_data[0]
                    if user_data[-1] == 1:
                        return redirect('/preferences')
                    else:
                        return redirect('/emotion_recognition')
    return render_template('homepage_with_login//login.html')


@app.route('/preferences_update', methods=['GET', 'POST'])
def preferences_update():
    if request.method == 'POST':
        fav_song = escape(request.form.get('song_name'))
        fav_artist = escape(request.form.get('artist_name'))
        fav_genre = escape(request.form.get('genre_name'))
        min_year = escape(request.form.get('min_year'))
        if 'username' in session:
            local_username = session['username']
            logged_user = User.query.filter_by(user_name=local_username).first()
            logged_user.pref_song = fav_song
            logged_user.pref_artist = fav_artist
            logged_user.pref_genre = fav_genre
            logged_user.min_year = min_year
            logged_user.first_login = 0
            db.session.commit()
    return redirect(url_for('emotion_recognition'))


@app.route('/preferences', methods=['GET', 'POST'])
def preferences():
    df = pd.read_csv('./resources/finalUsable.csv')
    song_names = df['title'].values.tolist()
    song_names = list(set(song_names))
    song_names.sort()
    artist_names = df['artist'].values.tolist()
    artist_names = list(set(artist_names))
    artist_names.sort()
    genre_name = df['genre'].values.tolist()
    genre_name = list(set(genre_name))
    genre_name.sort()
    return render_template('preferences//preferences.html', song_names=song_names, artist_names=artist_names,
                           genre_names=genre_name)


@app.route('/logout')
def logout():
    local_user = session['username']
    try:
        os.remove(f'./user_emotions/{local_user}.txt')
    except:
        pass
    session.pop('username', None)
    session.pop('emotion', None)
    session.pop('liked', None)
    session.pop('disliked', None)
    session.pop('happy_count', None)
    flash('You were logged out.')
    return redirect(url_for('index'))


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        ph = PasswordHasher()
        username = escape(request.form.get("username"))
        password = escape(request.form.get("password"))

        # create an object of the Profile class of models
        # and store data as a row in our datatable
        if username is not None and password is not None:
            check_user = User.query.filter_by(user_name=username).first()
            if check_user:
                flash('Username already existing!')
                return redirect('/register')
            else:
                my_ph = ph.hash(password)
                p = User(user_name=username, user_pass=my_ph, pref_song="", pref_artist="", pref_genre="", min_year=0,
                         first_login=1)
                db.session.add(p)
                db.session.commit()
                return redirect('/')
    else:
        return render_template('homepage_with_register//register.html')


def gen(video_capturer, local_username):
    filename = f'./user_emotions/{local_username}.txt'
    while True:
        success, frame = video_capturer.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier('./resources/haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            prediction = model.predict(cropped_img)
            cv2.putText(frame, emotion_dict[int(np.argmax(prediction))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 0),
                        1, cv2.LINE_AA)
            with open(filename, 'w') as file:
                file.write(emotion_dict[int(np.argmax(prediction))])
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/emotion_recognition', methods=['GET', 'POST'])
def emotion_recognition():
    return render_template('emotion_page//emotion.html')


@app.route('/emotion')
def emotion():
    global video
    if 'username' in session:
        if session['username']:
            local_username = session['username']
            session['emotion'] = ''
            return Response(gen(video, local_username),
                            mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/about')
def about():
    return render_template('about_page//about.html')


def recommend_based_on_song_name(gen_df, song_name, func_emotion, amount=1):
    distance = []
    song = gen_df[gen_df.title.str.lower() == song_name.lower()].head(1).values[0]
    # emotion = gen_df[gen_df.title.str.lower() == song_name.lower()].head(1).values[0][-2]
    rec = gen_df[gen_df.title.str.lower() != song_name.lower()]
    for func_songs in tqdm(rec.values):
        d = 0
        for col in np.arange(len(rec.columns)):
            if not col in [0, 1, 2, 3, 4, 6, 11, 19]:
                d = d + np.absolute(float(song[col]) - float(func_songs[col]))
        distance.append(d)
    rec['distance'] = distance
    rec = rec.sort_values('distance')
    rec = rec.loc[rec['emotion'] == func_emotion]
    columns = ['index', 'title', 'artist', 'popularity']
    return rec[columns][:amount]


def recommend_based_on_genre(gen_df, genre, func_emotion, amount=1):
    distance = []
    genre_df = gen_df[gen_df.genre.str.lower() == genre.lower()].head(1).values[0]
    rec = gen_df[gen_df.genre.str.lower() != genre.lower()]
    for func_songs in tqdm(rec.values):
        d = 0
        for col in np.arange(len(rec.columns)):
            if not col in [0, 1, 2, 3, 4, 6, 11, 19]:
                d = d + np.absolute(float(genre_df[col]) - float(func_songs[col]))
        distance.append(d)
    rec['distance'] = distance
    rec = rec.sort_values('distance')
    rec = rec.loc[rec['emotion'] == func_emotion]
    columns = ['index', 'title', 'artist', 'popularity']
    return rec[columns][:amount]


def recommend_based_on_genre_year(gen_df, genre, year_min, func_emotion, amount=1):
    distance = []
    genre_df = gen_df[gen_df.genre.str.lower() == genre.lower()].head(1).values[0]
    rec = gen_df[gen_df.genre.str.lower() != genre.lower()]
    for func_songs in tqdm(rec.values):
        d = 0
        for col in np.arange(len(rec.columns)):
            if not col in [0, 1, 2, 3, 4, 6, 11, 19]:
                d = d + np.absolute(float(genre_df[col]) - float(func_songs[col]))
        distance.append(d)
    rec['distance'] = distance
    rec = rec.sort_values('distance')
    rec = rec.loc[(rec['emotion'] == func_emotion) & (rec['release_date'] > year_min)]
    rec = rec.drop_duplicates(subset='title')
    columns = ['index', 'title', 'artist', 'popularity']
    return rec[columns][:amount]


def recommend_mix(gen_df, song_name, artist, genre, year_min, func_emotion, amount=1):
    distance = []
    mix_df = gen_df[(gen_df.genre.str.lower() == genre.lower()) |
                    (gen_df.artist.str.lower() == artist.lower()) |
                    (gen_df.title.str.lower() == song_name.lower())].head(1).values[0]
    rec = gen_df[(gen_df.genre.str.lower() != genre.lower()) &
                 (gen_df.title.str.lower() != song_name.lower()) &
                 (gen_df.artist.str.lower() != artist.lower())]
    for func_songs in tqdm(rec.values):
        d = 0
        for col in np.arange(len(rec.columns)):
            if not col in [0, 1, 2, 3, 4, 6, 11, 19]:
                d = d + np.absolute(float(mix_df[col]) - float(func_songs[col]))
        distance.append(d)
    rec['distance'] = distance
    rec = rec.sort_values('distance')
    rec = rec.loc[(rec['emotion'] == func_emotion) & (rec['release_date'] > year_min)]
    rec = rec.drop_duplicates(subset='title')
    columns = ['index', 'title', 'artist', 'popularity']
    return rec[columns][:amount]


def song_recommender(username, func_emotion):
    gen_df = pd.read_csv('./resources/finalUsable.csv')
    local_username = username
    liked_songs = []
    disliked_songs = []
    user_data = User.query.filter_by(user_name=local_username).first().get_data()
    local_id = user_data[0]
    rand_title = user_data[3]
    rand_artist = user_data[4]
    rand_year = int(user_data[6])
    rand_genre = user_data[5]
    liked_len = Appreciation.query.filter_by(user_id=local_id).count()
    if liked_len > 0:
        likes_dislikes_data = Appreciation.query.filter_by(user_id=local_id).all()
        for row in likes_dislikes_data:
            if row.liked == 1:
                if row.song_id not in liked_songs:
                    liked_songs.append(row.song_id)
            else:
                if row.song_id not in disliked_songs:
                    disliked_songs.append(row.song_id)
        rand_pos = random.randrange(0, len(liked_songs))
        df = pd.read_csv('./resources/finalUsable.csv')
        my_list = df.loc[df['index'] == liked_songs[rand_pos]].values.tolist()[0]
        rand_title = my_list[1]
        rand_artist = my_list[2]
        rand_year = int(my_list[3])
        rand_genre = my_list[4]

    # for user
    user_df = pd.DataFrame()
    user_emotion_df1 = recommend_based_on_song_name(gen_df, rand_title, func_emotion, amount=10)
    user_df = pd.concat([user_df, user_emotion_df1], ignore_index=True)
    user_emotion_df1 = recommend_based_on_genre(gen_df, rand_genre, func_emotion, amount=10)
    user_df = pd.concat([user_df, user_emotion_df1], ignore_index=True)
    user_emotion_df1 = recommend_based_on_genre_year(gen_df, rand_genre, rand_year, func_emotion, amount=10)
    user_df = pd.concat([user_df, user_emotion_df1], ignore_index=True)
    user_emotion_df1 = recommend_mix(gen_df, rand_title, rand_artist, rand_genre, rand_year, func_emotion, amount=10)
    user_df = pd.concat([user_df, user_emotion_df1], ignore_index=True)
    user_df = user_df.drop_duplicates(subset=['index', 'title'], keep='first')
    user_df = user_df[user_df.index.isin(disliked_songs) == False]

    # general good emotions
    good_df = pd.DataFrame()
    good_emotions = ['Happy', 'Neutral', 'Surprise']
    for emotion_iter in good_emotions:
        temp_df = recommend_based_on_song_name(gen_df, rand_title, emotion_iter, amount=4)
        good_df = pd.concat([good_df, temp_df], ignore_index=True)
        temp_df = recommend_based_on_genre(gen_df, rand_genre, emotion_iter, amount=4)
        good_df = pd.concat([good_df, temp_df], ignore_index=True)
        temp_df = recommend_based_on_genre_year(gen_df, rand_genre, rand_year, emotion_iter, amount=4)
        good_df = pd.concat([good_df, temp_df], ignore_index=True)
        temp_df = recommend_mix(gen_df, rand_title, rand_artist, rand_genre, rand_year, emotion_iter, amount=4)
        good_df = pd.concat([good_df, temp_df], ignore_index=True)
    good_df = good_df.drop_duplicates(subset=['index', 'title'], keep='first')
    good_df = good_df[good_df.index.isin(disliked_songs) == False]

    return user_df, good_df


def df_to_usable(df_to_list):
    """
    :param df_to_list: list of items in dataframe
    :return: a list of song IDs and a list of song hashes for youtube
    """
    id_list = []
    song_hash = []
    for item in df_to_list:
        id_list.append(item[0])
        name = item[1] + ' ' + item[2]
        name = name.replace(' ', '+')
        html = urllib.request.urlopen("https://www.youtube.com/results?search_query={0}".format(name.encode('utf-8')))
        video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
        song_hash.append(video_ids[0])
    return id_list, song_hash


@app.route('/songs', methods=['GET', 'POST'])
def songs():
    local_username = session['username']
    try:
        with open('./user_emotions/' + local_username + '.txt', 'r') as f:
            local_emotion = f.read().strip()
    except:
        return redirect(url_for('emotion_recognition'))
    session['emotion'] = local_emotion
    good_emotions = ['Happy', 'Neutral', 'Surprise']
    user_df, good_df = song_recommender(local_username, local_emotion)

    # get 4 songs based on knowledge
    if 'happy_count' not in session:
        session['happy_count'] = 99
    # get 4 songs sorted based on popularity
    # if it's in good emotion, we only take from the user_df
    if local_emotion in good_emotions:
        knowledge_list = user_df.iloc[:4, :].values.tolist()
        popularity_list = user_df.sort_values(by=["popularity"], ascending=False).iloc[:4, :].values.tolist()
        knowledge_id, knowledge_songs = df_to_usable(knowledge_list)
        popularity_id, popularity_songs = df_to_usable(popularity_list)
    else:
        if session['happy_count'] == 99:
            knowledge_list_user = user_df.iloc[:2, :].values.tolist()
            popularity_list_user = user_df.sort_values(by=["popularity"], ascending=False).iloc[:2, :].values.tolist()
            knowledge_list_good = good_df.iloc[:2, :].values.tolist()
            popularity_list_good = good_df.sort_values(by=["popularity"], ascending=False).iloc[:2, :].values.tolist()
            final_knowledge = knowledge_list_user + knowledge_list_good
            final_popularity = popularity_list_user + popularity_list_good
            knowledge_id, knowledge_songs = df_to_usable(final_knowledge)
            popularity_id, popularity_songs = df_to_usable(final_popularity)
            session['happy_count'] = 2
        else:
            # if there are liked songs
            if len(session['liked']) > 0:
                emotion_list = []
                df = pd.read_csv('./resources/finalUsable.csv')
                for sid in session['liked']:
                    songid = df[df['index'] == sid].index
                    songid = songid[0]
                    m_emotion = df.at[songid, 'emotion']
                    emotion_list.append(m_emotion)
                nmb_good_emotions, nmb_other_emotions = 0, 0
                for iter_emotion in emotion_list:
                    if iter_emotion in good_emotions:
                        nmb_good_emotions += 1
                    else:
                        nmb_other_emotions += 1
                percentage_good = float(nmb_good_emotions) / float(len(session['liked'])) * 100.0
                percentage_other = float(nmb_other_emotions) / float(len(session['liked'])) * 100.0
                local_cnt = session['happy_count']
                if percentage_good > 25.00 and percentage_other < 75.00:
                    if local_cnt < 4:
                        local_cnt += 1
                else:
                    if local_cnt > 1:
                        local_cnt -= 1
                session['happy_count'] = local_cnt

            nmb_of_songs_good = session['happy_count']
            nmb_of_user_songs = 4 - nmb_of_songs_good
            knowledge_list_good = good_df.iloc[:nmb_of_songs_good, :].values.tolist()
            popularity_list_good = good_df.sort_values(by=["popularity"],
                                                       ascending=False).iloc[:nmb_of_songs_good, :].values.tolist()
            if nmb_of_user_songs == 0:
                final_knowledge = knowledge_list_good
                final_popularity = popularity_list_good
            else:
                knowledge_list_user = user_df.iloc[:nmb_of_user_songs, :].values.tolist()
                popularity_list_user = user_df.sort_values(by=["popularity"],
                                                           ascending=False).iloc[:nmb_of_user_songs, :].values.tolist()
                final_knowledge = knowledge_list_user + knowledge_list_good
                final_popularity = popularity_list_user + popularity_list_good

            knowledge_id, knowledge_songs = df_to_usable(final_knowledge)
            popularity_id, popularity_songs = df_to_usable(final_popularity)

    session['liked'] = []
    session['disliked'] = []
    return render_template('new_recom//recompage.html', id1=knowledge_songs[0], id2=knowledge_songs[1],
                           id3=knowledge_songs[2], id4=knowledge_songs[3],
                           pid1=popularity_songs[0], pid2=popularity_songs[1], pid3=popularity_songs[2],
                           pid4=popularity_songs[3],
                           song_id1=knowledge_id[0], song_id2=knowledge_id[1], song_id3=knowledge_id[2],
                           song_id4=knowledge_id[3],
                           song_pid1=popularity_id[0], song_pid2=popularity_id[1], song_pid3=popularity_id[2],
                           song_pid4=popularity_id[3],
                           emotion=local_emotion.lower())


@app.route('/song_stats', methods=['POST'])
def song_stats():
    if request.method == 'POST':
        warning = False
        song_id = int(escape(request.form.get('song_id')))
        if not (0 < song_id < 9999):
            warning = True
        liked = int(escape(request.form.get('liked')))
        if liked not in [0, 1]:
            warning = True
        disliked = int(escape(request.form.get('disliked')))
        if disliked not in [0, 1]:
            warning = True
        # if sanitize is good
        if not warning:
            # get directly the user id
            if 'user_id' in session:
                local_id = session['user_id']
            # else query for it
            else:
                local_username = session['username']
                user_data = User.query.filter_by(user_name=local_username).first().get_data()
                local_id = user_data[0]
            liked_data = Appreciation.query.filter_by(user_id=local_id, song_id=song_id).count()
            # we checked if the song has been previously appreciated in either way
            if liked_data == 1:
                appreciation_data = Appreciation.query.filter_by(user_id=local_id, song_id=song_id).first().get_data()
                # if song was disliked and now liked
                if liked == 1 and appreciation_data[-1] == 0:
                    rating_user = Appreciation.query.filter_by(user_id=local_id, song_id=song_id).first()
                    rating_user.liked = 1
                    db.session.commit()
                    # add to popularity in .csv and to session variable
                    df = pd.read_csv('./resources/finalUsable.csv')
                    index_list = df[df['index'] == song_id].index
                    index_list = list(index_list)
                    df.iloc[index_list, 6] = df.loc[index_list].popularity + 1
                    df.to_csv('./resources/finalUsable.csv', index=False)
                # if song was liked and now disliked
                elif disliked == 1 and appreciation_data[-1] == 1:
                    rating_user = Appreciation.query.filter_by(user_id=local_id, song_id=song_id).first()
                    rating_user.liked = 0
                    db.session.commit()
                    # sub from popularity in .csv and to session variable
                    df = pd.read_csv('./resources/finalUsable.csv')
                    index_list = df[df['index'] == song_id].index
                    index_list = list(index_list)
                    df.iloc[index_list, 6] = df.loc[index_list].popularity - 1
                    df.to_csv('./resources/finalUsable.csv', index=False)
            # if song hasn't previously been liked or disliked
            else:
                if liked == 1:
                    liked_rating = 1
                    df = pd.read_csv('./resources/finalUsable.csv')
                    index_list = df[df['index'] == song_id].index
                    index_list = list(index_list)
                    df.iloc[index_list, 6] = df.loc[index_list].popularity + 1
                    df.to_csv('./resources/finalUsable.csv', index=False)
                else:
                    liked_rating = 0
                    df = pd.read_csv('./resources/finalUsable.csv')
                    index_list = df[df['index'] == song_id].index
                    index_list = list(index_list)
                    df.iloc[index_list, 6] = df.loc[index_list].popularity - 1
                    df.to_csv('./resources/finalUsable.csv', index=False)
                rating = Appreciation(user_id=local_id, song_id=song_id, liked=liked_rating)
                db.session.add(rating)
                db.session.commit()

            # we prepare data for next emotion recommendation, making sure the song hasn't been
            # appreciated multiple times
            if 'liked' in session and 'disliked' in session:
                liked_array = session['liked']
                disliked_array = session['disliked']
                if liked == 1:
                    if song_id in disliked_array:
                        disliked_array.remove(song_id)
                    liked_array.append(song_id)
                if disliked == 1:
                    if song_id in liked_array:
                        liked_array.remove(song_id)
                    disliked_array.append(song_id)
                liked_array = list(set(liked_array))
                disliked_array = list(set(disliked_array))
                session['liked'] = liked_array
                session['disliked'] = disliked_array
            return jsonify(status="success")


if __name__ == '__main__':
    app.run(host='localhost', port=5000, threaded=True, debug=True)
