# Shantusha Ramsoebhag - 1825217 - 05/06/2023
import streamlit as st
from turtle import title
import requests
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import math
import sklearn as sk
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
import time 
import os
# importing TF-IDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
# importing image display 
from IPython.display import Image


# -------------------------streamlit page design----------------------------
# Settings
st.set_option('deprecation.showPyplotGlobalUse', False)

# Set streamlit page to full width 
st.set_page_config(layout='wide')


# Maak een CSS-stijlblad
st.write("""
    <style>
        .stText {
            height: 150px; 
            overflow-y: scroll; 
        }
    </style>
""", unsafe_allow_html=True)


# -------------------------importing csv---------------------------

nail_df = pd.read_csv('nail_dataset_prototype_new.csv', sep=';')
#print(nail_df.head)

# -------------------------data cleaning---------------------------
# checking data types
#print(nail_df.info())

# remove all extra spaces from column titles
nail_df.columns = nail_df.columns.str.strip()

# removing the extra spaces at the end of all strings of column nail art
nail_df['nail art'] = nail_df['nail art'].str.rstrip()
nail_df['Image url'] = nail_df['Image url'].str.rstrip()
nail_df['color'] = nail_df['color'].str.rstrip()
nail_df['color group'] = nail_df['color group'].str.rstrip()
nail_df['shape'] = nail_df['shape'].str.rstrip()
nail_df['length'] = nail_df['length'].str.rstrip()
nail_df['nail art level'] = nail_df['nail art level'].str.rstrip()
nail_df['nail art style'] = nail_df['nail art style'].str.rstrip()

# removing the rows without nail art
nail_df = nail_df.drop(nail_df[nail_df['nail art'] == 'no'].index)


# -------------------------Feature engineering---------------------------

# Dictionary with the emotion connotations for colors
color_dict = {'red': 'Admired, Affectionate, Excited, Astonished, Brave, Cheer, Confident, Energetic,Happy, Joy, Joyful, Love, Lust, Merry, Passion, Powerful, Proud, Satisfied, Secure, Strong, Surprised, Triumphant', 

'yellow': 'Affectionate, Astonished, Beautiful, Cheer, Confident, Elated, Energetic, Excited, Friendly, Grateful, Happy, Hopeful, Inspired, Joy, Joyful, Kind, Merry, Nice, Proud, Relaxed, Safe, Satisfied, Surprised, Triumphant',

'orange': 'Adventurous, Ambitious, Confident, Creative, Excited, Happy, Joy, Love, Passion',

'blue': 'Admired, Beautiful, Brave, Calm, Cheer, Comfortable, Confident, Elated, Friendly, Grateful, Hopeful, Inspired, Joyful, Kind, Neutral, Nice, Peaceful, Proud, Relaxed, Safe, Satisfied, Secure, Triumphant',

'purple': 'Calm, Comfortable, Creative, Excited, Mysterious, Powerful, Wise',

'pink': 'Admired, Affectionate, Beautiful, Calm,  Comfortable, Energetic, Excited, Feminine, Friendly, Girly, Happy, Inspired, Joy, Kind, Love, Lust, Nice, Passion, Youthful',

'green': 'Calm, Comfortable, Grateful, Hopeful, Kind, Merry, Neutral, Peaceful, Safe',

'white': 'Astonished, Calm, Clean, Elated, Hopeful, Inspired, Love, Neutral, Passion, Peaceful, Relaxed, Safe, Secure, Surprised',

'gray': 'Calm, Neutral, Safe',

'gold': 'Brave, Wise',

'silver': 'Calm, Powerful',

'brown': 'Comfortable, Earthy, Neutral',

'nude': 'Comfortable, Earthy, Neutral',

'black': 'Comfortable, Confident, Elegant, Mysterious, Powerful, Sensual, Strong',
}

# Loop over each row in the dataframe and connect the color connotations for occasion
for i, row in nail_df.iterrows():
    mood_values = []
    for color in color_dict.keys():
        if color in row['color']:
            mood_values.append(color_dict[color])
    if mood_values:
        nail_df.at[i, 'mood'] = ', '.join(mood_values)
    else:
        nail_df.at[i, 'mood'] = ''

#print(nail_df)

# Dictionary with the color connotations for occasions

occasion_dict = {

'green': 'outdoors, holiday',

'blue': 'outdoors, engaged, birthday, vacation',

'white': 'outdoors, engaged, holiday, wedding',

'yellow': 'outdoors, birthday, holiday, vacation',

'pink': 'engaged, birthday',

'red': 'birthday, holiday, vacation, wedding',

'orange': 'vacation',

'gold': 'engaged, wedding',

}

# Loop over each row in the dataframe and connect the color connotations for occasion
for i, row in nail_df.iterrows():
    occasion_values = []
    for color in occasion_dict.keys():
        if color in row['color']:
            occasion_values.append(occasion_dict[color])
    if occasion_values:
        nail_df.at[i, 'occasion'] = ', '.join(occasion_values)
    else:
        nail_df.at[i, 'occasion'] = ''

#nail_df

# Removing all double values from all rows. 
nail_df['mood'] = nail_df['mood'].apply(lambda x: ', '.join(pd.Series(x.split(', ')).drop_duplicates()))
nail_df['occasion'] = nail_df['occasion'].apply(lambda x: ', '.join(pd.Series(x.split(', ')).drop_duplicates()))
nail_df['mood'] = nail_df['mood'].str.lower()
#print(nail_df)

# creating 1 string of all important data points
nail_df['sum data'] = nail_df['color'] + ", " + nail_df['color group'] + ", " + nail_df['shape'] + ", " + nail_df['length'] + ", " + nail_df['nail art'] + ", " + nail_df['nail art level'] + ", " + nail_df['nail art style'] + ", " + nail_df['topic effect'] + ", " + nail_df['season'] + ", " + nail_df['mood'] + ", " + nail_df['occasion']

#print(nail_df)


# -------------------------making the recommender system---------------------------


# Creating a function that transforms the input data to vectors and calculates the cosine simalarity

def find_similar_images(user_input):
    # Setting the vectorizer and transform the sum data column to vectors.

    vectorizer = TfidfVectorizer()

    vectorized_data = vectorizer.fit_transform(nail_df['sum data'])

    shape_features = ['square', 'round', 'squoval', 'almond', 'coffin', 'stiletto']
    color_features = ['red', 'yellow', 'orange', 'blue', 'purple', 'pink', 'green', 'white', 'gray', 'gold', 'silver', 'brown', 'nude', 'black']
    length_features = ['short', 'middle', 'long']
    nailartstyle_features = ['3d', 'abstract', 'animal', 'print', 'animals', 'art', 'ballons', 'beach', 'brush', 'strokes', 'bunny', 'butterfly', 'cartoon', 'cat', 'checkerboard', 'cherries', 'chevron', 'chicken', 'chilli', 'chrome', 'clouds', 'dots', 'drinks', 'easter', 'eggs', 'eyes', 'fantasy', 'figures', 'gold', 'flakes', 'flames', 'flowers', 'french', 'tips', 'frog', 'fruit', 'geometric', 'ghost', 'glitters', 'halloween', 'hearts', 'ladybird', 'lemon', 'leopard', 'mandela', 'marble', 'metallic', 'micky', 'mouse', 'moon', 'mushroom', 'mystery', 'ombre', 'orange', 'painting', 'paisley', 'palm', 'tree', 'patterns', 'reindeer', 'retro', 'reversed', 'tip', 'smileys', 'spiritual', 'splatter', 'stars', 'stickers', 'stones', 'strawberry', 'stripes', 'sun', 'sunflower', 'swirl', 'text', 'vacation', 'vegetable', 'watermelon', 'ying', 'yang', 'zebra']
    nailartlevel_features = ['simple', 'medium', 'complex']
    topcoat_features = ['shine', 'shimmer', 'glitters', 'matte']
    colorgroup_features = ['light', 'dark', 'pastel', 'neon', 'normal']
    mood_features = ['admired', 'adventurous', 'affectionate', 'ambitious', 'astonished', 'beautiful', 'brave', 'calm', 'cheer', 'clean', 'comfortable', 'confident', 'creative', 'earthy', 'elated', 'elegant', 'energetic','excited','feminine', 'friendly', 'girly', 'grateful', 'happy','hopeful','inspired', 'joy', 'joyful','kind', 'love', 'lust', 'merry', 'mysterious', 'neutral', 'nice', 'passion', 'peaceful', 'powerful', 'proud', 'relaxed','safe', 'satisfied', 'secure', 'sensual', 'strong', 'surprised', 'triumphant', 'wise', 'youthful']
    season_features = ['summer', 'spring', 'winter', 'autumn']
    occasion_features = ['outdoors', 'holiday', 'engaged', 'birthday', 'vacation', 'wedding']

    shape_color_indexes = [vectorizer.vocabulary_[f] for f in shape_features]
    color_indexes = [vectorizer.vocabulary_[f] for f in color_features]
    colorgroup_indexes = [vectorizer.vocabulary_[f] for f in colorgroup_features]
    length_indexes = [vectorizer.vocabulary_[f] for f in length_features]
    nailartstyle_indexes = [vectorizer.vocabulary_[f] for f in nailartstyle_features]
    nailartlevel_indexes = [vectorizer.vocabulary_[f] for f in nailartlevel_features]
    topcoat_indexes = [vectorizer.vocabulary_[f] for f in topcoat_features]
    mood_indexes = [vectorizer.vocabulary_[f] for f in mood_features]
    season_indexes = [vectorizer.vocabulary_[f] for f in season_features]
    occasion_indexes = [vectorizer.vocabulary_[f] for f in occasion_features]

    idf_1 = vectorizer.idf_[shape_color_indexes]
    idf_2 = vectorizer.idf_[color_indexes]
    idf_3 = vectorizer.idf_[length_indexes]
    idf_4 = vectorizer.idf_[nailartstyle_indexes]
    idf_5 = vectorizer.idf_[nailartlevel_indexes]
    idf_6 = vectorizer.idf_[topcoat_indexes]
    idf_7 = vectorizer.idf_[colorgroup_indexes]
    idf_8 = vectorizer.idf_[mood_indexes]
    idf_9 = vectorizer.idf_[season_indexes]
    idf_10 = vectorizer.idf_[occasion_indexes]

    # Adding weights to the variables

    tfidf_weights = vectorized_data.toarray()[0]

    tfidf_weights[shape_color_indexes] = idf_1 * 8
    tfidf_weights[color_indexes] = idf_2 * 10
    tfidf_weights[length_indexes] = idf_3 * 6
    tfidf_weights[nailartstyle_indexes] = idf_4 * 5
    tfidf_weights[nailartlevel_indexes] = idf_5 * 4
    tfidf_weights[topcoat_indexes] = idf_6 * 10
    tfidf_weights[colorgroup_indexes] = idf_7 * 10
    tfidf_weights[mood_indexes] = idf_8 * 4
    tfidf_weights[season_indexes] = idf_9 * 4
    tfidf_weights[occasion_indexes] = idf_10 * 4

    vectorized_data = np.multiply(vectorized_data.toarray(), tfidf_weights)

    # Converts the user input into a vector text
    user_input_vector = vectorizer.transform([user_input])

    # Calculate the cosine similarity between the user input and the nail designs
    similarity_scores = cosine_similarity(user_input_vector, vectorized_data)

    # Sort the nail designs by their cosine similarity score
    sorted_indexes = np.argsort(similarity_scores)[0][::-1]
    print(sorted_indexes)
    # Select the top 5 nail designs
    top_nail_designs = nail_df.iloc[sorted_indexes[:5]]
    print(top_nail_designs)

    for i, index in enumerate(sorted_indexes[:5]):
        similarity_score = similarity_scores[0][index]
        print(f"Similarity score for image {i+1}: {similarity_score}")
    #print(tfidf_weights_colors[nail_color_indexes])


    # Return the top 5 nail designs
    return top_nail_designs


# Introduction 
st.title('Nail art design recommender system')

col1, col2 = st.columns(2, gap='large')

with col1:
    st.subheader('Hi, I am Valerie')
    styling = '<div style="margin-top: -30px">______________</div>'
    st.caption(styling, unsafe_allow_html=True)
    st.write('<div class="stText">I can help you quickly and easily choose a nail design, and I hope to inspire you with the best nail designs that suit your preferences. But first, I want to get to know you better before I can help you. Note that this is a prototype; therefore, the results of the recommendations can still vary from each other since the dataset is limited.</div>', unsafe_allow_html=True)
    #button = st.button('Get started')
    
    #if button: 
mood = st.multiselect(
        'What feelings would you like your nails to evoke?',
        ['no preferences','admired', 'adventurous', 'affectionate', 'ambitious', 'astonished', 'beautiful', 'brave', 'calm', 'cheer', 'clean', 'comfortable', 'confident', 'creative', 'earthy', 'elated', 'elegant', 'energetic','excited','feminine', 'friendly', 'girly', 'grateful', 'happy','hopeful','inspired', 'joy', 'joyful','kind', 'love', 'lust', 'merry', 'mysterious', 'neutral', 'nice', 'passion', 'peaceful', 'powerful', 'proud', 'relaxed','safe', 'satisfied', 'secure', 'sensual', 'strong', 'surprised', 'triumphant', 'wise', 'youthful'])
    #user_input = " ".join(color)

occasion = st.multiselect(
        'For what occasion are you planning to do your nails for?',
        ['no preferences', 'summer', 'spring', 'winter', 'autumn', 'outdoors', 'holiday', 'engaged', 'birthday', 'vacation', 'wedding'])

color = st.multiselect(
        'What color do you like to wear?',
        ['no preferences', 'red', 'yellow', 'orange', 'blue', 'purple', 'pink', 'green', 'white', 'gray', 'gold', 'silver', 'brown', 'nude', 'black'])


color_group = st.multiselect(
        'What type of color are you looking for?',
        ['no preferences','light', 'dark', 'pastel', 'neon', 'normal'])

shape = st.selectbox(
        'What shape do you like for your nails?',
        ['no preferences','square', 'round', 'squoval', 'almond', 'coffin', 'stiletto'])

length = st.selectbox(
        'What nail length do you prefer?',
        ['no preferences','short', 'middle', 'long'])

nail_art_level = st.selectbox(
        'What type of nail art would you like?',
        ['no preferences', 'simple', 'medium', 'complex'])

nail_art = st.multiselect(
        'What nail art style do you like?',
        ['no preferences', 'abstract', 'animal print', 'animals', 'art', 'brush strokes', 'cartoon', 'checkerboard', 'chevron', 'chrome', 'dots', 'fantasy', 'gold flakes', 'flames', 'flowers', 'french tips', 'fruit', 'geometric', 'glitters', 'hearts', 'mandela', 'marble', 'metallic', 'mystery', 'ombre','paisley', 'patterns', 'retro', 'reversed tip', 'smileys', 'spiritual', 'splatter', 'stars', 'stickers', 'stones', 'stripes', 'swirl', 'text', 'vegetable'])

top_effect = st.multiselect(
        'What would your top layer be?',
        ['no preferences', 'shine', 'shimmer', 'glitters', 'matte'])

user_input = " ".join([str(color)] + [str(color_group)] + [str(shape)] + [str(length)] + [str(nail_art)] + [str(nail_art_level)] + [str(top_effect)] + [str(occasion)] + [str(mood)])

top_images = find_similar_images(user_input)

questions_answered = False
like_dislike_list = []

if color and color_group and shape and length and nail_art and nail_art_level and top_effect and occasion and mood:    
    questions_answered = True

if questions_answered:
    #index = 0  # Huidige index van de afbeelding
    num_images = len(top_images)

    for current_index in range(num_images):
        row = top_images.iloc[current_index]
        image_url = row['Image url']
        st.image(image_url) 
        col3, col4 = st.columns([0.1, 0.5])
        like_button = col3.button(f'👍', key=f'like_{current_index}')
        dislike_button = col4.button(f'👎', key=f'dislike_{current_index}')

        if like_button:
            like_dislike_list.append((image_url, 'like'))
        
        if dislike_button:
            like_dislike_list.append((image_url, 'dislike'))



with col2:
    st.image('https://img.freepik.com/free-vector/manicurist-applies-nail-polish-client-nail-salon_1150-43287.jpg?w=740&t=st=1683889286~exp=1683889886~hmac=0d701e82986c10a3b1362f0d896baa5f51b3bc368a3afc5e69965a3048fe8943', width=320)