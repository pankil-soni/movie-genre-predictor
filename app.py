import streamlit as st
import pickle as pkl
import pandas as pd
import requests
import re
import string
from nltk.corpus import stopwords
import nltk
stopwd = stopwords.words('english')
def clean_text(text):
    
    text= text.lower() # Lowercasing the text
    text = re.sub('Mail <svaradi@sprynet.com> for translation. ', '', text) # Removing unknown mail
    text = re.sub('-',' ',text.lower())   # Replacing `x-x` as `x x`
    text = re.sub(r'@\S+', '', text) # Removing mentions
    text = re.sub(r'http\S+', '', text) # Removing Links
    text = re.sub(f'[{string.punctuation}]', '', text) # Remove punctuations
    text = re.sub(f'[{string.digits}]', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text) # Removing unnecessary spaces
    text = re.sub(r'\s+[a-zA-Z]\s+', ' ', text) # Removing single characters
    
    words = nltk.tokenize.word_tokenize(text,language="english", preserve_line=True)
    text = " ".join([i for i in words if i not in stopwd and len(i)>2]) # Removing the stop words

    return text.strip()


# Function to load data
def load_data(file_name):
    return pkl.load(open(file_name, 'rb'))

# Function to load titles, dataframe, model, vectorizer, and encoder
load_titles = lambda: load_data('titles.pkl')
load_dataframe = lambda: load_data('dataframe.pkl')
load_model = lambda modelname: pkl.load(open(f'{modelname}.pkl', 'rb'))
load_vectorizer = lambda: load_data('Cvectorizer.pkl')
load_encoder = lambda: load_data('encoder.pkl')


# HTML template for displaying genre predictions
html_string = '''
<p style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif">
    <span style="font-weight: bold;font-size: x-large;">{} Genre:</span>
    <strong style="color: rgb(110,181,47)">"{}"</strong>
</p>
'''

# Streamlit configuration
st.set_page_config(layout='centered', page_title='Movie Genre Predictor',
                   page_icon="🎬", initial_sidebar_state='auto')

# Header and caption
st.header("Movie Genre Predictor")
st.caption("Made by Pankil Soni")
st.markdown('''<a href="https://www.linkedin.com/in/pankil-soni-5a0541170/" target="_blank"><svg xmlns="http://www.w3.org/2000/svg" height="16" width="14" viewBox="0 0 448 512"><!--!Font Awesome Free 6.5.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M100.3 448H7.4V148.9h92.9zM53.8 108.1C24.1 108.1 0 83.5 0 53.8a53.8 53.8 0 0 1 107.6 0c0 29.7-24.1 54.3-53.8 54.3zM447.9 448h-92.7V302.4c0-34.7-.7-79.2-48.3-79.2-48.3 0-55.7 37.7-55.7 76.7V448h-92.8V148.9h89.1v40.8h1.3c12.4-23.5 42.7-48.3 87.9-48.3 94 0 111.3 61.9 111.3 142.3V448z"/></svg></a> <a href="https://github.com/pankil-soni/" target="_blank"> <svg xmlns="http://www.w3.org/2000/svg" height="16" width="15.5" viewBox="0 0 496 512"><!--!Font Awesome Free 6.5.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3 .3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5 .3-6.2 2.3zm44.2-1.7c-2.9 .7-4.9 2.6-4.6 4.9 .3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3 .7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3 .3 2.9 2.3 3.9 1.6 1 3.6 .7 4.3-.7 .7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3 .7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3 .7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"/></svg></a>''',unsafe_allow_html=True)

# Model accuracies
accuracies = [
    {"Model": "Logistic Regression", "Train Accuracy": 0.9468998860022146, "Test Accuracy": 0.5159040590405904},
    {"Model": "BernoulliNB", "Train Accuracy": 0.9134487458883257, "Test Accuracy": 0.5271955719557195},
    {"Model": "MultinomialNB", "Train Accuracy": 0.9319468590753393, "Test Accuracy": 0.5165313653136532},
    {"Model": "Random Forest", "Train Accuracy": 1.0, "Test Accuracy": 0.48837638376383763},
    {"Model": "Support Vector", "Train Accuracy": 0.9908692943292223, "Test Accuracy": 0.4830811808118081}
]

# Model codes
model_codes = {"Logistic Regression": "lgc", "Random Forest": "rdmfr",
               "SVM": "svc", "Multinomial Naive Bayes": "mnb", "Bernoulli Naive Bayes": "bnb"}

# Columns for model selection and accuracies
col1, col2 = st.columns(2)
with col1:
    model = st.selectbox("Select the model to use", list(model_codes.keys()), key="model_select")
with col2:
    df = pd.DataFrame(accuracies)
    df.index = df["Model"]
    df.drop("Model", axis=1, inplace=True)
    st.write(df)

# Radio button for input selection
menu_id = st.radio("Select an option to enter the input", ("Search Movie", "Manually write title and description"))

if menu_id == "Search Movie":
    # Load titles and display options
    titles = load_titles()
    option = st.selectbox('Select the movie Name:', tuple(titles[::10]), key="movie_select")
    
    # Load data and display information
    df = load_dataframe()
    row = df[df['TITLE'] == option]
    glbtitle, glbdescription, glbimg, actual_genre = row["TITLE"].values[0], row["DESCRIPTION"].values[0], row["Poster_Link"].values[0], row["GENRE"].values[0]

    # Display information in columns
    col1, col2 = st.columns(2)
    with col1:
        st.header(f"Title: {glbtitle}")
        st.write(f"Description: {glbdescription}")

    with col2:
        try:
            image = requests.get(glbimg).content
            st.image(image, width=200)
        except Exception as e:
            st.error("Failed to load Poster")

    # Predict genre on button click
            
    if st.button('Predict Genre'):
        with st.spinner("Predicting Genre..."):
            result = load_model(model_codes[model]).predict(load_vectorizer().transform([clean_text(glbtitle + " " + glbdescription)]))
            result = load_encoder().inverse_transform(result)[0]
        st.markdown(html_string.format("Predicted", result.strip().title()), unsafe_allow_html=True)
        st.markdown(html_string.format("Actual", actual_genre.strip().title()), unsafe_allow_html=True)
else:
    # Manual input for title and description
    title = st.text_input('Enter the movie title:')
    desc = st.text_area('Enter the movie Description:')

    # Predict genre on button click
    if st.button('Predict Genre'):
        with st.spinner("Predicting Genre..."):
            result = load_model(model_codes[model]).predict(load_vectorizer().transform([clean_text(str(title) + " " + str(desc))]))
            result = load_encoder().inverse_transform(result)[0]
        st.markdown(html_string.format("Predicted", result.strip().title()), unsafe_allow_html=True)

# About the project section
st.header("About the Project")
st.markdown("""
<style>
.footer {
    position: relative;
    left: 0;
    bottom: 0;
    width: 100%;
    color: black;
    text-align: left;
}
.git {
    padding-top: 15px;
}
</style>
<div class="footer">
    This project is made using Streamlit and deployed on Heroku. <br>
    The movies in the search bar are taken from IMDB with the actual genres.<br>
    <p class="git">Made by <a href="https://github.com/pankil-soni/" target="_blank">
        <svg xmlns="http://www.w3.org/2000/svg" height="16" width="15.5" viewBox="0 0 496 512">
            <!--!Font Awesome Free 6.5.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.-->
            <path d="M165.9 397.4c0 2-2.3 3.6-5.2 3.6-3.3 .3-5.6-1.3-5.6-3.6 0-2 2.3-3.6 5.2-3.6 3-.3 5.6 1.3 5.6 3.6zm-31.1-4.5c-.7 2 1.3 4.3 4.3 4.9 2.6 1 5.6 0 6.2-2s-1.3-4.3-4.3-5.2c-2.6-.7-5.5 .3-6.2 2.3zm44.2-1.7c-2.9 .7-4.9 2.6-4.6 4.9 .3 2 2.9 3.3 5.9 2.6 2.9-.7 4.9-2.6 4.6-4.6-.3-1.9-3-3.2-5.9-2.9zM244.8 8C106.1 8 0 113.3 0 252c0 110.9 69.8 205.8 169.5 239.2 12.8 2.3 17.3-5.6 17.3-12.1 0-6.2-.3-40.4-.3-61.4 0 0-70 15-84.7-29.8 0 0-11.4-29.1-27.8-36.6 0 0-22.9-15.7 1.6-15.4 0 0 24.9 2 38.6 25.8 21.9 38.6 58.6 27.5 72.9 20.9 2.3-16 8.8-27.1 16-33.7-55.9-6.2-112.3-14.3-112.3-110.5 0-27.5 7.6-41.3 23.6-58.9-2.6-6.5-11.1-33.3 2.6-67.9 20.9-6.5 69 27 69 27 20-5.6 41.5-8.5 62.8-8.5s42.8 2.9 62.8 8.5c0 0 48.1-33.6 69-27 13.7 34.7 5.2 61.4 2.6 67.9 16 17.7 25.8 31.5 25.8 58.9 0 96.5-58.9 104.2-114.8 110.5 9.2 7.9 17 22.9 17 46.4 0 33.7-.3 75.4-.3 83.6 0 6.5 4.6 14.4 17.3 12.1C428.2 457.8 496 362.9 496 252 496 113.3 383.5 8 244.8 8zM97.2 352.9c-1.3 1-1 3.3 .7 5.2 1.6 1.6 3.9 2.3 5.2 1 1.3-1 1-3.3-.7-5.2-1.6-1.6-3.9-2.3-5.2-1zm-10.8-8.1c-.7 1.3 .3 2.9 2.3 3.9 1.6 1 3.6 .7 4.3-.7 .7-1.3-.3-2.9-2.3-3.9-2-.6-3.6-.3-4.3 .7zm32.4 35.6c-1.6 1.3-1 4.3 1.3 6.2 2.3 2.3 5.2 2.6 6.5 1 1.3-1.3 .7-4.3-1.3-6.2-2.2-2.3-5.2-2.6-6.5-1zm-11.4-14.7c-1.6 1-1.6 3.6 0 5.9 1.6 2.3 4.3 3.3 5.6 2.3 1.6-1.3 1.6-3.9 0-6.2-1.4-2.3-4-3.3-5.6-2z"/></svg> Pankil Soni</a></p>
</div>
""", unsafe_allow_html=True)