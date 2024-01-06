# Movie Genre Predictor

![Project Logo](https://img.icons8.com/?size=100&id=80698&format=png)

## Overview

Movie Genre Predictor is a web application built using Streamlit that predicts the genre of a movie based on its title and description. It utilizes machine learning models trained on movie data to provide genre predictions.

## Table of Contents

- [Demo](#demo)
- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Models and Accuracy](#models-and-accuracy)
- [Input Options](#input-options)
- [About the Project](#about-the-project)
- [License](#license)
- [Contact](#contact)

## Demo

![Demo GIF](https://im7.ezgif.com/tmp/ezgif-7-13f70baab3.gif)

## Features

- Predict movie genre from both manually entered and pre-existing movie titles.
- User-friendly web interface with movie information display.
- Multiple machine learning models for genre prediction.
- genre predictions with high model accuracy.

## Getting Started

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/pankil-soni/movie-genre-predictor.git
   cd movie-genre-predictor
## Install dependencies:

```
pip install -r requirements.txt --save @github/clipboard-copy-element
```
    
## Usage
Run the following command to start the Streamlit app:

```bash
Copy code
streamlit run app.py
Visit http://localhost:8501 in your web browser to access the application.
```
## Models and Accuracy

- $${\color{red}Logistic Regression}$$: Train Accuracy - **94.69%**, Test Accuracy - **51.59%**
- $${\color{red}BernoulliNB}$$: Train Accuracy - **91.34%**, Test Accuracy - **52.72%**
- $${\color{red}MultinomialNB}$$: Train Accuracy - **93.19%**, Test Accuracy - **51.65%**
- $${\color{red}Random Forest}$$: Train Accuracy - **100%**, Test Accuracy - **48.84%**
- $${\color{red}Support Vector}$$: Train Accuracy - **99.09%**, Test Accuracy - **48.31%**
## Input Options
The application provides two options for input:

1. Search Movie: Choose from pre-existing movie titles with actual genre information.
2. Manually Enter Title and Description: Enter your own movie title and description for genre prediction.

## About the Project
This project is a Movie Genre Predictor web application made using Streamlit and deployed on Streamlit Server. The movie data is sourced from IMDB, providing genre information for predictions.

Made by **Pankil Soni**

## License
This project is licensed under the MIT License.

## Contact
For inquiries or feedback, please contact your-pmsoni2016@gmail.com.
