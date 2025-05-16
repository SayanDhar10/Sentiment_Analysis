# Sentiment Analysis
This repository presents a detailed sentiment analysis project aimed at classifying text data into positive or negative sentiments using a Random Forest classifier. The core goal is to evaluate customer reviews—such as those from restaurants—and determine whether the expressed sentiment is positive (Liked) or negative (Disliked). The project employs advanced machine learning techniques and thorough text preprocessing to convert raw text into actionable insights. Key steps include data cleaning, feature extraction using TF-IDF, and training a reliable Random Forest model for accurate sentiment prediction. Comprehensive visualizations are also included to illustrate model performance and highlight important features, making the project a valuable resource for interpreting customer feedback.
# Features
* Text data preprocessing, including tokenization, stemming, and removal of stopwords
* Feature extraction from text using TF-IDF vectorization
* Sentiment prediction performed with a Random Forest classifier
* Visual representations such as the confusion matrix, feature importance chart, and probability distribution of predictions

# Methodology

1. Data Preprocessing
* Eliminate non-alphabetic characters from the text
* Convert all text to lowercase
* Perform tokenization to split text into individual words
* Remove stopwords using the NLTK library
* Apply stemming with the Porter Stemmer to reduce words to their root form

2. Feature Extraction
* Use TF-IDF vectorization to transform text into numerical feature representations

3. Model Training
* Train a Random Forest classifier on the processed feature set

4. Evaluation and Visualization
* Assess model performance using accuracy and a classification report
* Create visualizations including the confusion matrix, feature importance plot, and predicted probability distribution

