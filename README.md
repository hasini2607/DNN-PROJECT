# Sentiment Analysis using IMDb Movie Reviews

This project performs sentiment analysis on IMDb movie reviews using multiple machine learning and deep learning techniques. The goal is to classify reviews as either positive or negative.

## Project Structure

- Data extraction and preprocessing
- Feature engineering using Bag of Words and TF-IDF
- Training classical ML models (Logistic Regression, Naive Bayes, SVM)
- Training deep learning models (CNN, DNN, RNN, LSTM, FNN)
- Performance evaluation and visualization

## Dataset

- Source: IMDb movie reviews dataset
- Size: 50,000 reviews (25,000 positive, 25,000 negative)
- Balanced: Yes
- Format: CSV, extracted from Google Drive

## Tools and Libraries Used

- pandas, numpy, math
- nltk (stopword removal, lemmatization)
- scikit-learn (Logistic Regression, Naive Bayes, SVM, evaluation metrics)
- keras / tensorflow (deep learning models)
- matplotlib (visualization)

## Data Preprocessing

- Removed HTML tags and URLs
- Converted text to lowercase
- Removed stopwords
- Applied lemmatization
- Tokenized and padded reviews for neural network models

## Feature Extraction

- Bag of Words (CountVectorizer) using unigrams, bigrams, trigrams
- TF-IDF with n-gram range (1, 3)

## Classical Machine Learning Models

| Model                  | Accuracy |
|------------------------|----------|
| Logistic Regression    | ~89.8%   |
| Naive Bayes            | ~88.3%   |
| Support Vector Machine | ~89.6%   |

Evaluation included precision, recall, and F1-score.

## Deep Learning Models

| Model     | Validation Accuracy | Comments                          |
|-----------|---------------------|-----------------------------------|
| CNN       | ~89.5%              | Good at detecting patterns        |
| DNN       | ~88%                | Simple but prone to overfitting   |
| RNN (LSTM)| ~88%                | Captures sequence dependencies    |
| FNN       | ~86–88%             | Flattens features, overfits fast  |

Overfitting was observed in most deep models based on training vs validation trends.

## Model Evaluation

- Accuracy and loss graphs for training and validation sets
- Classification reports with precision, recall, and F1-scores
- Confusion matrix analysis

## Conclusion

- Logistic Regression and SVM gave the best classical model performance.
- CNN gave the best deep learning model results.
- Overfitting can be mitigated with dropout, early stopping, and regularization.

## Future Improvements

- Try Bidirectional LSTM or GRU for better sequence learning
- Implement attention mechanisms
- Explore transformer models like BERT

## Requirements

- Python 3.6 or above
- pandas
- numpy
- nltk
- scikit-learn
- tensorflow / keras
- matplotlib

## How to Run

1. Clone the repository
2. Install the required libraries
3. Load the dataset and run the notebook or script
4. Train and evaluate different models as needed


