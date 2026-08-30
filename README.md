# Arabic Cyberbullying Detection Using Deep Learning

An end-to-end deep learning project for detecting cyberbullying in Arabic tweets using LSTM and GRU recurrent neural networks.

The project includes text preprocessing, class imbalance handling, model training, model comparison, and final evaluation.

## Project Overview

Cyberbullying is a serious problem on social media, especially when harmful or insulting language is used against others.

The goal of this project is to automatically classify Arabic tweets into two categories:

- Non-bullying
- Bullying

The project compares multiple recurrent deep learning models and evaluates whether class weighting can improve the detection of bullying tweets.

## Dataset

The project uses the ArbCyD Arabic Cyberbullying Dataset.

The dataset contains:

- 10,000 tweets
- 6,204 non-bullying tweets
- 3,796 bullying tweets
- No missing values

The original columns are:

- `Tweets`
- `Domain`
- `Label`

The tweet text is used as the model input, while `Label` is the target variable.

## Project Workflow

1. Dataset loading and inspection
2. Target label exploration
3. Tweet length analysis
4. Label encoding
5. Stratified train-validation-test split
6. Text tokenization
7. Sequence padding
8. Baseline LSTM model
9. Class-weighted LSTM
10. Weighted GRU
11. Early stopping
12. Confusion matrix and classification report
13. Model comparison
14. Final model selection

## Data Preprocessing

The labels were encoded as:

- Non-bullying = 0
- Bullying = 1

The dataset was divided into:

- 70% training
- 15% validation
- 15% testing

Stratified splitting was used to preserve approximately the same class distribution in each subset.

The text was tokenized using a vocabulary limit of 10,000 words.

All sequences were padded or truncated to a maximum length of 30 tokens.

## Deep Learning Architecture

The baseline LSTM architecture consisted of:

- Embedding layer with 64-dimensional embeddings
- LSTM layer with 64 units
- Dropout layer
- Sigmoid output layer

The model used:

- Adam optimizer
- Binary cross-entropy loss
- Early stopping

## Models Compared

| Model | Test Accuracy | Bullying Precision | Bullying Recall | Bullying F1 |
|---|---:|---:|---:|---:|
| Baseline LSTM | 76.60% | 0.73 | 0.61 | 0.67 |
| Weighted LSTM | **77.07%** | 0.70 | **0.69** | **0.70** |
| Weighted GRU | 76.20% | **0.75** | 0.57 | 0.64 |

## Final Model

The Weighted LSTM was selected as the final model.

It achieved:

- **Test Accuracy:** 77.07%
- **Bullying Precision:** 0.70
- **Bullying Recall:** 0.69
- **Bullying F1-score:** 0.70

Class weighting improved the model's ability to detect bullying tweets compared with the baseline LSTM.

## Confusion Matrix

The final Weighted LSTM produced:

```text
[[762, 169],
 [175, 394]]
