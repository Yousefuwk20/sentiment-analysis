# Twitter Sentiment Analysis with AWS SageMaker

This repository contains a complete pipeline for performing sentiment analysis on Twitter data using AWS SageMaker’s BlazingText algorithm. The goal is to classify tweets as expressing positive or negative sentiment, using data from the Sentiment140 dataset.

## Project Overview

This project leverages the Sentiment140 dataset, which provides a large corpus of tweets labeled as either positive or negative, enabling us to train a machine learning model for sentiment classification. The model is trained using SageMaker’s BlazingText in supervised mode, optimized for large-scale text processing.

## Key Components

### 1. **Data Preprocessing**
   - **Cleaning**: Text data undergoes preprocessing to remove noise. This includes:
     - Lowercasing text
     - Removing URLs, mentions (@username), hashtags, special characters, and numbers.
   - **Labeling**: Tweets are labeled as `__label__0` (negative) or `__label__4` (positive) for BlazingText compatibility.
   - **Tokenization**: Text data is tokenized for the model, which works more effectively with preprocessed tokens.

### 2. **Data Visualization and Exploratory Analysis**
   - **Sentiment Distribution**: Visualizations reveal the class distribution, showing any imbalance between positive and negative tweets.
   - **Tweet Length Distribution**: Histogram analysis displays variations in tweet length, providing insight into tweet brevity or verbosity.
   - **Common Words**: Word clouds and bar charts illustrate the most frequent words in positive and negative tweets, giving a quick glimpse of sentiment-driving keywords.

### 3. **Model Training**
   - **AWS SageMaker’s BlazingText Algorithm**: We use BlazingText in supervised mode, a powerful tool optimized for text classification tasks. It’s scalable and designed to handle large text datasets efficiently.
   - **Training**: Training is carried out on SageMaker-managed instances, utilizing either `ml.m5.large` or `ml.t2.medium` instances.
   - **Hyperparameter Tuning**: Key hyperparameters such as learning rate and batch size are fine-tuned for optimal model performance.

### 4. **Model Deployment**
   - **SageMaker Endpoint**: The trained BlazingText model is deployed as a real-time endpoint on SageMaker, allowing for dynamic tweet sentiment prediction.
   - **Inference Pipeline**: The endpoint receives tweet text, preprocesses it, and outputs sentiment predictions, either positive or negative.

## Dataset

- **Sentiment140**: The dataset includes 1.6 million tweets labeled as either positive or negative.
- **Classes**:
   - **`__label__0`**: Negative sentiment
   - **`__label__4`**: Positive sentiment

## Project Structure

- **notebooks/**: Jupyter notebooks for data preprocessing, training, deployment, and visualization.
- **scripts/**: Scripts for preprocessing and inference functions.
- **data/**: Contains preprocessed data files used for training and evaluation.
- **docs/**: Documentation on model insights, reports, and analysis.

## Visualizations and Insights

Several visualizations are provided to understand the data and its structure, including:
- **Sentiment distribution chart** to see if classes are balanced.
- **Tweet length histogram** to analyze tweet length variations.
- **Word clouds** and **top words** per sentiment to highlight common terms in positive and negative tweets.

## Setup and Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/username/sentiment-analysis.git
   ```

2. **Set up AWS and SageMaker**:
   - Configure your AWS account and ensure you have permissions to access SageMaker and S3.
   - Upload the Sentiment140 dataset to an S3 bucket for seamless data access during training.

3. **Run the Notebooks**:
   - Navigate to the `notebooks/` directory and follow the Jupyter notebooks step-by-step for data preprocessing, training, and deployment.

## Requirements

- **AWS Services**: SageMaker, S3
- **Python Libraries**: `pandas`, `boto3`, `sagemaker`, `matplotlib`, `seaborn`, `wordcloud`, `scikit-learn`
  
Install dependencies using:
```bash
pip install -r requirements.txt
```
## LinkedIn Demo
A demonstration of this sentiment analysis project is available on my LinkedIn profile, showcasing the model's capabilities in real-time tweet sentiment prediction. Feel free to check it out for a visual walkthrough of the project's features and performance.

## Future Improvements

- **Additional Preprocessing**: Experiment with advanced NLP techniques like stemming and lemmatization.
- **Multiclass Sentiment Analysis**: Extend the model to classify neutral or mixed sentiments.
- **Hyperparameter Optimization**: Further tuning of BlazingText parameters for enhanced performance.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for feedback, bug reports, or improvements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

