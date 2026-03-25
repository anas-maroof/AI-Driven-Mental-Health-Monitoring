# AI-Driven-Mental-Health-Monitoring

This project performs **sentiment analysis on mental-health related text** using NLP and machine learning.

The model learns patterns in text statements and predicts mental health categories such as:

- Depression
- Normal
- Anxiety
- Suicidal
- Anxiety
- Bipolar
- Personality disorder
  
The pipeline used in this project:

Text Cleaning → NLP Processing → Feature Extraction → Class Balancing → Model Training

---

# Project Setup

Follow these steps to run the project locally.

---

## 1. Clone the Repository

## 2. Create conda environment

```bash
conda create -p <envname> python=3.11.4
```
## 3.  activate the environment

```bash
conda activate <pathtoenv>
```
## 4.  install requirements.txt

```bash
pip install -r requirements.txt
```
Main libraries used in the project include:

-numpy – numerical computation
-pandas – data manipulation
-scikit-learn – machine learning models
-nltk – NLP utilities
-spacy – advanced NLP processing
-tqdm – progress bars
-contractions – expand English contractions
-imbalanced-learn – handle class imbalance
-kagglehub – download Kaggle datasets

## 5. Download the spaCy Language Model
```bash
python -m spacy download en_core_web_sm
```
## 6.install ipykernel for vscode
```bash
pip install ipykernel
```
## 7.register your Python environment as a Jupyter kernel
```bash
python -m ipykernel install --user --name=<envname>
```
## 8. run notebook cell 


# NOTEBOOK OVERVIEW

## Raw Text Exploratory Data Analysis (EDA)
- Load the dataset and inspect its structure (shape, columns, data types)
- Check for missing values in text and label columns
- Identify duplicate records in the dataset
- Analyze class distribution and calculate percentage imbalance
- Compute text length and word count statistics on raw text
- Visualize label distribution for imbalance detection
- Display sample raw texts for manual inspection

## NLP Resource Initialization & Setup
- Initialize placeholders for spaCy model, stopwords, and lemmatizer
- Download required NLTK resources (WordNet, Stopwords, Punkt)
- Create English stopword set
- Initialize WordNet Lemmatizer
- Verify successful NLTK resource loading
- Load spaCy English model (en_core_web_sm)
- Enable spaCy-based lemmatization if available
- Handle errors gracefully with fallback mechanisms
- Ensure stable and robust NLP environment before preprocessing

## Data Loading & Preparation
- Initialize dataset path, column names, test size (20%), and random seed
- Load dataset from CSV with fallback dummy data if file is missing
- Clean column names and remove unnecessary index column
- Validate required text and label columns
- Handle missing values and ensure correct data types
- Extract unique labels and create numerical label mapping
- Separate features (X) and target (y)
- Perform stratified train-test split with fallback handling
- Display dataset sizes and confirm successful setup

## Advanced Text Preprocessing Pipeline

- Contraction expansion (e.g., can't → cannot)
- URL, mention, and hashtag normalization
- Acronym and slang expansion (e.g., idk → I do not know)
- Digit–word normalization (e.g., gr8 → great)
- Repeated character reduction (e.g., soooo → soo)
- Special character and noise removal
- Stopword removal to reduce irrelevant words
- Lemmatization to convert words to base form
- Short-text filtering for meaningful samples
- Dataset validation to ensure proper class distribution

## SMOTE Oversampling

* Convert cleaned text data into numerical feature vectors using **TF-IDF**
* Limit vocabulary to the top 5000 most important features
* Fit TF-IDF on training data and transform both training and test sets
* Initialize **SMOTE** with fixed random seed for reproducibility
* Apply SMOTE only on the training data to handle class imbalance
* Generate balanced training feature set and corresponding labels
* Keep test data unchanged to ensure fair model evaluation
* Store resampled training data and original test data separately for further modeling

### Word Cloud Visualization for Each Sentiment Class

- Generate **Word Clouds** to visualize the most frequent words for each sentiment class.
- Combine all cleaned training texts belonging to the same label.
- Iterate through each sentiment category using the `label_id` mapping.
- Create word clouds using the **WordCloud** library with a **white background** and **viridis colormap**.
- Disable **collocations** to highlight individual word frequencies clearly.
- Use already **preprocessed text** (stopwords removed during cleaning).
- Display generated word clouds using **Matplotlib**.
- Save each word cloud image separately in the **output directory** for further analysis.
- Handle cases where a class has **empty or insufficient text** to prevent runtime errors.

### N-gram Analysis for Each Sentiment Class

- Perform **N-gram analysis** to identify frequently occurring word patterns within each sentiment class.
- Create a temporary dataset combining cleaned training text and corresponding labels.
- Compute **Unigrams, Bigrams, and Trigrams** for each sentiment category.
- Use **CountVectorizer** to convert text into N-gram frequency features.
- Apply `min_df = 2` to ignore extremely rare N-grams and focus on meaningful patterns.
- Limit the vocabulary to **2000 features** to maintain computational efficiency.
- Extract the **top 10 most frequent N-grams** for each sentiment class.
- Visualize N-gram frequencies using **Seaborn bar plots**.
- Automatically adjust plot size for better readability.
- Save generated N-gram plots in the **output directory** for further analysis.
- Include error handling for cases where a class has **insufficient text or valid N-grams**.

### Baseline Model: Logistic Regression

- Implement **Logistic Regression** as the baseline model for multi-class mental health text classification.
- Convert cleaned text data into numerical features using **TF-IDF vectorization**.
- Configure TF-IDF with **5000 maximum features**, **minimum document frequency of 3**, and **(1,2) n-gram range** to capture both unigrams and bigrams.
- Fit the TF-IDF vectorizer on the **training dataset** and transform both training and test datasets.
- Perform **hyperparameter tuning using GridSearchCV** with different solvers (`liblinear`, `saga`), penalties (`l1`, `l2`), regularization strengths (`C`), and optional class balancing.
- Use **K-Fold Cross Validation** with shuffled splits and a fixed random seed for reproducibility.
- Optimize model performance using **weighted F1-score**, suitable for multi-class and imbalanced datasets.
- Include **fallback training with a default Logistic Regression model** if GridSearchCV fails.
- Train the final model using the processed TF-IDF training data.
- Evaluate model performance on the **test dataset** using:
  - Precision
  - Recall
  - F1-score
  - Accuracy
- Generate and visualize a **Confusion Matrix** to analyze prediction performance across all sentiment classes.
- Save the confusion matrix visualization in the **output directory** for reporting and analysis.

### Logistic Regression with SMOTE Dataset

- Train **Logistic Regression** on the **SMOTE-balanced dataset** to address class imbalance.
- Use the **TF-IDF feature vectors generated earlier** from the text preprocessing pipeline.
- Training dataset contains **84,987 samples**, while the test dataset contains **8,958 samples**.
- Perform **hyperparameter tuning using GridSearchCV** to identify the optimal model configuration.
- Evaluate different values of **regularization strength (C = 0.1, 1, 10)** with **L2 regularization**.
- Use **`liblinear` solver** suitable for smaller and sparse datasets such as TF-IDF matrices.
- Apply **K-Fold Cross Validation (3 splits)** with shuffling and a fixed random seed for reproducibility.
- Optimize model selection using **weighted F1-score**, which accounts for class imbalance across categories.
- Best model configuration obtained:
  - **C = 10**
  - **Penalty = L2**
  - **Solver = liblinear**
- Achieved **best cross-validation weighted F1-score of 0.865** on the training data.
- Evaluate the trained model on the **held-out test dataset**.
- Report performance metrics including **precision, recall, F1-score, and overall accuracy**.
- Achieved **69% test accuracy** on the SMOTE-balanced training setup.
- Generate and visualize a **Confusion Matrix** to analyze class-wise prediction performance.

### Logistic Regression with Cost-Sensitive Learning

- Implement **Logistic Regression with Cost-Sensitive Learning** to handle class imbalance without oversampling.
- Convert cleaned text into numerical features using **TF-IDF vectorization**.
- Configure TF-IDF with **5000 maximum features**, **minimum document frequency of 3**, and **(1,2) n-gram range** to capture both unigrams and bigrams.
- Fit the TF-IDF vectorizer on the **training dataset** and transform both training and test datasets.
- Compute **class weights automatically using `compute_class_weight`** to penalize misclassification of minority classes.
- Create a **class weight dictionary** and pass it to the Logistic Regression model to implement cost-sensitive learning.
- Perform **hyperparameter tuning using GridSearchCV** to select the best regularization strength.
- Evaluate values of **C = [0.1, 1, 10]** with **L2 regularization** and **liblinear solver**.
- Use **K-Fold Cross Validation** with shuffled splits and a fixed random seed for reproducibility.
- Optimize model selection using **weighted F1-score**, which is suitable for imbalanced multi-class classification.
- Include a **fallback model training step** if GridSearchCV fails or there are insufficient samples.
- Train the final model on the **TF-IDF training dataset with computed class weights**.
- Evaluate the model on the **test dataset** using **precision, recall, F1-score, and accuracy**.
- Generate and visualize a **Confusion Matrix** to analyze class-wise prediction performance.
- Save the confusion matrix plot in the **output directory** for reporting and analysis.


### Deep Learning Model: RoBERTa (Robust Transformers)

- Uses **roberta-base** for multi-class mental-health sentiment classification.
- Includes seed/device setup and environment checks before training.
- Performs tokenization with **RobertaTokenizerFast** using padding/truncation.
- Builds PyTorch **Dataset/DataLoader** pipelines for train/validation/test splits.
- Trains with **class-weighted loss + AdamW + warmup scheduler + gradient clipping**.
- Uses validation-based **early stopping** and saves the best checkpoint.
- Reports test **accuracy, classification report, and confusion matrix**.
- Reloads the **best saved model checkpoint** before final test evaluation.
- Uses stable batch-wise training and validation loops with per-epoch metric tracking.

### BERT-Fuse Hybrid Model

- Uses **bert-base-uncased** as the contextual encoder branch.
- Generates handcrafted features using **Bag-of-Words** and **TF-IDF**.
- Prepares tokenized + handcrafted inputs in a custom hybrid dataset pipeline.
- Uses a fused architecture: **Bi-LSTM + Bi-GRU + Transformer block + CNN**.
- Concatenates fused deep features with handcrafted vectors before classification.
- Trains with **AdamW**, scheduler, and monitored validation performance.
- Evaluates with **accuracy, classification report, and confusion matrix**.
- Includes reproducibility controls (fixed seeds and deterministic settings where applicable).
- Runs full train/validation/test workflow with checkpoint-based best-model selection.

