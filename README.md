# Financial News Classification & t-SNE Visualization

## Project Overview

This project explores machine learning approaches for representing, visualizing, and classifying financial news headlines. It focuses on four categories:

- **Upgrade:** Analyst upgrades or improved stock recommendations.
- **Downgrade:** Analyst downgrades or reduced stock recommendations.
- **Positive:** Favorable financial developments, such as strong earnings or revenue growth.
- **Negative:** Unfavorable financial developments, such as missed earnings expectations or declining performance.

The project uses TF-IDF vectorization, Truncated Singular Value Decomposition (SVD), t-distributed Stochastic Neighbor Embedding (t-SNE), and logistic regression to analyze financial headline representations.

It also compares machine learning classification against a rule-based keyword baseline.

[Slide Deck](https://docs.google.com/presentation/d/1vKvcyPZ9ExyO35IokbTPHvjCRCj81wwpp-HqLZYpCxo/edit?usp=sharing)

## Objectives

1. Generate a labeled dataset of synthetic financial headlines.
2. Convert textual headlines into numerical feature representations.
3. Reduce dimensionality using Truncated SVD.
4. Visualize headline embeddings using t-SNE at different sample sizes.
5. Train a logistic regression classifier to categorize financial headlines.
6. Compare model accuracy against a keyword-based classifier.
7. Evaluate generalization using standard stratified cross-validation and template-group holdout.
8. Save reusable models, experimental results, and visualizations.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| NumPy | Numerical computations |
| Pandas | Data manipulation and CSV processing |
| Scikit-learn | Feature engineering, dimensionality reduction, classification, and evaluation |
| Matplotlib | Data visualization |
| Joblib | Model serialization |

### Machine Learning Techniques

**TF-IDF (Term Frequency–Inverse Document Frequency)**

Converts financial headlines into numerical feature vectors based on the relative importance of words and two-word phrases.

**Truncated SVD**

Reduces high-dimensional TF-IDF representations into lower-dimensional numerical embeddings. The default embedding size is 40 dimensions.

**t-SNE**

Projects embeddings into two dimensions for visualization. It helps examine local grouping patterns among the financial headline categories.

**Logistic Regression**

Trains a supervised classifier using the TF-IDF and SVD features to predict one of the four headline categories.

**Cross-Validation**

Evaluates classification performance with two strategies:

- Stratified five-fold cross-validation using headline-level splits.
- Stratified five-fold cross-validation with template-group separation to evaluate generalization to unseen headline wording patterns.

## Dataset

The dataset is generated programmatically within `tsne_models.py`.

It consists of synthetic headlines based on financial news templates combined with company and financial institution names, along with additional manually specified example headlines.

The dataset contains the following fields:

| Column | Description |
|---|---|
| `headline` | Financial news headline |
| `label` | News classification category |
| `ticker` | Company name associated with the generated headline |
| `source` | Dataset source indicator |
| `template_group` | Identifier grouping headlines generated from the same template |

Duplicate headline strings are removed before model training.

**Note:** This project uses synthetic financial headlines rather than a live financial news API or an independently annotated real-world news dataset.

## Project Structure

```text
Combinatorial Modeling/
│
├── tsne_models.py
├── README.md
│
├── tsne_news_embeddings.png
├── tsne_predictive_power.png
│
└── models/
    ├── news_corpus.csv
    ├── news_embedding_svd.joblib
    ├── news_label_encoder.joblib
    ├── news_classifier.joblib
    ├── tsne_n60_embeddings.csv
    ├── tsne_n120_embeddings.csv
    ├── tsne_n180_embeddings.csv
    ├── tsne_n240_embeddings.csv
    └── tsne_metrics.json
```

The models directory and output files are automatically generated when the script executes successfully.

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/jayraval0/comp560project.git
cd comp560project
```

### 2. Create a Python virtual environment

Python 3.10 or later is recommended. Use a recent, compatible version of scikit-learn that supports `TSNE(max_iter=...)`.

```bash
python3 -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

For Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install the required libraries

```bash
python -m pip install numpy pandas matplotlib scikit-learn joblib
```

### 4. Execute the program

```bash
python tsne_models.py
```

The program automatically generates the financial headline dataset, trains the models, evaluates classification performance, and saves the visualizations and results.

No API keys, external financial news downloads, or manually supplied datasets are required.

## Machine Learning Pipeline

The project follows this workflow:

```text
Synthetic Financial Headlines
            |
            v
     TF-IDF Vectorization
            |
            v
       Truncated SVD
            |
            +---------------------+
            |                     |
            v                     v
      t-SNE (2D)          Logistic Regression
            |                     |
            v                     v
      Scatter Plots        Category Predictions
                                  |
                                  v
                         Cross-Validation
                                  |
                                  v
                       Keyword Baseline Comparison
```

t-SNE is used for visualization only. Classification accuracy is calculated from a model trained on TF-IDF and SVD features, not on t-SNE coordinates.

## Experiments

Four sample sizes are evaluated:

| Experiment | Headline Sample Size |
|---|---|
| 1 | 60 |
| 2 | 120 |
| 3 | 180 |
| 4 | 240 |

Each sample is selected using class-balanced stratified sampling.

For each experiment, the program:

1. Samples the required number of headlines.
2. Generates TF-IDF and SVD embeddings.
3. Computes two-dimensional t-SNE coordinates.
4. Evaluates logistic regression using cross-validation.
5. Evaluates the keyword baseline on the same validation folds.
6. Saves the coordinates and classification metrics.

## Generated Visualizations

### 1. t-SNE Headline Embeddings

**File:** `tsne_news_embeddings.png`

Creates a 2×2 panel of scatter plots using the four sample sizes.

Each point represents a financial headline, colored according to its assigned category.

The graphs are intended to illustrate how headline embeddings are arranged in two-dimensional space.

### 2. Classification Accuracy Comparison

**File:** `tsne_predictive_power.png`

Compares average cross-validation accuracy for:

- Logistic regression with headline-level splits.
- Keyword baseline with headline-level splits.
- Logistic regression with template-group holdout.
- Keyword baseline with template-group holdout.

The horizontal axis represents sample size, and the vertical axis represents mean validation accuracy.

## Saved Models and Results

The script saves three reusable model objects:

- `news_embedding_svd.joblib`: Fitted TF-IDF and Truncated SVD pipeline.
- `news_classifier.joblib`: Fitted logistic regression classifier.
- `news_label_encoder.joblib`: Encoder mapping classification labels to integers.

It also saves:

- `news_corpus.csv`: Synthetic headline dataset.
- `tsne_n*_embeddings.csv`: Two-dimensional coordinates and headline labels for each sample size.
- `tsne_metrics.json`: Cross-validation metrics, dataset information, and generated model paths.

### Example: Classifying a New Headline

After running the training script, the saved models can be loaded to classify a new financial news headline:

```python
import joblib
from pathlib import Path

model_dir = Path("models")

embedding_model = joblib.load(
    model_dir / "news_embedding_svd.joblib"
)
classifier = joblib.load(
    model_dir / "news_classifier.joblib"
)
encoder = joblib.load(
    model_dir / "news_label_encoder.joblib"
)

headline = [
    "Analysts upgrade Nvidia following strong earnings growth"
]

features = embedding_model.transform(headline)
prediction = classifier.predict(features)
label = encoder.inverse_transform(prediction)

print("Predicted category:", label[0])
```

Only load Joblib model files from trusted sources.

## Author

Jay Raval

GitHub: https://github.com/jayraval0

## Repository

https://github.com/jayraval0/comp560project
