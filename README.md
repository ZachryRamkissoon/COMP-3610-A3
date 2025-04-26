# COMP-3610-A3: Big Data Analytics Assignment 3

## Overview

This repository contains the code and resources for Assignment 3 of the COMP-3610 Big Data Analytics course. The project involves analyzing the McAuley-Lab/Amazon-Reviews-2023 dataset (~200GB, 34 categories) through a series of tasks: data acquisition, cleaning and preprocessing, exploratory data analysis (EDA), binary sentiment classification, recommender system development using Alternating Least Squares (ALS), and clustering using k-means. The tasks are implemented in Python using Jupyter notebooks, with libraries such as DuckDB, Pandas, Matplotlib, `implicit`, and `scikit-learn`.

The repository is organized by task, with each part containing the relevant scripts or notebooks. Due to the dataset’s size, we used strategies like sequential category processing and sampling to manage computational resources. 

## Repository Structure

- **Part_1/**: Data Acquisition (Task 1)
  - `download_amazon_reviews.py`: Script to download all 34 categories of the Amazon Reviews dataset using the `download_all_amazon_reviews` function.
  - **Purpose**: Downloads raw review and metadata files.

- **Part_2/**: Data Cleaning & Preprocessing (Task 2)
  - `merging.ipynb`: Initial notebook for merging review and metadata on `parent_asin` (exploratory, may contain redundant code).
  - `preprocessing.ipynb`: Main notebook for cleaning and preprocessing, including merging, handling missing values, removing duplicates, and adding derived columns (`review_length`, `year`).
  - **Purpose**: Processes raw data into cleaned Parquet files, which are used in subsequent tasks.

- **Part_3/**: Exploratory Data Analysis (EDA) (Task 3)
  - `EDA.ipynb`: Notebook for generating EDA plots (star rating histogram, top 10 categories, top 10 brands, time-based trend) and calculating the Pearson correlation between `review_length` and `rating`.
  - `plots/`: Directory containing the generated plots:
    - `star_rating_histogram.png`
    - `top_10_categories.png`
    - `top_10_brands.png`
    - `avg_rating_trend.png`
  - **Purpose**: Provides insights into the dataset’s distribution, trends, and correlations.

- **Part_4_&6/**: Binary Sentiment (Task 4) and Clustering (Task 6)
  - `adding_sentiment.ipynb`: Notebook for adding sentiment column.
  - **Purpose**: Preparing for logistic regression.
  - `logistic_regression_&_clustering.ipynb`: Notebook for binary sentiment classification using logistic regression and product segmentation using k-means clustering.
  - **Purpose**: Implements sentiment prediction (Positive/Negative based on ratings) and clusters products into 5 groups based on features like mean rating ,total reviews, brand and category.

- **Part_5/**: Recommender System (ALS) (Task 5)
  - `ALS.ipynb`: Notebook for building an ALS-based recommender system on a 0.01% sample of the data, including data setup, model training, evaluation (RMSE), and top-5 recommendations for 3 users.
  - **Purpose**: Demonstrates collaborative filtering using the `implicit` library.

### Running the Notebooks
- **Order of Execution**:
  1. **Part_1**: Run `download_amazon_reviews.py` to acquire the dataset.
  2. **Part_2**: Run `merging.ipynb` to merge the data. Run `preprocessing.ipynb` to clean and preprocess the data.
  3. **Part_3**: Run `EDA.ipynb` to perform EDA and generate plots.
  4. **Part_4_&6**: Run `adding_sentiment.ipynb` to add the sentiment column. Run `logistic_regression_&_cluster_with_summary.ipynb` for sentiment classification and clustering.
  5. **Part_5**: Run `ALS.ipynb` to build the ALS recommender system.

## Notes
- **Plots**: EDA plots are saved in `Part_3/plots/`.

## Team Members
- Alindo Goberdhan
- Jehlani Joseph
- Kenwyn Simon
- Zachry Ramkissoon