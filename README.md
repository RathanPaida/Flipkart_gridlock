# Flipkart_gridlock
PROJECT OVERVIEW
The approach centers on a Robust K-Fold Ensemble combining LightGBM and CatBoost models. To ensure the model remains accurate even when evaluated against private, unseen datasets, we utilize a Dynamic Time Anchoring strategy that automatically identifies the most recent historical data to drive lag-based predictions.

APPROACH
🛠 Approach & MethodologyEnsemble Strategy: 

We use a 5-Fold Cross-Validation strategy to train two distinct models (LightGBM and CatBoost) on the truth data.

Blending: Final predictions are a weighted ensemble of the two models (60% LightGBM, 40% CatBoost), which leverages the gradient-boosting strengths of both libraries to minimize RMSE/maximize R2.

Time Series Integrity: Instead of hardcoding specific days, the pipeline uses .max() to dynamically anchor features like lag_gh_hour to the latest available day in the training set, ensuring the model is always looking at the "freshest" data for momentum features.

Categorical Handling: The pipeline uses native categorical processing. LightGBM utilizes label-encoded integers (for GPU speed), while CatBoost processes strings natively.

FEATURE ENGINEERING
🧩 Feature Engineering Details

The feature set is engineered to capture both spatial and temporal dynamics:
Temporal:
Cyclic time features (hour_sin, hour_cos) and rush_hour indicators help capture daily traffic rhythm.

Spatial:
Geohash Decoding:
Derived latitude, longitude, and dist_from_center to capture spatial density.
 
Prefixes: Hierarchical geohash prefixes (geo_prefix4, geo_prefix5) to allow the model to learn localized neighborhood trends.

K-Means Clustering: Geographic clusters (n=20) group areas with similar traffic profiles.

Domain Interaction:
lane_lv_ratio: Captures the relationship between lane capacity and vehicle type.

Road_LV & Road_Weather: Combined interaction features that allow the model to distinguish between different road conditions/weather impact combinations.

Dynamic Lags: lag_gh_hour and expected_demand_adjusted use the previous day's traffic means to project growth and demand trends forward.

TOOLS AND DEPENDENCIES
🛠 Tools & Dependencies
This pipeline is optimized for GPU-accelerated environments (like Kaggle Notebooks).

Engines: LightGBM (GPU-enabled), CatBoost (GPU-enabled)
Data Science: Pandas, Numpy, Scikit-learn (Imputers, KFold).
Optimization: Optuna (for hyperparameter tuning).
Spatial Utils: pygeohash.


📂 Relevant Files
File Name                   Description
train.csv                   Input: Historical training data.
test.csv                    Input: Unseen test set.
submission.csv              Output: Final ensemble prediction file.


🚀 Execution Instructions
Install: 
Ensure all packages are installed: 
!pip install pandas numpy scikit-learn xgboost lightgbm catboost matplotlib     category_encoders pygeohash optuna.

Setup: Ensure you have your train.csv and test.csv in the current working directory.

Configure: Enable the GPU accelerator in your Kaggle/Colab notebook settings.

Run: Execute the pipeline cell-by-cell as provided in the notebook.
The pipeline will automatically generate the final submission file in the working directory.
