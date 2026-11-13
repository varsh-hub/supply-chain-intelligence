## Project Environment and Technology Setup

The Supply Chain Risk & Disruption Intelligence System was developed using Python as the primary programming language. Visual Studio Code (VS Code) was selected as the development environment because it provides an integrated terminal, source-code management, debugging capabilities, and support for Python development.

A Python virtual environment was created for the project to isolate the project dependencies from the system-wide Python installation. This helps maintain a consistent and reproducible development environment.

The following technologies and libraries were selected for the implementation:

| Technology / Library | Purpose |
|---|---|
| Python | Primary programming language |
| Pandas | Data manipulation and data preprocessing |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning and anomaly detection |
| Plotly | Interactive data visualization |
| Streamlit | Web application development |
| Joblib | Saving and loading trained machine-learning models |
| Jupyter Notebook | Exploratory data analysis and experimentation |
| Git | Version control |
| GitHub | Remote source-code repository |

The project dependencies are maintained in a `requirements.txt` file so that the required Python libraries can be installed consistently.

The project follows a modular architecture in which data processing, feature engineering, machine learning, anomaly detection, risk calculation, and recommendation generation are separated into individual Python modules. This improves maintainability and makes the system easier to test and extend.

## Initial Dataset Inspection

After placing the supply-chain dataset in the project's `data/raw/` directory, an initial data inspection was performed using Python and Pandas.

The purpose of this step was to understand the structure and quality of the raw dataset before applying any data cleaning or transformation techniques.

The inspection script examined:

- The number of rows and columns in the dataset.
- The names of all available columns.
- The first five records.
- The data type of each column.
- The number of missing values in each column.
- The number of duplicate records.

At this stage, no data cleaning or modification was performed. The raw dataset was kept unchanged so that its original structure could be analyzed before preprocessing.

The results of this inspection will be used to determine the appropriate data-cleaning techniques, feature-engineering methods, and machine-learning variables for the Supply Chain Risk & Disruption Intelligence System.

## Data Preprocessing and Feature Engineering

After completing the initial dataset inspection, a preprocessing pipeline was developed to prepare the raw supply-chain data for further analysis and machine-learning tasks.

The original raw dataset was preserved without modification. A separate processed dataset was generated at `data/processed/processed_data.csv`.

The preprocessing process included the following steps:

1. **Column Selection**  
   Relevant order, product, delivery, shipping, sales, and categorical variables were selected from the original 53 columns. Unnecessary and sensitive customer information was excluded from the machine-learning dataset.

2. **Character Encoding Handling**  
   The raw CSV contained characters that were not compatible with the default UTF-8 encoding. The dataset was therefore loaded using Latin-1 encoding to correctly interpret the existing CSV contents.

3. **Date Conversion**  
   The `order date (DateOrders)` and `shipping date (DateOrders)` columns were converted from text values into datetime format.

4. **Delivery Delay Feature**  
   A new feature called `delivery_delay_days` was created by subtracting the scheduled shipping duration from the actual shipping duration.

   ```
   delivery_delay_days =
   actual shipping days - scheduled shipping days
   ```

   Positive values indicate that shipping took longer than scheduled, while negative values indicate faster-than-scheduled shipping.

5. **Order-to-Shipping Duration**  
   A second feature called `order_to_shipping_days` was created by calculating the time between the order date and shipping date.

6. **Invalid Date Handling**  
   Records with invalid or missing values in the converted date columns were removed.

After preprocessing, the resulting dataset contained 180,519 records and 34 columns. The processed dataset contained no missing values and no duplicate records.

The processed dataset will be used as the input for exploratory analysis, feature engineering, anomaly detection, and machine-learning model development.

## Exploratory Data Analysis

After preprocessing the supply-chain dataset, exploratory data analysis (EDA) was performed to understand delivery-risk patterns and identify important relationships within the dataset.

The EDA was performed using Pandas for statistical analysis and Plotly for interactive visualizations.

### Dataset Overview

The processed dataset contains **180,519 records and 34 columns**.

The analysis found:

- 98,977 orders classified as late-delivery risk.
- 81,542 orders not classified as late-delivery risk.
- Approximately **54.83%** of the orders were associated with late-delivery risk.
- Approximately **45.17%** of the orders were not associated with late-delivery risk.

### Delivery Delay Analysis

A delivery-delay feature was previously created by comparing actual shipping duration with scheduled shipping duration.

The average delivery delay was approximately **0.57 days**, while the median delay was **1 day**.

The observed delay ranged from **-2 days to +4 days**. Negative values indicate shipments that were completed faster than the scheduled duration, while positive values indicate delays.

The distribution showed that a large number of orders experienced a delay of approximately one day.

### Shipping Mode Analysis

Late-delivery risk was analyzed across different shipping modes.

The results were:

| Shipping Mode | Late Delivery Risk |
|---|---:|
| First Class | 95.32% |
| Second Class | 76.63% |
| Same Day | 45.74% |
| Standard Class | 38.07% |

The analysis indicates a strong relationship between shipping mode and late-delivery risk in the dataset. First Class and Second Class shipments had substantially higher observed late-delivery risk than Same Day and Standard Class shipments.

This variable is therefore considered an important candidate for further investigation and machine-learning feature engineering.

### Market Analysis

Late-delivery risk was also compared across different markets.

The observed risk percentages were relatively similar:

- Europe: 55.21%
- Pacific Asia: 55.05%
- USCA: 54.80%
- Africa: 54.59%
- LATAM: 54.36%

This indicates that market-level differences in late-delivery risk are smaller than the differences observed between shipping modes.

### Product Category Analysis

Product categories were analyzed to identify categories with relatively high observed delivery risk.

The ten highest-risk categories included:

| Category | Late Delivery Risk |
|---|---:|
| Golf Bags & Carts | 68.85% |
| Lacrosse | 60.06% |
| Pet Supplies | 58.94% |
| Cameras | 58.11% |
| Strength Training | 57.66% |
| As Seen on TV! | 57.35% |
| Music | 57.14% |
| Accessories | 56.97% |
| Fitness Accessories | 56.96% |
| Books | 56.54% |

These results show that product category may also contain useful information for understanding delivery-risk patterns.

### EDA Visualizations

Three primary visualizations were generated:

1. **Late Delivery Risk Distribution**  
   This chart compares the number of orders classified as late-delivery risk and non-risk.

2. **Distribution of Delivery Delay**  
   This histogram shows the distribution of delivery delays across orders.

3. **Late Delivery Risk by Shipping Mode**  
   This chart compares the percentage of late-delivery risk across different shipping modes.

### Important Consideration for Machine Learning

The EDA identified several variables that describe the delivery outcome itself, such as actual shipping duration and calculated delivery delay.

These variables are useful for historical analysis, but they should not automatically be used as predictive-model inputs because they may only become available after the delivery process has already occurred.

Using such variables to predict late delivery could introduce **target leakage**, resulting in an unrealistically accurate machine-learning model.

Therefore, the predictive-model feature set will be designed around information that would realistically be available before or during the shipping process.

The EDA results will be used to guide feature selection, anomaly detection, predictive modelling, and the subsequent supply-chain risk engine.


## Predictive Feature Engineering and Target Design

After completing exploratory data analysis, the next stage was to prepare the dataset specifically for machine-learning prediction.

The objective of the predictive model is to estimate whether an order is likely to experience late delivery.

### Target Variable

The target variable selected for the classification model is:

`Late_delivery_risk`

The target contains two classes:

- `0` — No late-delivery risk
- `1` — Late-delivery risk

The dataset contains 180,519 records, with the following target distribution:

| Target | Records | Percentage |
|---|---:|---:|
| 0 | 81,542 | 45.17% |
| 1 | 98,977 | 54.83% |

The target distribution is reasonably balanced, so no major class-imbalance correction is required at this stage.

### Feature Selection

Potential predictive variables were selected from the processed dataset, including order, product, customer, geographical, financial, and shipping information.

Examples include:

- Shipping mode
- Scheduled shipping duration
- Order quantity
- Product price
- Sales
- Customer segment
- Market
- Country
- Region
- Product category
- Department
- Order type
- Product status

### Prevention of Data Leakage

Special attention was given to preventing target leakage.

Several columns describe the delivery outcome itself and therefore were excluded from the predictive feature set:

- `Days for shipping (real)`
- `delivery_delay_days`
- `Delivery Status`
- `shipping date (DateOrders)`
- `order_to_shipping_days`
- `Late_delivery_risk`

These variables may contain information that would only become available after the delivery process has occurred. Using them as predictive inputs could produce unrealistically high model performance.

Therefore, the model will be trained using information that is more representative of what would be available when assessing an order's delivery risk.

### Date Feature Engineering

The original order-date column was transformed into separate calendar features:

- `order_year`
- `order_month`
- `order_day_of_week`
- `order_hour`

The original `order date (DateOrders)` column was then removed from the model input.

The `Order Id` column was also excluded because it is an identifier rather than a meaningful business feature.

### Result

The resulting feature-engineering pipeline separates the dataset into:

- **X** — predictive input features
- **y** — late-delivery-risk target

The resulting feature set contains no missing values and is ready for categorical encoding, train/test splitting, and machine-learning model development.

This stage establishes a leakage-aware foundation for the predictive component of the Supply Chain Risk & Disruption Intelligence System.


## Machine Learning Baseline Model

After preparing the predictive features and target variable, a first machine-learning classification model was developed to predict late-delivery risk.

### Train/Test Split

The dataset was divided into training and testing subsets using an 80/20 split.

- Training records: **144,415**
- Testing records: **36,104**

Stratified sampling was used so that the distribution of the two target classes remained approximately consistent between the training and testing datasets.

### Categorical Feature Encoding

The dataset contains both numerical and categorical variables.

Categorical variables such as shipping mode, market, product category, customer segment, and geographical fields were converted into machine-readable numerical representations using one-hot encoding.

A preprocessing pipeline was used so that feature transformation and model training were handled consistently.

### Baseline Model

A **Random Forest Classifier** was selected as the first baseline machine-learning algorithm.

The model was configured with 100 decision trees and a fixed random state to make the experiment reproducible.

Random Forest was selected because it can capture nonlinear relationships between multiple supply-chain variables and can work effectively with a mixture of numerical and encoded categorical features.

### Baseline Model Performance

The model was evaluated using the unseen test dataset.

| Metric | Result |
|---|---:|
| Accuracy | 76.79% |
| Precision | 85.34% |
| Recall | 69.64% |
| F1 Score | 76.70% |

The model achieved an overall accuracy of approximately **76.79%**.

The precision for the late-delivery-risk class was approximately **85.34%**, indicating that when the model predicts an order as high late-delivery risk, the prediction is correct relatively often.

The recall for the late-delivery-risk class was approximately **69.64%**, meaning that the model successfully identified approximately 70% of the orders that actually belonged to the late-risk class.

The F1 score of **76.70%** provides a balanced measure of precision and recall.

### Baseline Model Interpretation

The results indicate that the selected supply-chain features contain meaningful information for predicting late-delivery risk.

However, this model is considered a **baseline model** rather than the final predictive model. Further analysis is required to understand feature importance, classification errors, and whether model performance can be improved through parameter tuning and alternative algorithms.

The next stage will therefore focus on model evaluation, feature importance, confusion-matrix analysis, and model improvement.