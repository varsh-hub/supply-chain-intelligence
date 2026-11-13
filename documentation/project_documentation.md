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