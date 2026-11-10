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