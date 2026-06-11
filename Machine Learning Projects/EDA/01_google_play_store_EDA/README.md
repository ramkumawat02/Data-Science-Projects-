# Google Play Store EDA

Exploratory Data Analysis project on the Google Play Store dataset. The notebook follows a CRISP-DM style workflow and focuses on understanding app categories, installs, reviews, ratings, pricing, and content rating patterns.

## Project Files

| File | Description |
| --- | --- |
| `01_EDA_Google_Play.ipynb` | Main Jupyter notebook containing data preparation, cleaning, feature conversion, visualizations, and insights. |
| `googleplaystore.csv` | Raw Google Play Store dataset with 10,841 rows and 13 columns. |
| `CRISP-DM-.png` | Image used in the notebook to explain the CRISP-DM process. |

## Dataset Columns

The dataset includes:

- `App`
- `Category`
- `Rating`
- `Reviews`
- `Size`
- `Installs`
- `Type`
- `Price`
- `Content Rating`
- `Genres`
- `Last Updated`
- `Current Ver`
- `Android Ver`

## Analysis Workflow

The notebook covers:

1. Data loading using Pandas.
2. Basic data profiling with `head`, `tail`, `sample`, `shape`, `describe`, and `info`.
3. Duplicate row detection and removal.
4. Data type correction for columns such as `Reviews`, `Size`, `Installs`, `Price`, and `Last Updated`.
5. Cleaning text-based numeric columns by removing symbols like `+`, `$`, and `,`.
6. Converting app size values from MB/KB into numeric format.
7. Extracting `day`, `month`, and `year` from the `Last Updated` column.
8. Removing duplicate apps while keeping the first record.
9. Separating categorical and numerical features.
10. Visual analysis using count plots, pie charts, bar plots, distribution plots, KDE plots, and box plots.

## Key Questions Explored

- Which app category is most common?
- What are the top 10 most popular categories?
- Which category has the highest total installs?
- What are the top installed apps by category and type?
- Which apps receive the most reviews?
- How are ratings, price, installs, and content ratings distributed?

## Main Insights

- `Family` is the most common app category in the dataset.
- Most apps are free rather than paid.
- App installs and reviews are highly skewed, with a small number of apps receiving very large user activity.
- Popular categories and highly installed categories are not always the same, so both count-based and install-based analysis are useful.
- Review volume highlights user engagement and can differ from install count.

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
```

## How to Run

1. Open the project folder.
2. Start Jupyter Notebook or JupyterLab.
3. Open `01_EDA_Google_Play.ipynb`.
4. Run the notebook cells from top to bottom.

Install the required Python libraries if needed:

```bash
pip install pandas numpy matplotlib seaborn jupyter
```

## Output

The final notebook provides a cleaned working dataframe and multiple visualizations that explain the structure of the Google Play Store dataset and highlight category, install, review, rating, and pricing trends.
