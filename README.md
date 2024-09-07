# TellCo Telecom Data Analysis

## Project Overview

This project aims to analyze the telecommunications dataset of TellCo, a mobile service provider in the Republic of Pefkakia. The goal is to provide insights into user behavior, engagement, and satisfaction to inform potential business growth opportunities.

## Business Need

Our investor specializes in acquiring undervalued assets. This analysis will help determine whether TellCo is a viable investment by identifying opportunities to enhance profitability through customer-focused strategies.

## Data

- **Source**: Monthly aggregation of xDR records.
- **Attributes**: Include session details such as duration, data usage, and application types.
- **Database**: PostgreSQL schema provided.

## Learning Outcomes

- Understand the business context and client interests.
- Extract insights using various data analysis techniques.
- Build an interactive dashboard for data exploration.
- Develop skills in Python, SQL, and data visualization.

## Project Structure

```plaintext
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── .gitignore
├── requirements.txt
├── README.md
├── src/
│   ├── __init__.py
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── tests/
│   ├── __init__.py
└── scripts/
    ├── __init__.py
    └── README.md
```

## Key Tasks

### Task 1: User Overview Analysis

- Identify top handsets and manufacturers.
- Analyze user behavior on applications.
- Handle missing values and outliers.

### Task 2: User Engagement Analysis

- Track user engagement metrics.
- Use k-means clustering to segment customers.

### Task 3: Experience Analytics

- Analyze network parameters and device characteristics.
- Perform clustering on user experience metrics.

### Task 4: Satisfaction Analysis

- Assign engagement and experience scores.
- Build a regression model to predict satisfaction.

## Tools and Technologies

- **Python**: pandas, numpy, scikit-learn, matplotlib, seaborn
- **SQL**: PostgreSQL
- **Dashboard**: Streamlit
- **Containerization**: Docker
- **CI/CD**: GitHub Actions

## Tutorials Schedule

- **Wednesday**: Data Pre-processing
- **Thursday**: EDA and Data Modeling
- **Friday**: Dashboard Development
- **Monday & Tuesday**: Question Answering

## Deliverables

- Interim Submission: Insights from Task 1 and 2 in slides.
- Final Submission: Comprehensive report on all tasks.
- GitHub repository link.
- Deployed dashboard link.

## References

- Exploratory Data Analysis in Python
- Non-Graphical Univariate Analysis
- Bivariate Analysis Techniques
- Correlation and PCA Analysis

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:nebiyu-ethio/tellco-data-insights.git
   ```
2. Navigate to the project directory:
   ```bash
   cd tellco-telecom-data-analysis
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

- Run scripts for data analysis and model training.
- Use the Streamlit dashboard for interactive data exploration.

## License

This project is licensed under the MIT License.

---

For more details, please refer to the project documentation and tutorials provided.
```

Feel free to customize the content as per your specific project details and requirements.
