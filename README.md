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

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone git@github.com:nebiyu-ethio/tellco-data-insights.git
   ```
2. Navigate to the project directory:
   ```bash
   cd tellco-data-insights
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Once the environment is set up and the packages are installed, run the project locally:

1. **Start the Streamlit Dashboard**  
   Navigate to the `app` directory in the `dashboard-dev`and start the Streamlit dashboard:
   ```bash
   cd app
   streamlit run main.py
   ```

   This will launch the dashboard, providing an interactive interface for exploring the analyzed data.

### Deployed Dashboard URL

The deployed version of the Streamlit dashboard can be accessed at: [TELLCO DATA INSIGHTS](https://tellco-data-insights.streamlit.app/).

## License

This project is licensed under the MIT License.
