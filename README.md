# World Population Analysis Dashboard

## Overview
This project is an interactive web application that provides insights into global population data. It scrapes data from Wikipedia, processes it, and presents various visualizations and analyses through an intuitive dashboard.

## Features
- Real-time data scraping from Wikipedia
- Interactive visualizations including:
  - Top 10 countries by population
  - Population distribution
  - Population by continent
- Country comparison tool
- Population prediction based on world percentage
- Searchable raw data table

## Technologies Used
- Python
- Streamlit
- Pandas
- BeautifulSoup
- Plotly
- Scikit-learn

## Setup and Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Local Installation
1. Clone the repository:
   ```
   git clone https://github.com/yourusername/world-population-analysis.git
   cd world-population-analysis
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Run the Streamlit app:
   ```
   streamlit run populationAnalysis.py
   ```

5. Open your web browser and go to `http://localhost:8501` to view the app.

## Deployment
This app is designed to be deployed on Streamlit Cloud. To deploy:

1. Fork this repository to your GitHub account.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud).
3. Create a new app and select your forked repository.
4. Set the main file path to `populationAnalysis.py`.
5. Deploy the app.

## Usage
- Use the "Refresh Data" button to fetch the latest data from Wikipedia.
- Explore different visualizations in the "Population Visualizations" section.
- Compare two countries using the dropdown menus in the "Country Comparison" section.
- Predict population based on world percentage in the "Population Predictor" section.
- Search and explore raw data in the "Raw Data" section.

## Contributing
Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Data source: Wikipedia's "List of countries and dependencies by population"
- Thanks to the Streamlit team for their excellent framework
