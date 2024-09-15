import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from io import StringIO
import re
import pycountry

# Set page config
st.set_page_config(layout="wide", page_title="World Population Insights", page_icon="🌍")

@st.cache_data
def scrape_wikipedia_data():
    url = "https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_population"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table', {'class': 'wikitable'})
        if table is None:
            st.error("Could not find the population table on the Wikipedia page.")
            return None
        df = pd.read_html(StringIO(str(table)))[0]
        return df
    except Exception as e:
        st.error(f"Error scraping data: {e}")
        return None

def get_continent(country_name):
    try:
        country = pycountry.countries.search_fuzzy(country_name)[0]
        alpha2 = country.alpha_2
        if alpha2 == 'AQ':
            return 'Antarctica'
        elif alpha2.startswith('A'):
            return 'Asia'
        elif alpha2.startswith('E'):
            return 'Europe'
        elif alpha2.startswith('F'):
            return 'Africa'
        elif alpha2 in ['AU', 'NZ']:
            return 'Oceania'
        elif alpha2.startswith('N') or alpha2.startswith('S'):
            return 'Americas'
        else:
            return 'Unknown'
    except:
        return 'Unknown'

def clean_data(df):
    if df is None:
        return None
    
    # Identify the correct column names
    country_col = [col for col in df.columns if 'country' in col.lower() or 'location' in col.lower()][0]
    population_col = [col for col in df.columns if 'population' in col.lower()][0]
    percentage_col = [col for col in df.columns if '%' in col.lower()][0]
    
    # Rename columns to standard names
    df = df.rename(columns={
        country_col: 'Country',
        population_col: 'Population',
        percentage_col: 'Percentage'
    })
    
    # Remove citations from the 'Country' column
    df['Country'] = df['Country'].apply(lambda x: re.sub(r'\[.*?\]', '', str(x)).strip())
    
    # Convert 'Population' to numeric
    df['Population'] = pd.to_numeric(df['Population'].astype(str).str.replace(',', ''), errors='coerce')
    
    # Convert 'Percentage' to numeric
    df['Percentage'] = pd.to_numeric(df['Percentage'].astype(str).str.rstrip('%'), errors='coerce') / 100.0
    
    # Drop rows with missing values in key columns
    df = df.dropna(subset=['Country', 'Population', 'Percentage'])
    
    # Add continent information
    df['Continent'] = df['Country'].apply(get_continent)
    
    # Remove 'Continent' column if all values are 'Unknown'
    if df['Continent'].nunique() == 1 and df['Continent'].iloc[0] == 'Unknown':
        df = df.drop('Continent', axis=1)
    
    # Exclude 'World' from the dataset
    df = df[df['Country'] != 'World']
    
    return df

def analyze_data(df):
    total_population = df['Population'].sum()
    average_population = df['Population'].mean()
    median_population = df['Population'].median()
    most_populous = df.nlargest(1, 'Population')
    least_populous = df.nsmallest(1, 'Population')
    
    return {
        'total_population': total_population,
        'average_population': average_population,
        'median_population': median_population,
        'most_populous': most_populous,
        'least_populous': least_populous
    }

def format_percentage(value):
    if value < 0.0001:  # For very small percentages
        return '< 0.0001%'
    else:
        return f'{value:.2%}'

def create_population_chart(df):
    top_10_countries = df.nlargest(10, 'Population')
    fig = px.bar(top_10_countries, x='Country', y='Population', 
                 title='Top 10 Countries by Population',
                 labels={'Population': 'Population', 'Country': 'Country'},
                 color='Continent' if 'Continent' in df.columns else None,
                 hover_data=['Percentage'])
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_population_distribution(df):
    fig = px.histogram(df, x='Population', nbins=50,
                       title='Distribution of Country Populations',
                       labels={'Population': 'Population'},
                       color='Continent' if 'Continent' in df.columns else None)
    fig.update_layout(bargap=0.1)
    return fig

def main():
    st.title('🌍 World Population Insights')
    st.write("Explore and analyze global population data")
    
    if st.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()
    
    df = scrape_wikipedia_data()
    df = clean_data(df)
    
    if df is None or df.empty:
        st.error("Failed to process the data. Please try again later.")
        return
    
    stats = analyze_data(df)
    
    # Display KPIs at the top
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total World Population", f"{stats['total_population']:,.0f}")
    col2.metric("Average Country Population", f"{stats['average_population']:,.0f}")
    col3.metric("Median Country Population", f"{stats['median_population']:,.0f}")
    col4.metric("Number of Countries", f"{len(df):,}")
    
    st.markdown("---")
    
    # Most and Least Populous Countries
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Most Populous Country")
        most_populous = stats['most_populous'].iloc[0]
        st.write(f"**{most_populous['Country']}**")
        st.write(f"Population: {most_populous['Population']:,}")
        st.write(f"Percentage of World: {format_percentage(most_populous['Percentage'])}")
    
    with col2:
        st.subheader("Least Populous Country")
        least_populous = stats['least_populous'].iloc[0]
        st.write(f"**{least_populous['Country']}**")
        st.write(f"Population: {least_populous['Population']:,}")
        st.write(f"Percentage of World: {format_percentage(least_populous['Percentage'])}")
    
    st.markdown("---")
    
    st.subheader("Population Visualizations")
    
    st.plotly_chart(create_population_chart(df), use_container_width=True)
    st.plotly_chart(create_population_distribution(df), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Raw Data")
    search_term = st.text_input("Search for a country")
    if search_term:
        filtered_df = df[df['Country'].str.contains(search_term, case=False)]
    else:
        filtered_df = df
    
    # Display only relevant columns and remove index
    columns_to_display = ['Country', 'Population', 'Percentage']
    if 'Continent' in filtered_df.columns:
        columns_to_display.append('Continent')
    
    st.dataframe(filtered_df[columns_to_display].reset_index(drop=True).style.format({
        'Population': '{:,.0f}',
        'Percentage': format_percentage
    }))

if __name__ == "__main__":
    main()
