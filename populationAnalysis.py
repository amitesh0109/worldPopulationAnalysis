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

# Custom CSS for better styling
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3 {
        color: #1e3a8a;
    }
    .stAlert > div {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
    .stButton>button {
        background-color: #3b82f6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

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

def create_population_chart(df):
    top_10_countries = df.nlargest(10, 'Population')
    fig = px.bar(top_10_countries, x='Country', y='Population', 
                 title='Top 10 Countries by Population',
                 labels={'Population': 'Population', 'Country': 'Country'},
                 color='Continent',
                 hover_data=['Percentage'])
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_population_distribution(df):
    fig = px.histogram(df, x='Population', nbins=50,
                       title='Distribution of Country Populations',
                       labels={'Population': 'Population'},
                       color='Continent')
    fig.update_layout(bargap=0.1)
    return fig

def create_continent_pie_chart(df):
    continent_data = df.groupby('Continent')['Population'].sum().reset_index()
    fig = px.pie(continent_data, values='Population', names='Continent', 
                 title='Population Distribution by Continent')
    return fig

def train_population_predictor(df):
    X = df[['Percentage']].values
    y = df['Population'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

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
        st.write(f"Percentage of World: {most_populous['Percentage']:.2%}")
    
    with col2:
        st.subheader("Least Populous Country")
        least_populous = stats['least_populous'].iloc[0]
        st.write(f"**{least_populous['Country']}**")
        st.write(f"Population: {least_populous['Population']:,}")
        st.write(f"Percentage of World: {least_populous['Percentage']:.2%}")
    
    st.markdown("---")
    
    st.subheader("Population Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Top 10 Countries", "Population Distribution", "Continents"])
    
    with tab1:
        st.plotly_chart(create_population_chart(df), use_container_width=True)
    
    with tab2:
        st.plotly_chart(create_population_distribution(df), use_container_width=True)
    
    with tab3:
        st.plotly_chart(create_continent_pie_chart(df), use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("Country Comparison")
    col1, col2 = st.columns(2)
    with col1:
        country1 = st.selectbox("Select first country", df['Country'].tolist(), key='country1')
    with col2:
        country2 = st.selectbox("Select second country", df['Country'].tolist(), index=1, key='country2')
    
    if country1 and country2:
        data1 = df[df['Country'] == country1].iloc[0]
        data2 = df[df['Country'] == country2].iloc[0]
        
        fig = go.Figure(data=[
            go.Bar(name=country1, x=['Population'], y=[data1['Population']]),
            go.Bar(name=country2, x=['Population'], y=[data2['Population']])
        ])
        fig.update_layout(title_text=f"Population Comparison: {country1} vs {country2}")
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        col1.metric(country1, f"{data1['Population']:,}", f"{data1['Percentage']:.2%} of world")
        col2.metric(country2, f"{data2['Population']:,}", f"{data2['Percentage']:.2%} of world")
    
    st.markdown("---")
    
    st.subheader("Population Predictor")
    st.write("Predict a country's population based on its percentage of world population.")
    
    model = train_population_predictor(df)
    
    user_input = st.number_input("Enter a percentage of world population (e.g., 0.5 for 0.5%)", min_value=0.0, max_value=100.0, value=1.0, step=0.1)
    
    if st.button("Predict Population"):
        predicted_population = model.predict([[user_input / 100]])[0]
        st.metric("Predicted Population", f"{predicted_population:,.0f}")
    
    st.markdown("---")
    
    st.subheader("Raw Data")
    search_term = st.text_input("Search for a country")
    if search_term:
        filtered_df = df[df['Country'].str.contains(search_term, case=False)]
    else:
        filtered_df = df
    
    # Display only relevant columns and remove index
    st.dataframe(filtered_df[['Country', 'Population', 'Percentage', 'Continent']].reset_index(drop=True).style.format({
        'Population': '{:,.0f}',
        'Percentage': '{:.2%}'
    }))

if __name__ == "__main__":
    main()
