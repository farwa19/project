import streamlit as st
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import os
import textwrap
import json
import cohere
import folium
import streamlit as st
from streamlit_folium import st_folium
# Set up Cohere client
co = cohere.Client("jmJBKshr4Zjt9rPD8YEtuhapL2RpgN9z92SIluep") 
from geopy.geocoders import Nominatim
st.set_page_config(
    page_title="Restaurant Recommendation System",
    page_icon=":🍴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Your Streamlit app code follows here
import pandas as pd
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
def get_recommendations_ml(cat, address):
    geolocator = Nominatim(user_agent="my-geocoding-app")
    location = geolocator.geocode(address)
    user_location = np.array([[location.latitude, location.longitude]])


    df_filtered = df[df['Category'] == cat]


    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(df_filtered[['latitude', 'longitude']])

    distances, indices = knn.kneighbors(user_location)

    
    recommendations = df_filtered.iloc[indices[0]]
    recommendations_sorted = recommendations.sort_values(by="Distance", ascending=True)
    return recommendations_sorted
df = pd.read_csv("kl.csv", encoding='latin-1')
m = df['Category'].unique()
st.title("Restaurant Recommendation System")
st.sidebar.header("Choose a Restaurant Type")
category = st.sidebar.selectbox("Select Category", m)
kl = None
search = st.sidebar.selectbox("Choose Search Criteria", ["Select Restaurant with Highest Rating", "Select Restaurant Closer to My Location"])
if search == "Select Restaurant Closer to My Location":
    address = st.sidebar.text_input("Enter your address", "")
    kl = address
    if address:
        st.write(f"Searching for restaurants near {address}...")
hk = 1
ok = st.sidebar.button("Find Restaurants")
if ok:
    if search == "Select Restaurant Closer to My Location" :
        st.write(f'Category = {category}')
        st.write(f'Address ={address}')
        

        
import json

def format_relevant_rows(relevant_rows):
    
    formatted_rows = ""
    for row in relevant_rows:
        formatted_rows += json.dumps(row, indent=2) + "\n\n"
    return formatted_rows

def search_csv(query, df, column_to_search='Title'): # Change 'Question' to the actual column name in your CSV
  """
  Searches a DataFrame for rows containing the query.

  Args:
    query: The string to search for.
    df: The Pandas DataFrame to search.
    column_to_search: The column in the DataFrame to search within.

  Returns:
    A list of relevant rows as dictionaries.
  """
  # Simple keyword-based search (you can use more advanced techniques if needed)

if search == "Select Restaurant Closer to My Location" :
  sugeest = get_recommendations_ml(category, address)
  print(sugeest)
  hk = sugeest 
  st.dataframe(sugeest, hide_index=True)
# User's query
  query = "List 5 food places with their reviws from database"

  # Search the CSV
  # You can specify the column to search if it's not 'Title'

  # Construct a prompt for Cohere, including relevant info from the CSV
  print('test')


  relevant_rows = format_relevant_rows(hk)

  print(relevant_rows)
  print("nnnnnnnnnnn")
  def search_csv(query, df, column_to_search='Title'): # Change 'Question' to the actual column name in your CSV
    """
    Searches a DataFrame for rows containing the query.

    Args:
      query: The string to search for.
      df: The Pandas DataFrame to search.
      column_to_search: The column in the DataFrame to search within.

    Returns:
      A list of relevant rows as dictionaries.
    """
    # Simple keyword-based search (you can use more advanced techniques if needed)


  # User's query
  query = "List 5 food places with their reviws"

  # Search the CSV
  # You can specify the column to search if it's not 'Title'

  # Construct a prompt for Cohere, including relevant info from the CSV

  print("nnnnnnnnnnn")
  prompt = f"""
  Tell the user about these restruants and its number of reveiws and location from our database only. Dont forget to tell the user abot Title  of the restruant!

  User query: {query}


  Relevant information from our database: {hk}

  Please provide a concise and informative answer to the user's query based on the available information. Give information about all the restruants given in database
  """

  # Generate a response using Cohere
  response = co.generate(
      model='command',
      prompt=prompt,

      temperature=0.5,

  )

  st.write(response.generations[0].text)
  st.title("Visualization")
  for index, row in hk.iterrows():
      print(row)  # Print each row of the DataFrame
      # Create a figure and axis
      name = row.get("Title")
      st.write(name)
      location = [row.get("latitude"), row.get("longitude")]  # Islamabad, Pakistan

      # Create a Folium map centered around the location
      map = folium.Map(location=location, zoom_start=12)

        # Add a marker to the map
      folium.Marker(
            location,
            popup="Islamabad, Pakistan",
            icon=folium.Icon(icon="info-sign")
        ).add_to(map)

        # Display the Folium map in Streamlit
      st_folium(map, width=700, height=500)
else:
    kl = df[df["Review_Points"] > 4.0]
    jk = kl[kl["Category"] == category]
    df_sorted = jk.sort_values('Reviews', ascending=False)
    high = df_sorted.head(10)
    st.dataframe(df_sorted.head(10),hide_index=True)
    query = "List 5 food places with highest reviws and  location from database. Write down the answer. Dont give me table"

  
    print('test')



   
    print("nnnnnnnnnnn")
    def search_csv(query, df, column_to_search='Title'): # Change 'Question' to the actual column name in your CSV
      """
      Searches a DataFrame for rows containing the query.

      Args:
        query: The string to search for.
        df: The Pandas DataFrame to search.
        column_to_search: The column in the DataFrame to search within.

      Returns:
        A list of relevant rows as dictionaries.
      """
      # Simple keyword-based search (you can use more advanced techniques if needed)


    # User's query
    query = "List 5 food places with their reviws"

    # Search the CSV
    # You can specify the column to search if it's not 'Title'

    # Construct a prompt for Cohere, including relevant info from the CSV

    print("nnnnnnnnnnn")
    prompt = f"""
    Give answer of 2 or 3 sentences
    Tell the user about these restruants and its number of reveiws and location from our database only. Dont forget to tell the user abot Title  of the restruant!

    User query: {query}


    Relevant information from our database: {high}

    Please provide a concise and informative answer to the user's query based on the available information. Give information about all the restruants given in database
    """

    # Generate a response using Cohere
    response = co.generate(
        model='command',
        prompt=prompt,

        temperature=0,

    )

    st.write(response.generations[0].text)
    print(high)
    st.title("Visualization")
    for index, row in high.iterrows():
      print(row)  # Print each row of the DataFrame
      # Create a figure and axis
      name = row.get("Title")
      st.write(name)
      location = [row.get("latitude"), row.get("longitude")]  # Islamabad, Pakistan

      # Create a Folium map centered around the location
      map = folium.Map(location=location, zoom_start=12)

        # Add a marker to the map
      folium.Marker(
            location,
            popup="Islamabad, Pakistan",
            icon=folium.Icon(icon="info-sign")
        ).add_to(map)

        # Display the Folium map in Streamlit
      st_folium(map, width=700, height=500)
