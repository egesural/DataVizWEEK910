import os
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# PAGE SETUP & PROJECT EXPLANATION
# ---------------------------------------------------------
st.set_page_config(page_title="Tesla Supercharger Spatial Analysis", layout="wide")

st.title("U.S. Tesla Supercharger Network")

st.markdown("""
**About This Project**  
This application performs a spatial analysis of Tesla Supercharger locations across the United States. 
Looking at raw station counts can be misleading because states with larger populations naturally have more infrastructure. 
To reveal the true density of the charging network, this project uses two distinct data joining techniques:

1. **Spatial Join:** Maps the exact GPS coordinates of each Supercharger station to its corresponding U.S. state boundary.
2. **Attribute Join & Normalization:** Connects the resulting state station counts to demographic population data, allowing us to calculate and map the **population-normalized rate** (Superchargers per 1 Million residents).
""")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------
@st.cache_data
def load_all_data():
    # 1. Load Supercharger CSV
    csv_path = os.path.join(BASE_DIR, "superchargers_us.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(BASE_DIR, "superchargers_us_2.csv")
    points_df = pd.read_csv(csv_path)
    points_df["kw_clean"] = points_df["kw"].fillna(250.0)

    # 2. Load State Boundaries GeoJSON
    geojson_path = os.path.join(BASE_DIR, "states.geojson")
    if not os.path.exists(geojson_path):
        geojson_path = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    
    states_gdf = gpd.read_file(geojson_path)
    states_gdf = states_gdf.set_crs("EPSG:4326", allow_override=True)

    # Standardize FIPS codes 
    if "STATEFP" in states_gdf.columns:
        states_gdf["fips"] = states_gdf["STATEFP"].astype(str).str.zfill(2)
    else:
        states_gdf["fips"] = states_gdf["id"].astype(str).str.zfill(2)

    # Standardize state names
    if "NAME" in states_gdf.columns:
        states_gdf["state_name"] = states_gdf["NAME"]
    else:
        states_gdf["state_name"] = states_gdf["name"]

    abbr = {
        "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
        "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "District of Columbia": "DC",
        "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL",
        "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA",
        "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN",
        "Mississippi": "MS", "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
        "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
        "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
        "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD",
        "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA",
        "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY", "Puerto Rico": "PR"
    }
    states_gdf["state_abbr"] = states_gdf["state_name"].map(abbr)

    # 3. Load Population Stats
    stats_path = os.path.join(BASE_DIR, "state_stats.csv")
    if os.path.exists(stats_path):
        stats_df = pd.read_csv(stats_path)
        stats_df.columns = [c.lower().strip() for c in stats_df.columns]
        if "fips" in stats_df.columns:
            stats_df["fips"] = stats_df["fips"].astype(str).str.replace(".0", "", regex=False).str.zfill(2)
    else:
        # Fallback dictionary if file is missing locally
        pop_dict = {
            "01": 5024279, "02": 733391, "04": 7151502, "05": 3011524, "06": 39538223,
            "08": 5773714, "09": 3605944, "10": 989948, "11": 689545, "12": 21538187,
            "13": 10711908, "15": 1455271, "16": 1839106, "17": 12812508, "18": 6785528,
            "19": 3190369, "20": 2937880, "21": 4505836, "22": 4657757, "23": 1362359,
            "24": 6177224, "25": 7029917, "26": 10077331, "27": 5706494, "28": 2961279,
            "29": 6154913, "30": 1084225, "31": 1961504, "32": 3104614, "33": 1377529,
            "34": 9288994, "35": 2117522, "36": 20201249, "37": 10439388, "38": 779094,
            "39": 11799448, "40": 3959353, "41": 4237256, "42": 13002700, "44": 1097379,
            "45": 5118425, "46": 886667, "47": 6910840, "48": 29145505, "49": 3271616,
            "50": 643077, "51": 8631393, "53": 7705281, "54": 1793716, "55": 5893718,
            "56": 576851, "72": 3285874
        }
        stats_df = pd.DataFrame({"fips": list(pop_dict.keys()), "population": list(pop_dict.values())})

    return points_df, states_gdf, stats_df

points_raw, states_gdf, stats_df = load_all_data()

# ---------------------------------------------------------
# SIDEBAR FILTERS (Shared controls)
# ---------------------------------------------------------
st.sidebar.header("Network Filters")
st.sidebar.write("Adjust these filters to update both the map and the charts simultaneously.")

min_stalls, max_stalls = int(points_raw["stalls"].min()), int(points_raw["stalls"].max())
stall_range = st.sidebar.slider("Station Size (Number of Stalls)", min_stalls, max_stalls, (min_stalls, max_stalls))
kw_filter = st.sidebar.slider("Minimum Charging Power (kW)", int(points_raw["kw_clean"].min()), int(points_raw["kw_clean"].max()), int(points_raw["kw_clean"].min()))

filtered_points = points_raw[
    (points_raw["stalls"] >= stall_range[0]) & 
    (points_raw["stalls"] <= stall_range[1]) & 
    (points_raw["kw_clean"] >= kw_filter)
].copy()

# ---------------------------------------------------------
# SPATIAL & ATTRIBUTE JOINS
# ---------------------------------------------------------
# 1. Spatial Join
points_gdf = gpd.GeoDataFrame(
    filtered_points, 
    geometry=gpd.points_from_xy(filtered_points["lon"], filtered_points["lat"]), 
    crs="EPSG:4326"
)
sjoined = gpd.sjoin(points_gdf, states_gdf[["state_name", "fips", "state_abbr", "geometry"]], how="left", predicate="within")

unmatched = sjoined[sjoined["index_right"].isna()]
matched = sjoined[sjoined["index_right"].notna()]

state_counts = matched.groupby(["fips", "state_name", "state_abbr"]).agg(
    station_count=("name", "count")
).reset_index()

base_states = pd.DataFrame(states_gdf[["fips", "state_name", "state_abbr"]])
state_summary = pd.merge(base_states, state_counts, on=["fips", "state_name", "state_abbr"], how="left")
state_summary["station_count"] = state_summary["station_count"].fillna(0)

# 2. Attribute Join
if "fips" in stats_df.columns:
    joined = pd.merge(state_summary, stats_df, on="fips", how="left")
else:
    joined = pd.merge(state_summary, stats_df, left_on="state_name", right_on="name", how="left")

# Find the population column automatically
pop_col = [c for c in joined.columns if "pop" in c.lower()][0]

# Calculate Normalized Rate
joined["stations_per_1m"] = (joined["station_count"] / joined[pop_col]) * 1_000_000

# Calculate 5-class Quantile scheme for map coloring
if len(joined["stations_per_1m"].dropna().unique()) >= 5:
    joined["quantile_class"] = pd.qcut(joined["stations_per_1m"].rank(method="first"), q=5, labels=["Q1 (Lowest 20%)", "Q2", "Q3", "Q4", "Q5 (Highest 20%)"])
else:
    joined["quantile_class"] = "All"

# ---------------------------------------------------------
# MAIN TABS
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["Geospatial Analysis & Methodology", "Interactive Data Exploration"])

with tab1:
    st.subheader("Population-Normalized Deployment")
    
    st.markdown("""
    This map utilizes two distinct geographic layers:
    * **Base Layer (Choropleth):** State polygons are shaded based on the normalized rate of Superchargers per 1 million residents. 
    * **Overlay Layer (Graduated Symbols):** Individual filtered station locations are plotted on top. The marker sizes represent the total number of stalls, and the marker colors represent power output (kW).
    """)

    fig1 = px.choropleth(
        joined, locations="state_abbr", locationmode="USA-states", color="stations_per_1m",
        hover_name="state_name", scope="usa", projection="albers usa", color_continuous_scale="Blues"
    )
    fig1.add_trace(go.Scattergeo(
        lon=filtered_points["lon"], lat=filtered_points["lat"], mode="markers",
        marker=dict(size=filtered_points["stalls"] / 2 + 3, color=filtered_points["kw_clean"], colorscale="YlOrRd"),
        text=filtered_points["name"], name="Superchargers"
    ))
    fig1.update_layout(height=550, margin=dict(l=0, r=0, b=0, t=30))
    st.plotly_chart(fig1, use_container_width=True)
   

    st.divider()
    st.subheader("Tracking Data Integrity")
    st.markdown("""
    When conducting a spatial join, some points may fail to match a bounding polygon. This typically occurs due to boundary generalizations (e.g., stations located right on a coastline or pier that fall just outside simplified state lines) or regional mismatches (e.g., stations located in U.S. Territories not included in the primary state boundary file). 
    
    The metrics below verify the successful matching rate for our dataset.
    """)
    col1, col2 = st.columns(2)
    col1.metric("Points Matched Successfully", len(matched))
    col2.metric("Unmatched Points (Failed Spatial Join)", len(unmatched))
    
    if len(unmatched) > 0:
        with st.expander("View Unmatched Stations"):
            st.dataframe(unmatched[["name", "city", "state", "lat", "lon"]])


with tab2:
    st.subheader("Drill Down by State")
    st.markdown("""
    Select a state directly on the map below. The histogram on the right will automatically update to reflect the distribution of Supercharger capabilities (Number of Stalls and kW Output) exclusively for the chosen state.
    """)
    
    c1, c2 = st.columns([1.2, 1])
    
    with c1:
        # 5-Class Quantile Map for discrete selection
        fig2 = px.choropleth(
            joined.dropna(subset=["quantile_class"]), locations="state_abbr", locationmode="USA-states",
            color="quantile_class", hover_name="state_name", scope="usa", projection="albers usa",
            category_orders={"quantile_class": ["Q1 (Lowest 20%)", "Q2", "Q3", "Q4", "Q5 (Highest 20%)"]},
            color_discrete_sequence=px.colors.sequential.Viridis,
            title="Click a State to Filter"
        )
        fig2.update_layout(clickmode="event+select", height=450, margin=dict(l=0, r=0, b=0, t=40))
        
        # Capture selection
        map_event = st.plotly_chart(fig2, on_select="rerun", key="map_select", use_container_width=True)

    # Process selection logic
    clicked = None
    if map_event and "selection" in map_event and len(map_event["selection"].get("points", [])) > 0:
        clicked = map_event["selection"]["points"][0].get("location")

    with c2:
        sub_df = filtered_points[filtered_points["state"] == clicked] if clicked else filtered_points
        current_title = f"Station Distribution: {clicked or 'All States'}"
        
        fig_hist = px.histogram(
            sub_df, x="stalls", color="kw_clean", 
            title=current_title,
            labels={"stalls": "Number of Stalls", "kw_clean": "Power Output (kW)"}
        )
        fig_hist.update_layout(height=450, margin=dict(l=0, r=0, b=0, t=40))
        st.plotly_chart(fig_hist, use_container_width=True)