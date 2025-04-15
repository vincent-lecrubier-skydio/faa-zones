import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import pydeck as pdk
import httpx
import re
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple
from mapbox_util import forward_geocode


def parse_lat_lon(s):
    pattern = r'[-+]?\d*\.\d+|\d+'
    matches = re.findall(pattern, s)
    if len(matches) == 2:
        return float(matches[0]), float(matches[1])
    return None


def main():
    st.set_page_config(page_title="FAA Zones GeoJSON downloader",
                       page_icon="🛩️", layout="wide")

    st.title("🛩️ FAA Zones GeoJSON Downloader")

    if "location" not in st.session_state:
        if "location" in st.query_params:
            location_param = st.query_params.get("location")
            st.session_state["location"] = location_param
        else:
            st.session_state["location"] = "3000 Clearview way, San Mateo, CA"

    location = st.text_input(
        "Location Address or Coordinates (lat,lon)", key="location")

    coords = forward_geocode(location)

    center_lat = None
    center_lon = None
    if coords:
        center_lat = coords[1]
        center_lon = coords[0]
    else:
        lat_lon = parse_lat_lon(location)
        if lat_lon:
            center_lat, center_lon = lat_lon
    if center_lat is None or center_lon is None:
        st.error(
            "Invalid location. Please enter a valid address or coordinates (lat,lon)")
        del st.query_params["location"]
        return
    else:
        st.query_params["location"] = location

    st.markdown(f"""
        Location Coordinates:

        ```python
        {center_lat}, {center_lon}
        ```
        """)

    radius_mi = st.number_input(
        "Radius around location (mi)", value=30.0, step=0.1)
    
    


main()
