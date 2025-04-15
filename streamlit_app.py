import streamlit as st
import pandas as pd
import json
import os
import tempfile
from datetime import datetime, timedelta
import pydeck as pdk
import httpx
import re
import asyncio
import numpy as np
from typing import List, Dict, Any, Tuple
import requests
import json
import pandas as pd
import time
import uuid
from typing import Tuple, Optional, Dict, List, Any
from mapbox_util import forward_geocode


def parse_lat_lon(s):
    pattern = r'[-+]?\d*\.\d+|\d+'
    matches = re.findall(pattern, s)
    if len(matches) == 2:
        return float(matches[0]), float(matches[1])
    return None


def calculate_bounding_box(lat: float, lon: float, radius_miles: float) -> Tuple[float, float, float, float]:
    """
    Calculate a bounding box given a center point and radius in miles.
    Returns (minx, miny, maxx, maxy) in degrees.
    """
    # Earth's radius in miles
    earth_radius_miles = 3963.2

    # Convert radius from miles to degrees for latitude
    # 1 degree of latitude is approximately 69.1 miles
    lat_degrees = radius_miles / 69.1

    # Convert radius from miles to degrees for longitude
    # Longitude degrees vary based on latitude
    lon_degrees = radius_miles / (69.1 * np.cos(np.radians(lat)))

    # Calculate the bounding box
    minx = lon - lon_degrees  # min longitude
    maxx = lon + lon_degrees  # max longitude
    miny = lat - lat_degrees  # min latitude
    maxy = lat + lat_degrees  # max latitude

    return (minx, miny, maxx, maxy)


# FAA ArcGIS FeatureServer Query API URL
FAA_UAS_FacilityMap_Data_url_template = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/"
    "FAA_UAS_FacilityMap_Data/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&"
    "geometry=%7B%22xmin%22%3A{}%2C%22ymin%22%3A{}%2C%22xmax%22%3A{}%2C%22ymax%22%3A{}%2C%22spatialReference%22%3A%7B%22wkid%22%3A4326%7D%7D"
    "&geometryType=esriGeometryEnvelope&inSR=4326"
    "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
)

Prohibited_Areas_url_template = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/"
    "Prohibited_Areas/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&"
    "geometry=%7B%22xmin%22%3A{}%2C%22ymin%22%3A{}%2C%22xmax%22%3A{}%2C%22ymax%22%3A{}%2C%22spatialReference%22%3A%7B%22wkid%22%3A4326%7D%7D"
    "&geometryType=esriGeometryEnvelope&inSR=4326"
    "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
)

Recreational_Flyer_Fixed_Sites_url_template = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/"
    "Recreational_Flyer_Fixed_Sites/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&"
    "geometry=%7B%22xmin%22%3A{}%2C%22ymin%22%3A{}%2C%22xmax%22%3A{}%2C%22ymax%22%3A{}%2C%22spatialReference%22%3A%7B%22wkid%22%3A4326%7D%7D"
    "&geometryType=esriGeometryEnvelope&inSR=4326"
    "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
)

Part_Time_National_Security_UAS_Flight_Restrictions_url_template = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/"
    "Part_Time_National_Security_UAS_Flight_Restrictions/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&"
    "geometry=%7B%22xmin%22%3A{}%2C%22ymin%22%3A{}%2C%22xmax%22%3A{}%2C%22ymax%22%3A{}%2C%22spatialReference%22%3A%7B%22wkid%22%3A4326%7D%7D"
    "&geometryType=esriGeometryEnvelope&inSR=4326"
    "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
)

FAA_Recognized_Identification_Areas_url_template = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/"
    "FAA_Recognized_Identification_Areas/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&"
    "geometry=%7B%22xmin%22%3A{}%2C%22ymin%22%3A{}%2C%22xmax%22%3A{}%2C%22ymax%22%3A{}%2C%22spatialReference%22%3A%7B%22wkid%22%3A4326%7D%7D"
    "&geometryType=esriGeometryEnvelope&inSR=4326"
    "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
)

DoD_Mar_13_url_template = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/"
    "DoD_Mar_13/FeatureServer/0/query?"
    "where=1%3D1&outFields=*&"
    "geometry=%7B%22xmin%22%3A{}%2C%22ymin%22%3A{}%2C%22xmax%22%3A{}%2C%22ymax%22%3A{}%2C%22spatialReference%22%3A%7B%22wkid%22%3A4326%7D%7D"
    "&geometryType=esriGeometryEnvelope&inSR=4326"
    "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
)


FAA_UAS_FacilityMap_0ft_url_template = (
    "https://services6.arcgis.com/ssFJjBXIUyZDrSYZ/arcgis/rest/services/"
    "FAA_UAS_FacilityMap_Data/FeatureServer/0/query?"
    "where=CEILING%3D0&outFields=*&"
    "geometry=%7B%22xmin%22%3A{}%2C%22ymin%22%3A{}%2C%22xmax%22%3A{}%2C%22ymax%22%3A{}%2C%22spatialReference%22%3A%7B%22wkid%22%3A4326%7D%7D"
    "&geometryType=esriGeometryEnvelope&inSR=4326"
    "&spatialRel=esriSpatialRelIntersects&outSR=4326&f=json"
)


def validate_bbox(minx: float, miny: float, maxx: float, maxy: float) -> bool:
    """Validate bounding box coordinates."""
    if not all(isinstance(x, (int, float)) for x in [minx, miny, maxx, maxy]):
        print("Error: All bounding box coordinates must be numbers")
        return False
    if minx >= maxx:
        print("Error: minx must be less than maxx")
        return False
    if miny >= maxy:
        print("Error: miny must be less than maxy")
        return False
    if not (-180 <= minx <= 180 and -180 <= maxx <= 180):
        print("Error: longitude values must be between -180 and 180")
        return False
    if not (-90 <= miny <= 90 and -90 <= maxy <= 90):
        print("Error: latitude values must be between -90 and 90")
        return False
    return True


def save_to_geojson(processed_feature, output_file, append=False):
    """
    Save a single GeoJSON feature to a line-delimited GeoJSON file.
    Each feature is written as a separate line.
    """
    mode = "a" if append else "w"
    with open(output_file, mode) as f:
        json.dump(processed_feature, f)
        f.write("\n")

    if not append:
        print(f"Started new GeoJSON output file: {output_file}")


def get_faa_data_single_layer(
    url_template: str,
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    dataset_name: str,
    output_file: str,
    progress_callback=None,
) -> Tuple[bool, int]:
    """
    Fetch FAA data for a given bounding box and dataset.
    Writes features to a standard GeoJSON file.
    Returns (success_status, feature_count)
    """
    print(f"\nRequesting {dataset_name} data...")
    # Validate bounding box
    if not validate_bbox(minx, miny, maxx, maxy):
        return False, 0
    if isinstance(url_template, tuple):
        url_template = "".join(url_template)

    # Initialize variables for pagination
    feature_count = 0
    result_offset = 0
    result_record_count = 1000  # Number of records to fetch per request
    more_records = True

    # Initialize the GeoJSON structure
    geojson_data: Dict[str, Any] = {
        "type": "FeatureCollection", "features": []}

    while more_records:
        pagination_params = (
            f"&resultOffset={result_offset}&resultRecordCount={result_record_count}"
        )
        url = url_template.format(minx, miny, maxx, maxy) + pagination_params

        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            features = data.get("features", [])

            if not features and result_offset == 0:
                print("No features found in response")
                return False, 0

            # Process features and add to collection
            for feature in features:
                geom = feature.get("geometry")
                if geom and geom.get("rings"):
                    processed_feature = {
                        "type": "Feature",
                        "geometry": {"type": "Polygon", "coordinates": geom["rings"]},
                        "properties": feature.get("attributes", {}),
                    }
                    geojson_data["features"].append(processed_feature)
                    feature_count += 1
                elif geom and all(k in geom for k in ["x", "y"]):
                    processed_feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [geom["x"], geom["y"]],
                        },
                        "properties": feature.get("attributes", {}),
                    }
                    geojson_data["features"].append(processed_feature)
                    feature_count += 1

            # Check if we've received fewer records than requested, indicating we've reached the end
            if len(features) < result_record_count:
                more_records = False
            else:
                result_offset += result_record_count
                print(f"  Processed {feature_count} features so far...")
                # Update progress if callback is provided
                if progress_callback:
                    progress_callback(dataset_name, feature_count)
                # Add a small delay between pagination requests
                time.sleep(0.5)

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {str(e)}")
            if "response" in locals() and hasattr(response, "text"):
                print("Response content:", response.text)
            return False, 0

    # Write the complete GeoJSON to file
    if feature_count > 0:
        with open(output_file, "w") as f:
            json.dump(geojson_data, f)
        print(
            f"Successfully wrote a total of {feature_count} features to {output_file}"
        )
        return True, feature_count
    else:
        print(f"No features to write for {dataset_name}")
        return False, 0


def get_faa_data_all_layers(region_bbox: Tuple[float, float, float, float], session_id: str, progress_bar=None):
    """
    Fetch FAA drone airspace data for a given bounding box and save to separate GeoJSON files.

    :param region_bbox: (minx, miny, maxx, maxy) bounding box coordinates
    :param session_id: Unique session ID to avoid filename collisions
    :param progress_bar: Optional progress bar to update during processing
    :return: List of tuples (filename, file_content, display_name, feature_count)
    """
    # Define all data sources
    data_sources = [
        (FAA_UAS_FacilityMap_Data_url_template, "faa_uas_facility_map"),
        (Prohibited_Areas_url_template, "faa_prohibited_areas"),
        (Recreational_Flyer_Fixed_Sites_url_template, "faa_recreational_sites"),
        (
            Part_Time_National_Security_UAS_Flight_Restrictions_url_template,
            "faa_national_security",
        ),
        (FAA_Recognized_Identification_Areas_url_template, "faa_identification_areas"),
        (DoD_Mar_13_url_template, "faa_dod_mar_13"),
        (FAA_UAS_FacilityMap_0ft_url_template, "faa_uas_facility_map_0ft"),
    ]

    # Create a temporary directory for this session
    temp_dir = tempfile.mkdtemp()
    results = []
    feature_counts = {}
    total_sources = len(data_sources)

    # Setup progress tracking
    if progress_bar:
        progress_bar.progress(0, text="Starting download...")

    # Function to update progress
    def update_progress(dataset_name, count):
        feature_counts[dataset_name] = count
        if progress_bar:
            current_progress = len(feature_counts) / total_sources
            progress_text = f"Processing {dataset_name}: {count} features found"
            progress_bar.progress(current_progress, text=progress_text)

    # Fetch and save data from each source
    for idx, (url_template, name) in enumerate(data_sources):
        if progress_bar:
            progress_text = f"Downloading {name}... ({idx+1}/{total_sources})"
            progress_bar.progress((idx) / total_sources, text=progress_text)

        # Create a unique filename for this dataset in this session
        unique_filename = f"{name}_{session_id}.geojson"
        output_file = os.path.join(temp_dir, unique_filename)

        success, feature_count = get_faa_data_single_layer(
            url_template,
            *region_bbox,
            name,
            output_file,
            progress_callback=update_progress
        )

        if success and feature_count > 0:
            # Read the file content
            with open(output_file, 'rb') as f:
                file_content = f.read()

            # Add to results with display name that includes feature count
            display_name = f"{name.replace('_', ' ').title()}"
            results.append((unique_filename, file_content,
                           display_name, feature_count))
        elif not success:
            print(f"Failed to retrieve data for {name}")

        # Add a delay between requests to be considerate of the FAA's servers
        time.sleep(1)

    if progress_bar:
        progress_bar.progress(1.0, text="Download complete!")

    return results


def combine_geojson_files(files_dict, style_properties=None):
    """
    Combine multiple GeoJSON files into a single GeoJSON with optional style properties.

    Args:
        files_dict: Dictionary of {file_name: file_content_bytes}
        style_properties: Dictionary of simplestyle properties to add to features

    Returns:
        Combined GeoJSON as bytes
    """
    combined = {"type": "FeatureCollection", "features": []}

    for name, content_bytes in files_dict.items():
        try:
            content = json.loads(content_bytes.decode('utf-8'))
            if "features" in content and isinstance(content["features"], list):
                for feature in content["features"]:
                    # Add style properties if provided
                    if style_properties and isinstance(style_properties, dict):
                        if "properties" not in feature:
                            feature["properties"] = {}
                        feature["properties"].update(style_properties)
                    combined["features"].append(feature)
        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
            continue

    return json.dumps(combined).encode('utf-8')


def main():
    st.set_page_config(page_title="FAA Zones GeoJSON downloader",
                       page_icon="🛩️", layout="wide")

    st.title("🛩️ FAA Zones GeoJSON Downloader")

    # Initialize session state for storing downloaded files if not already present
    if 'downloaded_files' not in st.session_state:
        st.session_state.downloaded_files = []

    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

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

    # Calculate bounding box
    minx, miny, maxx, maxy = calculate_bounding_box(
        center_lat, center_lon, radius_mi)

    # For display, reorder as north, west, south, east
    north, west, south, east = maxy, minx, miny, maxx
    region_bbox = (west, south, east, north)

    st.markdown(f"""
        Bounding Box (north, west, south, east):
        
        ```python
        {north:.6f}, {west:.6f}, {south:.6f}, {east:.6f}
        ```
        """)

    # Add a button to download FAA data
    if st.button("Get data for this region from FAA ➡️"):
        # Create a progress bar
        progress_bar = st.progress(0, text="Preparing to download FAA data...")

        with st.spinner("Downloading FAA data for the region... This may take a minute or two."):
            # We need to convert from west, south, east, north to minx, miny, maxx, maxy
            minx, miny, maxx, maxy = west, south, east, north

            # Call the function to download all layers
            downloaded_files = get_faa_data_all_layers(
                (minx, miny, maxx, maxy),
                st.session_state.session_id,
                progress_bar
            )

            # Store the results in session state
            st.session_state.downloaded_files = downloaded_files

            st.success("FAA data downloaded successfully!")

    # Display download buttons for any available files
    if st.session_state.downloaded_files:
        with st.expander("Downloaded Files Details", expanded=False):
            for filename, file_content, display_name, feature_count in st.session_state.downloaded_files:
                # Create a download button for each file
                st.download_button(
                    label=f"📄 {display_name} ({feature_count} features)",
                    data=file_content,
                    file_name=filename,
                    mime="application/json"
                )

        # Add combined download button for the specific files
        st.subheader("Combined Download")

        # Find the requested files
        combined_files = {}
        for filename, file_content, display_name, feature_count in st.session_state.downloaded_files:
            if 'faa_uas_facility_map_0ft' in filename or 'faa_dod_mar_13' in filename:
                combined_files[filename] = file_content

        if len(combined_files) > 0:
            # Define simplestyle properties (red contours, red fill with 0.1 opacity)
            style_properties = {
                "stroke": "#FF0000",        # Red outline
                "stroke-width": 2,          # Line width
                "stroke-opacity": 1.0,      # Fully opaque outline
                "fill": "#FF0000",          # Red fill
                "fill-opacity": 0.1         # 10% opacity for fill
            }

            # Combine the files with styling
            combined_geojson = combine_geojson_files(
                combined_files, style_properties)

            # Create a download button for the combined file
            st.download_button(
                label="📥 Download FAA Restricted Zones",
                data=combined_geojson,
                file_name=f"combined_restricted_zones_{st.session_state.session_id}.geojson",
                mime="application/json",
                help="Combines FAA 0ft Facility Map and DoD Mar 13 zones into a single geojson with red styling"
            )
        else:
            st.info(
                "Download data first to generate combined restricted zones file.")


main()
