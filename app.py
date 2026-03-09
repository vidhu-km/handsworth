import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import MiniMap, Draw
from streamlit_folium import st_folium
import branca.colormap as cm
from shapely.geometry import Point, shape
from pyproj import Transformer
from shapely.ops import transform as shapely_transform

# ==========================================================
# Page configuration
# ==========================================================
st.set_page_config(layout="wide", page_title="Bakken Map Viewer", page_icon="🗺️")

TOOLTIP_STYLE = "font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);border:1px solid #333;border-radius:3px;"
NULL_STYLE = {"fillColor": "#ffffff", "fillOpacity": 0, "color": "#888", "weight": 0.25}

# ==========================================================
# Optimized Helpers
# ==========================================================

def safe_range(series):
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty: return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    return (lo, hi) if lo != hi else (lo - 1, lo + 1)

def endpoint_of_geom(geom):
    if geom is None or geom.is_empty: return None
    if geom.geom_type == "LineString": return Point(geom.coords[-1])
    if geom.geom_type == "MultiLineString": return Point(geom.geoms[-1].coords[-1])
    return geom if geom.geom_type == "Point" else None

def fmt_val(v):
    if pd.isna(v): return "—"
    return f"{v:,.0f}" if isinstance(v, (int, float)) and abs(v) > 100 else f"{v:.3f}"

# ==========================================================
# Load & Cache Data (The "Speed" Engine)
# ==========================================================
@st.cache_resource(show_spinner="Optimizing spatial layers...")
def load_data():
    # Load all spatial files
    files = {
        "lines": "lines.shp", "points": "points.shp", "grid": "ooipsectiongrid.shp",
        "units": "Bakken Units.shp", "land": "Bakken Land.shp", "handsworth": "Handsworth Units.shp"
    }
    gdfs = {k: gpd.read_file(v) for k, v in files.items() if v}
    
    well_df = pd.read_excel("wells.xlsx", sheet_name=0)
    section_df = pd.read_excel("wells.xlsx", sheet_name=1)

    # Standardize CRS and Simplify Geometries (Massive speed boost)
    for key in gdfs:
        if gdfs[key].crs is None: gdfs[key].set_crs(epsg=26913, inplace=True)
        gdfs[key] = gdfs[key].to_crs(epsg=26913)
        # Simplify grid and units to reduce Leaflet payload
        if key in ["grid", "units", "handsworth", "land"]:
            gdfs[key]["geometry"] = gdfs[key].geometry.simplify(25, preserve_topology=True)

    # Data Cleaning
    well_df["UWI"] = well_df["UWI"].astype(str).str.strip()
    section_df["Section"] = section_df["Section"].astype(str).str.strip()
    gdfs["grid"]["Section"] = gdfs["grid"]["Section"].astype(str).str.strip()

    # Identify Numeric Cols
    well_numeric_cols = well_df.select_dtypes(include=[np.number]).columns.tolist()
    sec_numeric_cols = section_df.select_dtypes(include=[np.number]).columns.tolist()

    # Optimized Well Merge
    gdfs["lines"]["UWI"] = gdfs["lines"]["UWI"].astype(str).str.strip()
    gdfs["points"]["UWI"] = gdfs["points"]["UWI"].astype(str).str.strip()
    
    existing_wells = pd.concat([
        gdfs["lines"][["UWI", "geometry"]],
        gdfs["points"][~gdfs["points"]["UWI"].isin(gdfs["lines"]["UWI"])][["UWI", "geometry"]]
    ]).merge(well_df, on="UWI", how="left")
    
    existing_wells = gpd.GeoDataFrame(existing_wells, crs="EPSG:26913")
    section_enriched = gdfs["grid"].merge(section_df, on="Section", how="left")

    return (existing_wells, section_enriched, gdfs["units"], gdfs["land"], gdfs["handsworth"],
            well_df, section_df, well_numeric_cols, sec_numeric_cols)

(existing_wells_gdf, section_enriched_gdf, units_gdf, land_gdf, handsworth_gdf,
 well_df, section_df, WELL_NUMERIC_COLS, SEC_NUMERIC_COLS) = load_data()

# ==========================================================
# Sidebar & Filtering
# ==========================================================
st.sidebar.title("🗺️ Control Panel")
view_mode = st.sidebar.radio("View Mode", ["Section View", "Well View"])
section_gradient = st.sidebar.selectbox("Gradient: Section Grid", ["None"] + SEC_NUMERIC_COLS)

# Filtering logic
filter_ranges = {}
target_cols = SEC_NUMERIC_COLS if view_mode == "Section View" else WELL_NUMERIC_COLS
target_gdf = section_enriched_gdf if view_mode == "Section View" else existing_wells_gdf

with st.sidebar.expander("🔍 Data Filters", expanded=True):
    for col in target_cols[:8]: # Limit sidebar clutter
        lo, hi = safe_range(target_gdf[col])
        filter_ranges[col] = st.slider(col, lo, hi, (lo, hi))

def apply_filters(df, ranges):
    mask = np.ones(len(df), dtype=bool)
    for col, (f_lo, f_hi) in ranges.items():
        if col in df.columns:
            mask &= ((df[col] >= f_lo) & (df[col] <= f_hi)) | df[col].isna()
    return df[mask]

filtered_sections = apply_filters(section_enriched_gdf, filter_ranges if view_mode == "Section View" else {})
filtered_wells = apply_filters(existing_wells_gdf, filter_ranges if view_mode == "Well View" else {})

# ==========================================================
# Map Construction
# ==========================================================
st.title("🗺️ Bakken Map Viewer")

# Reproject only the filtered subset to save time
section_display = filtered_sections.to_crs(4326)
units_display = units_gdf.to_crs(4326)
land_display = land_gdf.to_crs(4326)
handsworth_display = handsworth_gdf.to_crs(4326)
wells_display = filtered_wells.to_crs(4326)

bounds = section_display.total_bounds
m = folium.Map(location=[(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], zoom_start=10, tiles="CartoDB positron")

# ---- Layers ----
# Handsworth Units (New Layer)
folium.GeoJson(handsworth_display, name="Handsworth Units", 
               style_function=lambda _: {"color": "#e91e63", "weight": 2, "fillOpacity": 0.1}).add_to(m)

# Bakken Land
folium.GeoJson(land_display, name="Bakken Land", show=False,
               style_function=lambda _: {"fillColor": "#fff9c4", "color": "#fbc02d", "weight": 0.5, "fillOpacity": 0.2}).add_to(m)

# Section Grid (With Dynamic Styling)
if section_gradient != "None":
    vals = section_display[section_gradient].dropna()
    if not vals.empty:
        colormap = cm.LinearColormap(["#f7fcf5", "#00441b"], vmin=vals.min(), vmax=vals.max()).to_step(n=7)
        colormap.caption = section_gradient
        m.add_child(colormap)
        
        def style_fn(feature):
            v = feature["properties"].get(section_gradient)
            return {"fillColor": colormap(v), "fillOpacity": 0.5, "weight": 0.2, "color": "white"} if v is not None else NULL_STYLE
    else: style_fn = lambda _: NULL_STYLE
else: style_fn = lambda _: {"fillColor": "#ebf5fb", "color": "#2e86c1", "weight": 0.3, "fillOpacity": 0.2}

folium.GeoJson(section_display, name="Section Grid", style_function=style_fn,
               tooltip=folium.GeoJsonTooltip(fields=["Section"] + SEC_NUMERIC_COLS[:3], sticky=True)).add_to(m)

# Wells (Simplified as CircleMarkers for speed if point, GeoJson if line)
well_fg = folium.FeatureGroup(name="Wells")
for _, row in wells_display.iterrows():
    if row.geometry.geom_type == 'Point':
        folium.CircleMarker([row.geometry.y, row.geometry.x], radius=2, color="black", fill=True).add_to(well_fg)
    else:
        folium.GeoJson(row.geometry, style_function=lambda _: {"color": "black", "weight": 1}).add_to(well_fg)
well_fg.add_to(m)

Draw(export=False, draw_options={"polyline":False, "circle":False, "marker":False}).add_to(m)
folium.LayerControl().add_to(m)

# Render
map_output = st_folium(m, use_container_width=True, height=700, returned_objects=["all_drawings"])

# ==========================================================
# Selection Tool Logic
# ==========================================================
if map_output and map_output.get("all_drawings"):
    last_draw = map_output["all_drawings"][-1]
    poly_geom = shape(last_draw["geometry"])
    
    # Transform drawn poly back to data CRS
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)
    poly_data_crs = shapely_transform(lambda x, y: transformer.transform(x, y), poly_geom)
    
    # Spatial Join
    if view_mode == "Section View":
        hits = section_enriched_gdf[section_enriched_gdf.intersects(poly_data_crs)]
        cols = SEC_NUMERIC_COLS
    else:
        hits = existing_wells_gdf[existing_wells_gdf.intersects(poly_data_crs)]
        cols = WELL_NUMERIC_COLS

    if not hits.empty:
        st.write(f"### 📊 Selection Results ({len(hits)} items)")
        st.dataframe(hits.drop(columns="geometry").describe().loc[['mean', 'sum', 'count']])
        st.download_button("Download CSV", hits.drop(columns="geometry").to_csv(), "selection.csv")
    else:
        st.info("No data found in selected area.")