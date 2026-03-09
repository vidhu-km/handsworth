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
import matplotlib.colors as mcolors
import matplotlib.cm as mpl_cm

# ==========================================================
# Page configuration
# ==========================================================
st.set_page_config(layout="wide", page_title="Bakken Map Viewer", page_icon="🗺️")

TOOLTIP_STYLE = (
    "font-size:11px;padding:3px 6px;background:rgba(255,255,255,0.92);"
    "border:1px solid #333;border-radius:3px;"
)
NULL_STYLE = {"fillColor": "#ffffff", "fillOpacity": 0, "color": "#888", "weight": 0.25}

# ==========================================================
# Helpers
# ==========================================================

def safe_range(series):
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return 0.0, 1.0
    lo, hi = float(vals.min()), float(vals.max())
    if lo == hi:
        return (0.0, 1.0) if lo == 0 else (lo - abs(lo) * 0.1, lo + abs(lo) * 0.1)
    return lo, hi


def midpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return geom.interpolate(0.5, normalized=True)
    if geom.geom_type == "MultiLineString":
        return max(geom.geoms, key=lambda g: g.length).interpolate(0.5, normalized=True)
    if geom.geom_type == "Point":
        return geom
    return geom.centroid


def endpoint_of_geom(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return Point(list(geom.coords)[-1])
    if geom.geom_type == "MultiLineString":
        return Point(list(geom.geoms[-1].coords)[-1])
    if geom.geom_type == "Point":
        return geom
    return None


def fmt_val(v):
    if pd.isna(v):
        return "—"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) > 100 else f"{v:.3f}"
    return str(v)


# ==========================================================
# Load data
# ==========================================================
@st.cache_resource(show_spinner="Loading spatial data …")
def load_data():
    lines = gpd.read_file("lines.shp")
    points = gpd.read_file("points.shp")
    grid = gpd.read_file("ooipsectiongrid.shp")
    units = gpd.read_file("Bakken Units.shp")
    land = gpd.read_file("Bakken Land.shp")

    well_df = pd.read_excel("wells.xlsx", sheet_name=0)
    section_df = pd.read_excel("wells.xlsx", sheet_name=1)

    for gdf in [lines, points, grid, units, land]:
        if gdf.crs is None:
            gdf.set_crs(epsg=26913, inplace=True)
        gdf.to_crs(epsg=26913, inplace=True)

    grid["Section"] = grid["Section"].astype(str).str.strip()
    grid["geometry"] = grid.geometry.simplify(50, preserve_topology=True)

    well_df["UWI"] = well_df["UWI"].astype(str).str.strip()
    if "Section" in well_df.columns:
        well_df["Section"] = well_df["Section"].astype(str).str.strip()
    section_df["Section"] = section_df["Section"].astype(str).str.strip()

    # Coerce all numeric columns
    for col in well_df.columns:
        if col not in ("UWI", "Section"):
            well_df[col] = pd.to_numeric(well_df[col], errors="coerce")
    for col in section_df.columns:
        if col != "Section":
            section_df[col] = pd.to_numeric(section_df[col], errors="coerce")

    well_numeric_cols = [
        c for c in well_df.columns
        if c not in ("UWI", "Section") and pd.api.types.is_numeric_dtype(well_df[c])
    ]
    sec_numeric_cols = [
        c for c in section_df.columns
        if c != "Section" and pd.api.types.is_numeric_dtype(section_df[c])
    ]

    # Build existing wells geometry (lines preferred, points as fallback)
    lines["UWI"] = lines["UWI"].astype(str).str.strip()
    points["UWI"] = points["UWI"].astype(str).str.strip()
    lines_with_uwi = lines[["UWI", "geometry"]].copy()
    points_only = points[~points["UWI"].isin(lines_with_uwi["UWI"])][["UWI", "geometry"]].copy()
    existing_wells = gpd.GeoDataFrame(
        pd.concat([lines_with_uwi, points_only], ignore_index=True),
        geometry="geometry", crs=lines.crs,
    )
    # Merge all well data onto geometries
    existing_wells = existing_wells.merge(well_df, on="UWI", how="left")

    # Enrich section grid
    section_enriched = grid.merge(section_df, on="Section", how="left")

    return (existing_wells, section_enriched, units, land,
            well_df, section_df, well_numeric_cols, sec_numeric_cols)


(existing_wells_gdf, section_enriched_gdf, units_gdf, land_gdf,
 well_df, section_df, WELL_NUMERIC_COLS, SEC_NUMERIC_COLS) = load_data()

# ==========================================================
# Transformer
# ==========================================================
transformer_to_4326 = Transformer.from_crs("EPSG:26913", "EPSG:4326", always_xy=True)

# ==========================================================
# Sidebar
# ==========================================================
st.sidebar.title("🗺️ Map Settings")

view_mode = st.sidebar.radio("View Mode", ["Section View", "Well View"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Section Grid Gradient")
section_gradient = st.sidebar.selectbox("Colour sections by", ["None"] + SEC_NUMERIC_COLS)

# ---------- Filters ----------
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Filters")

if view_mode == "Section View":
    filter_cols = SEC_NUMERIC_COLS
    filter_target = section_enriched_gdf
else:
    filter_cols = WELL_NUMERIC_COLS
    filter_target = existing_wells_gdf

filter_ranges = {}
for col in filter_cols:
    lo, hi = safe_range(filter_target[col])
    if lo == hi:
        continue
    f_lo, f_hi = st.sidebar.slider(col, lo, hi, (lo, hi), key=f"filter_{col}")
    filter_ranges[col] = (f_lo, f_hi)

# Apply filters
def apply_filters(df, ranges):
    mask = pd.Series(True, index=df.index)
    for col, (f_lo, f_hi) in ranges.items():
        mask &= ((df[col] >= f_lo) & (df[col] <= f_hi)) | df[col].isna()
    return mask

section_mask = apply_filters(section_enriched_gdf, filter_ranges if view_mode == "Section View" else {})
well_mask = apply_filters(existing_wells_gdf, filter_ranges if view_mode == "Well View" else {})

filtered_sections = section_enriched_gdf[section_mask].copy()
filtered_wells = existing_wells_gdf[well_mask].copy()

n_sec = int(section_mask.sum())
n_well = int(well_mask.sum())

if view_mode == "Section View":
    st.sidebar.markdown(f"**{n_sec}** / {len(section_enriched_gdf)} sections pass filters")
else:
    st.sidebar.markdown(f"**{n_well}** / {len(existing_wells_gdf)} wells pass filters")

# ==========================================================
# Prepare display data (reproject to 4326)
# ==========================================================
section_display = filtered_sections.to_crs(4326)
units_display = units_gdf.to_crs(4326)
land_display = land_gdf.to_crs(4326)

# For wells: compute a representative point for each well (endpoint for lines, point for points)
wells_for_display = filtered_wells.copy()
wells_for_display["_rep_point"] = wells_for_display.geometry.apply(endpoint_of_geom)
wells_display = wells_for_display.to_crs(4326)

# Also keep a 26913 version with rep points for polygon selection
wells_for_selection = filtered_wells.copy()
wells_for_selection["_rep_point"] = wells_for_selection.geometry.apply(
    lambda g: endpoint_of_geom(g) if g.geom_type != "Point" else g
)

# ==========================================================
# Title
# ==========================================================
st.title("🗺️ Bakken Map Viewer")

# ==========================================================
# Build map
# ==========================================================
bounds = section_enriched_gdf.total_bounds
cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
clon, clat = transformer_to_4326.transform(cx, cy)

m = folium.Map(location=[clat, clon], zoom_start=11, tiles="CartoDB positron")
MiniMap(toggle_display=True, position="bottomleft").add_to(m)

# ---- Draw tool ----
Draw(
    export=False,
    position="topleft",
    draw_options={
        "polyline": False,
        "rectangle": True,
        "circle": False,
        "circlemarker": False,
        "marker": False,
        "polygon": {
            "allowIntersection": False,
            "shapeOptions": {"color": "#ff7800", "weight": 2, "fillOpacity": 0.1},
        },
    },
    edit_options={"edit": False},
).add_to(m)

# ---- Bakken Land ----
folium.FeatureGroup(name="Bakken Land", show=True).add_child(
    folium.GeoJson(land_display.to_json(), style_function=lambda _: {
        "fillColor": "#fff9c4", "color": "#fff9c4", "weight": 0.5, "fillOpacity": 0.2,
    })
).add_to(m)

# ---- Section Grid ----
if section_gradient != "None" and section_gradient in section_display.columns:
    grad_vals = section_display[section_gradient].dropna()
    if not grad_vals.empty:
        colormap = cm.LinearColormap(
            ["#f7fcf5", "#74c476", "#00441b"],
            vmin=float(grad_vals.min()), vmax=float(grad_vals.max()),
        ).to_step(n=7)
        colormap.caption = section_gradient
        m.add_child(colormap)
        sec_style = lambda feat, _col=section_gradient, _cm=colormap: (
            {"fillColor": _cm(feat["properties"].get(_col)), "fillOpacity": 0.45,
             "color": "white", "weight": 0.3}
            if feat["properties"].get(_col) is not None
            and not (isinstance(feat["properties"].get(_col), float)
                     and np.isnan(feat["properties"].get(_col)))
            else NULL_STYLE
        )
    else:
        sec_style = lambda _: NULL_STYLE
else:
    sec_style = lambda _: NULL_STYLE

sec_tip_fields = [c for c in section_display.columns if c != "geometry"]
section_fg = folium.FeatureGroup(
    name="Section Grid",
    show=(view_mode == "Section View" or section_gradient != "None"),
)
folium.GeoJson(
    section_display.to_json(), style_function=sec_style,
    highlight_function=lambda _: {"weight": 2, "color": "black", "fillOpacity": 0.5},
    tooltip=folium.GeoJsonTooltip(
        fields=sec_tip_fields, aliases=[f"{f}:" for f in sec_tip_fields],
        localize=True, sticky=True,
        style=TOOLTIP_STYLE,
    ),
).add_to(section_fg)
section_fg.add_to(m)

# ---- Units ----
folium.FeatureGroup(name="Units", show=True).add_child(
    folium.GeoJson(units_display.to_json(), style_function=lambda _: {
        "color": "black", "weight": 2, "fillOpacity": 0, "interactive": False,
    })
).add_to(m)

# ---- Existing Wells ----
well_fg = folium.FeatureGroup(name="Existing Wells", show=True)
well_tip_cols = [c for c in wells_display.columns if c not in ("geometry", "_rep_point")]

line_wells = wells_display[wells_display.geometry.type.isin(["LineString", "MultiLineString"])]
point_wells = wells_display[wells_display.geometry.type == "Point"]

if not line_wells.empty:
    # Invisible wide hitbox for tooltips
    tip_data = line_wells.drop(columns=["_rep_point"], errors="ignore").copy()
    for c in tip_data.columns:
        if c != "geometry" and tip_data[c].dtype == object:
            tip_data[c] = tip_data[c].astype(str)

    folium.GeoJson(
        tip_data.to_json(),
        style_function=lambda _: {"color": "transparent", "weight": 15, "opacity": 0},
        highlight_function=lambda _: {"weight": 15, "color": "#555", "opacity": 0.3},
        tooltip=folium.GeoJsonTooltip(
            fields=[c for c in tip_data.columns if c != "geometry"],
            aliases=[f"{c}:" for c in tip_data.columns if c != "geometry"],
            localize=True, sticky=True, style=TOOLTIP_STYLE,
        ),
    ).add_to(well_fg)

    # Visible thin line
    vis_data = line_wells.drop(columns=["_rep_point"], errors="ignore").copy()
    for c in vis_data.columns:
        if c != "geometry" and vis_data[c].dtype == object:
            vis_data[c] = vis_data[c].astype(str)
    folium.GeoJson(
        vis_data.to_json(),
        style_function=lambda _: {"color": "black", "weight": 0.5, "opacity": 0.8},
    ).add_to(well_fg)

    # Endpoints
    for _, row in line_wells.iterrows():
        ep = endpoint_of_geom(row.geometry)
        if ep is not None:
            folium.CircleMarker(
                location=[ep.y, ep.x], radius=1,
                color="black", fill=True, fill_color="black", fill_opacity=0.8, weight=1,
            ).add_to(well_fg)

for _, row in point_wells.iterrows():
    tip_parts = []
    for col in well_tip_cols:
        if col not in row.index or pd.isna(row[col]):
            continue
        tip_parts.append(f"<b>{col}:</b> {fmt_val(row[col])}")
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x], radius=2,
        color="black", fill=True, fill_color="black", fill_opacity=0.9, weight=1,
        tooltip=folium.Tooltip("<br>".join(tip_parts), sticky=True, style=TOOLTIP_STYLE),
    ).add_to(well_fg)

well_fg.add_to(m)

folium.LayerControl(collapsed=True).add_to(m)

# ==========================================================
# Render map & capture drawn polygons
# ==========================================================
map_data = st_folium(m, use_container_width=True, height=900, returned_objects=["all_drawings"])

# ==========================================================
# Polygon selection / summation
# ==========================================================
st.markdown("---")
st.header("📐 Polygon Selection Tool")
st.caption("Draw a polygon or rectangle on the map to sum values within it.")

drawings = map_data.get("all_drawings") if map_data else None

if drawings and len(drawings) > 0:
    # Use the last drawn polygon
    last_drawing = drawings[-1]
    drawn_geom_4326 = shape(last_drawing["geometry"])

    # Transform to 26913 for spatial operations
    transformer_to_26913 = Transformer.from_crs("EPSG:4326", "EPSG:26913", always_xy=True)

    from shapely.ops import transform as shapely_transform
    drawn_geom_26913 = shapely_transform(
        lambda x, y, z=None: transformer_to_26913.transform(x, y), drawn_geom_4326
    )

    drawn_gdf = gpd.GeoDataFrame(
        [{"geometry": drawn_geom_26913}], crs="EPSG:26913"
    )

    if view_mode == "Section View":
        # Find sections intersecting the drawn polygon
        hits = gpd.sjoin(
            section_enriched_gdf, drawn_gdf, how="inner", predicate="intersects"
        )
        if hits.empty:
            st.info("No sections found within the drawn polygon.")
        else:
            st.success(f"**{len(hits)}** sections found within polygon.")

            # Show individual rows
            display_cols = ["Section"] + SEC_NUMERIC_COLS
            display_cols = [c for c in display_cols if c in hits.columns]
            detail_df = hits[display_cols].reset_index(drop=True)
            st.dataframe(detail_df, use_container_width=True)

            # Sum numeric columns
            sums = hits[SEC_NUMERIC_COLS].sum(numeric_only=True)
            means = hits[SEC_NUMERIC_COLS].mean(numeric_only=True)
            counts = hits[SEC_NUMERIC_COLS].count()

            summary = pd.DataFrame({
                "Metric": sums.index,
                "Sum": sums.values,
                "Mean": means.values,
                "Count (non-null)": counts.values,
            })
            st.subheader("Aggregate Summary")
            st.dataframe(summary, use_container_width=True)

            st.download_button(
                "📥 Download Selection (CSV)",
                data=detail_df.to_csv(index=False),
                file_name="polygon_sections.csv", mime="text/csv",
            )

    else:  # Well View
        # Use representative point (endpoint/midpoint) for containment check
        rep_gdf = wells_for_selection.copy()
        rep_gdf = rep_gdf[rep_gdf["_rep_point"].notna()].copy()
        rep_gdf = rep_gdf.set_geometry(
            gpd.GeoSeries(rep_gdf["_rep_point"], crs="EPSG:26913")
        )

        hits = gpd.sjoin(rep_gdf, drawn_gdf, how="inner", predicate="within")

        if hits.empty:
            st.info("No wells found within the drawn polygon.")
        else:
            st.success(f"**{len(hits)}** wells found within polygon.")

            display_cols = ["UWI"] + WELL_NUMERIC_COLS
            if "Section" in hits.columns:
                display_cols.insert(1, "Section")
            display_cols = [c for c in display_cols if c in hits.columns]
            detail_df = hits[display_cols].reset_index(drop=True)
            st.dataframe(detail_df, use_container_width=True)

            sums = hits[WELL_NUMERIC_COLS].sum(numeric_only=True)
            means = hits[WELL_NUMERIC_COLS].mean(numeric_only=True)
            counts = hits[WELL_NUMERIC_COLS].count()

            summary = pd.DataFrame({
                "Metric": sums.index,
                "Sum": sums.values,
                "Mean": means.values,
                "Count (non-null)": counts.values,
            })
            st.subheader("Aggregate Summary")
            st.dataframe(summary, use_container_width=True)

            st.download_button(
                "📥 Download Selection (CSV)",
                data=detail_df.to_csv(index=False),
                file_name="polygon_wells.csv", mime="text/csv",
            )
else:
    st.info("Draw a polygon or rectangle on the map to select and aggregate data.")