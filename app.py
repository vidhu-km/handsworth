import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import MiniMap, Draw
from streamlit_folium import st_folium
import branca.colormap as cm
from shapely.geometry import shape
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
import json

# ── Config ────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Bakken Unitization Screener", page_icon="🛢️")

TIP_CSS = "font-size:11px;padding:3px 6px;background:rgba(255,255,255,.92);border:1px solid #333;border-radius:3px;"
NULL_STYLE = {"fillColor": "#ffffff", "fillOpacity": 0, "color": "#888", "weight": 0.25}
CRS_WORK = "EPSG:26913"
CRS_MAP = "EPSG:4326"
TO_4326 = Transformer.from_crs(CRS_WORK, CRS_MAP, always_xy=True)
TO_26913 = Transformer.from_crs(CRS_MAP, CRS_WORK, always_xy=True)


# ── Helpers ───────────────────────────────────────────────
def safe_range(s):
    v = s.replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    return (lo - abs(lo) * 0.1, lo + abs(lo) * 0.1) if lo == hi else (lo, hi)


def rep_point(g):
    """Representative point: endpoint for lines, centroid otherwise."""
    if g is None or g.is_empty:
        return None
    if g.geom_type == "LineString":
        c = list(g.coords)
        return gpd.points_from_xy([c[-1][0]], [c[-1][1]])[0]
    if g.geom_type == "MultiLineString":
        c = list(g.geoms[-1].coords)
        return gpd.points_from_xy([c[-1][0]], [c[-1][1]])[0]
    if g.geom_type == "Point":
        return g
    return g.centroid


def fmt(v):
    if pd.isna(v):
        return "—"
    if isinstance(v, float):
        return f"{v:,.0f}" if abs(v) > 100 else f"{v:.3f}"
    return str(v)


# ── Load & cache everything (runs once) ──────────────────
@st.cache_resource(show_spinner="Loading spatial data…")
def load_all():
    # Read shapefiles
    lines = gpd.read_file("lines.shp")
    points = gpd.read_file("points.shp")
    grid = gpd.read_file("ooipsectiongrid.shp")
    bakken_units = gpd.read_file("Bakken Units.shp")
    handsworth_units = gpd.read_file("Handsworth Units.shp")
    land = gpd.read_file("Bakken Land.shp")

    # Excel
    well_df = pd.read_excel("wells.xlsx", sheet_name=0)
    section_df = pd.read_excel("wells.xlsx", sheet_name=1)

    # Standardise CRS
    for gdf in [lines, points, grid, bakken_units, handsworth_units, land]:
        if gdf.crs is None:
            gdf.set_crs(CRS_WORK, inplace=True)
        gdf.to_crs(CRS_WORK, inplace=True)

    # Simplify heavy geometries
    grid["geometry"] = grid.geometry.simplify(50, preserve_topology=True)
    land["geometry"] = land.geometry.simplify(50, preserve_topology=True)

    # Clean keys
    grid["Section"] = grid["Section"].astype(str).str.strip()
    well_df["UWI"] = well_df["UWI"].astype(str).str.strip()
    section_df["Section"] = section_df["Section"].astype(str).str.strip()
    if "Section" in well_df.columns:
        well_df["Section"] = well_df["Section"].astype(str).str.strip()
    lines["UWI"] = lines["UWI"].astype(str).str.strip()
    points["UWI"] = points["UWI"].astype(str).str.strip()

    # Coerce numerics
    for df, skip in [(well_df, {"UWI", "Section"}), (section_df, {"Section"})]:
        for c in df.columns:
            if c not in skip:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    well_num = [c for c in well_df.columns if c not in ("UWI", "Section") and pd.api.types.is_numeric_dtype(well_df[c])]
    sec_num = [c for c in section_df.columns if c != "Section" and pd.api.types.is_numeric_dtype(section_df[c])]

    # Build wells GeoDataFrame (lines preferred, points fallback)
    pts_only = points[~points["UWI"].isin(lines["UWI"])][["UWI", "geometry"]]
    wells_geom = gpd.GeoDataFrame(
        pd.concat([lines[["UWI", "geometry"]], pts_only], ignore_index=True),
        geometry="geometry", crs=CRS_WORK,
    ).merge(well_df, on="UWI", how="left")
    wells_geom["_rep"] = wells_geom.geometry.apply(rep_point)

    # Section enriched
    sec_geo = grid.merge(section_df, on="Section", how="left")

    # Pre-reproject display copies to 4326 & serialise GeoJSON once
    sec_4326 = sec_geo.to_crs(CRS_MAP)
    land_4326 = land.to_crs(CRS_MAP)
    bu_4326 = bakken_units.to_crs(CRS_MAP)
    hu_4326 = handsworth_units.to_crs(CRS_MAP)
    wells_4326 = wells_geom.to_crs(CRS_MAP)
    wells_4326["_rep4326"] = wells_4326.geometry.apply(rep_point)

    # Serialise static layers once
    land_json = land_4326.to_json()
    bu_json = bu_4326.to_json()
    hu_json = hu_4326.to_json()

    return dict(
        wells=wells_geom, wells_4326=wells_4326,
        sections=sec_geo, sec_4326=sec_4326,
        land_json=land_json, bu_json=bu_json, hu_json=hu_json,
        well_df=well_df, section_df=section_df,
        well_num=well_num, sec_num=sec_num,
    )


D = load_all()

# ── Sidebar ───────────────────────────────────────────────
sb = st.sidebar
sb.title("🛢️ Unitization Screener")

view_mode = sb.radio("View Mode", ["Section View", "Well View"])

sb.markdown("---")
sb.subheader("💧 Waterflood Scenario")
oil_price = sb.slider("Oil Price ($/bbl)", 30.0, 120.0, 70.0, 1.0)
wf_uplift = sb.slider("Waterflood RF Uplift (%)", 0.0, 100.0, 25.0, 1.0,
                       help="Percent increase in recovery factor if waterflood is implemented")

sb.markdown("---")
sb.subheader("🎨 Section Colouring")
gradient_choices = ["None", "Incremental WF Oil (bbl)"] + D["sec_num"]
section_gradient = sb.selectbox("Colour sections by", gradient_choices)

sb.markdown("---")
sb.subheader("🔍 Filters")

is_sec = view_mode == "Section View"
f_cols = D["sec_num"] if is_sec else D["well_num"]
f_target = D["sections"] if is_sec else D["wells"]

ranges = {}
for col in f_cols:
    lo, hi = safe_range(f_target[col])
    if lo == hi:
        continue
    r = sb.slider(col, lo, hi, (lo, hi), key=f"f_{col}")
    if r != (lo, hi):
        ranges[col] = r


def apply_mask(df, rngs):
    m = pd.Series(True, index=df.index)
    for c, (lo, hi) in rngs.items():
        m &= ((df[c] >= lo) & (df[c] <= hi)) | df[c].isna()
    return m


sec_mask = apply_mask(D["sections"], ranges if is_sec else {})
well_mask = apply_mask(D["wells"], ranges if not is_sec else {})

sb.caption(f"{'Sections' if is_sec else 'Wells'}: **{int(sec_mask.sum() if is_sec else well_mask.sum())}** / {len(D['sections'] if is_sec else D['wells'])}")


# ── Compute waterflood incremental columns on sections ───
def compute_wf(sec_df, uplift_pct):
    """Add waterflood-derived columns. Expects OOIP and RF (recovery factor) columns."""
    df = sec_df.copy()
    # Try to detect OOIP and RF columns
    ooip_col = next((c for c in df.columns if "ooip" in c.lower()), None)
    rf_col = next((c for c in df.columns if "rf" in c.lower() or "recovery" in c.lower()), None)
    eur_col = next((c for c in df.columns if "eur" in c.lower()), None)

    if ooip_col and rf_col:
        df["Incremental WF Oil (bbl)"] = df[ooip_col] * df[rf_col] * (uplift_pct / 100.0)
        df["Total RF w/ WF"] = df[rf_col] * (1 + uplift_pct / 100.0)
        df["Total Recoverable (bbl)"] = df[ooip_col] * df["Total RF w/ WF"]
    elif ooip_col and eur_col:
        # Derive RF from EUR / OOIP
        df["_rf_derived"] = (df[eur_col] / df[ooip_col]).clip(0, 1)
        df["Incremental WF Oil (bbl)"] = df[ooip_col] * df["_rf_derived"] * (uplift_pct / 100.0)
        df["Total RF w/ WF"] = df["_rf_derived"] * (1 + uplift_pct / 100.0)
        df["Total Recoverable (bbl)"] = df[ooip_col] * df["Total RF w/ WF"]
        df.drop(columns=["_rf_derived"], inplace=True)
    elif ooip_col:
        # Assume a default primary RF of ~8% for Bakken if nothing else available
        default_rf = 0.08
        df["Incremental WF Oil (bbl)"] = df[ooip_col] * default_rf * (uplift_pct / 100.0)
        df["Total RF w/ WF"] = default_rf * (1 + uplift_pct / 100.0)
        df["Total Recoverable (bbl)"] = df[ooip_col] * df["Total RF w/ WF"]

    df["WF Incremental Revenue ($)"] = df.get("Incremental WF Oil (bbl)", 0) * oil_price
    return df


sec_enriched = compute_wf(D["sections"], wf_uplift)
sec_enriched_4326 = compute_wf(D["sec_4326"], wf_uplift)

# Apply section mask to display data
sec_display = sec_enriched_4326[sec_mask].copy()
wells_display = D["wells_4326"][well_mask].copy()

# Update numeric column list to include new computed cols
WF_COLS = [c for c in ["Incremental WF Oil (bbl)", "Total RF w/ WF",
                        "Total Recoverable (bbl)", "WF Incremental Revenue ($)"]
           if c in sec_enriched.columns]
ALL_SEC_NUM = D["sec_num"] + WF_COLS

# ── Build map ─────────────────────────────────────────────
st.title("🛢️ Bakken Unitization Screening Tool")

col_info1, col_info2, col_info3 = st.columns(3)
total_incr = sec_enriched.loc[sec_mask, "Incremental WF Oil (bbl)"].sum() if "Incremental WF Oil (bbl)" in sec_enriched.columns else 0
total_rev = sec_enriched.loc[sec_mask, "WF Incremental Revenue ($)"].sum() if "WF Incremental Revenue ($)" in sec_enriched.columns else 0
ooip_col_name = next((c for c in sec_enriched.columns if "ooip" in c.lower()), None)
total_ooip = sec_enriched.loc[sec_mask, ooip_col_name].sum() if ooip_col_name else 0

col_info1.metric("Total OOIP (filtered sections)", f"{total_ooip:,.0f} bbl")
col_info2.metric(f"Incremental WF Oil @ {wf_uplift:.0f}% uplift", f"{total_incr:,.0f} bbl")
col_info3.metric(f"Incremental Revenue @ ${oil_price:.0f}/bbl", f"${total_rev:,.0f}")

bounds = D["sections"].total_bounds
cx, cy = (bounds[0] + bounds[2]) / 2, (bounds[1] + bounds[3]) / 2
clon, clat = TO_4326.transform(cx, cy)

m = folium.Map(location=[clat, clon], zoom_start=11, tiles="CartoDB positron",
               prefer_canvas=True)
MiniMap(toggle_display=True, position="bottomleft").add_to(m)

Draw(
    export=False, position="topleft",
    draw_options=dict(polyline=False, circle=False, circlemarker=False, marker=False,
                      rectangle=True,
                      polygon=dict(allowIntersection=False,
                                   shapeOptions=dict(color="#ff7800", weight=2, fillOpacity=0.1))),
    edit_options=dict(edit=False),
).add_to(m)

# Land
folium.GeoJson(D["land_json"], name="Bakken Land",
               style_function=lambda _: {"fillColor": "#fff9c4", "color": "#fff9c4",
                                          "weight": 0.5, "fillOpacity": 0.15}).add_to(m)

# Section grid with optional gradient
grad_col = section_gradient
if grad_col != "None" and grad_col in sec_display.columns:
    vals = sec_display[grad_col].dropna()
    if not vals.empty:
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmin == vmax:
            vmax = vmin + 1
        cmap = cm.LinearColormap(["#f7fcf5", "#74c476", "#00441b"],
                                  vmin=vmin, vmax=vmax).to_step(7)
        cmap.caption = grad_col
        m.add_child(cmap)

        def sec_style(feat, _c=grad_col, _m=cmap):
            v = feat["properties"].get(_c)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return {"fillColor": _m(v), "fillOpacity": 0.5, "color": "white", "weight": 0.3}
            return NULL_STYLE
    else:
        sec_style = lambda _: NULL_STYLE
else:
    sec_style = lambda _: NULL_STYLE

tip_fields = [c for c in sec_display.columns if c != "geometry"]
folium.GeoJson(
    sec_display.to_json(), name="Section Grid", style_function=sec_style,
    highlight_function=lambda _: {"weight": 2, "color": "black", "fillOpacity": 0.55},
    tooltip=folium.GeoJsonTooltip(fields=tip_fields, aliases=[f"{f}:" for f in tip_fields],
                                   localize=True, sticky=True, style=TIP_CSS),
).add_to(m)

# Bakken Units
folium.GeoJson(D["bu_json"], name="Bakken Units",
               style_function=lambda _: {"color": "black", "weight": 2,
                                          "fillOpacity": 0, "dashArray": "5 3"}).add_to(m)

# Handsworth Units
folium.GeoJson(D["hu_json"], name="Handsworth Units",
               style_function=lambda _: {"color": "#d32f2f", "weight": 2.5,
                                          "fillOpacity": 0.05, "fillColor": "#ef9a9a"}).add_to(m)

# Wells — bulk GeoJson (no per-row loop)
well_tip_cols = [c for c in wells_display.columns if c not in ("geometry", "_rep4326", "_rep")]

# Lines
line_mask = wells_display.geometry.type.isin(["LineString", "MultiLineString"])
if line_mask.any():
    lw = wells_display[line_mask].drop(columns=["_rep4326", "_rep"], errors="ignore")
    for c in lw.columns:
        if c != "geometry" and lw[c].dtype == object:
            lw[c] = lw[c].astype(str)
    lw_json = lw.to_json()
    # Hitbox
    folium.GeoJson(lw_json, style_function=lambda _: {"color": "transparent", "weight": 14, "opacity": 0},
                   highlight_function=lambda _: {"weight": 14, "color": "#555", "opacity": 0.3},
                   tooltip=folium.GeoJsonTooltip(
                       fields=[c for c in lw.columns if c != "geometry"],
                       aliases=[f"{c}:" for c in lw.columns if c != "geometry"],
                       localize=True, sticky=True, style=TIP_CSS),
                   name="Well Lines (tooltip)").add_to(m)
    # Visible
    folium.GeoJson(lw_json, style_function=lambda _: {"color": "black", "weight": 0.5, "opacity": 0.8},
                   name="Well Lines").add_to(m)
    # Endpoints as a separate point layer (bulk)
    eps = wells_display.loc[line_mask, "_rep4326"].dropna()
    if not eps.empty:
        ep_gdf = gpd.GeoDataFrame(geometry=list(eps), crs=CRS_MAP)
        folium.GeoJson(ep_gdf.to_json(),
                       marker=folium.CircleMarker(radius=1, color="black",
                                                   fill=True, fill_color="black",
                                                   fill_opacity=0.8, weight=1),
                       name="Well Endpoints").add_to(m)

# Points
pt_mask = wells_display.geometry.type == "Point"
if pt_mask.any():
    pw = wells_display[pt_mask].drop(columns=["_rep4326", "_rep"], errors="ignore")
    for c in pw.columns:
        if c != "geometry" and pw[c].dtype == object:
            pw[c] = pw[c].astype(str)
    folium.GeoJson(
        pw.to_json(),
        marker=folium.CircleMarker(radius=2, color="black", fill=True,
                                    fill_color="black", fill_opacity=0.9, weight=1),
        tooltip=folium.GeoJsonTooltip(
            fields=[c for c in pw.columns if c != "geometry"],
            aliases=[f"{c}:" for c in pw.columns if c != "geometry"],
            localize=True, sticky=True, style=TIP_CSS),
        name="Well Points").add_to(m)

folium.LayerControl(collapsed=True).add_to(m)

# ── Render ────────────────────────────────────────────────
map_data = st_folium(m, use_container_width=True, height=850, returned_objects=["all_drawings"])

# ── Polygon selection ─────────────────────────────────────
st.markdown("---")
st.header("📐 Polygon Selection — Unitization Analysis")
st.caption("Draw a polygon/rectangle on the map to evaluate waterflood potential in that area.")

drawings = map_data.get("all_drawings") if map_data else None

if drawings and len(drawings) > 0:
    drawn_4326 = shape(drawings[-1]["geometry"])
    drawn_26913 = shapely_transform(lambda x, y, z=None: TO_26913.transform(x, y), drawn_4326)
    drawn_gdf = gpd.GeoDataFrame([{"geometry": drawn_26913}], crs=CRS_WORK)

    if is_sec:
        hits = gpd.sjoin(sec_enriched, drawn_gdf, how="inner", predicate="intersects")
        if hits.empty:
            st.info("No sections in polygon.")
        else:
            st.success(f"**{len(hits)}** sections selected")

            # ── Summary metrics ──
            c1, c2, c3, c4 = st.columns(4)
            sel_ooip = hits[ooip_col_name].sum() if ooip_col_name else 0
            sel_incr = hits["Incremental WF Oil (bbl)"].sum() if "Incremental WF Oil (bbl)" in hits.columns else 0
            sel_rev = hits["WF Incremental Revenue ($)"].sum() if "WF Incremental Revenue ($)" in hits.columns else 0
            sel_tot = hits["Total Recoverable (bbl)"].sum() if "Total Recoverable (bbl)" in hits.columns else 0

            c1.metric("OOIP in Selection", f"{sel_ooip:,.0f} bbl")
            c2.metric("Incremental WF Oil", f"{sel_incr:,.0f} bbl")
            c3.metric("Total Recoverable w/ WF", f"{sel_tot:,.0f} bbl")
            c4.metric("Incremental Revenue", f"${sel_rev:,.0f}")

            # Sensitivity table: show uplift at 10%, 25%, 50%, 75%
            st.subheader("Waterflood Uplift Sensitivity")
            sens_rows = []
            for pct in [10, 15, 20, 25, 30, 40, 50, 75]:
                tmp = compute_wf(hits, pct)
                sens_rows.append({
                    "Uplift %": pct,
                    "Incremental Oil (bbl)": tmp["Incremental WF Oil (bbl)"].sum() if "Incremental WF Oil (bbl)" in tmp.columns else 0,
                    "Revenue ($)": (tmp["Incremental WF Oil (bbl)"].sum() * oil_price) if "Incremental WF Oil (bbl)" in tmp.columns else 0,
                })
            st.dataframe(pd.DataFrame(sens_rows), use_container_width=True, hide_index=True)

            # Detail table
            show_cols = ["Section"] + [c for c in ALL_SEC_NUM if c in hits.columns]
            detail = hits[show_cols].reset_index(drop=True)

            with st.expander("Section Detail Table", expanded=False):
                st.dataframe(detail, use_container_width=True)

            # Aggregates
            agg_cols = [c for c in ALL_SEC_NUM if c in hits.columns]
            agg = pd.DataFrame({
                "Metric": agg_cols,
                "Sum": [hits[c].sum() for c in agg_cols],
                "Mean": [hits[c].mean() for c in agg_cols],
                "Count": [hits[c].count() for c in agg_cols],
            })
            with st.expander("Aggregate Summary", expanded=False):
                st.dataframe(agg, use_container_width=True)

            st.download_button("📥 Download CSV", detail.to_csv(index=False),
                               "polygon_sections.csv", "text/csv")

    else:  # Well view
        rg = D["wells"].loc[well_mask].copy()
        rg = rg[rg["_rep"].notna()].set_geometry(gpd.GeoSeries(rg["_rep"], crs=CRS_WORK))
        hits = gpd.sjoin(rg, drawn_gdf, how="inner", predicate="within")
        if hits.empty:
            st.info("No wells in polygon.")
        else:
            st.success(f"**{len(hits)}** wells selected")
            show_cols = ["UWI"] + (["Section"] if "Section" in hits.columns else []) + D["well_num"]
            show_cols = [c for c in show_cols if c in hits.columns]
            detail = hits[show_cols].reset_index(drop=True)
            st.dataframe(detail, use_container_width=True)

            agg = pd.DataFrame({
                "Metric": D["well_num"],
                "Sum": [hits[c].sum() for c in D["well_num"]],
                "Mean": [hits[c].mean() for c in D["well_num"]],
                "Count": [hits[c].count() for c in D["well_num"]],
            })
            with st.expander("Aggregate Summary"):
                st.dataframe(agg, use_container_width=True)
            st.download_button("📥 Download CSV", detail.to_csv(index=False),
                               "polygon_wells.csv", "text/csv")
else:
    st.info("Draw a polygon or rectangle on the map to evaluate waterflood unitization potential.")