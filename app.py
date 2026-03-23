import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import json
from shapely.geometry import shape, Point
from shapely.ops import transform as shapely_transform
from pyproj import Transformer
import streamlit.components.v1 as components

st.set_page_config(layout="wide", page_title="Bakken WF Section Screener", page_icon="🛢️")

# ── Constants ─────────────────────────────────────────
NULL_STY = {"fillColor": "#fff", "fillOpacity": 0, "color": "#888", "weight": 0.25}
CRS_W = "EPSG:26913"
CRS_M = "EPSG:4326"
TO4 = Transformer.from_crs(CRS_W, CRS_M, always_xy=True)
TO26 = Transformer.from_crs(CRS_M, CRS_W, always_xy=True)

WELL_NUM = ["Hz Length (m)", "Cuml", "EUR"]
WELL_CAT = ["Well Type", "Status", "Objective", "Injector", "Operator"]
HEEL_TOL = 1.0

# Put your Mapbox token here or use an env variable
MAPBOX_TOKEN = st.secrets.get("MAPBOX_TOKEN", "YOUR_MAPBOX_TOKEN_HERE")


def safe_range(s):
    v = s.replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    return (lo - abs(lo) * 0.1, lo + abs(lo) * 0.1) if lo == hi else (lo, hi)


def heel_point(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return Point(geom.coords[0])
    if geom.geom_type == "MultiLineString":
        return Point(list(geom.geoms[0].coords)[0])
    if geom.geom_type == "Point":
        return geom
    return None


def toe_point(geom):
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return Point(geom.coords[-1])
    if geom.geom_type == "MultiLineString":
        return Point(list(geom.geoms[-1].coords)[-1])
    if geom.geom_type == "Point":
        return geom
    return None


def _group_heels(legs):
    heels = [(idx, heel_point(row.geometry)) for idx, row in legs.iterrows()]
    used = set()
    groups = []
    for i, (idx_i, hi) in enumerate(heels):
        if idx_i in used or hi is None:
            continue
        grp = [idx_i]
        used.add(idx_i)
        for j, (idx_j, hj) in enumerate(heels):
            if idx_j in used or hj is None:
                continue
            if hi.distance(hj) <= HEEL_TOL:
                grp.append(idx_j)
                used.add(idx_j)
        groups.append(grp)
    return groups


@st.cache_resource(show_spinner="Loading spatial data…")
def load():
    # ── Read the two well shapefiles ──
    existing = gpd.read_file("existing.shp")
    inventory = gpd.read_file("inventory.shp")
    existing["_source"] = "existing"
    inventory["_source"] = "inventory"
    lines = gpd.GeoDataFrame(
        pd.concat([existing, inventory], ignore_index=True),
        geometry="geometry",
    )

    points = gpd.read_file("points.shp")
    grid = gpd.read_file("ooipsectiongrid.shp")
    bu = gpd.read_file("Bakken Units.shp")
    hu = gpd.read_file("Handsworth Units.shp")
    land = gpd.read_file("Bakken Land.shp")
    wdf = pd.read_excel("wells.xlsx", sheet_name=0)
    sdf = pd.read_excel("wells.xlsx", sheet_name=1)

    for g in [lines, points, grid, bu, hu, land]:
        if g.crs is None:
            g.set_crs(CRS_W, inplace=True)
        g.to_crs(CRS_W, inplace=True)

    grid["geometry"] = grid.geometry.simplify(50, preserve_topology=True)
    land["geometry"] = land.geometry.simplify(50, preserve_topology=True)

    grid["Section"] = grid["Section"].astype(str).str.strip()
    wdf["UWI"] = wdf["UWI"].astype(str).str.strip()
    wdf["Section"] = wdf["Section"].astype(str).str.strip()
    sdf["Section"] = sdf["Section"].astype(str).str.strip()
    lines["UWI"] = lines["UWI"].astype(str).str.strip()
    points["UWI"] = points["UWI"].astype(str).str.strip()

    for c in ["Cuml", "EUR"]:
        if c in wdf.columns:
            wdf[c] = pd.to_numeric(wdf[c], errors="coerce")
    if "SectionOOIP" in sdf.columns:
        sdf["SectionOOIP"] = pd.to_numeric(sdf["SectionOOIP"], errors="coerce")

    # ── assemble all well geometries ──
    pts_only = points[~points["UWI"].isin(lines["UWI"])][["UWI", "geometry"]].copy()
    pts_only["_source"] = "existing"

    all_geom = gpd.GeoDataFrame(
        pd.concat(
            [lines[["UWI", "geometry", "_source"]], pts_only],
            ignore_index=True,
        ),
        geometry="geometry", crs=CRS_W,
    )

    line_mask = all_geom.geometry.geom_type.isin(["LineString", "MultiLineString"])
    legs = all_geom[line_mask].copy().reset_index(drop=True)
    pt_wells = all_geom[~line_mask].copy().reset_index(drop=True)

    # ── group multilaterals by heel ───
    groups = _group_heels(legs)

    leg_to_gid = {}
    group_meta = {}
    for gid, member_idxs in enumerate(groups):
        uwis = legs.loc[member_idxs, "UWI"].tolist()
        uwi00 = next((u for u in uwis if u.endswith("00")), uwis[0])
        is_ml = len(member_idxs) > 1
        label = uwi00 + " ML" if is_ml else uwi00
        src = legs.loc[member_idxs[0], "_source"]
        group_meta[gid] = dict(label=label, uwi00=uwi00, is_ml=is_ml,
                                member_idxs=member_idxs, source=src)
        for mi in member_idxs:
            leg_to_gid[mi] = gid

    legs["_gid"] = legs.index.map(leg_to_gid)
    legs["_leg_length_m"] = legs.geometry.length
    grp_len = legs.groupby("_gid")["_leg_length_m"].sum().rename("_total_length_m")

    grp_rows = []
    for gid, meta in group_meta.items():
        row = {"_gid": gid, "Well": meta["label"], "UWI": meta["uwi00"],
               "_source": meta["source"]}
        match = wdf[wdf["UWI"] == meta["uwi00"]]
        if not match.empty:
            r = match.iloc[0]
            for c in wdf.columns:
                if c == "UWI":
                    continue
                row[c] = r[c]
        grp_rows.append(row)
    grp_attr = pd.DataFrame(grp_rows)
    grp_attr = grp_attr.merge(grp_len.reset_index(), on="_gid", how="left")
    grp_attr.rename(columns={"_total_length_m": "Hz Length (m)"}, inplace=True)

    for col in ["Cuml", "EUR"]:
        if col in grp_attr.columns:
            grp_attr[f"_{col}_per_m"] = (
                grp_attr[col] / grp_attr["Hz Length (m)"]
            ).replace([np.inf, -np.inf], np.nan)

    # ── spatial overlay: legs × section grid ──
    legs_ov = legs[["_gid", "geometry"]].copy()
    sec_ov = grid[["Section", "geometry"]].copy()
    overlay = gpd.overlay(legs_ov, sec_ov, how="intersection")
    overlay["_int_length_m"] = overlay.geometry.length

    rate_cols = [c for c in grp_attr.columns if c.endswith("_per_m")]
    overlay = overlay.merge(grp_attr[["_gid"] + rate_cols], on="_gid", how="left")

    for col in ["Cuml", "EUR"]:
        pm = f"_{col}_per_m"
        if pm in overlay.columns:
            overlay[f"_alloc_{col}"] = overlay[pm] * overlay["_int_length_m"]

    alloc_cols = [c for c in overlay.columns if c.startswith("_alloc_")]
    sec_alloc = overlay.groupby("Section")[alloc_cols].sum().reset_index()
    sec_alloc.rename(columns={
        "_alloc_Cuml": "SectionCuml",
        "_alloc_EUR": "SectionEUR",
    }, inplace=True)

    sec = grid.merge(sdf, on="Section", how="left")
    sec = sec.merge(sec_alloc, on="Section", how="left")

    for c in ["SectionCuml", "SectionEUR"]:
        if c in sec.columns:
            sec[c] = sec[c].fillna(0)

    if "SectionOOIP" in sec.columns:
        sec["SectionOOIP"] = sec["SectionOOIP"].fillna(0)
        sec["SectionRF"] = np.where(
            sec["SectionOOIP"] > 0,
            sec["SectionCuml"] / sec["SectionOOIP"],
            np.nan,
        )
        sec["SectionURF"] = np.where(
            sec["SectionOOIP"] > 0,
            sec["SectionEUR"] / sec["SectionOOIP"],
            np.nan,
        )
    else:
        sec["SectionRF"] = np.nan
        sec["SectionURF"] = np.nan

    # ── wells display GeoDataFrame ──
    legs_base = legs[["_gid", "_source", "geometry"]].copy()
    attr_cols = [c for c in grp_attr.columns
                 if not c.startswith("_") or c in ("_gid", "_source")]
    legs_disp = legs_base.merge(
        grp_attr[attr_cols], on="_gid", how="left", suffixes=("", "_grp"),
    )
    legs_disp.drop(columns=["_gid", "_source_grp"], inplace=True, errors="ignore")

    if not pt_wells.empty:
        pt_disp = pt_wells[["UWI", "_source", "geometry"]].merge(
            wdf, on="UWI", how="left",
        )
        pt_disp["Well"] = pt_disp["UWI"]
        pt_disp["Hz Length (m)"] = 0.0

    display_cols = (
        ["Well", "UWI", "Section", "Hz Length (m)", "Cuml", "EUR",
         "Well Type", "Status", "Objective", "Injector", "Operator",
         "On Prod Date", "Last Prod Date", "On Inj Date", "Last Inj Date",
         "_source"]
    )

    for c in display_cols:
        if c not in legs_disp.columns:
            legs_disp[c] = np.nan

    frames = [legs_disp[display_cols + ["geometry"]]]

    if not pt_wells.empty:
        for c in display_cols:
            if c not in pt_disp.columns:
                pt_disp[c] = np.nan
        frames.append(pt_disp[display_cols + ["geometry"]])

    wells_final = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry", crs=CRS_W,
    )
    wells_final["_rep"] = wells_final.geometry.apply(toe_point)

    # ── overlay JSON ──
    land_j = land.to_crs(CRS_M).__geo_interface__
    bu_j = bu.to_crs(CRS_M).__geo_interface__
    hu_j = hu.to_crs(CRS_M).__geo_interface__

    return wells_final, sec, land_j, bu_j, hu_j


wells_gdf, sec_gdf, land_geojson, bu_geojson, hu_geojson = load()

SEC_NUM = [c for c in ["SectionOOIP", "SectionCuml", "SectionRF",
                        "SectionEUR", "SectionURF"] if c in sec_gdf.columns]


def add_wf(df, uplift):
    d = df.copy()
    if "SectionOOIP" in d.columns and "SectionRF" in d.columns:
        d["WF Incremental Oil (bbl)"] = d["SectionOOIP"] * (uplift / 100)
        d["Total URF w/ WF"] = d["SectionURF"] + (uplift / 100)
        d["Total Recoverable (bbl)"] = d["SectionOOIP"] * d["Total URF w/ WF"]
    return d


# ── Sidebar ───────────────────────────────────────────
sb = st.sidebar
sb.title("🛢️ WF Unit Screener")

sb.subheader("💧 Waterflood Scenario")
oil_price = sb.slider("Netback ($/bbl)", 0.0, 75.0, 35.0, 1.0)
wf_uplift = sb.slider("Waterflood RF Uplift (% pts)", 0.0, 10.0, 5.9, 0.1,
                       help="Additive percentage-point increase in recovery factor")

sb.markdown("---")
sb.subheader("🎨 Section Colouring")
WF_COLS = ["WF Incremental Oil (bbl)", "Total URF w/ WF", "Total Recoverable (bbl)"]
grad_opts = ["None"] + WF_COLS + SEC_NUM
section_gradient = sb.selectbox("Colour sections by", grad_opts, index=1)

sb.markdown("---")
sb.subheader("🔍 Section Filters")
sec_ranges = {}
for c in SEC_NUM:
    if c not in sec_gdf.columns:
        continue
    lo, hi = safe_range(sec_gdf[c])
    if lo == hi:
        continue
    r = sb.slider(c, lo, hi, (lo, hi), key=f"sf_{c}")
    if r != (lo, hi):
        sec_ranges[c] = r

sb.markdown("---")
sb.subheader("🔍 Well Filters")
well_num_ranges = {}
for c in WELL_NUM:
    if c not in wells_gdf.columns:
        continue
    lo, hi = safe_range(wells_gdf[c])
    if lo == hi:
        continue
    r = sb.slider(c, lo, hi, (lo, hi), key=f"wf_{c}")
    if r != (lo, hi):
        well_num_ranges[c] = r

well_cat_filters = {}
for c in WELL_CAT:
    if c not in wells_gdf.columns:
        continue
    opts = sorted(wells_gdf[c].dropna().unique().astype(str))
    if not opts:
        continue
    sel = sb.multiselect(c, opts, default=opts, key=f"wc_{c}")
    if len(sel) < len(opts):
        well_cat_filters[c] = sel


def mask_num(df, rngs):
    m = pd.Series(True, index=df.index)
    for c, (lo, hi) in rngs.items():
        if c in df.columns:
            m &= ((df[c] >= lo) & (df[c] <= hi)) | df[c].isna()
    return m


sec_mask = mask_num(sec_gdf, sec_ranges)
well_mask = mask_num(wells_gdf, well_num_ranges)
for c, vals in well_cat_filters.items():
    if c in wells_gdf.columns:
        well_mask &= wells_gdf[c].astype(str).isin(vals) | wells_gdf[c].isna()

sb.caption(
    f"Sections: **{sec_mask.sum()}** / {len(sec_gdf)}  •  "
    f"Wells: **{well_mask.sum()}** / {len(wells_gdf)}"
)

# ── Compute WF on filtered sections ──────────────────
sec_wf = add_wf(sec_gdf[sec_mask], wf_uplift)
sec_wf["WF Incremental Netback ($)"] = (
    sec_wf.get("WF Incremental Oil (bbl)", 0) * oil_price
)
sec_disp = sec_wf.to_crs(CRS_M)
wells_disp = wells_gdf[well_mask].to_crs(CRS_M)

ALL_SEC = SEC_NUM + [
    c for c in WF_COLS + ["WF Incremental Netback ($)"] if c in sec_wf.columns
]

st.title("🛢️ Bakken WF Section Screening Tool")
st.caption("Click sections on the map to select them, then press **Run Analysis** below.")

# ── Prepare GeoJSON data for Mapbox ──────────────────


@st.cache_data(show_spinner=False)
def prepare_map_data(_sec_disp, _wells_disp, _land_geojson, _bu_geojson, _hu_geojson,
                     gradient_col, wf_uplift_val):
    """Prepare all GeoJSON strings for the map. Cached to avoid recomputation."""

    # Section grid GeoJSON — include all numeric properties for tooltip
    sec_props_cols = ["Section"] + [
        c for c in _sec_disp.columns
        if c not in ("geometry", "_source") and _sec_disp[c].dtype in [np.float64, np.int64, float, int, object]
    ]
    sec_props_cols = [c for c in sec_props_cols if c in _sec_disp.columns]
    sec_export = _sec_disp[sec_props_cols + ["geometry"]].copy()

    # Convert any problematic types for JSON serialization
    for c in sec_export.columns:
        if c != "geometry":
            sec_export[c] = sec_export[c].apply(
                lambda x: None if (isinstance(x, float) and np.isnan(x)) else x
            )

    sec_geojson = json.loads(sec_export.to_json())

    # Compute gradient color stops if needed
    color_stops = None
    if gradient_col != "None" and gradient_col in _sec_disp.columns:
        vals = _sec_disp[gradient_col].dropna()
        if not vals.empty:
            vmin = float(vals.min())
            vmax = float(vals.max())
            if vmin == vmax:
                vmax = vmin + 1
            color_stops = {"col": gradient_col, "min": vmin, "max": vmax}

    # Wells GeoJSON — split by source and geometry type
    ttip_exclude = {"geometry", "_rep", "_source"}
    wells_existing_lines = None
    wells_inventory_lines = None
    wells_existing_points = None
    wells_inventory_points = None

    is_inv = _wells_disp["_source"] == "inventory"

    for source_mask, source_name in [(~is_inv, "existing"), (is_inv, "inventory")]:
        subset = _wells_disp[source_mask]
        if subset.empty:
            continue

        # Drop internal columns
        export_cols = [c for c in subset.columns if c not in ttip_exclude]
        sub_export = subset[export_cols].copy()
        for c in sub_export.columns:
            if c != "geometry" and sub_export[c].dtype == object:
                sub_export[c] = sub_export[c].astype(str)
            elif c != "geometry":
                sub_export[c] = sub_export[c].apply(
                    lambda x: None if (isinstance(x, float) and np.isnan(x)) else x
                )

        lm = sub_export.geometry.geom_type.isin(["LineString", "MultiLineString"])
        pm = sub_export.geometry.geom_type == "Point"

        if lm.any():
            gj = json.loads(sub_export[lm].to_json())
            if source_name == "existing":
                wells_existing_lines = gj
            else:
                wells_inventory_lines = gj

        if pm.any():
            gj = json.loads(sub_export[pm].to_json())
            if source_name == "existing":
                wells_existing_points = gj
            else:
                wells_inventory_points = gj

    return (sec_geojson, color_stops,
            wells_existing_lines, wells_inventory_lines,
            wells_existing_points, wells_inventory_points)


(sec_geojson, color_stops,
 wells_existing_lines, wells_inventory_lines,
 wells_existing_points, wells_inventory_points) = prepare_map_data(
    sec_disp, wells_disp, land_geojson, bu_geojson, hu_geojson,
    section_gradient, wf_uplift
)

# Compute map center
bnds = sec_gdf.total_bounds
cx, cy = (bnds[0] + bnds[2]) / 2, (bnds[1] + bnds[3]) / 2
clon, clat = TO4.transform(cx, cy)

# ── Build the Mapbox GL JS HTML ──────────────────────


def build_mapbox_html(
    center_lng, center_lat, zoom,
    sec_geojson, color_stops,
    land_geojson, bu_geojson, hu_geojson,
    wells_existing_lines, wells_inventory_lines,
    wells_existing_points, wells_inventory_points,
    mapbox_token,
):
    """Build a standalone HTML page with Mapbox GL JS map."""

    # Serialize data
    sec_json_str = json.dumps(sec_geojson)
    land_json_str = json.dumps(land_geojson)
    bu_json_str = json.dumps(bu_geojson)
    hu_json_str = json.dumps(hu_geojson)
    wel_lines_str = json.dumps(wells_existing_lines) if wells_existing_lines else "null"
    wil_lines_str = json.dumps(wells_inventory_lines) if wells_inventory_lines else "null"
    wel_pts_str = json.dumps(wells_existing_points) if wells_existing_points else "null"
    wil_pts_str = json.dumps(wells_inventory_points) if wells_inventory_points else "null"

    # Color expression for sections
    if color_stops:
        col = color_stops["col"]
        vmin = color_stops["min"]
        vmax = color_stops["max"]
        fill_color_expr = f"""[
            'case',
            ['has', '{col}'],
            ['interpolate', ['linear'],
                ['get', '{col}'],
                {vmin}, '#f7fcf5',
                {(vmin + vmax) / 2}, '#74c476',
                {vmax}, '#00441b'
            ],
            'rgba(255,255,255,0)'
        ]"""
        fill_opacity_expr = """[
            'case',
            ['boolean', ['feature-state', 'selected'], false],
            0.7,
            0.4
        ]"""
    else:
        fill_color_expr = "'rgba(200,200,200,0.1)'"
        fill_opacity_expr = """[
            'case',
            ['boolean', ['feature-state', 'selected'], false],
            0.3,
            0.05
        ]"""

    # Build tooltip fields from section properties
    tooltip_fields = []
    if sec_geojson and sec_geojson.get("features"):
        props = sec_geojson["features"][0].get("properties", {})
        tooltip_fields = list(props.keys())

    tooltip_html_parts = []
    for f in tooltip_fields:
        tooltip_html_parts.append(
            f"'<tr><td style=\"font-weight:bold;padding:2px 6px;\">{f}</td>"
            f"<td style=\"padding:2px 6px;\">' + "
            f"(props['{f}'] != null ? (typeof props['{f}'] === 'number' ? props['{f}'].toLocaleString(undefined, {{maximumFractionDigits:4}}) : props['{f}']) : '—') + "
            f"'</td></tr>'"
        )
    tooltip_rows_js = " +\n                    ".join(tooltip_html_parts) if tooltip_html_parts else "''"

    html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<script src="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.js"></script>
<link href="https://api.mapbox.com/mapbox-gl-js/v3.3.0/mapbox-gl.css" rel="stylesheet" />
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
    #map {{ width: 100%; height: 850px; }}

    #controls {{
        position: absolute; top: 10px; right: 10px; z-index: 10;
        background: rgba(255,255,255,0.95); border-radius: 8px;
        padding: 12px 16px; box-shadow: 0 2px 12px rgba(0,0,0,0.2);
        max-width: 260px; font-size: 13px;
    }}
    #controls h3 {{ margin: 0 0 8px 0; font-size: 14px; }}
    #controls .count {{ color: #666; margin-bottom: 8px; }}

    #run-btn {{
        display: inline-block; padding: 8px 20px;
        background: #2563eb; color: white; border: none;
        border-radius: 6px; cursor: pointer; font-size: 13px;
        font-weight: 600; width: 100%; margin-top: 4px;
        transition: background 0.2s;
    }}
    #run-btn:hover {{ background: #1d4ed8; }}
    #run-btn:disabled {{ background: #94a3b8; cursor: not-allowed; }}

    #clear-btn {{
        display: inline-block; padding: 6px 16px;
        background: #ef4444; color: white; border: none;
        border-radius: 6px; cursor: pointer; font-size: 12px;
        width: 100%; margin-top: 6px;
        transition: background 0.2s;
    }}
    #clear-btn:hover {{ background: #dc2626; }}

    .mapboxgl-popup-content {{
        padding: 8px 12px; font-size: 11px; max-height: 300px;
        overflow-y: auto; border-radius: 6px;
    }}
    .mapboxgl-popup-content table {{ border-collapse: collapse; }}
    .mapboxgl-popup-content td {{ border-bottom: 1px solid #eee; }}

    #legend {{
        position: absolute; bottom: 30px; left: 10px; z-index: 10;
        background: rgba(255,255,255,0.92); border-radius: 6px;
        padding: 8px 12px; font-size: 11px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }}
    #legend .bar {{
        width: 150px; height: 12px; border-radius: 3px;
        background: linear-gradient(to right, #f7fcf5, #74c476, #00441b);
        margin: 4px 0;
    }}
    #legend .labels {{ display: flex; justify-content: space-between; font-size: 10px; }}

    .section-selected {{
        /* managed via feature-state */
    }}
</style>
</head>
<body>

<div id="map"></div>

<div id="controls">
    <h3>📐 Section Selection</h3>
    <div class="count">Selected: <strong><span id="sel-count">0</span></strong> sections</div>
    <button id="run-btn" disabled>Run Analysis</button>
    <button id="clear-btn">Clear Selection</button>
</div>

{"" if not color_stops else f'''
<div id="legend">
    <div><strong>{color_stops["col"]}</strong></div>
    <div class="bar"></div>
    <div class="labels">
        <span>{color_stops["min"]:,.2f}</span>
        <span>{color_stops["max"]:,.2f}</span>
    </div>
</div>
'''}

<script>
    mapboxgl.accessToken = '{mapbox_token}';

    const map = new mapboxgl.Map({{
        container: 'map',
        style: 'mapbox://styles/mapbox/light-v11',
        center: [{center_lng}, {center_lat}],
        zoom: {zoom},
        antialias: true
    }});

    map.addControl(new mapboxgl.NavigationControl(), 'top-left');

    // Data
    const secData = {sec_json_str};
    const landData = {land_json_str};
    const buData = {bu_json_str};
    const huData = {hu_json_str};
    const welLines = {wel_lines_str};
    const wilLines = {wil_lines_str};
    const welPts = {wel_pts_str};
    const wilPts = {wil_pts_str};

    // Selection state
    const selectedSections = new Set();
    let featureIdMap = {{}};  // Section string -> numeric feature id

    // Assign numeric IDs to section features for feature-state
    secData.features.forEach((f, i) => {{
        f.id = i;
        featureIdMap[f.properties.Section] = i;
    }});

    function updateUI() {{
        document.getElementById('sel-count').textContent = selectedSections.size;
        document.getElementById('run-btn').disabled = selectedSections.size === 0;
    }}

    map.on('load', () => {{

        // ── Land layer ──
        map.addSource('land', {{ type: 'geojson', data: landData }});
        map.addLayer({{
            id: 'land-fill',
            type: 'fill',
            source: 'land',
            paint: {{
                'fill-color': '#fff9c4',
                'fill-opacity': 0.15
            }}
        }});

        // ── Units layers ──
        map.addSource('bu', {{ type: 'geojson', data: buData }});
        map.addLayer({{
            id: 'bu-outline',
            type: 'line',
            source: 'bu',
            paint: {{ 'line-color': '#000', 'line-width': 2.5 }}
        }});

        map.addSource('hu', {{ type: 'geojson', data: huData }});
        map.addLayer({{
            id: 'hu-outline',
            type: 'line',
            source: 'hu',
            paint: {{ 'line-color': '#000', 'line-width': 2.5 }}
        }});

        // ── Section grid ──
        map.addSource('sections', {{
            type: 'geojson',
            data: secData,
            promoteId: 'Section'
        }});

        // Fill layer — colored by gradient + selection highlight
        map.addLayer({{
            id: 'sections-fill',
            type: 'fill',
            source: 'sections',
            paint: {{
                'fill-color': {fill_color_expr},
                'fill-opacity': {fill_opacity_expr}
            }}
        }});

        // Selection highlight (separate layer on top for bold border)
        map.addLayer({{
            id: 'sections-selected-outline',
            type: 'line',
            source: 'sections',
            paint: {{
                'line-color': [
                    'case',
                    ['boolean', ['feature-state', 'selected'], false],
                    '#ff7800',
                    'rgba(0,0,0,0)'
                ],
                'line-width': [
                    'case',
                    ['boolean', ['feature-state', 'selected'], false],
                    3,
                    0
                ]
            }}
        }});

        // Grid lines (thin)
        map.addLayer({{
            id: 'sections-outline',
            type: 'line',
            source: 'sections',
            paint: {{
                'line-color': '#888',
                'line-width': 0.3
            }}
        }});

        // ── Wells ──
        if (welLines) {{
            map.addSource('wells-existing-lines', {{ type: 'geojson', data: welLines }});
            map.addLayer({{
                id: 'wells-existing-lines',
                type: 'line',
                source: 'wells-existing-lines',
                paint: {{ 'line-color': '#000', 'line-width': 1, 'line-opacity': 0.8 }}
            }});
        }}
        if (wilLines) {{
            map.addSource('wells-inventory-lines', {{ type: 'geojson', data: wilLines }});
            map.addLayer({{
                id: 'wells-inventory-lines',
                type: 'line',
                source: 'wells-inventory-lines',
                paint: {{ 'line-color': '#ef4444', 'line-width': 1, 'line-opacity': 0.8 }}
            }});
        }}
        if (welPts) {{
            map.addSource('wells-existing-pts', {{ type: 'geojson', data: welPts }});
            map.addLayer({{
                id: 'wells-existing-pts',
                type: 'circle',
                source: 'wells-existing-pts',
                paint: {{ 'circle-radius': 2.5, 'circle-color': '#000', 'circle-opacity': 0.9 }}
            }});
        }}
        if (wilPts) {{
            map.addSource('wells-inventory-pts', {{ type: 'geojson', data: wilPts }});
            map.addLayer({{
                id: 'wells-inventory-pts',
                type: 'circle',
                source: 'wells-inventory-pts',
                paint: {{ 'circle-radius': 2.5, 'circle-color': '#ef4444', 'circle-opacity': 0.9 }}
            }});
        }}

        // ── Click handler for section selection ──
        map.on('click', 'sections-fill', (e) => {{
            if (!e.features || e.features.length === 0) return;

            const feature = e.features[0];
            const secId = feature.properties.Section;

            if (selectedSections.has(secId)) {{
                selectedSections.delete(secId);
                map.setFeatureState(
                    {{ source: 'sections', id: feature.id }},
                    {{ selected: false }}
                );
            }} else {{
                selectedSections.add(secId);
                map.setFeatureState(
                    {{ source: 'sections', id: feature.id }},
                    {{ selected: true }}
                );
            }}

            updateUI();
        }});

        // ── Hover tooltip for sections ──
        const popup = new mapboxgl.Popup({{
            closeButton: false, closeOnClick: false,
            maxWidth: '320px', offset: 15
        }});

        map.on('mousemove', 'sections-fill', (e) => {{
            map.getCanvas().style.cursor = 'pointer';
            if (!e.features || e.features.length === 0) return;

            const props = e.features[0].properties;
            const html = '<table>' +
                    {tooltip_rows_js} +
                    '</table>';
            popup.setLngLat(e.lngLat).setHTML(html).addTo(map);
        }});

        map.on('mouseleave', 'sections-fill', () => {{
            map.getCanvas().style.cursor = '';
            popup.remove();
        }});

        // ── Well hover tooltips ──
        const wellPopup = new mapboxgl.Popup({{
            closeButton: false, closeOnClick: false,
            maxWidth: '300px', offset: 10
        }});

        const wellLayers = [
            'wells-existing-lines', 'wells-inventory-lines',
            'wells-existing-pts', 'wells-inventory-pts'
        ].filter(id => map.getLayer(id));

        wellLayers.forEach(layerId => {{
            map.on('mousemove', layerId, (e) => {{
                if (!e.features || e.features.length === 0) return;
                const props = e.features[0].properties;
                let html = '<table>';
                for (const [k, v] of Object.entries(props)) {{
                    if (k.startsWith('_')) continue;
                    html += '<tr><td style="font-weight:bold;padding:2px 6px;">' + k +
                            '</td><td style="padding:2px 6px;">' +
                            (v != null ? v : '—') + '</td></tr>';
                }}
                html += '</table>';
                wellPopup.setLngLat(e.lngLat).setHTML(html).addTo(map);
            }});
            map.on('mouseleave', layerId, () => {{
                wellPopup.remove();
            }});
        }});

        // ── Clear button ──
        document.getElementById('clear-btn').addEventListener('click', () => {{
            selectedSections.forEach(secId => {{
                const fid = featureIdMap[secId];
                if (fid !== undefined) {{
                    map.setFeatureState(
                        {{ source: 'sections', id: fid }},
                        {{ selected: false }}
                    );
                }}
            }});
            selectedSections.clear();
            updateUI();
        }});

        // ── Run button — send data back to Streamlit ──
        document.getElementById('run-btn').addEventListener('click', () => {{
            const selected = Array.from(selectedSections);
            // Send to Streamlit via query params workaround
            const data = JSON.stringify(selected);
            // Use window.parent.postMessage for Streamlit communication
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                value: data
            }}, '*');

            // Also store in a hidden element for polling
            let el = document.getElementById('selected-data');
            if (!el) {{
                el = document.createElement('div');
                el.id = 'selected-data';
                el.style.display = 'none';
                document.body.appendChild(el);
            }}
            el.setAttribute('data-sections', data);

            // Visual feedback
            const btn = document.getElementById('run-btn');
            btn.textContent = 'Sent! (' + selected.length + ' sections)';
            btn.style.background = '#16a34a';
            setTimeout(() => {{
                btn.textContent = 'Run Analysis';
                btn.style.background = '#2563eb';
            }}, 2000);
        }});

    }});
</script>
</body>
</html>
"""
    return html


mapbox_html = build_mapbox_html(
    clon, clat, 11,
    sec_geojson, color_stops,
    land_geojson, bu_geojson, hu_geojson,
    wells_existing_lines, wells_inventory_lines,
    wells_existing_points, wells_inventory_points,
    MAPBOX_TOKEN,
)

# ── Render map ────────────────────────────────────────
# Since Mapbox GL JS in an iframe can't easily communicate back to Streamlit
# via st.components.v1.html, we use a two-step approach:
# 1. The map lets users click sections and shows the selection
# 2. Users manually enter selected sections OR we use a custom component

# For the simplest reliable approach: render the map, and provide a text input
# where users can paste section IDs (the map shows them on click),
# OR use streamlit-javascript to capture the postMessage.

# However, the MOST practical approach is to use a proper bidirectional
# Streamlit component. Let's create a lightweight one:

# We'll use the streamlit bidirectional component API inline.

import hashlib

# Create a hash of the HTML to use as component key
html_hash = hashlib.md5(mapbox_html.encode()).hexdigest()[:8]

# Render the interactive map
map_component_value = components.html(mapbox_html, height=900, scrolling=False)

st.markdown("---")

# ── Section Selection Input ──────────────────────────
# Since components.html doesn't support bidirectional communication natively,
# provide a text-area where users type/paste section IDs from the map.
# The map clearly shows section IDs on hover for easy reference.

st.header("📐 Section Analysis")
st.caption(
    "**Click sections** on the map above to identify them (section IDs shown on hover). "
    "Then enter the section IDs below, one per line or comma-separated."
)

# Also provide a multiselect for convenience
all_section_ids = sorted(sec_wf["Section"].unique().tolist())

# Initialize session state for selected sections
if "selected_sections" not in st.session_state:
    st.session_state.selected_sections = []

col_input1, col_input2 = st.columns([2, 1])

with col_input1:
    selected_sections = st.multiselect(
        "Select sections (searchable)",
        options=all_section_ids,
        default=st.session_state.selected_sections,
        key="section_multiselect",
        help="Search and select section IDs. You can see IDs by hovering on the map.",
    )

with col_input2:
    raw_text = st.text_area(
        "Or paste section IDs (comma or newline separated)",
        height=100,
        key="section_text",
        help="Paste section IDs from the map here",
    )

    if raw_text.strip():
        # Parse comma or newline separated
        parsed = [
            s.strip() for s in raw_text.replace(",", "\n").split("\n") if s.strip()
        ]
        # Merge with multiselect
        combined = list(set(selected_sections + [s for s in parsed if s in all_section_ids]))
        selected_sections = combined
        invalid = [s for s in parsed if s not in all_section_ids]
        if invalid:
            st.warning(f"Unknown sections ignored: {', '.join(invalid[:10])}")

run_analysis = st.button(
    "🚀 Run Analysis",
    type="primary",
    disabled=len(selected_sections) == 0,
)

if run_analysis and selected_sections:
    st.session_state.selected_sections = selected_sections

    # Filter to selected sections
    sec_hits = sec_wf[sec_wf["Section"].isin(selected_sections)].copy()

    # Wells in those sections (by Section column match or spatial intersection)
    well_in_sec = wells_gdf[well_mask]
    if "Section" in well_in_sec.columns:
        well_hits = well_in_sec[well_in_sec["Section"].isin(selected_sections)]
    else:
        # Spatial join fallback
        sec_geom = sec_gdf[sec_gdf["Section"].isin(selected_sections)][["Section", "geometry"]]
        well_hits = gpd.sjoin(well_in_sec, sec_geom, how="inner", predicate="intersects")
        well_hits = well_hits.drop(columns=["index_right", "Section_right"], errors="ignore")

    well_unique = (
        well_hits.drop_duplicates(subset="Well")
        if "Well" in well_hits.columns
        else well_hits
    )

    if sec_hits.empty and well_hits.empty:
        st.info("No data for selected sections.")
    else:
        if not sec_hits.empty:
            st.subheader(f"📊 {len(sec_hits)} Sections Selected")
            c1, c2, c3, c4 = st.columns(4)
            so = (
                sec_hits["SectionOOIP"].sum()
                if "SectionOOIP" in sec_hits.columns else 0
            )
            si = (
                sec_hits["WF Incremental Oil (bbl)"].sum()
                if "WF Incremental Oil (bbl)" in sec_hits.columns else 0
            )
            sr = si * oil_price
            st_ = (
                sec_hits["Total Recoverable (bbl)"].sum()
                if "Total Recoverable (bbl)" in sec_hits.columns else 0
            )
            c1.metric("OOIP", f"{so:,.0f} bbl")
            c2.metric("WF Incremental Oil", f"{si:,.0f} bbl")
            c3.metric("Total Recoverable w/ WF", f"{st_:,.0f} bbl")
            c4.metric("Incremental Netback", f"${sr:,.0f}")

            st.subheader("Waterflood Uplift Sensitivity")
            sens = []
            for p in [1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30]:
                tmp = add_wf(sec_hits, p)
                inc = (
                    tmp["WF Incremental Oil (bbl)"].sum()
                    if "WF Incremental Oil (bbl)" in tmp.columns else 0
                )
                tot = (
                    tmp["Total Recoverable (bbl)"].sum()
                    if "Total Recoverable (bbl)" in tmp.columns else 0
                )
                sens.append({
                    "Uplift (% pts)": p,
                    "Incremental Oil (bbl)": round(inc),
                    "Total Recoverable (bbl)": round(tot),
                    f"Netback @ ${oil_price:.0f}/bbl": round(inc * oil_price),
                })
            st.dataframe(
                pd.DataFrame(sens), use_container_width=True, hide_index=True,
            )

            scols = ["Section"] + [
                c for c in ALL_SEC if c in sec_hits.columns
            ]
            sdet = sec_hits[scols].reset_index(drop=True)
            with st.expander("Section Detail", expanded=False):
                st.dataframe(sdet, use_container_width=True)
            st.download_button(
                "📥 Sections CSV", sdet.to_csv(index=False),
                "selected_sections.csv", "text/csv",
            )

        if not well_unique.empty:
            st.subheader(f"🛢️ {len(well_unique)} Wells in Selected Sections")
            wc = (
                ["Well", "UWI", "Section", "_source"]
                + [c for c in WELL_NUM if c in well_unique.columns]
                + [c for c in WELL_CAT if c in well_unique.columns]
            )
            wc = [c for c in wc if c in well_unique.columns]
            wdet = well_unique[wc].reset_index(drop=True)

            wk1, wk2, wk3 = st.columns(3)
            wk1.metric(
                "Total EUR",
                f"{well_unique['EUR'].sum():,.0f} bbl"
                if "EUR" in well_unique.columns else "—",
            )
            wk2.metric(
                "Total Cuml",
                f"{well_unique['Cuml'].sum():,.0f} bbl"
                if "Cuml" in well_unique.columns else "—",
            )
            wk3.metric(
                "Avg Hz Length",
                f"{well_unique['Hz Length (m)'].mean():,.0f} m"
                if "Hz Length (m)" in well_unique.columns else "—",
            )

            with st.expander("Well Detail", expanded=False):
                st.dataframe(wdet, use_container_width=True)

            wagg_cols = [c for c in WELL_NUM if c in well_unique.columns]
            wagg = pd.DataFrame({
                "Metric": wagg_cols,
                "Sum": [well_unique[c].sum() for c in wagg_cols],
                "Mean": [well_unique[c].mean() for c in wagg_cols],
                "Count": [well_unique[c].count() for c in wagg_cols],
            })
            with st.expander("Well Aggregates", expanded=False):
                st.dataframe(wagg, use_container_width=True)
            st.download_button(
                "📥 Wells CSV", wdet.to_csv(index=False),
                "selected_wells.csv", "text/csv", key="dl_wells",
            )

elif not run_analysis:
    st.info(
        "👆 Click sections on the map (hover to see IDs), select them above, "
        "then press **Run Analysis**."
    )