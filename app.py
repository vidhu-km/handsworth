import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import pydeck as pdk
from shapely.geometry import Point
from pyproj import Transformer
import json

st.set_page_config(layout="wide", page_title="Bakken WF Section Screener", page_icon="🛢️")

CRS_W = "EPSG:26913"
CRS_M = "EPSG:4326"
TO4 = Transformer.from_crs(CRS_W, CRS_M, always_xy=True)

WELL_NUM = ["Hz Length (m)", "Cuml", "EUR"]
WELL_CAT = ["Well Type", "Status", "Objective", "Injector", "Operator"]
HEEL_TOL = 1.0


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


def _val_color(v, vmin, vmax):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return [200, 200, 200, 40]
    t = 0.5 if vmax == vmin else max(0.0, min(1.0, (v - vmin) / (vmax - vmin)))
    r = int(247 - t * 247)
    g = int(252 - t * 184)
    b = int(245 - t * 218)
    return [r, g, b, 160]


@st.cache_resource(show_spinner="Loading spatial data…")
def load():
    existing = gpd.read_file("existing.shp")
    inventory = gpd.read_file("inventory.shp")
    existing["_source"] = "existing"
    inventory["_source"] = "inventory"
    lines = gpd.GeoDataFrame(
        pd.concat([existing, inventory], ignore_index=True), geometry="geometry",
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

    pts_only = points[~points["UWI"].isin(lines["UWI"])][["UWI", "geometry"]].copy()
    pts_only["_source"] = "existing"
    all_geom = gpd.GeoDataFrame(
        pd.concat([lines[["UWI", "geometry", "_source"]], pts_only], ignore_index=True),
        geometry="geometry", crs=CRS_W,
    )

    line_mask = all_geom.geometry.geom_type.isin(["LineString", "MultiLineString"])
    legs = all_geom[line_mask].copy().reset_index(drop=True)
    pt_wells = all_geom[~line_mask].copy().reset_index(drop=True)

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
        row = {"_gid": gid, "Well": meta["label"], "UWI": meta["uwi00"], "_source": meta["source"]}
        match = wdf[wdf["UWI"] == meta["uwi00"]]
        if not match.empty:
            r = match.iloc[0]
            for c in wdf.columns:
                if c != "UWI":
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

    overlay = gpd.overlay(
        legs[["_gid", "geometry"]].copy(),
        grid[["Section", "geometry"]].copy(),
        how="intersection",
    )
    overlay["_int_length_m"] = overlay.geometry.length
    rate_cols = [c for c in grp_attr.columns if c.endswith("_per_m")]
    overlay = overlay.merge(grp_attr[["_gid"] + rate_cols], on="_gid", how="left")

    for col in ["Cuml", "EUR"]:
        pm = f"_{col}_per_m"
        if pm in overlay.columns:
            overlay[f"_alloc_{col}"] = overlay[pm] * overlay["_int_length_m"]

    alloc_cols = [c for c in overlay.columns if c.startswith("_alloc_")]
    sec_alloc = overlay.groupby("Section")[alloc_cols].sum().reset_index()
    sec_alloc.rename(columns={"_alloc_Cuml": "SectionCuml", "_alloc_EUR": "SectionEUR"}, inplace=True)

    sec = grid.merge(sdf, on="Section", how="left").merge(sec_alloc, on="Section", how="left")
    for c in ["SectionCuml", "SectionEUR"]:
        if c in sec.columns:
            sec[c] = sec[c].fillna(0)
    if "SectionOOIP" in sec.columns:
        sec["SectionOOIP"] = sec["SectionOOIP"].fillna(0)
        sec["SectionRF"] = np.where(sec["SectionOOIP"] > 0, sec["SectionCuml"] / sec["SectionOOIP"], np.nan)
        sec["SectionURF"] = np.where(sec["SectionOOIP"] > 0, sec["SectionEUR"] / sec["SectionOOIP"], np.nan)
    else:
        sec["SectionRF"] = np.nan
        sec["SectionURF"] = np.nan

    legs_base = legs[["_gid", "_source", "geometry"]].copy()
    attr_cols = [c for c in grp_attr.columns if not c.startswith("_") or c in ("_gid", "_source")]
    legs_disp = legs_base.merge(grp_attr[attr_cols], on="_gid", how="left", suffixes=("", "_grp"))
    legs_disp.drop(columns=["_gid", "_source_grp"], inplace=True, errors="ignore")

    if not pt_wells.empty:
        pt_disp = pt_wells[["UWI", "_source", "geometry"]].merge(wdf, on="UWI", how="left")
        pt_disp["Well"] = pt_disp["UWI"]
        pt_disp["Hz Length (m)"] = 0.0

    display_cols = [
        "Well", "UWI", "Section", "Hz Length (m)", "Cuml", "EUR",
        "Well Type", "Status", "Objective", "Injector", "Operator",
        "On Prod Date", "Last Prod Date", "On Inj Date", "Last Inj Date", "_source",
    ]
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
        pd.concat(frames, ignore_index=True), geometry="geometry", crs=CRS_W,
    )
    wells_final["_rep"] = wells_final.geometry.apply(toe_point)
    return wells_final, sec, land, bu, hu


wells_gdf, sec_gdf, land_gdf, bu_gdf, hu_gdf = load()
SEC_NUM = [c for c in ["SectionOOIP", "SectionCuml", "SectionRF", "SectionEUR", "SectionURF"] if c in sec_gdf.columns]


def add_wf(df, uplift):
    d = df.copy()
    if "SectionOOIP" in d.columns and "SectionRF" in d.columns:
        d["WF Incremental Oil (bbl)"] = d["SectionOOIP"] * (uplift / 100)
        d["Total URF w/ WF"] = d["SectionURF"] + (uplift / 100)
        d["Total Recoverable (bbl)"] = d["SectionOOIP"] * d["Total URF w/ WF"]
    return d


# ── Session state init ────────────────────────────────
if "selected_sections" not in st.session_state:
    st.session_state.selected_sections = set()
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

# ── Sidebar ───────────────────────────────────────────
sb = st.sidebar
sb.title("🛢️ WF Unit Screener")

n_sel = len(st.session_state.selected_sections)
sb.markdown(f"**{n_sel} section{'s' if n_sel != 1 else ''} selected**")

col_clear, col_run = sb.columns(2)
with col_clear:
    if st.button("❌ Clear", disabled=(n_sel == 0), use_container_width=True):
        st.session_state.selected_sections = set()
        st.session_state.run_analysis = False
        st.rerun()
with col_run:
    if st.button("🚀 Run", type="primary", disabled=(n_sel == 0), use_container_width=True):
        st.session_state.run_analysis = True

if st.session_state.selected_sections:
    sb.caption("Selected: " + ", ".join(sorted(st.session_state.selected_sections)))

sb.markdown("---")
sb.subheader("💧 Waterflood Scenario")
oil_price = sb.slider("Netback ($/bbl)", 0.0, 75.0, 35.0, 1.0)
wf_uplift = sb.slider("Waterflood RF Uplift (% pts)", 0.0, 10.0, 5.9, 0.1)

sb.markdown("---")
sb.subheader("🎨 Section Colouring")
WF_COLS = ["WF Incremental Oil (bbl)", "Total URF w/ WF", "Total Recoverable (bbl)"]
grad_opts = ["None"] + WF_COLS + SEC_NUM
section_gradient = sb.selectbox("Colour sections by", grad_opts, index=1)

sb.markdown("---")
sb.subheader("🗺️ Map Layers")
show_land = sb.checkbox("Bakken Land", value=True)
show_bu = sb.checkbox("Bakken Units", value=True)
show_hu = sb.checkbox("Handsworth Units", value=True)
show_grid = sb.checkbox("Section Grid", value=True)
show_existing = sb.checkbox("Existing Wells", value=True)
show_inventory = sb.checkbox("Inventory Wells", value=True)
show_endpoints = sb.checkbox("Well Endpoints", value=True)

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

sb.caption(f"Sections: **{sec_mask.sum()}** / {len(sec_gdf)}  •  Wells: **{well_mask.sum()}** / {len(wells_gdf)}")

# ── Filtered data ─────────────────────────────────────
sec_wf = add_wf(sec_gdf[sec_mask], wf_uplift)
sec_wf["WF Incremental Netback ($)"] = sec_wf.get("WF Incremental Oil (bbl)", 0) * oil_price
wells_disp = wells_gdf[well_mask].copy()
ALL_SEC = SEC_NUM + [c for c in WF_COLS + ["WF Incremental Netback ($)"] if c in sec_wf.columns]

# ── Title ─────────────────────────────────────────────
st.title("🛢️ Bakken WF Section Screening Tool")
st.caption("🖱️ Click sections on the map to select/deselect. Then hit **Run** in the sidebar.")

# ── Build map data ────────────────────────────────────
def to_geojson_4326(gdf):
    return json.loads(gdf.to_crs(CRS_M).to_json())


selected_secs = st.session_state.selected_sections
HIGHLIGHT_COLOR = [30, 130, 230, 180]

sec_4326 = sec_wf.to_crs(CRS_M).copy()
for c in sec_4326.columns:
    if c == "geometry":
        continue
    if sec_4326[c].dtype in [np.float64, np.float32]:
        sec_4326[c] = sec_4326[c].replace([np.inf, -np.inf], np.nan)
        sec_4326[c] = sec_4326[c].where(sec_4326[c].notna(), None)

gc = section_gradient
if gc != "None" and gc in sec_4326.columns:
    vals = sec_wf[gc].replace([np.inf, -np.inf], np.nan).dropna()
    vmin, vmax = (float(vals.min()), float(vals.max())) if not vals.empty else (0, 1)
    if vmin == vmax:
        vmax = vmin + 1
else:
    vmin, vmax = 0, 1

fill_colors, line_colors = [], []
for _, row in sec_4326.iterrows():
    sec_id = str(row.get("Section", ""))
    if sec_id in selected_secs:
        fill_colors.append(HIGHLIGHT_COLOR)
        line_colors.append([30, 130, 230, 255])
    else:
        if gc != "None" and gc in sec_4326.columns:
            fill_colors.append(_val_color(row.get(gc), vmin, vmax))
        else:
            fill_colors.append([200, 200, 200, 40])
        line_colors.append([136, 136, 136, 180])

sec_4326["_fill_color"] = fill_colors
sec_4326["_line_color"] = line_colors

for c in sec_4326.columns:
    if c in ("geometry", "_fill_color", "_line_color"):
        continue
    if sec_4326[c].dtype in [np.float64, np.float32]:
        sec_4326[c] = sec_4326[c].apply(lambda x: round(x, 4) if x is not None else None)

sec_geojson = json.loads(sec_4326.to_json())
for i, feat in enumerate(sec_geojson["features"]):
    feat["properties"]["_fill_color"] = fill_colors[i]
    feat["properties"]["_line_color"] = line_colors[i]

# ── Build layers ──────────────────────────────────────
layers = []

if show_land:
    layers.append(pdk.Layer(
        "GeoJsonLayer", data=to_geojson_4326(land_gdf),
        get_fill_color=[255, 249, 196, 40], get_line_color=[255, 249, 196, 130],
        get_line_width=10, pickable=False,
    ))
if show_bu:
    layers.append(pdk.Layer(
        "GeoJsonLayer", data=to_geojson_4326(bu_gdf),
        get_fill_color=[0, 0, 0, 0], get_line_color=[0, 0, 0, 255],
        get_line_width=30, pickable=False,
    ))
if show_hu:
    layers.append(pdk.Layer(
        "GeoJsonLayer", data=to_geojson_4326(hu_gdf),
        get_fill_color=[0, 0, 0, 0], get_line_color=[0, 0, 0, 255],
        get_line_width=30, pickable=False,
    ))
if show_grid:
    layers.append(pdk.Layer(
        "GeoJsonLayer", data=sec_geojson,
        get_fill_color="properties._fill_color", get_line_color="properties._line_color",
        get_line_width=5, pickable=True, auto_highlight=True,
        highlight_color=[255, 200, 0, 120],
    ))

is_inv = wells_disp["_source"] == "inventory"
for subset, color, show_flag in [
    (wells_disp[~is_inv], [0, 0, 0, 200], show_existing),
    (wells_disp[is_inv], [220, 30, 30, 200], show_inventory),
]:
    if not show_flag or subset.empty:
        continue
    lm = subset.geometry.geom_type.isin(["LineString", "MultiLineString"])

    if lm.any():
        lw = subset[lm].drop(columns=["_rep"], errors="ignore").to_crs(CRS_M).copy()
        for c in lw.columns:
            if c == "geometry":
                continue
            if lw[c].dtype in [np.float64, np.float32]:
                lw[c] = lw[c].replace([np.inf, -np.inf], np.nan).where(lambda x: x.notna(), None)
            elif lw[c].dtype == object:
                lw[c] = lw[c].astype(str)
        layers.append(pdk.Layer(
            "GeoJsonLayer", data=json.loads(lw.drop(columns=["_source"], errors="ignore").to_json()),
            get_line_color=color, get_line_width=15, pickable=True,
            auto_highlight=True, highlight_color=[255, 255, 0, 150],
        ))

    if show_endpoints and lm.any():
        eps = subset.loc[lm.values, "_rep"].dropna()
        if not eps.empty:
            ep_gdf = gpd.GeoDataFrame(geometry=list(eps), crs=CRS_W).to_crs(CRS_M)
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=[{"position": [g.x, g.y]} for g in ep_gdf.geometry],
                get_position="position", get_fill_color=color, get_radius=30, pickable=False,
            ))

    pm_mask = subset.geometry.geom_type == "Point"
    if pm_mask.any():
        pw = subset[pm_mask].drop(columns=["_rep"], errors="ignore").to_crs(CRS_M).copy()
        for c in pw.columns:
            if c == "geometry":
                continue
            if pw[c].dtype in [np.float64, np.float32]:
                pw[c] = pw[c].replace([np.inf, -np.inf], np.nan).where(lambda x: x.notna(), None)
            elif pw[c].dtype == object:
                pw[c] = pw[c].astype(str)
        layers.append(pdk.Layer(
            "GeoJsonLayer", data=json.loads(pw.drop(columns=["_source"], errors="ignore").to_json()),
            get_fill_color=color, get_radius=40, point_radius_min_pixels=2, pickable=True,
        ))

# ── Map ───────────────────────────────────────────────
bnds = sec_gdf.total_bounds
cx, cy = (bnds[0] + bnds[2]) / 2, (bnds[1] + bnds[3]) / 2
clon, clat = TO4.transform(cx, cy)

tooltip = {
    "html": (
        "<div style='font-size:11px;max-width:400px;padding:4px;'>"
        "<b>Section:</b> {Section}<br/>"
        "<b>SectionOOIP:</b> {SectionOOIP}<br/>"
        "<b>SectionCuml:</b> {SectionCuml}<br/>"
        "<b>SectionEUR:</b> {SectionEUR}<br/>"
        "<b>SectionRF:</b> {SectionRF}<br/>"
        "<b>SectionURF:</b> {SectionURF}<br/>"
        "<b>WF Incremental Oil:</b> {WF Incremental Oil (bbl)}<br/>"
        "<b>Total URF w/ WF:</b> {Total URF w/ WF}<br/>"
        "<b>Well:</b> {Well}<br/>"
        "<b>UWI:</b> {UWI}<br/>"
        "<b>Cuml:</b> {Cuml}<br/>"
        "<b>EUR:</b> {EUR}<br/>"
        "<b>Operator:</b> {Operator}<br/>"
        "</div>"
    ),
    "style": {"backgroundColor": "rgba(255,255,255,0.95)", "color": "#333",
              "border": "1px solid #333", "borderRadius": "3px"},
}

event = st.pydeck_chart(
    pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=clat, longitude=clon, zoom=11, pitch=0, bearing=0),
        map_style="light",
        tooltip=tooltip,
    ),
    use_container_width=True,
    height=850,
    on_select="rerun",
    selection_mode="single-object",
)

# ── Handle click → toggle selection (NO analysis yet) ─
if event and event.selection:
    sel_objects = event.selection.get("objects", {})
    clicked_section = None
    for key, obj_list in sel_objects.items():
        if isinstance(obj_list, list):
            for obj in obj_list:
                if isinstance(obj, dict) and "Section" in obj:
                    clicked_section = str(obj["Section"])
                    break
        if clicked_section:
            break

    if clicked_section:
        if clicked_section in st.session_state.selected_sections:
            st.session_state.selected_sections.discard(clicked_section)
        else:
            st.session_state.selected_sections.add(clicked_section)
        # Reset run flag so analysis doesn't auto-show on new selections
        st.session_state.run_analysis = False
        st.rerun()

# ── Analysis (ONLY when Run is clicked) ───────────────
st.markdown("---")
st.header("📐 Selected Sections — Analysis")

if not st.session_state.run_analysis:
    if st.session_state.selected_sections:
        st.info(f"**{len(st.session_state.selected_sections)}** sections selected. Click **🚀 Run** in the sidebar to analyze.")
    else:
        st.info("Click sections on the map to select them, then click **🚀 Run** in the sidebar.")
else:
    sel_secs = st.session_state.selected_sections
    st.caption(f"**{len(sel_secs)}** sections analyzed.")

    sec_hits = sec_wf[sec_wf["Section"].isin(sel_secs)].copy()
    well_hits = wells_gdf[well_mask & wells_gdf["Section"].isin(sel_secs)].copy()
    well_unique = well_hits.drop_duplicates(subset="Well") if "Well" in well_hits.columns else well_hits

    if sec_hits.empty and well_hits.empty:
        st.info("No data for selected sections.")
    else:
        if not sec_hits.empty:
            st.subheader(f"📊 {len(sec_hits)} Sections")
            c1, c2, c3, c4 = st.columns(4)
            so = sec_hits["SectionOOIP"].sum() if "SectionOOIP" in sec_hits.columns else 0
            si = sec_hits["WF Incremental Oil (bbl)"].sum() if "WF Incremental Oil (bbl)" in sec_hits.columns else 0
            sr = si * oil_price
            st_ = sec_hits["Total Recoverable (bbl)"].sum() if "Total Recoverable (bbl)" in sec_hits.columns else 0
            c1.metric("OOIP", f"{so:,.0f} bbl")
            c2.metric("WF Incremental Oil", f"{si:,.0f} bbl")
            c3.metric("Total Recoverable w/ WF", f"{st_:,.0f} bbl")
            c4.metric("Incremental Netback", f"${sr:,.0f}")

            st.subheader("Waterflood Uplift Sensitivity")
            sens = []
            for p in [1, 2, 3, 5, 7.5, 10, 15, 20, 25, 30]:
                tmp = add_wf(sec_hits, p)
                inc = tmp["WF Incremental Oil (bbl)"].sum() if "WF Incremental Oil (bbl)" in tmp.columns else 0
                tot = tmp["Total Recoverable (bbl)"].sum() if "Total Recoverable (bbl)" in tmp.columns else 0
                sens.append({
                    "Uplift (% pts)": p,
                    "Incremental Oil (bbl)": round(inc),
                    "Total Recoverable (bbl)": round(tot),
                    f"Netback @ ${oil_price:.0f}/bbl": round(inc * oil_price),
                })
            st.dataframe(pd.DataFrame(sens), use_container_width=True, hide_index=True)

            scols = ["Section"] + [c for c in ALL_SEC if c in sec_hits.columns]
            sdet = sec_hits[scols].reset_index(drop=True)
            with st.expander("Section Detail", expanded=False):
                st.dataframe(sdet, use_container_width=True)
            st.download_button("📥 Sections CSV", sdet.to_csv(index=False), "selected_sections.csv", "text/csv")

        if not well_unique.empty:
            st.subheader(f"🛢️ {len(well_unique)} Wells in Selected Sections")
            wc = ["Well", "UWI", "Section", "_source"] + \
                 [c for c in WELL_NUM if c in well_unique.columns] + \
                 [c for c in WELL_CAT if c in well_unique.columns]
            wc = [c for c in wc if c in well_unique.columns]
            wdet = well_unique[wc].reset_index(drop=True)

            wk1, wk2, wk3 = st.columns(3)
            wk1.metric("Total EUR", f"{well_unique['EUR'].sum():,.0f} bbl" if "EUR" in well_unique.columns else "—")
            wk2.metric("Total Cuml", f"{well_unique['Cuml'].sum():,.0f} bbl" if "Cuml" in well_unique.columns else "—")
            wk3.metric("Avg Hz Length", f"{well_unique['Hz Length (m)'].mean():,.0f} m" if "Hz Length (m)" in well_unique.columns else "—")

            with st.expander("Well Detail", expanded=False):
                st.dataframe(wdet, use_container_width=True)
            st.download_button("📥 Wells CSV", wdet.to_csv(index=False), "selected_wells.csv", "text/csv", key="dl_wells")