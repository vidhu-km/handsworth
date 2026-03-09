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

st.set_page_config(layout="wide", page_title="Bakken Unitization Screener", page_icon="🛢️")

TIP = "font-size:11px;padding:3px 6px;background:rgba(255,255,255,.92);border:1px solid #333;border-radius:3px;"
NULL_STY = {"fillColor": "#fff", "fillOpacity": 0, "color": "#888", "weight": 0.25}
CRS_W, CRS_M = "EPSG:26913", "EPSG:4326"
TO4 = Transformer.from_crs(CRS_W, CRS_M, always_xy=True)
TO26 = Transformer.from_crs(CRS_M, CRS_W, always_xy=True)

WELL_NUM = ["Hz Length (m)", "Cuml", "Norm Cuml", "EUR", "Norm EUR",
            "1Y Cuml", "Norm 1Y Cuml", "IP90", "Norm IP90"]
WELL_CAT = ["Well Type", "Status", "Objective", "Injector", "Operator"]
SEC_NUM = ["SectionOOIP", "SectionCuml", "SectionRF", "SectionEUR", "SectionURF"]


def safe_range(s):
    v = s.replace([np.inf, -np.inf], np.nan).dropna()
    if v.empty:
        return 0.0, 1.0
    lo, hi = float(v.min()), float(v.max())
    return (lo - abs(lo) * .1, lo + abs(lo) * .1) if lo == hi else (lo, hi)


def rep_pt(g):
    if g is None or g.is_empty:
        return None
    if g.geom_type == "LineString":
        return gpd.points_from_xy([g.coords[-1][0]], [g.coords[-1][1]])[0]
    if g.geom_type == "MultiLineString":
        c = list(g.geoms[-1].coords)
        return gpd.points_from_xy([c[-1][0]], [c[-1][1]])[0]
    if g.geom_type == "Point":
        return g
    return g.centroid


@st.cache_resource(show_spinner="Loading spatial data…")
def load():
    lines = gpd.read_file("lines.shp")
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

    for c in WELL_NUM:
        if c in wdf.columns:
            wdf[c] = pd.to_numeric(wdf[c], errors="coerce")
    for c in SEC_NUM:
        if c in sdf.columns:
            sdf[c] = pd.to_numeric(sdf[c], errors="coerce")

    pts_only = points[~points["UWI"].isin(lines["UWI"])][["UWI", "geometry"]]
    wells = gpd.GeoDataFrame(
        pd.concat([lines[["UWI", "geometry"]], pts_only], ignore_index=True),
        geometry="geometry", crs=CRS_W,
    ).merge(wdf, on="UWI", how="left")
    wells["_rep"] = wells.geometry.apply(rep_pt)

    sec = grid.merge(sdf, on="Section", how="left")

    land_j = land.to_crs(CRS_M).to_json()
    bu_j = bu.to_crs(CRS_M).to_json()
    hu_j = hu.to_crs(CRS_M).to_json()

    return wells, sec, land_j, bu_j, hu_j


wells_gdf, sec_gdf, land_json, bu_json, hu_json = load()


def add_wf(df, uplift):
    d = df.copy()
    if "SectionOOIP" in d.columns and "SectionRF" in d.columns:
        d["WF Incremental Oil (bbl)"] = d["SectionOOIP"] * d["SectionRF"] * (uplift / 100)
        d["Total RF w/ WF"] = d["SectionRF"] * (1 + uplift / 100)
        d["Total Recoverable (bbl)"] = d["SectionOOIP"] * d["Total RF w/ WF"]
    return d


# ── Sidebar ───────────────────────────────────────────
sb = st.sidebar
sb.title("🛢️ Unitization Screener")

sb.subheader("💧 Waterflood Scenario")
oil_price = sb.slider("Oil Price ($/bbl)", 30.0, 120.0, 70.0, 1.0)
wf_uplift = sb.slider("Waterflood RF Uplift (%)", 0.0, 100.0, 25.0, 1.0,
                       help="% increase in recovery factor from waterflood")

sb.markdown("---")
sb.subheader("🎨 Section Colouring")
WF_COLS = ["WF Incremental Oil (bbl)", "Total RF w/ WF", "Total Recoverable (bbl)"]
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
        m &= ((df[c] >= lo) & (df[c] <= hi)) | df[c].isna()
    return m


sec_mask = mask_num(sec_gdf, sec_ranges)
well_mask = mask_num(wells_gdf, well_num_ranges)
for c, vals in well_cat_filters.items():
    well_mask &= wells_gdf[c].astype(str).isin(vals) | wells_gdf[c].isna()

sb.caption(f"Sections: **{sec_mask.sum()}** / {len(sec_gdf)}  •  Wells: **{well_mask.sum()}** / {len(wells_gdf)}")

# ── Compute WF on filtered sections ──────────────────
sec_wf = add_wf(sec_gdf[sec_mask], wf_uplift)
sec_wf["WF Incremental Revenue ($)"] = sec_wf.get("WF Incremental Oil (bbl)", 0) * oil_price
sec_disp = sec_wf.to_crs(CRS_M)
wells_disp = wells_gdf[well_mask].to_crs(CRS_M)

ALL_SEC = SEC_NUM + [c for c in WF_COLS + ["WF Incremental Revenue ($)"] if c in sec_wf.columns]

# ── KPIs ──────────────────────────────────────────────
st.title("🛢️ Bakken Unitization Screening Tool")
k1, k2, k3, k4 = st.columns(4)
t_ooip = sec_wf["SectionOOIP"].sum() if "SectionOOIP" in sec_wf.columns else 0
t_incr = sec_wf["WF Incremental Oil (bbl)"].sum() if "WF Incremental Oil (bbl)" in sec_wf.columns else 0
t_rev = sec_wf["WF Incremental Revenue ($)"].sum() if "WF Incremental Revenue ($)" in sec_wf.columns else 0
t_tot = sec_wf["Total Recoverable (bbl)"].sum() if "Total Recoverable (bbl)" in sec_wf.columns else 0
k1.metric("OOIP (filtered)", f"{t_ooip:,.0f} bbl")
k2.metric(f"WF Incremental @ {wf_uplift:.0f}%", f"{t_incr:,.0f} bbl")
k3.metric("Total Recoverable w/ WF", f"{t_tot:,.0f} bbl")
k4.metric(f"Incremental Rev @ ${oil_price:.0f}", f"${t_rev:,.0f}")

# ── Map ───────────────────────────────────────────────
bnds = sec_gdf.total_bounds
cx, cy = (bnds[0] + bnds[2]) / 2, (bnds[1] + bnds[3]) / 2
clon, clat = TO4.transform(cx, cy)

m = folium.Map(location=[clat, clon], zoom_start=11, tiles="CartoDB positron", prefer_canvas=True)
MiniMap(toggle_display=True, position="bottomleft").add_to(m)
Draw(export=False, position="topleft",
     draw_options=dict(polyline=False, circle=False, circlemarker=False, marker=False,
                       rectangle=True,
                       polygon=dict(allowIntersection=False,
                                    shapeOptions=dict(color="#ff7800", weight=2, fillOpacity=0.1))),
     edit_options=dict(edit=False)).add_to(m)

# Land
folium.GeoJson(land_json, name="Bakken Land",
               style_function=lambda _: {"fillColor": "#fff9c4", "color": "#fff9c4",
                                          "weight": 0.5, "fillOpacity": 0.15}).add_to(m)

# Section grid
gc = section_gradient
if gc != "None" and gc in sec_disp.columns:
    vals = sec_disp[gc].dropna()
    if not vals.empty:
        vn, vx = float(vals.min()), float(vals.max())
        if vn == vx:
            vx = vn + 1
        cmap = cm.LinearColormap(["#f7fcf5", "#74c476", "#00441b"], vmin=vn, vmax=vx).to_step(7)
        cmap.caption = gc
        m.add_child(cmap)
        _s = lambda f, _c=gc, _m=cmap: (
            {"fillColor": _m(f["properties"].get(_c)), "fillOpacity": 0.5,
             "color": "white", "weight": 0.3}
            if f["properties"].get(_c) is not None
            and not (isinstance(f["properties"].get(_c), float) and np.isnan(f["properties"].get(_c)))
            else NULL_STY)
    else:
        _s = lambda _: NULL_STY
else:
    _s = lambda _: NULL_STY

stf = [c for c in sec_disp.columns if c != "geometry"]
folium.GeoJson(
    sec_disp.to_json(), name="Section Grid", style_function=_s,
    highlight_function=lambda _: {"weight": 2, "color": "black", "fillOpacity": 0.55},
    tooltip=folium.GeoJsonTooltip(fields=stf, aliases=[f"{f}:" for f in stf],
                                   localize=True, sticky=True, style=TIP)).add_to(m)

# Units
folium.GeoJson(bu_json, name="Bakken Units",
               style_function=lambda _: {"color": "black", "weight": 2,
                                          "fillOpacity": 0, "dashArray": "5 3"}).add_to(m)
folium.GeoJson(hu_json, name="Handsworth Units",
               style_function=lambda _: {"color": "#d32f2f", "weight": 2.5,
                                          "fillOpacity": 0.05, "fillColor": "#ef9a9a"}).add_to(m)

# Wells bulk
wcols = [c for c in wells_disp.columns if c not in ("geometry", "_rep")]
lm = wells_disp.geometry.type.isin(["LineString", "MultiLineString"])
if lm.any():
    lw = wells_disp[lm].drop(columns=["_rep"], errors="ignore")
    for c in lw.columns:
        if c != "geometry" and lw[c].dtype == object:
            lw[c] = lw[c].astype(str)
    lj = lw.to_json()
    folium.GeoJson(lj, name="Wells (hitbox)",
                   style_function=lambda _: {"color": "transparent", "weight": 14, "opacity": 0},
                   highlight_function=lambda _: {"weight": 14, "color": "#555", "opacity": 0.3},
                   tooltip=folium.GeoJsonTooltip(
                       fields=[c for c in lw.columns if c != "geometry"],
                       aliases=[f"{c}:" for c in lw.columns if c != "geometry"],
                       localize=True, sticky=True, style=TIP)).add_to(m)
    folium.GeoJson(lj, name="Well Lines",
                   style_function=lambda _: {"color": "black", "weight": 0.5, "opacity": 0.8}).add_to(m)
    eps = wells_disp.loc[lm].geometry.apply(rep_pt).dropna()
    if not eps.empty:
        folium.GeoJson(gpd.GeoDataFrame(geometry=list(eps), crs=CRS_M).to_json(),
                       name="Well Endpoints",
                       marker=folium.CircleMarker(radius=1, color="black", fill=True,
                                                   fill_color="black", fill_opacity=.8, weight=1)).add_to(m)

pm = wells_disp.geometry.type == "Point"
if pm.any():
    pw = wells_disp[pm].drop(columns=["_rep"], errors="ignore")
    for c in pw.columns:
        if c != "geometry" and pw[c].dtype == object:
            pw[c] = pw[c].astype(str)
    folium.GeoJson(pw.to_json(), name="Well Points",
                   marker=folium.CircleMarker(radius=2, color="black", fill=True,
                                               fill_color="black", fill_opacity=.9, weight=1),
                   tooltip=folium.GeoJsonTooltip(
                       fields=[c for c in pw.columns if c != "geometry"],
                       aliases=[f"{c}:" for c in pw.columns if c != "geometry"],
                       localize=True, sticky=True, style=TIP)).add_to(m)

folium.LayerControl(collapsed=True).add_to(m)
map_data = st_folium(m, use_container_width=True, height=850, returned_objects=["all_drawings"])

# ── Polygon selection ─────────────────────────────────
st.markdown("---")
st.header("📐 Polygon Selection — Unitization Analysis")
st.caption("Draw a polygon/rectangle to evaluate waterflood potential & well inventory.")

drawings = map_data.get("all_drawings") if map_data else None

if drawings and len(drawings) > 0:
    d4 = shape(drawings[-1]["geometry"])
    d26 = shapely_transform(lambda x, y, z=None: TO26.transform(x, y), d4)
    dgdf = gpd.GeoDataFrame([{"geometry": d26}], crs=CRS_W)

    # Sections
    sec_hits = gpd.sjoin(sec_wf, dgdf, how="inner", predicate="intersects")
    # Wells
    wrep = wells_gdf[well_mask].copy()
    wrep = wrep[wrep["_rep"].notna()].set_geometry(gpd.GeoSeries(wrep["_rep"], crs=CRS_W))
    well_hits = gpd.sjoin(wrep, dgdf, how="inner", predicate="within")

    if sec_hits.empty and well_hits.empty:
        st.info("No data in polygon.")
    else:
        # Section results
        if not sec_hits.empty:
            st.subheader(f"📊 {len(sec_hits)} Sections Selected")
            c1, c2, c3, c4 = st.columns(4)
            so = sec_hits["SectionOOIP"].sum()
            si = sec_hits["WF Incremental Oil (bbl)"].sum() if "WF Incremental Oil (bbl)" in sec_hits.columns else 0
            sr = si * oil_price
            st_ = sec_hits["Total Recoverable (bbl)"].sum() if "Total Recoverable (bbl)" in sec_hits.columns else 0
            c1.metric("OOIP", f"{so:,.0f} bbl")
            c2.metric("WF Incremental Oil", f"{si:,.0f} bbl")
            c3.metric("Total Recoverable w/ WF", f"{st_:,.0f} bbl")
            c4.metric("Incremental Revenue", f"${sr:,.0f}")

            st.subheader("Waterflood Uplift Sensitivity")
            sens = []
            for p in [5, 10, 15, 20, 25, 30, 40, 50, 75]:
                tmp = add_wf(sec_hits, p)
                inc = tmp["WF Incremental Oil (bbl)"].sum() if "WF Incremental Oil (bbl)" in tmp.columns else 0
                sens.append({"Uplift %": p, "Incremental Oil (bbl)": round(inc),
                             f"Revenue @ ${oil_price:.0f}/bbl": round(inc * oil_price)})
            st.dataframe(pd.DataFrame(sens), use_container_width=True, hide_index=True)

            scols = ["Section"] + [c for c in ALL_SEC if c in sec_hits.columns]
            sdet = sec_hits[scols].reset_index(drop=True)
            with st.expander("Section Detail", expanded=False):
                st.dataframe(sdet, use_container_width=True)
            st.download_button("📥 Sections CSV", sdet.to_csv(index=False),
                               "polygon_sections.csv", "text/csv")

        # Well results
        if not well_hits.empty:
            st.subheader(f"🛢️ {len(well_hits)} Wells Selected")
            wc = ["UWI", "Section"] + WELL_NUM + WELL_CAT
            wc = [c for c in wc if c in well_hits.columns]
            wdet = well_hits[wc].reset_index(drop=True)

            wk1, wk2, wk3 = st.columns(3)
            wk1.metric("Total EUR", f"{well_hits['EUR'].sum():,.0f} bbl" if "EUR" in well_hits.columns else "—")
            wk2.metric("Total Cuml", f"{well_hits['Cuml'].sum():,.0f} bbl" if "Cuml" in well_hits.columns else "—")
            wk3.metric("Avg Norm EUR", f"{well_hits['Norm EUR'].mean():,.1f} bbl/m" if "Norm EUR" in well_hits.columns else "—")

            with st.expander("Well Detail", expanded=False):
                st.dataframe(wdet, use_container_width=True)

            wagg_cols = [c for c in WELL_NUM if c in well_hits.columns]
            wagg = pd.DataFrame({
                "Metric": wagg_cols,
                "Sum": [well_hits[c].sum() for c in wagg_cols],
                "Mean": [well_hits[c].mean() for c in wagg_cols],
                "Count": [well_hits[c].count() for c in wagg_cols],
            })
            with st.expander("Well Aggregates", expanded=False):
                st.dataframe(wagg, use_container_width=True)
            st.download_button("📥 Wells CSV", wdet.to_csv(index=False),
                               "polygon_wells.csv", "text/csv", key="dl_wells")
else:
    st.info("Draw a polygon or rectangle on the map to evaluate waterflood unitization potential.")