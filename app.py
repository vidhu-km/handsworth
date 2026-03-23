import streamlit as st
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
from folium.plugins import MiniMap, Draw
from streamlit_folium import st_folium
import branca.colormap as cm
from shapely.geometry import shape, Point
from shapely.ops import transform as shapely_transform
from pyproj import Transformer

st.set_page_config(layout="wide", page_title="Bakken WF Section Screener", page_icon="🛢️")

TIP = "font-size:11px;padding:3px 6px;background:rgba(255,255,255,.92);border:1px solid #333;border-radius:3px;"
NULL_STY = {"fillColor": "#fff", "fillOpacity": 0, "color": "#888", "weight": 0.25}
CRS_W = "EPSG:26913"
CRS_M = "EPSG:4326"
TO4 = Transformer.from_crs(CRS_W, CRS_M, always_xy=True)
TO26 = Transformer.from_crs(CRS_M, CRS_W, always_xy=True)

WELL_NUM = ["Hz Length (m)", "Cuml", "EUR"]
WELL_CAT = ["Well Type", "Status", "Objective", "Injector", "Operator"]
HEEL_TOL = 1.0

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER REGISTRY — add / remove / reorder / edit layers here.
#
# Each entry is a dict with these keys:
#   "name"      : layer name in the layer control
#   "file"      : shapefile path
#   "style"     : dict passed to style_function (static style)
#   "simplify"  : optional tolerance (metres) to simplify geometries
#   "tooltip"   : True to add a tooltip with all attribute fields (default False)
#
# Layers render in the order listed (first = bottom).
# ═══════════════════════════════════════════════════════════════════════════════
OVERLAY_LAYERS = [
    # ── 1. Bakken Land ────────────────────────────────────────────────────────
    {
        "name": "Bakken Land",
        "file": "Bakken Land.shp",
        "style": {
            "fillColor": "#fff9c4",
            "color": "#fff9c4",
            "weight": 0.5,
            "fillOpacity": 0.15,
        },
        "simplify": 50,
    },
    # ── 2. Bakken Units ──────────────────────────────────────────────────────
    {
        "name": "Bakken Units",
        "file": "Bakken Units.shp",
        "style": {
            "color": "black",
            "weight": 3,
            "fillOpacity": 0,
        },
    },
    # ── 3. Handsworth Units ──────────────────────────────────────────────────
    {
        "name": "Handsworth Units",
        "file": "Handsworth Units.shp",
        "style": {
            "color": "black",
            "weight": 3,
            "fillOpacity": 0,
        },
    },
    # ── 4. Bakken Sw Line ────────────────────────────────────────────────────
    {
        "name": "Bakken Sw Line",
        "file": "Bakken Sw Line_plyln.shp",
        "style": {
            "color": "#0000ff",
            "weight": 2,
            "fillOpacity": 0,
        },
        "tooltip": True,
    },
    # ── 5. T1 and T2 Contours ───────────────────────────────────────────────
    {
        "name": "T1 and T2 Contours",
        "file": "T1 and T2 Contours_expanded_plyln.shp",
        "style": {
            "color": "#ff0000",
            "weight": 1.5,
            "fillOpacity": 0,
            "dashArray": "6 4",
        },
        "tooltip": True,
    },
    # ── COPY-PASTE TEMPLATE (uncomment & fill in) ────────────────────────────
    # {
    #     "name": "My New Layer",
    #     "file": "my_layer.shp",
    #     "style": {
    #         "color": "purple",
    #         "weight": 2,
    #         "fillOpacity": 0,
    #     },
    #     "simplify": None,
    #     "tooltip": True,
    # },
]
# ═══════════════════════════════════════════════════════════════════════════════


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
    existing = gpd.read_file("lines.shp")
    inventory = gpd.read_file("inventory.shp")
    existing["_source"] = "existing"
    inventory["_source"] = "inventory"
    lines = gpd.GeoDataFrame(
        pd.concat([existing, inventory], ignore_index=True),
        geometry="geometry",
    )

    points = gpd.read_file("points.shp")
    grid = gpd.read_file("ooipsectiongrid.shp")

    # ── Load overlay layers from the registry ─────────
    overlay_jsons = {}
    for layer_def in OVERLAY_LAYERS:
        lname = layer_def["name"]
        try:
            gdf = gpd.read_file(layer_def["file"])
            if gdf.crs is None:
                gdf.set_crs(CRS_W, inplace=True)
            gdf.to_crs(CRS_W, inplace=True)
            simp = layer_def.get("simplify")
            if simp:
                gdf["geometry"] = gdf.geometry.simplify(simp, preserve_topology=True)
            overlay_jsons[lname] = gdf.to_crs(CRS_M).to_json()
        except Exception as e:
            st.warning(f"Could not load layer '{lname}': {e}")
            overlay_jsons[lname] = None

    # ── Well attribute table (single-sheet xlsx) ──────
    wdf = pd.read_excel("wells.xlsx")

    for g in [lines, points, grid]:
        if g.crs is None:
            g.set_crs(CRS_W, inplace=True)
        g.to_crs(CRS_W, inplace=True)

    grid["geometry"] = grid.geometry.simplify(50, preserve_topology=True)

    grid["Section"] = grid["Section"].astype(str).str.strip()
    wdf["UWI"] = wdf["UWI"].astype(str).str.strip()
    wdf["Section"] = wdf["Section"].astype(str).str.strip()
    lines["UWI"] = lines["UWI"].astype(str).str.strip()
    points["UWI"] = points["UWI"].astype(str).str.strip()

    for c in ["Cuml", "EUR"]:
        if c in wdf.columns:
            wdf[c] = pd.to_numeric(wdf[c], errors="coerce")

    # ── OOIP comes from the grid shapefile attribute ──
    if "OOIP" in grid.columns:
        grid["SectionOOIP"] = pd.to_numeric(grid["OOIP"], errors="coerce").fillna(0)
    else:
        grid["SectionOOIP"] = 0.0

    # ── assemble all well geometries (keep _source tag) ──
    pts_only = points[~points["UWI"].isin(lines["UWI"])][["UWI", "geometry"]].copy()
    pts_only["_source"] = "existing"

    all_geom = gpd.GeoDataFrame(
        pd.concat(
            [lines[["UWI", "geometry", "_source"]],
             pts_only],
            ignore_index=True,
        ),
        geometry="geometry", crs=CRS_W,
    )

    # ── separate lines vs points ──────────────────────
    line_mask = all_geom.geometry.geom_type.isin(["LineString", "MultiLineString"])
    legs = all_geom[line_mask].copy().reset_index(drop=True)
    pt_wells = all_geom[~line_mask].copy().reset_index(drop=True)

    # ── group multilaterals by heel ───────────────────
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

    # ── group attribute table (from 00 UWI row) ──────
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

    # ── spatial overlay: legs × section grid ──────────
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

    # ── build section GeoDataFrame ────────────────────
    sec = grid.copy()
    sec = sec.merge(sec_alloc, on="Section", how="left")

    for c in ["SectionCuml", "SectionEUR"]:
        if c in sec.columns:
            sec[c] = sec[c].fillna(0)

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

    # ── wells display GeoDataFrame ────────────────────
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

    return wells_final, sec, overlay_jsons


wells_gdf, sec_gdf, overlay_jsons = load()

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

# ── Map ───────────────────────────────────────────────
bnds = sec_gdf.total_bounds
cx, cy = (bnds[0] + bnds[2]) / 2, (bnds[1] + bnds[3]) / 2
clon, clat = TO4.transform(cx, cy)

m = folium.Map(
    location=[clat, clon], zoom_start=11,
    tiles="CartoDB positron", prefer_canvas=True,
)
MiniMap(toggle_display=True, position="bottomleft").add_to(m)
Draw(
    export=False, position="topleft",
    draw_options=dict(
        polyline=False, circle=False, circlemarker=False, marker=False,
        rectangle=True,
        polygon=dict(
            allowIntersection=False,
            shapeOptions=dict(color="#ff7800", weight=2, fillOpacity=0.1),
        ),
    ),
    edit_options=dict(edit=False),
).add_to(m)

# ── Overlay layers (rendered in registry order) ──────
for layer_def in OVERLAY_LAYERS:
    lname = layer_def["name"]
    lj = overlay_jsons.get(lname)
    if lj is None:
        continue
    style = layer_def["style"]
    kwargs = dict(
        name=lname,
        style_function=lambda _, _s=style: _s,
    )
    if layer_def.get("tooltip"):
        # Parse geojson to get field names for tooltip
        import json as _json
        _parsed = _json.loads(lj)
        _props = _parsed.get("features", [{}])[0].get("properties", {})
        _fields = [k for k in _props.keys()]
        if _fields:
            kwargs["tooltip"] = folium.GeoJsonTooltip(
                fields=_fields,
                aliases=[f"{f}:" for f in _fields],
                localize=True, sticky=True, style=TIP,
            )
    folium.GeoJson(lj, **kwargs).add_to(m)

# ── Section grid colouring ────────────────────────────
gc = section_gradient
if gc != "None" and gc in sec_disp.columns:
    vals = sec_disp[gc].dropna()
    if not vals.empty:
        vn, vx = float(vals.min()), float(vals.max())
        if vn == vx:
            vx = vn + 1
        cmap = cm.LinearColormap(
            ["#f7fcf5", "#74c476", "#00441b"], vmin=vn, vmax=vx,
        ).to_step(7)
        cmap.caption = gc
        m.add_child(cmap)

        def _s(f, _c=gc, _m=cmap):
            v = f["properties"].get(_c)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                return {
                    "fillColor": _m(v), "fillOpacity": 0.5,
                    "color": "white", "weight": 0.3,
                }
            return NULL_STY
    else:
        _s = lambda _: NULL_STY
else:
    _s = lambda _: NULL_STY

stf = [c for c in sec_disp.columns if c not in ("geometry", "_source")]
folium.GeoJson(
    sec_disp.to_json(), name="Section Grid", style_function=_s,
    highlight_function=lambda _: {
        "weight": 2, "color": "black", "fillOpacity": 0.55,
    },
    tooltip=folium.GeoJsonTooltip(
        fields=stf, aliases=[f"{f}:" for f in stf],
        localize=True, sticky=True, style=TIP,
    ),
).add_to(m)

# ── Wells on map (split by source: existing=black, inventory=red) ──
ttip_exclude = {"geometry", "_rep", "_source"}
tip_cols_all = [c for c in wells_disp.columns if c not in ttip_exclude]

is_inv = wells_disp["_source"] == "inventory"
wells_existing = wells_disp[~is_inv]
wells_inventory = wells_disp[is_inv]

for subset, color, layer_suffix in [
    (wells_existing, "black", "Existing"),
    (wells_inventory, "red", "Inventory"),
]:
    if subset.empty:
        continue

    layer_group = folium.FeatureGroup(name=f"Wells ({layer_suffix})")

    lm = subset.geometry.geom_type.isin(["LineString", "MultiLineString"])

    if lm.any():
        lw = subset[lm].drop(columns=["_rep", "_source"], errors="ignore").copy()

        for c in lw.columns:
            if c != "geometry" and lw[c].dtype == object:
                lw[c] = lw[c].astype(str)

        lj = lw.to_json()
        tip_fields = [c for c in lw.columns if c != "geometry"]

        folium.GeoJson(
            lj,
            style_function=lambda _, _col=color: {
                "color": "transparent",
                "weight": 14,
                "opacity": 0,
            },
            highlight_function=lambda _, _col=color: {
                "weight": 14,
                "color": _col,
                "opacity": 0.3,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tip_fields,
                aliases=[f"{c}:" for c in tip_fields],
                localize=True,
                sticky=True,
                style=TIP,
            ),
        ).add_to(layer_group)

        folium.GeoJson(
            lj,
            style_function=lambda _, _col=color: {
                "color": _col,
                "weight": 0.5,
                "opacity": 0.8,
            },
        ).add_to(layer_group)

        eps = subset.loc[lm.values, "_rep"].dropna()
        if not eps.empty:
            folium.GeoJson(
                gpd.GeoDataFrame(geometry=list(eps), crs=CRS_M).to_json(),
                marker=folium.CircleMarker(
                    radius=1,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=0.8,
                    weight=1,
                ),
            ).add_to(layer_group)

    layer_group.add_to(m)

    pm_mask = subset.geometry.geom_type == "Point"
    if pm_mask.any():
        pw = subset[pm_mask].drop(columns=["_rep", "_source"], errors="ignore").copy()
        for c in pw.columns:
            if c != "geometry" and pw[c].dtype == object:
                pw[c] = pw[c].astype(str)
        tip_fields_p = [c for c in pw.columns if c != "geometry"]
        folium.GeoJson(
            pw.to_json(), name=f"Well Points ({layer_suffix})",
            marker=folium.CircleMarker(
                radius=2, color=color, fill=True,
                fill_color=color, fill_opacity=0.9, weight=1,
            ),
            tooltip=folium.GeoJsonTooltip(
                fields=tip_fields_p,
                aliases=[f"{c}:" for c in tip_fields_p],
                localize=True, sticky=True, style=TIP,
            ),
        ).add_to(m)

folium.LayerControl(collapsed=True).add_to(m)
map_data = st_folium(
    m, use_container_width=True, height=850,
    returned_objects=["all_drawings"],
)

# ── Polygon selection ─────────────────────────────────
st.markdown("---")
st.header("📐 Polygon Selection — Section Analysis")
st.caption(
    "Draw a polygon/rectangle to evaluate potential waterflood uplift."
)

drawings = map_data.get("all_drawings") if map_data else None

if drawings and len(drawings) > 0:
    polys = [shape(d["geometry"]) for d in drawings if d["geometry"]["type"] == "Polygon"]

    if len(polys) > 0:
        from shapely.ops import unary_union

        combined = unary_union(polys)

        d26 = shapely_transform(
            lambda x, y, z=None: TO26.transform(x, y),
            combined
        )

        dgdf = gpd.GeoDataFrame([{"geometry": d26}], crs=CRS_W)

    # Sections — intersects
    sec_hits = gpd.sjoin(sec_wf, dgdf, how="inner", predicate="intersects")
    sec_hits = sec_hits.drop(columns=["index_right"], errors="ignore")

    # Wells — intersects
    well_hits = gpd.sjoin(
        wells_gdf[well_mask], dgdf, how="inner", predicate="intersects",
    )
    well_hits = well_hits.drop(columns=["index_right"], errors="ignore")

    well_unique = (
        well_hits.drop_duplicates(subset="Well")
        if "Well" in well_hits.columns
        else well_hits
    )

    if sec_hits.empty and well_hits.empty:
        st.info("No data in polygon.")
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

            scols = ["Section"] + [
                c for c in ALL_SEC if c in sec_hits.columns
            ]
            sdet = sec_hits[scols].reset_index(drop=True)
            with st.expander("Section Detail", expanded=False):
                st.dataframe(sdet, use_container_width=True)
            st.download_button(
                "📥 Sections CSV", sdet.to_csv(index=False),
                "polygon_sections.csv", "text/csv",
            )
else:
    st.info(
        "Draw a polygon or rectangle on the map to evaluate "
        "waterflood unitization potential."
    )