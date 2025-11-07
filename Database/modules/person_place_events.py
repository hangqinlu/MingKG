# -*- coding: utf-8 -*-
"""
人物 × 地点事件（陈列 + 二维地图 + 溯源）
- 事件：BirthEvent / OfficeAtPlaceEvent / TaskEvent
- 关系标签：生 / 职任 / 任务执行
- 地图底图：Carto 灰无注记
- 仅保留“人物子图”（高度 1400，标注统一用现代名称）
- 模块化入口：run(st)
- 不调用 set_page_config；兼容主入口注入的全局数据源（st.session_state['kd_data_path']）与 st.session_state['graph']。
"""

import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union

import folium
from streamlit_folium import st_folium
from rdflib import Graph, URIRef, RDF, Literal
from rdflib.namespace import RDFS, SKOS, FOAF
import streamlit as st

# ========= 常量与命名 =========
NS = "http://mingkg.org/ontology/imperial_exam.owl#"
NT_PATH_DEFAULT = r"C:\Users\卢航青\Desktop\本体结构\ontology_places_merged_postcheck_admin_1758527615.nt"

# 类名（本地名）
CLASS_PERSON = "Person"
CLASS_PLACE  = "Place"
CLASS_BIRTH  = "BirthEvent"
CLASS_OFFICE = "OfficeAtPlaceEvent"
CLASS_TASK   = "TaskEvent"

# 关系名（本地名）
OP_HAS_PLACE     = "hasPlace"
OP_ABOUT         = "about"
OP_DERIVED_FROM  = "derivedFrom"

# 数据属性（PropAssertion & TextProvenance）
DP_PA_PROP   = "prop"
DP_PA_VAL    = "value"
DP_PA_VALN   = "value_norm"
DP_TP_CONF   = "record_confidence"
DP_TP_SRC    = "Text_source"
DP_TP_BODY   = "Text_body"

# 人名显示候选
PERSON_NAME_KEYS = [
    "姓名", "name", "label", "rdfs_label", "标题", "title",
    str(FOAF.name), str(RDFS.label), str(SKOS.prefLabel)
]

# 地点名称键
PLACE_HIS_NAME   = "历史名称"
PLACE_MOD_NAME   = "现代名称"

# 溯源可见属性（本地名）
WANTED_PA_FOR_PROV = {PLACE_HIS_NAME, PLACE_MOD_NAME}

# ========= CSS（仅注入样式，不 set_page_config）=========
CSS = """
<style>
.block-container { max-width:96vw; padding-left:8px; padding-right:8px; }
.small { color:#64748b; font-size:12px; }
.hint  { color:#94a3b8; font-size:12px; }
.card  { border:1px solid #ececec; border-radius:12px; padding:12px 14px; margin:10px 0; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.04); }
.event-head { font-weight:600; margin-bottom:6px; }
.mark  { background:#fde68a; padding:0 3px; border-radius:4px; }
.badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#eef2ff; color:#0f172a; margin-left:8px; border:1px solid #e2e8f0;}
.sec-title { font-weight:600; margin:8px 0 6px; }
</style>
"""

# ========= 会话键（带前缀防冲突）=========
KEY_GRAPH        = "graph"                  # 与主入口约定
KEY_LAST_QUERY   = "__ppe_last_query__"
KEY_NAME_PREF    = "__ppe_name_pref__"
KEY_DRAW_LINES   = "__ppe_draw_lines__"
KEY_PROV_NODE    = "__ppe_prov_node__"
KEY_LOADED_FILE  = "__ppe_loaded_file__"    # 本模块沿用的“已加载文件路径”

# ========= 通用工具 =========
_ZW = {u"\u200b", u"\u200c", u"\u200d", u"\ufeff"}
def norm(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if (not ch.isspace()) and (ch not in _ZW))
    return s

def localname(u: Union[URIRef, str]) -> str:
    s = str(u)
    for sep in ("#", "/", ":"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s

def pred_tail(p: URIRef) -> str:
    return localname(p)

@st.cache_data(show_spinner=False)
def load_graph_any(path: str) -> Graph:
    p = Path(path)
    data = p.read_bytes()
    g = Graph()
    # 尽量鲁棒的解析顺序
    try:
        g.parse(data=data, format="nt")
    except Exception:
        try:
            g = Graph(); g.parse(data=data, format="turtle")
        except Exception:
            g = Graph(); g.parse(data=data)
    return g

def is_instance_of(g: Graph, inst: URIRef, class_local: str) -> bool:
    for t in g.objects(inst, RDF.type):
        if localname(t) == class_local:
            return True
    return False

def find_instances(g: Graph, class_local: str) -> List[URIRef]:
    out = []
    for s, t in g.subject_objects(RDF.type):
        if isinstance(s, URIRef) and localname(t) == class_local:
            out.append(s)
    return out

def get_literals_map(g: Graph, node: URIRef) -> Dict[str, List[str]]:
    res: Dict[str, List[str]] = {}
    for p, o in g.predicate_objects(node):
        if isinstance(o, Literal):
            k = localname(p)
            res.setdefault(k, []).append(str(o).strip())
    return res

def get_first_display(props: Dict[str, List[str]], keys: List[str]) -> str:
    for k in keys:
        lk = localname(k)
        if lk in props and props[lk]:
            return props[lk][0]
    for vs in props.values():
        for v in vs:
            if v.strip():
                return v
    return ""

def get_place_for_event(g: Graph, evt: URIRef) -> Optional[URIRef]:
    for p, o in g.predicate_objects(evt):
        if isinstance(o, URIRef) and localname(p) == OP_HAS_PLACE and is_instance_of(g, o, CLASS_PLACE):
            return o
    for p, o in g.predicate_objects(evt):
        if isinstance(o, URIRef) and is_instance_of(g, o, CLASS_PLACE):
            return o
    return None

def relation_label_by_event(g: Graph, evt: URIRef) -> str:
    if is_instance_of(g, evt, CLASS_BIRTH):
        return "生"
    if is_instance_of(g, evt, CLASS_OFFICE):
        return "职任"
    if is_instance_of(g, evt, CLASS_TASK):
        return "任务执行"
    return "事件"

# ========= 溯源（繁简体联动高亮） =========
TS_MAP = {
    "蘇":"苏","劉":"刘","張":"张","趙":"赵","錢":"钱","孫":"孙","國":"国","會":"会","試":"试","鄉":"乡",
    "進":"进","舉":"举","階":"阶","級":"级","歷":"历","鄭":"郑","黃":"黄","萬":"万","陳":"陈","楊":"杨",
    "馬":"马","許":"许","鄧":"邓","吳":"吴","葉":"叶","羅":"罗","齊":"齐","祿":"禄","祯":"祯","禎":"祯"
}
def t2s(s: str) -> str: return "".join(TS_MAP.get(ch, ch) for ch in s)
def s2t(s: str):
    inv = getattr(s2t, "_inv", None)
    if inv is None:
        inv = {v: k for k, v in TS_MAP.items()}
        s2t._inv = inv
    return "".join(inv.get(ch, ch) for ch in s)

def fuzzy_contains_name(text: str, names: List[str]) -> bool:
    if not text or not names:
        return False
    t_raw = norm(text); t_s = t2s(t_raw); t_t = s2t(t_raw)
    for name in names:
        n_raw = norm(name); n_s = t2s(n_raw); n_t = s2t(n_raw)
        if (n_raw and n_raw in t_raw) or (n_s and n_s in t_s) or (n_t and n_t in t_t):
            return True
    return False

def objects_literals(g: Graph, s: URIRef, prop_uri: str) -> List[Literal]:
    return [o for o in g.objects(s, URIRef(prop_uri)) if isinstance(o, Literal)]

def txt_list(g: Graph, s: URIRef, prop_uri: str) -> List[str]:
    vals, seen = [], set()
    for lit in objects_literals(g, s, URIRef(prop_uri)):
        v = str(lit).strip()
        if v and v not in seen:
            seen.add(v); vals.append(v)
    return vals

def prop_assertions_about(g: Graph, node: URIRef) -> List[URIRef]:
    return [pa for pa in g.subjects(URIRef(NS + OP_ABOUT), node)]

def pa_core_tuple(g: Graph, pa: URIRef) -> Tuple[str, str]:
    prop = next(iter(txt_list(g, pa, NS + DP_PA_PROP)), "")
    valn = next(iter(txt_list(g, pa, NS + DP_PA_VALN)), "")
    if valn:
        return prop, valn
    val = next(iter(txt_list(g, pa, NS + DP_PA_VAL)), "")
    return prop, val

def pa_to_provenances(g: Graph, pa: URIRef) -> List[URIRef]:
    return [tp for tp in g.objects(pa, URIRef(NS + OP_DERIVED_FROM)) if isinstance(tp, URIRef)]

def provenance_groups_for(node: URIRef, g: Graph, person_aliases: List[str]) -> List[Dict]:
    groups: Dict[Tuple[str,str], Dict] = {}
    def _n(s): return norm(s or "")
    for pa in prop_assertions_about(g, node):
        prop, val = pa_core_tuple(g, pa)
        if not prop or not val or (localname(prop) not in WANTED_PA_FOR_PROV and prop not in WANTED_PA_FOR_PROV):
            continue
        for tp in pa_to_provenances(g, pa):
            srcs  = txt_list(g, tp, NS + DP_TP_SRC)
            confs = txt_list(g, tp, NS + DP_TP_CONF)
            bodys = txt_list(g, tp, NS + DP_TP_BODY)
            src  = srcs[0] if srcs else ""
            body = bodys[0] if bodys else ""
            if not body: continue
            if not fuzzy_contains_name(body, person_aliases): continue
            key = (_n(src), _n(body))
            G = groups.setdefault(key, {"src": src or "（未知书目）", "body": body, "conf": "", "items": set()})
            if confs:
                try:
                    cur = float(G["conf"]) if G["conf"] else -1e9
                    valf = float(confs[0])
                    if valf > cur:
                        G["conf"] = confs[0]
                except Exception:
                    pass
            G["items"].add((localname(prop) if prop else "", val))
    ordered = sorted(groups.values(), key=lambda d: (0 if d["src"] else 1, d["src"], d["body"]))
    return ordered

def highlight_value(text: str, value: str) -> str:
    if not text or not value:
        return text
    t = text
    cands = list(dict.fromkeys([value, t2s(value), s2t(value)]))
    for v in cands:
        vv = v.strip()
        if vv:
            t = t.replace(vv, f"<span class='mark'>{vv}</span>")
    return t

def person_aliases(g: Graph, p: URIRef) -> List[str]:
    props = get_literals_map(g, p)
    vals = []
    for k in PERSON_NAME_KEYS:
        lk = localname(k)
        vals.extend(props.get(lk, []))
    out, seen = [], set()
    for v in vals:
        v = v.strip()
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out

# ========= 坐标解析 =========
def parse_coord(coord_str: str) -> Tuple[float,float]:
    """返回 (lng, lat)，支持 '24.8100°N, 113.5920°E' / '24.8100N,113.5920E' 等。"""
    coord_str = coord_str.replace('，', ',').replace(' ', '')
    parts = coord_str.split(',')
    if len(parts) != 2:
        raise ValueError("格式不正确")
    lat_str, lng_str = parts
    def get_num(s):
        import re as _re
        num = float(_re.findall(r"[-+]?\d*\.\d+|\d+", s)[0])
        if ('S' in s) or ('南' in s):
            num = -num
        if ('W' in s) or ('西' in s):
            num = -num
        return num
    lat = get_num(lat_str)
    lng = get_num(lng_str)
    return lng, lat  # 注意返回顺序

COORD_KEYS_MERGED = ["经纬坐标", "经纬度", "经纬", "坐标", "坐標", "coordinates"]
LAT_KEYS = ["纬度", "緯度", "latitude", "lat"]
LON_KEYS = ["经度", "經度", "longitude", "lon", "lng"]

def place_coordinates_from_rdf(g: Graph, place: URIRef) -> Optional[Tuple[float, float]]:
    if place is None: return None
    props = get_literals_map(g, place)
    for k in COORD_KEYS_MERGED:
        for v in props.get(k, []):
            try:
                lng, lat = parse_coord(v)
                return (lat, lng)  # Folium 需要 (lat, lng)
            except Exception:
                continue
    lat_val = None; lon_val = None
    for k in LAT_KEYS:
        if k in props and props[k]:
            try:
                import re as _re
                lat_val = float(_re.findall(r"[-+]?\d*\.\d+|\d+", props[k][0])[0])
            except Exception: pass
    for k in LON_KEYS:
        if k in props and k in props and props[k]:
            try:
                import re as _re
                lon_val = float(_re.findall(r"[-+]?\d*\.\d+|\d+", props[k][0])[0])
            except Exception: pass
    if lat_val is not None and lon_val is not None:
        return (lat_val, lon_val)
    return None

# ========= 搜索 =========
def search_persons(g: Graph, query: str) -> List[Tuple[str, str, str]]:
    persons = find_instances(g, CLASS_PERSON)
    qn = norm(query.strip())
    hits = []
    for p in persons:
        props = get_literals_map(g, p)
        names = []
        for k in PERSON_NAME_KEYS:
            lk = localname(k)
            names.extend(props.get(lk, []))
        names = [n for n in names if n.strip()]
        if not names:
            continue
        reason = ""
        for v in names:
            if qn and norm(v) == qn:
                reason = f"[等值命中] {v}"; break
        if not reason:
            for v in names:
                if qn and qn in norm(v):
                    reason = f"[包含命中] {v}"; break
        if reason:
            hits.append((str(p), names[0], reason))
    hits.sort(key=lambda t: t[1])
    return hits[:30]

# ========= 回调 =========
def set_prov(node_iri: str, prop_name: str, prop_value: str, aliases: List[str]):
    st.session_state[KEY_PROV_NODE] = ("place_or_event", node_iri, prop_name, prop_value, aliases)

# ========= 新增：与入口对接的全局加载器（方案 A）=========
def _ensure_graph_from_global() -> Tuple[Optional[Graph], bool]:
    """
    若主入口已设置 kd_data_path，则自动加载 Graph 并写入：
      - st.session_state['graph'] / KEY_GRAPH
      - st.session_state[KEY_LOADED_FILE]
    返回 (Graph or None, 是否本次重新加载)
    """
    kd_path = (st.session_state.get("kd_data_path") or "").strip()
    if not kd_path:
        return None, False
    cur_loaded = st.session_state.get(KEY_LOADED_FILE, "")
    g = st.session_state.get(KEY_GRAPH)
    if g is not None and cur_loaded == kd_path:
        return g, False
    try:
        g = load_graph_any(kd_path)
        st.session_state[KEY_GRAPH] = g
        st.session_state[KEY_LOADED_FILE] = kd_path
        return g, True
    except Exception as e:
        st.error(f"全局数据源加载失败：{e}")
        return None, False

# ========= 对外入口 =========
def run(st):
    st.markdown(CSS, unsafe_allow_html=True)
    st.subheader("🗺️ 科举人物迁移路线检索")

    # 优先复用主入口的全局数据源（kd_data_path）
    g, reloaded = _ensure_graph_from_global()

    # 侧栏：仅当未设置全局数据源时，才显示本模块的手动加载 UI；其余参数仍常驻
    with st.sidebar:
        st.header("📁 数据")
        if g is None:
            data_default = st.session_state.get(KEY_LOADED_FILE, NT_PATH_DEFAULT)
            data_file = st.text_input("RDF 路径（.nt/.ttl）", value=data_default, key="__ppe_path_input__")
            if st.button("加载数据", type="primary", use_container_width=True, key="__ppe_btn_load__"):
                try:
                    g2 = load_graph_any(data_file)
                    st.session_state[KEY_GRAPH] = g2
                    st.session_state[KEY_LOADED_FILE] = data_file
                    st.success(f"已加载 RDF：{data_file}（triples={len(g2)}）")
                except Exception as e:
                    st.error(f"RDF 加载失败：{e}")
        else:
            st.success(f"已连接全局数据源：{st.session_state.get(KEY_LOADED_FILE,'')}")
            if reloaded:
                st.info("已根据全局数据源自动加载。")

        name_pref = st.radio("地点展示名称", ["历史名称优先", "现代名称优先"],
                             index=0 if st.session_state.get(KEY_NAME_PREF, "历史")=="历史" else 1, key="__ppe_namepref__")
        draw_lines = st.checkbox("地图绘制迁移折线", value=st.session_state.get(KEY_DRAW_LINES, True), key="__ppe_draw__")
        st.session_state[KEY_NAME_PREF] = "历史" if name_pref=="历史名称优先" else "现代"
        st.session_state[KEY_DRAW_LINES] = draw_lines

    g = st.session_state.get(KEY_GRAPH)
    if not g:
        st.info("👈 先在侧栏加载 RDF 数据文件，或在主入口应用全局数据源后自动复用。")
        return

    # 顶部搜索（8/4）
    c1, c2 = st.columns([8,4])
    with c1:
        q = st.text_input("输入人名关键字（繁简均可：蘇國瓓 / 苏国瓓）", value=st.session_state.get(KEY_LAST_QUERY, ""))
    with c2:
        if st.button("🔎 搜索", type="primary", use_container_width=True):
            st.session_state[KEY_LAST_QUERY] = q

    # 左右 8/4
    left, right = st.columns([8,4], gap="large")

    with left:
        hits = []
        if st.session_state.get(KEY_LAST_QUERY, "").strip():
            hits = search_persons(g, st.session_state[KEY_LAST_QUERY])

        if not hits:
            st.info("输入人名并点击“搜索”。")
            return

        st.success(f"命中 {len(hits)} 人（展示前 {len(hits)}）")
        st.divider()

        prefer_hist = (st.session_state.get(KEY_NAME_PREF, "历史") == "历史")
        draw_lines_flag = bool(st.session_state.get(KEY_DRAW_LINES, True))

        for idx, (person_iri, _, _) in enumerate(hits, 1):
            person = URIRef(person_iri)
            props = get_literals_map(g, person)
            pname = get_first_display(props, PERSON_NAME_KEYS) or localname(person)
            aliases = person_aliases(g, person)

            # 收集地点事件
            events: Set[URIRef] = set()
            for p_, e in g.predicate_objects(person):
                if isinstance(e, URIRef) and (is_instance_of(g, e, CLASS_BIRTH) or is_instance_of(g, e, CLASS_OFFICE) or is_instance_of(g, e, CLASS_TASK)):
                    events.add(e)
            for e, p_ in g.subject_predicates(person):
                if isinstance(e, URIRef) and (is_instance_of(g, e, CLASS_BIRTH) or is_instance_of(g, e, CLASS_OFFICE) or is_instance_of(g, e, CLASS_TASK)):
                    events.add(e)
            if not events:
                st.markdown(f"### 👤 {pname}  <span class='hint'>（无地点相关事件）</span>", unsafe_allow_html=True)
                st.markdown("---")
                continue

            # 陈列分桶
            type_buckets: Dict[str, List[URIRef]] = {"生": [], "职任": [], "任务执行": [], "事件": []}
            for evt in events:
                type_buckets[relation_label_by_event(g, evt)].append(evt)

            st.markdown(f"### 👤 {pname} <span class='hint'>（{person_iri}）</span>", unsafe_allow_html=True)

            # 逐桶陈列 & 收集地图点
            points: List[Tuple[float,float,str,str,str]] = []  # (lat, lng, rel, his, mod)
            for rel in ["生", "职任", "任务执行", "事件"]:
                evts = sorted(type_buckets.get(rel, []), key=lambda x: str(x))
                if not evts: continue
                with st.expander(f"📂 {rel}（{len(evts)}）", expanded=False):
                    for i, evt in enumerate(evts):
                        place = get_place_for_event(g, evt)
                        place_props = get_literals_map(g, place) if place else {}
                        his = (place_props.get(PLACE_HIS_NAME, [""]) or [""])[0] if place_props else ""
                        mod = (place_props.get(PLACE_MOD_NAME, [""]) or [""])[0] if place_props else ""
                        disp_text = (his or mod) if prefer_hist else (mod or his)
                        disp_text = disp_text or "（未标注地点）"
                        eid_short = localname(evt)

                        st.markdown(
                            f"<div class='card'><div class='event-head'>• 事件：{eid_short} <span class='badge'>{rel}</span></div><div>地点：{disp_text}</div>",
                            unsafe_allow_html=True
                        )

                        # 溯源 Chip
                        btn_key_h = f"chip_his_{hash((person_iri, str(evt), 'his', his))}"
                        btn_key_m = f"chip_mod_{hash((person_iri, str(evt), 'mod', mod))}"
                        col_a, col_b = st.columns(2)
                        with col_a:
                            if his:
                                st.button(f"历史名称：{his}", key=btn_key_h, on_click=set_prov,
                                          args=(str(place) if place else str(evt), PLACE_HIS_NAME, his, aliases),
                                          use_container_width=True)
                        with col_b:
                            if mod:
                                st.button(f"现代名称：{mod}", key=btn_key_m, on_click=set_prov,
                                          args=(str(place) if place else str(evt), PLACE_MOD_NAME, mod, aliases),
                                          use_container_width=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                        # 坐标入图
                        coords = place_coordinates_from_rdf(g, place) if place else None
                        if coords:
                            lat, lng = coords
                            points.append((lat, lng, rel, his, mod))

            # 人物子图（仅此；高度 1400；标注现代名称）
            if points:
                sub_center = [points[0][0], points[0][1]]
                m_sub = folium.Map(location=sub_center, zoom_start=6, tiles=None, control_scale=True)
                folium.TileLayer(
                    tiles="https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
                    attr='&copy; <a href="https://carto.com/">CARTO</a>',
                    name="Carto Positron (No Labels)",
                    overlay=False, control=False
                ).add_to(m_sub)

                coords_line = []
                for (lat, lng, rel, his, mod) in points:
                    modern_name = (mod or his or "（未标注现代名称）")
                    html = f"""
                    <div style='font-size:13px;line-height:1.6;min-width:140px;max-width:260px;'>
                      <b style='color:#205493;'>人物：</b>{pname}<br>
                      <b style='color:#205493;'>关系：</b>{rel}<br>
                      <b style='color:#205493;'>现代名称：</b>{modern_name}<br>
                      <span style='color:#888;'>坐标：</span>{lat:.4f}, {lng:.4f}
                    </div>
                    """
                    folium.CircleMarker(
                        [lat, lng], radius=4, color="#4b5563", weight=1, fill=True, fill_opacity=0.85
                    ).add_to(m_sub)
                    mk = folium.Marker(
                        [lat, lng],
                        icon=folium.Icon(icon='map-marker', prefix='fa', color='gray'),
                        popup=folium.Popup(html, max_width=300, min_width=160, show=False)
                    )
                    folium.Tooltip(f"{pname} · {rel} · {modern_name}").add_to(mk)
                    mk.add_to(m_sub)
                    coords_line.append([lat, lng])

                if draw_lines_flag and len(coords_line) >= 2:
                    folium.PolyLine(coords_line, color="#4b5563", weight=3, opacity=0.7).add_to(m_sub)

                st.markdown("<div class='sec-title'>🗺️ 人物子图（标注：现代名称）</div>", unsafe_allow_html=True)
                st_folium(m_sub, width=None, height=1400)

            st.markdown("---")

    with right:
        st.markdown("### 溯源")
        sel = st.session_state.get(KEY_PROV_NODE)
        if not sel:
            st.info("点击左侧任一『历史名称 / 现代名称』Chip 查看溯源。只展示包含该人物‘姓名/字号’的摘录，并对所点值做繁简体联动高亮。")
        else:
            _type, node_iri, prop_name, prop_value, aliases = sel
            node = URIRef(node_iri)
            st.markdown(f"**定位**：地点 / 事件节点  |  **断言**：{prop_name} = {prop_value}")

            groups = provenance_groups_for(node, g, aliases)
            filtered = [G for G in groups if (prop_name, prop_value) in G["items"]]
            if not filtered:
                st.warning("未找到与此值直接匹配且含人名的溯源。（可能该值暂无溯源或溯源文本未含人名/字号）")
            else:
                for i, G in enumerate(filtered, 1):
                    with st.expander(f"#{i} 书目：{G['src']}", expanded=(i==1)):
                        st.caption("证明： " + "； ".join([f"{p} = {v}" for p, v in sorted(G["items"])]))
                        if G["conf"]:
                            st.write(f"**可信度：** {G['conf']}")
                        st.write("**摘录（自动高亮，含繁简体）**：", unsafe_allow_html=True)
                        st.markdown(highlight_value(G["body"], prop_value), unsafe_allow_html=True)
