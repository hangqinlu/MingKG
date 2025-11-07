# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
from collections import defaultdict, deque
import re, html, json, os, math
import numpy as np

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from rdflib import Graph, URIRef, RDF, Literal

# Folium（GIS）
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ========= 兼容性：rerun =========
def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# —— 繁简 —— #
try:
    from opencc import OpenCC
    _CC_T2S = OpenCC("t2s"); _CC_S2T = OpenCC("s2t")
    def to_s(s: str) -> str: return _CC_T2S.convert(s or "")
    def to_t(s: str) -> str: return _CC_S2T.convert(s or "")
except Exception:
    def to_s(s: str) -> str: return s or ""
    def to_t(s: str) -> str: return s or ""

PLOT_FONT = dict(family="SimHei, Microsoft YaHei, Arial Unicode MS, Noto Sans CJK SC", size=14)

# ===== 路径（按需替换） =====
DEFAULT_DATA           = r"C:\Users\卢航青\Desktop\owl_inspect_20250925_171447\ontology_places_merged.nt"
PROVINCE_GEOJSON       = r"C:\Users\卢航青\Desktop\geo\china_provinces.geojson"   # 省面
CITY_GEOJSON           = r"C:\Users\卢航青\Desktop\geo\china_cities.geojson"      # 地级市面

# ===== 本体要素 =====
CLASS_PERSON = "Person"
CLASS_PLACE  = "Place"
CLASS_BIRTH  = "BirthEvent"
CLASS_OFFICE = "OfficeAtPlaceEvent"
CLASS_TASK   = "TaskEvent"
CLASS_PARTICIPATION = "ParticipationEvent"
CLASS_IMPERIAL_EXAM = "ImperialExam"
CLASS_PROPASSERTION = "PropAssertion"
CLASS_TEXTPROV      = "TextProvenance"

OP_SUB_PLACE     = "isSubPlaceOf"
OP_HAS_PLACE     = "hasPlace"
OP_TOOK_PLACE_AT = "tookPlaceAt"
OP_HAPPENED_IN   = "happenedIn"
PLACE_PRED_NAMES = {OP_HAS_PLACE, OP_TOOK_PLACE_AT, OP_HAPPENED_IN}

OP_PARTICIPATES  = "participatesIn"
OP_HAS_EXAM      = "hasExam"

DP_PLACE_MODERN  = "现代名称"
DP_PLACE_HIST    = "历史名称"
DP_ADMIN_LEVEL   = "现代区划层级"

PERSON_NAME_KEYS = ["姓名","name","label","rdfs_label","标题","title",
                    "http://xmlns.com/foaf/0.1/name",
                    "http://www.w3.org/2000/01/rdf-schema#label",
                    "http://www.w3.org/2004/02/skos/core#prefLabel"]

DP_ACADEMIC   = "学术专长"
DP_HUJI_TYPE  = "户籍类型"

NS = "http://mingkg.org/ontology/imperial_exam.owl#"
P_ABOUT = URIRef(NS + "about")
P_PROP  = URIRef(NS + "prop")
P_VALN  = URIRef(NS + "value_norm")
P_VAL   = URIRef(NS + "value")
PA_SCHOOL_TYPE_KEYS = {"学籍类型","學籍類型","学籍_类型","學籍_類型","类型","類型"}

NS_ABOUT        = "about"
NS_DERIVED_FROM = "derivedFrom"
NS_CONTAINS     = "contains"
NS_TP_BODY      = "Text_body"
NS_TP_CONF      = "record_confidence"
NS_TP_SOURCE_KEYS = ["Text_Source","Text_source","来源","书名","source","Source","book","Book","Title","题名"]

LEVEL_MAP = {
    "省": {"省","特别行政区","自治区"},
    "市": {"市","州","盟","地区"},
    "县/区/旗": {"县","区","旗","林区","特区","新区","市辖区"},
    "乡镇": {"乡","镇","街道","民族乡","苏木"},
    "村/社区": {"社区","村","嘎查","嘎查村","居委会","村委会"},
}
NEXT_LEVEL = {"省":"市","市":"县/区/旗","县/区/旗":"乡镇","乡镇":"村/社区","村/社区":None}
PARENT_LEVEL = {"省":None,"市":"省","县/区/旗":"市","乡镇":"县/区/旗","村/社区":"乡镇"}

# —— 中国 bbox —— #
CHINA_BOUNDS = dict(min_lat=18.0, max_lat=54.5, min_lon=73.0, max_lon=135.5)

# ===== 年号 =====
MING = [("洪武",1368,1398),("建文",1398,1402),("永乐",1403,1424),("洪熙",1425,1425),
        ("宣德",1426,1435),("正统",1436,1449),("景泰",1450,1456),("天顺",1457,1464),
        ("成化",1465,1487),("弘治",1488,1505),("正德",1506,1521),("嘉靖",1522,1566),
        ("隆庆",1567,1572),("万历",1573,1620),("泰昌",1620,1620),("天启",1621,1627),
        ("崇祯",1628,1644)]
QING = [("顺治",1644,1661),("康熙",1662,1722),("雍正",1735,1735),("乾隆",1736,1795),
        ("嘉庆",1796,1820),("道光",1821,1850),("咸丰",1851,1861),("同治",1862,1874),
        ("光绪",1875,1908),("宣统",1909,1911)]
ERA_LIST = MING + QING
ERA_ORDER = [e[0] for e in ERA_LIST]

# ===== 工具 =====
def localname(u: Union[URIRef, str]) -> str:
    s = str(u or "")
    for sep in ("#", "/", ":"):
        if sep in s: s = s.rsplit(sep, 1)[-1]
    return s

def is_instance_of(g: Graph, inst: URIRef, class_local: str) -> bool:
    for t in g.objects(inst, RDF.type):
        if localname(t) == class_local:
            return True
    return False

def get_literals(g: Graph, node: URIRef) -> Dict[str, List[str]]:
    res: Dict[str, List[str]] = {}
    for p, o in g.predicate_objects(node):
        if isinstance(o, Literal):
            res.setdefault(localname(p), []).append(str(o).strip())
    return res

def first_lit(props: Dict[str, List[str]], keys: List[str], default="") -> str:
    for k in keys:
        if k in props and props[k]:
            for v in props[k]:
                if v: return v
    for vs in props.values():
        for v in vs:
            if v: return v
    return default

def tokenize_admin_name(full: str) -> List[str]:
    if not full: return []
    s = re.sub(r"[·•·\s、，,；;]+","", full)
    sufs = ["特别行政区","自治区","自治州","自治县","地区","市辖区",
            "省","市","州","盟","县","区","旗","乡","镇","街道","民族乡","苏木",
            "社区","村","嘎查","嘎查村","居委会","村委会"]
    sufs = sorted(set(sufs), key=len, reverse=True)
    out, buf = [], ""
    for ch in s:
        buf += ch
        for suf in sufs:
            if buf.endswith(suf):
                out.append(buf); buf=""; break
    if buf: out.append(buf)
    return out

def canon_level_suffix(s: str) -> str:
    return {"市辖区":"区","自治县":"县","自治州":"州","自治旗":"旗"}.get((s or "").strip(), (s or "").strip())

def level_from_name(modern: str) -> str:
    segs = tokenize_admin_name(modern)
    if not segs: return ""
    last = segs[-1]
    for suf in ["市辖区","自治县","自治州","自治旗","特别行政区","自治区",
                "省","市","州","盟","县","区","旗","乡","镇","街道","民族乡","苏木",
                "社区","村","嘎查","嘎查村","居委会","村委会"]:
        if last.endswith(suf): return canon_level_suffix(suf)
    return ""

def highlight_html(text: str, terms: List[str]) -> str:
    if not text: return ""
    t = text
    uniq = sorted({*terms, *[to_s(x) for x in terms], *[to_t(x) for x in terms]}, key=len, reverse=True)
    for v in uniq:
        if v: t = t.replace(v, f"<span style='background:#fde68a;padding:0 2px;border-radius:3px'>{v}</span>")
    return t

def era_of_year(y: int) -> Optional[str]:
    for n,a,b in ERA_LIST:
        if a <= y <= b: return n
    return None

# ===== 底图配置 =====
def basemap_conf(name: str):
    tiles = {
        "Carto·灰（全注记）": (
            "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
            "© OpenStreetMap © CARTO", ["a","b","c","d"]),
        "Carto·灰（无注记）": (
            "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
            "© OpenStreetMap © CARTO", ["a","b","c","d"]),
        "Carto·Voyager（无注记）": (
            "https://{s}.basemaps.cartocdn.com/rastertiles/voyager_nolabels/{z}/{x}/{y}{r}.png",
            "© OpenStreetMap © CARTO", ["a","b","c","d"]),
        "Esri·World Gray Canvas": (
            "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            "Tiles © Esri — Esri, DeLorme, NAVTEQ", None),
        "OpenTopoMap": (
            "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
            "© OpenStreetMap contributors, SRTM | © OpenTopoMap (CC-BY-SA)", ["a","b","c"]),
        "GeoQ·蓝黑": (
            "https://map.geoq.cn/ArcGIS/rest/services/ChinaOnlineStreetPurplishBlue/MapServer/tile/{z}/{y}/{x}",
            "© GeoQ & OpenStreetMap", None),
        "高德矢量": (
            "https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=7&x={x}&y={y}&z={z}",
            "© 高德地图", ["1","2","3","4"]),
        "高德影像": (
            "https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}",
            "© 高德影像", ["1","2","3","4"]),
        "无底图（纯白，仅边界）": ("", " ", None),
        "高德矢量（无注记）": (
            "https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png",
            "© OpenStreetMap © CARTO", ["a","b","c","d"]),
    }
    return tiles.get(name, tiles["Carto·灰（无注记）"])

# ===== Mercator =====
def _mercator_y(lat_deg: float) -> float:
    lat = max(-85.05112878, min(85.05112878, float(lat_deg)))
    rad = math.radians(lat)
    return math.log(math.tan(math.pi/4.0 + rad/2.0))

# ===== 坐标解析 =====
LAT_KEYS={"纬度","lat","latitude","Lat","Latitude","N"}
LON_KEYS={"经度","lon","longitude","Lon","Longitude","E","Lng","lng","Long"}
_re_lat=re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*([NS])?", re.I)
_re_lon=re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*([EW])?", re.I)
_re_pair=re.compile(r"(-?\d+(?:\.\d+)?)\s*°?\s*[NS]?\s*[,，]\s*(-?\d+(?:\.\d+)?)\s*°?\s*[EW]?", re.I)
def parse_lat_lon_from_props(props: Dict[str, List[str]]) -> Tuple[Optional[float], Optional[float]]:
    lat_val, lon_val=None, None
    for _, vals in props.items():
        for v in vals:
            m=_re_pair.search(v)
            if m:
                a,b=float(m.group(1)), float(m.group(2))
                lat_val, lon_val = (b,a) if abs(a)>abs(b) else (a,b)
                mlat=_re_lat.search(v); mlon=_re_lon.search(v)
                if mlat and mlat.group(2): lat_val = -abs(lat_val) if mlat.group(2).upper()=="S" else abs(lat_val)
                if mlon and mlon.group(2): lon_val = -abs(lon_val) if mlon.group(2).upper()=="W" else abs(lon_val)
                return lat_val, lon_val
    for k in LAT_KEYS:
        if k in props and props[k]:
            m=_re_lat.search(props[k][0])
            if m:
                lat_val=float(m.group(1))
                if m.group(2) and m.group(2).upper()=="S": lat_val=-abs(lat_val)
    for k in LON_KEYS:
        if k in props and props[k]:
            m=_re_lon.search(props[k][0])
            if m:
                lon_val=float(m.group(1))
                if m.group(2) and m.group(2).upper()=="W": lon_val=-abs(lon_val)
    return lat_val, lon_val

# ===== RDF 加载与索引（缓存两段式） =====
@st.cache_resource(show_spinner=False)
def load_graph(path: str) -> Graph:
    fmt = {".nt":"nt",".ttl":"turtle",".rdf":"xml",".owl":"xml",".xml":"xml"}.get(Path(path).suffix.lower(), None)
    g = Graph(); g.parse(path, format=(fmt or "nt")); return g

@st.cache_resource(show_spinner=True, hash_funcs={Graph: lambda obj: id(obj)})
def _build_indices_from_graph(_g: Graph):
    g = _g

    pred_by_local: Dict[str, Set[URIRef]] = defaultdict(set)
    for _, p, _ in g.triples((None, None, None)):
        pred_by_local[localname(p)].add(p)

    persons, places, births, offices, tasks, parts, exams, props, tps = set(), set(), set(), set(), set(), set(), set(), set(), set()
    for s, t in g.subject_objects(RDF.type):
        if not isinstance(s, URIRef): continue
        ln = localname(t)
        if ln == CLASS_PERSON: persons.add(s)
        elif ln == CLASS_PLACE: places.add(s)
        elif ln == CLASS_BIRTH: births.add(s)
        elif ln == CLASS_OFFICE: offices.add(s)
        elif ln == CLASS_TASK: tasks.add(s)
        elif ln == CLASS_PARTICIPATION: parts.add(s)
        elif ln == CLASS_IMPERIAL_EXAM: exams.add(s)
        elif ln == CLASS_PROPASSERTION: props.add(s)
        elif ln == CLASS_TEXTPROV: tps.add(s)

    place_meta: Dict[str, Dict[str, Union[str, float, None]]] = {}
    for pl in places:
        props_l = get_literals(g, pl)
        mod = first_lit(props_l, [DP_PLACE_MODERN], "") or first_lit(props_l, [DP_PLACE_HIST], localname(pl))
        lv  = first_lit(props_l, [DP_ADMIN_LEVEL, "区划层级"], "") or level_from_name(mod)
        lat, lon = parse_lat_lon_from_props(props_l)
        place_meta[str(pl)] = {"modern": mod, "level": canon_level_suffix(lv), "lat": lat, "lon": lon}

    child2parent: Dict[str, Set[str]] = defaultdict(set)
    parent2children: Dict[str, Set[str]] = defaultdict(set)
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef) and localname(p) == OP_SUB_PLACE:
            if is_instance_of(g, s, CLASS_PLACE) and is_instance_of(g, o, CLASS_PLACE):
                child2parent[str(s)].add(str(o)); parent2children[str(o)].add(str(s))

    allowed_place_preds: Set[URIRef] = set()
    for nm in PLACE_PRED_NAMES:
        allowed_place_preds |= pred_by_local.get(nm, set())

    def strict_event_place(e: URIRef) -> Optional[str]:
        for p,o in g.predicate_objects(e):
            if (p in allowed_place_preds) and isinstance(o, URIRef) and is_instance_of(g, o, CLASS_PLACE):
                return str(o)
        return None

    def is_clean_office(e: URIRef) -> bool:
        T = {localname(t) for t in g.objects(e, RDF.type)}
        return (CLASS_OFFICE in T) and (CLASS_BIRTH not in T)
    def is_clean_task(e: URIRef) -> bool:
        T = {localname(t) for t in g.objects(e, RDF.type)}
        return (CLASS_TASK in T) and (CLASS_BIRTH not in T)

    office_events = {e for e in offices if is_clean_office(e)}
    task_events   = {e for e in tasks   if is_clean_task(e)}

    office_ev2place, task_ev2place, birth_ev2place = {}, {}, {}
    for e in office_events:
        pl = strict_event_place(e)
        if pl: office_ev2place[str(e)] = pl
    for e in task_events:
        pl = strict_event_place(e)
        if pl: task_ev2place[str(e)] = pl
    for e in births:
        pl = strict_event_place(e)
        if pl: birth_ev2place[str(e)] = pl

    office_ev2persons: Dict[str, Set[str]] = defaultdict(set)
    task_ev2persons  : Dict[str, Set[str]] = defaultdict(set)
    birth_ev2persons : Dict[str, Set[str]] = defaultdict(set)
    person2office_ev : Dict[str, Set[str]] = defaultdict(set)
    person2task_ev   : Dict[str, Set[str]] = defaultdict(set)
    person2birth_ev  : Dict[str, Set[str]] = defaultdict(set)

    def link_evt_person(evt_set: Set[URIRef], ev2persons: Dict[str, Set[str]], p2ev: Dict[str, Set[str]]):
        for e in evt_set:
            eid = str(e)
            for p,o in g.predicate_objects(e):
                if isinstance(o, URIRef) and is_instance_of(g, o, CLASS_PERSON):
                    ev2persons[eid].add(str(o)); p2ev[str(o)].add(eid)
            for s2,p in g.subject_predicates(e):
                if isinstance(s2, URIRef) and is_instance_of(g, s2, CLASS_PERSON):
                    ev2persons[eid].add(str(s2)); p2ev[str(s2)].add(eid)

    link_evt_person(office_events, office_ev2persons, person2office_ev)
    link_evt_person(task_events,   task_ev2persons,   person2task_ev)
    link_evt_person(births,        birth_ev2persons,  person2birth_ev)

    part2exam: Dict[str, str] = {}
    for pe in parts:
        for p, ex in g.predicate_objects(pe):
            if isinstance(ex, URIRef) and localname(p) == OP_HAS_EXAM and is_instance_of(g, ex, CLASS_IMPERIAL_EXAM):
                part2exam[str(pe)] = str(ex)

    def exam_is_palace(ex_uri: str) -> bool:
        ex = URIRef(ex_uri)
        if any(("殿试" in str(o)) for _, o in g.predicate_objects(ex) if isinstance(o, Literal)): return True
        return ("殿试" in localname(ex)) or ("Palace" in localname(ex))

    year_pat = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
    def exam_years(ex_uri: str) -> List[int]:
        yrs = set()
        ex = URIRef(ex_uri)
        for _, lit in g.predicate_objects(ex):
            if isinstance(lit, Literal):
                for m in year_pat.findall(str(lit)):
                    y = int(m)
                    if 1000 <= y <= 2100: yrs.add(y)
        return sorted(yrs)

    exam_people: Set[str] = set()
    person_exam_years: Dict[str, Set[int]] = defaultdict(set)
    for s, t in g.subject_objects(RDF.type):
        if not isinstance(s, URIRef) or localname(t) != CLASS_PERSON: continue
        pid = str(s)
        pe_nodes = set()
        for p, pe in g.predicate_objects(s):
            if isinstance(pe, URIRef) and localname(p) == OP_PARTICIPATES and is_instance_of(g, pe, CLASS_PARTICIPATION):
                pe_nodes.add(str(pe))
        for pe, p in g.subject_predicates(s):
            if isinstance(pe, URIRef) and localname(p) == OP_PARTICIPATES and is_instance_of(g, pe, CLASS_PARTICIPATION):
                pe_nodes.add(str(pe))
        keep = False; yrs=set()
        for peid in pe_nodes:
            exid = part2exam.get(peid)
            if not exid: continue
            if exam_is_palace(exid):
                keep = True
                for y in exam_years(exid): yrs.add(y)
        if keep: exam_people.add(pid)
        if yrs: person_exam_years[pid] |= yrs

    person_birth_places: Dict[str, Set[str]] = defaultdict(set)
    for eid, ppl in birth_ev2persons.items():
        pl = birth_ev2place.get(eid)
        if not pl: continue
        for p in ppl:
            person_birth_places[p].add(pl)

    pred_index = {k: set(v) for k,v in pred_by_local.items()}

    return dict(
        g=g, place_meta=place_meta, child2parent=child2parent, parent2children=parent2children,
        office_ev2place=office_ev2place, task_ev2place=task_ev2place, birth_ev2place=birth_ev2place,
        office_ev2persons=office_ev2persons, task_ev2persons=task_ev2persons, birth_ev2persons=birth_ev2persons,
        person2office_ev=person2office_ev, person2task_ev=person2task_ev, person2birth_ev=person2birth_ev,
        exam_people=exam_people, person_exam_years={k:list(v) for k,v in person_exam_years.items()},
        person_birth_places=person_birth_places, pred_index={k:[str(x) for x in v] for k,v in pred_index.items()}
    )

@st.cache_resource(show_spinner=True)
def build_indices(path: str):
    g = load_graph(path)
    return _build_indices_from_graph(g)

def restore_pred_index(d: Dict[str, List[str]]) -> Dict[str, Set[URIRef]]:
    return {k: {URIRef(x) for x in v} for k,v in d.items()}

# ===== 层级/行政工具 =====
def ancestors_of(pid: str, child2parent: Dict[str, Set[str]]) -> List[str]:
    res, q, seen = [], deque([pid]), {pid}
    while q:
        cur = q.popleft()
        for pa in child2parent.get(cur, set()):
            if pa not in seen:
                seen.add(pa); res.append(pa); q.append(pa)
    return res

def descendants_of(pid: str, parent2children: Dict[str, Set[str]]) -> List[str]:
    res, q, seen = [], deque([pid]), {pid}
    while q:
        cur = q.popleft()
        for ch in parent2children.get(cur, set()):
            if ch not in seen:
                seen.add(ch); res.append(ch); q.append(ch)
    return res

def is_view_level_node(place_id: str, level_choice: str, place_meta: Dict[str, Dict[str, Union[str,float,None]]]) -> bool:
    meta = place_meta.get(place_id, {})
    lv = meta.get("level",""); name = str(meta.get("modern",""))
    if level_choice == "省":
        segs = tokenize_admin_name(name)
        if len(segs)==1 and name.endswith("市"): return True
    return lv in LEVEL_MAP[level_choice]

def lift_to_level(place_id: str, level_choice: str, place_meta, child2parent) -> Optional[str]:
    if is_view_level_node(place_id, level_choice, place_meta): return place_id
    for a in ancestors_of(place_id, child2parent):
        if is_view_level_node(a, level_choice, place_meta): return a
    return None

def level_options_map(pm: Dict[str, Dict[str, Union[str,float,None]]]) -> Dict[str, List[Tuple[str,str]]]:
    L: Dict[str, List[Tuple[str,str]]] = defaultdict(list)
    for pid, m in pm.items():
        lv = m.get("level",""); mod = m.get("modern","")
        if isinstance(mod, str) and lv: L[lv].append((mod, pid))
    for k in L: L[k].sort(key=lambda t: t[0])
    return L

def collect_descendants_at_level(root: str, target_level: str,
                                 parent2children: Dict[str, Set[str]],
                                 place_meta: Dict[str, Dict[str, Union[str,float,None]]]) -> Set[str]:
    result: Set[str] = set()
    for pid in [root] + descendants_of(root, parent2children):
        mod = place_meta.get(pid, {}).get("modern","")
        lv  = place_meta.get(pid, {}).get("level","")
        if lv in LEVEL_MAP.get(target_level, set()) or \
           (target_level=="省" and str(mod).endswith("市") and len(tokenize_admin_name(str(mod))) == 1):
            result.add(pid)
    return result

# ===== 人物属性 =====
def person_display_name(g: Graph, pid: str) -> str:
    props = get_literals(g, URIRef(pid))
    for k in PERSON_NAME_KEYS:
        ln = localname(k)
        if ln in props and props[ln]:
            for v in props[ln]:
                if v: return v
    return localname(URIRef(pid))

def pa_school_types_of_person(g: Graph, person: URIRef) -> List[str]:
    out, seen = [], set()
    for pa in g.subjects(P_ABOUT, person):
        props_txt=[]
        for lit in g.objects(pa, P_PROP):
            if isinstance(lit, Literal):
                s = str(lit).strip()
                if s: props_txt.append(s)
        if not props_txt: continue
        ok=False
        for s in props_txt:
            ln=localname(s)
            if ln in PA_SCHOOL_TYPE_KEYS: ok=True; break
        if not ok:
            for s in props_txt:
                if ("学籍" in s or "學籍" in s) and ("类型" in s or "類型" in s): ok=True; break
        if not ok: continue
        val=""
        for lit in g.objects(pa, P_VALN):
            if isinstance(lit, Literal) and str(lit).strip(): val=str(lit).strip(); break
        if not val:
            for lit in g.objects(pa, P_VAL):
                if isinstance(lit, Literal) and str(lit).strip(): val=str(lit).strip(); break
        if not val: continue
        k = to_s(re.sub(r"\s+","", val))
        if k and k not in seen: seen.add(k); out.append(val)
    return out

def all_person_prop_values(g: Graph, pred_index: Dict[str, Set[URIRef]], local: str) -> List[str]:
    vals, seen = [], set()
    for s, t in g.subject_objects(RDF.type):
        if not isinstance(s, URIRef) or localname(t)!=CLASS_PERSON: continue
        for pred in pred_index.get(local, set()):
            for lit in g.objects(s, pred):
                if isinstance(lit, Literal):
                    v=str(lit).strip()
                    if v and v not in seen: seen.add(v); vals.append(v)
    vals.sort(); return vals

def all_school_types(g: Graph) -> List[str]:
    vals, seen = [], set()
    for s, t in g.subject_objects(RDF.type):
        if not isinstance(s, URIRef) or localname(t)!=CLASS_PERSON: continue
        for v in pa_school_types_of_person(g, s):
            k = to_s(re.sub(r"\s+","", v))
            if k and k not in seen: seen.add(k); vals.append(v)
    vals.sort(); return vals

# ===== 溯源 =====
def restore_pred(d: Dict[str, List[str]]) -> Dict[str, Set[URIRef]]:
    return {k: {URIRef(x) for x in v} for k,v in d.items()}

def tp_fields(g: Graph, pred_index: Dict[str, Set[URIRef]], tp: URIRef) -> Tuple[str,str,str]:
    DP_BODY = pred_index.get("Text_body", set())
    DP_CONF = pred_index.get("record_confidence", set())
    DP_SRC  = set()
    for key in ["Text_Source","Text_source","来源","书名","source","Source","book","Book","Title","题名"]:
        DP_SRC |= pred_index.get(key, set())
    src, body, conf = "", "", ""
    for p,o in g.predicate_objects(tp):
        if isinstance(o, Literal):
            if (p in DP_BODY) and not body: body=str(o).strip()
            if (p in DP_CONF) and not conf: conf=str(o).strip()
            if (p in DP_SRC)  and not src : src =str(o).strip()
    return (src or "（未知书目）"), body, conf

def provenance_for_person(g: Graph, pred_index: Dict[str, Set[URIRef]], person_id: str, related: List[str], disp: str) -> List[Dict]:
    P_ABT = pred_index.get("about", set())
    P_DFR = pred_index.get("derivedFrom", set())
    P_CON = pred_index.get("contains", set())
    key_s = to_s(re.sub(r"\s+","", disp or ""))
    def ok(body: str) -> bool: return key_s in to_s(re.sub(r"\s+","", body or ""))

    groups: Dict[Tuple[str,str], Dict] = {}
    def put(tp: URIRef, label: str):
        src, body, conf = tp_fields(g, pred_index, tp)
        if not body or not ok(body): return
        k = (src.strip(), body.strip())
        if k not in groups: groups[k] = {"src":src,"body":body,"conf":conf,"labels":set([label])}
        else:
            groups[k]["labels"].add(label)
            try:
                if conf and (not groups[k]["conf"] or float(conf)>float(groups[k]["conf"])): groups[k]["conf"]=conf
            except: pass

    for u in related:
        node = URIRef(u)
        for P in P_CON:
            for tp in g.subjects(P, node): put(tp, localname(node))
        for P in P_ABT:
            for pa in g.subjects(P, node):
                for P2 in P_DFR:
                    for tp in g.objects(pa, P2): put(tp, localname(node))

    out = sorted([{"src":k[0],"body":k[1],"conf":v["conf"],"labels":sorted(v["labels"])} for k,v in groups.items()],
                 key=lambda d: (d["src"], d["body"]))
    return out

# ===== 统计图 =====
def grouped_banded_bar(df_top: pd.DataFrame,
                       view_level: str,
                       place_meta: Dict[str, Dict[str, Union[str,float,None]]],
                       child2parent: Dict[str, Set[str]]) -> Optional[go.Figure]:
    if df_top.empty:
        return None

    def parent_of(pid: str, parent_level: Optional[str]) -> Optional[str]:
        if parent_level is None: return None
        for a in [pid] + ancestors_of(pid, child2parent):
            if is_view_level_node(a, parent_level, place_meta): return a
        return None

    parent_level = PARENT_LEVEL.get(view_level)
    if not parent_level:
        d2 = df_top.copy().sort_values(["人数","地点"], ascending=[False, True]).head(120)
        fig = px.bar(d2, x="人数", y="地点", orientation="h", text="人数", template="plotly_white")
        fig.update_traces(textposition="outside", marker_line_width=0.6, marker_line_color="rgba(0,0,0,0.35)")
        fig.update_layout(
            height=int(max(440, min(1400, d2.shape[0]*22 + 180))),
            margin=dict(l=16, r=16, t=52, b=40),
            font=PLOT_FONT, xaxis_title="人数（去重）", yaxis_title=None,
            uniformtext_minsize=10, uniformtext_mode="hide", bargap=0.22
        )
        return fig

    records = []
    for _, r in df_top.iterrows():
        pid  = r["place_id"]; name = r["地点"]; val  = int(r["人数"])
        pa = parent_of(pid, PARENT_LEVEL.get(view_level))
        pa_name = place_meta.get(pa, {}).get("modern", pa or "（无上层）")
        records.append({"上层": pa_name, "地点": name, "人数": val})

    dff = pd.DataFrame(records)
    totals_by_parent = dff.groupby("上层")["人数"].sum().sort_values(ascending=False)
    parent_order = list(totals_by_parent.index)
    dff["上层"] = pd.Categorical(dff["上层"], categories=parent_order, ordered=True)
    dff = dff.sort_values(["上层", "人数", "地点"], ascending=[True, False, True])
    labels = dff["地点"].tolist()
    dense_vertical = (len(labels) > 26) or (sum(len(s) for s in labels)/max(1,len(labels)) > 6)

    base_rgb = [
        (230, 99, 132),  (54, 162, 235), (255, 206, 86),  (75, 192, 192),
        (153, 102, 255), (255, 159, 64), (199, 199, 199), (255, 99, 71),
        (60, 179, 113),  (123, 104, 238), (255, 215, 0),  (100, 149, 237)
    ]
    def rgba(rgb, a): return f"rgba({rgb[0]},{rgb[1]},{rgb[2]},{a})"
    parent_bar= {p: rgba(base_rgb[i % len(base_rgb)], 0.85) for i, p in enumerate(parent_order)}

    if dense_vertical:
        y_order = dff["地点"].tolist()
        colors = [parent_bar[row["上层"]] for _, row in dff.iterrows()]
        fig = go.Figure(go.Bar(
            x=dff["人数"], y=y_order, orientation="h", text=dff["人数"],
            marker=dict(color=colors, line=dict(width=0.5, color="rgba(0,0,0,0.35)"))
        ))
        fig.update_traces(textposition="outside")
        fig.update_layout(
            template="plotly_white",
            height=int(max(520, min(1600, len(y_order)*22 + 220))),
            margin=dict(l=28, r=24, t=32, b=36),
            font=PLOT_FONT, xaxis_title="人数（去重）", yaxis_title=None,
            uniformtext_minsize=10, uniformtext_mode="hide", bargap=0.22, hovermode="closest",
        )
        fig.update_traces(hovertemplate="地点：%{y}<br>人数：%{x}<extra></extra>")
        return fig

    x_order = dff["地点"].tolist()
    colors = [parent_bar[row["上层"]] for _, row in dff.iterrows()]
    fig = go.Figure(go.Bar(
        x=x_order, y=dff["人数"], text=dff["人数"],
        marker=dict(color=colors, line=dict(width=0.5, color="rgba(0,0,0,0.35)"))
    ))
    fig.update_traces(textposition="outside")
    fig.update_layout(
        template="plotly_white",
        height=int(max(520, min(1400, len(x_order) * 26 + 220))),
        margin=dict(l=24, r=16, t=92, b=130),
        font=PLOT_FONT, xaxis_title=None, yaxis_title="人数（去重）",
        uniformtext_minsize=10, uniformtext_mode="hide", bargap=0.25, hovermode="closest",
    )
    fig.update_traces(hovertemplate="地点：%{x}<br>人数：%{y}<extra></extra>")
    return fig

# ===== GeoJSON / 边界线 =====
@st.cache_data(show_spinner=False)
def load_geojson(path: str) -> Optional[dict]:
    try:
        if not path or not os.path.exists(path):
            return None
        txt = None
        for enc in ("utf-8", "utf-8-sig", "gb18030"):
            try:
                with open(path, "r", encoding=enc) as f:
                    txt = f.read()
                break
            except Exception:
                continue
        if txt is None:
            return None
        data = json.loads(txt)
        if isinstance(data, dict) and data.get("type") == "Topology":
            return None
        return data
    except Exception:
        return None

def add_plain_white_background(m: folium.Map):
    folium.map.CustomPane("bg-pane", z_index=300).add_to(m)
    world = {"type":"FeatureCollection","features":[{"type":"Feature","properties":{},
             "geometry":{"type":"Polygon","coordinates":[[[-180,-89.9],[180,-89.9],[180,89.9],[-180,89.9],[-180,-89.9]]]}}]}
    folium.GeoJson(world, style_function=lambda f: dict(color="#ffffff", weight=0, fillColor="#ffffff", fillOpacity=1.0),
                   pane="bg-pane", control=False).add_to(m)

def add_admin_boundary_lines(m: folium.Map, gj: dict, color="#777777", weight=1, opacity=1.0, dash=None, name=None):
    if not gj: return
    folium.map.CustomPane("boundary-pane", z_index=500).add_to(m)
    def sty(_):
        s = dict(color=color, weight=weight, opacity=opacity, fill=False, fillOpacity=0.0)
        if dash: s["dashArray"] = dash
        return s
    folium.GeoJson(gj, name=(name or "boundary"), style_function=sty, highlight_function=None,
                   control=False, pane="boundary-pane").add_to(m)

# ===== 分桶 + 人名 + 溯源（使用运行期的全局上下文） =====
def render_bucket_section(df_bucket_like: pd.DataFrame,
                          bucket_people: Dict[str, Set[str]],
                          etype: str):
    st.session_state.setdefault("sel_person", None)
    st.session_state.setdefault("sel_person_label", "")
    st.session_state.setdefault("sel_related", [])

    def set_sel(pid: str, label: str, related: List[str]):
        st.session_state["sel_person"] = pid
        st.session_state["sel_person_label"] = label
        st.session_state["sel_related"] = related

    if df_bucket_like is None or df_bucket_like.empty:
        st.info("无分桶。")
        return

    # 表格
    dfv = df_bucket_like.copy()
    dfv["排名"] = range(1, len(dfv)+1)
    total = dfv["人数"].sum() if len(dfv) else 1
    dfv["占比"] = (dfv["人数"] / total).round(4)
    dfv["累计占比"] = (dfv["占比"].cumsum()).round(4)
    dfv = dfv[["排名","地点","人数","占比","累计占比","place_id"]]
    st.dataframe(dfv.drop(columns=["place_id"]).head(30), use_container_width=True, height=420)

    csv = dfv.drop(columns=["place_id"]).to_csv(index=False, encoding="utf-8-sig")
    st.download_button("下载分桶表（CSV）", data=csv, file_name="分桶统计.csv", mime="text/csv")

    # 详情（人名）
    for _, row in df_bucket_like.iterrows():
        pname, cnt, pid_place = row["地点"], int(row["人数"]), row["place_id"]
        with st.expander(f"{pname} —— 人数：{cnt}", expanded=False):
            child_level = NEXT_LEVEL.get(applied_core["view_level"])
            if child_level:
                child_nodes = collect_descendants_at_level(pid_place, child_level, parent2children, place_meta)
                child_people: Dict[str, Set[str]] = defaultdict(set)
                for p in (bucket_people.get(pid_place, set()) or []):
                    if etype == "任职":
                        p2ev = S["person2office_ev"]; evp = S["office_ev2place"]
                    elif etype == "差遣":
                        p2ev = S["person2task_ev"];   evp = S["task_ev2place"]
                    else:
                        p2ev = S["person2birth_ev"];  evp = S["birth_ev2place"]
                    for e in p2ev.get(p, set()):
                        pl = evp.get(e)
                        if pl:
                            lifted_child = lift_to_level(pl, child_level, place_meta, child2parent)
                            if lifted_child and (lifted_child in child_nodes):
                                child_people[lifted_child].add(p)
                det = []
                for ch_id, ppl in child_people.items():
                    det.append((place_meta[ch_id]["modern"], len(ppl), ch_id))
                det.sort(key=lambda x: x[1], reverse=True)
                if det:
                    dfc = pd.DataFrame(det, columns=["下级地点","人数","place_id"])
                    subfig = px.bar(dfc.head(40), x="人数", y="下级地点", orientation="h", text="人数", template="plotly_white")
                    subfig.update_traces(textposition="outside", marker_line_width=0.6, marker_line_color="rgba(0,0,0,0.35)")
                    subfig.update_layout(xaxis_title="人数（去重）", yaxis_title=None,
                                         margin=dict(l=8,r=8,t=6,b=8), font=PLOT_FONT,
                                         height=260, uniformtext_minsize=10, uniformtext_mode="hide")
                    st.plotly_chart(subfig, use_container_width=True, config={"displaylogo": False})
                else:
                    st.caption("（该桶在下一层级无统计结果）")

            plist = sorted(list(bucket_people.get(pid_place, set())), key=lambda x: person_display_name(g, x))
            cols = st.columns(3)
            for i, p in enumerate(plist[:600]):
                nm = person_display_name(g, p)
                related = []
                if etype == "任职":
                    p2ev = S["person2office_ev"]; evp = S["office_ev2place"]
                elif etype == "差遣":
                    p2ev = S["person2task_ev"];   evp = S["task_ev2place"]
                else:
                    p2ev = S["person2birth_ev"];  evp = S["birth_ev2place"]
                for e in (p2ev.get(p, set()) or []):
                    pl = evp.get(e)
                    if pl: related.extend([e, pl])
                with cols[i % 3]:
                    st.button(nm, key=f"btn_{pid_place}_{p}",
                              on_click=set_sel, args=(p, nm, list(dict.fromkeys(related))),
                              use_container_width=True)

# ===== GIS 主图（已去除“主视域”遮罩与选择） =====
def render_gis_map_from_buckets(df_top: pd.DataFrame,
                                bucket_people: Dict[str, Set[str]],
                                *,
                                show_heat: bool, heat_radius: int, heat_blur: int,
                                heat_min_op: float, heat_boost: float,
                                label_scale: float, offset_px: int, show_points: bool,
                                basemap_choice: str,
                                show_province: bool, show_city: bool,
                                etype: str, view_level: str,
                                map_height: int = 1400):
    if df_top.empty:
        st.warning("GIS：无可视化数据（统计结果为空）。")
        return

    st.markdown(f"""
        <style>
        section.main .block-container {{padding-top: 0.8rem;}}
        h3, h4, h5 {{margin-top: 0.2rem; margin-bottom: 0.4rem;}}
        .element-container:has(iframe) {{margin-bottom: 0.35rem;}}
        .tight-hr {{margin: 10px 0 12px 0; border: none; border-top: 1px solid #ddd;}}
        .prov-panel {{max-height: {int(map_height)-60}px; overflow: auto; padding-right: 6px;}}
        </style>
    """, unsafe_allow_html=True)

    # —— 聚合点坐标 —— #
    def bucket_coord(pid: str) -> Optional[Tuple[float,float]]:
        m = place_meta.get(pid, {})
        lat, lon = m.get("lat"), m.get("lon")
        if isinstance(lat, (int,float)) and isinstance(lon, (int,float)):
            return float(lat), float(lon)
        acc_lat, acc_lon, n = 0.0, 0.0, 0
        for did in [pid] + descendants_of(pid, parent2children):
            dm = place_meta.get(did, {})
            la, lo = dm.get("lat"), dm.get("lon")
            if isinstance(la, (int,float)) and isinstance(lo, (int,float)):
                acc_lat += float(la); acc_lon += float(lo); n += 1
        if n>0: return acc_lat/n, acc_lon/n
        return None

    pts = []
    for _, r in df_top.iterrows():
        bid = r["place_id"]; name = r["地点"]; cnt = int(r["人数"])
        xy = bucket_coord(bid)
        if not xy: continue
        pts.append(dict(bucket_id=bid, name=name, lat=xy[0], lon=xy[1], weight=cnt))
    if not pts:
        st.warning("GIS：所有聚合桶均缺坐标，无法渲染。")
        return
    df_gis = pd.DataFrame(pts)

    center_lat = float(np.clip(df_gis["lat"].mean(), CHINA_BOUNDS["min_lat"], CHINA_BOUNDS["max_lat"]))
    center_lon = float(np.clip(df_gis["lon"].mean(), CHINA_BOUNDS["min_lon"], CHINA_BOUNDS["max_lon"]))
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4,
                   tiles=None, max_bounds=False, min_zoom=3, max_zoom=9,
                   control_scale=True, prefer_canvas=True,
                   world_copy_jump=False, no_wrap=True)

    china_bounds = [[CHINA_BOUNDS["min_lat"], CHINA_BOUNDS["min_lon"]],
                    [CHINA_BOUNDS["max_lat"], CHINA_BOUNDS["max_lon"]]]
    m.fit_bounds(china_bounds)

    # —— 底图 —— #
    if basemap_choice == "无底图（纯白，仅边界）":
        add_plain_white_background(m)
    else:
        url, attr, subs = basemap_conf(basemap_choice)
        if subs:
            for s in subs[:1]:
                folium.TileLayer(tiles=url.replace("{s}", s), attr=attr, name=basemap_choice,
                                 control=False, no_wrap=True).add_to(m)
        else:
            folium.TileLayer(tiles=url, attr=attr, name=basemap_choice,
                             control=False, no_wrap=True).add_to(m)

    # —— 行政边界线（可选） —— #
    if show_province:
        gj_prov = load_geojson(PROVINCE_GEOJSON)
        add_admin_boundary_lines(m, gj_prov, color="#555555", weight=1.2, opacity=1.0, name="省界")
    if show_city:
        gj_city = load_geojson(CITY_GEOJSON)
        add_admin_boundary_lines(m, gj_city, color="#888888", weight=0.9, opacity=0.9, dash="3,3", name="市界")

    # —— 热力/点/标签 —— #
    if show_heat:
        RED_GRADIENT = {"0.0":"#ffecec","0.25":"#ffb3b3","0.5":"#ff6666","0.75":"#ff3333","1.0":"#ff0000"}
        heat_df = df_gis.copy(); heat_df["w2"] = heat_df["weight"] * float(heat_boost)
        heat_data = heat_df[["lat","lon","w2"]].values.tolist()
        HeatMap(heat_data, radius=int(heat_radius), blur=int(heat_blur),
                min_opacity=float(heat_min_op), max_zoom=8, gradient=RED_GRADIENT).add_to(m)

    if show_points:
        for _, r in df_gis.iterrows():
            folium.CircleMarker([float(r["lat"]), float(r["lon"])], radius=3, weight=1,
                                color="#2b6cb0", fill=True, fill_opacity=0.35).add_to(m)

    folium.map.CustomPane("label-pane", z_index=960).add_to(m)
    def make_label_html(name: str, count: int, scale: float=1.0, offset_px:int=10) -> str:
        base_name_px  = int(14 * max(0.6, scale))
        base_count_px = int(12 * max(0.6, scale))
        name_safe = html.escape(str(name))
        return f"""
        <div style="position: relative; transform: translate(-50%, -100%); text-align:center; white-space:nowrap;">
            <div style="font-weight:700; font-size:{base_name_px}px; line-height:1.1; color:#111; text-shadow:
                -1px -1px 0 #fff, 1px -1px 0 #fff, -1px 1px 0 #fff, 1px 1px 0 #fff;">{name_safe}</div>
            <div style="position: relative; top:{offset_px}px; display:inline-block; padding:1px 4px; border-radius:10px;
                background: rgba(17,17,17,.78); color:#fff; font-size:{base_count_px}px; line-height:1.0;">{int(count)}</div>
        </div>"""
    for _, r in df_gis.iterrows():
        html_div = make_label_html(r["name"], int(r["weight"]), label_scale, offset_px)
        folium.Marker([float(r["lat"]), float(r["lon"])],
                      icon=folium.DivIcon(html=html_div), pane="label-pane").add_to(m)

    # —— 布局 —— #
    top_left, top_right = st.columns([8,4], gap="large")
    with top_left:
        st.markdown(f"##### GIS 主图 · {etype} · {view_level}层级")
        st_folium(m, width=None, height=1400, returned_objects=[], key="map_gis_high")

    with top_right:
        st.markdown("##### 溯源")
        st.markdown("<div class='prov-panel'>", unsafe_allow_html=True)
        sel_p = st.session_state.get("sel_person")
        if not sel_p:
            st.info("在下方分桶中点击人名以查看溯源。")
        else:
            nm = st.session_state.get("sel_person_label","")
            related = st.session_state.get("sel_related", [])
            provs = provenance_for_person(g, pred_index, sel_p, related, nm)
            if not provs:
                st.warning("无匹配的文本溯源。")
            else:
                for i, ent in enumerate(provs[:60], 1):
                    src, body, conf, labels = ent["src"], ent["body"], ent["conf"], ent["labels"]
                    head = f"#{i} 书目：{src}" + (f"｜可信度：{conf}" if conf else "")
                    with st.expander(head, expanded=(i==1)):
                        if labels: st.caption("关联节点：" + "、".join(labels))
                        st.markdown(highlight_html(body, [nm]), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<hr class='tight-hr'/>", unsafe_allow_html=True)
    st.markdown("#### 分类明细")
    render_bucket_section(df_top, bucket_people, etype)

# ========= 主入口 =========
def run():
    global S, g, place_meta, child2parent, parent2children, pred_index, exam_people, applied_core

    # ========== 侧栏：数据/运行 ==========
    with st.sidebar:
        st.header("数据 / 运行")
        DATA_PATH = st.text_input("RDF 路径", value=DEFAULT_DATA)
        if st.button("清缓存并重建", type="primary", use_container_width=True):
            st.cache_data.clear(); st.cache_resource.clear(); _safe_rerun()

    S = build_indices(DATA_PATH)
    g = S["g"]
    place_meta = S["place_meta"]
    child2parent = S["child2parent"]
    parent2children = S["parent2children"]
    pred_index = restore_pred_index(S["pred_index"])
    exam_people = S["exam_people"]

    st.title("地点项计量")

    # ======= 初始化状态 =======
    def _init_once():
        if "applied_gis" not in st.session_state:
            st.session_state["applied_gis"] = dict(
                show_heat=True, heat_radius=24, heat_blur=18, heat_min_op=0.22, heat_boost=2.0,
                label_scale=1.0, offset_px=10, show_points=False,
                basemap="Carto·灰（无注记）",
                show_province=True, show_city=True
            )
        if "applied_filters" not in st.session_state:
            st.session_state["applied_filters"] = dict(
                enabled=False,
                geo_lvl=None, geo_node=None, sub_lvl=None, sub_geo_node=None,
                sel_school=[], sel_acad=[], sel_huji=[],
                left_era=None, right_era=None,
                apply_attr_era_to_birth=False,
            )
        if "applied_core" not in st.session_state:
            st.session_state["applied_core"] = dict(
                view_mode="GIS 地图",
                etype="任职", view_level="省", birth_mode="排除矛盾"
            )
        if "draft_gis" not in st.session_state:
            st.session_state["draft_gis"] = dict(st.session_state["applied_gis"])
        if "draft_filters" not in st.session_state:
            st.session_state["draft_filters"] = dict(st.session_state["applied_filters"])
        if "draft_core" not in st.session_state:
            st.session_state["draft_core"] = dict(view_mode=st.session_state["applied_core"]["view_mode"])
    _init_once()

    # ======= 侧栏：快速控制（即时生效）=======
    with st.sidebar:
        st.markdown("---")
        st.subheader("")
        etype_now = st.selectbox("类型", ["任职","差遣","诞生"],
                                 index=["任职","差遣","诞生"].index(st.session_state["applied_core"].get("etype","任职")))
        level_now = st.selectbox("区划层级", ["省","市","县/区/旗","乡镇","村/社区"],
                                 index=["省","市","县/区/旗","乡镇","村/社区"].index(st.session_state["applied_core"].get("view_level","省")))
        birth_mode_now = st.selectbox("矛盾处理", ["排除矛盾","包括矛盾"],
                                      index=["排除矛盾","包括矛盾"].index(st.session_state["applied_core"].get("birth_mode","排除矛盾")))
        st.session_state["applied_core"]["etype"] = etype_now
        st.session_state["applied_core"]["view_level"] = level_now
        st.session_state["applied_core"]["birth_mode"] = birth_mode_now

    # ======= 侧栏：功能框（点击“更新视图”应用）=======
    with st.sidebar:
        st.markdown("---")
        st.header("功能框")

        dcore = st.session_state["draft_core"]
        dcore["view_mode"] = st.radio("可视化模式", ["统计柱状图","GIS 地图"],
                                      index=["统计柱状图","GIS 地图"].index(dcore.get("view_mode","GIS 地图")))

        dfilt = st.session_state["draft_filters"]
        st.markdown("**阈值**")
        dfilt["enabled"] = st.checkbox("启用（人物/地域/时间）", value=bool(dfilt.get("enabled", False)))

        if dfilt["enabled"]:
            lv_opts = level_options_map(place_meta)
            order_lv = ["省","市","县/区/旗","乡镇","村/社区"]
            exist_lv = [lv for lv in order_lv if lv in lv_opts] or sorted(lv_opts.keys())

            with st.expander("地域条件（出生地）", expanded=False):
                geo_lvl = st.selectbox("阈值层级", ["（未选择）"] + exist_lv,
                                       index=(["（未选择）"] + exist_lv).index(dfilt.get("geo_lvl") or "（未选择）"))
                dfilt["geo_lvl"] = None if geo_lvl == "（未选择）" else geo_lvl
                dfilt["geo_node"] = None; dfilt["sub_lvl"] = None; dfilt["sub_geo_node"] = None
                if dfilt["geo_lvl"]:
                    geo_pairs = lv_opts.get(dfilt["geo_lvl"], [])
                    geo_dict = {"（未选择）": None}
                    geo_dict.update({mod: uri for (mod, uri) in geo_pairs})
                    geo_mod = st.selectbox("阈值地点（现代名称）", list(geo_dict.keys()), index=0)
                    dfilt["geo_node"] = geo_dict.get(geo_mod)
                    refine = st.checkbox("细化到下层", value=False)
                    if refine and dfilt["geo_node"]:
                        sub_levels = [lv for lv in order_lv if lv != dfilt["geo_lvl"]]
                        if NEXT_LEVEL.get(dfilt["geo_lvl"]):
                            start_lv = NEXT_LEVEL[dfilt["geo_lvl"]]
                            sub_levels = sub_levels[sub_levels.index(start_lv):]
                        sub_levels = [lv for lv in sub_levels if lv in exist_lv]
                        if sub_levels:
                            dfilt["sub_lvl"] = st.selectbox("下层层级", sub_levels, index=0)
                            cand_nodes = collect_descendants_at_level(dfilt["geo_node"], dfilt["sub_lvl"], parent2children, place_meta)
                            pair = sorted([(place_meta[x]["modern"], x) for x in cand_nodes], key=lambda t: t[0])
                            sd = {"（未选择）": None}; sd.update({n:i for n,i in pair})
                            sub_mod = st.selectbox("下层地点（现代名称）", list(sd.keys()), index=0)
                            dfilt["sub_geo_node"] = sd.get(sub_mod)

            with st.expander("人物条件", expanded=False):
                opt_school = all_school_types(g)
                opt_acad   = all_person_prop_values(g, pred_index, "学术专长")
                opt_huji   = all_person_prop_values(g, pred_index, "户籍类型")
                dfilt["sel_school"] = st.multiselect("学籍类别", opt_school, default=dfilt.get("sel_school", []))
                dfilt["sel_acad"]   = st.multiselect("学术专长", opt_acad, default=dfilt.get("sel_acad", []))
                dfilt["sel_huji"]   = st.multiselect("户籍类别", opt_huji, default=dfilt.get("sel_huji", []))

            with st.expander("时间条件（年号范围）", expanded=False):
                left_era  = st.selectbox("左界（含）", ["（未选择）"] + ERA_ORDER, index=(["（未选择）"] + ERA_ORDER).index(dfilt.get("left_era") or "（未选择）"))
                right_era = st.selectbox("右界（含）", ["（未选择）"] + ERA_ORDER, index=(["（未选择）"] + ERA_ORDER).index(dfilt.get("right_era") or "（未选择）"))
                dfilt["left_era"]  = None if left_era  == "（未选择）" else left_era
                dfilt["right_era"] = None if right_era == "（未选择）" else right_era

            dfilt["apply_attr_era_to_birth"] = st.checkbox("将人物属性/时期阈值应用到“诞生”统计（默认关闭）",
                                                           value=bool(dfilt.get("apply_attr_era_to_birth", False)))

        dgis = st.session_state["draft_gis"]
        st.markdown("---")
        enable_gis_settings = st.checkbox("启用视图设置", value=False)
        if enable_gis_settings:
            st.subheader("GIS 视图设置")
            dgis["show_heat"] = st.checkbox("显示热力层（红色）", dgis.get("show_heat", True))
            dgis["heat_radius"] = st.slider("热力半径", 5, 60, dgis.get("heat_radius",24), 1)
            dgis["heat_blur"] = st.slider("热力模糊", 1, 40, dgis.get("heat_blur",18), 1)
            dgis["heat_min_op"] = st.slider("热力最小不透明度", 0.0, 1.0, float(dgis.get("heat_min_op",0.22)), 0.02)
            dgis["heat_boost"] = st.slider("热力权重放大（×）", 0.5, 10.0, float(dgis.get("heat_boost",2.0)), 0.1)
            st.markdown("---")
            dgis["label_scale"] = st.slider("中文标签字号倍率", 0.6, 2.5, float(dgis.get("label_scale",1.0)), 0.1)
            dgis["offset_px"] = st.slider("人数标签垂直偏移（px）", -30, 30, int(dgis.get("offset_px",10)), 1)
            dgis["show_points"] = st.checkbox("显示参考点（淡色圆点）", bool(dgis.get("show_points", False)))
            st.markdown("---")
            basemap_options = [
                "无底图（纯白，仅边界）",
                "Carto·灰（全注记）",
                "Carto·灰（无注记）",
                "Carto·Voyager（无注记）",
                "Esri·World Gray Canvas",
                "OpenTopoMap",
                "GeoQ·蓝黑",
                "高德矢量",
                "高德影像",
                "高德矢量（无注记）",
            ]
            default_choice = dgis.get("basemap","Carto·灰（无注记）")
            if default_choice not in basemap_options:
                default_choice = "Carto·灰（无注记）"
            dgis["basemap"] = st.selectbox("底图（学术友好型）", basemap_options, index=basemap_options.index(default_choice))
            st.caption("建议论文/报告：Carto·灰（无注记） 或 Esri·World Gray Canvas。Topo 分析：OpenTopoMap。")

        if st.button("更新视图", type="primary", use_container_width=True):
            st.session_state["applied_gis"] = dict(dgis)
            st.session_state["applied_filters"] = dict(dfilt)
            st.session_state["applied_core"]["view_mode"] = dcore["view_mode"]
            _safe_rerun()

    # ======= 读取“已应用”参数 =======
    AG = st.session_state["applied_gis"]
    AF = st.session_state["applied_filters"]
    applied_core = st.session_state["applied_core"]

    view_mode = applied_core["view_mode"]
    etype = applied_core["etype"]
    view_level = applied_core["view_level"]
    birth_contra_mode = applied_core["birth_mode"]

    # ======= 筛选判定 =======
    def in_birth_subtree(pid: str,
                         geo_node: Optional[str], sub_geo_node: Optional[str],
                         parent2children: Dict[str, Set[str]], child2parent: Dict[str, Set[str]]) -> bool:
        if (geo_node is None) and (sub_geo_node is None):
            return True
        root = sub_geo_node or geo_node
        subtree = {root, *descendants_of(root, parent2children)}
        for bp in S["person_birth_places"].get(pid, set()):
            if (bp in subtree) or any(pa in subtree for pa in ancestors_of(bp, child2parent)):
                return True
        return False

    def person_literals_batch(pid: str, local: str) -> Set[str]:
        node = URIRef(pid)
        vals=set()
        for pred in pred_index.get(local, set()):
            for lit in g.objects(node, pred):
                if isinstance(lit, Literal) and str(lit).strip():
                    vals.add(str(lit).strip())
        return vals

    def pa_school_types_of_person_fast(pid: str) -> Set[str]:
        return set(pa_school_types_of_person(g, URIRef(pid)))

    def pass_attr(pid: str, sel_school: List[str], sel_acad: List[str], sel_huji: List[str]) -> bool:
        if sel_school:
            mine = {to_s(re.sub(r"\s+","", v)) for v in pa_school_types_of_person_fast(pid)}
            need = {to_s(re.sub(r"\s+","", v)) for v in sel_school}
            if not (mine & need): return False
        if sel_acad:
            vals = person_literals_batch(pid, "学术专长")
            if not (vals & set(sel_acad)): return False
        if sel_huji:
            vals = person_literals_batch(pid, "户籍类型")
            if not (vals & set(sel_huji)): return False
        return True

    def pass_era(pid: str, left_era: Optional[str], right_era: Optional[str]) -> bool:
        if not left_era or not right_era: return True
        li, ri = ERA_ORDER.index(left_era), ERA_ORDER.index(right_era)
        if li > ri: li, ri = ri, li
        ERA_WINDOW = set(ERA_ORDER[li:ri+1])
        yrs = S["person_exam_years"].get(pid, [])
        if not yrs: return False
        for y in yrs:
            n = era_of_year(y)
            if n and n in ERA_WINDOW: return True
        return False

    # ======= 基准人群 =======
    P = set(exam_people)
    if AF.get("enabled", False):
        P = {p for p in P if in_birth_subtree(p, AF.get("geo_node"), AF.get("sub_geo_node"), parent2children, child2parent)}
        apply_attr_era_to_birth = bool(AF.get("apply_attr_era_to_birth", False))
        if (etype in ("任职","差遣")) or (etype == "诞生" and apply_attr_era_to_birth):
            P = {p for p in P if pass_attr(p, AF.get("sel_school", []), AF.get("sel_acad", []), AF.get("sel_huji", []))}
            P = {p for p in P if pass_era(p, AF.get("left_era"), AF.get("right_era"))}

    # ======= 事件池 =======
    if etype == "任职":
        ev2place = S["office_ev2place"]; p2events = S["person2office_ev"]
    elif etype == "差遣":
        ev2place = S["task_ev2place"];   p2events = S["person2task_ev"]
    else:
        ev2place = S["birth_ev2place"];  p2events = S["person2birth_ev"]

    # ======= 诞生矛盾 =======
    def split_birth_conflict_at_level(Pset: Set[str], level_choice: str) -> Tuple[Set[str], Set[str]]:
        non_conflict, conflict = set(), set()
        for pid in Pset:
            places = S["person_birth_places"].get(pid, set())
            lifted = set()
            for pl in places:
                anc = lift_to_level(pl, level_choice, place_meta, child2parent)
                if anc: lifted.add(anc)
            if len(lifted) >= 2: conflict.add(pid)
            elif lifted: non_conflict.add(pid)
        return non_conflict, conflict

    birth_non_conf, birth_conf = set(), set()
    if etype == "诞生":
        birth_non_conf, birth_conf = split_birth_conflict_at_level(P, view_level)

    # ======= 聚合 =======
    def aggregate_at_level(Pset: Set[str], level_choice: str, etype_: str, birth_mode: str
                           ) -> Tuple[pd.DataFrame, Dict[str, Set[str]]]:
        bucket_people: Dict[str, Set[str]] = defaultdict(set)
        if etype_ == "诞生" and birth_mode == "排除矛盾":
            P_use = Pset & birth_non_conf
        else:
            P_use = set(Pset)
        for pid in P_use:
            for e in p2events.get(pid, set()):
                pl = ev2place.get(e)
                if not pl: continue
                lifted = lift_to_level(pl, level_choice, place_meta, child2parent)
                if not lifted: continue
                bucket_people[lifted].add(pid)

        rows = []
        for k,v in bucket_people.items():
            name = place_meta.get(k, {}).get("modern", localname(URIRef(k)))
            rows.append((name, len(v), k))
        rows.sort(key=lambda t: t[1], reverse=True)
        return pd.DataFrame(rows, columns=["地点","人数","place_id"]), bucket_people

    df_top, bucket_people = aggregate_at_level(P, view_level, etype, birth_contra_mode)

    # ======= 页面呈现 =======
    if view_mode == "统计柱状图":
        L, R = st.columns([7,5], gap="large")
        with L:
            st.subheader(f"{etype} · {view_level}层级")
            if df_top.empty:
                st.info("无统计结果。")
            else:
                fig = grouped_banded_bar(df_top, view_level, place_meta, child2parent)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            st.markdown("#### 分桶")
            render_bucket_section(df_top, bucket_people, etype)

        with R:
            st.subheader("溯源")
            sel_p = st.session_state.get("sel_person")
            if not sel_p:
                st.info("点击左侧分桶中的人名以查看溯源。")
            else:
                nm = st.session_state.get("sel_person_label","")
                related = st.session_state.get("sel_related", [])
                provs = provenance_for_person(g, pred_index, sel_p, related, nm)
                if not provs:
                    st.warning("无匹配的文本溯源。")
                else:
                    for i, ent in enumerate(provs[:80], 1):
                        src, body, conf, labels = ent["src"], ent["body"], ent["conf"], ent["labels"]
                        head = f"#{i} 书目：{src}" + (f"｜可信度：{conf}" if conf else "")
                        with st.expander(head, expanded=(i==1)):
                            if labels: st.caption("关联节点：" + "、".join(labels))
                            st.markdown(highlight_html(body, [nm]), unsafe_allow_html=True)
    else:
        render_gis_map_from_buckets(
            df_top=df_top, bucket_people=bucket_people,
            show_heat=bool(AG["show_heat"]), heat_radius=int(AG["heat_radius"]), heat_blur=int(AG["heat_blur"]),
            heat_min_op=float(AG["heat_min_op"]), heat_boost=float(AG["heat_boost"]),
            label_scale=float(AG["label_scale"]), offset_px=int(AG["offset_px"]), show_points=bool(AG["show_points"]),
            basemap_choice=AG["basemap"], show_province=bool(AG["show_province"]), show_city=bool(AG["show_city"]),
            etype=etype, view_level=view_level,
            map_height=1400
        )

    # ===== 底部口径 =====
    st.caption(
        "口径：任职=OfficeAtPlaceEvent、差遣=TaskEvent（均排除兼具BirthEvent）；诞生=BirthEvent。"
        "事件→地点严格限于 hasPlace / tookPlaceAt / happenedIn。"
        "地图默认底图为“Carto·灰（无注记）”；已移除主视域选择与遮罩。"
    )
