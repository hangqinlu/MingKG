# -*- coding: utf-8 -*-
"""
殿试统计与溯源（固定年份轴｜年号着色与括揽｜人物筛选：地理+人物属性）
主入口兼容：封装 run()，移除模块层 set_page_config 与立即执行。其余处理逻辑保持不变。
依赖：pip install streamlit rdflib pandas plotly opencc-python-reimplemented
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
from collections import defaultdict, deque
import re, unicodedata

import streamlit as st
import pandas as pd
import plotly.express as px
from rdflib import Graph, URIRef, RDF, Literal
from opencc import OpenCC

# ============== 样式/字体（page_config 交由主入口设置） ==============
FONT_FALLBACK = (
    "Noto Sans CJK SC, Source Han Sans SC, Microsoft YaHei UI, Microsoft YaHei, "
    "PingFang SC, Hiragino Sans GB, WenQuanYi Micro Hei, SimHei, Arial Unicode MS, Arial, sans-serif"
)
PLOT_FONT = dict(family=FONT_FALLBACK, size=15)

# ============== 基本配置 ==============
DEFAULT_DATA = r"C:\Users\卢航青\Desktop\本体结构\ontology_appt_merged.nt"

# 类 / 谓词
CLASS_PERSON        = "Person"
CLASS_PARTICIPATION = "ParticipationEvent"
CLASS_IMPERIAL_EXAM = "ImperialExam"
CLASS_BIRTH         = "BirthEvent"
CLASS_PLACE         = "Place"

OP_PARTICIPATES     = "participatesIn"     # Person <-> ParticipationEvent
OP_HAS_EXAM         = "hasExam"            # ParticipationEvent -> ImperialExam
OP_BORN_IN_EVENT    = "bornInEvent"        # Person -> BirthEvent
OP_EVENT_HAS_PLACE  = "hasPlace"           # BirthEvent -> Place
OP_SUB_PLACE        = "isSubPlaceOf"       # Place 层级

# 溯源键名（兼容多写法）
NS_ABOUT, NS_DERIVED_FROM, NS_CONTAINS = "about", "derivedFrom", "contains"
NS_TP_BODY, NS_TP_CONF = "Text_body", "record_confidence"
NS_TP_SOURCE_KEYS = ["Text_Source","Text_source","来源","书名","source","Source","book","Book","Title","题名"]

# Place 属性
DP_PLACE_MOD_NAME, DP_PLACE_HIS_NAME, DP_PLACE_ADMIN_LEVEL = "现代名称", "历史名称", "现代区划层级"

# 人名属性
PERSON_NAME_KEYS = ["姓名","name","label","rdfs_label","标题","title"]

# 人物属性筛选（数据属性/PropAssertion）
DP_ACADEMIC = "学术专长"
DP_HUJI     = "户籍类型"
PA_SCHOOL_TYPE_KEYS = {"学籍类型","學籍類型","学籍_类型","學籍_類型","类型","類型"}
NS = "http://mingkg.org/ontology/imperial_exam.owl#"
P_ABOUT = URIRef(NS + "about")
P_PROP  = URIRef(NS + "prop")
P_VALN  = URIRef(NS + "value_norm")
P_VAL   = URIRef(NS + "value")

# 殿试识别
PALACE_LEVEL_KEYS   = ["考试等级","科举等级","考试级别","等级","level","Level","examLevel","ExamLevel"]
PALACE_LEVEL_TOKENS = ["殿试","殿","Palace","御试","廷对"]

# ============== 工具 ==============
CC_T2S = OpenCC('t2s')
CC_S2T = OpenCC('s2t')
def to_s(s: str) -> str: return CC_T2S.convert(s or "")
def to_t(s: str) -> str: return CC_S2T.convert(s or "")

def norm_text(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch for ch in s if not ch.isspace() and ch not in ("\u200b","\u200c","\u200d","\ufeff"))

def localname(u: Union[URIRef, str]) -> str:
    s = str(u)
    for sep in ("#", "/", ":"):
        if sep in s: s = s.rsplit(sep, 1)[-1]
    return s

def is_instance_of(g: Graph, inst: URIRef, class_local: str) -> bool:
    for t in g.objects(inst, RDF.type):
        if localname(t) == class_local: return True
    return False

def get_literals(g: Graph, node: URIRef) -> Dict[str, List[str]]:
    res: Dict[str, List[str]] = {}
    for p, o in g.predicate_objects(node):
        if isinstance(o, Literal):
            res.setdefault(localname(p), []).append(str(o).strip())
    return res

def first_lit(props: Dict[str, List[str]], keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in props and props[k]:
            for v in props[k]:
                if v: return v
    for vs in props.values():
        for v in vs:
            if v: return v
    return default

def person_display_name(g: Graph, person_uri: str) -> str:
    props = get_literals(g, URIRef(person_uri))
    for k in PERSON_NAME_KEYS:
        if k in props and props[k]: return props[k][0]
    return localname(URIRef(person_uri))

def highlight_html(text: str, terms: List[str]) -> str:
    if not text: return ""
    t = text
    uniq_terms = sorted({*terms,*[to_s(x) for x in terms],*[to_t(x) for x in terms]}, key=len, reverse=True)
    for v in uniq_terms:
        if v: t = t.replace(v, f"<span style='background:#fde68a;padding:0 2px;border-radius:3px'>{v}</span>")
    return t

# ============== 年号与固定举办年 ==============
MING_ERA_MAP: List[Tuple[str, int, int]] = [
    ("洪武", 1368, 1398), ("建文", 1399, 1402), ("永乐", 1403, 1424), ("洪熙", 1425, 1425),
    ("宣德", 1426, 1435), ("正统", 1436, 1449), ("景泰", 1450, 1456), ("天顺", 1457, 1464),
    ("成化", 1465, 1487), ("弘治", 1488, 1505), ("正德", 1506, 1521), ("嘉靖", 1522, 1566),
    ("隆庆", 1567, 1572), ("万历", 1573, 1620), ("泰昌", 1620, 1620), ("天启", 1621, 1627),
    ("崇祯", 1628, 1644),
]
def era_of_year(year: int) -> Optional[str]:
    if year == 1620: return "泰昌"
    for era, a, b in MING_ERA_MAP:
        if a <= year <= b: return era
    return None

HOSTED_YEARS = [
    1385, 1388, 1391, 1394, 1397,
    1400,
    1404, 1406, 1411, 1414, 1415, 1418, 1421,
    1427, 1430, 1433,
    1436, 1439, 1442, 1445, 1448,
    1451, 1454,
    1457, 1460, 1464,
    1466, 1469, 1472, 1475, 1478, 1481, 1484, 1487,
    1488, 1490, 1493, 1496, 1499, 1502, 1505,
    1508, 1511, 1514, 1517, 1521,
    1523, 1526, 1529, 1532, 1535, 1538, 1541, 1544, 1547, 1550, 1553, 1556, 1559, 1562, 1565,
    1568, 1571,
    1574, 1577, 1580, 1583, 1586, 1589, 1592, 1595, 1598, 1601, 1604, 1607, 1610, 1613, 1616, 1619,
    1622, 1625,
    1628, 1631, 1634, 1637, 1640, 1643
]
ERA_HOSTED = {
    "洪武":[1385,1388,1391,1394,1397],
    "建文":[1400],
    "永乐":[1404,1406,1411,1414,1415,1418,1421],
    "洪熙":[],
    "宣德":[1427,1430,1433],
    "正统":[1436,1439,1442,1445,1448],
    "景泰":[1451,1454],
    "天顺":[1457,1460,1464],
    "成化":[1466,1469,1472,1475,1478,1481,1484,1487],
    "弘治":[1488,1490,1493,1496,1499,1502,1505],
    "正德":[1508,1511,1514,1517,1521],
    "嘉靖":[1523,1526,1529,1532,1535,1538,1541,1544,1547,1550,1553,1556,1559,1562,1565],
    "隆庆":[1568,1571],
    "万历":[1574,1577,1580,1583,1586,1589,1592,1595,1598,1601,1604,1607,1610,1613,1616,1619],
    "泰昌":[],
    "天启":[1622,1625],
    "崇祯":[1628,1631,1634,1637,1640,1643],
}

# ============== 缓存：加载与索引 ==============
@st.cache_resource(show_spinner=False)
def load_graph(path: str) -> Graph:
    fmt = {".nt":"nt",".ttl":"turtle",".rdf":"xml",".owl":"xml",".xml":"xml"}.get(Path(path).suffix.lower(), None)
    g = Graph(); g.parse(path, format=(fmt or "nt"))
    return g

@st.cache_data(show_spinner=True)
def build_indices(path: str):
    g = load_graph(path)

    # 实例收集
    persons, particip, exams, births, places = set(), set(), set(), set(), set()
    for s, t in g.subject_objects(RDF.type):
        if not isinstance(s, URIRef): continue
        ln = localname(t)
        if ln == CLASS_PERSON: persons.add(s)
        elif ln == CLASS_PARTICIPATION: particip.add(s)
        elif ln == CLASS_IMPERIAL_EXAM: exams.add(s)
        elif ln == CLASS_BIRTH: births.add(s)
        elif ln == CLASS_PLACE: places.add(s)

    # 谓词索引
    pred_index_raw: Dict[str, Set[URIRef]] = defaultdict(set)
    for s, p, o in g.triples((None, None, None)):
        pred_index_raw[localname(p)].add(p)
    DP_SRC_SET = set()
    for key in NS_TP_SOURCE_KEYS:
        DP_SRC_SET |= set(pred_index_raw.get(key, []))
    pred_index = {
        "ABOUT": set(pred_index_raw.get(NS_ABOUT, [])),
        "DFROM": set(pred_index_raw.get(NS_DERIVED_FROM, [])),
        "CONTAINS": set(pred_index_raw.get(NS_CONTAINS, [])),
        "BODY": set(pred_index_raw.get(NS_TP_BODY, [])),
        "CONF": set(pred_index_raw.get(NS_TP_CONF, [])),
        "SRC_SET": DP_SRC_SET,
        "DP_ACADEMIC": set(pred_index_raw.get(DP_ACADEMIC, [])),
        "DP_HUJI": set(pred_index_raw.get(DP_HUJI, [])),
    }

    # 年份解析与殿试判定
    YEAR_RE = re.compile(r"(1[0-9]{3})")
    def parse_exam_year_from_node(ex: URIRef) -> Optional[int]:
        props = get_literals(g, ex)
        for vs in props.values():
            for t in vs:
                for m in YEAR_RE.findall(t or ""):
                    try: return int(m)
                    except: pass
        for m in YEAR_RE.findall(localname(ex)):
            try: return int(m)
            except: pass
        return None

    def is_palace_exam(ex: URIRef) -> bool:
        props = get_literals(g, ex)
        for k in PALACE_LEVEL_KEYS:
            for v in props.get(k, []):
                if any(tok in (v or "") for tok in PALACE_LEVEL_TOKENS):
                    return True
        ln = localname(ex)
        return ("Palace" in ln) or ("殿" in ln) or ("御试" in ln) or ("廷对" in ln)

    # Place 属性与层级
    def canon_level_suffix(s: str) -> str:
        return {"市辖区":"区","自治县":"县","自治州":"州","自治旗":"旗","嘎查村":"村"}.get((s or "").strip(), (s or "").strip())

    def _clean(s: str) -> str: return re.sub(r"[·•·\s、，,；;]+", "", (s or "").strip())

    def last_unit_suffix_from_full_name(fullname: str) -> str:
        s = _clean(fullname)
        sufs = ["特别行政区","自治区","自治州","自治县","自治旗","地区","市辖区","省","市","州","盟","县","区","旗",
                "乡","镇","街道","民族乡","苏木","社区","村","嘎查","嘎查村","居委会","村委会"]
        sufs = sorted(set(sufs), key=len, reverse=True)
        buf, tokens = "", []
        for ch in s:
            buf += ch
            for suf in sufs:
                if buf.endswith(suf):
                    tokens.append(buf); buf = ""; break
        if buf: tokens.append(buf)
        if not tokens: return ""
        last = tokens[-1]
        for suf in sufs:
            if last.endswith(suf): return canon_level_suffix(suf)
        return ""

    place_mod: Dict[str, str] = {}
    place_lvl: Dict[str, str] = {}
    for pl in places:
        props = get_literals(g, pl)
        mod = first_lit(props, [DP_PLACE_MOD_NAME, DP_PLACE_HIS_NAME], "") or localname(pl)
        lvl = first_lit(props, [DP_PLACE_ADMIN_LEVEL], "")
        if not lvl and mod: lvl = last_unit_suffix_from_full_name(mod)
        place_mod[str(pl)] = mod
        place_lvl[str(pl)] = canon_level_suffix(lvl)

    # 层级边（方向自适配）
    parent2children: Dict[str, List[str]] = defaultdict(list)
    child2parents: Dict[str, List[str]] = defaultdict(list)
    for s, p, o in g.triples((None, None, None)):
        if not (isinstance(s, URIRef) and isinstance(o, URIRef)): continue
        if localname(p) == OP_SUB_PLACE and is_instance_of(g, s, CLASS_PLACE) and is_instance_of(g, o, CLASS_PLACE):
            parent2children[str(o)].append(str(s))
            child2parents[str(s)].append(str(o))
    def _avg_outdeg(d: Dict[str, List[str]]) -> float:
        if not d: return 0.0
        return sum(len(v) for v in d.values())/max(1,len(d))
    if _avg_outdeg(parent2children) < 0.1:
        parent2children.clear(); child2parents.clear()
        for s, p, o in g.triples((None, None, None)):
            if not (isinstance(s, URIRef) and isinstance(o, URIRef)): continue
            if localname(p) == OP_SUB_PLACE and is_instance_of(g, s, CLASS_PLACE) and is_instance_of(g, o, CLASS_PLACE):
                parent2children[str(s)].append(str(o))  # s=parent
                child2parents[str(o)].append(str(s))     # o=child

    # 出生地（精确链）
    person_birth_place: Dict[str, str] = {}
    for person in persons:
        pid = str(person)
        be_nodes: Set[URIRef] = set()
        for p, e in g.predicate_objects(person):
            if localname(p) == OP_BORN_IN_EVENT and isinstance(e, URIRef) and is_instance_of(g, e, CLASS_BIRTH):
                be_nodes.add(e)
        for e, p in g.subject_predicates(person):
            if localname(p) == OP_BORN_IN_EVENT and isinstance(e, URIRef) and is_instance_of(g, e, CLASS_BIRTH):
                be_nodes.add(e)
        target_place = None
        for e in be_nodes:
            for p, pl in g.predicate_objects(e):
                if localname(p) == OP_EVENT_HAS_PLACE and isinstance(pl, URIRef) and is_instance_of(g, pl, CLASS_PLACE):
                    target_place = str(pl); break
            if target_place: break
        if target_place: person_birth_place[pid] = target_place

    # Participation → Palace Exam；Exam 年份
    person2parts: Dict[str, Set[str]] = defaultdict(set)
    for pe in particip:
        peid = str(pe)
        for person, p2 in g.subject_predicates(pe):
            if isinstance(person, URIRef) and localname(p2) == OP_PARTICIPATES and is_instance_of(g, person, CLASS_PERSON):
                person2parts[str(person)].add(peid)
        for p2, person in g.predicate_objects(pe):
            if localname(p2) == OP_PARTICIPATES and isinstance(person, URIRef) and is_instance_of(g, person, CLASS_PERSON):
                person2parts[str(person)].add(peid)

    part2exams_palace: Dict[str, List[str]] = defaultdict(list)
    exam2year: Dict[str, Optional[int]] = {}
    for pe in particip:
        peid = str(pe)
        for p, ex in g.predicate_objects(pe):
            if localname(p) == OP_HAS_EXAM and isinstance(ex, URIRef) and is_instance_of(g, ex, CLASS_IMPERIAL_EXAM):
                if is_palace_exam(ex):
                    exid = str(ex)
                    part2exams_palace[peid].append(exid)
                    if exid not in exam2year:
                        exam2year[exid] = parse_exam_year_from_node(ex)

    # 层级选择源
    level_to_places: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for uri, mod in place_mod.items():
        lvl = (place_lvl.get(uri, "") or "").strip()
        if not mod or not lvl: continue
        level_to_places[lvl].append((mod, uri))
    for lvl in level_to_places:
        level_to_places[lvl].sort(key=lambda x: x[0])

    # 人物属性索引（直连 + PropAssertion 学籍）
    def is_lit(x) -> bool: return isinstance(x, Literal) and str(x).strip() != ""
    def all_literals(node: URIRef, pred_set: Set[URIRef]) -> List[str]:
        vals, seen = [], set()
        for p in pred_set:
            for lit in g.objects(node, p):
                if is_lit(lit):
                    s = str(lit).strip()
                    if s and s not in seen: seen.add(s); vals.append(s)
        return vals

    def neighbors_one_hop(node: URIRef) -> Set[URIRef]:
        out = set()
        for p, o in g.predicate_objects(node):
            if isinstance(o, URIRef): out.add(o)
        for s, p in g.subject_predicates(node):
            if isinstance(s, URIRef): out.add(s)
        return out

    def school_types_of_person(person: URIRef) -> List[str]:
        out, seen = [], set()
        def collect_from(n: URIRef):
            for pa in g.subjects(P_ABOUT, n):
                props_txt = [str(l).strip() for l in g.objects(pa, P_PROP) if is_lit(l)]
                if not props_txt: continue
                ok = any((t in PA_SCHOOL_TYPE_KEYS) or (("学籍" in t or "學籍" in t) and ("类型" in t or "類型" in t)) for t in props_txt)
                if not ok: continue
                val = next((str(l).strip() for l in g.objects(pa, P_VALN) if is_lit(l)), "") \
                    or next((str(l).strip() for l in g.objects(pa, P_VAL) if is_lit(l)), "")
                if not val: continue
                k = norm_text(to_s(val))
                if k and k not in seen:
                    seen.add(k); out.append(val)
        collect_from(n=person)
        for nb in neighbors_one_hop(person):
            collect_from(nb)
        return out

    persons_all = set(str(x) for x in persons)
    person_academic: Dict[str, List[str]] = {}
    person_huji: Dict[str, List[str]]     = {}
    person_school: Dict[str, List[str]]   = {}
    opts_academic, opts_huji, opts_school = set(), set(), set()

    for s in persons:
        pid = str(s)
        ac = all_literals(s, pred_index["DP_ACADEMIC"])
        hu = all_literals(s, pred_index["DP_HUJI"])
        sc = school_types_of_person(s)
        person_academic[pid] = ac
        person_huji[pid]     = hu
        person_school[pid]   = sc
        opts_academic.update(ac); opts_huji.update(hu); opts_school.update(sc)

    return {
        "g": g,
        "place_mod": place_mod,
        "place_lvl": place_lvl,
        "parent2children": parent2children,
        "child2parents": child2parents,
        "person_birth_place": person_birth_place,
        "person2parts": person2parts,
        "part2exams_palace": part2exams_palace,
        "exam2year": exam2year,
        "pred_index": pred_index,
        "level_to_places": level_to_places,
        # 人物属性
        "person_academic": person_academic,
        "person_huji": person_huji,
        "person_school": person_school,
        "opts_academic": sorted(opts_academic),
        "opts_huji": sorted(opts_huji),
        "opts_school": sorted(opts_school),
    }

# ============== 溯源（人 × 指定殿试集合） ==============
def collect_provenance_for_person_and_exam_years(
    g: Graph,
    pred_index: Dict[str, Set[URIRef]],
    person_id: str,
    exams_of_interest: Set[str],   # 空=不限
):
    P_ABOUT, P_DFROM, P_CONTAINS = pred_index["ABOUT"], pred_index["DFROM"], pred_index["CONTAINS"]
    DP_BODY, DP_CONF, DP_SRC_SET = pred_index["BODY"], pred_index["CONF"], set(pred_index["SRC_SET"])

    pname = person_display_name(g, person_id)
    key_s = to_s(re.sub(r"\s+", "", pname or ""))

    def tp_fields(tp: URIRef) -> Tuple[str,str,str]:
        src, body, conf = "", "", ""
        for p, o in g.predicate_objects(tp):
            if isinstance(o, Literal):
                if (p in DP_BODY) and not body: body = str(o).strip()
                if (p in DP_CONF) and not conf: conf = str(o).strip()
                if (p in DP_SRC_SET) and not src: src = str(o).strip()
        return (src or "（未知书目）"), body, conf

    def body_contains_person(body: str) -> bool:
        return key_s in to_s(re.sub(r"\s+", "", body or ""))

    targets_exams: Set[URIRef] = set()
    for p, pe in g.predicate_objects(URIRef(person_id)):
        if isinstance(pe, URIRef) and localname(p) == OP_PARTICIPATES and is_instance_of(g, pe, CLASS_PARTICIPATION):
            for p2, ex in g.predicate_objects(pe):
                if localname(p2) == OP_HAS_EXAM and isinstance(ex, URIRef):
                    targets_exams.add(ex)
    for pe, p in g.subject_predicates(URIRef(person_id)):
        if isinstance(pe, URIRef) and localname(p) == OP_PARTICIPATES and is_instance_of(g, pe, CLASS_PARTICIPATION):
            for p2, ex in g.predicate_objects(pe):
                if localname(p2) == OP_HAS_EXAM and isinstance(ex, URIRef):
                    targets_exams.add(ex)
    if exams_of_interest:
        targets_exams = {URIRef(e) for e in exams_of_interest}

    groups: Dict[Tuple[str,str], Dict] = {}
    def upsert(src: str, body: str, conf: str, exam_uri: Optional[str]):
        if not body or not body_contains_person(body): return
        key = (src.strip(), body.strip())
        cur = groups.get(key)
        if cur is None:
            groups[key] = {"src": src or "（未知书目）", "body": body, "conf": conf, "exams": set()}
        if exam_uri: groups[key]["exams"].add(exam_uri)
        try:
            if conf and (not groups[key]["conf"] or float(conf) > float(groups[key]["conf"])): groups[key]["conf"] = conf
        except: pass

    for ex in targets_exams:
        for P in P_CONTAINS:
            for tp in g.subjects(P, ex):
                src, body, conf = tp_fields(tp); upsert(src, body, conf, str(ex))
        for P in P_ABOUT:
            for pa in g.subjects(P, ex):
                for P2 in P_DFROM:
                    for tp in g.objects(pa, P2):
                        src, body, conf = tp_fields(tp); upsert(src, body, conf, str(ex))

    out = sorted(
        [{"src":k[0], "body":k[1], "conf":v["conf"], "exams":sorted(v["exams"])} for k, v in groups.items()],
        key=lambda d: (d["src"], d["body"])
    )
    return out

# ============== 主入口（UI 执行） ==============
def run(default_path: Optional[str] = None):
    """在主入口 app.py 中调用：run()。默认路径可通过参数覆盖。"""
    # —— 页面样式注入（保持原 UI 外观） —— #
    st.markdown(
        f"""
        <style>
          html, body, [class*="css"], .stApp, .stMarkdown, .stText, .stCaption, .stExpander, .stButton, .stTextInput, .stSelectbox {{
            font-family: {FONT_FALLBACK} !important;
          }}
          .year-list-box {{
            margin-top:.25rem; padding:.5rem .7rem; background:#f0f9ff;
            border:1px solid #e0f2fe; border-radius:10px;
          }}
          .quote-box {{
            padding:.6rem .75rem; border-left:4px solid #dbeafe; background:#f8fafc;
            border-radius:8px; line-height:1.6;
          }}
          .person-card {{
            padding:.8rem 1rem; border-radius:12px; border:1px solid #e5e7eb; background:#fafafa;
          }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # ============== 侧边栏（参数与阈值） ==============
    with st.sidebar:
        st.header("数据与参数")
        DATA_PATH = st.text_input("RDF 路径（NT/TTL/RDF/OWL）", value=(default_path or DEFAULT_DATA))
        chart_mode = st.radio("主图模式", ["具体年份", "年号汇总"], index=0)
        contradiction_mode = st.selectbox("矛盾处理", ["排除矛盾","包括矛盾"], index=0,
                                          help="矛盾：同一人关联≥2个殿试实例。")
        st.markdown("---")
        st.header("阈值")
        enable_person_filters = st.checkbox("启用（人物·地域）", value=False)
        # 先放置占位变量（未启用时保持默认值）
        sel_lvl = None; selected_region_node = None; region_title = "全域"
        sel_school, sel_acad, sel_huji = [], [], []
        if st.button("清缓存重建", use_container_width=True, type="primary"):
            st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

    # ============== 加载索引后，渲染筛选控件 ==============
    state = build_indices(DATA_PATH)
    g                 = state["g"]
    place_mod         = state["place_mod"]
    parent2children   = state["parent2children"]
    person_birth      = state["person_birth_place"]
    person2parts      = state["person2parts"]
    part2exams_palace = state["part2exams_palace"]
    exam2year         = state["exam2year"]
    pred_index        = state["pred_index"]
    level_to_places   = state["level_to_places"]

    opts_school       = state["opts_school"]
    opts_academic     = state["opts_academic"]
    opts_huji         = state["opts_huji"]

    if enable_person_filters:
        with st.sidebar:
            st.caption("已启用人物筛选。")
            with st.expander("地域条件（出生地）", expanded=True):
                LEVEL_ORDER = ["省","特别行政区","自治区","市","州","盟","地区","县","区","旗","林区","特区","新区","乡","镇","街道","地区办事处","社区","村"]
                existing_lvls = [lvl for lvl in LEVEL_ORDER if lvl in level_to_places] or sorted(level_to_places.keys())
                sel_lvl = st.selectbox("区划层级", existing_lvls, index=0 if existing_lvls else 0, key="sel_lvl_geo")
                options = level_to_places.get(sel_lvl, [])
                mod2uri = {mod: uri for mod, uri in options}
                sel_mod = st.selectbox("现代名称", list(mod2uri.keys()) if options else ["（本层级暂无）"], key="sel_mod_geo")
                selected_region_node = mod2uri.get(sel_mod)
                region_title = sel_mod or "（未选）"
            with st.expander("人物条件", expanded=True):
                sel_school = st.multiselect("学籍类型", options=opts_school, default=[], key="sel_school")
                sel_acad   = st.multiselect("学术专长", options=opts_academic, default=[], key="sel_acad")
                sel_huji   = st.multiselect("户籍类型", options=opts_huji, default=[], key="sel_huji")

    # ============== 地域闭包与人物集合 ==============
    def closure_descendants(root_id: str) -> Set[str]:
        if not root_id: return set()
        out = {root_id}
        q, seen = deque([root_id]), {root_id}
        while q:
            u = q.popleft()
            for v in parent2children.get(u, []):
                if v not in seen:
                    seen.add(v); out.add(v); q.append(v)
        return out

    if enable_person_filters and selected_region_node:
        region_nodes = closure_descendants(selected_region_node)
        title_scope = f"{region_title}"
    else:
        region_nodes = set(place_mod.keys())  # 全域
        title_scope = "全域"

    # 地理过滤
    people_geofiltered: Set[str] = {pid for pid, bpl in person_birth.items() if bpl in region_nodes}

    # 人物属性过滤
    person_academic = state["person_academic"]
    person_huji     = state["person_huji"]
    person_school   = state["person_school"]

    def pass_attr(pid: str) -> bool:
        if not enable_person_filters:
            return True
        if sel_school:
            vals = [norm_text(to_s(v)) for v in person_school.get(pid, [])]
            if not any(norm_text(to_s(x)) in vals for x in sel_school): return False
        if sel_acad:
            vals = [norm_text(to_s(v)) for v in person_academic.get(pid, [])]
            if not any(norm_text(to_s(x)) in vals for x in sel_acad): return False
        if sel_huji:
            vals = [norm_text(to_s(v)) for v in person_huji.get(pid, [])]
            if not any(norm_text(to_s(x)) in vals for x in sel_huji): return False
        return True

    people_in_region: Set[str] = {pid for pid in people_geofiltered if pass_attr(pid)}

    # ============== 逐人→殿试→年份（主图用固定轴） ==============
    person_exam_years: Dict[str, Set[int]] = defaultdict(set)
    person2exams_ids: Dict[str, Set[str]] = defaultdict(set)
    unknown_year_people: Set[str] = set()

    for pid in people_in_region:
        for peid in person2parts.get(pid, set()):
            for exid in part2exams_palace.get(peid, []):
                person2exams_ids[pid].add(exid)
                y = exam2year.get(exid)
                if y is None:
                    unknown_year_people.add(pid)
                else:
                    person_exam_years[pid].add(y)

    # 矛盾集合（一个人多场殿试）
    conflict_people: Set[str] = {pid for pid, exs in person2exams_ids.items() if len(exs) >= 2}
    exclude_conflict = (contradiction_mode == "排除矛盾")

    # 年份 → 去重人（仅 HOSTED_YEARS）
    year2people: Dict[int, Set[str]] = defaultdict(set)
    for pid, years in person_exam_years.items():
        for y in years:
            if y in HOSTED_YEARS:
                year2people[y].add(pid)

    def _count_year_people(y: int) -> int:
        ppl = set(year2people.get(y, set()))
        if exclude_conflict: ppl -= conflict_people
        return len(ppl)

    # ============== 主图 ==============
    if chart_mode == "具体年份":
        st.title(f"科举项计量")
        df_bar = pd.DataFrame({
            "年份": HOSTED_YEARS,
            "人数": [_count_year_people(y) for y in HOSTED_YEARS],
            "年号": [era_of_year(y) for y in HOSTED_YEARS],
        })
        fig = px.bar(
            df_bar, x="年份", y="人数", color="年号", text="人数",
            title=f"{title_scope} · 各举办年份殿试人数（{contradiction_mode}）"
        )
        fig.update_traces(textposition="outside", hovertemplate="年份=%{x}<br>人数=%{y}<extra></extra>")

        # 年号括揽（灰底 + 顶部标签）
        shapes, annotations = [], []
        era_buckets: Dict[str, List[int]] = defaultdict(list)
        for y in HOSTED_YEARS:
            e = era_of_year(y)
            if e: era_buckets[e].append(y)
        for era, ys in era_buckets.items():
            if not ys: continue
            x0, x1 = min(ys) - 0.45, max(ys) + 0.45
            shapes.append(dict(
                type="rect", xref="x", yref="paper", x0=x0, x1=x1, y0=0, y1=1,
                fillcolor="rgba(160,160,160,0.07)", line=dict(width=0), layer="below"
            ))
            annotations.append(dict(
                x=(min(ys)+max(ys))/2, y=1.06, xref="x", yref="paper",
                text=f"{era}", showarrow=False, font=dict(family=FONT_FALLBACK, size=14)
            ))
        fig.update_layout(
            xaxis_title=None, yaxis_title="人数（去重）",
            margin=dict(l=10,r=10,t=84,b=90),
            font=PLOT_FONT,
            xaxis=dict(tickmode="array", tickvals=HOSTED_YEARS, ticktext=[str(x) for x in HOSTED_YEARS]),
            uniformtext_minsize=10, uniformtext_mode="hide",
            legend_title="年号",
            shapes=shapes, annotations=annotations,
            bargap=0.25
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    else:
        st.title(f"殿试年号分布（年号汇总｜{title_scope}）")
        era_people_unique: Dict[str, Set[str]] = defaultdict(set)
        for y in HOSTED_YEARS:
            e = era_of_year(y)
            if e:
                ppl = set(year2people.get(y, set()))
                if exclude_conflict: ppl -= conflict_people
                era_people_unique[e] |= ppl

        unknown_people = []
        for pid in people_in_region:
            if pid in person2exams_ids and len(person_exam_years.get(pid, set()) & set(HOSTED_YEARS)) == 0:
                if (not exclude_conflict) or (pid not in conflict_people):
                    unknown_people.append(pid)

        ordered_eras = [e for e, a, b in MING_ERA_MAP]
        rows = [{"年号": e, "人数": len(era_people_unique.get(e, set()))} for e in ordered_eras]
        rows.append({"年号": "时间未知", "人数": len(unknown_people)})
        df_era_bar = pd.DataFrame(rows)
        fig = px.bar(
            df_era_bar, x="年号", y="人数", text="人数",
            title=f"{title_scope} · 各年号殿试关联人数（去重；{contradiction_mode}）"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            xaxis_title=None, yaxis_title="人数（去重）",
            margin=dict(l=10, r=10, t=60, b=70),
            font=PLOT_FONT,
            xaxis=dict(tickmode="array", tickvals=df_era_bar["年号"], ticktext=df_era_bar["年号"]),
            uniformtext_minsize=10, uniformtext_mode="hide",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    # ============== 左右两栏：左=年号分类（固定举办年），右=溯源 ==============
    left, right = st.columns([7,5], gap="large")

    with left:
        st.markdown("## 年号分类")
        st.caption("点击年号展开“小柱状图”（横轴为该年号的举办年；计数为 0 亦显示）。展开年份卡查看人物；点击人名于右侧显示溯源。含“时间未知”。")
        q_bucket = st.text_input("检索年号或年份（例：嘉靖 / 15；可留空）", value="").strip()

        # 年份 → 人物（去重；按矛盾策略）
        year_people_list: Dict[int, List[str]] = {}
        for y in HOSTED_YEARS:
            ppl = set(year2people.get(y, set()))
            if exclude_conflict: ppl -= conflict_people
            year_people_list[y] = sorted(list(ppl), key=lambda pid: person_display_name(g, pid))

        ordered_eras = [e for e, a, b in MING_ERA_MAP]
        for era in ordered_eras:
            years_axis = ERA_HOSTED.get(era, [])
            if q_bucket and (q_bucket not in era and not any(str(y).startswith(q_bucket) for y in years_axis)):
                continue

            era_unique: Set[str] = set()
            for y in years_axis:
                era_unique |= set(year_people_list.get(y, []))
            sub_title = f"{era}（{years_axis[0]}–{years_axis[-1]}） · {len(era_unique)}" if years_axis else f"{era}（无举办年份） · 0"

            with st.expander(sub_title, expanded=False):
                if not years_axis:
                    st.info("该年号未举办殿试。")
                    continue

                # 年号内小图（固定横轴）
                dfe = pd.DataFrame({"年份": years_axis, "人数": [len(year_people_list.get(y, [])) for y in years_axis]})
                dfe["年号"] = era
                chart = px.bar(
                    dfe, x="年份", y="人数", color="年号", text="人数",
                    title=f"{era} 年号内举办年份殿试人数（{title_scope}；{contradiction_mode}）"
                )
                chart.update_traces(textposition="outside", hovertemplate="年份=%{x}<br>人数=%{y}<extra></extra>")
                chart.update_layout(
                    xaxis_title=None, yaxis_title="人数（去重）",
                    margin=dict(l=10,r=10,t=60,b=50),
                    font=PLOT_FONT,
                    xaxis=dict(tickmode="array", tickvals=years_axis, ticktext=[str(x) for x in years_axis]),
                    uniformtext_minsize=10, uniformtext_mode="hide",
                    showlegend=False, bargap=0.25
                )
                st.plotly_chart(chart, use_container_width=True, config={"displaylogo": False})

                # 年份 → 人物
                st.markdown("<div class='year-list-box'>📅 <b>年份清单</b>（点击展开查看人物）</div>", unsafe_allow_html=True)
                for y in years_axis:
                    ppl = year_people_list.get(y, [])
                    with st.expander(f"🗓️ {y} 年 — 人数：{len(ppl)}（{contradiction_mode}）", expanded=False):
                        if not ppl:
                            st.info("该年暂无人物。")
                        else:
                            cols = st.columns(4)
                            for i, pid in enumerate(ppl):
                                name = person_display_name(g, pid)
                                if cols[i % 4].button(name, key=f"btn_person_{era}_{y}_{pid}"):
                                    st.session_state["sel_era"] = era
                                    st.session_state["sel_year"] = y
                                    st.session_state["sel_person"] = pid

        # 时间未知
        unk_people = []
        for pid in people_in_region:
            if pid in person2exams_ids and len(person_exam_years.get(pid, set()) & set(HOSTED_YEARS)) == 0:
                if (not exclude_conflict) or (pid not in conflict_people):
                    unk_people.append(pid)
        with st.expander(f"时间未知 · {len(unk_people)}", expanded=False):
            if not unk_people:
                st.info("暂无无法解析年份的殿试记录。")
            else:
                cols = st.columns(4)
                for i, pid in enumerate(sorted(unk_people, key=lambda x: person_display_name(g, x))[:400]):
                    if cols[i % 4].button(person_display_name(g, pid), key=f"btn_person_unknown_{pid}"):
                        st.session_state["sel_era"] = "时间未知"
                        st.session_state["sel_year"] = None
                        st.session_state["sel_person"] = pid

    with right:
        st.markdown("## 溯源")
        st.caption("按书目聚合高亮摘录（自动高亮人名）。")
        sel_era  = st.session_state.get("sel_era")
        sel_year = st.session_state.get("sel_year")
        sel_person = st.session_state.get("sel_person")

        if not sel_person:
            st.info("提示：在左侧选择人物以查看溯源。")
        else:
            pname = person_display_name(g, sel_person)
            exams_of_this_person = set()
            for peid in person2parts.get(sel_person, set()):
                exams_of_this_person |= set(part2exams_palace.get(peid, []))
            target_exams = set()
            if sel_year is not None:
                target_exams = {eid for eid in exams_of_this_person if exam2year.get(eid) == sel_year}

            provs = collect_provenance_for_person_and_exam_years(
                g=g, pred_index=pred_index, person_id=sel_person, exams_of_interest=target_exams
            )
            st.markdown(
                (
                    "<div class='person-card'><div style='font-size:1.05rem; font-weight:700;'>{}</div>"
                    "<div style='opacity:.75;'>{}{}</div></div>"
                ).format(pname, sel_era or "", ("" if sel_year is None else " · " + str(sel_year))),
                unsafe_allow_html=True
            )
            st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

            if not provs:
                st.info("未检索到溯源文本（或正文不含该人名）。")
            else:
                for i, ent in enumerate(provs[:80], 1):
                    src, body, conf = ent["src"], ent["body"], ent["conf"]
                    head = f"📘 书目：{src}" + (f"｜可信度：{conf}" if conf else "")
                    with st.expander(head, expanded=(i == 1)):
                        st.write("**摘录**：", unsafe_allow_html=True)
                        st.markdown(f"<div class='quote-box'>{highlight_html(body, [pname])}</div>", unsafe_allow_html=True)

    # ============== 说明 ==============
    st.markdown("---")
    st.caption(
        "流程：若启用人物筛选，则在左侧侧栏设置“出生地层级 + 学籍/学术专长/户籍”；"
        "逐人追 Participation→殿试 Exam→解析年份；主图与年号分类固定采用【举办年份轴】，颜色=年号，"
        "顶部灰底与年号标签实现“年号括揽”。"
    )
