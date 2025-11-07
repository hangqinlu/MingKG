# -*- coding: utf-8 -*-
"""
官职实例多维统计 + 溯源（交互柱状图）—— 主入口兼容版
保留原全部逻辑，仅将页面执行封装为 run()，移除模块层 set_page_config。
"""

from pathlib import Path
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional, Set, Union
import unicodedata, re

import streamlit as st
import pandas as pd
import plotly.express as px

from rdflib import Graph, URIRef, Literal, RDF
from rdflib.namespace import RDFS, SKOS, FOAF

# —— 画图字体（页面配置交由主入口） —— #
PLOT_FONT = dict(family="SimHei, Microsoft YaHei, Noto Sans CJK SC, Arial Unicode MS, Arial", size=14)

# —— 默认数据路径（可在侧栏覆盖） —— #
DEFAULT_DATA = r"C:\Users\卢航青\Desktop\本体结构\ontology_academic_fixed_positions_full.nt"

# —— 对象属性（localname） —— #
P_APPOINTED_IN      = "appointedIn"      # Person → AppointmentEvent
P_HAS_POSITION      = "hasPosition"      # AppointmentEvent → OfficialPosition

# —— 殿试链与出生地链（localname） —— #
OP_PARTICIPATES_IN  = "participatesIn"   # Person → ParticipationEvent
OP_HAS_EXAM         = "hasExam"          # ParticipationEvent → ImperialExam
DP_EXAM_LEVEL       = "考试等级"           # ImperialExam → Literal（殿试）
CLASS_IMPERIAL_EXAM = "ImperialExam"

OP_BORN_IN_EVENT    = "bornInEvent"      # Person → BirthEvent
OP_EVENT_HAS_PLACE  = "hasPlace"         # BirthEvent → Place
OP_SUB_PLACE        = "isSubPlaceOf"     # Place 层级
CLASS_PLACE         = "Place"
CLASS_BIRTH         = "BirthEvent"

# —— OfficialPosition 的常用数据属性（localname） —— #
DP_FIELDS_HUMAN = [
    "原始称谓", "官职名称", "核心职称", "层级", "通用层级", "机构", "职系",
    "修饰_方位", "修饰_副", "地名", "官阶",
]
DP_FIELDS_ALIGN = [
    "对齐码_core", "对齐码_inst", "对齐码_tier",
    "对齐码_loc_core", "对齐码_loc_inst", "对齐码_loc_full",
]

# —— 溯源谓词（真实 IRI） —— #
NS = "http://mingkg.org/ontology/imperial_exam.owl#"
P_ABOUT        = URIRef(NS + "about")
P_DERIVED_FROM = URIRef(NS + "derivedFrom")
P_CONTAINS     = URIRef(NS + "contains")
TP_CONF        = URIRef(NS + "record_confidence")
TP_SRC         = URIRef(NS + "Text_source")
TP_BODY        = URIRef(NS + "Text_body")
P_PROP         = URIRef(NS + "prop")
P_VALN         = URIRef(NS + "value_norm")
P_VAL          = URIRef(NS + "value")

# —— 人名属性 —— #
PERSON_NAME_KEYS = {
    "姓名", "name", "label", "rdfs_label", "标题", "title",
    str(FOAF.name), str(RDFS.label), str(SKOS.prefLabel)
}

# —— 年号映射（明） —— #
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

# —— 文本规整 / 繁简容错 —— #
_ZW = {u"\u200b", u"\u200c", u"\u200d", u"\ufeff"}
def norm(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if (not ch.isspace()) and (ch not in _ZW))
    return s

TS_MAP = {"蘇":"苏","劉":"刘","張":"张","趙":"赵","錢":"钱","孫":"孙","國":"国","會":"会","試":"试","鄉":"乡",
          "進":"进","舉":"举","階":"阶","級":"级","歷":"历","鄭":"郑","黃":"黄","萬":"万","陳":"陈","楊":"杨",
          "馬":"马","許":"许","鄧":"邓","吳":"吴","葉":"叶","羅":"罗","齊":"齐","祿":"禄","祯":"祯","禎":"祯"}
def t2s(s: str) -> str: return "".join(TS_MAP.get(ch, ch) for ch in (s or ""))

def localname(u: Union[URIRef, str]) -> str:
    s = str(u)
    if "#" in s: s = s.rsplit("#", 1)[-1]
    elif "/" in s: s = s.rsplit("/", 1)[-1]
    return s

def is_literal_nonempty(x) -> bool:
    return isinstance(x, Literal) and str(x).strip() != ""

# —— 对齐码中文化 —— #
INSTITUTION_ABBR = {
    "都察院": "JCY", "吏部": "LIBU", "户部": "HUBU", "礼部": "LIPU",
    "兵部": "BINGBU", "刑部": "XINGBU", "工部": "GONGBU",
    "太常寺": "TAICHANGSI", "光禄寺": "GUANGLUSI", "太仆寺": "TAIPUSI",
    "大理寺": "DALISI", "鸿胪寺": "HONGLUSI", "通政司": "TONGZHENGSI",
    "宗人府": "ZONGRENFU", "詹事府": "ZHANSHIFU", "国子监": "GUOZIJIAN",
    "布政使司": "BUZHENGSI", "按察使司": "ANCHASI",
    "五军都督府": "WJDD", "都督府": "DUDUFU",
    "都指挥使司": "DUZHIHUISI", "卫所": "WEISUO",
    "锦衣卫": "JINYIWEI", "散官": "SANGUAN",
}
ABBR2CN = {v: k for k, v in INSTITUTION_ABBR.items()}

def label_from_align(code: str, fallback_tier: str = "") -> str:
    if not code: return ""
    s = str(code).strip()
    try:
        if s.startswith("CORE/"):
            return s.split("/", 1)[1]
        if s.startswith("INST/"):
            _, inst, core = s.split("/", 2)
            return f"{ABBR2CN.get(inst, inst)} / {core}"
        if s.startswith("TIER/"):
            parts = s.split("/")
            inst = parts[1] if len(parts) > 1 else ""
            core = parts[2] if len(parts) > 2 else ""
            last = parts[3] if len(parts) > 3 else ""
            tier = fallback_tier or (last if last else "")
            inst_cn = ABBR2CN.get(inst, inst)
            return f"{inst_cn} / {core}" if not tier or tier == core else f"{inst_cn} / {core} / {tier}"
        if s.startswith("LOC_CORE/"):
            _, rest = s.split("/", 1)
            place, core = rest.split("/", 1)
            return f"{place.replace('Place#','')} / {core}"
        if s.startswith("LOC_INST/"):
            _, rest = s.split("/", 1)
            place, inst, core = rest.split("/")[:3]
            return f"{place.replace('Place#','')} / {ABBR2CN.get(inst, inst)} / {core}"
        if s.startswith("LOC_FULL/"):
            head = s.split(";")[0]
            parts = head.split("/")
            place = parts[1].replace("Place#","") if len(parts)>1 else ""
            inst  = parts[2] if len(parts)>2 else ""
            core  = parts[3] if len(parts)>3 else ""
            last  = parts[4] if len(parts)>4 else ""
            tier  = fallback_tier or (last if last else "")
            inst_cn = ABBR2CN.get(inst, inst)
            return f"{place} / {inst_cn} / {core}" if not tier or tier==core else f"{place} / {inst_cn} / {core} / {tier}"
    except:
        pass
    return s

def pos_label(pos_attrs: dict, op_iri: str, key: str) -> str:
    a = pos_attrs.get(op_iri, {})
    if key in {"对齐码_core","对齐码_inst","对齐码_tier","对齐码_loc_core","对齐码_loc_inst","对齐码_loc_full"}:
        tier = a.get("层级") or a.get("通用层级") or ""
        return label_from_align(a.get(key, ""), fallback_tier=tier)
    if key == "层级": return a.get("层级") or a.get("通用层级") or ""
    if key == "官阶": return a.get("官阶", "")
    return a.get(key, "") or ""

def dim2_label(pos_attrs: dict, op_iri: str, which: str) -> str:
    a = pos_attrs.get(op_iri, {})
    return label_from_align(a.get("对齐码_inst","")) if which=="机构×核心" else label_from_align(a.get("对齐码_loc_core",""))

def dim3_label(pos_attrs: dict, op_iri: str, which: str) -> str:
    a = pos_attrs.get(op_iri, {})
    if which=="机构×核心×层级":
        tier = a.get("层级") or a.get("通用层级") or ""
        return label_from_align(a.get("对齐码_tier",""), fallback_tier=tier)
    return label_from_align(a.get("对齐码_loc_inst",""))

def dim4_label(pos_attrs: dict, op_iri: str) -> str:
    a = pos_attrs.get(op_iri, {})
    tier = a.get("层级") or a.get("通用层级") or ""
    return label_from_align(a.get("对齐码_loc_full",""), fallback_tier=tier)

# —— 加载 —— #
@st.cache_resource(show_spinner=False)
def load_graph(path: str) -> Graph:
    fmt = {".nt":"nt",".ttl":"turtle",".rdf":"xml",".owl":"xml",".xml":"xml"}.get(Path(path).suffix.lower(), None)
    g = Graph(); g.parse(path, format=(fmt or "turtle"))
    return g

# —— 建索引 —— #
@st.cache_resource(show_spinner=True)
def build_indices(path: str):
    g = load_graph(path)

    pred_by_local: Dict[str, Set[URIRef]] = defaultdict(set)
    for _, p, _ in g.triples((None, None, None)):
        pred_by_local[localname(p)].add(p)

    preds_app_in = pred_by_local.get(P_APPOINTED_IN, set())
    preds_has_pos = pred_by_local.get(P_HAS_POSITION, set())

    def is_official(s: URIRef) -> bool:
        return any(localname(t)=="OfficialPosition" for _,_,t in g.triples((s, RDF.type, None)))

    pos_nodes: List[URIRef] = [s for s, t in g.subject_objects(RDF.type) if is_official(s)]

    def first_dp(subj: URIRef, local: str) -> str:
        for pred in pred_by_local.get(local, set()):
            for o in g.objects(subj, pred):
                if is_literal_nonempty(o): return str(o).strip()
        return ""

    pos_attrs: Dict[str, Dict[str, str]] = {}
    for op in pos_nodes:
        row = {}
        row["原始称谓"] = first_dp(op, "原始称谓") or first_dp(op, "官职名称")
        for k in DP_FIELDS_HUMAN:
            if k == "原始称谓": continue
            v = first_dp(op, k) or (first_dp(op, "通用层级") if k=="层级" else "")
            row[k] = v
        for k in DP_FIELDS_ALIGN:
            row[k] = first_dp(op, k)
        pos_attrs[str(op)] = row

    evt_to_pos: Dict[str, Set[str]] = defaultdict(set)
    for e, p, op in g.triples((None, None, None)):
        if p in preds_has_pos and isinstance(op, URIRef):
            evt_to_pos[str(e)].add(str(op))

    person_to_evts: Dict[str, Set[str]] = defaultdict(set)
    for person, p, evt in g.triples((None, None, None)):
        if p in preds_app_in and isinstance(evt, URIRef):
            person_to_evts[str(person)].add(str(evt))

    pos_persons_all: Dict[str, Set[str]] = defaultdict(set)
    for person, evts in person_to_evts.items():
        for e in evts:
            for op in evt_to_pos.get(e, set()):
                pos_persons_all[op].add(person)

    name_predicates: Set[URIRef] = set()
    for key in PERSON_NAME_KEYS:
        name_predicates |= pred_by_local.get(localname(URIRef(key)) if str(key).startswith("http") else key, set())

    def all_literals_by_local(node: URIRef, local: str) -> List[str]:
        vals, seen = [], set()
        for pred in pred_by_local.get(local, set()):
            for lit in g.objects(node, pred):
                if is_literal_nonempty(lit):
                    s = str(lit).strip()
                    if s and s not in seen:
                        seen.add(s); vals.append(s)
        return vals

    # 学籍类型（PropAssertion）
    PA_SCHOOL_TYPE_KEYS = {"学籍类型","學籍類型","学籍_类型","學籍_類型","类型","類型"}

    persons_all: Set[URIRef] = set(URIRef(p) for p in person_to_evts.keys())

    def one_hop_neighbors(person: URIRef) -> Set[URIRef]:
        out: Set[URIRef] = set()
        for p, o in g.predicate_objects(person):
            if isinstance(o, URIRef): out.add(o)
        for s, p in g.subject_predicates(person):
            if isinstance(s, URIRef): out.add(s)
        return out

    def pa_school_types_of_person(person: URIRef) -> List[str]:
        out, seen = [], set()
        def collect_from(node: URIRef):
            for pa in g.subjects(P_ABOUT, node):
                props_txt = [str(l).strip() for l in g.objects(pa, P_PROP) if is_literal_nonempty(l)]
                if not props_txt: continue
                ok = any((localname(s) in PA_SCHOOL_TYPE_KEYS) or (("学籍" in s or "學籍" in s) and ("类型" in s or "類型" in s)) for s in props_txt)
                if not ok: continue
                val = next((str(l).strip() for l in g.objects(pa, P_VALN) if is_literal_nonempty(l)), "") \
                      or next((str(l).strip() for l in g.objects(pa, P_VAL) if is_literal_nonempty(l)), "")
                if not val: continue
                k = norm(t2s(val))
                if k and k not in seen:
                    seen.add(k); out.append(val)
        collect_from(person)
        for nb in one_hop_neighbors(person):
            collect_from(nb)
        return out

    person_academic: Dict[str, List[str]]    = {}
    person_huji_type: Dict[str, List[str]]   = {}
    person_school_type: Dict[str, List[str]] = {}
    opts_academic, opts_huji_type, opts_school_type = set(), set(), set()

    for p in persons_all:
        vals_ac = all_literals_by_local(p, "学术专长")
        vals_hu = all_literals_by_local(p, "户籍类型")
        vals_sc = pa_school_types_of_person(p)

        person_academic[str(p)]    = vals_ac
        person_huji_type[str(p)]   = vals_hu
        person_school_type[str(p)] = vals_sc

        opts_academic.update(vals_ac)
        opts_huji_type.update(vals_hu)
        opts_school_type.update(vals_sc)

    # 殿试年份解析
    YEAR_RE = re.compile(r"(1[0-9]{3})")
    pred_part = pred_by_local.get(OP_PARTICIPATES_IN, set())
    pred_has_exam = pred_by_local.get(OP_HAS_EXAM, set())
    pred_exam_level = pred_by_local.get(DP_EXAM_LEVEL, set())

    p_to_pes: Dict[str, Set[str]] = defaultdict(set)
    pe_to_ex: Dict[str, str] = {}
    ex_to_level: Dict[str, str] = {}
    exam2year: Dict[str, Optional[int]] = {}

    for pred in pred_part:
        for s, _, pe in g.triples((None, pred, None)):
            if isinstance(s, URIRef) and isinstance(pe, URIRef):
                p_to_pes[str(s)].add(str(pe))
    for pred in pred_has_exam:
        for pe, _, ex in g.triples((None, pred, None)):
            if isinstance(pe, URIRef) and isinstance(ex, URIRef):
                pe_to_ex[str(pe)] = str(ex)
    for pred in pred_exam_level:
        for ex, _, lit in g.triples((None, pred, None)):
            if isinstance(ex, URIRef) and is_literal_nonempty(lit):
                ex_to_level[str(ex)] = str(lit).strip()

    def parse_exam_year(ex: URIRef) -> Optional[int]:
        for _, _, lit in g.triples((ex, None, None)):
            if is_literal_nonempty(lit):
                for m in YEAR_RE.findall(str(lit)):
                    try: return int(m)
                    except: pass
        for m in YEAR_RE.findall(localname(ex)):
            try: return int(m)
            except: pass
        return None

    for ex in set(URIRef(x) for x in pe_to_ex.values()):
        is_exam = any(localname(t) == CLASS_IMPERIAL_EXAM for _, _, t in g.triples((ex, RDF.type, None)))
        if not is_exam: continue
        exam2year[str(ex)] = parse_exam_year(ex)

    persons_with_dianshi: Set[str] = set()
    for pid in persons_all:
        sid = str(pid)
        ok = False
        for pe in p_to_pes.get(sid, set()):
            ex = pe_to_ex.get(pe)
            if not ex: continue
            if norm(t2s(ex_to_level.get(ex, ""))) == norm("殿试"):
                ok = True; break
        if ok: persons_with_dianshi.add(sid)

    # 地域索引（Place 名称、层级、isSubPlaceOf；Person 出生地）
    DP_PLACE_MOD_NAME, DP_PLACE_HIS_NAME, DP_PLACE_ADMIN_LEVEL = "现代名称", "历史名称", "现代区划层级"
    place_mod: Dict[str, str] = {}
    place_lvl: Dict[str, str] = {}

    def is_type(node: URIRef, local_type: str) -> bool:
        return any(localname(t)==local_type for _,_,t in g.triples((node, RDF.type, None)))

    def first_dp_local(s: URIRef, local: str) -> str:
        for pred in pred_by_local.get(local, set()):
            for o in g.objects(s, pred):
                if is_literal_nonempty(o): return str(o).strip()
        return ""

    for s, _, _ in g.triples((None, RDF.type, None)):
        if any(localname(t)==CLASS_PLACE for _,_,t in g.triples((s, RDF.type, None))):
            mod = first_dp_local(s, DP_PLACE_MOD_NAME) or first_dp_local(s, DP_PLACE_HIS_NAME) or localname(s)
            lvl = first_dp_local(s, DP_PLACE_ADMIN_LEVEL)
            place_mod[str(s)] = mod; place_lvl[str(s)] = lvl

    parent2children: Dict[str, Set[str]] = defaultdict(set)
    for s, p, o in g.triples((None, None, None)):
        if localname(p) == OP_SUB_PLACE and isinstance(s, URIRef) and isinstance(o, URIRef):
            parent2children[str(o)].add(str(s))

    person_birth_place: Dict[str, str] = {}
    for person in persons_all:
        target = None
        for p, e in g.predicate_objects(person):
            if localname(p) == OP_BORN_IN_EVENT and isinstance(e, URIRef) and is_type(e, CLASS_BIRTH):
                for p2, pl in g.predicate_objects(e):
                    if localname(p2) == OP_EVENT_HAS_PLACE and isinstance(pl, URIRef) and is_type(pl, CLASS_PLACE):
                        target = str(pl); break
            if target: break
        if not target:
            for e, p in g.subject_predicates(person):
                if localname(p) == OP_BORN_IN_EVENT and isinstance(e, URIRef) and is_type(e, CLASS_BIRTH):
                    for p2, pl in g.predicate_objects(e):
                        if localname(p2) == OP_EVENT_HAS_PLACE and isinstance(pl, URIRef) and is_type(pl, CLASS_PLACE):
                            target = str(pl); break
                if target: break
        if target: person_birth_place[str(person)] = target

    level_to_places: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for uri, mod in place_mod.items():
        lvl = (place_lvl.get(uri, "") or "").strip()
        if not mod: continue
        level_to_places[lvl].append((mod, uri))
    for k in list(level_to_places.keys()):
        level_to_places[k].sort(key=lambda x: x[0])

    single_dims = [
        ("职系", "职系"), ("机构", "机构"), ("核心职称", "核心职称"),
        ("层级", "层级"), ("官阶", "官阶"), ("地名", "地名"),
        ("对齐码_core", "对齐码_core"), ("对齐码_inst", "对齐码_inst"),
        ("对齐码_tier", "对齐码_tier"), ("对齐码_loc_core", "对齐码_loc_core"),
        ("对齐码_loc_inst", "对齐码_loc_inst"), ("对齐码_loc_full", "对齐码_loc_full"),
    ]

    return dict(
        g=g,
        pos_attrs=pos_attrs,
        pos_persons_all={k:list(v) for k,v in pos_persons_all.items()},
        single_dims=single_dims,
        pred_by_local={k:list(v) for k,v in pred_by_local.items()},
        name_predicates=[str(p) for p in name_predicates],
        person_academic=person_academic, person_huji_type=person_huji_type, person_school_type=person_school_type,
        opts_academic=sorted(opts_academic), opts_huji_type=sorted(opts_huji_type), opts_school_type=sorted(opts_school_type),
        p_to_pes={k:list(v) for k,v in p_to_pes.items()}, pe_to_ex=pe_to_ex, ex_to_level=ex_to_level, exam2year=exam2year,
        persons_with_dianshi=list(persons_with_dianshi),
        place_mod=place_mod, place_lvl=place_lvl, parent2children={k:list(v) for k,v in parent2children.items()},
        person_birth_place=person_birth_place, level_to_places={k:v for k,v in level_to_places.items()},
    )

# —— 页面绘图 —— #
def plot_bar(df: pd.DataFrame, topn: int = 30, title: str = "") -> bool:
    if df.empty:
        st.info("无统计结果。"); return False
    d = df.copy().sort_values("人数", ascending=False).head(int(topn))
    d["类别显示"] = d["类别"].map(lambda s: s if len(s)<=46 else s[:45]+"…")
    fig = px.bar(d, x="类别显示", y="人数", text="人数", title=title)
    fig.update_traces(textposition="outside")
    fig.update_layout(
        xaxis_title=None, yaxis_title="人数（去重）",
        uniformtext_minsize=10, uniformtext_mode="hide",
        margin=dict(l=10,r=10,t=60,b=90),
        font=PLOT_FONT, template="plotly_white",
        xaxis=dict(tickangle=25, categoryorder="total descending"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    return True

# —— 溯源工具（运行期依赖 g） —— #
def provenance_for_official_position(g: Graph, op_iri: str) -> List[Dict]:
    TP_SRC_R, TP_BODY_R, TP_CONF_R = URIRef(NS + "Text_source"), URIRef(NS + "Text_body"), URIRef(NS + "record_confidence")
    P_ABOUT_R, P_DERIVED_R, P_CONTAINS_R = URIRef(NS + "about"), URIRef(NS + "derivedFrom"), URIRef(NS + "contains")

    def tp_fields(tp: URIRef) -> Tuple[str, str, str]:
        src  = next((str(x).strip() for x in g.objects(tp, TP_SRC_R)  if is_literal_nonempty(x)), "")
        body = next((str(x).strip() for x in g.objects(tp, TP_BODY_R) if is_literal_nonempty(x)), "")
        conf = next((str(x).strip() for x in g.objects(tp, TP_CONF_R) if is_literal_nonempty(x)), "")
        return src, body, conf

    node = URIRef(op_iri)
    groups: Dict[Tuple[str,str], Dict] = {}
    def add_group(tp: URIRef):
        src, body, conf = tp_fields(tp)
        if not body: return
        key = (norm(src), norm(body))
        G = groups.setdefault(key, {"src": src or "（未知书目）", "body": body, "conf": ""})
        if conf:
            try:
                cur = float(G["conf"]) if G["conf"] else -1e9
                if float(conf) > cur: G["conf"] = conf
            except: pass
    for pa in g.subjects(P_ABOUT_R, node):
        for tp in g.objects(pa, P_DERIVED_R):
            if isinstance(tp, URIRef): add_group(tp)
    for tp in g.subjects(P_CONTAINS_R, node):
        if isinstance(tp, URIRef): add_group(tp)
    return sorted(groups.values(), key=lambda d: (0 if d["src"] else 1, d["src"], d["body"]))

# —— 主入口 —— #
def run(default_path: Optional[str] = None):
    """在主入口（app.py）中调用：opm.run()"""
    data_default = default_path or DEFAULT_DATA

    with st.sidebar:
        st.header("数据")
        data_file = st.text_input("NT/TTL/RDF/OWL 路径", value=data_default)
        if st.button("重建索引", type="primary", use_container_width=True):
            st.cache_data.clear(); st.cache_resource.clear(); st.rerun()

        st.markdown("---")
        st.header("阈值")
        enable_all_filters = st.checkbox("启用（人物·地域·时间）", value=False)

        only_dianshi = False
        sel_school, sel_acad, sel_huji = [], [], []
        selected_region_node, region_title = None, "全域"
        era_left = era_right = "不限"

        if enable_all_filters:
            st.caption("已开启阈值。可在下列分组中设置具体条件。")
            # — 人物 —
            with st.expander("人物条件", expanded=True):
                # 其余人物条件在下方逻辑中使用
                only_dianshi = st.checkbox("仅统计通过殿试者", value=False)
                # 其余三项需要 options，等索引载入后再 render（占位）
                st.session_state["__need_populate_opts__"] = True

    # —— 索引与图 —— #
    state = build_indices(data_file)
    g: Graph = state["g"]
    pos_attrs: Dict[str, Dict[str, str]] = state["pos_attrs"]
    pos_persons_all: Dict[str, Set[str]] = {k: set(v) for k, v in state["pos_persons_all"].items()}
    single_dims = state["single_dims"]

    pred_by_local: Dict[str, Set[URIRef]] = {k: set(v) for k, v in state["pred_by_local"].items()}
    name_predicates: Set[URIRef] = set(URIRef(p) for p in state["name_predicates"])

    # 人名缓存（依赖上面的 pred_by_local/name_predicates/g）
    @st.cache_data(show_spinner=False)
    def person_name_cache(person_iri: str) -> str:
        s = URIRef(person_iri)
        for pred in pred_by_local.get("姓名", set()):
            for lit in g.objects(s, pred):
                if is_literal_nonempty(lit): return str(lit).strip()
        for pred in name_predicates:
            for lit in g.objects(s, pred):
                if is_literal_nonempty(lit): return str(lit).strip()
        return localname(s)

    # —— 补渲染人物阈值选项（需索引） —— #
    with st.sidebar:
        if enable_all_filters:
            with st.expander("人物条件", expanded=True):
                sel_school = st.multiselect("学籍类型", options=state["opts_school_type"], default=[])
                sel_acad   = st.multiselect("学术专长", options=state["opts_academic"], default=[])
                sel_huji   = st.multiselect("户籍类型", options=state["opts_huji_type"], default=[])
            # — 地域 —
            with st.expander("地域条件（出生地）", expanded=False):
                level_to_places = state["level_to_places"]
                LEVEL_ORDER = ["省","特别行政区","自治区","市","州","盟","地区","县","区","旗","林区","特区","新区","乡","镇","街道","社区","村"]
                existing_lvls = [lvl for lvl in LEVEL_ORDER if lvl in level_to_places] or list(level_to_places.keys())
                sel_lvl = st.selectbox("区划层级", existing_lvls, index=0 if existing_lvls else 0)
                options = level_to_places.get(sel_lvl, [])
                mod2uri = {mod: uri for mod, uri in options}
                sel_mod = st.selectbox("现代名称", list(mod2uri.keys()) if options else ["（无可选项）"])
                selected_region_node = mod2uri.get(sel_mod)
                region_title = sel_mod or "（未选）"
            # — 时间 —
            with st.expander("时间条件（年号范围）", expanded=False):
                era_labels = ["不限"] + [e for e, a, b in MING_ERA_MAP]
                era_left  = st.selectbox("左边界", era_labels, index=0)
                era_right = st.selectbox("右边界", era_labels, index=0)

    # =========================
    #       阈值计算
    # =========================
    person_academic    = state["person_academic"]
    person_huji_type   = state["person_huji_type"]
    person_school_type = state["person_school_type"]
    persons_with_dianshi: Set[str] = set(state["persons_with_dianshi"])

    # 地域闭包
    parent2children = {k: set(v) for k, v in state["parent2children"].items()}
    def closure_descendants(root_id: str) -> Set[str]:
        if not root_id: return set()
        out = {root_id}; q = deque([root_id]); seen = {root_id}
        while q:
            u = q.popleft()
            for v in parent2children.get(u, set()):
                if v not in seen:
                    seen.add(v); out.add(v); q.append(v)
        return out

    if enable_all_filters and selected_region_node:
        region_nodes = closure_descendants(selected_region_node)
    else:
        region_nodes = set(state["place_mod"].keys())  # 全域
    person_birth_place = state["person_birth_place"]

    # 时间范围
    def era_range(era: str) -> Optional[Tuple[int,int]]:
        if era == "不限": return None
        for e, a, b in MING_ERA_MAP:
            if e == era: return (a, b)
        return None

    left_range  = era_range(era_left)  if enable_all_filters else None
    right_range = era_range(era_right) if enable_all_filters else None
    if left_range and right_range:
        year_min, year_max = min(left_range[0], right_range[0]), max(left_range[1], right_range[1])
    elif left_range:
        year_min, year_max = left_range
    elif right_range:
        year_min, year_max = right_range
    else:
        year_min = year_max = None

    p_to_pes = {k: set(v) for k, v in state["p_to_pes"].items()}
    pe_to_ex = state["pe_to_ex"]
    ex_to_level = state["ex_to_level"]
    exam2year = state["exam2year"]

    def person_any_exam_in_range(pid: str) -> bool:
        if not enable_all_filters or (year_min is None and year_max is None):
            return True
        for pe in p_to_pes.get(pid, set()):
            ex = pe_to_ex.get(pe)
            if not ex: continue
            if norm(t2s(ex_to_level.get(ex, ""))) != norm("殿试"):
                continue
            y = exam2year.get(ex)
            if isinstance(y, int):
                if (year_min is None or y >= year_min) and (year_max is None or y <= year_max):
                    return True
        return False

    def person_in_region(pid: str) -> bool:
        if not enable_all_filters:
            return True
        bpl = person_birth_place.get(pid)
        return bool(bpl and bpl in region_nodes)

    def person_pass_filters(pid: str) -> bool:
        if enable_all_filters:
            if only_dianshi and (pid not in persons_with_dianshi):
                return False
            if sel_school:
                vals = [norm(t2s(v)) for v in person_school_type.get(pid, [])]
                if not any(norm(t2s(x)) in vals for x in sel_school): return False
            if sel_acad:
                vals = [norm(t2s(v)) for v in person_academic.get(pid, [])]
                if not any(norm(t2s(x)) in vals for x in sel_acad): return False
            if sel_huji:
                vals = [norm(t2s(v)) for v in person_huji_type.get(pid, [])]
                if not any(norm(t2s(x)) in vals for x in sel_huji): return False
        if not person_in_region(pid): return False
        if not person_any_exam_in_range(pid): return False
        return True

    # 对每官职的担任人集合应用过滤
    pos_persons_filtered: Dict[str, Set[str]] = {
        op: {pid for pid in ppl if person_pass_filters(pid)} for op, ppl in pos_persons_all.items()
    }

    # =========================
    #        可视化
    # =========================
    st.title("职官项计量")
    scope_desc = "地域：全域" if not (enable_all_filters and selected_region_node) else f"地域：{region_title}"
    if not (enable_all_filters and (year_min is not None or year_max is not None)):
        time_desc = "时间：不限"
    else:
        time_desc = f"时间：{year_min or '…'}–{year_max or '…'}（按殿试年份）"
    person_desc = "人物：全体" if not enable_all_filters else "人物：已设置条件"
    st.caption(f"口径：每类别统计担任过该类别下任一官职的去重人物数。当前：{person_desc} ｜ {scope_desc} ｜ {time_desc}。")

    st.session_state.setdefault("prov_op", None)
    def set_prov(op_iri: str, title: str):
        st.session_state["prov_op"] = (op_iri, title)

    tab1, tab2 = st.tabs(["单维统计", "多维统计"])

    # —— 单维 —— #
    with tab1:
        colA, colC = st.columns([3,2])
        with colA:
            dim_display = st.selectbox(
                "选择维度",
                options=[name for name, _ in single_dims],
                index=0,
                help="对齐码自动转中文标签；含“官阶”。"
            )
        with colC:
            topn = st.slider("Top N", min_value=5, max_value=100, value=30, step=5, key="topn_single")

        dim_key = next(k for name,k in single_dims if name == dim_display)

        bucket_persons: Dict[str, Set[str]] = defaultdict(set)
        for op_iri, persons in pos_persons_filtered.items():
            label = pos_label(pos_attrs, op_iri, dim_key)
            if label: bucket_persons[label].update(persons)

        df_dim = pd.DataFrame([(b, len(ps)) for b, ps in bucket_persons.items() if b],
                              columns=["类别", "人数"]).sort_values("人数", ascending=False)

        left, right = st.columns([7,5], gap="large")
        with left:
            plot_bar(df_dim, topn=topn, title=f"{dim_display}：Top {topn}")
            st.markdown("#### 类别明细")
            if df_dim.empty:
                st.info("无数据（阈值可能过严或该范围为空）。")
            else:
                for label, cnt in df_dim.values.tolist():
                    with st.expander(f"{label}  —  人数：{cnt}", expanded=False):
                        op_rows = []
                        for op_iri, persons in pos_persons_filtered.items():
                            if pos_label(pos_attrs, op_iri, dim_key) == label:
                                title2 = (pos_attrs.get(op_iri, {}).get("原始称谓")
                                          or pos_attrs.get(op_iri, {}).get("官职名称")
                                          or op_iri.rsplit("#",1)[-1])
                                op_rows.append((title2, len(persons), op_iri))
                        op_rows.sort(key=lambda x: x[1], reverse=True)
                        cols = st.columns(3)
                        for i, (title2, pcnt, opiri) in enumerate(op_rows):
                            with cols[i % 3]:
                                st.button(
                                    f"{title2}（{pcnt}）",
                                    key=f"prov_btn_{hash((label,opiri))}",
                                    on_click=set_prov,
                                    args=(opiri, title2),
                                    use_container_width=True
                                )

        with right:
            st.markdown("### 溯源")
            sel = st.session_state.get("prov_op")
            if not sel:
                st.info("在左侧列表中选择官职以查看溯源。")
            else:
                op_iri, title = sel
                st.write(f"**官职：** {title}")
                incumbents = []
                for piri in sorted(pos_persons_filtered.get(op_iri, set())):
                    nm = person_name_cache(piri)
                    if nm: incumbents.append(nm)
                incumbents = sorted(list(dict.fromkeys(incumbents)))
                if incumbents:
                    st.write("**担任人（已应用阈值）：** " + "、".join(incumbents))
                for i, G in enumerate(provenance_for_official_position(g, op_iri), 1):
                    with st.expander(f"#{i} 书目：{G['src']}", expanded=(i==1)):
                        if G.get("conf"): st.write(f"可信度：{G['conf']}")
                        st.markdown(
                            "<div style='padding:.6rem .75rem;border-left:4px solid #dbeafe;background:#f8fafc;border-radius:8px;line-height:1.6'>"
                            + G["body"] +
                            "</div>",
                            unsafe_allow_html=True
                        )

    # —— 多维 —— #
    with tab2:
        colA, colB, colC = st.columns([3,3,2])
        with colA:
            whichN_label = st.selectbox("维度数量", options=["二维（2D）", "三维（3D）", "四维（4D）"], index=0)
        with colB:
            subtype = None
            if whichN_label.startswith("二维"):
                subtype = st.selectbox("二维组合", options=["机构×核心", "地点×核心"], index=0)
            elif whichN_label.startswith("三维"):
                subtype = st.selectbox("三维组合", options=["机构×核心×层级", "地点×机构×核心"], index=0)
        with colC:
            topn2 = st.slider("Top N", min_value=5, max_value=100, value=30, step=5, key="topn2")

        def get_label_for(op: str) -> str:
            if whichN_label.startswith("二维"):
                return dim2_label(pos_attrs, op, subtype or "机构×核心")
            elif whichN_label.startswith("三维"):
                return dim3_label(pos_attrs, op, subtype or "机构×核心×层级")
            else:
                return dim4_label(pos_attrs, op)

        bucket_personsN: Dict[str, Set[str]] = defaultdict(set)
        for op_iri, persons in pos_persons_filtered.items():
            lab = get_label_for(op_iri)
            if lab: bucket_personsN[lab].update(persons)
        dfN = pd.DataFrame([(b, len(ps)) for b, ps in bucket_personsN.items() if b],
                           columns=["类别", "人数"]).sort_values("人数", ascending=False)

        left2, right2 = st.columns([7,5], gap="large")
        with left2:
            title = whichN_label + (f" · {subtype}" if subtype else "")
            plot_bar(dfN, topn=topn2, title=f"{title}：Top {topn2}")
            st.markdown("#### 类别明细")
            if dfN.empty:
                st.info("无数据。")
            else:
                for label, cnt in dfN.values.tolist():
                    with st.expander(f"{label}  —  人数：{cnt}", expanded=False):
                        op_rows = []
                        for op_iri, persons in pos_persons_filtered.items():
                            if get_label_for(op_iri) == label:
                                title2 = (pos_attrs.get(op_iri, {}).get("原始称谓")
                                          or pos_attrs.get(op_iri, {}).get("官职名称")
                                          or op_iri.rsplit("#",1)[-1])
                                op_rows.append((title2, len(persons), op_iri))
                        op_rows.sort(key=lambda x: x[1], reverse=True)
                        cols = st.columns(3)
                        for i, (title2, pcnt, opiri) in enumerate(op_rows):
                            with cols[i % 3]:
                                st.button(
                                    f"{title2}（{pcnt}）",
                                    key=f"prov_btnN_{hash((label,opiri))}",
                                    on_click=set_prov,
                                    args=(opiri, title2),
                                    use_container_width=True
                                )

        with right2:
            st.markdown("### 溯源")
            sel = st.session_state.get("prov_op")
            if not sel:
                st.info("在左侧选择官职以查看溯源。")
            else:
                op_iri, title = sel
                st.write(f"**官职：** {title}")
                incumbents = []
                for piri in sorted(pos_persons_filtered.get(op_iri, set())):
                    nm = person_name_cache(piri)
                    if nm: incumbents.append(nm)
                incumbents = sorted(list(dict.fromkeys(incumbents)))
                if incumbents:
                    st.write("**担任人（已应用阈值）：** " + "、".join(incumbents))
                for i, G in enumerate(provenance_for_official_position(g, op_iri), 1):
                    with st.expander(f"#{i} 书目：{G['src']}", expanded=(i==1)):
                        if G.get("conf"): st.write(f"可信度：{G['conf']}")
                        st.markdown(
                            "<div style='padding:.6rem .75rem;border-left:4px solid #dbeafe;background:#f8fafc;border-radius:8px;line-height:1.6'>"
                            + G["body"] +
                            "</div>",
                            unsafe_allow_html=True
                        )
