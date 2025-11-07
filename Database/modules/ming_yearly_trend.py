# modules/ming_yearly_trend.py
# -*- coding: utf-8 -*-
# 📈 明代逐年趋势（官职维度 + 人物属性维度）
# 分母：当年“殿试人物”（participatesIn→hasExam 为 ImperialExam 且等级含“殿试”）
# 分子：分母中“命中当前计量对象（桶）”的人物（去重）
# ✅ 人名点击可溯源（人名高亮；基于 TextPassage: contains / about→derivedFrom）
# 🔒 强制“仅殿试时间触发计数”
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional, Union
import re, unicodedata, datetime

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from rdflib import Graph, URIRef, Literal, RDF
from rdflib.namespace import RDFS, SKOS, FOAF

# ========== 样式与常量（纯定义；无 UI 调用） ==========
PLOT_FONT = dict(family="Inter, Segoe UI, SimHei, Microsoft YaHei, Noto Sans CJK SC, Arial", size=14)
THEME = dict(
    bg="#ffffff",
    grid="rgba(0,0,0,0.06)",
    text="#1f2937",
    subtext="#4b5563",
    bar="#5B8FF9",
    bar_edge="rgba(0,0,0,0.15)",
    line="#5AD8A6",
    line_pts_edge="#ffffff",
    ratio="#9270CA",
    ratio_fill_base=(146,112,202),
    eras_rgb=[
        (91,143,249), (90,216,166), (246,189,22),
        (232,104,162), (78,201,176), (170,120,200),
        (255,157,77), (124,201,240), (140,140,140),
        (116,148,217), (130,202,157), (249,123,114)
    ]
)
DEFAULT_DATA = r"C:\Users\卢航青\Desktop\本体结构\ontology_appt_merged.nt"

MING_ERAS: List[Tuple[str, int, int]] = [
    ("洪武", 1368, 1398), ("建文", 1399, 1402), ("永乐", 1403, 1424), ("洪熙", 1425, 1425),
    ("宣德", 1426, 1435), ("正统", 1436, 1449), ("景泰", 1450, 1456), ("天顺", 1457, 1464),
    ("成化", 1465, 1487), ("弘治", 1488, 1505), ("正德", 1506, 1521), ("嘉靖", 1522, 1566),
    ("隆庆", 1567, 1572), ("万历", 1573, 1620), ("泰昌", 1620, 1620), ("天启", 1621, 1627),
    ("崇祯", 1628, 1644),
]
def era_of_year(y: int) -> Optional[str]:
    for n, a, b in MING_ERAS:
        if a <= y <= b: return n
    return None

# ========== 工具 ==========
def localname(u: Union[URIRef, str]) -> str:
    s = str(u)
    if "#" in s: s = s.rsplit("#", 1)[-1]
    elif "/" in s: s = s.rsplit("/", 1)[-1]
    return s

def is_lit(x) -> bool:
    return isinstance(x, Literal) and str(x).strip() != ""

_ZW = {u"\u200b", u"\u200c", u"\u200d", u"\ufeff"}
def nfkc_strip(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if (not ch.isspace()) and (ch not in _ZW))
    return s.strip()

# OpenCC：全局繁↔简
from opencc import OpenCC
CC_T2S = OpenCC("t2s")
CC_S2T = OpenCC("s2t")
def to_s(s: str) -> str: return CC_T2S.convert(s or "")
def to_t(s: str) -> str: return CC_S2T.convert(s or "")
def norm(s: str) -> str: return nfkc_strip(to_s(s or ""))

# ========== RDF 常量（localname） ==========
OP_PARTICIPATES_IN = "participatesIn"
OP_HAS_EXAM        = "hasExam"
DP_EXAM_LEVEL      = "考试等级"
CLASS_IMPERIAL_EXAM= "ImperialExam"

P_APPOINTED_IN     = "appointedIn"
P_HAS_POSITION     = "hasPosition"
CLASS_OFFICIAL     = {"OfficialPosition"}

POS_DIM_KEYS = [("职系","职系"), ("机构","机构"), ("核心职称","核心职称")]
PERSON_DIM_KEYS = [("学籍类型","学籍类型"), ("户籍类型","户籍类型"), ("学术专长","学术专长")]

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

PERSON_NAME_KEYS = {
    "姓名","name","label","rdfs_label","标题","title",
    str(FOAF.name), str(RDFS.label), str(SKOS.prefLabel)
}
PA_SCHOOL_TYPE_KEYS = {"学籍类型","學籍類型","学籍_类型","學籍_類型","类型","類型"}

# ========== 缓存：加载与索引（纯函数；无 UI） ==========
@st.cache_resource(show_spinner=False)
def load_graph(path: str) -> Graph:
    suf = Path(path).suffix.lower()
    fmt = {".nt":"nt",".ttl":"turtle",".rdf":"xml",".owl":"xml",".xml":"xml"}.get(suf, "turtle")
    g = Graph(); g.parse(path, format=fmt); return g

@st.cache_resource(show_spinner=True)
def build_indices(path: str):
    g = load_graph(path)

    # 谓词 localname 索引
    pred_by_local: Dict[str, Set[URIRef]] = defaultdict(set)
    for _, p, _ in g.triples((None, None, None)):
        pred_by_local[localname(p)].add(p)

    # 人名谓词
    name_predicates: Set[URIRef] = set()
    for key in PERSON_NAME_KEYS:
        lname = localname(key) if isinstance(key, str) else localname(URIRef(key))
        name_predicates |= pred_by_local.get(lname, set())

    # 官职实例与字段
    def is_official(node: URIRef) -> bool:
        for _,_,t in g.triples((node, RDF.type, None)):
            if localname(t) in CLASS_OFFICIAL: return True
        return False

    pos_nodes: List[URIRef] = []
    for s,_,t in g.triples((None, RDF.type, None)):
        if is_official(s): pos_nodes.append(s)

    def first_dp(node: URIRef, local: str) -> str:
        for pred in pred_by_local.get(local, set()):
            for lit in g.objects(node, pred):
                if is_lit(lit): return str(lit).strip()
        return ""

    pos_attrs: Dict[str, Dict[str, str]] = {}
    for op in pos_nodes:
        row = {
            "职系": first_dp(op, "职系"),
            "机构": first_dp(op, "机构"),
            "核心职称": first_dp(op, "核心职称"),
            "原始称谓": first_dp(op, "原始称谓") or first_dp(op, "官职名称"),
        }
        pos_attrs[str(op)] = row

    # 人 → 任职（官职去重集合）
    preds_app_in  = pred_by_local.get(P_APPOINTED_IN, set())
    preds_has_pos = pred_by_local.get(P_HAS_POSITION, set())

    person2events: Dict[str, Set[str]] = defaultdict(set)
    for s, p, e in g.triples((None, None, None)):
        if p in preds_app_in and isinstance(e, URIRef):
            person2events[str(s)].add(str(e))

    event2pos: Dict[str, Set[str]] = defaultdict(set)
    for e, p, op in g.triples((None, None, None)):
        if p in preds_has_pos and isinstance(op, URIRef):
            event2pos[str(e)].add(str(op))

    person2positions: Dict[str, Set[str]] = defaultdict(set)
    for pid, evts in person2events.items():
        for e in evts:
            for op in event2pos.get(e, set()):
                person2positions[pid].add(op)

    # 科举：参与 → 考试 → 年份/等级（殿试识别）
    YEAR_RE = re.compile(r"(1[0-9]{3}|20[0-9]{2})")
    preds_part = pred_by_local.get(OP_PARTICIPATES_IN, set())
    preds_has_exam = pred_by_local.get(OP_HAS_EXAM, set())
    preds_exam_level = pred_by_local.get(DP_EXAM_LEVEL, set())

    p_to_pes: Dict[str, Set[str]] = defaultdict(set)
    pe_to_ex: Dict[str, str] = {}
    ex_to_level: Dict[str, str] = {}

    for pred in preds_part:
        for s,_,pe in g.triples((None, pred, None)):
            if isinstance(s, URIRef) and isinstance(pe, URIRef):
                p_to_pes[str(s)].add(str(pe))
    for pred in preds_has_exam:
        for pe,_,ex in g.triples((None, pred, None)):
            if isinstance(pe, URIRef) and isinstance(ex, URIRef):
                pe_to_ex[str(pe)] = str(ex)
    for pred in preds_exam_level:
        for ex,_,lit in g.triples((None, pred, None)):
            if isinstance(ex, URIRef) and is_lit(lit):
                ex_to_level[str(ex)] = str(lit).strip()

    def parse_exam_year(ex: URIRef) -> Optional[int]:
        for _,_,lit in g.triples((ex, None, None)):
            if is_lit(lit):
                for m in YEAR_RE.findall(str(lit)):
                    try:
                        y = int(m)
                        if 1000 <= y <= 2100:
                            return y
                    except: pass
        for m in YEAR_RE.findall(localname(ex)):
            try:
                y = int(m)
                if 1000 <= y <= 2100:
                    return y
            except: pass
        return None

    PALACE_TOKENS = ["殿试","殿","御试","廷对","Palace"]
    def is_palace_exam(ex_id: str) -> bool:
        lv = ex_to_level.get(ex_id, "")
        if any(tok in (lv or "") for tok in PALACE_TOKENS): return True
        ln = localname(ex_id)
        return any(tok in ln for tok in PALACE_TOKENS)

    year2people_exam: Dict[int, Set[str]] = defaultdict(set)
    person2palace_exams: Dict[str, Set[str]] = defaultdict(set)
    for pid, pes in p_to_pes.items():
        for pe in pes:
            ex = pe_to_ex.get(pe)
            if not ex: continue
            if not is_palace_exam(ex):
                continue
            y = parse_exam_year(URIRef(ex))
            if isinstance(y, int) and 1368 <= y <= 1644:
                year2people_exam[y].add(pid)
            person2palace_exams[pid].add(ex)

    # 人物属性维度
    def all_literals_by_local(node: URIRef, local: str) -> List[str]:
        vals, seen = [], set()
        for pred in pred_by_local.get(local, set()):
            for lit in g.objects(node, pred):
                if is_lit(lit):
                    s = str(lit).strip()
                    if s and s not in seen:
                        seen.add(s); vals.append(s)
        return vals

    def one_hop_neighbors(person: URIRef) -> Set[URIRef]:
        out: Set[URIRef] = set()
        for p, o in g.predicate_objects(person):
            if isinstance(o, URIRef): out.add(o)
        for s, p in g.subject_predicates(person):
            if isinstance(s, URIRef): out.add(s)
        return out

    def pa_school_types_of_person(person: URIRef) -> List[str]:
        out, seen = [], set()
        def collect_from_node(node: URIRef):
            for pa in g.subjects(P_ABOUT, node):
                props_txt = [str(l).strip() for l in g.objects(pa, P_PROP) if is_lit(l)]
                if not props_txt: continue
                ok = any((s in PA_SCHOOL_TYPE_KEYS) or (("学籍" in s or "學籍" in s) and ("类型" in s or "類型" in s)) for s in props_txt)
                if not ok: continue
                val = next((str(l).strip() for l in g.objects(pa, P_VALN) if is_lit(l)), "") \
                      or next((str(l).strip() for l in g.objects(pa, P_VAL) if is_lit(l)), "")
                if not val: continue
                k = nfkc_strip(val)
                if k and k not in seen:
                    seen.add(k); out.append(val)
        collect_from_node(person)
        for nb in one_hop_neighbors(person):
            collect_from_node(nb)
        return out

    persons_all: Set[str] = set(p_to_pes.keys())
    person_academic: Dict[str, List[str]]    = {}
    person_huji_type: Dict[str, List[str]]   = {}
    person_school_type: Dict[str, List[str]] = {}
    for pid in persons_all:
        node = URIRef(pid)
        person_academic[pid]    = all_literals_by_local(node, "学术专长")
        person_huji_type[pid]   = all_literals_by_local(node, "户籍类型")
        person_school_type[pid] = pa_school_types_of_person(node)

    # 维度可选值
    dim_options: Dict[str, List[str]] = {}
    for _, key in POS_DIM_KEYS:
        vals: Set[str] = set()
        for op, row in pos_attrs.items():
            v = (row.get(key) or "").strip()
            if v: vals.add(v)
        dim_options[key] = sorted(vals)

    def collect_values(mapping: Dict[str, List[str]]) -> List[str]:
        pool: Set[str] = set()
        for arr in mapping.values():
            for v in arr:
                if v: pool.add(v.strip())
        return sorted(pool)
    dim_options["学术专长"] = collect_values(person_academic)
    dim_options["户籍类型"] = collect_values(person_huji_type)
    dim_options["学籍类型"] = collect_values(person_school_type)

    return dict(
        g=g,
        pred_by_local=pred_by_local,
        name_predicates=name_predicates,
        pos_attrs=pos_attrs,
        person2positions=person2positions,
        year2people_exam=year2people_exam,
        person2palace_exams=person2palace_exams,
        person_academic=person_academic,
        person_huji_type=person_huji_type,
        person_school_type=person_school_type,
        dim_options=dim_options,
    )

# 解决 UnhashableParamError：把 Graph 作为下划线参数 + graph_nonce
@st.cache_data(show_spinner=False)
def _person_name_cache(person_iri: str,
                       pred_by_local: Dict[str, Set[URIRef]],
                       name_predicates: Set[URIRef],
                       _g: Graph,
                       graph_nonce: int = 0) -> str:
    node = URIRef(person_iri)
    for p in pred_by_local.get("姓名", set()):
        for lit in _g.objects(node, p):
            if is_lit(lit): return str(lit).strip()
    for p in name_predicates:
        for lit in _g.objects(node, p):
            if is_lit(lit): return str(lit).strip()
    return localname(node)

# ========== 页面入口 ==========
def run(configure_page: bool = True, default_data_path: Optional[str] = None):
    # 1) 页面配置（可由主入口关闭）
    if configure_page and not st.session_state.get("_page_configured", False):
        st.set_page_config(page_title="时空计量", layout="wide")
        st.session_state["_page_configured"] = True

    # 2) 侧栏参数
    with st.sidebar:
        st.header("数据 / 运行")
        DATA_PATH = st.text_input("RDF 路径", value=default_data_path or DEFAULT_DATA)
        if st.button("清缓存并重建", type="primary", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

        st.markdown("---")
        contradiction_mode = st.selectbox(
            "矛盾处理", ["排除矛盾","包括矛盾"], index=0,
            help="矛盾：同一人连到≥2个殿试实例（participatesIn→hasExam→ImperialExam）。"
        )
        st.caption("提示：选择“排除矛盾”时，矛盾人物在所有统计维度中统一剔除。")

        era_alpha = st.slider(
            "年号底色强度（α）", min_value=0.08, max_value=0.30, value=0.16, step=0.01,
            help="轻度增强区分度；建议不超过 0.25。"
        )

    # 3) 构建索引
    S = build_indices(DATA_PATH)
    g = S["g"]
    pred_by_local = S["pred_by_local"]
    name_predicates = S["name_predicates"]
    pos_attrs = S["pos_attrs"]
    person2positions = S["person2positions"]
    year2people_exam = S["year2people_exam"]
    person2palace_exams = S["person2palace_exams"]
    person_academic = S["person_academic"]
    person_huji_type = S["person_huji_type"]
    person_school_type = S["person_school_type"]
    dim_options = S["dim_options"]

    # 4) 轻薄封装的人名缓存（避免读取模块级全局）
    def person_name_cache(person_iri: str) -> str:
        return _person_name_cache(person_iri, pred_by_local, name_predicates, g, id(g))

    # 5) 溯源（用闭包里的 g）
    def tp_fields(tp: URIRef) -> Tuple[str, str, str]:
        src  = next((str(x).strip() for x in g.objects(tp, TP_SRC)  if is_lit(x)), "")
        body = next((str(x).strip() for x in g.objects(tp, TP_BODY) if is_lit(x)), "")
        conf = next((str(x).strip() for x in g.objects(tp, TP_CONF) if is_lit(x)), "")
        return src, body, conf

    def highlight_name(body_html: str, name: str) -> str:
        if not name: return body_html
        nm_s = to_s(name); nm_t = to_t(name)
        for key in sorted({nm_s, nm_t}, key=len, reverse=True):
            if key:
                body_html = body_html.replace(key, f"<span style='background:#fde68a'>{key}</span>")
        return body_html

    def provenance_for_person(pid: str, person_name: str) -> List[Dict]:
        node = URIRef(pid)
        related: Set[URIRef] = {node}
        for p, o in g.predicate_objects(node):
            if isinstance(o, URIRef): related.add(o)
        for s, p in g.subject_predicates(node):
            if isinstance(s, URIRef): related.add(s)

        key = norm(person_name)
        groups: Dict[Tuple[str,str], Dict] = {}

        def add_group(tp: URIRef):
            src, body, conf = tp_fields(tp)
            if not body: return
            if key and (key not in norm(body)):
                return
            k = (nfkc_strip(to_s(src)), nfkc_strip(to_s(body)))
            G = groups.setdefault(k, {"src": src or "（未知书目）", "body": body, "conf": ""})
            if conf:
                try:
                    cur = float(G["conf"]) if G["conf"] else -1e18
                    if float(conf) > cur: G["conf"] = conf
                except: pass

        for rel in related:
            for tp in g.subjects(P_CONTAINS, rel):
                if isinstance(tp, URIRef): add_group(tp)
        for rel in related:
            for pa in g.subjects(P_ABOUT, rel):
                for tp in g.objects(pa, P_DERIVED_FROM):
                    if isinstance(tp, URIRef): add_group(tp)

        out = sorted(groups.values(), key=lambda d: (0 if d["src"] else 1, d["src"], d["body"]))
        for it in out:
            it["body"] = highlight_name(it["body"], person_name)
        return out

    # 6) 顶部 UI：维度与目标桶
    st.title("时空计量")
    st.caption("分母=当年殿试人物（仅殿试触发）；分子=分母中命中当前计量对象（桶）的人物（去重）。")

    colA, colB, colC = st.columns([3,3,6])
    with colA:
        dim_source = st.selectbox("维度来源", ["官职维度", "人物属性维度"], index=0)
    with colB:
        if dim_source == "官职维度":
            dim_name = st.selectbox("选择", options=[x[0] for x in POS_DIM_KEYS], index=0)
            dim_key = dict(POS_DIM_KEYS)[dim_name]
        else:
            dim_name = st.selectbox("选择", options=[x[0] for x in PERSON_DIM_KEYS], index=0)
            dim_key = dict(PERSON_DIM_KEYS)[dim_name]
    with colC:
        cand = dim_options.get(dim_key, [])
        target_bucket = st.selectbox("属性", ["（请选择）"] + cand, index=0)

    st.divider()
    chart_mode = st.radio(
        "主图显示",
        ["比例", "人数"],
        index=0, horizontal=True,
        help="比例：命中比例；人数：分母为柱，分子为折线。"
    )

    if not cand:
        st.warning(f"在数据中未发现“{dim_key}”的有效取值，请检查本体字段或数据填充。")
        return
    if target_bucket == "（请选择）":
        st.info("请选择一个计量对象（桶）。")
        return

    target_norm = nfkc_strip(target_bucket)

    # 7) 矛盾人物集合 & 开关
    conflict_people: Set[str] = {pid for pid, exs in person2palace_exams.items() if len(exs) >= 2}
    EXCLUDE_CONFLICT = (contradiction_mode == "排除矛盾")

    # 8) 命中判定
    def person_hits_bucket(pid: str) -> bool:
        if dim_source == "官职维度":
            for op in person2positions.get(pid, set()):
                row = pos_attrs.get(op, {})
                lab = (row.get(dim_key) or "").strip()
                if lab and nfkc_strip(lab) == target_norm:
                    return True
            return False
        else:
            if dim_key == "学术专长":
                arr = person_academic.get(pid, [])
            elif dim_key == "户籍类型":
                arr = person_huji_type.get(pid, [])
            else:
                arr = person_school_type.get(pid, [])
            return any(nfkc_strip(v) == target_norm for v in arr)

    # 9) 逐年统计
    rows = []
    years = sorted([y for y in year2people_exam.keys() if 1368 <= y <= 1644])
    for y in years:
        denom_people = set(year2people_exam[y])
        if EXCLUDE_CONFLICT:
            denom_people -= conflict_people
        den = len(denom_people)
        num = sum(1 for pid in denom_people if person_hits_bucket(pid))
        ratio = (num / den) if den > 0 else None
        rows.append(dict(年份=y, 有效样本数=den, 目标命中数=num, 比例=ratio))
    df = pd.DataFrame(rows)
    if df.empty:
        st.info("没有可统计的数据。")
        return

    # 10) 年号底色 & 标注
    def rgba_str(rgb, a):
        r,g,b = rgb
        return f"rgba({r},{g},{b},{a})"

    def era_shapes_and_labels(x_years: List[int], alpha: float):
        shapes, annotations = [], []
        if not x_years:
            return shapes, annotations
        x_min, x_max = min(x_years), max(x_years)
        cidx = 0
        for era, a, b in MING_ERAS:
            L, R = max(a, x_min), min(b, x_max)
            if L <= R:
                color = rgba_str(THEME["eras_rgb"][cidx % len(THEME["eras_rgb"])], alpha)
                cidx += 1
                shapes.append(dict(type="rect", xref="x", yref="paper",
                                   x0=L-0.5, x1=R+0.5, y0=0, y1=1,
                                   fillcolor=color, line=dict(width=0), layer="below"))
                annotations.append(dict(
                    x=(L+R)/2, y=1.06, xref="x", yref="paper", text=era,
                    showarrow=False, font=dict(family=PLOT_FONT["family"], size=13, color=THEME["subtext"])
                ))
        return shapes, annotations

    years_all = df["年份"].tolist()
    den_arr   = df["有效样本数"].tolist()
    num_arr   = df["目标命中数"].tolist()
    ratio_arr = [ (num/den) if den>0 else None for den, num in zip(den_arr, num_arr) ]

    def base_layout(fig, x_years, yaxis_title):
        shapes, annotations = era_shapes_and_labels(x_years, era_alpha)
        fig.update_layout(
            template="simple_white",
            height=560, margin=dict(l=24, r=20, t=92, b=70), font=PLOT_FONT,
            paper_bgcolor=THEME["bg"], plot_bgcolor=THEME["bg"],
            xaxis=dict(title=None, tickmode="linear", dtick=1, showgrid=False, tickfont=dict(color=THEME["text"])),
            yaxis=dict(title=yaxis_title, showgrid=True, gridcolor=THEME["grid"], zeroline=False, tickfont=dict(color=THEME["text"])),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="rgba(255,255,255,0.98)", bordercolor="rgba(0,0,0,0.15)",
                            font=dict(size=13, family=PLOT_FONT["family"], color=THEME["text"])),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                        bgcolor="rgba(255,255,255,0.6)", bordercolor="rgba(0,0,0,0.08)", borderwidth=1),
            shapes=shapes, annotations=annotations
        )

    def export_png(fig, filename_hint: str):
        """把图导出为 PNG 字节流（需要 kaleido）。"""
        try:
            import plotly.io as pio
            png_bytes = pio.to_image(fig, format="png", scale=2)  # 高清导出
            return png_bytes, None
        except Exception as e:
            return None, str(e)

    # 11) 主图
    if st.session_state.get("_dummy_spacing_once") is None:
        st.session_state["_dummy_spacing_once"] = True
    st.subheader("")

    if chart_mode.startswith("比例"):
        x = [int(y) for y, r in zip(years_all, ratio_arr) if r is not None]
        y = [float(r) for r in ratio_arr if r is not None]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines+markers",
            name=f"{dim_name}＝{target_bucket}",
            line=dict(width=3.2, color=THEME["ratio"], shape="spline", smoothing=0.45),
            marker=dict(size=7.5, line=dict(width=1, color=THEME["line_pts_edge"]), symbol="circle"),
            fill="tozeroy", fillcolor=rgba_str(THEME["ratio_fill_base"], 0.12),
            hovertemplate="公元年：%{x}<br>比例：%{y:.2%}<extra></extra>"
        ))
        base_layout(fig, x, "比例（目标命中数 / 有效样本数）")

        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        col_dl, _ = st.columns([1,5])
        with col_dl:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"明代_比例_{dim_name}_{target_bucket}_{ts}.png"
            png, _ = export_png(fig, fname)
            if png:
                st.download_button("⬇️ 下载当前图（PNG）", data=png, file_name=fname, mime="image/png")
            else:
                st.info("若要下载图片，请先安装依赖：`pip install -U kaleido`。")

    else:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=years_all, y=den_arr, name="分母：当年殿试人物",
            marker=dict(color=THEME["bar"], line=dict(color=THEME["bar_edge"], width=0.5)),
            opacity=0.72,
            hovertemplate="公元年：%{x}<br>分母人数：%{y:d}<extra></extra>"
        ))
        custom_ratio = [ (n/d if d>0 else None) for d, n in zip(den_arr, num_arr) ]
        fig.add_trace(go.Scatter(
            x=years_all, y=num_arr, mode="lines+markers",
            name=f"分子：命中人数（{dim_name}＝{target_bucket}）",
            line=dict(width=3.2, color=THEME["line"], shape="spline", smoothing=0.45),
            marker=dict(size=7.5, line=dict(width=1, color=THEME["line_pts_edge"]), symbol="circle"),
            hovertemplate="公元年：%{x}<br>命中人数：%{y:d}<br>当年比例：%{customdata:.2%}<extra></extra>",
            customdata=custom_ratio
        ))
        fig.update_traces(selector=dict(type="bar"), offsetgroup="den", cliponaxis=False)
        base_layout(fig, years_all, "人数（人）")

        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        col_dl, _ = st.columns([1,5])
        with col_dl:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"明代_人数_{dim_name}_{target_bucket}_{ts}.png"
            png, _ = export_png(fig, fname)
            if png:
                st.download_button("⬇️ 下载当前图（PNG）", data=png, file_name=fname, mime="image/png")
            else:
                st.info("若要下载图片，请先安装依赖：`pip install -U kaleido`。")

    # 12) 年号 → 公元年 人名分桶（点击溯源）
    from collections import defaultdict as _dd
    ERA2YEARS: Dict[str, List[int]] = _dd(list)
    for y in years:
        ERA2YEARS[era_of_year(y) or "（未定年号）"].append(y)
    for k in list(ERA2YEARS.keys()):
        ERA2YEARS[k].sort()

    st.session_state.setdefault("sel_person", None)
    st.session_state.setdefault("sel_name", "")
    def select_person(pid: str, nm: str):
        st.session_state["sel_person"] = pid
        st.session_state["sel_name"] = nm

    left, right = st.columns([7,5], gap="large")
    with left:
        st.markdown("### 分类明细")
        era_order = [e for (e,_,_) in MING_ERAS] + [k for k in ERA2YEARS.keys() if k not in [e for (e,_,_) in MING_ERAS]]
        for era in era_order:
            yrs = ERA2YEARS.get(era, [])
            if not yrs: continue
            era_den = 0
            era_num = 0
            for y in yrs:
                denom_people = set(year2people_exam[y])
                if EXCLUDE_CONFLICT:
                    denom_people -= conflict_people
                era_den += len(denom_people)
                era_num += sum(1 for pid in denom_people if person_hits_bucket(pid))
            with st.expander(f"年号「{era}」｜科举人数 {era_den} / 命中人数 {era_num}", expanded=False):
                for y in yrs:
                    denom_people = set(year2people_exam[y])
                    if EXCLUDE_CONFLICT:
                        denom_people -= conflict_people
                    hits = [pid for pid in denom_people if person_hits_bucket(pid)]
                    others = sorted(list(denom_people - set(hits)))
                    title = f"🗓️ 公元 {y} ｜ 分母 {len(denom_people)} / 分子 {len(hits)}"
                    with st.expander(title, expanded=False):
                        cols = st.columns(2)
                        with cols[0]:
                            st.markdown("**命中（人物）**")
                            if not hits: st.write("—")
                            else:
                                cols2 = st.columns(3)
                                for i, pid in enumerate(sorted(hits, key=lambda x: person_name_cache(x))):
                                    nm = person_name_cache(pid)
                                    cols2[i % 3].button(nm, key=f"hit_{era}_{y}_{pid}", on_click=select_person, args=(pid, nm))
                        with cols[1]:
                            st.markdown("**未命中**")
                            if not others: st.write("—")
                            else:
                                cols3 = st.columns(3)
                                for i, pid in enumerate(sorted(others, key=lambda x: person_name_cache(x))):
                                    nm = person_name_cache(pid)
                                    cols3[i % 3].button(nm, key=f"miss_{era}_{y}_{pid}", on_click=select_person, args=(pid, nm))

    with right:
        st.markdown("### 溯源")
        sel_pid = st.session_state.get("sel_person")
        if not sel_pid:
            st.info("提示：点击左侧任意人名以查看溯源文本。")
        else:
            nm = st.session_state.get("sel_name","")
            st.write(f"**人物：** {nm}（{sel_pid}）")
            provs = provenance_for_person(sel_pid, nm)
            if not provs:
                st.warning("未找到文本溯源（含人名的正文未命中）。")
            else:
                for i, ent in enumerate(provs[:60], 1):
                    src, body, conf = ent["src"], ent["body"], ent.get("conf","")
                    title = f"#{i} 书目：{src}" + (f"｜可信度：{conf}" if conf else "")
                    with st.expander(title, expanded=(i==1)):
                        st.markdown(
                            f"<div style='padding:.6rem .75rem;border-left:4px solid #dbeafe;background:#f8fafc;border-radius:8px;line-height:1.6'>{body}</div>",
                            unsafe_allow_html=True
                        )

# 允许单文件直跑
if __name__ == "__main__":
    run(configure_page=True, default_data_path=DEFAULT_DATA)

__all__ = ["run"]
