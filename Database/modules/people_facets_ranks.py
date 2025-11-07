# modules/people_facets_ranks.py
# -*- coding: utf-8 -*-
# 人物数据属性分面（学籍类型/学术专长/户籍类型） + 殿试名次分桶
# 兼容主入口：
#   - 使用 run() 无参数
#   - 不使用 st.experimental_rerun，统一 _safe_rerun()
#   - 不在模块中 set_page_config（交由主入口）
# 修复：UnhashableParamError —— 通过 @st.cache_resource 的 _build_indices_from_graph(_g)

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
from collections import defaultdict, deque
import re, unicodedata

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from rdflib import Graph, URIRef, RDF, Literal
from rdflib.namespace import RDFS, SKOS, FOAF

# ====== 全局常量（可按需修改路径） ======
PLOT_FONT = dict(family="Noto Sans CJK SC, Microsoft YaHei UI, Arial Unicode MS, Arial, sans-serif", size=14)
DEFAULT_DATA = r"C:\Users\卢航青\Desktop\本体结构\ontology_academic_fixed.nt"

# ====== 本体常量 ======
NS = "http://mingkg.org/ontology/imperial_exam.owl#"
P_ABOUT = URIRef(NS + "about")
P_PROP  = URIRef(NS + "prop")
P_VALN  = URIRef(NS + "value_norm")
P_VAL   = URIRef(NS + "value")

# 关系/属性本地名
OP_PARTICIPATES_IN = "participatesIn"      # Person -> ParticipationEvent
OP_HAS_EXAM        = "hasExam"             # ParticipationEvent -> ImperialExam
OP_SUB_PLACE       = "isSubPlaceOf"        # Place -> Place
OP_HAS_PLACE       = "hasPlace"            # Event -> Place
OP_TOOK_PLACE_AT   = "tookPlaceAt"
OP_HAPPENED_IN     = "happenedIn"
PLACE_PRED_NAMES   = {OP_HAS_PLACE, OP_TOOK_PLACE_AT, OP_HAPPENED_IN}

DP_EXAM_LEVEL      = "考试等级"
DP_JIADI_LEVEL     = "甲第等级"
DP_RANK            = "名次"

# 类 & 姓名谓词
CLASS_PERSON_NAMES = {"Person", "人物"}
CLASS_PLACE        = "Place"
CLASS_BIRTH        = "BirthEvent"
CLASS_PARTICIPATION= "ParticipationEvent"
CLASS_IMPERIAL_EXAM= "ImperialExam"

PERSON_NAME_KEYS = {
    "姓名","name","label","rdfs_label","标题","title",
    str(FOAF.name), str(RDFS.label), str(SKOS.prefLabel)
}

# 人物属性分面
DP_SCHOOL_TYPE = "学籍类型"      # PropAssertion（prop/value_norm）
DP_ACADEMIC    = "学术专长"
DP_HUJI_TYPE   = "户籍类型"      # 只计官籍/军籍/民籍
ALLOWED_HUJI   = {"官籍","军籍","民籍"}
PA_SCHOOL_TYPE_KEYS = {"学籍类型","學籍類型","学籍_类型","學籍_類型","类型","類型"}

# 年号（时期阈值）
MING = [
    ("洪武",1368,1398),("建文",1399,1402),("永乐",1403,1424),("洪熙",1425,1425),
    ("宣德",1426,1435),("正统",1436,1449),("景泰",1450,1456),("天顺",1457,1464),
    ("成化",1465,1487),("弘治",1488,1505),("正德",1506,1521),("嘉靖",1522,1566),
    ("隆庆",1567,1572),("万历",1573,1620),("泰昌",1620,1620),("天启",1621,1627),
    ("崇祯",1628,1644)
]
QING = [
    ("顺治",1644,1661),("康熙",1662,1722),("雍正",1723,1735),("乾隆",1736,1795),
    ("嘉庆",1796,1820),("道光",1821,1850),("咸丰",1851,1861),("同治",1862,1874),
    ("光绪",1875,1908),("宣统",1909,1911)
]
ERA_LIST = MING + QING
ERA_ORDER = [e[0] for e in ERA_LIST]
def era_of_year(y: int) -> Optional[str]:
    for n,a,b in ERA_LIST:
        if a <= y <= b: return n
    return None

# ====== 兼容性：rerun 包装 ======
def _safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()  # 旧版兼容
        except Exception:
            pass

# ====== 繁简工具（可无 opencc） ======
try:
    from opencc import OpenCC
    _CC_T2S = OpenCC("t2s"); _CC_S2T = OpenCC("s2t")
    def to_s(s: str) -> str: return _CC_T2S.convert(s or "")
    def to_t(s: str) -> str: return _CC_S2T.convert(s or "")
except Exception:
    def to_s(s: str) -> str: return s or ""
    def to_t(s: str) -> str: return s or ""

def strip_ws(s: str) -> str: return re.sub(r"\s+", "", s or "")
def norm_token(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or ""); s = strip_ws(s); return to_s(s)

def contains_kw(body: str, kw: str) -> bool:
    if not body or not kw: return False
    b_s, b_t = strip_ws(to_s(body)), strip_ws(to_t(body))
    k_s, k_t = strip_ws(to_s(kw)),  strip_ws(to_t(kw))
    return (k_s in b_s) or (k_t in b_t)

def localname(u: Union[URIRef, str]) -> str:
    s = str(u or "")
    for sep in ("#", "/", ":"):
        if sep in s: s = s.rsplit(sep, 1)[-1]
    return s

# ====== 名次解析（用于分桶统计） ======
CN_NUM = {'零':0,'〇':0,'○':0,'Ｏ':0,'一':1,'二':2,'两':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10,'百':100,'千':1000}
def parse_cn_numeral(s: str) -> Optional[int]:
    s = (s or "").replace("第","").replace("名","").replace("位","")
    if re.fullmatch(r"\d{1,6}", s):
        try: return int(s)
        except: return None
    total, num = 0, 0
    for ch in s:
        if ch in ("十","百","千"):
            mul = CN_NUM[ch]
            if num == 0: num = 1
            total += num * mul
            num = 0
        else:
            v = CN_NUM.get(ch)
            if v is None: return None
            num = v
    total += num
    if s.startswith("十") and total < 10: total += 10
    return total or None

RANK_CTX_PATTS = [
    r"第\s*([零〇○Ｏ一二两三四五六七八九十百千0-9]{1,6})\s*名",
    r"([零〇○Ｏ一二两三四五六七八九十百千0-9]{1,6})\s*名(?!额)"
]
COUNT_HINTS = ["共","计","名额","人数","额数","榜额","编入","录取名额"]
def extract_ranks_from_text(text: str) -> List[int]:
    text = text or ""
    ranks = []
    for pat in RANK_CTX_PATTS:
        for m in re.finditer(pat, text):
            v = parse_cn_numeral(m.group(1))
            if isinstance(v, int) and v > 0:
                ranks.append(v)
    if not ranks and not any(h in text for h in COUNT_HINTS):
        for m in re.finditer(r"\d{1,4}", text):
            v = int(m.group(0))
            if 0 < v < 100000: ranks.append(v)
        for m in re.finditer(r"[零〇○Ｏ一二两三四五六七八九十百千]{1,6}", text):
            v = parse_cn_numeral(m.group(0))
            if isinstance(v, int) and 0 < v < 100000:
                ranks.append(v)
    seen, out = set(), []
    for v in ranks:
        if v not in seen:
            seen.add(v); out.append(v)
    return out

def rank_token_variants(rank: Optional[int]) -> List[str]:
    if not isinstance(rank, int) or rank <= 0: return []
    base = str(rank); return [f"第{base}名", f"{base}名", f"第 {base} 名"]

# ====== 资源缓存：加载 RDF ======
@st.cache_resource(show_spinner=False)
def load_graph(path: str) -> Graph:
    suf = Path(path).suffix.lower()
    fmt = {".nt":"nt",".ttl":"turtle",".rdf":"xml",".owl":"xml",".xml":"xml"}.get(suf, "turtle")
    g = Graph(); g.parse(path, format=fmt); return g

# ====== 资源缓存：由 Graph 构建索引（注意 _g + hash_funcs） ======
@st.cache_resource(show_spinner=True, hash_funcs={Graph: lambda obj: id(obj)})
def _build_indices_from_graph(_g: Graph):
    # 重要：统一以 g 引用，避免误用未定义变量
    g: Graph = _g

    pred_by_local: Dict[str, Set[URIRef]] = defaultdict(set)
    for _, p, _ in g.triples((None, None, None)):
        pred_by_local[localname(p)].add(p)

    persons: Set[URIRef] = set()
    places : Set[URIRef] = set()
    parts  : Set[URIRef] = set()
    exams  : Set[URIRef] = set()
    births : Set[URIRef] = set()
    for s, t in g.subject_objects(RDF.type):
        if not isinstance(s, URIRef): continue
        ln = localname(t)
        if ln in CLASS_PERSON_NAMES: persons.add(s)
        elif ln == CLASS_PLACE: places.add(s)
        elif ln == CLASS_PARTICIPATION: parts.add(s)
        elif ln == CLASS_IMPERIAL_EXAM: exams.add(s)
        elif ln == CLASS_BIRTH: births.add(s)

    # 文本属性
    def lit_vals(node: Optional[URIRef], key_local: str) -> List[str]:
        if not node: return []
        vals, seen = [], set()
        for pred in pred_by_local.get(key_local, set()):
            for lit in g.objects(node, pred):
                if isinstance(lit, Literal):
                    s = str(lit).strip()
                    if s and s not in seen:
                        seen.add(s); vals.append(s)
        return vals

    # Place 元数据
    place_meta: Dict[str, Dict[str, str]] = {}
    for pl in places:
        mod = (lit_vals(pl, "现代名称") or lit_vals(pl, "历史名称") or [localname(pl)])[0]
        lv  = (lit_vals(pl, "现代区划层级") or [""])[0]
        place_meta[str(pl)] = {"modern": mod, "level": lv}

    # 地点树
    child2parent: Dict[str, Set[str]] = defaultdict(set)
    parent2children: Dict[str, Set[str]] = defaultdict(set)
    for s, p, o in g:
        if isinstance(s, URIRef) and isinstance(o, URIRef) and localname(p) == OP_SUB_PLACE:
            child2parent[str(s)].add(str(o)); parent2children[str(o)].add(str(s))

    # Participation → Exam
    p2pe: Dict[URIRef, List[URIRef]] = defaultdict(list)
    for pred in pred_by_local.get(OP_PARTICIPATES_IN, set()):
        for s, _, pe in g.triples((None, pred, None)):
            if isinstance(s, URIRef) and isinstance(pe, URIRef):
                p2pe[s].append(pe)

    pe2ex: Dict[URIRef, URIRef] = {}
    for pred in pred_by_local.get(OP_HAS_EXAM, set()):
        for pe, _, ex in g.triples((None, pred, None)):
            if isinstance(pe, URIRef) and isinstance(ex, URIRef):
                pe2ex[pe] = ex

    # 事件地点（严格）
    allowed_place_preds: Set[URIRef] = set()
    for nm in PLACE_PRED_NAMES:
        allowed_place_preds |= pred_by_local.get(nm, set())
    def strict_event_place(e: URIRef) -> Optional[str]:
        for p,o in g.predicate_objects(e):
            if (p in allowed_place_preds) and isinstance(o, URIRef):
                if o in places: return str(o)
        return None

    # 出生：BirthEvent -> Place -> Person
    births_set = set(births)
    birth_ev2place: Dict[str, str] = {}
    birth_ev2persons: Dict[str, Set[str]] = defaultdict(set)
    for e in births_set:
        pl = strict_event_place(e)
        if pl: birth_ev2place[str(e)] = pl
        for s2,_ in g.subject_predicates(e):
            if isinstance(s2, URIRef) and s2 in persons:
                birth_ev2persons[str(e)].add(str(s2))
        for _,o in g.predicate_objects(e):
            if isinstance(o, URIRef) and o in persons:
                birth_ev2persons[str(e)].add(str(o))
    person_birth_places: Dict[str, Set[str]] = defaultdict(set)
    for beid, ppl in birth_ev2persons.items():
        pl = birth_ev2place.get(beid)
        if not pl: continue
        for pid in ppl:
            person_birth_places[pid].add(pl)

    # 人名
    def person_name(p: URIRef) -> str:
        for key in PERSON_NAME_KEYS:
            for pred in pred_by_local.get(localname(key), set()):
                for lit in g.objects(p, pred):
                    if isinstance(lit, Literal):
                        s = str(lit).strip()
                        if s: return s
        return localname(p)

    # 判定殿试
    def is_dianshi(pe: URIRef) -> bool:
        ex = pe2ex.get(pe)
        texts = []
        if ex: texts += lit_vals(ex, DP_EXAM_LEVEL)
        texts += lit_vals(pe, DP_EXAM_LEVEL)
        return any(("殿试" in t) or ("殿試" in t) or ("Palace" in t) for t in texts)

    # 年份抽取（时期）
    year_pat = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
    def exam_years(ex_uri: Optional[URIRef], pe: URIRef) -> List[int]:
        yrs = set()
        nodes = [pe] + ([ex_uri] if ex_uri else [])
        for node in nodes:
            if not node: continue
            for _, lit in g.predicate_objects(node):
                if isinstance(lit, Literal):
                    for m in year_pat.findall(str(lit)):
                        y = int(m)
                        if 1000 <= y <= 2100: yrs.add(y)
        return sorted(yrs)

    # participation（仅殿试）
    records = []
    exam_people: Set[str] = set()
    person_exam_years: Dict[str, Set[int]] = defaultdict(set)
    person_main_ranks: Dict[str, List[int]] = defaultdict(list)
    person_main_ranks_by_domain: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {"二甲":[], "三甲":[]})

    def norm_jiadi(s: str) -> str:
        s = s or ""
        if "二甲" in s: return "二甲"
        if "三甲" in s: return "三甲"
        if "一甲" in s: return "一甲"
        return "未明"

    for p in sorted(list(persons), key=lambda x: str(x)):
        pes = p2pe.get(p, [])
        keep = False
        yrs = set()
        for pe in pes:
            if not is_dianshi(pe): continue  # 仅殿试
            keep = True
            ex = pe2ex.get(pe)

            jiadi = "未明"
            vals_j = lit_vals(pe, DP_JIADI_LEVEL) + (lit_vals(ex, DP_JIADI_LEVEL) if ex else [])
            for s in vals_j:
                if s: jiadi = norm_jiadi(s); break

            ranks = []
            for t in lit_vals(pe, DP_RANK):
                rs = extract_ranks_from_text(t)
                if rs: ranks += rs
            # 去重保序
            seen, uniq = set(), []
            for v in ranks:
                if v not in seen:
                    seen.add(v); uniq.append(v)
            main_rank = uniq[0] if uniq else None

            yrs |= set(exam_years(ex, pe))
            records.append(dict(
                person=str(p), name=person_name(p),
                pe=str(pe), ex=str(ex) if ex else "",
                jiadi=jiadi, ranks=uniq, main_rank=main_rank
            ))
            if main_rank:
                person_main_ranks[str(p)].append(int(main_rank))
                if jiadi in ("二甲","三甲"):
                    person_main_ranks_by_domain[str(p)][jiadi].append(int(main_rank))
        if keep: exam_people.add(str(p))
        if yrs:  person_exam_years[str(p)] |= yrs

    # 人物原子属性 & 学籍 PropAssertion
    def person_literals(node: URIRef, local: str) -> List[str]:
        vals, seen = [], set()
        for pred in pred_by_local.get(local, set()):
            for lit in g.objects(node, pred):
                if isinstance(lit, Literal):
                    s = str(lit).strip()
                    if s and s not in seen:
                        seen.add(s); vals.append(s)
        return vals

    def pa_school_types_of_person(person: URIRef) -> List[str]:
        out, seen = [], set()
        for pa in g.subjects(P_ABOUT, person):
            props_txt=[]
            for lit in g.objects(pa, URIRef(NS+"prop")):
                if isinstance(lit, Literal):
                    s = str(lit).strip()
                    if s: props_txt.append(s)
            if not props_txt: continue
            ok=False
            for s in props_txt:
                if localname(s) in PA_SCHOOL_TYPE_KEYS: ok=True; break
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
            k = norm_token(val)
            if k and k not in seen:
                seen.add(k); out.append(val)
        return out

    return dict(
        g=g, pred_by_local=pred_by_local,
        persons=sorted(list(persons), key=lambda x: str(x)),
        records=records,
        exam_people=exam_people,
        person_exam_years={k:list(v) for k,v in person_exam_years.items()},
        person_main_ranks=person_main_ranks,
        person_main_ranks_by_domain=person_main_ranks_by_domain,
        place_meta=place_meta,
        child2parent=child2parent,
        parent2children=parent2children,
        person_birth_places=person_birth_places,
        person_literals=person_literals,
        pa_school_types_of_person=pa_school_types_of_person,
    )

# ====== 资源缓存：便捷入口（用路径） ======
@st.cache_resource(show_spinner=True)
def build_indices(path: str):
    g = load_graph(path)
    return _build_indices_from_graph(g)

# ====== 溯源（精准 + 兜底） ======
def tp_fields(g: Graph, pred_by_local: Dict[str, Set[URIRef]], tp: URIRef) -> Tuple[str,str,str]:
    DP_BODY = pred_by_local.get("Text_body", set())
    DP_CONF = pred_by_local.get("record_confidence", set())
    DP_SRC  = set()
    for k in ["Text_Source","Text_source","来源","书名","source","Source","book","Book","Title","题名"]:
        DP_SRC |= pred_by_local.get(k, set())
    src, body, conf = "", "", ""
    for p,o in g.predicate_objects(tp):
        if isinstance(o, Literal):
            if (p in DP_BODY) and not body: body=str(o).strip()
            if (p in DP_CONF) and not conf: conf=str(o).strip()
            if (p in DP_SRC)  and not src : src =str(o).strip()
    return (src or "（未知书目）"), body, conf

def _collect_with_filter(g: Graph, pred_by_local: Dict[str, Set[URIRef]], nodes: List[str], body_ok_fn):
    P_ABT = pred_by_local.get("about", set())
    P_DFR = pred_by_local.get("derivedFrom", set())
    P_CON = pred_by_local.get("contains", set())
    groups: Dict[Tuple[str,str], Dict] = {}
    def put(tp: URIRef, labels: List[str]):
        src, body, conf = tp_fields(g, pred_by_local, tp)
        if not body_ok_fn(body): return
        k = (src.strip(), body.strip())
        if k not in groups:
            groups[k] = {"src":src or "（未知书目）","body":body,"conf":conf,"labels":set(labels)}
        else:
            groups[k]["labels"] |= set(labels)
            try:
                if conf and (not groups[k]["conf"] or float(conf)>float(groups[k]["conf"])): groups[k]["conf"]=conf
            except: pass
    for u in nodes:
        node = URIRef(u)
        for P in P_CON:
            for tp in g.subjects(P, node): put(tp, [localname(node)])
        for P in P_ABT:
            for pa in g.subjects(P, node):
                for P2 in P_DFR:
                    for tp in g.objects(pa, P2): put(tp, [localname(node)])
    out = sorted([{"src":k[0],"body":k[1],"conf":v["conf"],"labels":sorted(v["labels"])} for k,v in groups.items()],
                 key=lambda d: (d["src"], d["body"]))
    return out

def provenance_for(g, pred_by_local, nodes, person_name: Optional[str], rank: Optional[int], fallback_person_only=True):
    def body_ok_strict(body: str) -> bool:
        if not body: return False
        if person_name and (not contains_kw(body, person_name)): return False
        if isinstance(rank, int) and rank>0:
            toks = rank_token_variants(rank)
            if toks and not any(contains_kw(body, tok) for tok in toks): return False
        return True
    provs = _collect_with_filter(g, pred_by_local, nodes, body_ok_strict)
    used_fallback = False
    if fallback_person_only and person_name and isinstance(rank,int) and rank>0 and len(provs)==0:
        provs = _collect_with_filter(g, pred_by_local, nodes, lambda b: bool(b) and contains_kw(b, person_name))
        used_fallback = True
    return provs, used_fallback

# ====== 入口 ======
def run(st):
    # ====== 侧栏：数据加载 ======
    with st.sidebar:
        st.header("数据")
        data_file = st.text_input("RDF 路径", value=DEFAULT_DATA)
        if st.button("加载/重载", type="primary", use_container_width=True):
            st.cache_resource.clear()
            _safe_rerun()

    # 先构建索引
    S = build_indices(data_file)
    g = S["g"]; pred_by_local = S["pred_by_local"]
    records = S["records"]
    exam_people = S["exam_people"]
    person_exam_years = S["person_exam_years"]
    person_main_ranks = S["person_main_ranks"]
    person_main_ranks_by_domain = S["person_main_ranks_by_domain"]
    place_meta = S["place_meta"]
    child2parent = S["child2parent"]
    parent2children = S["parent2children"]
    person_birth_places = S["person_birth_places"]
    person_literals = S["person_literals"]
    pa_school_types_of_person = S["pa_school_types_of_person"]

    if not g:
        st.info("👈 请先在左侧加载 RDF 数据文件"); return

    # ====== 侧栏：阈值（可选） ======
    st.session_state.setdefault("thresholds", dict(
        enable=False,
        # 地理（出生地）
        place_level="（未选择）", place_id=None,
        # 人物属性
        school=[], acad=[], huji=[],
        # 时期
        left_era="（未选择）", right_era="（未选择）",
    ))
    TH = st.session_state["thresholds"]

    def level_options(place_meta: Dict[str, Dict[str, str]]) -> Dict[str, List[Tuple[str,str]]]:
        Lvl: Dict[str, List[Tuple[str,str]]] = defaultdict(list)
        for pid, m in place_meta.items():
            lv = (m.get("level","") or "").strip()
            mod = m.get("modern","")
            if lv and mod: Lvl[lv].append((mod, pid))
        for k in Lvl: Lvl[k].sort(key=lambda t: t[0])
        return Lvl

    lv_opts = level_options(place_meta)
    exist_lv = sorted(lv_opts.keys())

    with st.sidebar:
        st.divider()
        st.subheader("阈值（可选）")
        enable_threshold = st.checkbox("启用阈值", value=TH.get("enable", False), key="__enable_threshold")

        # 读取旧值
        sel_level = TH.get("place_level","（未选择）")
        target_place = TH.get("place_id", None)
        left_era  = TH.get("left_era","（未选择）")
        right_era = TH.get("right_era","（未选择）")

        if enable_threshold:
            # —— 地理（出生地） ——
            st.markdown("**地理条件（出生地）**")
            cL, cR = st.columns(2)
            with cL:
                sel_level = st.selectbox(
                    "出生地层级", ["（未选择）"]+exist_lv,
                    index=(["（未选择）"]+exist_lv).index(TH.get("place_level","（未选择）"))
                )
            with cR:
                target_place = None
                if sel_level != "（未选择）":
                    all_pairs = lv_opts.get(sel_level, [])
                    place_names = [nm for nm,_ in all_pairs]
                    name2id = dict(all_pairs)
                    default_name = "（未选择）"
                    if TH.get("place_id"):
                        for nm,pid in all_pairs:
                            if pid == TH["place_id"]:
                                default_name = nm; break
                    sel_place_name = st.selectbox(
                        "选择地点（现代名）", ["（未选择）"]+place_names,
                        index=(["（未选择）"]+place_names).index(default_name) if default_name in (["（未选择）"]+place_names) else 0
                    )
                    if sel_place_name != "（未选择）":
                        target_place = name2id.get(sel_place_name)
                else:
                    target_place = None

            st.markdown("---")

            # —— 人物属性 ——
            st.markdown("**人物条件**")
            def all_school_types():
                vals, seen = [], set()
                for p in [URIRef(x) for x in sorted(list(exam_people))]:
                    for v in pa_school_types_of_person(p):
                        k = norm_token(v)
                        if k and k not in seen:
                            seen.add(k); vals.append(v)
                vals.sort(); return vals
            def all_vals(local):
                vals, seen = [], set()
                for p in [URIRef(x) for x in sorted(list(exam_people))]:
                    for v in S["person_literals"](p, local):
                        k = norm_token(v)
                        if k and k not in seen:
                            seen.add(k); vals.append(v)
                vals.sort(); return vals

            cols_attr = st.columns(3)
            with cols_attr[0]:
                sel_school = st.multiselect("学籍类型", all_school_types(), default=TH.get("school", []), key="_tmp_school")
            with cols_attr[1]:
                sel_acad = st.multiselect("学术专长", all_vals(DP_ACADEMIC), default=TH.get("acad", []), key="_tmp_acad")
            with cols_attr[2]:
                sel_huji = st.multiselect("户籍类型（官/军/民）", ["官籍","军籍","民籍"], default=TH.get("huji", []), key="_tmp_huji")

            st.markdown("---")

            # —— 时期 ——
            st.markdown("**时间条件（年号范围）**")
            colE1, colE2 = st.columns(2)
            with colE1:
                left_era  = st.selectbox("左界（含）", ["（未选择）"]+ERA_ORDER,
                                         index=(["（未选择）"]+ERA_ORDER).index(TH.get("left_era","（未选择）")))
            with colE2:
                right_era = st.selectbox("右界（含）", ["（未选择）"]+ERA_ORDER,
                                         index=(["（未选择）"]+ERA_ORDER).index(TH.get("right_era","（未选择）")))

            if st.button("应用阈值", type="primary", use_container_width=True):
                st.session_state["thresholds"] = dict(
                    enable=True,
                    place_level=sel_level, place_id=target_place,
                    school=st.session_state.get("_tmp_school", TH.get("school", [])),
                    acad=st.session_state.get("_tmp_acad", TH.get("acad", [])),
                    huji=st.session_state.get("_tmp_huji", TH.get("huji", [])),
                    left_era=left_era, right_era=right_era,
                )
                _safe_rerun()
        else:
            if st.button("应用阈值（关闭）", type="primary", use_container_width=True):
                st.session_state["thresholds"] = dict(
                    enable=False,
                    place_level="（未选择）", place_id=None,
                    school=[], acad=[], huji=[],
                    left_era="（未选择）", right_era="（未选择）",
                )
                _safe_rerun()

    TH = st.session_state["thresholds"]

    # 仅“有殿试”的人物集合
    P_base_all = [URIRef(p) for p in sorted(list(exam_people))]

    # ====== 顶部：分面选择（已删除“户籍地”） ======
    st.title("人物项计量")
    colA, colB = st.columns([5,5])
    with colA:
        facet = st.selectbox("选择分面", [
            "学籍类型（PropAssertion）","学术专长","户籍类型",
            "殿试名次"
        ], index=0)
    with colB:
        top_n = st.slider("Top N（按人数降序）", 5, 40, 15, 5)

    # ====== 公共：工具函数 ======
    def person_name(node: URIRef) -> str:
        for key in PERSON_NAME_KEYS:
            for pred in pred_by_local.get(localname(key), set()):
                for lit in g.objects(node, pred):
                    if isinstance(lit, Literal):
                        s = str(lit).strip()
                        if s: return s
        return localname(node)

    def descendants_of(pid: str) -> List[str]:
        res, q, seen = [], deque([pid]), {pid}
        while q:
            cur = q.popleft()
            for ch in parent2children.get(cur, set()):
                if ch not in seen:
                    seen.add(ch); res.append(ch); q.append(ch)
        return res
    def ancestors_of(pid: str) -> List[str]:
        res, q, seen = [], deque([pid]), {pid}
        while q:
            cur = q.popleft()
            for pa in child2parent.get(cur, set()):
                if pa not in seen:
                    seen.add(pa); res.append(pa); q.append(pa)
        return res

    # —— 阈值过滤（避免当前分面自筛；户籍地分面已删除，保留逻辑不影响行为） ——
    def pass_filters_person(pid: str, facet_name: str) -> bool:
        if not TH.get("enable", False):
            return True

        node = URIRef(pid)

        # 1) 地理（出生地）
        sel_root = TH.get("place_id")
        if sel_root and facet_name != "户籍地":
            subtree = {sel_root, *descendants_of(sel_root)}
            ok_home = False
            for bp in person_birth_places.get(pid, set()):
                if (bp in subtree) or any(pa in subtree for pa in ancestors_of(bp)):
                    ok_home = True; break
            if not ok_home: return False

        # 2) 人物属性（排除当前分面自身）
        sel_school = TH.get("school", [])
        if sel_school and facet_name != "学籍类型（PropAssertion）":
            mine = {norm_token(v) for v in pa_school_types_of_person(node)}
            need = {norm_token(v) for v in sel_school}
            if not (mine & need): return False

        sel_acad = TH.get("acad", [])
        if sel_acad and facet_name != "学术专长":
            mine = set(S["person_literals"](node, DP_ACADEMIC))
            if not (mine & set(sel_acad)): return False

        sel_huji = TH.get("huji", [])
        if sel_huji and facet_name != "户籍类型":
            mine = {to_s(v).strip() for v in S["person_literals"](node, DP_HUJI_TYPE)}
            mine = {v for v in mine if v in ALLOWED_HUJI}
            if not (mine & set(sel_huji)): return False

        # 3) 时期
        le, re = TH.get("left_era"), TH.get("right_era")
        if le and re and le!="（未选择）" and re!="（未选择）":
            li, ri = ERA_ORDER.index(le), ERA_ORDER.index(re)
            if li>ri: li, ri = ri, li
            window = set(ERA_ORDER[li:ri+1])
            yrs = person_exam_years.get(pid, [])
            if not yrs: return False
            ok=False
            for y in yrs:
                en = era_of_year(y)
                if en and en in window: ok=True; break
            if not ok: return False

        return True

    # 溯源面板状态
    st.session_state.setdefault("sel_person", None)
    st.session_state.setdefault("sel_related", [])
    st.session_state.setdefault("sel_name", "")
    st.session_state.setdefault("sel_rank", None)

    # ====== 布局：左（图与列表）—右（溯源） ======
    L, R = st.columns([7,5], gap="large")

    # ====== 左侧：数据属性三分面（学籍类型 / 学术专长 / 户籍类型） ======
    def facet_values(p: URIRef, facet_name: str) -> List[str]:
        if facet_name == "学籍类型（PropAssertion）":
            return S["pa_school_types_of_person"](p)
        elif facet_name == "学术专长":
            return S["person_literals"](p, DP_ACADEMIC)
        elif facet_name == "户籍类型":
            vals = []
            for v in S["person_literals"](p, DP_HUJI_TYPE):
                v_s = to_s(v).strip()
                if v_s in ALLOWED_HUJI:
                    vals.append(v_s)
            return vals
        return []

    with L:
        if facet != "殿试名次":
            st.caption("主柱状图不显示“未填”；右侧显示缺失人数。点击人名查看右侧溯源。户籍类型仅计官籍/军籍/民籍。")
            persons_filtered = [p for p in P_base_all if pass_filters_person(str(p), facet)]

            # 聚合
            bucket: Dict[str, List[URIRef]] = defaultdict(list)
            missing_list: List[URIRef] = []
            for p in persons_filtered:
                vals = [v for v in facet_values(p, facet) if v and strip_ws(v)]
                if not vals:
                    missing_list.append(p); continue
                for v in vals:
                    bucket[norm_token(v)].append(p)

            # 主图（不含未填）
            items_sorted = sorted(bucket.items(), key=lambda kv: (-len(kv[1]), kv[0]))
            items_top = items_sorted[:top_n]
            df_counts = pd.DataFrame([(k, len(v)) for k,v in items_top], columns=["分面值","人数"])

            st.subheader(f"属性 · {facet}")
            c1, c2 = st.columns([7,3])
            with c1:
                if df_counts.empty:
                    st.info("无可绘制数据（可能全部缺失或阈值过滤过严）。")
                else:
                    fig = px.bar(df_counts, x="分面值", y="人数", hover_data=["分面值","人数"])
                    fig.update_traces(text=df_counts["人数"], textposition="outside")
                    fig.update_layout(template="plotly_white", font=PLOT_FONT,
                                      margin=dict(l=10,r=10,t=40,b=80),
                                      xaxis=dict(tickangle=28, automargin=True, showgrid=True, gridcolor="rgba(0,0,0,0.08)"),
                                      yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.08)"))
                    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            with c2:
                st.metric("缺失人数（未填）", len(missing_list))

            # 分桶名单（含未填）+ 点击溯源
            st.markdown("### 分类明细")
            max_show = st.slider("每类显示前 N 人", 10, 200, 50, 10)

            for k, plist in items_top:
                with st.expander(f"{k}（{len(plist)}）", expanded=False):
                    rows = sorted([(person_name(p), str(p)) for p in plist], key=lambda t: t[0])
                    cols = st.columns(3)
                    for i, (nm, iri) in enumerate(rows[:max_show]):
                        with cols[i % 3]:
                            def _set_sel(pid=iri, nm=nm):
                                st.session_state["sel_person"] = pid
                                st.session_state["sel_related"] = [pid]
                                st.session_state["sel_name"] = nm
                                st.session_state["sel_rank"] = None
                            st.button(nm, key=f"facet_{facet}_{k}_{iri}", on_click=_set_sel, use_container_width=True)
                    if len(rows) > max_show:
                        with st.expander(f"显示全部（共 {len(rows)}）", expanded=False):
                            cols2 = st.columns(3)
                            for j, (nm, iri) in enumerate(rows[max_show:]):
                                with cols2[j % 3]:
                                    def _set_sel2(pid=iri, nm=nm):
                                        st.session_state["sel_person"] = pid
                                        st.session_state["sel_related"] = [pid]
                                        st.session_state["sel_name"] = nm
                                        st.session_state["sel_rank"] = None
                                    st.button(nm, key=f"facet_all_{facet}_{k}_{iri}", on_click=_set_sel2, use_container_width=True)

            # 未填桶
            with st.expander(f"未填（{len(missing_list)}）", expanded=False):
                rows = sorted([(person_name(p), str(p)) for p in missing_list], key=lambda t: t[0])
                cols = st.columns(3)
                for i, (nm, iri) in enumerate(rows[:max_show]):
                    with cols[i % 3]:
                        def _set_sel3(pid=iri, nm=nm):
                            st.session_state["sel_person"] = pid
                            st.session_state["sel_related"] = [pid]
                            st.session_state["sel_name"] = nm
                            st.session_state["sel_rank"] = None
                        st.button(nm, key=f"facet_missing_{facet}_{iri}", on_click=_set_sel3, use_container_width=True)
                if len(rows) > max_show:
                    with st.expander(f"显示全部（共 {len(rows)}）", expanded=False):
                        cols2 = st.columns(3)
                        for j, (nm, iri) in enumerate(rows[max_show:]):
                            with cols2[j % 3]:
                                def _set_sel4(pid=iri, nm=nm):
                                    st.session_state["sel_person"] = pid
                                    st.session_state["sel_related"] = [pid]
                                    st.session_state["sel_name"] = nm
                                    st.session_state["sel_rank"] = None
                                st.button(nm, key=f"facet_missing_all_{facet}_{iri}", on_click=_set_sel4, use_container_width=True)

    # ====== 左侧：殿试名次（分桶） ======
    with L:
        if facet == "殿试名次":
            st.markdown("##")

            # 人群预筛（仅用启用的阈值：地理/人物属性/时期）
            persons_filtered = [p for p in P_base_all if pass_filters_person(str(p), facet)]
            person_ids = {str(p) for p in persons_filtered}

            # === 拿全名次（ParticipationEvent 下多名次逐条展开） ===
            df_rank = pd.DataFrame(records)
            df_rank = df_rank[df_rank["jiadi"].isin(["二甲", "三甲"])].copy()
            df_rank = df_rank[df_rank["person"].isin(person_ids)].copy()
            df_rank["ranks"] = df_rank["ranks"].apply(lambda xs: xs if isinstance(xs, list) else [])
            df_rank = df_rank.explode("ranks", ignore_index=True)
            df_rank = df_rank[df_rank["ranks"].notna()].copy()
            df_rank["名次"] = df_rank["ranks"].astype(int)
            df_rank.drop(columns=["ranks"], inplace=True)

            # 桶设置（展示参数）
            colBS, colCAP, colCON = st.columns(3)
            with colBS:
                bucket_size = st.selectbox("名次分桶步长", [10,20,50,100], index=0)
            with colCAP:
                max_rank_cap = st.number_input("名次上限（各域生成连续桶）", 50, 3000, 400, 50)
            with colCON:
                contradiction_mode = st.selectbox("矛盾处理（同人同域多名次）", ["排除矛盾","包括矛盾"], index=0)
            include_conflict = (contradiction_mode == "包括矛盾")

            def bucket_label_0_based(r: int, step: int) -> str:
                start = ((max(r,1)-1)//step)*step
                end = start + step
                return f"{start}-{end}"
            def ordered_buckets(step: int, cap: int) -> List[str]:
                return [f"{a}-{a+step}" for a in range(0, int(cap), int(step))]
            buckets_seq = ordered_buckets(bucket_size, int(max_rank_cap))

            df_rank["名次桶"] = df_rank["名次"].map(lambda x: bucket_label_0_based(int(x), bucket_size))
            ER = df_rank[df_rank["jiadi"]=="二甲"].copy()
            SR = df_rank[df_rank["jiadi"]=="三甲"].copy()

            # 矛盾标记 & 清洗
            def domain_clean(dfx: pd.DataFrame) -> pd.DataFrame:
                if dfx.empty:
                    return dfx
                dfx = dfx.copy()
                dfx["person_jiadi"] = dfx["person"] + "||" + dfx["jiadi"]
                dup_counts = dfx.groupby("person_jiadi")["名次"].nunique()
                conflict_keys = set(dup_counts[dup_counts >= 2].index)
                dfx["矛盾"] = dfx["person_jiadi"].isin(conflict_keys)
                if not include_conflict:
                    dfx = dfx[~dfx["矛盾"]].copy()
                return dfx

            ER = domain_clean(ER)
            SR = domain_clean(SR)

            # 计数
            def count_domain(dfx: pd.DataFrame, labels: List[str], include_conflict: bool) -> pd.DataFrame:
                if dfx.empty:
                    return pd.DataFrame({"名次桶": labels, "人数": [0]*len(labels)})
                if include_conflict:
                    c = dfx.groupby("名次桶").size()
                else:
                    c = dfx.groupby("名次桶")["person"].nunique()
                c = c.reindex(labels, fill_value=0).reset_index()
                c.columns = ["名次桶", "人数"]
                return c

            cnt_er = count_domain(ER, buckets_seq, include_conflict)
            cnt_sr = count_domain(SR, buckets_seq, include_conflict)

            if cnt_er.empty and cnt_sr.empty:
                st.info("无可绘数据（可能名次缺失或阈值过滤过严）。")
            else:
                fig = go.Figure()
                color_er = "rgba(54,162,235,0.85)"
                color_sr = "rgba(255,159,64,0.85)"
                bg_er    = "rgba(54,162,235,0.10)"
                bg_sr    = "rgba(255,159,64,0.10)"

                fig.add_trace(go.Bar(
                    x=[f"二甲|{b}" for b in buckets_seq], y=cnt_er["人数"], name="二甲", text=cnt_er["人数"],
                    marker=dict(color=color_er, line=dict(width=0.5, color="rgba(0,0,0,0.35)"))
                ))
                fig.add_trace(go.Bar(
                    x=[f"三甲|{b}" for b in buckets_seq], y=cnt_sr["人数"], name="三甲", text=cnt_sr["人数"],
                    marker=dict(color=color_sr, line=dict(width=0.5, color="rgba(0,0,0,0.35)"))
                ))

                tickvals = [f"二甲|{b}" for b in buckets_seq] + [f"三甲|{b}" for b in buckets_seq]
                ticktext = buckets_seq + buckets_seq

                shapes = [
                    dict(type="rect", xref="paper", yref="paper", x0=0.0, x1=0.5, y0=0.0, y1=1.0,
                         fillcolor=bg_er, line=dict(width=0), layer="below"),
                    dict(type="rect", xref="paper", yref="paper", x0=0.5, x1=1.0, y0=0.0, y1=1.0,
                         fillcolor=bg_sr, line=dict(width=0), layer="below"),
                ]
                annotations = [
                    dict(x=0.25, y=1.08, xref="paper", yref="paper", text="二甲", showarrow=False,
                         font=dict(size=16)),
                    dict(x=0.75, y=1.08, xref="paper", yref="paper", text="三甲", showarrow=False,
                         font=dict(size=16)),
                ]
                height = int(max(560, min(2000, len(tickvals)*18 + 260)))
                fig.update_traces(textposition="outside")
                fig.update_layout(
                    barmode="group", template="plotly_white", height=height,
                    margin=dict(l=24, r=16, t=96, b=140), font=PLOT_FONT,
                    xaxis=dict(type="category", tickmode="array", tickvals=tickvals, ticktext=ticktext,
                               tickangle=34, automargin=True, showgrid=False),
                    yaxis=dict(title="记录数（含矛盾多名次）" if include_conflict else "人数（去重）",
                               showgrid=True, gridcolor="rgba(0,0,0,0.08)", zeroline=False),
                    uniformtext_minsize=10, uniformtext_mode="hide",
                    bargap=0.25, shapes=shapes, annotations=annotations,
                    hovermode="closest", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                fig.update_traces(hovertemplate="名次桶：%{x}<br>人数：%{y}<extra></extra>")
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

            # 分桶名单 + 点击溯源
            def _list_domain(domain_name: str, dfx: pd.DataFrame):
                if dfx.empty:
                    st.caption(f"（{domain_name} 无数据）"); return
                st.markdown(f"### {domain_name}")
                dfx = dfx.copy()
                dfx["名次桶"] = dfx["名次"].map(lambda x: bucket_label_0_based(int(x), bucket_size))
                for b in buckets_seq:
                    rows_b = dfx[dfx["名次桶"]==b]
                    if rows_b.empty: continue
                    head = rows_b.sort_values(["名次","name"]).drop_duplicates(subset=["person"])
                    with st.expander(f"{b} —— {domain_name} · 人数 {head.shape[0]}", expanded=False):
                        cols = st.columns(3)
                        for i, row in enumerate(head.itertuples(index=False)):
                            nm, pid, peid, exid = row.name, row.person, row.pe, row.ex
                            rnk = getattr(row, "名次", None)
                            is_conflict = bool(getattr(row, "矛盾", False))
                            def _set_sel(pid=pid, peid=peid, exid=exid, nm=nm, rnk=rnk):
                                st.session_state["sel_person"] = pid
                                st.session_state["sel_related"] = [pid, peid, exid]
                                st.session_state["sel_name"] = nm
                                st.session_state["sel_rank"] = int(rnk) if pd.notna(rnk) else None
                            with cols[i % 3]:
                                badge = " ⚠矛盾" if is_conflict else ""
                                label = f"{nm}" + (f"（{rnk}）" if pd.notna(rnk) else "") + badge
                                st.button(label, key=f"btn_{domain_name}_{b}_{pid}_{peid}",
                                          on_click=_set_sel, use_container_width=True)

            if 'ER' in locals() and 'SR' in locals():
                _list_domain("二甲", ER)
                _list_domain("三甲", SR)

    # ====== 右侧：统一溯源 ======
    with R:
        st.markdown("### 溯源")
        sel_p = st.session_state.get("sel_person")
        if not sel_p:
            st.info("在左侧点击任意人名以查看溯源。")
        else:
            nm = st.session_state.get("sel_name","")
            rk = st.session_state.get("sel_rank", None)
            nodes = [x for x in st.session_state.get("sel_related", []) if x]
            provs, used_fallback = provenance_for(g, pred_by_local, nodes, person_name=nm, rank=rk, fallback_person_only=True)
            if not provs:
                st.warning("未找到文本溯源（人名或名次均未命中）。")
            else:
                if used_fallback and isinstance(rk, int):
                    st.info("提示：未命中“人名+名次”，已降级为“仅人名”。")
                for i, ent in enumerate(provs[:80], 1):
                    src, body, conf, labels = ent["src"], ent["body"], ent["conf"], ent["labels"]
                    head = f"#{i} 书目：{src}" + (f"｜可信度：{conf}" if conf else "")
                    with st.expander(head, expanded=(i==1)):
                        if labels: st.caption("关联节点：" + "、".join(labels))
                        body_show = body
                        if nm:
                            for s in {nm, to_s(nm), to_t(nm)}:
                                if s:
                                    body_show = body_show.replace(s, f"<span style='background:#fde68a'>{s}</span>")
                        if isinstance(rk, int) and rk > 0 and not used_fallback:
                            for tok in rank_token_variants(rk):
                                if tok:
                                    body_show = body_show.replace(tok, f"<span style='background:#c7f9cc'>{tok}</span>")
                        st.markdown(body_show, unsafe_allow_html=True)

    st.caption(
        "口径与说明：\n"
        "• 选择“分面”进行聚合展示；户籍类型仅计官籍/军籍/民籍。\n"
        "• 左侧“阈值（可选）”包含地理（出生地）、人物属性、时期；仅在勾选并应用后生效，且自动避免当前分面自筛。\n"
        "• 名次分桶为统计浏览功能；二域（二甲/三甲）独立分桶；‘包括矛盾’将同人同域多名次逐条计并以⚠标记。"
    )
