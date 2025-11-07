# -*- coding: utf-8 -*-
"""
人物社会关系浏览（分桶=1跳；图谱=多跳；索引提速 + 溯源追踪｜右侧溯源）
- 模块化入口：run(st)
- 不调用 set_page_config；可直接被主入口集成
- 复用主入口注入的 st.session_state['graph']；若无则本模块侧栏可加载并构建索引
- 左右布局 8/4；分桶仅列“对方人物”，点击姓名在右侧渲染溯源
- 溯源筛选仅以“中心人物”的姓名/字号为滤网；对方人物仅用于高亮
"""

import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union
from collections import deque, defaultdict

import streamlit as st
from rdflib import Graph, URIRef, RDF, Literal
from rdflib.namespace import RDFS, SKOS, FOAF
from pyvis.network import Network
from opencc import OpenCC
_cc_t2s = OpenCC('t2s')
_cc_s2t = OpenCC('s2t')

# ========= 常量 & 命名 =========
REL_KEYS_LOCAL = {"关系类型", "關係類型", "RelationType"}
PERSON_NAME_KEYS = {
    "姓名", "name", "label", "rdfs_label", "标题", "title",
    str(FOAF.name), str(RDFS.label), str(SKOS.prefLabel)
}
PRED_PERSON2EVENT_LOCAL = "socialRelationEvent"
CLASS_PERSON_NAMES = {"Person", "人物"}

NS = "http://mingkg.org/ontology/imperial_exam.owl#"
P_ABOUT        = URIRef(NS + "about")
P_DERIVED_FROM = URIRef(NS + "derivedFrom")
P_CONTAINS     = URIRef(NS + "contains")
P_PROP         = URIRef(NS + "prop")
P_VALN         = URIRef(NS + "value_norm")
P_VAL          = URIRef(NS + "value")
TP_CONF        = URIRef(NS + "record_confidence")
TP_SRC         = URIRef(NS + "Text_source")
TP_BODY        = URIRef(NS + "Text_body")

# ========= UI 样式（仅注入 CSS，不 set_page_config）=========
CSS = """
<style>
.block-container { max-width:96vw; padding-left:8px; padding-right:8px; }
.small { color:#64748b; font-size:12px; }
.hint  { color:#94a3b8; font-size:12px; }
.card  { border:1px solid #ececec; border-radius:12px; padding:12px 14px; margin:10px 0; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.04); }
</style>
"""

# ========= 文本规整 & 繁简 =========
_ZW = {u"\u200b", u"\u200c", u"\u200d", u"\ufeff"}

# —— 新增：与入口对接的全局加载器（放在 import 之后） ——
def _detect_format(p: str) -> Optional[str]:
    suf = Path(p).suffix.lower()
    return {".nt":"nt",".ttl":"turtle",".rdf":"xml",".owl":"xml",".xml":"xml"}.get(suf, None)

def _ensure_graph_from_global():
    """若入口已设置 kd_data_path，则自动加载 Graph 并建索引；返回 (Graph or None, loaded: bool)"""
    kd_path = st.session_state.get("kd_data_path", "") or ""
    if not kd_path:
        return None, False
    # 如果已加载且路径未变，直接复用
    cur_loaded = st.session_state.get(KEY_LOADED_FILE, "")
    g = st.session_state.get(KEY_GRAPH)
    if g is not None and cur_loaded == kd_path and st.session_state.get(KEY_P2E):
        return g, False

    # 否则重载
    try:
        fmt = _detect_format(kd_path) or "turtle"
        g = Graph(); g.parse(kd_path, format=fmt)
        st.session_state[KEY_GRAPH] = g
        st.session_state[KEY_LOADED_FILE] = kd_path
        build_indices(g)
        return g, True
    except Exception as e:
        st.error(f"全局数据源加载失败：{e}")
        return None, False



def norm(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if (not ch.isspace()) and (ch not in _ZW))
    return s

def localname(u: Union[URIRef, str]) -> str:
    s = str(u or "")
    for sep in ("#", "/", ":"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s

TS_MAP = {
    "蘇":"苏","劉":"刘","張":"张","趙":"赵","錢":"钱","孫":"孙","國":"国","會":"会","試":"试","鄉":"乡",
    "進":"进","舉":"举","階":"阶","級":"级","歷":"历","鄭":"郑","黃":"黄","萬":"万","陳":"陈","楊":"杨",
    "馬":"马","許":"许","鄧":"邓","吳":"吴","葉":"叶","羅":"罗","齊":"齐","祿":"禄","祯":"祯","禎":"祯"
}
def t2s(s: str) -> str: return "".join(TS_MAP.get(ch, ch) for ch in s)
def s2t(s: str):
    inv = getattr(s2t, "_inv", None)
    if inv is None:
        inv = {v:k for k,v in TS_MAP.items()}
        s2t._inv = inv
    return "".join(inv.get(ch, ch) for ch in s)

def fuzzy_contains_any(text: str, names: List[str]) -> bool:
    if not text or not names: return False
    t_raw = norm(text); t_s, t_t = _cc_t2s.convert(t_raw), _cc_s2t.convert(t_raw)
    for name in names:
        n_raw = norm(name); n_s, n_t = _cc_t2s.convert(n_raw), _cc_s2t.convert(n_raw)
        if (n_raw and n_raw in t_raw) or (n_s and n_s in t_s) or (n_t and n_t in t_t):
            return True
    return False


def highlight_terms(text: str, terms: List[str]) -> str:
    if not text or not terms: return text
    t = text
    variants = []
    for n in terms:
        variants.extend([n, t2s(n), s2t(n)])
    ordered, seen = [], set()
    for v in variants:
        v = v.strip()
        if v and v not in seen:
            seen.add(v); ordered.append(v)
    for v in sorted(ordered, key=len, reverse=True):
        t = t.replace(v, f"<span style='background:#fde68a;padding:0 3px;border-radius:4px'>{v}</span>")
    return t

# ========= 内部状态键（带前缀，避免与其它模块冲突）=========
KEY_LOADED_FILE = "__sr_loaded__"
KEY_GRAPH       = "graph"                  # 与主入口约定
KEY_LAST_QUERY  = "__sr_last_query__"
KEY_PROV_NODE   = "__sr_prov_node__"
# 索引
KEY_PRED_BY_LOCAL   = "__sr_pred_by_local__"
KEY_NAME_PREDS      = "__sr_name_preds__"
KEY_SRE_PREDS       = "__sr_sre_preds__"
KEY_RELTYPE_PREDS   = "__sr_reltype_preds__"
KEY_P2E             = "__sr_person_to_events__"
KEY_E2M             = "__sr_event_to_members__"
KEY_E2R             = "__sr_event_to_reltype__"
KEY_NAME_CACHE      = "__sr_person_name_cache__"

# ========= 构建索引 =========
def build_indices(g: Graph) -> None:
    pred_by_local: Dict[str, Set[URIRef]] = {}
    for _, p, _ in g.triples((None, None, None)):
        pred_by_local.setdefault(localname(p), set()).add(p)

    name_preds: List[URIRef] = []
    for key in PERSON_NAME_KEYS:
        name_preds.extend(list(pred_by_local.get(localname(key), set())))

    sre_preds: Set[URIRef] = set(pred_by_local.get(PRED_PERSON2EVENT_LOCAL, set()))
    reltype_preds: Set[URIRef] = set()
    for k in REL_KEYS_LOCAL:
        reltype_preds |= pred_by_local.get(k, set())

    person_to_events: Dict[URIRef, List[URIRef]] = defaultdict(list)
    event_to_members: Dict[URIRef, List[URIRef]] = defaultdict(list)
    if sre_preds:
        for s, p, o in g.triples((None, None, None)):
            if p in sre_preds and isinstance(o, URIRef):
                if o not in person_to_events[s]:
                    person_to_events[s].append(o)
                if s not in event_to_members[o]:
                    event_to_members[o].append(s)

    event_to_reltype: Dict[URIRef, str] = {}
    for evt in event_to_members.keys():
        val = None
        for rp in reltype_preds:
            for lit in g.objects(evt, rp):
                if isinstance(lit, Literal):
                    x = str(lit).strip()
                    if x:
                        val = x; break
            if val: break
        event_to_reltype[evt] = val or "(未标注关系类型)"

    person_name_cache: Dict[URIRef, str] = {}
    def _fill(person: URIRef):
        for pred in name_preds:
            for lit in g.objects(person, pred):
                if isinstance(lit, Literal):
                    t = str(lit).strip()
                    if t:
                        person_name_cache[person] = t; return
    for p in person_to_events.keys():
        _fill(p)
    for members in event_to_members.values():
        for p in members:
            if p not in person_name_cache:
                _fill(p)

    st.session_state[KEY_PRED_BY_LOCAL] = pred_by_local
    st.session_state[KEY_NAME_PREDS]    = name_preds
    st.session_state[KEY_SRE_PREDS]     = sre_preds
    st.session_state[KEY_RELTYPE_PREDS] = reltype_preds
    st.session_state[KEY_P2E]           = person_to_events
    st.session_state[KEY_E2M]           = event_to_members
    st.session_state[KEY_E2R]           = event_to_reltype
    st.session_state[KEY_NAME_CACHE]    = person_name_cache

# ========= 基础查询 =========
def person_name(person: URIRef) -> str:
    cache = st.session_state.get(KEY_NAME_CACHE, {})
    if person in cache: return cache[person]
    nm = localname(person); cache[person] = nm
    st.session_state[KEY_NAME_CACHE] = cache
    return nm

def aggregate_names(p: URIRef, g: Graph) -> List[str]:
    names, seen = [], set()
    pred_by_local = st.session_state.get(KEY_PRED_BY_LOCAL, {})
    for key in PERSON_NAME_KEYS:
        for pred in pred_by_local.get(localname(key), set()):
            for lit in g.objects(p, pred):
                if isinstance(lit, Literal):
                    s = str(lit).strip()
                    if s and s not in seen:
                        seen.add(s); names.append(s)
    if not names: names.append(localname(p))
    return names

def person_candidates(g: Graph) -> List[URIRef]:
    persons: Set[URIRef] = set(st.session_state.get(KEY_P2E, {}).keys())
    for pred in st.session_state.get(KEY_NAME_PREDS, []):
        for s, _, o in g.triples((None, pred, None)):
            if isinstance(o, Literal):
                persons.add(s)
    for s, _, t in g.triples((None, RDF.type, None)):
        if localname(t) in CLASS_PERSON_NAMES:
            persons.add(s)
    return sorted(persons, key=lambda x: str(x))

def events_of_person(person: URIRef) -> List[URIRef]:
    return list(dict.fromkeys(st.session_state.get(KEY_P2E, {}).get(person, [])))

def relation_type_of_event(evt: URIRef) -> str:
    return st.session_state.get(KEY_E2R, {}).get(evt, "(未标注关系类型)")

def counterpart_people(evt: URIRef, self_person: URIRef) -> List[URIRef]:
    members = st.session_state.get(KEY_E2M, {}).get(evt, [])
    return [m for m in members if m != self_person]

# ========= 溯源 =========
def prop_assertions_about(g: Graph, node: URIRef) -> List[URIRef]:
    return [pa for pa in g.subjects(P_ABOUT, node)]

def pa_to_tp(g: Graph, pa: URIRef) -> List[URIRef]:
    return [tp for tp in g.objects(pa, P_DERIVED_FROM) if isinstance(tp, URIRef)]

def textprov_contains_event(g: Graph, evt: URIRef) -> List[URIRef]:
    return [tp for tp in g.subjects(P_CONTAINS, evt)]

def tp_fields(g: Graph, tp: URIRef) -> Tuple[str, str, str]:
    src  = next((str(x).strip() for x in g.objects(tp, TP_SRC)  if isinstance(x, Literal) and str(x).strip()), "")
    body = next((str(x).strip() for x in g.objects(tp, TP_BODY) if isinstance(x, Literal) and str(x).strip()), "")
    conf = next((str(x).strip() for x in g.objects(tp, TP_CONF) if isinstance(x, Literal) and str(x).strip()), "")
    return src, body, conf

def provenance_groups_for_event_center_only(g: Graph, evt: URIRef, center_aliases: List[str]) -> List[Dict]:
    groups: Dict[Tuple[str, str], Dict] = {}
    def _n(s): return norm(s or "")

    # A: PA → TP
    for pa in prop_assertions_about(g, evt):
        for tp in pa_to_tp(g, pa):
            src, body, conf = tp_fields(g, tp)
            if not body: continue
            if not fuzzy_contains_any(body, center_aliases):  # 仅中心人物作为滤网
                continue
            key = (_n(src), _n(body))
            G = groups.setdefault(key, {"src": src or "（未知书目）", "body": body, "conf": ""})
            if conf:
                try:
                    cur = float(G["conf"]) if G["conf"] else -1e9
                    valf = float(conf)
                    if valf > cur: G["conf"] = conf
                except Exception:
                    pass

    # B: contains → TP
    for tp in textprov_contains_event(g, evt):
        src, body, conf = tp_fields(g, tp)
        if not body: continue
        if not fuzzy_contains_any(body, center_aliases):
            continue
        key = (_n(src), _n(body))
        G = groups.setdefault(key, {"src": src or "（未知书目）", "body": body, "conf": ""})
        if conf:
            try:
                cur = float(G["conf"]) if G["conf"] else -1e9
                valf = float(conf)
                if valf > cur: G["conf"] = conf
            except Exception:
                pass

    ordered = sorted(groups.values(), key=lambda d: (0 if d["src"] else 1, d["src"], d["body"]))
    return ordered

# ========= 图谱 =========
def expand_graph_people_only(g: Graph, center: URIRef, depth: int = 2, max_nodes: int = 300)\
        -> Tuple[Dict[str, Dict], List[Tuple[str, str, str]]]:
    nodes: Dict[str, Dict] = {}
    edges: List[Tuple[str, str, str]] = []
    q = deque([(center, 0)])
    nodes[str(center)] = {"label": person_name(center), "level": 0}
    visited: Set[str] = {str(center)}

    p2e = st.session_state.get(KEY_P2E, {})
    e2m = st.session_state.get(KEY_E2M, {})
    e2r = st.session_state.get(KEY_E2R, {})

    while q and len(nodes) < max_nodes:
        cur, d = q.popleft()
        if d >= depth: continue
        for evt in p2e.get(cur, []):
            rel = e2r.get(evt, "(未标注关系类型)")
            for o in e2m.get(evt, []):
                if o == cur: continue
                s_iri, t_iri = str(cur), str(o)
                if t_iri not in nodes and len(nodes) < max_nodes:
                    nodes[t_iri] = {"label": person_name(o), "level": d + 1}
                if t_iri in nodes:
                    edges.append((s_iri, t_iri, rel))
                if t_iri not in visited and (d + 1) <= depth and len(nodes) < max_nodes:
                    visited.add(t_iri); q.append((o, d + 1))
    return nodes, edges

def render_pyvis_graph(nodes: Dict[str, Dict], edges: List[Tuple[str, str, str]], center_iri: str, merge_edges: bool = True):
    net = Network(height="680px", width="100%", directed=False, notebook=False)
    net.toggle_physics(True)
    COLOR_CENTER = "#ff9f43"; COLOR_L0 = "#feca57"; COLOR_L1 = "#54a0ff"; COLOR_L2 = "#1dd1a1"; COLOR_OTH = "#c8d6e5"
    for iri, info in nodes.items():
        lv = info.get("level", 99)
        label = info.get("label") or localname(iri)
        color = COLOR_OTH
        if iri == center_iri: color = COLOR_CENTER
        elif lv == 0: color = COLOR_L0
        elif lv == 1: color = COLOR_L1
        elif lv >= 2: color = COLOR_L2
        net.add_node(iri, label=label, title=iri, color=color, shape="dot")
    if merge_edges:
        merged: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        for s, t, rel in edges:
            a, b = sorted([s, t]); merged[(a, b)].add(rel or "(未标注关系类型)")
        for (a, b), relset in merged.items():
            label = "、".join(sorted(relset)); title = "；".join(sorted(relset))
            net.add_edge(a, b, label=label, title=title)
    else:
        for s, t, rel in edges:
            r = rel or "(未标注关系类型)"
            net.add_edge(s, t, label=r, title=r)
    st.components.v1.html(net.generate_html(), height=700, scrolling=True)

# ========= 对外入口 =========
def run(st):
    st.markdown(CSS, unsafe_allow_html=True)
    st.subheader("👥 科举人物社会关系检索")

    # 复用主入口注入的图谱；否则在本模块侧栏提供加载与建索引
    g: Optional[Graph] = st.session_state.get(KEY_GRAPH)
    # —— 复用主入口数据源：若入口已应用数据源，则不再显示本模块的文件选择 UI ——
    with st.sidebar:
        st.header("📁 数据")
        g_global, reloaded = _ensure_graph_from_global()
        if g_global is None:
            # 仍支持单独使用本模块（无全局路径时，保留原有输入）
            default_path = st.session_state.get(KEY_LOADED_FILE, "")
            data_file = st.text_input("NT/TTL/RDF/OWL 路径", value=default_path)
            build_index_btn = st.button("加载/更新 并构建索引", type="primary", use_container_width=True)
            if build_index_btn:
                try:
                    fmt = _detect_format(data_file) or "turtle"
                    g = Graph();
                    g.parse(data_file, format=fmt)
                    st.session_state[KEY_GRAPH] = g
                    st.session_state[KEY_LOADED_FILE] = data_file
                    build_indices(g)
                    st.success(f"已加载并建索引：{data_file}（triples={len(g)}）")
                except Exception as e:
                    st.error(f"加载失败：{e}")
        else:
            st.success(f"已连接全局数据源：{st.session_state.get(KEY_LOADED_FILE)}")
            if reloaded:
                st.info("已根据全局数据源自动加载并构建索引。")

    g = st.session_state.get(KEY_GRAPH)
    if not g or not st.session_state.get(KEY_P2E):
        st.info("👈 请在侧栏选择数据文件并点击『加载/更新 并构建索引』。")
        return

    # 搜索栏（8/4）
    c1, c2 = st.columns([8,4])
    with c1:
        q = st.text_input("输入人名关键字（繁简均可）", value=st.session_state.get(KEY_LAST_QUERY, ""))
    with c2:
        if st.button("🔎 搜索", type="primary", use_container_width=True):
            st.session_state[KEY_LAST_QUERY] = q

    # 搜索候选
    def search_persons(query: str) -> List[Tuple[str, str]]:
        qn = norm(query or "")
        if not qn: return []
        hits: List[Tuple[str, str]] = []
        for p in person_candidates(g):
            nm = person_name(p)
            if nm and (norm(nm) == qn or qn in norm(nm)):
                hits.append((str(p), nm))
        hits.sort(key=lambda t: t[1])
        return hits[:50]

    hits: List[Tuple[str, str]] = []
    if st.session_state.get(KEY_LAST_QUERY, "").strip():
        hits = search_persons(st.session_state[KEY_LAST_QUERY])

    if not hits:
        st.info("输入人名并点击“搜索”。")
        return

    st.success(f"命中 {len(hits)} 人")
    st.divider()

    # 参数区
    st.subheader("选择图谱参数")
    colA, colB, colC, colD = st.columns([4,2,2,2])
    with colA:
        center_choice = st.selectbox(
            "选择中心人物（分桶=1跳；图谱=多跳）",
            options=[f"{nm}｜{iri}" for iri, nm in hits],
            index=0
        )
    with colB:
        depth = st.slider("图谱展开跳数（多跳）", min_value=1, max_value=3, value=2)
    with colC:
        max_nodes = st.number_input("最大节点数", min_value=20, max_value=2000, value=300, step=10)
    with colD:
        merge_edges = st.checkbox("合并同对人物多条边标签", value=True)

    choice_iri = center_choice.split("｜", 1)[-1]
    center_person = URIRef(choice_iri)
    center_aliases = aggregate_names(center_person, g)
    center_name = next((nm for iri, nm in hits if iri == choice_iri), localname(choice_iri))

    st.markdown(f"### 👤 {center_name}")

    # 分桶（左 8）+ 溯源（右 4）
    left, right = st.columns([8,4], gap="large")

    # 溯源回调
    def set_prov_rel(event_iri: str, center_aliases_: List[str], target_aliases: List[str]):
        st.session_state[KEY_PROV_NODE] = dict(
            mode="rel", event_iri=event_iri,
            center_aliases=center_aliases_,
            target_aliases=target_aliases
        )

    with left:
        evts = events_of_person(center_person)
        if not evts:
            st.info("（无社会关系事件）")
        else:
            buckets: Dict[str, List[URIRef]] = {}
            for e in evts:
                reltype = relation_type_of_event(e)
                buckets.setdefault(reltype or "(未标注关系类型)", []).append(e)

            for rel in sorted(buckets.keys(), key=lambda s: (s == "(未标注关系类型)", s)):
                group = buckets[rel]
                with st.expander(f"🏷️ 关系：{rel}（{len(group)}）", expanded=False):
                    for e in sorted(group, key=lambda x: str(x)):
                        others = counterpart_people(e, center_person)
                        if not others:
                            st.write(f"- 事件：`{localname(e)}` ｜ 对方：**（无）**")
                            continue
                        # 对方人物按钮（触发右侧溯源）
                        names_for_caption = []
                        for o in others:
                            o_name = person_name(o)
                            o_aliases = aggregate_names(o, g)
                            st.button(
                                o_name,
                                key=f"btn_{hash((str(e), o_name))}",
                                on_click=set_prov_rel,
                                args=(str(e), center_aliases, o_aliases)
                            )
                            names_for_caption.append(o_name)
                        st.caption(f"事件：`{localname(e)}` ｜ 对方：{'、'.join(names_for_caption)}")

    with right:
        st.markdown("### 溯源")
        sel = st.session_state.get(KEY_PROV_NODE)
        if not sel:
            st.info("在左侧点击任意『对方人物』姓名以查看溯源。溯源仅以中心人物姓名/字号为筛选，对方人物仅用于高亮。")
        else:
            event_iri = sel.get("event_iri")
            evt = URIRef(event_iri)
            target_aliases = sel.get("target_aliases", [])
            groups = provenance_groups_for_event_center_only(g, evt, center_aliases)

            if not groups:
                st.warning("未找到包含中心人物姓名的证据文本。（多见于当前数据未为该事件写入 PA/contains 溯源）")
            else:
                for i, G in enumerate(groups, 1):
                    st.write(f"#{i} 书目：{G['src']}" + (f" ｜ 可信度：{G['conf']}" if G.get("conf") else ""))
                    body_hl = highlight_terms(G["body"], target_aliases + center_aliases)
                    st.markdown(body_hl, unsafe_allow_html=True)
                    st.markdown("---")

    # 图谱（多跳）
    st.markdown("#### 社会关系图谱")
    nodes, edges = expand_graph_people_only(g, center_person, depth=depth, max_nodes=max_nodes)
    render_pyvis_graph(nodes, edges, center_iri=str(center_person), merge_edges=merge_edges)
    st.divider()
