# modules/person_profile_attrs.py
# -*- coding: utf-8 -*-
"""
人物数据属性履历（模块化）
- 暴露 run(st) 供主入口调用。
- 优先复用主入口设置的全局数据源：st.session_state['kd_data_path'] → 自动加载 Graph 并使用；
  若未设置，则回落到本模块侧栏的“手动加载”。
- 布局：主内容 8 / 溯源 4；顶部搜索栏 8 / 4；减小左右留白。
- 溯源：PropAssertion.about → TextProvenance（繁简体联动高亮）。
"""

import unicodedata
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Union

import streamlit as st
from rdflib import Graph, URIRef, RDF, Literal
from rdflib.namespace import RDFS, SKOS, FOAF

# ====== 常量 ======
NS = "http://mingkg.org/ontology/imperial_exam.owl#"
ABOUT        = NS + "about"          # PropAssertion -> owl:Thing
DERIVED_FROM = NS + "derivedFrom"    # PropAssertion -> TextProvenance

PA_PROP   = NS + "prop"
PA_VAL    = NS + "value"
PA_VALN   = NS + "value_norm"
TP_CONF   = NS + "record_confidence"
TP_SRC    = NS + "Text_source"
TP_BODY   = NS + "Text_body"

# Person 数据属性候选（本地名；按你的本体）
PERSON_DP_LOCAL = [
    "姓名", "字", "学术专长", "学籍", "家庭排行", "户籍地", "户籍类型", "生年",
    str(RDFS.label).rsplit("/",1)[-1],  # rdfs_label（容错）
]

# 允许溯源的属性（白名单）
ALLOWED_FOR_PROV = set(PERSON_DP_LOCAL) | {"姓名", "字", "学术专长", "学籍", "家庭排行", "户籍地", "户籍类型", "生年"}

# ====== 样式（放这里以便 run 内注入一次，不调用 set_page_config） ======
CSS = """
<style>
.block-container {
  max-width: 96vw;
  padding-left: 8px;
  padding-right: 8px;
}
.hint { color:#94a3b8; font-size:12px; }
.small { color:#64748b; font-size:12px; }
.card { border:1px solid #ececec; border-radius:12px; padding:12px 14px; margin:10px 0; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.04); }
.grid6 { display:grid; grid-template-columns: repeat(3, minmax(0, 1fr)); grid-gap:8px; margin-top:8px; }
.chip { width:100%; border:1px solid #c7d2fe; background:#eef2ff; color:#1e293b; border-radius:999px; padding:8px 10px; font-size:13px; text-align:center; cursor:pointer;}
.chip:hover { background:#e0e7ff;}
.mark { background: #fde68a; padding: 0 3px; border-radius: 4px; }
.badge {display:inline-block; padding:2px 8px; border-radius:999px; font-size:12px; background:#eef2ff; color:#0f172a; margin-left:8px; border:1px solid #e2e8f0;}
</style>
"""

# ====== 工具函数（纯函数） ======
_ZW = {u"\u200b", u"\u200c", u"\u200d", u"\ufeff"}

def norm(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if (not ch.isspace()) and (ch not in _ZW))
    return s

def localname_str(u: Union[URIRef, str]) -> str:
    s = str(u)
    for sep in ("#", "/", ":"):
        if sep in s:
            s = s.rsplit(sep, 1)[-1]
    return s

@st.cache_data(show_spinner=False)
def load_graph_any(path: str) -> Graph:
    p = Path(path)
    data = p.read_bytes()
    g = Graph()
    # 容错：优先 nt；失败退回 turtle；再退回 rdflib 自动判定
    try:
        g.parse(data=data, format="nt")
    except Exception:
        try:
            g = Graph(); g.parse(data=data, format="turtle")
        except Exception:
            g = Graph(); g.parse(data=data)
    return g

def find_instances(g: Graph, class_local: str) -> List[URIRef]:
    out = []
    for s, t in g.subject_objects(RDF.type):
        if isinstance(s, URIRef) and localname_str(t) == class_local:
            out.append(s)
    return out

def get_literals_map(g: Graph, node: URIRef) -> Dict[str, List[str]]:
    res: Dict[str, List[str]] = {}
    for p, o in g.predicate_objects(node):
        if isinstance(o, Literal):
            k = localname_str(p)
            res.setdefault(k, []).append(str(o).strip())
    return res

def get_first_display(props: Dict[str, List[str]], keys_local: List[str]) -> str:
    for k in keys_local:
        lk = localname_str(k)
        if lk in props and props[lk]:
            return props[lk][0]
    # 兜底：任意非空文字属性
    for vs in props.values():
        for v in vs:
            if v.strip():
                return v
    return ""

# —— 溯源（繁简体联动） —— #
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

def fuzzy_contains_name(text: str, names: List[str]) -> bool:
    if not text or not names: return False
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
    return [pa for pa in g.subjects(URIRef(ABOUT), node)]

def pa_core_tuple(g: Graph, pa: URIRef) -> Tuple[str, str]:
    prop = next(iter(txt_list(g, pa, PA_PROP)), "")
    valn = next(iter(txt_list(g, pa, PA_VALN)), "")
    if valn:
        return prop, valn
    val = next(iter(txt_list(g, pa, PA_VAL)), "")
    return prop, val

def pa_to_provenances(g: Graph, pa: URIRef) -> List[URIRef]:
    return [tp for tp in g.objects(pa, URIRef(DERIVED_FROM)) if isinstance(tp, URIRef)]

def highlight_value(text: str, value: str) -> str:
    if not text or not value: return text
    t = text
    cands = list(dict.fromkeys([value, t2s(value), s2t(value)]))
    for v in cands:
        vv = v.strip()
        if vv:
            t = t.replace(vv, f"<span class='mark'>{vv}</span>")
    return t

def person_aliases(g: Graph, p: URIRef) -> List[str]:
    props = get_literals_map(g, p)
    candidates = ["姓名", "name", "label", "rdfs_label", "标题", "title",
                  str(FOAF.name), str(RDFS.label), str(SKOS.prefLabel)]
    vals = []
    for k in candidates:
        lk = localname_str(k)
        vals.extend(props.get(lk, []))
    out, seen = [], set()
    for v in vals:
        v = (v or "").strip()
        if v and v not in seen:
            seen.add(v); out.append(v)
    return out

def provenance_groups_for(node: URIRef, g: Graph, person_alias: List[str], allowed_props: Set[str]) -> List[Dict]:
    groups: Dict[Tuple[str,str], Dict] = {}
    def _n(s): return norm(s or "")
    for pa in prop_assertions_about(g, node):
        prop, val = pa_core_tuple(g, pa)
        if not prop or not val:
            continue
        prop_local = localname_str(prop)
        if prop_local not in allowed_props and prop not in allowed_props:
            continue
        for tp in pa_to_provenances(g, pa):
            srcs  = txt_list(g, tp, TP_SRC)
            confs = txt_list(g, tp, TP_CONF)
            bodys = txt_list(g, tp, TP_BODY)
            src  = srcs[0] if srcs else ""
            body = bodys[0] if bodys else ""
            if not body:
                continue
            if not fuzzy_contains_name(body, person_alias):
                continue
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
            G["items"].add((prop_local if prop_local else str(prop), val))
    ordered = sorted(groups.values(), key=lambda d: (0 if d["src"] else 1, d["src"], d["body"]))
    return ordered

# ====== 新增：与入口对接的全局加载器（方案 A） ======
def _ensure_graph_from_global() -> Tuple[Optional[Graph], bool]:
    """
    若入口已设置 kd_data_path，则自动加载 Graph 并写入：
      - st.session_state['graph']
      - st.session_state['loaded_file']
    返回 (Graph or None, 是否本次重新加载)
    """
    kd_path = (st.session_state.get("kd_data_path") or "").strip()
    if not kd_path:
        return None, False
    cur_loaded = st.session_state.get("loaded_file", "")
    g = st.session_state.get("graph")
    if g is not None and cur_loaded == kd_path:
        return g, False
    try:
        g = load_graph_any(kd_path)
        st.session_state["graph"] = g
        st.session_state["loaded_file"] = kd_path
        return g, True
    except Exception as e:
        st.error(f"全局数据源加载失败：{e}")
        return None, False

# ====== 页面子逻辑（只在 run 内调用） ======
def _search_persons(g: Graph, query: str) -> List[Tuple[str, str, str]]:
    persons = find_instances(g, "Person")
    qn = norm(query.strip())
    hits = []
    for p in persons:
        props = get_literals_map(g, p)
        names = []
        for k in ["姓名", str(FOAF.name), str(RDFS.label), str(SKOS.prefLabel), "标题", "title", "label", "rdfs_label"]:
            lk = localname_str(k)
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
    hits.sort(key=lambda t: (t[1]))
    return hits[:50]

def _render_person_block(g: Graph, person_iri: str):
    person = URIRef(person_iri)
    props = get_literals_map(g, person)
    display_name = get_first_display(props, ["姓名", str(FOAF.name), str(RDFS.label), str(SKOS.prefLabel)]) or localname_str(person)
    st.markdown(f"### {display_name}  <span class='hint'>{person_iri}</span>", unsafe_allow_html=True)

    # 组装展示键值（只取白名单）
    kvs: List[Tuple[str, str]] = []
    for key in PERSON_DP_LOCAL:
        lk = localname_str(key)
        vals = props.get(lk, [])
        for v in vals:
            v = v.strip()
            if v:
                kvs.append((lk, v))

    if not kvs:
        st.info("（该人物暂无可展示的数据属性）")
        return

    # 渲染 Chip（点击入溯源）
    st.markdown("<div class='card'><div class='small'>数据属性</div>", unsafe_allow_html=True)
    st.markdown("<div class='grid6'>", unsafe_allow_html=True)
    aliases = person_aliases(g, person)

    def _set_prov(node_iri: str, prop_name: str, prop_value: str, aliases: List[str]):
        st.session_state["__pp_attrs_prov__"] = (node_iri, prop_name, prop_value, aliases)

    for i, (k, v) in enumerate(kvs):
        key_btn = f"pp_attrs_chip_{hash((person_iri, k, v))}_{i}"
        st.button(f"{k}：{v}", key=key_btn, on_click=_set_prov,
                  args=(person_iri, k, v, aliases), use_container_width=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# ====== 对外入口 ======
def run(st):
    """主入口：由 app.py 调用 person_profile_attrs.run(st)"""
    # 仅在本模块页面渲染时注入 CSS（不重复 set_page_config）
    st.markdown(CSS, unsafe_allow_html=True)

    # 优先复用主入口的全局数据源（kd_data_path）
    g, reloaded = _ensure_graph_from_global()

    # 侧栏：仅当全局未设置 kd_data_path 时，才显示本模块的手动加载 UI
    if g is None:
        with st.sidebar:
            st.header("📁 数据")
            data_file = st.text_input("NT/TTL/OWL/RDF 路径", value=st.session_state.get("loaded_file", ""))
            if st.button("加载数据", type="primary", use_container_width=True):
                try:
                    g = load_graph_any(data_file)
                    st.session_state["graph"] = g
                    st.session_state["loaded_file"] = data_file
                    st.success(f"已加载：{data_file}（triples={len(g)}）")
                except Exception as e:
                    st.error(f"加载失败：{e}")
    else:
        # 全局数据源已连接时给出明确提示
        with st.sidebar:
            st.header("📁 数据")
            st.success(f"已连接全局数据源：{st.session_state.get('loaded_file','')}")
            if reloaded:
                st.info("已根据全局数据源自动加载。")

    g = st.session_state.get("graph")
    st.subheader("👤 人物数据属性检索")

    if not g:
        st.info("👈 先在侧栏加载本体数据文件（.nt/.ttl/.owl/.rdf），或在主入口应用全局数据源后自动复用。")
        return

    # 顶部搜索（8/4）
    col_q1, col_q2 = st.columns([8,4])
    with col_q1:
        q = st.text_input("输入人名关键字（繁简均可）", value=st.session_state.get("__pp_attrs_last_query__", ""))
    with col_q2:
        if st.button("🔎 搜索", type="primary", use_container_width=True):
            st.session_state["__pp_attrs_last_query__"] = q

    # 命中列表
    hits: List[Tuple[str,str,str]] = []
    if st.session_state.get("__pp_attrs_last_query__", "").strip():
        hits = _search_persons(g, st.session_state["__pp_attrs_last_query__"])

    if not hits:
        st.info("输入人名并点击“搜索”。")
        return

    st.success(f"命中 {len(hits)} 人（展示前 {len(hits)}）")
    st.divider()

    # 主视图 8/4
    left, right = st.columns([8,4], gap="large")

    with left:
        for idx, (person_iri, _display_name, _reason) in enumerate(hits, 1):
            _render_person_block(g, person_iri)
            st.divider()

    with right:
        st.markdown("### 溯源")
        sel = st.session_state.get("__pp_attrs_prov__")
        if not sel:
            st.info("点击左侧任一属性 Chip 查看溯源。只展示包含该人物‘姓名/字号’的摘录，并对所点值做繁简体联动高亮。")
        else:
            node_iri, prop_name, prop_value, aliases = sel
            node = URIRef(node_iri)
            st.markdown(f"**定位**：人物节点  |  **断言**：{prop_name} = {prop_value}")

            groups = provenance_groups_for(node, g, aliases, ALLOWED_FOR_PROV)
            filtered = [G for G in groups if (prop_name, prop_value) in G["items"]]

            if not filtered:
                st.warning("未找到与此值直接匹配且含人名的溯源。（可能该值暂无溯源或溯源文本未含人名/字号）")
            else:
                for i, G in enumerate(filtered, 1):
                    with st.expander(f"#{i} 书目：{G['src']}", expanded=(i==1)):
                        st.caption("证明： " + "； ".join([f"{p} = {v}" for p,v in sorted(G["items"])]))
                        if G["conf"]:
                            st.write(f"**可信度：** {G['conf']}")
                        st.write("**摘录（自动高亮，含繁简体）**：", unsafe_allow_html=True)
                        st.markdown(highlight_value(G["body"], prop_value), unsafe_allow_html=True)
