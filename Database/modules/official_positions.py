# -*- coding: utf-8 -*-
"""
官职履历展示（Person × OfficialPosition，按“职系”折叠；忽略事件）
- 供主入口 app.py 调用：official_positions.run(st)
- 优先复用主入口注入的全局数据源：
    st.session_state['kd_data_path']  -> 自动加载 Graph 到 st.session_state['graph']
  若未设置全局数据源，则在本模块侧栏提供“加载数据”入口。
- 职系面板内：直接列出官职实例“官职名称”（更醒目），提供溯源 Chip
- 详细信息：默认隐藏“对齐码_*”字段（可按实例开关显示）
- 布局：8/4；去两侧留白；所有 session_state 带模块前缀以防冲突
"""

import unicodedata
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Union

import streamlit as st
from rdflib import Graph, URIRef, Literal
from rdflib.namespace import RDFS, SKOS, FOAF

# ====== 基本配置 ======
NS = "http://mingkg.org/ontology/imperial_exam.owl#"

# —— 对象属性（URI）——
APPOINTED_IN = NS + "appointedIn"    # Person -> AppointmentEvent
HAS_POSITION = NS + "hasPosition"    # AppointmentEvent -> OfficialPosition
ABOUT        = NS + "about"          # PropAssertion -> owl:Thing
DERIVED_FROM = NS + "derivedFrom"    # PropAssertion -> TextProvenance

# —— 数据属性（PropAssertion & TextProvenance）——
PA_PROP  = NS + "prop"
PA_VAL   = NS + "value"
PA_VALN  = NS + "value_norm"
TP_CONF  = NS + "record_confidence"
TP_SRC   = NS + "Text_source"
TP_BODY  = NS + "Text_body"

# —— 文本名候选 —— #
NAME_PROPS = [
    NS + "姓名", NS + "字",
    str(RDFS.label), str(SKOS.prefLabel), str(FOAF.name),
]
POSITION_NAME_PROPS = [
    str(RDFS.label), str(SKOS.prefLabel),
    NS + "官职名称", NS + "原始称谓",
]

# —— 官职目标“本地名”清单（不看命名空间，只看尾巴）—— #
OP_LOCAL_KEYS_ORDERED = [
    "核心职称", "层级", "机构", "职系",
    "修饰_方位", "修饰_副", "地名",
    "对齐码_core", "对齐码_inst", "对齐码_tier",
    "对齐码_loc_core", "对齐码_loc_inst", "对齐码_loc_full",
    "原始称谓", "官职名称",
]
ALIGN_KEYS_PREFIX = ("对齐码_",)  # 对齐码字段前缀集合
WANTED_PA_FOR_PROV: Set[str] = set(OP_LOCAL_KEYS_ORDERED)

# —— 层级排序（执行 → 分管 → 决策 → 其他）—— #
TIER_ORDER = {"执行": 0, "分管": 1, "决策": 2}
def tier_rank(t: str) -> int:
    return TIER_ORDER.get(t or "", 3)

# ====== 样式（仅在 run 内注入；不调用 set_page_config） ======
CSS = """
<style>
.block-container { max-width:96vw; padding-left:8px; padding-right:8px; }
.card { border:1px solid #ececec; border-radius:16px; padding:14px 16px; margin:10px 0; background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.04); }
.badge {display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px; background:#f1f5f9; color:#0f172a; margin-right:6px; border:1px solid #e2e8f0;}
.hint { color:#94a3b8; font-size:12px; }
.small { color:#64748b; font-size:12px; }
.grid6 { display:grid; grid-template-columns: repeat(6, minmax(0, 1fr)); grid-gap:8px; margin-top:6px; }
.chip { width:100%; border:1px solid #c7d2fe; background:#eef2ff; color:#1e293b; border-radius:999px; padding:8px 10px; font-size:13px; text-align:center; cursor:pointer; }
.chip:hover { background:#e0e7ff; }
.mark { background:#fde68a; padding:0 3px; border-radius:4px; }

/* —— 官职实例行（更清晰） —— */
.op-item { display:flex; align-items:center; justify-content:space-between;
  padding:10px 12px; border:1px solid #e5e7eb; border-radius:12px; margin:8px 0;
  background:#fafafa; }
.op-item:hover { background:#f5f7fa; }
.op-name { font-weight:700; font-size:15px; color:#111827; }
.op-meta { color:#6b7280; font-size:12px; margin-left:10px; }
</style>
"""

# ====== 工具函数 ======
_ZW = {u"\u200b", u"\u200c", u"\u200d", u"\ufeff"}
def norm(s: str) -> str:
    if s is None: return ""
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if (not ch.isspace()) and (ch not in _ZW))
    return s

def iri_tail(u: URIRef) -> str:
    s = str(u)
    if "#" in s: s = s.rsplit("#", 1)[-1]
    elif "/" in s: s = s.rsplit("/", 1)[-1]
    return s

@st.cache_data(show_spinner=False)
def load_graph_any(path: str) -> Graph:
    p = Path(path)
    data = p.read_bytes()
    g = Graph()
    # 更鲁棒的判定顺序
    try:
        g.parse(data=data, format="nt")
    except Exception:
        try:
            g = Graph(); g.parse(data=data, format="turtle")
        except Exception:
            g = Graph(); g.parse(data=data)  # 交给 rdflib 自动判定
    return g

def objects_literals(g: Graph, s: URIRef, prop_uri: str) -> List[Literal]:
    return [o for o in g.objects(s, URIRef(prop_uri)) if isinstance(o, Literal)]

def txt_list(g: Graph, s: URIRef, prop_uri: str) -> List[str]:
    vals, seen = [], set()
    for lit in objects_literals(g, s, URIRef(prop_uri)):
        v = str(lit).strip()
        if v and v not in seen:
            seen.add(v); vals.append(v)
    return vals

def persons_with_appts(g: Graph) -> Set[URIRef]:
    return set(s for s,_,_ in g.triples((None, URIRef(APPOINTED_IN), None)) if isinstance(s, URIRef))

def appts_of(g: Graph, p: URIRef) -> List[URIRef]:
    return [o for _,_,o in g.triples((p, URIRef(APPOINTED_IN), None)) if isinstance(o, URIRef)]

def positions_of_appt(g: Graph, e: URIRef) -> List[URIRef]:
    return [o for _,_,o in g.triples((e, URIRef(HAS_POSITION), None)) if isinstance(o, URIRef)]

def pred_tail(p) -> str:
    s = str(p)
    if "#" in s: return s.rsplit("#", 1)[-1]
    if "/" in s: return s.rsplit("/", 1)[-1]
    return s

def values_by_localname(g: Graph, subj: URIRef, local_name: str) -> List[str]:
    out, seen = [], set()
    for _, p, o in g.triples((subj, None, None)):
        if pred_tail(p) == local_name and isinstance(o, Literal):
            v = str(o).strip()
            if v and v not in seen:
                seen.add(v); out.append(v)
    return out

def pick_label(g: Graph, node: URIRef, props: List[str]) -> str:
    for u in props:
        for lit in objects_literals(g, node, URIRef(u)):
            s = str(lit).strip()
            if s: return s
    return ""

def aggregate_names(g: Graph, p: URIRef) -> List[str]:
    vals, seen = [], set()
    for u in NAME_PROPS:
        for lit in objects_literals(g, p, URIRef(u)):
            s = str(lit).strip()
            if s and s not in seen:
                seen.add(s); vals.append(s)
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

# —— 溯源（去重 + 高亮值，仅取含人名的文本） —— #
TS_MAP = {"蘇":"苏","劉":"刘","張":"张","趙":"赵","錢":"钱","孫":"孙","國":"国","會":"会","試":"试","鄉":"乡",
          "進":"进","舉":"举","階":"阶","級":"级","歷":"历","鄭":"郑","黃":"黄","萬":"万","陳":"陈","楊":"杨",
          "馬":"马","許":"许","鄧":"邓","吳":"吴","葉":"叶","羅":"罗","齊":"齐","祿":"禄","祯":"祯","禎":"祯"}
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

def provenance_groups_for(node: URIRef, g: Graph, person_aliases: List[str]) -> List[Dict]:
    groups: Dict[Tuple[str,str], Dict] = {}
    def _n(s): return norm(s or "")
    for pa in prop_assertions_about(g, node):
        prop, val = pa_core_tuple(g, pa)
        # 这里的 prop 是 PropAssertion.prop 的文本值（通常是“本地名”或原字段名）
        if not prop or not val or (prop not in WANTED_PA_FOR_PROV and pred_tail(prop) not in WANTED_PA_FOR_PROV):
            continue
        for tp in pa_to_provenances(g, pa):
            srcs  = txt_list(g, tp, TP_SRC)
            confs = txt_list(g, tp, TP_CONF)
            bodys = txt_list(g, tp, TP_BODY)
            src = srcs[0] if srcs else ""
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
            G["items"].add((prop, val))
    ordered = sorted(groups.values(), key=lambda d: (0 if d["src"] else 1, d["src"], d["body"]))
    return ordered

def highlight_value(text: str, value: str) -> str:
    if not text or not value: return text
    t = text
    cands = list(dict.fromkeys([value, t2s(value), s2t(value)]))
    for v in cands:
        vv = v.strip()
        if vv:
            t = t.replace(vv, f"<span class='mark'>{vv}</span>")
    return t

# ====== 内部状态键（加前缀，避免与其他模块冲突） ======
KEY_LAST_QUERY = "__op_last_query__"
KEY_HITS       = "__op_hits__"
KEY_PROV       = "__op_prov__"  # (node_type, node_iri, prop_name, prop_value, aliases)

# ====== 全局数据源接入（方案 A） ======
def _ensure_graph_from_global() -> Tuple[Optional[Graph], bool]:
    """
    若主入口已设置 kd_data_path，则自动加载 Graph 并写入：
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
        st.session_state.graph = g
        st.session_state.loaded_file = kd_path
        return g, True
    except Exception as e:
        st.error(f"全局数据源加载失败：{e}")
        return None, False

# ====== 辅助 ======
def values_safe_first(lst: List[str], default: str="—") -> str:
    return (lst[0] if lst else default) or default

def search_persons(g: Graph, query: str) -> List[Tuple[str, str, str]]:
    persons = persons_with_appts(g)
    qn = norm(query.strip())
    hits = []
    for p in persons:
        names = aggregate_names(g, p)
        if not names: continue
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
    return hits[:30]

def bucket_positions_by_family(g: Graph, ops: List[URIRef]) -> Dict[str, List[URIRef]]:
    buckets: Dict[str, List[URIRef]] = {}
    for op in ops:
        families = values_by_localname(g, op, "职系")
        if not families:
            buckets.setdefault("(未识别职系)", []).append(op)
        else:
            for fam in sorted(set(families)):
                buckets.setdefault(fam, []).append(op)
    return buckets

def sort_positions_for_bucket(g: Graph, pos_list: List[URIRef]) -> List[URIRef]:
    def key(op):
        tier  = values_safe_first(values_by_localname(g, op, "层级"), "")
        inst  = values_safe_first(values_by_localname(g, op, "机构"), "")
        core  = values_safe_first(values_by_localname(g, op, "核心职称"), "")
        label = pick_label(g, op, POSITION_NAME_PROPS) or iri_tail(op)
        return (tier_rank(tier), inst, core, label, str(op))
    return sorted(pos_list, key=key)

def render_position_name(g: Graph, op: URIRef) -> str:
    name = (values_by_localname(g, op, "官职名称")
            or values_by_localname(g, op, "原始称谓")
            or [pick_label(g, op, POSITION_NAME_PROPS)]
            or ["—"])[0] or "—"
    return name

# ====== 对外入口 ======
def run(st):
    # 注入样式（不 set_page_config，避免与主入口冲突）
    st.markdown(CSS, unsafe_allow_html=True)

    # 优先复用主入口的全局数据源
    g, reloaded = _ensure_graph_from_global()

    # 若主入口未注入图谱，这里提供加载入口；若已复用全局，则隐藏文件输入
    if g is None and (st.session_state.get("graph") is None):
        with st.sidebar:
            st.header("📁 数据")
            data_file = st.text_input("NT/TTL/OWL 路径", value=st.session_state.get("loaded_file", ""))
            if st.button("加载数据", type="primary", use_container_width=True):
                try:
                    g = load_graph_any(data_file)
                    st.session_state.graph = g
                    st.session_state.loaded_file = data_file
                    st.success(f"已加载：{data_file}（triples={len(g)}）")
                except Exception as e:
                    st.error(f"加载失败：{e}")
    else:
        with st.sidebar:
            st.header("📁 数据")
            st.success(f"已连接全局数据源：{st.session_state.get('loaded_file','')}")
            if reloaded:
                st.info("已根据全局数据源自动加载。")

    g: Optional[Graph] = st.session_state.get("graph")
    st.subheader("科举人物官职信息检索")

    if not g:
        st.info("👈 先在侧栏加载本体数据文件（.nt/.ttl/.owl），或在主入口设置全局数据源后自动复用。")
        return

    # 检索栏（8/4）
    col_q1, col_q2 = st.columns([8,4])
    with col_q1:
        q = st.text_input("输入人名关键字（繁简均可：蘇國瓓 / 苏国瓓）", value=st.session_state.get(KEY_LAST_QUERY, ""))
    with col_q2:
        if st.button("🔎 搜索", type="primary", use_container_width=True):
            st.session_state[KEY_LAST_QUERY] = q

    # 搜索
    hits: List[Tuple[str,str,str]] = []
    if st.session_state.get(KEY_LAST_QUERY, "").strip():
        hits = search_persons(g, st.session_state[KEY_LAST_QUERY])

    if not hits:
        st.info("输入人名并点击“搜索”。")
        return

    st.success(f"命中 {len(hits)} 人（展示前 {len(hits)} ）")
    st.divider()

    # 主视图：左（8）/右（4）
    left, right = st.columns([8,4], gap="large")

    # 溯源回调
    def set_prov(node_type: str, node_iri: str, prop_name: str, prop_value: str, aliases: List[str]):
        st.session_state[KEY_PROV] = (node_type, node_iri, prop_name, prop_value, aliases)

    with left:
        for idx, (person_iri, display_name, reason) in enumerate(hits, 1):
            person = URIRef(person_iri)
            aliases = aggregate_names(g, person)
            st.markdown(f"### {idx}. {display_name}  <span class='hint'>{reason}</span>", unsafe_allow_html=True)

            # 收集官职实例
            ops, seen = [], set()
            for e in appts_of(g, person):
                for op in positions_of_appt(g, e):
                    if op not in seen:
                        seen.add(op); ops.append(op)
            st.markdown(f"<span class='small'>官职实例数：{len(ops)}</span>", unsafe_allow_html=True)

            buckets = bucket_positions_by_family(g, ops)
            if not buckets:
                st.info("（无官职实例）")
                st.divider()
                continue

            # —— 职系折叠：展开后列“官职名称”卡片行（不显示层级/核心职称） —— #
            for fam in sorted(buckets.keys(), key=lambda s: (s=="(未识别职系)", s)):
                pos_list_sorted = sort_positions_for_bucket(g, buckets[fam])
                with st.expander(f"🏷️ 职系：{fam}（{len(pos_list_sorted)}）", expanded=False):
                    for k, op in enumerate(pos_list_sorted, 1):
                        name = render_position_name(g, op)
                        iri_short = iri_tail(op)

                        # 左右两列：实例行 + 该字段溯源按钮
                        colA, colB = st.columns([7,5])
                        with colA:
                            st.markdown(
                                f"<div class='op-item'>"
                                f"  <div class='op-name'>{k}. {name}</div>"
                                f"  <div class='op-meta'>ID：{iri_short}</div>"
                                f"</div>",
                                unsafe_allow_html=True
                            )
                        with colB:
                            prov_key = f"prov_name_chip_{hash((str(op), name))}"
                            st.button("官职名称·溯源", key=prov_key, on_click=set_prov,
                                      args=("position", str(op), "官职名称", name, aliases),
                                      use_container_width=True)

                        # 明细（默认隐藏“对齐码_*”）
                        with st.expander("详细信息", expanded=False):
                            key_show_align = f"show_align_{hash(str(op))}"
                            show_align = st.checkbox("显示对齐码字段", value=False, key=key_show_align)

                            chips: List[Tuple[str,str]] = []
                            for local_name in OP_LOCAL_KEYS_ORDERED:
                                if (not show_align) and any(local_name.startswith(prefix) for prefix in ALIGN_KEYS_PREFIX):
                                    continue
                                for v in values_by_localname(g, op, local_name):
                                    chips.append((local_name, v))

                            if chips:
                                st.markdown("<div class='grid6'>", unsafe_allow_html=True)
                                for j, (pk, pv) in enumerate(chips):
                                    key = f"pos_chip_{hash((str(op), pk, pv))}_{j}"
                                    st.button(f"{pk}：{pv}", key=key, on_click=set_prov,
                                              args=("position", str(op), pk, pv, aliases),
                                              use_container_width=True)
                                st.markdown("</div>", unsafe_allow_html=True)
                            else:
                                st.markdown("<div class='hint'>（无可展示的官职属性）</div>", unsafe_allow_html=True)

            st.divider()

    with right:
        st.markdown("### 溯源")
        sel = st.session_state.get(KEY_PROV)
        if not sel:
            st.info("点击左侧任一官职属性 Chip 或『官职名称·溯源』查看溯源。只展示包含该人物‘姓名/字号’的文本，‘书目+摘录’去重，并自动高亮所点的具体值。")
        else:
            node_type, node_iri, prop_name, prop_value, aliases = sel
            node = URIRef(node_iri)
            st.markdown(f"**定位**：官职  |  **断言**：{prop_name} = {prop_value}")

            groups = provenance_groups_for(node, g, aliases)
            filtered = [G for G in groups if (prop_name, prop_value) in G["items"]]

            if not filtered:
                st.warning("未找到与此值直接匹配且含人名的溯源。（可能该值暂无溯源或溯源文本未含人名/字号）")
            else:
                for i, G in enumerate(filtered, 1):
                    with st.expander(f"#{i} 书目：{G['src']}", expanded=(i==1)):
                        st.caption("证明： " + "； ".join([f"{p} = {v}" for p,v in sorted(G['items'])]))
                        if G["conf"]:
                            st.write(f"**可信度：** {G['conf']}")
                        st.write("**摘录（自动高亮）**：", unsafe_allow_html=True)
                        st.markdown(highlight_value(G["body"], prop_value), unsafe_allow_html=True)
