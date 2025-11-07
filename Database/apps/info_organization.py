# apps/info_organization.py
# -*- coding: utf-8 -*-
"""
信息组织模块（封装版）：
- 作为主入口 app.py 的子页面模块引入
- 提供 run(st, configure_page=False)
- 不在此页面 set_page_config，避免与主入口冲突
依赖：streamlit, networkx, pyvis, pandas, transformers, torch, pyyaml
"""

from __future__ import annotations
import json, os, io, time, uuid, re, ast, html
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import pandas as pd
import networkx as nx
from pyvis.network import Network
from transformers import AutoTokenizer, AutoModel
import torch
import yaml
from SPLR.model import RModel
from SPLR.inference import ner_inference
from SPLR.ds import call_openai
from SPLR.utils import build_type_index
from SPLR.osplr_prompt import build_prompt


def _style(st):
    st.markdown(
        """
        <style>
        :root { --fg:#111; --muted:#6b7280; }
        html, body, [class*="css"] {
          font-family: "Noto Sans CJK SC","Source Han Sans SC","Microsoft YaHei UI","Segoe UI","Helvetica Neue",Arial,sans-serif;
          color: var(--fg);
        }
        .block-title{ font-weight:700; font-size:1.1rem; margin:6px 0 8px; }
        .hint{ color:var(--muted); font-size:.9rem; margin-bottom:8px; }
        .hr{ height:1px; background:#e5e7eb; margin:14px 0 18px;}
        .kgtitle{ font-size:1.0rem; font-weight:600; margin:8px 0 6px;}
        .placeholder{ border:1px dashed #d1d5db; border-radius:10px; padding:18px; color:#6b7280; text-align:center; }
        .table-min td, .table-min th { padding:6px 8px !important; font-size:.9rem !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def esc(s: Any) -> str:
    return html.escape("" if s is None else str(s), quote=True)


# ================== 配置与模型 ==================
def load_config() -> Dict[str, Any]:
    # apps/info_organization.py -> 项目根 -> configs/config.yaml
    base_dir = Path(__file__).resolve().parent.parent
    cfg_path = base_dir / 'configs' / 'config.yaml'
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_ner_model_impl(cfg: Dict[str, Any]):
    model_path = cfg['model']['checkpoint_path']
    model_dir = cfg['model']['pretrained_dir']
    ner_type_file = cfg['model']['ner_type_file']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    type_2_index, index_2_type = build_type_index(ner_type_file)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    bert = AutoModel.from_pretrained(model_dir)
    model = RModel(bert, len(type_2_index)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return tokenizer, model, index_2_type, device

def cache_resources(st):
    # 独立的缓存装饰器实例化，避免在 import 时绑定
    @st.cache_resource(show_spinner=True)
    def _cached(cfg_hash: str):
        cfg = load_config()
        return _load_ner_model_impl(cfg)
    return _cached


# ================== 本体规则 ==================
NODE_LABELS = {"Person", "Official Position", "ImperialExam", "Place", "Text"}
REL_LABELS  = {"参与", "担任", "社会关系", "生", "任务执行", "职任", "隶属"}

def node_color(ntype: str) -> str:
    return {
        "Person": "#0ea5e9",
        "Official Position": "#10b981",
        "ImperialExam": "#f59e0b",
        "Place": "#8b5cf6",
        "Text": "#64748b"
    }.get(ntype, "#94a3b8")

def human_name_for_node(label: str, props: Dict[str, Any]) -> str:
    if label == "Person":            return props.get("姓名","").strip() or "未知人物"
    if label == "Official Position": return props.get("官职名称","").strip() or "未知官职"
    if label == "ImperialExam":      return props.get("考试等级","").strip() or "科举"
    if label == "Place":             return props.get("历史名称","").strip() or props.get("现代名称","").strip() or "未知地点"
    if label == "Text":              return (props.get("content","") or "").strip()[:20] or "文本片段"
    return label

def node_tooltip(ntype: str, props: Dict[str, Any]) -> str:
    lines = [f"<b>{esc(ntype)}</b>"]
    for k,v in props.items():
        if v is None or (isinstance(v, str) and not v.strip()): continue
        lines.append(f"{esc(k)}: {esc(v)}")
    return "<br>".join(lines)


# ================== 解析（容错到 Python 字面量） ==================
def normalize_extractions(extracted: Any) -> List[Dict[str, Any]]:
    if isinstance(extracted, dict): return [extracted]
    if isinstance(extracted, list): return extracted
    if isinstance(extracted, str):
        s = extracted.strip()
        m = re.match(r"^```(?:json|JSON|python|py)?\s*(.*?)\s*```$", s, flags=re.S)
        if m: s = m.group(1).strip()
        try:
            data = json.loads(s); return data if isinstance(data, list) else [data]
        except: pass
        items = []
        for ln in s.splitlines():
            ln = ln.strip()
            if not ln: continue
            try: items.append(json.loads(ln))
            except: continue
        if items: return items
        def _sanitize_json_like(text: str) -> str:
            t = text.replace("\ufeff","")
            def _drop_indices(m):
                inner = m.group(1)
                inner = re.sub(r'(?m)^\s*\d+\s*:\s*', '', inner)
                inner = re.sub(r'(?<!")\b\d+\s*:\s*(?=\{)', '', inner)
                return "[" + inner + "]"
            t = re.sub(r'^\s*\[(.*)\]\s*$', _drop_indices, t, flags=re.S)
            t = re.sub(r'"\s*\n\s*"', '",\n"', t)
            t = re.sub(r'\}\s*\n\s*\{', '},\n{', t)
            t = re.sub(r'(\}|\]|\d)\s*\n\s*"', r'\1,\n"', t)
            t = re.sub(r'\}\s*\n*\d+\s*:\s*\{', '},\n{', t)
            t = re.sub(r',\s*,', ',', t)
            t = re.sub(r',(\s*[\}\]])', r'\1', t)
            ts = t.strip()
            if not (ts.startswith('[') and ts.endswith(']')):
                if ts.startswith('{') and ts.endswith('}'): t = f'[{t}]'
            return t
        s2 = _sanitize_json_like(s)
        try:
            data = json.loads(s2); return data if isinstance(data, list) else [data]
        except: pass
        try:
            data = ast.literal_eval(s); return data if isinstance(data, list) else [data]
        except: return []
    return []

# ================== 图构建（严格两遍；仅连接已建节点） ==================
def build_graph_from_json_strict(extractions: List[Dict[str, Any]]) -> Tuple[nx.MultiDiGraph, Dict[str, List[Dict[str, Any]]]]:
    G = nx.MultiDiGraph()
    node_index_by_label: Dict[str, List[Dict[str, Any]]] = {}
    used_name_count: Dict[str, int] = {}

    def uniq_name(name: str) -> str:
        if name not in used_name_count:
            used_name_count[name] = 1
            return name
        used_name_count[name] += 1
        return f"{name}#{used_name_count[name]}"

    # PASS 1: 节点
    for obj in extractions:
        if not isinstance(obj, dict): continue
        t = (obj.get("type") or "").strip().lower()
        if t != "node":
            continue
        label = obj.get("label")
        if label not in NODE_LABELS:
            continue
        props = obj.get("properties", {}) or {}
        disp_name = human_name_for_node(label, props)
        node_id = uniq_name(disp_name)
        G.add_node(
            node_id,
            label=disp_name,
            ntype=label,
            color=node_color(label),
            title=node_tooltip(label, props),
            **props
        )
        node_index_by_label.setdefault(label, []).append({"id": node_id, "props": props})

    # 按关系引用匹配节点（严格等值匹配）
    def find_node_id_from_ref(ref: Dict[str, Any]) -> Optional[str]:
        if not isinstance(ref, dict): return None
        r_label = ref.get("label")
        if r_label not in NODE_LABELS: return None
        candidates = node_index_by_label.get(r_label, [])
        ref_kv = {k: v for k, v in ref.items() if k != "label" and v not in (None, "")}
        if not ref_kv: return None
        for item in candidates:
            props = item["props"] or {}
            ok = True
            for k, v in ref_kv.items():
                if str(props.get(k, "")).strip() != str(v).strip():
                    ok = False; break
            if ok:
                return item["id"]
        return None

    # PASS 2: 关系（仅连接已建节点）
    for obj in extractions:
        if not isinstance(obj, dict): continue
        t = (obj.get("type") or "").strip().lower()
        if t != "relationship":
            continue
        rel_label = obj.get("label")
        if rel_label not in REL_LABELS:
            continue
        start_ref = obj.get("start", {}) or {}
        end_ref   = obj.get("end", {}) or {}
        props     = obj.get("properties", {}) or {}

        u = find_node_id_from_ref(start_ref)
        v = find_node_id_from_ref(end_ref)
        if u is None or v is None:
            # 在页面提示（非致命）
            continue

        eid = str(uuid.uuid4())[:8]
        title_lines = [f"<b>{esc(rel_label)}</b>"] + [
            f"{esc(k)}: {esc(val)}" for k, val in props.items()
            if val not in (None, "") and not (isinstance(val, str) and not val.strip())
        ]
        G.add_edge(u, v, key=eid, label=rel_label, title="<br>".join(title_lines), **props)

    return G, node_index_by_label

# ================== 图可视化 ==================
def graph_to_pyvis_html(G: Optional[nx.MultiDiGraph], height: int = 620) -> str:
    nt = Network(height=f"{height}px", width="100%", notebook=False, directed=True,
                 bgcolor="#ffffff", font_color="#111111")
    nt.barnes_hut()
    if G is not None:
        for nid, data in G.nodes(data=True):
            nt.add_node(nid, label=data.get("label",""), title=data.get("title",""),
                        color=data.get("color","#94a3b8"), shape="dot", size=18)
        for u, v, k, ed in G.edges(keys=True, data=True):
            nt.add_edge(u, v, title=ed.get("title",""), label=ed.get("label",""),
                        arrows="to", physics=True, smooth=True)
    options = {
        "nodes": {"font": {"size": 14}, "borderWidth": 1},
        "edges": {"font": {"size": 12, "align": "horizontal"}, "arrows": {"to": {"enabled": True}}},
        "physics": {"stabilization": True},
        "interaction": {"hover": True, "tooltipDelay": 120, "navigationButtons": True, "keyboard": True}
    }
    nt.set_options(json.dumps(options, ensure_ascii=False))
    return nt.generate_html()

# ================== 页面入口 ==================
def run(st, configure_page: bool = False):
    _style(st)

    # -- 会话状态 --
    if "io_graph" not in st.session_state:
        st.session_state.io_graph: Optional[nx.MultiDiGraph] = None
    if "io_ner_results" not in st.session_state:
        st.session_state.io_ner_results: List[Dict[str, List[str]]] = []
    if "io_line_texts" not in st.session_state:
        st.session_state.io_line_texts: List[str] = []
    if "io_extractions" not in st.session_state:
        st.session_state.io_extractions: List[List[Dict[str, Any]]] = []

    st.markdown('<div class="block-title">自动构建知识图谱</div>', unsafe_allow_html=True)
    st.markdown('<div class="hint">流程：输入 → NER 实时 → 抽取 → 图谱实时；严格按 JSON（只建 node/relationship）。</div>', unsafe_allow_html=True)

    mode = st.radio("输入模式", ["逐行文档", "手动输入"], index=0, horizontal=True, key="io_mode")

    doc_display_mode = None
    if mode == "逐行文档":
        doc_display_mode = st.radio("逐行文档 · 图谱显示策略", ["逐条覆盖显示（清除旧数据）", "累计显示（不清除）"],
                                    index=0, horizontal=True, key="io_disp_mode")

    with st.expander("输入文本", expanded=True):
        raw_lines: List[str] = []
        if mode == "逐行文档":
            up = st.file_uploader("上传 TXT（UTF-8，每行一条）", type=["txt"], key="io_uploader")
            if up is not None:
                content = up.read().decode("utf-8", errors="ignore")
                raw_lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
                st.success(f"已读取 {len(raw_lines)} 行。")
        else:
            text = st.text_area("在此输入（每行一条）", height=180, key="io_textarea",
                                placeholder="示例：\n萬士英，會試第二百九十三名，第三甲二百三十二名賜同進士出身…")
            raw_lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]

    run_btn = st.button("开始抽取并构图（实时显示）", type="primary", use_container_width=True, key="io_run")

    # 左右两列
    ner_col, graph_col = st.columns([5, 7], gap="large")

    with ner_col:
        st.markdown('<div class="block-title">NER 结果</div>', unsafe_allow_html=True)
        ner_placeholder = st.empty()
        if not st.session_state.io_ner_results:
            ner_placeholder.markdown('<div class="placeholder">尚无 NER 结果。</div>', unsafe_allow_html=True)
        else:
            with ner_placeholder.container():
                if mode == "逐行文档" and doc_display_mode and doc_display_mode.startswith("逐条覆盖"):
                    st.markdown(f'<div class="kgtitle">文本</div>', unsafe_allow_html=True)
                    st.write(st.session_state.io_line_texts[-1])
                    st.markdown('<div class="kgtitle">实体表</div>', unsafe_allow_html=True)
                    _render_ner_table(st, st.session_state.io_ner_results[-1])
                else:
                    tabs = st.tabs([f"第{i+1}行" for i in range(len(st.session_state.io_ner_results))])
                    for i, tab in enumerate(tabs):
                        with tab:
                            st.markdown(f'<div class="kgtitle">文本</div>', unsafe_allow_html=True)
                            st.write(st.session_state.io_line_texts[i])
                            st.markdown('<div class="kgtitle">实体表</div>', unsafe_allow_html=True)
                            _render_ner_table(st, st.session_state.io_ner_results[i])

    with graph_col:
        st.markdown('<div class="block-title">知识图谱（可拖拽）</div>', unsafe_allow_html=True)
        graph_placeholder = st.empty()
        if st.session_state.io_graph is None or st.session_state.io_graph.number_of_nodes() == 0:
            graph_placeholder.markdown('<div class="placeholder">尚未生成图谱。</div>', unsafe_allow_html=True)
        else:
            html_page = graph_to_pyvis_html(st.session_state.io_graph, height=620)
            graph_placeholder.components.v1.html(html_page, height=640, scrolling=True)

    # 实时处理
    if run_btn:
        if not raw_lines:
            st.warning("未检测到有效文本，请先输入 / 上传。")
            return

        # 懒加载模型（带缓存）
        cfg = load_config()
        cached_loader = cache_resources(st)
        tokenizer, ner_model, index_2_type, device = cached_loader(cfg_hash=str(hash(json.dumps(cfg, ensure_ascii=False))))

        # 清空历史
        st.session_state.io_ner_results = []
        st.session_state.io_line_texts = []
        st.session_state.io_extractions = []
        st.session_state.io_graph = nx.MultiDiGraph()

        ner_placeholder.empty()
        graph_placeholder.empty()

        accumulating = True
        if mode == "逐行文档" and doc_display_mode and doc_display_mode.startswith("逐条覆盖"):
            accumulating = False

        total = len(raw_lines)
        prog = st.progress(0.0, text="处理中…")

        overall_graph = nx.MultiDiGraph() if accumulating else None

        for i, line in enumerate(raw_lines, 1):
            # 1) NER
            ner_raw = ner_inference(line, tokenizer, device, ner_model, index_2_type)
            if isinstance(ner_raw, dict) and all(isinstance(v, list) for v in ner_raw.values()):
                ner_dict = ner_raw
            else:
                tmp: Dict[str, List[str]] = {}
                if isinstance(ner_raw, list):
                    tmp["实体"] = [str(x) for x in ner_raw]
                elif isinstance(ner_raw, dict):
                    for k,v in ner_raw.items():
                        tmp[k] = v if isinstance(v, list) else [v]
                ner_dict = tmp or {"NER": [str(ner_raw)]}

            # —— NER 界面更新 —— #
            if accumulating:
                st.session_state.io_ner_results.append(ner_dict)
                st.session_state.io_line_texts.append(line)
                with ner_placeholder.container():
                    tabs = st.tabs([f"第{j+1}行" for j in range(len(st.session_state.io_ner_results))])
                    for j, tab in enumerate(tabs):
                        with tab:
                            st.markdown(f'<div class="kgtitle">文本</div>', unsafe_allow_html=True)
                            st.write(st.session_state.io_line_texts[j])
                            st.markdown('<div class="kgtitle">实体表</div>', unsafe_allow_html=True)
                            _render_ner_table(st, st.session_state.io_ner_results[j])
            else:
                st.session_state.io_ner_results = [ner_dict]
                st.session_state.io_line_texts = [line]
                with ner_placeholder.container():
                    st.markdown(f'<div class="kgtitle">文本</div>', unsafe_allow_html=True)
                    st.write(line)
                    st.markdown('<div class="kgtitle">实体表</div>', unsafe_allow_html=True)
                    _render_ner_table(st, ner_dict)

            # 2) LLM 抽取
            prompt = build_prompt(line, ner_raw)
            extracted_raw = call_openai(prompt)
            norm = normalize_extractions(extracted_raw)

            # 3) 构图
            G_line, _ = build_graph_from_json_strict(norm)
            if accumulating:
                overall_graph = nx.compose(overall_graph, G_line)
                st.session_state.io_graph = overall_graph
                G_to_show = overall_graph
            else:
                st.session_state.io_graph = G_line
                G_to_show = G_line

            # —— 图谱更新 —— #
            with graph_placeholder.container():
                html_page = graph_to_pyvis_html(G_to_show, height=620)
                st.components.v1.html(html_page, height=640, scrolling=True)

            prog.progress(i/total, text=f"已完成 {i}/{total}")
            time.sleep(0.02)

        prog.empty()
        st.success("抽取与构图完成。")


# ================== 辅助渲染 ==================
def _render_ner_table(st, ner_dict: Dict[str, List[str]]):
    if not isinstance(ner_dict, dict) or not ner_dict:
        st.caption("未识别到命名实体。"); return
    rows = []
    for cat, items in ner_dict.items():
        if not isinstance(items, list): continue
        for it in items:
            rows.append({"类别": cat, "项": it})
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.caption("未识别到命名实体。")
