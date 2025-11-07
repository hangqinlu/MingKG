# -*- coding: utf-8 -*-
"""
根据 Excel 外部知识表为 JSON 中的 Place 节点批量补齐现代名称、区划层级、经纬度（仅在已有键上覆盖；不新增任何键），含繁简转换。
匹配顺序：
  1) 全名匹配（保留行政后缀，如“普安衛/淮安府”）
  2) 兜底：去一次末尾行政后缀后再匹配

Excel 需要的列名：
- 历史地名
- 现代名称
- 现代行政区划
- 现代政府驻地经纬度   （如 "30.0333°N, 120.8833°E"）

JSON 顶层为列表；元素含 "nodes"/"relationships"；Place 节点属性里历史名键支持 "歷史名稱" 或 "历史名称"。
写回规则（重要）：只覆盖已有键，不新增键或结构；Excel 值为空则不覆盖。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
import json
import re
import os, argparse

def _parse_args():
    ap = argparse.ArgumentParser(description="脚本1：Excel→JSON 补齐（外部配置）")
    ap.add_argument("--excel", dest="excel", default=os.getenv("EXCEL_PATH", r"C:\Users\卢航青\Desktop\地点实例规范.xlsx"),
                    help="Excel 路径（或设置环境变量 EXCEL_PATH）")
    ap.add_argument("--json-in", dest="json_in", default=os.getenv("JSON_IN_PATH", r"C:\Users\卢航青\PycharmProjects\pythonProject11\OSPLR-main\data\group2.json"),
                    help="输入 JSON 路径（或设置环境变量 JSON_IN_PATH）")
    ap.add_argument("--json-out", dest="json_out", default=os.getenv("JSON_OUT_PATH", r"C:\Users\卢航青\PycharmProjects\pythonProject11\图属性数据——OWL数据\data\group2.json"),
                    help="输出 JSON 路径（或设置环境变量 JSON_OUT_PATH）")
    return ap.parse_args()

_args = _parse_args()
EXCEL_PATH   = _args.excel
JSON_IN_PATH = _args.json_in
JSON_OUT_PATH= _args.json_out
# ---------- 繁简转换（可选） ----------
try:
    from opencc import OpenCC
    _CC_T2S = OpenCC("t2s")
    _CC_S2T = OpenCC("s2t")
    def to_s(s: str) -> str: return _CC_T2S.convert(s or "")
    def to_t(s: str) -> str: return _CC_S2T.convert(s or "")
except Exception:
    def to_s(s: str) -> str: return s or ""
    def to_t(s: str) -> str: return s or ""

# ---------- 规范化 ----------
_PUNCT = re.compile(r"[\s\.,;:/\\|·•\-\u3000\u00A0\(\)（）\[\]【】{}“”\"'‘’`~!@#$%^&*+=<>?？！，。、《》：；]+")
_SUFFIXES = ("衛","州","府","縣","郡","路","道","廳","司","堡","城","寨","里","鄉","村","鎮","縣城")

def normalize_base(raw: str) -> str:
    if raw is None: return ""
    s = str(raw).strip()
    s = _PUNCT.sub("", s)
    return s.lower()

def strip_suffix_once(s: str) -> str:
    if not s: return s
    for suf in _SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf):
            return s[:-len(suf)]
    return s

def key_variants_full(name: str) -> List[str]:
    cand = set()
    for v in (name, to_s(name), to_t(name)):
        k = normalize_base(v)
        if k: cand.add(k)
    return list(cand)

def key_variants_stripped(name: str) -> List[str]:
    cand = set()
    for v in (name, to_s(name), to_t(name)):
        k = strip_suffix_once(normalize_base(v))
        if k: cand.add(k)
    return list(cand)

# ---------- 坐标解析 ----------
def parse_latlon(text: Any) -> Tuple[Optional[float], Optional[float]]:
    if text is None: return (None, None)
    s = str(text).strip().replace("，", ",")
    if any(c in s.upper() for c in ("N","S","E","W")) or any(c in s for c in ("北","南","东","西","東")):
        parts = re.split(r"[;, ]+", s)
        lat = lon = None
        for p in parts:
            up = p.upper()
            m = re.search(r"([+-]?\d+(?:\.\d+)?)", p)
            if not m: continue
            val = float(m.group(1))
            if ("N" in up) or ("北" in p): lat = abs(val)
            if ("S" in up) or ("南" in p): lat = -abs(val)
            if ("E" in up) or ("东" in p) or ("東" in p): lon = abs(val)
            if ("W" in up) or ("西" in p): lon = -abs(val)
        return (lat, lon)
    try:
        a, b = [float(x) for x in s.split(",")]
        if abs(a) <= 90 and abs(b) <= 180: return (a, b)     # (lat,lon)
        if abs(b) <= 90 and abs(a) <= 180: return (b, a)     # (lon,lat)
    except Exception:
        pass
    return (None, None)

# ---------- 构建“双层索引”：先 full 后 strip ----------
def build_indexes(df: pd.DataFrame):
    req = ["历史地名","现代名称","现代行政区划","现代政府驻地经纬度"]
    for c in req:
        if c not in df.columns:
            raise ValueError(f"Excel 缺少列：{c}（现有列：{list(df.columns)}）")

    idx_full: Dict[str, Dict[str, Any]] = {}
    idx_strip: Dict[str, Dict[str, Any]] = {}

    for _, row in df.iterrows():
        his = str(row.get("历史地名") or "").strip()
        modern = str(row.get("现代名称") or "").strip()
        level  = str(row.get("现代行政区划") or "").strip()
        coord  = str(row.get("现代政府驻地经纬度") or "").strip()
        lat, lon = parse_latlon(coord)
        rec = {"modern": modern, "level": level, "coord_text": coord, "lat": lat, "lon": lon}

        for k in key_variants_full(his):
            idx_full[k] = rec
        for k in key_variants_stripped(his):
            if k not in idx_strip:
                idx_strip[k] = rec

    return idx_full, idx_strip

# ---------- 仅在已有键上覆盖的助手 ----------
def overwrite_if_present(container: Dict[str, Any], key: str, value: Any):
    """
    只在 container 里已存在 key 且 value 非空/非 None 时覆盖；否则不动。
    对于数值型 lat/lon，允许 0，但不允许 None。
    """
    if not isinstance(container, dict):
        return
    if key in container:
        if isinstance(value, (int, float)):
            if value is not None:
                container[key] = value
        else:
            if value is not None and str(value) != "":
                container[key] = value

# ---------- JSON 写回（不新增任何键/结构） ----------
def fill_place_properties_in_place(obj: Dict[str, Any], know: Dict[str, Any]):
    """
    obj 可以是节点的 properties，也可以是 relationship 的 end（两者都是 dict）。
    只覆盖 obj 中已有的键：
      - 現代名稱 / 现代名称
      - 區劃層級 / 行政区划(若你原结构使用此名)
      - 經緯座標 / 经纬坐标(若你原结构使用此名)
      - lat / lon（仅当原本就存在这些键时）
    不新增任何键，不创建 properties。
    """
    # 名称与层级（兼容繁/简两种命名）
    overwrite_if_present(obj, "現代名稱", know.get("modern", ""))
    overwrite_if_present(obj, "现代名称", know.get("modern", ""))

    overwrite_if_present(obj, "區劃層級", know.get("level", ""))
    overwrite_if_present(obj, "行政区划", know.get("level", ""))
    overwrite_if_present(obj, "现代行政区划", know.get("level", ""))  # 如果你的 JSON 正好用这个键

    # 坐标文本
    overwrite_if_present(obj, "經緯座標", know.get("coord_text", ""))
    overwrite_if_present(obj, "经纬坐标", know.get("coord_text", ""))

    # 数值坐标（仅当原来就有这些键）
    lat, lon = know.get("lat"), know.get("lon")
    if lat is not None:
        overwrite_if_present(obj, "lat", float(lat))
        overwrite_if_present(obj, "緯度", float(lat))
        overwrite_if_present(obj, "纬度", float(lat))
    if lon is not None:
        overwrite_if_present(obj, "lon", float(lon))
        overwrite_if_present(obj, "經度", float(lon))
        overwrite_if_present(obj, "经度", float(lon))

def match_place(his_name: str, idx_full: Dict[str, Dict[str, Any]], idx_strip: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not his_name: return None
    for k in key_variants_full(his_name):
        hit = idx_full.get(k)
        if hit: return hit
    for k in key_variants_stripped(his_name):
        hit = idx_strip.get(k)
        if hit: return hit
    return None

# ---------- 主流程 ----------
def process(json_in: Path, json_out: Path, excel_path: Path):
    df = pd.read_excel(excel_path, dtype=str)
    idx_full, idx_strip = build_indexes(df)

    data = json.loads(json_in.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON 顶层应为列表。")

    unmatched_nodes = 0
    updated_nodes = 0
    updated_rel_ends = 0

    for item in data:
        # nodes → Place
        for node in item.get("nodes", []):
            if node.get("label") != "Place":
                continue
            props = node.get("properties")  # 不新建
            if not isinstance(props, dict):
                unmatched_nodes += 1  # 无 properties 则跳过
                continue
            his = props.get("歷史名稱") or props.get("历史名称")
            if not his:
                unmatched_nodes += 1
                continue

            hit = match_place(his, idx_full, idx_strip)
            if hit:
                fill_place_properties_in_place(props, hit)
                updated_nodes += 1
            else:
                unmatched_nodes += 1

        # relationships → end.label == Place
        for rel in item.get("relationships", []):
            end = rel.get("end", {})
            if not isinstance(end, dict) or end.get("label") != "Place":
                continue
            his2 = end.get("歷史名稱") or end.get("历史名称")
            if not his2:
                continue
            hit2 = match_place(his2, idx_full, idx_strip)
            if hit2:
                # 直接在 end 上“存在即覆盖”
                fill_place_properties_in_place(end, hit2)
                updated_rel_ends += 1

    # 确保输出目录存在
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== 完成 ===")
    print(f"Excel: {excel_path}")
    print(f"JSON 输入: {json_in}")
    print(f"JSON 输出: {json_out}")
    print(f"Place 节点更新（仅覆盖已有键）：{updated_nodes}")
    print(f"Place 节点未匹配：{unmatched_nodes}")
    print(f"关系 end.Place 补齐（仅覆盖已有键）：{updated_rel_ends}")

if __name__ == "__main__":
    process(Path(JSON_IN_PATH), Path(JSON_OUT_PATH), Path(EXCEL_PATH))
