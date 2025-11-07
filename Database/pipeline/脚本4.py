# -*- coding: utf-8 -*-
"""
三科统一规范化流水线（殿试 + 乡试 + 会试 | 鲁棒加载 + 解析与写回 + 合并与后处理 + 导出）
- 殿试逻辑完整保留（成化专项修复；弱信息保留&重挂；同年合并；未规范→规范覆盖；状态降级）
- 乡试/会试：与殿试一致的解析流程与写回；分科使用各自 ParticipationEvent 状态字段
- 稳健加载：rdflib → RDF/XML → owlready2（类/属性按 name 或 IRI 后缀双通道查找）
"""

import os
import re
import csv
import uuid
import datetime
import logging
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any, Set

# ==== 日志 ====
LOG_LEVEL = logging.INFO
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("UnifiedExamPipeline")

# ==== 命令行参数 ====
def _parse_args():
    ap = argparse.ArgumentParser(description="三科统一规范化流水线")
    ap.add_argument("--src", dest="src",
                    default=os.getenv("ONTO_FILE", r"C:\Users\卢航青\Desktop\owl_inspect_XXXX\ontology_dedup.nt"),
                    help="输入 NT/TTL/OWL/RDF（ONTO_FILE）")
    ap.add_argument("--out-dir", default=os.getenv("OUT_DIR", r"C:\Users\卢航青\Desktop\本体结构"),
                    help="输出目录（OUT_DIR）")
    ap.add_argument("--include-southern-ming", action="store_true",
                    default=os.getenv("INCLUDE_SOUTHERN_MING","True").lower()=="true",
                    help="是否包含南明年号")
    return ap.parse_args()

_args = _parse_args()
SRC = _args.src
OUT_DIR   = _args.out_dir
INCLUDE_SOUTHERN_MING = _args.include_southern_ming

os.makedirs(OUT_DIR, exist_ok=True)
OUT_OWL   = os.path.join(OUT_DIR, "ontology_dedup_updated.owl")
OUT_TTL   = os.path.join(OUT_DIR, "ontology_dedup_updated.ttl")
OUT_NT    = os.path.join(OUT_DIR, "ontology_dedup_updated.nt")
CSV_DIAN  = os.path.join(OUT_DIR, "per_person_dianshi_summary.csv")
CSV_XIANG = os.path.join(OUT_DIR, "per_person_xiangshi_summary.csv")
CSV_HUI   = os.path.join(OUT_DIR, "per_person_huishi_summary.csv")

# ========= 年号区间 =========
MING_ERAS = [
    ("洪武", 1368, 1398), ("建文", 1399, 1402), ("永乐", 1403, 1424), ("洪熙", 1425, 1425),
    ("宣德", 1426, 1435), ("正统", 1436, 1449), ("景泰", 1450, 1456), ("天顺", 1457, 1464),
    ("成化", 1465, 1487), ("弘治", 1488, 1505), ("正德", 1506, 1521), ("嘉靖", 1522, 1566),
    ("隆庆", 1567, 1572), ("万历", 1573, 1620), ("泰昌", 1620, 1620), ("天启", 1621, 1627),
    ("崇祯", 1628, 1644),
]
SOUTHERN_MING_ERAS = [("弘光", 1645, 1645), ("隆武", 1645, 1646), ("绍武", 1647, 1647), ("永历", 1646, 1662)]
if INCLUDE_SOUTHERN_MING:
    MING_ERAS += SOUTHERN_MING_ERAS
ERA2SPAN: Dict[str, Tuple[int, int]] = {name: (start, end) for name, start, end in MING_ERAS}

# ========= 干支 =========
HEAVENLY = list("甲乙丙丁戊己庚辛壬癸")
EARTHLY  = list("子丑寅卯辰巳午未申酉戌亥")
VALID_GZ = {HEAVENLY[i % 10] + EARTHLY[i % 12] for i in range(60)}

def gz_of_year(year: int) -> str:
    return HEAVENLY[(year - 4) % 10] + EARTHLY[(year - 4) % 12]

def years_matching_gz(gz: str, start: int, end: int) -> List[int]:
    return [yy for yy in range(start, end + 1) if gz_of_year(yy) == gz]

# ========= OCR/录入混淆 & 年号别名 =========
STEM_CONFUSION = {"已": "己", "巳": "己", "攵": "戊", "王": "壬"}
BRANCH_CONFUSION = {"成": "辰", "戍": "戌", "扌": "戌"}
ERA_ALIASES = {
    "龙庆": "隆庆", "隆庆": "隆庆",
    "万曆": "万历", "万厲": "万历", "萬曆": "万历", "萬歷": "万历",
    "正統": "正统", "天順": "天顺", "永樂": "永乐", "崇禎": "崇祯",
    "弘光": "弘光", "隆武": "隆武", "紹武": "绍武", "永曆": "永历", "永历": "永历",
}

# ========= 工具：文本与中文数字 =========
CN_NUM = {
    "零":0, "〇":0, "○":0, "Ｏ":0,
    "一":1, "元":1, "正":1, "两":2, "二":2, "三":3, "四":4, "五":5, "六":6, "七":7, "八":8, "九":9,
    "十":10, "廿":20, "卅":30,
    "壹":1, "贰":2, "貳":2, "叁":3, "參":3, "肆":4, "伍":5, "陆":6, "陸":6, "柒":7, "捌":8, "玖":9, "拾":10
}
def cn_ordinal_to_int(s: str) -> Optional[int]:
    s = (s or "").strip().replace("年","").replace("秊","").replace("年","")
    if re.fullmatch(r"\d{1,3}", s): return int(s)
    if s.startswith("廿"):
        rest = s[1:] or "零"; return 20 + (CN_NUM.get(rest, 0) if rest in CN_NUM else cn_ordinal_to_int(rest) or 0)
    if s.startswith("卅"):
        rest = s[1:] or "零"; return 30 + (CN_NUM.get(rest, 0) if rest in CN_NUM else cn_ordinal_to_int(rest) or 0)
    if "十" in s or "拾" in s:
        s = s.replace("拾","十"); left, _, right = s.partition("十")
        ten = 10 if left in ("","零") else CN_NUM.get(left, 0) * 10
        if ten == 0 and left not in ("","零"): return None
        return ten + (CN_NUM.get(right, 0) if right in CN_NUM else (cn_ordinal_to_int(right) or 0))
    acc = 0
    for ch in s:
        if ch in CN_NUM: acc = acc*10 + CN_NUM[ch]
        else: return None
    return acc or None

def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", "", (s or "").strip())
    for alias, canon in ERA_ALIASES.items():
        if alias in s: s = s.replace(alias, canon)
    s = s.replace("秊","年").replace("年","年").replace("·","").replace("・","").replace("‧","").replace("．",".")
    return s

def levenshtein(a: str, b: str) -> int:
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[j] = min(dp[j] + 1, dp[j-1] + 1, prev + cost)
            prev = cur
    return dp[m]

def canonicalize_stem(ch: str) -> str:
    return ch if ch in HEAVENLY else STEM_CONFUSION.get(ch, ch)

def canonicalize_branch(ch: str) -> str:
    return ch if ch in EARTHLY else BRANCH_CONFUSION.get(ch, ch)

# ========= 更鲁棒的抽取 =========
def extract_gz_candidates(txt: str):
    out = []
    s = normalize_text(txt or "")
    pairs = set(s[i:i+2] for i in range(max(len(s) - 1, 0)))
    for token in pairs:
        if len(token) < 2: continue
        c1, c2 = token[0], token[1]
        if c1 in HEAVENLY and c2 in EARTHLY and (c1+c2) in VALID_GZ:
            out.append({"gz": c1+c2, "confidence": "high", "from": token}); continue
        c1c, c2c = canonicalize_stem(c1), canonicalize_branch(c2)
        if c1c in HEAVENLY and c2c in EARTHLY and (c1c+c2c) in VALID_GZ:
            out.append({"gz": c1c+c2c, "confidence": "medium", "from": token}); continue
        for cand in VALID_GZ:
            if levenshtein(token, cand) <= 1:
                out.append({"gz": cand, "confidence": "low", "from": token}); break
    rank = {"high":3,"medium":2,"low":1}
    best = {}
    for item in out:
        g = item["gz"]
        if g not in best or rank[item["confidence"]] > rank[best[g]["confidence"]]:
            best[g] = item
    return list(best.values())

def extract_era_candidates(txt: str):
    s = normalize_text(txt or ""); cands = []
    for era in ERA2SPAN.keys():
        if era in s: cands.append({"era": era, "from": era})
    for alias, canon in ERA_ALIASES.items():
        if alias in s: cands.append({"era": canon, "from": alias})
    seen = set(); out = []
    for it in cands:
        if it["era"] in ERA2SPAN and it["era"] not in seen:
            seen.add(it["era"]); out.append(it)
    return out

def extract_ord_candidate(txt: str) -> Optional[Tuple[str, int]]:
    s = normalize_text(txt or "")
    m = re.search(r"((?:元|正|两|[〇零一二三四五六七八九十百廿卅壹贰貳叁參肆伍陆陸柒捌玖拾]{1,4}|\d{1,3}))[年秊年]", s)
    if m:
        raw = m.group(1); val = cn_ordinal_to_int(raw); return (raw, val) if val is not None else None
    m2 = re.search(
        r"(?:洪武|建文|永乐|洪熙|宣德|正统|景泰|天顺|成化|弘治|正德|嘉靖|隆庆|万历|泰昌|天启|崇祯|弘光|隆武|绍武|永历)"
        r"[·、，,．.\-—─~～]*"
        r"((?:元|正|两|[〇零一二三四五六七八九十百廿卅壹贰貳叁參肆伍陆陸柒捌玖拾]{1,4}|\d{1,3}))(?![年秊年])",
        s
    )
    if m2:
        raw = m2.group(1); val = cn_ordinal_to_int(raw); return (raw, val) if val is not None else None
    return None

def extract_gy_candidate(txt: str) -> Optional[int]:
    s = normalize_text(txt or "")
    m = re.search(r"(1[3-6]\d{2})年?", s)
    return int(m.group(1)) if m else None

# ========= 时间解析 =========
def parse_time_mixed(raw: str) -> Dict[str, Any]:
    out = {"raw": raw, "gy": None, "era": None, "ord_raw": None, "ord": None, "gz": None, "debug": {}}
    gy = extract_gy_candidate(raw)
    if gy: out["gy"] = gy
    eras = extract_era_candidates(raw)
    if eras:
        out["era"] = eras[0]["era"]; out["debug"]["era_from"] = eras[0]["from"]
    ord_pair = extract_ord_candidate(raw)
    if ord_pair:
        out["ord_raw"], out["ord"] = ord_pair
    gz_list = extract_gz_candidates(raw)
    if gz_list:
        gz_best = sorted(gz_list, key=lambda d: {"high":3,"medium":2,"low":1}[d["confidence"]], reverse=True)[0]
        out["gz"] = gz_best["gz"]
        out["debug"]["gz_from"] = gz_best["from"]
        out["debug"]["gz_confidence"] = gz_best["confidence"]
    if out.get("era") == "成化":
        log.debug("PARSE[成化] raw=%r era=%s ord=%s gz=%s gy=%s",
                  raw, out.get("era"), out.get("ord"), out.get("gz"), out.get("gy"))
    return out

def infer_years_from_parsed(parsed: Dict[str, Any]) -> Tuple[str, List[int], Dict[str, Any]]:
    era, ordv, gz, gy = parsed.get("era"), parsed.get("ord"), parsed.get("gz"), parsed.get("gy")
    extras = {"era": era, "ord": ordv, "gz": gz, "confidence": parsed.get("debug", {}).get("gz_confidence")}
    if era == "成化":
        log.debug("INFER[成化] input: ord=%s gz=%s gy=%s", ordv, gz, gy)
    if gy:
        if era == "成化": log.debug("INFER[成化] => 确定 gy=%s", gy)
        return "确定", [gy], extras
    if era and (ordv is not None) and era in ERA2SPAN:
        s, e = ERA2SPAN[era]; y = s + (ordv - 1)
        if s <= y <= e:
            if era == "成化": log.debug("INFER[成化] era+ord => 确定 y=%s", y)
            return "确定", [y], extras
        else:
            if era == "成化": log.debug("INFER[成化] era+ord 越界 (start=%s,end=%s,ord=%s,y=%s)", s,e,ordv,y)
            return "不可解析", [], {**extras, "note": "序数越界"}
    if era and gz and era in ERA2SPAN:
        s, e = ERA2SPAN[era]
        years = [yy for yy in range(s, e+1) if gz_of_year(yy) == gz]
        years = sorted(list(dict.fromkeys(years)))
        if era == "成化": log.debug("INFER[成化] era+gz 匹配=%s", years)
        if len(years) == 1:
            return "确定", years, extras
        elif len(years) > 1:
            return "不确定", years, extras
        else:
            return "不可解析", [], {**extras, "note": "年号区间内无匹配干支"}
    if gz and not era:
        overall_start = min(s for s, e in ERA2SPAN.values())
        overall_end   = max(e for s, e in ERA2SPAN.values())
        years = [yy for yy in range(overall_start, overall_end+1) if gz_of_year(yy) == gz]
        years = sorted(list(dict.fromkeys(years)))
        return ("不确定", years, extras) if years else ("不可解析", [], {**extras, "note": "全区间无匹配干支"})
    return "不可解析", [], extras

# ========= 鲁棒加载 =========
def load_ontology_robust(path: str):
    """
    支持 .owl/.rdf/.xml 直接载入；.nt/.ttl/.n3 走 rdflib 转 RDF/XML 再 owlready2 载入。
    """
    import tempfile
    from rdflib import Graph
    from owlready2 import get_ontology, default_world

    p = str(Path(path)); ext = Path(p).suffix.lower()
    rdfxml_path = p
    if ext in (".nt",".ttl",".n3"):
        g = Graph()
        fmt = {"nt":"nt","ttl":"turtle","n3":"n3"}[ext[1:]]
        g.parse(p, format=fmt)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".rdf", prefix="onto_rdfxml_")
        g.serialize(destination=tmp.name, format="xml")
        rdfxml_path = tmp.name
    from owlready2 import onto_path
    onto_path.append(os.path.dirname(rdfxml_path))
    world = default_world
    onto = get_ontology(rdfxml_path).load()
    return world, onto, Path(rdfxml_path).parent

# ========= OWL API：类/属性鲁棒解析 =========
def _ends_with_any(iri: str, names) -> bool:
    iri = iri or ""
    for n in names:
        if iri.endswith("#"+n) or iri.endswith("/"+n): return True
    return False

def must_get_class(onto, world, names):
    for c in list(onto.classes()):
        try:
            if c.name in names or _ends_with_any(getattr(c, "iri",""), names): return c
        except: pass
    for c in list(world.classes()):
        try:
            if c.name in names or _ends_with_any(getattr(c, "iri",""), names): return c
        except: pass
    raise RuntimeError(f"未找到类：{names}")

def must_get_objprop(onto, world, names):
    for p in list(onto.object_properties()):
        try:
            if p.name in names or _ends_with_any(getattr(p, "iri",""), names): return p
        except: pass
    for p in list(world.object_properties()):
        try:
            if p.name in names or _ends_with_any(getattr(p, "iri",""), names): return p
        except: pass
    raise RuntimeError(f"未找到对象属性：{names}")

def must_get_dataprop(onto, world, names):
    for p in list(onto.data_properties()):
        try:
            if p.name in names or _ends_with_any(getattr(p, "iri",""), names): return p
        except: pass
    for p in list(world.data_properties()):
        try:
            if p.name in names or _ends_with_any(getattr(p, "iri",""), names): return p
        except: pass
    raise RuntimeError(f"未找到数据属性：{names}")

# ========= rdflib→owlready2 加载 =========
world, onto, _tmpdir = load_ontology_robust(SRC)
log.info("本体已加载：%s", SRC)

# ========= 关键类/属性 =========
PersonCls             = must_get_class(onto, world, ["Person","人物"])
ParticipationEventCls = must_get_class(onto, world, ["ParticipationEvent","参与事件"])
ImperialExamCls       = must_get_class(onto, world, ["ImperialExam","科举考试","考试"])
participatesIn        = must_get_objprop(onto, world, ["participatesIn"])
hasExam               = must_get_objprop(onto, world, ["hasExam"])

# ========= DP/OP Utility =========
from owlready2 import default_world, destroy_entity, onto_path
def dp_get_all(inst, dp_name: str):
    try:
        v = getattr(inst, dp_name); return list(v) if isinstance(v, list) else ([] if v is None else [v])
    except Exception:
        try:
            prop = getattr(onto, dp_name); return list(prop[inst])
        except Exception:
            return []

def dp_get_one(inst, dp_name: str):
    vs = dp_get_all(inst, dp_name)
    return vs[0] if vs else None

def dp_clear(inst, dp_name: str):
    try: setattr(inst, dp_name, None)
    except Exception:
        try: setattr(inst, dp_name, [])
        except Exception:
            try:
                prop = getattr(onto, dp_name); prop.__delitem__(inst)
            except Exception: pass

def dp_set_single(inst, dp_name: str, value):
    if isinstance(value, (list, tuple, set)):
        value = next(iter(value), None)
    if value is not None and not isinstance(value, (str, int, float, bool, datetime.datetime, datetime.date)):
        value = str(value)
    try:
        setattr(inst, dp_name, value); return
    except Exception: pass
    try:
        prop = getattr(onto, dp_name); prop[inst] = [value]; return
    except Exception: pass
    try:
        dp_clear(inst, dp_name); setattr(inst, dp_name, value)
    except Exception:
        log.warning("无法写入数据属性 %s=%s on %s", dp_name, value, inst)

def op_get_all(inst, op_name: str) -> List[Any]:
    try:
        v = getattr(inst, op_name, [])
        return list(v) if isinstance(v, list) else ([v] if v else [])
    except Exception:
        try:
            prop = getattr(onto, op_name); return list(prop[inst])
        except Exception:
            return []

def op_add_unique(inst_subj, op_name: str, inst_obj):
    cur = op_get_all(inst_subj, op_name)
    if inst_obj in cur: return
    cur.append(inst_obj)
    try:
        setattr(inst_subj, op_name, cur)
    except Exception:
        try:
            prop = getattr(onto, op_name); prop[inst_subj] = cur
        except Exception:
            log.warning("无法添加对象属性 %s(%s -> %s)", op_name, inst_subj, inst_obj)

# ========= PropAssertion / TextProvenance =========
def make_uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

def _get_propassertion_class():
    pa = onto.search_one(iri="*PropAssertion")
    if pa is not None: return pa
    try: return onto.PropAssertion
    except AttributeError:
        raise RuntimeError("未找到 PropAssertion 类（请检查本体类名/IRI）")

def create_prop_assertion(about_inst, prop_name: str, value: Any,
                          value_norm: Optional[Any], derived_provs: List[Any]) -> Any:
    PA = _get_prop_assertion_class_cached()
    pa = PA(make_uid("PropAssertion"))
    _safe_set_list(pa, "prop", [prop_name])
    _safe_set_list(pa, "value", [str(value)])
    if value_norm is not None:
        _safe_set_list(pa, "value_norm", [str(value_norm)])
    op_add_unique(pa, "about", about_inst)
    for prov in derived_provs or []:
        op_add_unique(pa, "derivedFrom", prov)
        op_add_unique(prov, "contains", about_inst)
    try: op_add_unique(about_inst, "hasPropAssertion", pa)
    except Exception: pass
    return pa

# 小工具（避免属性名差异导致异常）
def _safe_set_list(inst, attr, lst):
    try:
        setattr(inst, attr, lst)
    except Exception:
        try:
            prop = getattr(onto, attr); prop[inst] = lst
        except Exception:
            pass

# 缓存 PA 类
_PA_CACHE = None
def _get_prop_assertion_class_cached():
    global _PA_CACHE
    if _PA_CACHE is None:
        _PA_CACHE = _get_propassertion_class()
    return _PA_CACHE

def prop_assertions_about(inst) -> List[Any]:
    out = []
    try:
        PAC = _get_prop_assertion_class_cached()
        for pa in PAC.instances():
            if inst in op_get_all(pa, "about"): out.append(pa)
    except Exception: pass
    return out

def provenances_for_exam_time(exam) -> List[Any]:
    provs = []
    for pa in prop_assertions_about(exam):
        if dp_get_one(pa, "prop") in ("考试时间","考试时间_原文","考试时间_规范","年号","序数","干支","时间解析说明"):
            provs += op_get_all(pa, "derivedFrom")
    provs = list(dict.fromkeys(provs))
    if provs: return provs
    # contains(exam)
    TP = onto.search_one(iri="*TextProvenance") or getattr(onto, "TextProvenance", None)
    if TP:
        for tp in TP.instances():
            if exam in op_get_all(tp, "contains"):
                provs.append(tp)
        provs = list(dict.fromkeys(provs))
        if provs: return provs
        # fallback: 造一个低可信度溯源
        auto_tp = TP(make_uid("TextProvenance"))
        _safe_set_list(auto_tp, "record_confidence", ["low"])
        _safe_set_list(auto_tp, "Text_source", ["auto-normalized"])
        _safe_set_list(auto_tp, "Text_body", ["Generated provenance for normalization"])
        op_add_unique(auto_tp, "contains", exam)
        return [auto_tp]
    return []

# ========= 信息量打分 & 原文择优 =========
def info_score_for_exam(exam) -> int:
    score = 0
    props = {(dp_get_one(pa, "prop") or ""): pa for pa in prop_assertions_about(exam)}
    for key in ("年号","序数","干支","考试时间_规范"):
        if key in props: score += 1
    for f in ("主考官","考试地点","科举政策"):
        if dp_get_one(exam, f): score += 1
    score += len(prop_assertions_about(exam))
    used = 0
    for evt in ParticipationEventCls.instances():
        if exam in op_get_all(evt, "hasExam"): used += 1
    score += used
    return score

def score_time_text(txt: str) -> tuple:
    s = normalize_text(txt or "")
    if not s: return (0,0,0,0,0,0)
    has_gy  = 1 if extract_gy_candidate(s) else 0
    eras    = extract_era_candidates(s)
    ordpair = extract_ord_candidate(s)
    gzlist  = extract_gz_candidates(s)
    has_era = 1 if eras else 0
    has_ord = 1 if ordpair else 0
    has_gz  = 1 if gzlist else 0
    return (has_gy, 1 if (has_era and has_ord) else 0, 1 if (has_era and has_gz) else 0,
            1 if (not has_era and has_gz) else 0, 1 if (has_era and not(has_ord or has_gz)) else 0,
            len(s))

def pick_best_time_text(exam) -> str:
    cands = []
    era = ordv = gz = None
    for pa in prop_assertions_about(exam):
        p = dp_get_one(pa, "prop")
        v = dp_get_one(pa, "value") or dp_get_one(pa, "value_norm")
        if not v: continue
        if p == "考试时间_原文": cands.append(str(v))
        elif p == "年号": era = str(v)
        elif p == "序数":
            vv = str(v); ordv = vv if vv.endswith("年") else vv + "年"
        elif p == "干支": gz = str(v)
    if era and ordv: cands.append(f"{era}{ordv}")
    if era and gz:   cands.append(f"{era}{gz}")
    if era:          cands.append(f"{era}中")
    if gz:           cands.append(gz)
    v = dp_get_one(exam, "考试时间")
    if isinstance(v, (str, int)) and str(v).strip(): cands.append(str(v))
    if not cands: return ""
    best = sorted(cands, key=score_time_text, reverse=True)[0]
    if "成化" in normalize_text(best):
        log.debug("PICK[成化] exam=%s best=%r pool=%r", getattr(exam,"name",""), best, cands)
    return best

# ========= 业务：检索 =========
def is_level(exam, level_text: str) -> bool:
    lv = dp_get_one(exam, "考试等级")
    return (lv == level_text) or (isinstance(lv, str) and level_text in lv)

def is_dianshi_exam(exam) -> bool:
    return is_level(exam, "殿试") or "Palace" in (getattr(exam, "name","") or "")

def all_persons() -> List[Any]:
    try: return list(PersonCls.instances())
    except: return []

def person_name(p) -> str:
    return dp_get_one(p, "姓名") or getattr(p, "name", "UnknownPerson")

def get_participations(p) -> List[Any]:
    return list(op_get_all(p, "participatesIn"))

def participation_exam(evt) -> Optional[Any]:
    exs = op_get_all(evt, "hasExam")
    return exs[0] if exs else None

# ========= 逐事件解析完整度 =========
def evaluate_time_completeness(evt, exam, era_hint: Optional[str]=None) -> Dict[str, Any]:
    raw_str = pick_best_time_text(exam)
    parsed  = parse_time_mixed(raw_str)
    status, years, extras = infer_years_from_parsed(parsed)
    era_hint_used = None
    if status == "不确定" and parsed.get("gz") and not parsed.get("era") and era_hint:
        if era_hint in ERA2SPAN:
            s,e = ERA2SPAN[era_hint]
            cand = list(dict.fromkeys(y for y in years if s <= y <= e))
            if cand:
                years = cand; era_hint_used = era_hint
    gy_unique = None
    if status == "确定" and years: gy_unique = int(years[0]); level = "L3"
    elif status == "不确定" and years: level = "L1"
    else: level = "L0"
    sc = info_score_for_exam(exam)
    return {"evt": evt, "exam": exam, "raw": raw_str, "parsed": parsed,
            "status": status, "years": years, "extras": extras,
            "level": level, "gy_unique": gy_unique, "era_hint_used": era_hint_used,
            "info_score": sc}

# ========= 人级前置筛选（按层级） =========
def preselect_per_person_for_level(p, level_text: str) -> Dict[str, List[Dict[str, Any]]]:
    events = []
    for evt in get_participations(p):
        ex = participation_exam(evt)
        if ex and is_level(ex, level_text):
            events.append(evt)
    if not events:
        return {"FULL_KEEP":[], "DELETE":[], "EXPAND":[], "UNPARSE":[]}

    # 人级年号提示
    era_hints = Counter()
    for evt in events:
        ex = participation_exam(evt)
        raw = pick_best_time_text(ex)
        m = re.search(r"(洪武|建文|永乐|洪熙|宣德|正统|景泰|天顺|成化|弘治|正德|嘉靖|隆庆|万历|泰昌|天启|崇祯|弘光|隆武|绍武|永历)", normalize_text(str(raw)))
        if m: era_hints[m.group(1)] += 1
        for pa in prop_assertions_about(ex):
            if dp_get_one(pa,"prop") in ("考试时间_原文","年号"):
                txt = str(dp_get_one(pa,"value") or dp_get_one(pa,"value_norm") or "")
                m = re.search(r"(洪武|建文|永乐|洪熙|宣德|正统|景泰|天顺|成化|弘治|正德|嘉靖|隆庆|万历|泰昌|天启|崇祯|弘光|隆武|绍武|永历)", normalize_text(txt))
                if m: era_hints[m.group(1)] += 1
    era_hint = era_hints.most_common(1)[0][0] if era_hints else None

    items = [evaluate_time_completeness(evt, participation_exam(evt), era_hint=era_hint) for evt in events]
    full = [it for it in items if it["level"] in ("L3","L2")]
    if full:
        by_gy: Dict[int, List[Dict[str,Any]]] = defaultdict(list)
        for it in full:
            if it["gy_unique"] is not None:
                by_gy[int(it["gy_unique"])].append(it)
        full_keep = []
        for gy, bucket in by_gy.items():
            best = sorted(bucket, key=lambda x: x["info_score"], reverse=True)[0]
            full_keep.append(best)
        full_ids = set((it["evt"], it["exam"]) for it in full_keep)
        to_delete = [it for it in items if (it["evt"], it["exam"]) not in full_ids]
        return {"FULL_KEEP": full_keep, "DELETE": to_delete, "EXPAND": [], "UNPARSE": []}
    else:
        to_expand = [it for it in items if it["level"] == "L1"]
        unpars   = [it for it in items if it["level"] == "L0"]
        return {"FULL_KEEP": [], "DELETE": [], "EXPAND": to_expand, "UNPARSE": unpars}

# ========= 通用：按层级创建/复用考试 =========
def find_or_create_exam_by_year(level_text: str, year: int, template_exam: Any, provs: List[Any]) -> Any:
    for ex in ImperialExamCls.instances():
        if is_level(ex, level_text) and str(dp_get_one(ex, "考试时间")) == str(year):
            return ex
    ex_new = ImperialExamCls(f"ImperialExam_{level_text}_{year}")
    dp_set_single(ex_new, "考试等级", level_text)
    dp_set_single(ex_new, "考试时间", str(year))
    for pv in provs or []: op_add_unique(pv, "contains", ex_new)
    return ex_new

def set_event_status(evt, status_prop: str, status_val: str, provs: List[Any]):
    create_prop_assertion(evt, status_prop, status_val, status_val, provs)

# ========= 弱信息重挂 / 合并 / 删除 =========
def migrate_exam_edges(src_exam, dst_exam):
    if src_exam == dst_exam: return
    for evt in ParticipationEventCls.instances():
        if src_exam in op_get_all(evt, "hasExam"):
            lst = [x for x in op_get_all(evt, "hasExam") if x != src_exam]
            try: setattr(evt, "hasExam", lst)
            except: pass
            op_add_unique(evt, "hasExam", dst_exam)
    for pa in prop_assertions_about(src_exam):
        abouts = op_get_all(pa, "about")
        if src_exam in abouts:
            abouts = [dst_exam if x == src_exam else x for x in abouts]
            try: setattr(pa, "about", list(dict.fromkeys(abouts)))
            except: pass
    TP = onto.search_one(iri="*TextProvenance") or getattr(onto, "TextProvenance", None)
    if TP:
        for tp in TP.instances():
            cont = op_get_all(tp, "contains")
            changed = False
            if src_exam in cont and dst_exam not in cont:
                cont.append(dst_exam); changed = True
            if src_exam in cont:
                cont = [x for x in cont if x != src_exam]; changed = True
            if changed:
                try: setattr(tp, "contains", list(dict.fromkeys(cont)))
                except: pass

def safe_destroy(inst):
    try: destroy_entity(inst)
    except Exception as e: log.warning("删除失败：%s  (%s)", getattr(inst, "name", inst), e)

# ========= 清污：考试时间 =========
def sanitize_time_field():
    fixed, cleared = 0, 0
    for ex in ImperialExamCls.instances():
        raw = getattr(ex, "考试时间", None)
        if raw is None: continue
        if not isinstance(raw, list):
            if not isinstance(raw, (str, int, float, bool, datetime.datetime, datetime.date)):
                try: setattr(ex, "考试时间", str(raw)); fixed += 1
                except: pass
            continue
        val = raw; depth_guard = 0
        while isinstance(val, list) and val and depth_guard < 10:
            val = val[0]; depth_guard += 1
        if isinstance(val, list) or val is None:
            dp_clear(ex, "考试时间"); cleared += 1
        else:
            try:
                setattr(ex, "考试时间", str(val)); fixed += 1
            except:
                try:
                    dp_clear(ex, "考试时间"); setattr(ex, "考试时间", str(val)); fixed += 1
                except:
                    cleared += 1
    if fixed or cleared:
        log.info("[SANITIZE] 考试时间 修复=%d 清空=%d", fixed, cleared)

# ========= 殿试专属：未规范→规范覆盖合并 =========
def is_canonical_exam(exam) -> bool:
    if not is_dianshi_exam(exam): return False
    y = dp_get_one(exam, "考试时间")
    if isinstance(y, int) or (isinstance(y, str) and y.isdigit()): return True
    has_norm = has_era = has_gz = False
    for pa in prop_assertions_about(exam):
        prop = dp_get_one(pa, "prop")
        if prop == "考试时间_规范": has_norm = True
        elif prop == "年号": has_era = True
        elif prop == "干支": has_gz = True
    if has_norm: return True
    if has_era and has_gz: return True
    return False

def extract_era_gz_from_exam(exam) -> Tuple[Optional[str], Optional[str]]:
    era = gz = None
    for pa in prop_assertions_about(exam):
        prop = dp_get_one(pa, "prop")
        val  = dp_get_one(pa, "value_norm") or dp_get_one(pa, "value")
        if prop == "年号" and val: era = str(val)
        elif prop == "干支" and val: gz = str(val)
    if not era or not gz:
        best = pick_best_time_text(exam)
        if best:
            parsed = parse_time_mixed(best)
            if not era and parsed.get("era"): era = parsed.get("era")
            if not gz and parsed.get("gz"): gz = parsed.get("gz")
    y = dp_get_one(exam, "考试时间")
    if (not gz) and y and str(y).isdigit(): gz = gz_of_year(int(y))
    return (era, gz)

def canonical_targets_for_person(p, era_nc: Optional[str], gz_nc: Optional[str]) -> List[Any]:
    exams = set()
    for evt in get_participations(p):
        for ex in op_get_all(evt, "hasExam"):
            if is_dianshi_exam(ex): exams.add(ex)
    cands = []
    for ex in exams:
        if not is_canonical_exam(ex): continue
        era_c, gz_c = None, None
        for pa in prop_assertions_about(ex):
            prop = dp_get_one(pa, "prop")
            val  = dp_get_one(pa, "value_norm") or dp_get_one(pa, "value")
            if prop == "年号" and val: era_c = str(val)
            elif prop == "干支" and val: gz_c = str(val)
        y = dp_get_one(ex, "考试时间")
        if (not gz_c) and y and str(y).isdigit(): gz_c = gz_of_year(int(y))
        if (not era_c) and y and str(y).isdigit():
            yy = int(y)
            for nm,(s,e) in ERA2SPAN.items():
                if s <= yy <= e: era_c = nm; break
        match_score = 0
        if gz_nc and gz_c and gz_nc == gz_c: match_score += 1
        if era_nc and era_c and era_nc == era_c: match_score += 1
        if match_score > 0: cands.append((match_score, ex))
    cands.sort(key=lambda t: (t[0], info_score_for_exam(t[1])), reverse=True)
    return [ex for _, ex in cands]

def postprocess_merge_uncanonical_exams():
    affected = 0
    for p in all_persons():
        exams_all = set()
        for evt in get_participations(p):
            for ex in op_get_all(evt, "hasExam"):
                if is_dianshi_exam(ex): exams_all.add(ex)
        uncanon = [ex for ex in exams_all if not is_canonical_exam(ex)]
        if not uncanon: continue
        for ex_nc in uncanon:
            era_nc, gz_nc = extract_era_gz_from_exam(ex_nc)
            if not era_nc and not gz_nc: continue
            targets = canonical_targets_for_person(p, era_nc, gz_nc)
            if not targets: continue
            if len(targets) > 1 and era_nc and gz_nc:
                strict = []
                for ex in targets:
                    era_c, gz_c = extract_era_gz_from_exam(ex)
                    if era_c == era_nc and gz_c == gz_nc: strict.append(ex)
                if len(strict) == 1: targets = strict
            if len(targets) != 1: continue
            dst = targets[0]
            migrate_exam_edges(ex_nc, dst)
            safe_destroy(ex_nc)
            affected += 1
    if affected:
        log.info("[CLEANUP DONE] 未规范殿试合并删除 %d 个", affected)

# ========= 主处理（按层级） =========
def process_all_for_level(level_text: str, status_prop: str):
    persons = all_persons()
    log.info("[START] %s 规范化：人物=%d", level_text, len(persons))
    person_to_years确定: Dict[Any, Set[int]] = defaultdict(set)

    for idx, p in enumerate(persons, 1):
        pname = person_name(p)
        plan = preselect_per_person_for_level(p, level_text)
        log.info("  (%s) [%d/%d] %s | KEEP=%d DEL=%d EXP=%d UNP=%d",
                 level_text, idx, len(persons), pname,
                 len(plan["FULL_KEEP"]), len(plan["DELETE"]), len(plan["EXPAND"]), len(plan["UNPARSE"]))

        # 建同年最佳考试索引
        gy_to_best_exam: Dict[int, Any] = {}
        for it in plan["FULL_KEEP"]:
            gy = it.get("gy_unique")
            if gy is not None: gy_to_best_exam[int(gy)] = it["exam"]

        # 弱信息事件：保留并尽量重挂到同年的最佳考试
        for it in plan["DELETE"]:
            evt, ex = it["evt"], it["exam"]
            gy = None
            try:
                if it.get("status") == "确定":
                    cand_years = it.get("years") or []
                    gy = int(cand_years[0]) if cand_years else it.get("gy_unique")
                if gy is None:
                    y_raw = dp_get_one(ex, "考试时间")
                    if y_raw and str(y_raw).isdigit(): gy = int(y_raw)
            except Exception: gy = None

            if gy is not None and gy in gy_to_best_exam:
                best_exam = gy_to_best_exam[gy]
                if best_exam != ex:
                    cur_exams = [e for e in op_get_all(evt, "hasExam") if e != ex]
                    try: setattr(evt, "hasExam", cur_exams)
                    except: pass
                    op_add_unique(evt, "hasExam", best_exam)
                    for pa in prop_assertions_about(ex):
                        if dp_get_one(pa, "prop") in ("考试时间","考试时间_原文","考试时间_规范","年号","序数","干支","时间解析说明"):
                            for tp in op_get_all(pa, "derivedFrom"):
                                op_add_unique(tp, "contains", best_exam)

        # FULL_KEEP：写回
        for it in plan["FULL_KEEP"]:
            evt, exam = it["evt"], it["exam"]
            gy = it["gy_unique"]
            provs = provenances_for_exam_time(exam)

            parsed, extras = it["parsed"], it["extras"]
            raw_txt = parsed.get("raw") or dp_get_one(exam, "考试时间") or ""
            if raw_txt: create_prop_assertion(exam, "考试时间_原文", raw_txt, raw_txt, provs)

            dp_set_single(exam, "考试时间", str(gy))
            create_prop_assertion(exam, "考试时间_规范", gy, gy, provs)

            dbg = parsed.get("debug", {}); note = []
            if extras.get("era") and dbg.get("era_from"): note.append(f"年号源自片段：{dbg['era_from']} -> {extras['era']}")
            if extras.get("gz") and dbg.get("gz_from"):   note.append(f"干支纠错：{dbg['gz_from']} -> {extras['gz']} (conf={dbg.get('gz_confidence')})")
            if note: create_prop_assertion(exam, "时间解析说明", "；".join(note), "；".join(note), provs)

            era = extras.get("era")
            if not era:
                for nm,(s,e) in ERA2SPAN.items():
                    if s <= gy <= e: era = nm; break
            if era:
                create_prop_assertion(exam, "年号", era, era, provs)
                s,_ = ERA2SPAN.get(era, (None,None))
                if s is not None:
                    create_prop_assertion(exam, "序数", parsed.get("ord_raw") or (gy - s + 1), (gy - s + 1), provs)
            create_prop_assertion(exam, "干支", gz_of_year(gy), gz_of_year(gy), provs)

            set_event_status(evt, status_prop, "确定", provs)
            person_to_years确定[p].add(gy)

        # EXPAND：多候选
        for it in plan["EXPAND"]:
            evt, exam = it["evt"], it["exam"]
            parsed, extras, years = it["parsed"], it["extras"], it["years"]
            provs = provenances_for_exam_time(exam)
            raw_txt = parsed.get("raw") or dp_get_one(exam, "考试时间") or ""
            for y in years:
                ex2 = find_or_create_exam_by_year(level_text, int(y), exam, provs)
                op_add_unique(evt, "hasExam", ex2)
                dp_set_single(ex2, "考试等级", level_text)
                dp_set_single(ex2, "考试时间", str(y))
                if raw_txt: create_prop_assertion(ex2, "考试时间_原文", raw_txt, raw_txt, provs)
                create_prop_assertion(ex2, "考试时间_规范", y, y, provs)
                dbg = parsed.get("debug", {}); note = []
                if extras.get("era") and dbg.get("era_from"): note.append(f"年号源自片段：{dbg['era_from']} -> {extras['era']}")
                if extras.get("gz") and dbg.get("gz_from"):   note.append(f"干支纠错：{dbg['gz_from']} -> {extras['gz']} (conf={dbg.get('gz_confidence')})")
                if note: create_prop_assertion(ex2, "时间解析说明", "；".join(note), "；".join(note), provs)
                era = extras.get("era")
                if not era:
                    for nm,(s,e) in ERA2SPAN.items():
                        if s <= y <= e: era = nm; break
                if era:
                    create_prop_assertion(ex2, "年号", era, era, provs)
                    s,_ = ERA2SPAN.get(era, (None,None))
                    if s is not None:
                        create_prop_assertion(ex2, "序数", parsed.get("ord_raw") or (y - s + 1), (y - s + 1), provs)
                create_prop_assertion(ex2, "干支", gz_of_year(y), gz_of_year(y), provs)
            set_event_status(evt, status_prop, "不确定", provs)

        # 同一人多个“确定年”→ 冲突
        if len(person_to_years确定[p]) >= 2:
            years_set = set(person_to_years确定[p])
            for evt in get_participations(p):
                ex = participation_exam(evt)
                if not ex or not is_level(ex, level_text): continue
                y = dp_get_one(ex, "考试时间")
                y = int(y) if (y and str(y).isdigit()) else None
                if y in years_set:
                    set_event_status(evt, status_prop, "冲突", provenances_for_exam_time(ex))

    # 全局同年合并
    for y in list(range(1000, 2101)):
        while True:
            cur = [ex for ex in ImperialExamCls.instances()
                   if is_level(ex, level_text) and str(dp_get_one(ex, "考试时间")) == str(y)]
            if len(cur) <= 1: break
            cur_sorted = sorted(cur, key=lambda e: info_score_for_exam(e), reverse=True)
            survivor, victims = cur_sorted[0], cur_sorted[1:]
            for v in victims:
                migrate_exam_edges(v, survivor)
                safe_destroy(v)

# ========= 后处理：按层级，场次>2 => 不确定 =========
def postprocess_status_by_count_for_level(level_text: str, status_prop: str):
    changed = 0
    for p in all_persons():
        exams = set()
        for evt in get_participations(p):
            for ex in op_get_all(evt, "hasExam"):
                if is_level(ex, level_text): exams.add(ex)
        if len(exams) > 2:
            for evt in get_participations(p):
                related = any(is_level(ex, level_text) for ex in op_get_all(evt, "hasExam"))
                if not related: continue
                # 清理旧状态
                for pa in list(prop_assertions_about(evt)):
                    if dp_get_one(pa, "prop") == status_prop:
                        try: destroy_entity(pa)
                        except: pass
                some_exam = next(iter(exams)) if exams else None
                provs = provenances_for_exam_time(some_exam) if some_exam else []
                set_event_status(evt, status_prop, "不确定", provs)
                changed += 1
    if changed:
        log.info("[POST] %s 状态降级完成：%d 条", level_text, changed)
    else:
        log.info("[POST] %s 状态降级：无需修改", level_text)

# ========= 导出 CSV =========
def export_per_person_csv_for_level(level_text: str, status_prop: str, out_csv: str):
    rows = []
    for p in all_persons():
        pname = person_name(p)
        for evt in get_participations(p):
            exs = op_get_all(evt, "hasExam")
            if not exs: continue
            status = ""
            for pa in prop_assertions_about(evt):
                if dp_get_one(pa, "prop") == status_prop:
                    status = dp_get_one(pa, "value_norm") or dp_get_one(pa, "value") or ""
                    break
            for ex in exs:
                if not is_level(ex, level_text): continue
                y  = dp_get_one(ex, "考试时间")
                era = ord_norm = gz = ""
                for pa in prop_assertions_about(ex):
                    prop = dp_get_one(pa, "prop")
                    if prop == "年号":
                        era = dp_get_one(pa, "value_norm") or dp_get_one(pa, "value") or era
                    elif prop == "序数":
                        ord_norm = dp_get_one(pa, "value_norm") or dp_get_one(pa, "value") or ord_norm
                    elif prop == "干支":
                        gz = dp_get_one(pa, "value_norm") or dp_get_one(pa, "value") or gz
                provs = provenances_for_exam_time(ex)
                source = dp_get_one(provs[0], "Text_source") or dp_get_one(provs[0], "Text_Source") if provs else ""
                conf   = dp_get_one(provs[0], "record_confidence") if provs else ""
                rows.append({
                    "person_name": pname,
                    "exam_level": level_text,
                    "exam_status": status,
                    "gregorian_year": str(y or ""),
                    "era_name": str(era),
                    "era_ordinal": str(ord_norm),
                    "sexagenary": str(gz),
                    "exam_id": getattr(ex, "name", ""),
                    "participation_event_id": getattr(evt, "name", ""),
                    "provenance_source": str(source or ""),
                    "provenance_conf": str(conf or ""),
                })
    if not rows:
        log.warning("[%s] CSV 导出：没有可导出的行", level_text); return
    with open(out_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    log.info("[OK] [%s] 逐人 CSV：%s（%d 行）", level_text, out_csv, len(rows))

# ========= 保存 =========
def save_all():
    try:
        onto.save(file=OUT_OWL, format="rdfxml")
        log.info("[OK] OWL 保存（RDF/XML）：%s", OUT_OWL)
    except Exception as e:
        log.warning("OWL 保存失败：%s", e)
    try:
        g = default_world.as_rdflib_graph()
        ttl = g.serialize(format="turtle")
        with open(OUT_TTL, "wb") as f:
            f.write(ttl if isinstance(ttl, (bytes, bytearray)) else ttl.encode("utf-8"))
        nt = g.serialize(format="nt")
        with open(OUT_NT, "wb") as f:
            f.write(nt if isinstance(nt, (bytes, bytearray)) else nt.encode("utf-8"))
        log.info("[OK] TTL/NT 导出：%s / %s", OUT_TTL, OUT_NT)
    except Exception as e:
        log.warning("TTL/NT 导出失败：%s", e)

# ========= 主入口 =========
if __name__ == "__main__":
    log.info("[START] 三科统一规范化流水线（成化修复 + 鲁棒加载 + 分科状态）")
    sanitize_time_field()

    # —— 殿试 ——
    process_all_for_level("殿试", "殿试状态")
    # 殿试未规范 → 规范覆盖 & 合并（仅殿试有此后处理）
    postprocess_merge_uncanonical_exams()
    postprocess_status_by_count_for_level("殿试", "殿试状态")
    export_per_person_csv_for_level("殿试", "殿试状态", CSV_DIAN)

    # —— 乡试 ——
    process_all_for_level("乡试", "乡试状态")
    postprocess_status_by_count_for_level("乡试", "乡试状态")
    export_per_person_csv_for_level("乡试", "乡试状态", CSV_XIANG)

    # —— 会试 ——
    process_all_for_level("会试", "会试状态")
    postprocess_status_by_count_for_level("会试", "会试状态")
    export_per_person_csv_for_level("会试", "会试状态", CSV_HUI)

    save_all()
    log.info("[DONE] 全部完成")
