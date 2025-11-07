# -*- coding: utf-8 -*-
import re
import json
from typing import Dict, Any, Tuple, List, Optional
import pandas as pd
import difflib

# =========================
# 写死的 Excel 路径
# =========================
CIVIL_XLSX   = r"C:\Users\卢航青\Desktop\民政系统.xlsx"
MIL_XLSX     = r"C:\Users\卢航青\Desktop\军事系统.xlsx"
TUSI_XLSX    = r"C:\Users\卢航青\Desktop\土司系统.xlsx"

CIVIL_SHEET  = None
MIL_SHEET    = None
TUSI_SHEET   = None

# 相似度阈值（OPS）
SIM_THRESHOLD = 0.80

# =========================
# 规范化 & 词表
# =========================
PROVINCE_HINTS = [
    "直隶","湖广","江西","浙江","四川","贵州","云南","山东","山西",
    "河南","陕西","两广","福建","广东","广西","江南","江北"
]
NORMALIZE_MAP = {
    "布政司": "布政使司",
    "行省": "布政使司",
    "都轉司": "都司",
    "宣撫司": "宣抚司",
    "安撫司": "安抚司",
    "長官司": "长官司",
    "鄉": "乡"
}
FORBID_ZHOU_TOKENS = {p for p in PROVINCE_HINTS if p.endswith("州")}

SUFFIXES = [
    "行都指挥使司","都指挥使司","布政使司","军民府","直隶州",
    "宣慰司","宣抚司","安抚司","长官司",
    "千户所","百户所","都司",
    "府","州","县","厅","卫","司","所"
]
CIVIL_SUFFIXES     = {"直隶州","府","州","县","厅","军民府"}
MILITARY_SUFFIXES  = {"行都指挥使司","都指挥使司","都司","卫","千户所","百户所"}
TUSI_SUFFIXES      = {"宣慰司","宣抚司","安抚司","长官司"}

# 后缀→系统锁定（包含“长官司”=土司，但查询时不带“长官司”）
SUFFIX_2_SYS: Dict[str, str] = {}
for s in CIVIL_SUFFIXES:
    SUFFIX_2_SYS[s] = "民政"
for s in MILITARY_SUFFIXES:
    SUFFIX_2_SYS[s] = "军事卫所"
for s in TUSI_SUFFIXES:
    SUFFIX_2_SYS[s] = "土司"

# =========================
# 文本清洗
# =========================
def normalize_text(s: str) -> str:
    if s is None: return ""
    if isinstance(s, float) and pd.isna(s): return ""
    def to_half_width(text: str) -> str:
        out = []
        for ch in str(text):
            code = ord(ch)
            if code == 0x3000: code = 32
            elif 0xFF01 <= code <= 0xFF5E: code -= 0xFEE0
            out.append(chr(code))
        return "".join(out)
    s = to_half_width(str(s).strip())
    s = re.sub(r"\s+", "", s)
    for k, v in NORMALIZE_MAP.items():
        s = s.replace(k, v)
    s = re.sub(r"[|｜／/、，,。．\.；;：:·・]+", "", s)
    return s

def startswith_province(text: str) -> Tuple[str, str]:
    for p in PROVINCE_HINTS:
        if text.startswith(p):
            return p, text[len(p):]
    return "", text

def strip_province_prefix(name: str) -> str:
    for p in PROVINCE_HINTS:
        if name.startswith(p):
            return name[len(p):]
    return name

# =========================
# A) 层层剥离：省 → 若干 token(基底+后缀) → 尾部无后缀核心
# =========================
SUF_ALT = "|".join(map(re.escape, SUFFIXES))
TOKEN_RE = re.compile(rf"^([一-龥A-Za-z0-9]+?)({SUF_ALT})")

def peel_hierarchy(raw_text: str) -> Dict[str, Any]:
    """
    返回：
      {
        "province": 省名或"",
        "tokens": [{"base":基底,"suffix":后缀,"full":基底+后缀}, ...]  # 按出现顺序
        "tail_core": 尾部无后缀核心或"",
        "core_candidates": 从更具体到更泛的核心候选（保序去重）
      }
    """
    s = normalize_text(raw_text)
    province, rem = startswith_province(s)

    tokens: List[Dict[str, str]] = []
    safety = 0
    while rem and safety < 300:
        safety += 1
        m = TOKEN_RE.match(rem)
        if not m:
            break
        base_raw, suf = m.group(1), m.group(2)

        # 关键改动：仅当后缀不是“长官司”时，才剥掉省名前缀；“长官司”保留省名
        if suf == "长官司":
            base = base_raw
        else:
            base = strip_province_prefix(base_raw)

        tokens.append({"base": base, "suffix": suf, "full": base + suf})
        rem = rem[m.end():]

    tail_core = ""
    if rem and re.fullmatch(r"[一-龥A-Za-z0-9]+", rem):
        tail_core = strip_province_prefix(rem)

    # 候选核心：尾部核心 > 最后一个token基底 > 之前token基底（逆序）
    core_candidates: List[str] = []
    seen = set()
    def _push(x: str):
        nx = normalize_text(x)
        if nx and nx not in seen:
            core_candidates.append(x); seen.add(nx)

    if tail_core: _push(tail_core)
    for t in reversed(tokens):
        _push(t["base"])

    return {
        "province": province,
        "tokens": tokens,
        "tail_core": tail_core,
        "core_candidates": core_candidates
    }

# =========================
# 2) Excel 索引构建（通用，含别名括注）
# =========================
def _ensure_df(obj):
    if isinstance(obj, dict):
        if obj:
            key = sorted(obj.keys())[0]
            return obj[key]
        raise ValueError("Excel 工作表为空。")
    return obj

def _extract_aliases(cell: str) -> Tuple[str, List[str]]:
    s = normalize_text(cell)
    if not s: return "", []
    m = re.match(r"^([一-龥A-Za-z0-9]+?)(（.+）)$", s)
    if not m:
        return s, []
    main = m.group(1)
    note = re.sub(r"[（）]", "", m.group(2))
    note = re.sub(r"^(曾用名|别名)(：)?", "", note)
    note = re.sub(r"[、,，/／;； ]+", "|", note)
    aliases = [a for a in note.split("|") if a]
    return main, aliases

def build_generic_index(excel_path: str, sheet_name=None):
    read_obj = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=str)
    df = _ensure_df(read_obj).fillna("")
    df.columns = [("" if c is None else str(c)).strip() for c in df.columns]
    cols_raw = list(df.columns)
    cols = {c.strip(): c for c in cols_raw if c is not None}

    def pick_col(tag: str) -> Optional[str]:
        for k, v in cols.items():
            if tag in k: return v
        return None

    c1 = pick_col("（一级）")
    c2 = pick_col("（二级）")
    c3 = pick_col("（三级）")
    c4 = pick_col("（四级）")

    def fuzzy_pick(keys: List[str]) -> Optional[str]:
        for k, v in cols.items():
            if any(kw in k for kw in keys): return v
        return None

    if not c1: c1 = fuzzy_pick(["承宣布政使司","布政使司","省","一级","上级"])
    if not c2: c2 = fuzzy_pick(["府","军民府","直隶州","二级","都司","卫","宣慰司","长官司"])
    if not c3: c3 = fuzzy_pick(["州","所","三级","属"])
    if not c4: c4 = fuzzy_pick(["县","四级","属"])

    level_cols = [c1 or "", c2 or "", c3 or "", c4 or ""]
    if not level_cols[0] or not level_cols[1]:
        raise ValueError(f"无法识别层级列：{cols_raw}")

    idx_lv2: Dict[str, Dict[str, str]] = {}
    idx_lv3: Dict[str, Dict[str, str]] = {}
    idx_lv4: Dict[str, Dict[str, str]] = {}

    for _, row in df.iterrows():
        l1 = normalize_text(row.get(level_cols[0], ""))
        l2_raw = normalize_text(row.get(level_cols[1], ""))
        l3 = normalize_text(row.get(level_cols[2], "")) if level_cols[2] else ""
        l4 = normalize_text(row.get(level_cols[3], "")) if level_cols[3] else ""
        if not l1 or not l2_raw: continue

        l2_main, l2_aliases = _extract_aliases(l2_raw)
        if not l2_main: continue
        names_lv2 = {l2_main} | set(l2_aliases)

        for nm in names_lv2:
            idx_lv2[nm] = {"L1": l1, "L2": l2_main}
        if l3:
            idx_lv3[l3] = {"L1": l1, "L2": l2_main, "L3": l3}
        if l4:
            idx_lv4[l4] = {"L1": l1, "L2": l2_main, "L3": l3, "L4": l4}

    return {"levels": level_cols, "idx_lv2": idx_lv2, "idx_lv3": idx_lv3, "idx_lv4": idx_lv4}

# =========================
# 3) OPS + 核心约束
# =========================
_SUFFIX_RX = re.compile(r"(布政使司|军民府|直隶州|都司|卫|府|州|县|所|宣慰司|宣抚司|安抚司|长官司|厅)$")

def _suffix(name: str) -> str:
    s = normalize_text(name)
    m = _SUFFIX_RX.search(s)
    return m.group(1) if m else ""

def _ops_span(query: str, target: str) -> Optional[Tuple[int, int]]:
    q = list(normalize_text(query))
    t = list(normalize_text(target))
    if not q or not t: return None
    pos = []; ti = 0
    for ch in q:
        found = False
        while ti < len(t):
            if t[ti] == ch:
                pos.append(ti); ti += 1; found = True; break
            ti += 1
        if not found: return None
    return (pos[0], pos[-1])

def _ops_score(query: str, target: str) -> float:
    window = _ops_span(query, target)
    if window is None: return -1.0
    start, end = window
    span_len = end - start + 1
    qlen = len(normalize_text(query))
    tlen = len(normalize_text(target))
    redundancy = max(0, span_len - qlen)
    base = 1.0 - redundancy / max(tlen, 1)
    if _suffix(query) and _suffix(query) == _suffix(target):
        base += 0.05
    if start == 0:
        base += 0.05
    return min(base, 1.0)

_CORE_HIT_RX_TPL = r"({core})(布政使司|军民府|直隶州|都司|卫|府|州|县|厅|司|所|长官司)?"

def _contains_core(key: str, core: str) -> bool:
    if not core:
        return True
    s = normalize_text(key); c = normalize_text(core)
    if not c: return True
    if c in s: return True
    pat = re.compile(_CORE_HIT_RX_TPL.replace("{core}", re.escape(c)))
    return pat.search(s) is not None

def _best_match_ops_with_score(name: str,
                               pool: Dict[str, Dict[str, str]],
                               required_core: Optional[str] = None) -> Tuple[Optional[str], float]:
    if not pool: return (None, -1.0)
    name_n = normalize_text(name)

    # 先在“包含核心”的子集中找
    if required_core:
        cands = {k:v for k,v in pool.items() if _contains_core(k, required_core)}
        if not cands:
            cands = pool
    else:
        cands = pool

    # OPS → difflib
    best_k, best_s = None, -1.0
    for k in cands.keys():
        s = _ops_score(name_n, k)
        if s > best_s:
            best_k, best_s = k, s
    if best_s < 0:
        for k in cands.keys():
            s = difflib.SequenceMatcher(None, name_n, normalize_text(k)).ratio()
            if s > best_s:
                best_k, best_s = k, s
    return best_k, best_s

def _build_chain(rec: dict) -> str:
    chain = [rec.get("L1",""), rec.get("L2","")]
    if rec.get("L3"): chain.append(rec["L3"])
    if rec.get("L4"): chain.append(rec["L4"])
    return "".join(chain)

def compose_canonical_from_idx_ops(unit: str, idx: dict, threshold: float = SIM_THRESHOLD,
                                   required_core: Optional[str] = None) -> Tuple[str, bool, float]:
    """
    (规范文本, matched, score)
    - 若 required_core 存在，胜出项需包含核心，否则视为 unmatched。
    """
    name = normalize_text(unit)
    if not name:
        return unit, False, 0.0

    # 精确命中
    for lv in ("idx_lv4","idx_lv3","idx_lv2"):
        if name in idx[lv]:
            r = idx[lv][name]; val = _build_chain(r)
            if required_core and not _contains_core(val, required_core):
                return unit, False, 0.0
            return val, True, 1.0

    # 模糊
    best_val, best_score = None, -1.0
    for level in ("idx_lv4","idx_lv3","idx_lv2"):
        best_key, score = _best_match_ops_with_score(name, idx[level], required_core=required_core)
        if best_key is not None and score > best_score:
            r = idx[level][best_key]
            best_val, best_score = _build_chain(r), score

    if best_val is not None and best_score >= threshold:
        if required_core and not _contains_core(best_val, required_core):
            return unit, False, best_score
        return best_val, True, best_score
    return unit, False, (best_score if best_score >= 0 else 0.0)

# =============== 无后缀兜底（跨三表） ===============
def fallback_search_all_systems(raw_text: str, civil_idx, mil_idx, tusi_idx,
                                threshold=SIM_THRESHOLD, required_core: Optional[str]=None):
    q = normalize_text(raw_text)
    civ_val, _, civ_sc = compose_canonical_from_idx_ops(q, civil_idx, threshold=0.0, required_core=required_core)
    mil_val, _, mil_sc = compose_canonical_from_idx_ops(q, mil_idx,   threshold=0.0, required_core=required_core)
    tusi_val, _, tusi_sc = compose_canonical_from_idx_ops(q, tusi_idx, threshold=0.0, required_core=required_core)

    scores = {"民政": civ_sc, "军事卫所": mil_sc, "土司": tusi_sc}
    vals   = {"民政": civ_val, "军事卫所": mil_val, "土司": tusi_val}

    max_sc = max(scores.values()); eps = 1e-9
    winners = [sys for sys, sc in scores.items() if abs(sc - max_sc) <= eps]

    out = {"民政": [], "军事卫所": [], "土司": []}
    for sys in winners:
        v = vals[sys]
        if max_sc >= threshold and (not required_core or _contains_core(v, required_core)):
            out[sys].append({"value": v, "matched": True, "tie": len(winners) > 1})
        else:
            out[sys].append({"value": q if not required_core else required_core,
                             "matched": False, "tie": len(winners) > 1})
    return out

# =========================
# 4) 新逻辑：有“行政后缀”→ 直接锁定系统匹配；无后缀→ 再做模糊匹配
#    特别规则：『长官司』锁定土司，但查询仅用 base（不带“长官司”）
# =========================
def normalize_all_systems(raw_text: str, _unused_min_units: Dict[str, List[str]],
                          civil_idx, mil_idx, tusi_idx) -> Dict[str, Any]:
    peel = peel_hierarchy(raw_text)
    province = peel["province"]
    tokens   = peel["tokens"]
    cores    = peel["core_candidates"]
    main_core = cores[0] if cores else ""

    sys2idx = {"民政": civil_idx, "军事卫所": mil_idx, "土司": tusi_idx}
    out = {"民政": [], "军事卫所": [], "土司": []}

    # 1) 先处理“带后缀”的最小单位 → 直接锁定系统
    for t in tokens:
        suf = t["suffix"]
        sys_lock = SUFFIX_2_SYS.get(suf, None)
        if not sys_lock:
            continue
        idx = sys2idx[sys_lock]

        # 关键改动：
        # - 长官司：只用 base 查询（且我们在 peel 时保留了省名，如“贵州长官司”→ base="贵州"）
        # - 其他后缀：仍用 full（base+suffix）查询
        query = t["base"] if suf == "长官司" else t["full"]

        # 核心约束：优先用 token.base；若无则用 main_core
        required = (t["base"] or main_core) or None

        val, ok, _ = compose_canonical_from_idx_ops(query, idx, SIM_THRESHOLD, required_core=required)
        if ok:
            out[sys_lock].append({"value": val, "matched": True})

    # 2) 若目前还没有任何命中，再处理“无后缀核心”→ 跨系统模糊匹配
    if all(len(v) == 0 for v in out.values()):
        query_core = main_core or (tokens[-1]["base"] if tokens else "") or raw_text
        fb = fallback_search_all_systems(query_core, civil_idx, mil_idx, tusi_idx,
                                         threshold=SIM_THRESHOLD, required_core=main_core or None)
        out = fb

    # 3) 省级泛项降权（有省+核心时，过滤“贵州卫/贵州都司”这类泛匹配）
    if province and (main_core or (tokens and tokens[-1]["base"])):
        bad_set = {f"{province}卫", f"{province}都司"}
        def _post_clean(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            new = []
            for it in items:
                v = normalize_text(it.get("value",""))
                if it.get("matched") and v in bad_set:
                    new.append({"value": main_core or tokens[-1]["base"], "matched": False, "tie": it.get("tie", False)})
                else:
                    new.append(it)
            return new
        out["军事卫所"] = _post_clean(out["军事卫所"])

    # 同值去重（保序）
    def dedup(lst: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        seenv=set(); r=[]
        for it in lst:
            nv = normalize_text(it["value"])
            if nv and nv not in seenv:
                r.append(it); seenv.add(nv)
        return r

    out["民政"] = dedup(out["民政"])
    out["军事卫所"] = dedup(out["军事卫所"])
    out["土司"] = dedup(out["土司"])

    return out

# =========================
# 5) REPL 调试端口
# =========================
def main():
    try:
        civil_idx = build_generic_index(CIVIL_XLSX, sheet_name=CIVIL_SHEET)
        mil_idx   = build_generic_index(MIL_XLSX,   sheet_name=MIL_SHEET)
        tusi_idx  = build_generic_index(TUSI_XLSX,  sheet_name=TUSI_SHEET)
    except Exception as e:
        print(f"[错误] 读取/索引 Excel 失败：{e}")
        return

    print("=== 剥离最小单位 → 有后缀锁定系统；无后缀再模糊 → OPS+阈值+核心约束 ===")
    print("（特别规则：『长官司』锁定“土司”，但匹配时去掉‘长官司’，仅用 base 检索；剥离时保留省名基底）")
    print(f"阈值: {SIM_THRESHOLD:.2f}")
    print(f"[民政] {CIVIL_XLSX}\n[军事] {MIL_XLSX}\n[土司] {TUSI_XLSX}")
    print("输入文本；:quit 退出。\n")

    while True:
        try:
            s = input("place> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。"); break
        if not s:
            continue
        if s.lower() in {":quit", ":exit"}:
            print("退出。"); break

        peel = peel_hierarchy(s)
        min_units = {
            "民政": [t["full"] for t in peel["tokens"] if t["suffix"] in CIVIL_SUFFIXES],
            "军事卫所": [t["full"] for t in peel["tokens"] if t["suffix"] in MILITARY_SUFFIXES or t["suffix"]=="卫"],
            "土司": [t["full"] for t in peel["tokens"] if t["suffix"] in TUSI_SUFFIXES],
        }
        canon = normalize_all_systems(s, min_units, civil_idx, mil_idx, tusi_idx)

        print(json.dumps({
            "原文": s,
            "抽取": {
                "省": peel["province"],
                "token序列": peel["tokens"],
                "尾部核心": peel["tail_core"],
                "核心候选": peel["core_candidates"]
            },
            "最小单位": min_units,
            "最规范表达": canon
        }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
