# -*- coding: utf-8 -*-
"""
strict_writeback_merge_authoritative_palace_and_provincial.pyjiaob

- 殿试（共享安全 + 同年合并 + 后处理）
- 乡试筛选（以殿试权威年为锚，选“早于殿试且最近”的乡试为权威；差值上限 16 年）
- 年份解析严格化：必须出现「进士/進士/进土/登科」关键词，且只接受同处的「(明)?年号 + (元/正/数) 年」
- OCR/异写容错：進士/进土、題名碑錄/题名碑录、登科錄/登科录 等；清洗常见噪音字符
"""

import re, tempfile, uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional, Any

from rdflib import Graph
from owlready2 import World, destroy_entity

# ===== 路径（按需改） =====
# ===== 路径（外部传入，兼容脚本5产物） =====
import os
import argparse
from pathlib import Path



import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
else:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")



def _parse_args():
    ap = argparse.ArgumentParser(description="脚本6：基于文献锁定的后处理（外部配置）")
    # 支持 --src（首选）与 --onto（兼容上游习惯），都映射到 dest="src"
    ap.add_argument(
        "--src", "--onto", dest="src", required=False,
        default=os.getenv(
            "ONTO_FILE",  # 若上游用 ONTO_FILE 传入则优先
            os.path.join(os.getenv("OUT_DIR", str(Path.cwd() / "本体结构")), "ontology_dedup_乡会殿处理.nt")
        ),
        help="输入 NT 文件（通常为脚本5输出的 ontology_dedup_乡会殿处理.nt）"
    )
    ap.add_argument(
        "--out-dir", dest="out_dir", required=False,
        default=os.getenv("OUT_DIR", str(Path.cwd() / "本体结构")),
        help="输出目录（OUT_DIR）"
    )
    ap.add_argument(
        "--max-prov-gap", dest="max_prov_gap", type=int, required=False,
        default=int(os.getenv("MAX_PROV_GAP", "16")),
        help="乡试距离殿试最大允许差值（年），默认 16"
    )
    return ap.parse_args()

_args = _parse_args()
SRC = _args.src
OUT_DIR = _args.out_dir
MAX_PROV_GAP = _args.max_prov_gap

Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

OUT_OWL = str(Path(OUT_DIR) / "ontology_dedup_updated.owl")
OUT_TTL = str(Path(OUT_DIR) / "ontology_dedup_updated.ttl")
OUT_NT  = str(Path(OUT_DIR) / "ontology_dedup_经过文献锁定处理.nt")

print(f"[CFG] SRC={SRC}")
print(f"[CFG] OUT_DIR={OUT_DIR}")
print(f"[CFG] MAX_PROV_GAP={MAX_PROV_GAP}")
print(f"[CFG] OUT_OWL={OUT_OWL}")
print(f"[CFG] OUT_TTL={OUT_TTL}")
print(f"[CFG] OUT_NT={OUT_NT}")

# ===== 简繁转换 + OCR 噪音清洗 =====
# 优先用 OpenCC；若不可用则退化为字典替换
try:
    from opencc import OpenCC
    _CC_T2S = OpenCC("t2s")
    def _cc_t2s(s: str) -> str:
        return _CC_T2S.convert(s or "")
except Exception:
    _cc_t2s = None

# 统一/纠错映射（含繁简、常见异写、OCR混淆）
_BASE_MAP = {
    "進":"进","錄":"录","題":"题","鄉":"乡","國":"国","萬":"万","祿":"禄",
    "歷":"历","曆":"历","統":"统","順":"顺","啟":"启","禎":"祯","樂":"乐","慶":"庆",
    "龍":"隆","歷":"历","歲":"岁","時":"时","間":"间","紀":"纪",
    "國朝歷科題名碑錄":"国朝历科题名碑录","題名碑錄":"题名碑录","登科錄":"登科录","進士登科錄":"进士登科录",
    # 年号别名
    "萬曆":"万历","萬歷":"万历","万曆":"万历","万厯":"万历","正統":"正统","天順":"天顺","天啟":"天启",
    "崇禎":"崇祯","永樂":"永乐","隆慶":"隆庆","龍慶":"隆庆","永歷":"永历",
    # 关键词容错（OCR）
    "進土":"进士","進士":"进士",
}

# 常见 OCR 噪音/装饰字符：零宽/中点/破折/罕见标点等
_OCR_NOISE_PATTERN = re.compile(
    r"[\u200b\u200c\u200d\uFEFF\u2060\u00AD"          # 零宽/软连字符
    r"\u2027\u2219\u22C5\u00B7\u30FB\u2022"           # 各种中点/圆点
    r"\uFE30-\uFE6F"                                  # CJK 标点宽度变体
    r"\uFFF0-\uFFFF"                                  # 特殊占位
    r"\u2010-\u2015\u2E3A\u2E3B\u2500\u2501"          # 连字符/破折/水平线
    r"\u3000"                                         # 全角空格
    r"\u2028\u2029"                                   # 分隔符
    r"]"
)

def _strip_ocr_noise(s: str) -> str:
    s = _OCR_NOISE_PATTERN.sub("", s or "")
    # 常见拉丁/全角标点混排统一去除（保留“年/科”等关键字）
    s = re.sub(r"[《》〈〉【】\[\]（）\(\)·・•—\-:：,，．。/／\s]+", "", s)
    return s

def t2s(s: str) -> str:
    s = s or ""
    if _cc_t2s:
        s = _cc_t2s(s)
    for a,b in _BASE_MAP.items():
        s = s.replace(a,b)
    # 二次兜底：把单独出现的“进土”也替换为“进士”
    s = s.replace("进土", "进士")
    return s

# ===== 载入（rdflib -> RDF/XML -> owlready2）=====
def load_ontology_robust(src_path: str):
    p = Path(src_path)
    if not p.exists(): raise FileNotFoundError(src_path)
    fmt = {".nt":"nt",".ttl":"turtle",".rdf":"xml",".owl":"xml",".xml":"xml"}.get(p.suffix.lower(),"nt")
    g = Graph(); g.parse(str(p), format=fmt)
    td = tempfile.TemporaryDirectory()
    owl_path = Path(td.name)/(p.stem+"_tmp.owl")
    g.serialize(destination=str(owl_path), format="xml", encoding="utf-8")
    world = World(); onto = world.get_ontology(str(owl_path)).load()
    return world, onto, td
world, onto, _tmpdir = load_ontology_robust(SRC)

# ===== 取类/属性 =====
def _ends(iri: str, names: List[str]) -> bool:
    return any((iri or "").endswith("#"+n) or (iri or "").endswith("/"+n) for n in names)
def must_get_class(names):
    for c in list(onto.classes()):
        if c.name in names or _ends(getattr(c,"iri",""), names): return c
    for c in list(world.classes()):
        if c.name in names or _ends(getattr(c,"iri",""), names): return c
    raise RuntimeError(f"no class {names}")
def must_get_objprop(names):
    for p in list(onto.object_properties()):
        if p.name in names or _ends(getattr(p,"iri",""), names): return p
    for p in list(world.object_properties()):
        if p.name in names or _ends(getattr(p,"iri",""), names): return p
    raise RuntimeError(f"no objprop {names}")
def must_get_dataprop(names):
    for p in list(onto.data_properties()):
        if p.name in names or _ends(getattr(p,"iri",""), names): return p
    for p in list(world.data_properties()):
        if p.name in names or _ends(getattr(p,"iri",""), names): return p
    raise RuntimeError(f"no dataprop {names}")

Person             = must_get_class(["Person","人物"])
ParticipationEvent = must_get_class(["ParticipationEvent","参与事件"])
ImperialExam       = must_get_class(["ImperialExam","科举考试","考试"])
PropAssertion      = must_get_class(["PropAssertion"])
TextProvenance     = must_get_class(["TextProvenance"])

participatesIn = must_get_objprop(["participatesIn"])
hasExam        = must_get_objprop(["hasExam"])
about          = must_get_objprop(["about"])
derivedFrom    = must_get_objprop(["derivedFrom"])
contains       = must_get_objprop(["contains"])

dp_exam_level  = must_get_dataprop(["考试等级"])
dp_exam_time   = must_get_dataprop(["考试时间"])
dp_pa_prop     = must_get_dataprop(["prop"])
dp_pa_value    = must_get_dataprop(["value"])
dp_pa_valnorm  = must_get_dataprop(["value_norm"])
dp_tp_source   = must_get_dataprop(["Text_source","Text_Source"])
dp_tp_body     = must_get_dataprop(["Text_body"])
dp_name        = must_get_dataprop(["姓名"])
dp_zhi         = must_get_dataprop(["字"])

# ======= 个体一致性工具 =======
def same_individual(a, b) -> bool:
    if a is b: return True
    ia = getattr(a, "iri", None); ib = getattr(b, "iri", None)
    if ia and ib and ia == ib: return True
    na = getattr(a, "name", None); nb = getattr(b, "name", None)
    return bool(na and nb and na == nb)

def list_has_individual(lst: List[Any], target: Any) -> bool:
    return any(same_individual(x, target) for x in lst)

def list_without_individual(lst: List[Any], target: Any) -> List[Any]:
    return [x for x in lst if not same_individual(x, target)]

def iri_of(x) -> str:
    return getattr(x,"iri", getattr(x,"name",""))

def key_of(x) -> str:
    return getattr(x, "iri", None) or getattr(x, "name", "")

# ===== 基础工具 =====
def op_list(inst, prop) -> List[Any]:
    try: return list(prop[inst])
    except Exception: return []
def op_add_unique(subj, prop, obj):
    cur = op_list(subj, prop)
    if list_has_individual(cur, obj): return
    try: prop[subj].append(obj)
    except Exception: pass
def dp_get_all(inst, dp) -> List[Any]:
    try: return list(dp[inst])
    except Exception: return []
def dp_get_one(inst, dp):
    vs = dp_get_all(inst, dp); return vs[0] if vs else None

def person_aliases(p) -> List[str]:
    out=[]
    for dp in (dp_name, dp_zhi):
        for v in dp_get_all(p, dp):
            s=str(v).strip()
            if s and s not in out: out.append(s)
    if not out:
        nm=getattr(p,"name","")
        if nm: out.append(nm)
    return out

def contains_name(text: str, names: List[str]) -> bool:
    t = t2s(text or "")
    for nm in names:
        if nm and nm in t: return True
    return False

# ===== 年号/干支/别名 =====
MING_ERAS = [
    ("洪武", 1368, 1398), ("建文", 1399, 1402), ("永乐", 1403, 1424), ("洪熙", 1425, 1425),
    ("宣德", 1426, 1435), ("正统", 1436, 1449), ("景泰", 1450, 1456), ("天顺", 1457, 1464),
    ("成化", 1465, 1487), ("弘治", 1488, 1505), ("正德", 1506, 1521), ("嘉靖", 1522, 1566),
    ("隆庆", 1567, 1572), ("万历", 1573, 1620), ("泰昌", 1620, 1620), ("天启", 1621, 1627),
    ("崇祯", 1628, 1644), ("弘光", 1645, 1645), ("隆武", 1645, 1646), ("绍武", 1647, 1647), ("永历", 1646, 1662),
]
ERA2SPAN = {n:(s,e) for (n,s,e) in MING_ERAS}
def era_of_year(y: int) -> Optional[Tuple[str,int]]:
    for name, s, e in MING_ERAS:
        if s <= y <= e:
            return name, s
    return None
STEMS="甲乙丙丁戊己庚辛壬癸"; BRANCHES="子丑寅卯辰巳午未申酉戌亥"
def ganzhi_of_year(y: int) -> str:
    off = (y - 1984) % 60
    return STEMS[off % 10] + BRANCHES[off % 12]
DIGITS  = "零一二三四五六七八九"
def int_to_cn_ordinal(n: int) -> str:
    if n<=0: return str(n)
    if n<=10: return "十" if n==10 else DIGITS[n]
    tens, ones = divmod(n,10)
    if n<20: return "十"+(DIGITS[ones] if ones else "")
    return (DIGITS[tens]+"十"+(DIGITS[ones] if ones else "")) if tens<10 else str(n)

ERA_ALIASES = {
    "萬曆":"万历","萬歷":"万历","万曆":"万历","万厯":"万历","正統":"正统","天順":"天顺","天啟":"天启",
    "崇禎":"崇祯","永樂":"永乐","隆慶":"隆庆","龍慶":"隆庆","永歷":"永历"
}

# —— 标题规范化与关键词 —— #
def _norm_bib_text(s: str) -> str:
    s = t2s(s or "").strip()
    # 统一一些系列名称与关键词
    s = s.replace("題名碑錄", "题名碑录").replace("國朝歷科題名碑錄", "国朝历科题名碑录")
    s = s.replace("登科錄", "登科录").replace("進士登科錄","进士登科录")
    # 关键词容错：進土/进土 一律视为 进士
    s = s.replace("進土","进士").replace("进土","进士").replace("進士","进士")
    return s

# 解析必须满足的特征词（含 OCR 变体）
REQUIRED_FEATURES = ("进士","登科","进土")  # “进土”保留在集合中，且在 _norm_bib_text 中已统一为“进士”

def normalize_title(s: str) -> str:
    s = t2s((s or "").strip())
    for a,c in ERA_ALIASES.items():
        s = s.replace(a,c)
    s = _norm_bib_text(s)
    # 去掉 OCR 噪音/装饰字符，保留“年/科”等关键结构字符
    s = _strip_ocr_noise(s)
    return s

def _has_required_features(s_norm: str) -> bool:
    # 标题或窗口中必须出现“进士/登科/进土”任一项；_norm_bib_text 已把进土→进士，但这里仍兼容
    return any(k in s_norm for k in REQUIRED_FEATURES)

def cn_ordinal_to_int(s: str) -> Optional[int]:
    """
    把“元/正/汉字数字/阿拉伯数字(+可带‘年’)”转为整数序数。
    例：'元年'→1, '正'→1, '十七'→17, '17'→17
    """
    s = (s or "").strip().replace("年","")
    if not s:
        return None
    # ★ 新增：直接支持 “元 / 正”
    if s in {"元", "正"}:
        return 1
    if re.fullmatch(r"\d{1,3}", s):
        return int(s)

    # 兼容 “拾” 与常见汉字数字写法
    s = s.replace("拾", "十")
    if "十" in s:
        a, _, b = s.partition("十")
        # “十”=10；“二十”=20；“二十三”=23；“十七”=17
        tens = 10 if a == "" else "零一二三四五六七八九".find(a) * 10
        if tens < 0:
            return None
        ones = 0 if b == "" else "零一二三四五六七八九".find(b)
        if ones < 0:
            return None
        return tens + ones

    # 纯汉字数字（不含“十”）：如 “三”“二三”
    val = 0
    for ch in s:
        d = "零一二三四五六七八九".find(ch)
        if d < 0:
            return None
        val = val * 10 + d
    return val


def extract_year_infos_from_title(title: str) -> List[Tuple[str,str,int]]:
    """
    严格解析：(年号, 原序词, 公元年)
    规则：
      1) 标题必须包含 “进士/登科/进土” 之一（规范化后）
      2) 只接受同处的「(明)?年号 + (元/正/数字/汉字数字) + 可选‘年’」
      3) 匹配片段附近 ±12 字符须出现 “进士/登科/进土”
      4) 放宽年号与序词之间的 OCR 噪音（0~3 个杂字符）
    """
    s = normalize_title(title)
    if not s or not _has_required_features(s):
        return []

    eras = [n for (n, _, __) in MING_ERAS]
    pat_era = "(?:" + "|".join(map(re.escape, eras)) + ")"

    out: List[Tuple[str, str, int]] = []
    # ★ 允许年号与序词之间夹少量 OCR 杂字符，且 “年” 可选
    rx = re.compile(
        rf"(?:明)?({pat_era})[^\u4e00-\u9fff0-9]{{0,3}}(元|正|[零一二三四五六七八九十拾\d]{{1,4}})\s*年?"
    )

    for m in rx.finditer(s):
        era, ord_raw = m.group(1), m.group(2)
        start, end = m.span()
        # 关键词邻近判断（±12 字符）
        l = max(0, start - 12); r = min(len(s), end + 12)
        if not _has_required_features(s[l:r]):
            continue

        k = cn_ordinal_to_int(ord_raw)   # ★ 现在 “元/正” 会得到 1
        if k is None:
            continue
        s_year, e_year = ERA2SPAN.get(era, (None, None))
        if s_year is None:
            continue
        y = s_year + (k - 1)
        if s_year <= y <= e_year:
            tup = (era, ord_raw, y)
            if tup not in out:
                out.append(tup)

    return out

# ===== 判定：殿试 / 乡试 =====
def is_palace_exam(exam) -> bool:
    for v in dp_get_all(exam, dp_exam_level):
        if "殿试" in t2s(str(v)): return True
    for pa in PropAssertion.instances():
        if list_has_individual(op_list(pa, about), exam):
            p  = dp_get_one(pa, dp_pa_prop)
            vn = dp_get_one(pa, dp_pa_valnorm) or dp_get_one(pa, dp_pa_value)
            if p in ("考试等级","考试级别") and vn and "殿试" in t2s(str(vn)):
                return True
    return False

def is_provincial_exam(exam) -> bool:
    for v in dp_get_all(exam, dp_exam_level):
        s = t2s(str(v))
        if "乡试" in s: return True
    for pa in PropAssertion.instances():
        if list_has_individual(op_list(pa, about), exam):
            p  = dp_get_one(pa, dp_pa_prop)
            vn = dp_get_one(pa, dp_pa_valnorm) or dp_get_one(pa, dp_pa_value)
            if p in ("考试等级","考试级别") and vn:
                s = t2s(str(vn))
                if "乡试" in s: return True
    return False

def is_palace_event(evt) -> bool:
    if any(is_palace_exam(x) for x in op_list(evt, hasExam)): return True
    for pa in PropAssertion.instances():
        if list_has_individual(op_list(pa, about), evt):
            p  = dp_get_one(pa, dp_pa_prop)
            vn = dp_get_one(pa, dp_pa_valnorm) or dp_get_one(pa, dp_pa_value)
            if p in ("考试等级","考试级别") and vn and "殿试" in t2s(str(vn)):
                return True
    return False

def is_provincial_event(evt) -> bool:
    if any(is_provincial_exam(x) for x in op_list(evt, hasExam)): return True
    for pa in PropAssertion.instances():
        if list_has_individual(op_list(pa, about), evt):
            p  = dp_get_one(pa, dp_pa_prop)
            vn = dp_get_one(pa, dp_pa_valnorm) or dp_get_one(pa, dp_pa_value)
            if p in ("考试等级","考试级别") and vn and "乡试" in t2s(str(vn)):
                return True
    return False

# ===== 书目+溯源收集（只留含人名的溯源 Text）=====
def collect_sources_and_exams_for_person(p) -> Tuple[List[Any], Dict[str, Dict[str, Any]]]:
    aliases = person_aliases(p)
    exams: List[Any] = []
    srcs: Dict[str, Dict[str, Any]] = {}
    # 事件侧
    for evt in op_list(p, participatesIn):
        if not is_palace_event(evt): continue
        for pa in PropAssertion.instances():
            if not list_has_individual(op_list(pa, about), evt): continue
            for tp in op_list(pa, derivedFrom):
                if not any(contains_name(str(b),aliases) for b in dp_get_all(tp, dp_tp_body)):
                    continue
                for s in [str(x).strip() for x in dp_get_all(tp, dp_tp_source) if str(x).strip()]:
                    key = normalize_title(s)
                    R = srcs.setdefault(key, {"src":s,"tps":set(),"infos":[]})
                    R["tps"].add(tp)
                    infos = extract_year_infos_from_title(s)
                    if infos:
                        for info in infos:
                            if info not in R["infos"]: R["infos"].append(info)
    # 殿试考试侧
    for evt in op_list(p, participatesIn):
        if not is_palace_event(evt): continue
        for ex in op_list(evt, hasExam):
            if not is_palace_exam(ex): continue
            if not list_has_individual(exams, ex):
                exams.append(ex)
            for pa in PropAssertion.instances():
                if not list_has_individual(op_list(pa, about), ex): continue
                for tp in op_list(pa, derivedFrom):
                    if not any(contains_name(str(b),aliases) for b in dp_get_all(tp, dp_tp_body)):
                        continue
                    for s in [str(x).strip() for x in dp_get_all(tp, dp_tp_source) if str(x).strip()]:
                        key = normalize_title(s)
                        R = srcs.setdefault(key, {"src":s,"tps":set(),"infos":[]})
                        R["tps"].add(tp)
                        infos = extract_year_infos_from_title(s)
                        if infos:
                            for info in infos:
                                if info not in R["infos"]: R["infos"].append(info)
    return exams, srcs

# ===== 属性名归一 & 时间断言同义词 =====
def normalize_prop_name(p: Any) -> str:
    s = t2s(str(p or "").strip())
    s = re.sub(r"\s+", "", s)
    s = s.replace("_", "").replace("-", "")
    s = s.replace("（", "(").replace("）", ")")
    for a,b in {"號":"号","紀":"纪","歲":"岁","時":"时","間":"间","規":"规","範":"范","歷":"历","曆":"历","萬":"万"}.items():
        s = s.replace(a,b)
    return s

_TIME_PROP_SYNONYMS = {
    "年号","年名","纪年","年号名",
    "序数","序次","年次","年序","序",
    "干支","天干地支","岁次","干支纪年",
    "考试时间规范","考试时间(规范)","考试时间（规范）","考试时间_规范","规范考试时间","规范时间","规范纪年","规范年",
    "考试时间原文","考试时间_原文","原始考试时间",
    "考试时间",
}
_TIME_PROP_KEYS_NORM = {normalize_prop_name(x) for x in _TIME_PROP_SYNONYMS}
def is_time_prop_name(p: Any) -> bool:
    pn = normalize_prop_name(p)
    if pn in _TIME_PROP_KEYS_NORM: return True
    if any(k in pn for k in ("考试时间","纪年","年号","干支","岁次")): return True
    return False

# ===== 断言工具 =====
def create_pa(exam, prop_name: str, value: Any, value_norm: Optional[Any], provs: List[Any]):
    pa = PropAssertion(f"PropAssertion_{uuid.uuid4().hex[:12]}")
    try: dp_pa_prop[pa].append(prop_name)
    except Exception: pass
    if value is not None:
        try: dp_pa_value[pa].append(str(value))
        except Exception: pass
    if value_norm is not None:
        try: dp_pa_valnorm[pa].append(str(value_norm))
        except Exception: pass
    op_add_unique(pa, about, exam)
    for tp in provs or []:
        op_add_unique(pa, derivedFrom, tp)
        op_add_unique(tp, contains, exam)
    return pa

def rebuild_canonical_time_block(exam, canonical: Dict[str, Any], provs: List[Any]):
    removed = 0
    for pa in list(PropAssertion.instances()):
        if not list_has_individual(op_list(pa, about), exam): continue
        p  = dp_get_one(pa, dp_pa_prop)
        if is_time_prop_name(p):
            try: destroy_entity(pa); removed += 1
            except Exception: pass
    if removed: print(f"    [-] 清理旧时间断言 {removed} 条")

    year   = str(canonical["year"])
    era    = canonical.get("era")
    ord_i  = canonical.get("ord_int")
    ganzhi = canonical.get("ganzhi")
    create_pa(exam, "考试时间_规范", year, year, provs)
    if era:   create_pa(exam, "年号", era, era, provs)
    if ord_i: create_pa(exam, "序数", int_to_cn_ordinal(ord_i), str(ord_i), provs)
    if ganzhi:create_pa(exam, "干支", ganzhi, ganzhi, provs)

def purge_time_assertions_to_canonical(exam, canonical: Dict[str, Any]):
    era_ok  = str(canonical.get("era") or "")
    ord_ok  = str(canonical.get("ord_int") or "")
    gz_ok   = str(canonical.get("ganzhi") or "")
    year_ok = str(canonical.get("year"))
    removed = 0
    for pa in list(PropAssertion.instances()):
        if not list_has_individual(op_list(pa, about), exam): continue
        p_raw = dp_get_one(pa, dp_pa_prop)
        if not is_time_prop_name(p_raw): continue
        vn = str(dp_get_one(pa, dp_pa_valnorm) or ""); vv = str(dp_get_one(pa, dp_pa_value) or "")
        val = vn if vn else vv
        pn = normalize_prop_name(p_raw); keep = True
        if "年号" in pn or "纪年" in pn: keep = (era_ok == val)
        elif "序" in pn:
            if ord_ok.isdigit(): keep = (ord_ok == val or int_to_cn_ordinal(int(ord_ok)) == val)
            else: keep = False
        elif "干支" in pn or "岁次" in pn: keep = (gz_ok == val)
        elif "规范" in pn or "考试时间" in pn: keep = (year_ok == val)
        if not keep:
            try: destroy_entity(pa); removed += 1
            except Exception: pass
    if removed: print(f"    [-] 二次净化：移除与 canonical 不一致时间断言 {removed} 条")

# ===== 共享判定等（保持不变） =====
def persons_of_event(evt) -> List[Any]:
    owners = []
    for p in Person.instances():
        if list_has_individual(op_list(p, participatesIn), evt):
            owners.append(p)
    return owners

def event_is_shared_with_others(evt, current_person) -> bool:
    for owner in persons_of_event(evt):
        if not same_individual(owner, current_person):
            return True
    return False

def exam_is_shared_with_others(exam, current_person) -> bool:
    for evt in ParticipationEvent.instances():
        if list_has_individual(op_list(evt, hasExam), exam):
            for owner in persons_of_event(evt):
                if not same_individual(owner, current_person):
                    return True
    return False

def collect_time_provs_for_exam(exam) -> List[Any]:
    tps = set()
    for pa in PropAssertion.instances():
        if list_has_individual(op_list(pa, about), exam) and is_time_prop_name(dp_get_one(pa, dp_pa_prop)):
            for tp in op_list(pa, derivedFrom):
                tps.add(tp)
    return list(tps)

def clone_event_for_person(src_evt, person) -> Any:
    evt_new = ParticipationEvent(f"ParticipationEvent_{uuid.uuid4().hex[:8]}")
    for pa in list(PropAssertion.instances()):
        if not list_has_individual(op_list(pa, about), src_evt): continue
        pa_new = PropAssertion(f"PropAssertion_{uuid.uuid4().hex[:12]}")
        try: dp_pa_prop[pa_new].append(dp_get_one(pa, dp_pa_prop))
        except: pass
        val = dp_get_one(pa, dp_pa_value)
        if val is not None:
            try: dp_pa_value[pa_new].append(val)
            except: pass
        valn = dp_get_one(pa, dp_pa_valnorm)
        if valn is not None:
            try: dp_pa_valnorm[pa_new].append(valn)
            except: pass
        op_add_unique(pa_new, about, evt_new)
        for tp in op_list(pa, derivedFrom):
            op_add_unique(pa_new, derivedFrom, tp); op_add_unique(tp, contains, evt_new)
    for tp in TextProvenance.instances():
        if list_has_individual(op_list(tp, contains), src_evt):
            op_add_unique(tp, contains, evt_new)
    cur = op_list(person, participatesIn)
    newlist = [evt_new if same_individual(x, src_evt) else x for x in cur]
    try: participatesIn[person] = newlist
    except Exception: pass
    print(f"    [CLONE-EVT] 为人物【{dp_get_one(person, dp_name) or person.name}】克隆 ParticipationEvent：{getattr(src_evt,'name','evt')} -> {evt_new.name}")
    return evt_new

def rewire_person_events_to_exam(person, target_exam, event_pred) -> int:
    cnt = 0
    evts = list(op_list(person, participatesIn))
    for evt in evts:
        if not event_pred(evt):
            continue
        if event_is_shared_with_others(evt, person):
            evt = clone_event_for_person(evt, person)
        try:
            hasExam[evt] = [target_exam]
        except Exception:
            try: hasExam[evt] = []
            except Exception: pass
            op_add_unique(evt, hasExam, target_exam)
        cnt += 1
    return cnt
def rewire_person_palace_events_to_exam(person, target_exam) -> int:
    return rewire_person_events_to_exam(person, target_exam, is_palace_event)
def rewire_person_provincial_events_to_exam(person, target_exam) -> int:
    return rewire_person_events_to_exam(person, target_exam, is_provincial_event)

def clone_exam_for_person(src_exam, target_year: int, person, canonical: Dict[str, Any], provs: List[Any], level_text: str) -> Any:
    ex_new = ImperialExam(f"ImperialExam_{target_year}_{uuid.uuid4().hex[:8]}")
    try: dp_exam_level[ex_new].append(level_text)
    except Exception: pass
    try: dp_exam_time[ex_new] = [int(target_year)]
    except Exception: dp_exam_time[ex_new] = [str(target_year)]
    rebuild_canonical_time_block(ex_new, canonical, provs)
    purge_time_assertions_to_canonical(ex_new, canonical)
    for pa in PropAssertion.instances():
        if not list_has_individual(op_list(pa, about), src_exam): continue
        prop = dp_get_one(pa, dp_pa_prop)
        if is_time_prop_name(prop):
            continue
        val  = dp_get_one(pa, dp_pa_value)
        valn = dp_get_one(pa, dp_pa_valnorm)
        create_pa(ex_new, prop, val, valn, op_list(pa, derivedFrom))
    for tp in TextProvenance.instances():
        if list_has_individual(op_list(tp, contains), src_exam):
            op_add_unique(tp, contains, ex_new)
    kind = "殿试" if level_text=="殿试" else "乡试"
    print(f"    [CLONE-{kind}] 为人物【{dp_get_one(person, dp_name) or person.name}】克隆：{getattr(src_exam,'name','Exam')} -> {ex_new.name}（{target_year}）")
    return ex_new

def migrate_non_time_assertions(loser, survivor):
    existing = set()
    for pa in PropAssertion.instances():
        if not list_has_individual(op_list(pa, about), survivor): continue
        p2 = dp_get_one(pa, dp_pa_prop)
        if is_time_prop_name(p2): continue
        p  = str(p2 or "")
        vn = str(dp_get_one(pa, dp_pa_valnorm) or "")
        vv = str(dp_get_one(pa, dp_pa_value) or "")
        key = (normalize_prop_name(p), vn if vn else "VAL::"+vv)
        existing.add(key)
    for pa in list(PropAssertion.instances()):
        if not list_has_individual(op_list(pa, about), loser): continue
        p_raw = dp_get_one(pa, dp_pa_prop)
        if is_time_prop_name(p_raw):
            try: destroy_entity(pa)
            except Exception: pass
            continue
        p  = str(p_raw or "")
        vn = str(dp_get_one(pa, dp_pa_valnorm) or "")
        vv = str(dp_get_one(pa, dp_pa_value) or "")
        key = (normalize_prop_name(p), vn if vn else "VAL::"+vv)
        if key in existing:
            for pa2 in PropAssertion.instances():
                if not list_has_individual(op_list(pa2, about), survivor): continue
                p2  = dp_get_one(pa2, dp_pa_prop)
                if is_time_prop_name(p2): continue
                vn2 = str(dp_get_one(pa2, dp_pa_valnorm) or "")
                vv2 = str(dp_get_one(pa2, dp_pa_value) or "")
                k2  = (normalize_prop_name(p2), vn2 if vn2 else "VAL::"+vv2)
                if k2 == key:
                    for tp in op_list(pa, derivedFrom):
                        op_add_unique(pa2, derivedFrom, tp)
                        op_add_unique(tp, contains, survivor)
                    break
            try: destroy_entity(pa)
            except Exception: pass
        else:
            new_about = [survivor if same_individual(x, loser) else x for x in op_list(pa, about)]
            try: about[pa] = new_about
            except Exception: pass
            for tp in op_list(pa, derivedFrom):
                op_add_unique(tp, contains, survivor)

def move_exam_edges(loser, survivor):
    for evt in ParticipationEvent.instances():
        exs = op_list(evt, hasExam)
        if any(same_individual(x, loser) for x in exs):
            kept = [x for x in exs if not same_individual(x, loser)]
            try: hasExam[evt] = kept
            except Exception: pass
            op_add_unique(evt, hasExam, survivor)
    for tp in TextProvenance.instances():
        cont = op_list(tp, contains)
        changed=False
        if any(same_individual(x, loser) for x in cont):
            cont = [x for x in cont if not same_individual(x, loser)]; changed=True
        if not any(same_individual(x, survivor) for x in cont):
            cont.append(survivor); changed=True
        if changed:
            try: contains[tp] = cont
            except Exception: pass
    try: destroy_entity(loser)
    except Exception: pass

def gc_orphan_exams():
    used_keys = set()
    for evt in ParticipationEvent.instances():
        for ex in op_list(evt, hasExam):
            used_keys.add(key_of(ex))
    removed = 0
    for ex in list(ImperialExam.instances()):
        if key_of(ex) and key_of(ex) not in used_keys:
            try: destroy_entity(ex); removed += 1
            except Exception: pass
    if removed:
        print(f"[CLEANUP] 回收无引用考试实例 {removed} 个")

def canonical_for_year(year: int) -> Dict[str, Any]:
    era_name, ord_int = None, None
    era_span = era_of_year(int(year))
    if era_span:
        era_name, era_start = era_span
        ord_int = int(year) - era_start + 1
    return {"year": int(year), "era": era_name, "ord_int": ord_int, "ganzhi": ganzhi_of_year(int(year))}

PERSON_AUTHORITY: Dict[str, Dict[str, Any]] = {}

def info_score_for_exam(exam) -> int:
    score = 0
    for pa in PropAssertion.instances():
        if not list_has_individual(op_list(pa, about), exam): continue
        if not is_time_prop_name(dp_get_one(pa, dp_pa_prop)):
            score += 1
    used = 0
    for evt in ParticipationEvent.instances():
        if list_has_individual(op_list(evt, hasExam), exam):
            used += 1
    score += used * 2
    score += max(0, 50 - len(getattr(exam, "name", "")))
    return score

def normalized_year_for_exam(exam) -> Optional[int]:
    y = dp_get_one(exam, dp_exam_time)
    if y is not None:
        s = str(y).strip()
        if s.isdigit(): return int(s)
    for pa in PropAssertion.instances():
        if not list_has_individual(op_list(pa, about), exam): continue
        if normalize_prop_name(dp_get_one(pa, dp_pa_prop)) in _TIME_PROP_KEYS_NORM:
            v = dp_get_one(pa, dp_pa_valnorm) or dp_get_one(pa, dp_pa_value)
            s = str(v or "").strip()
            if s.isdigit(): return int(s)
    return None

def merge_palace_exams_by_year():
    year_map: Dict[int, List[Any]] = defaultdict(list)
    for ex in ImperialExam.instances():
        if not is_palace_exam(ex): continue
        y = normalized_year_for_exam(ex)
        if y is None: continue
        year_map[y].append(ex)
    print(f"\n—— 殿试按规范年合并：发现 {len(year_map)} 个年份键 ——")
    total_merged = 0
    for y, all_ex in year_map.items():
        unique: Dict[str, Any] = {}
        for ex in all_ex:
            k = key_of(ex)
            if k and k not in unique: unique[k] = ex
        exams = list(unique.values())
        if len(exams) <= 1: continue
        cur_sorted = sorted(exams, key=lambda e: info_score_for_exam(e), reverse=True)
        survivor, victims = cur_sorted[0], cur_sorted[1:]
        print(f"  [MERGE-YEAR] {y} 年：{len(exams)} → 1；幸存者={getattr(survivor,'name','Exam')}")
        canonical = canonical_for_year(y)
        provs = collect_time_provs_for_exam(survivor)
        try: dp_exam_time[survivor] = [y]
        except Exception: dp_exam_time[survivor] = [str(y)]
        rebuild_canonical_time_block(survivor, canonical, provs)
        purge_time_assertions_to_canonical(survivor, canonical)
        for v in victims:
            if same_individual(v, survivor): continue
            migrate_non_time_assertions(v, survivor)
            move_exam_edges(v, survivor)
            print(f"    [-] 合并 {getattr(v,'name','Exam')} → {getattr(survivor,'name','Exam')}")
            total_merged += 1
    if total_merged == 0:
        print("[MERGE-YEAR] 无需合并")
    else:
        print(f"[MERGE-YEAR] 合并完成：共合并 {total_merged} 个实例")

def process_palace_all():
    persons = list(Person.instances())
    print(f"[INFO] 参与人数：{len(persons)}")
    for i, p in enumerate(persons, 1):
        aliases = person_aliases(p)
        pname = aliases[0] if aliases else getattr(p,"name","（未命名人物）")
        exams, srcs = collect_sources_and_exams_for_person(p)
        print(f"\n===== 人物 {i}/{len(persons)}：{pname} =====")
        if not exams:
            print("  [SKIP] 无殿试实例"); continue

        vote = Counter(); year2_tps: Dict[int, Set[Any]] = defaultdict(set)
        for rec in srcs.values():
            s = rec["src"]
            infos = rec.get("infos", [])
            for (_, _, y) in infos:
                vote[y] += 1; year2_tps[y].update(rec.get("tps", set()))

        if not vote:
            print("  [INFO] 无法从溯源标题严格解析出“年号+（元/正/数）年”（且需伴随进士/登科/进土），跳过写回")
            continue

        mc = vote.most_common(); top = mc[0][1]
        candidates = sorted([y for y,c in mc if c==top])
        chosen = int(candidates[0])
        provs  = list(year2_tps.get(chosen, []))
        canonical = canonical_for_year(chosen)
        print(f"  [判定] 权威年份 = {chosen}  （多数票={top}；平票={candidates}）")
        print(f"         规范：年号={canonical['era'] or '—'}；序数={canonical['ord_int'] or '—'}；干支={canonical['ganzhi']}")

        survivor = None
        for ex in exams:
            if str(dp_get_one(ex, dp_exam_time)) == str(chosen):
                if exam_is_shared_with_others(ex, p):
                    survivor = clone_exam_for_person(ex, chosen, p, canonical, provs, level_text="殿试")
                else:
                    survivor = ex
                    linked = rewire_person_palace_events_to_exam(p, survivor)
                    if linked:
                        print(f"  [LINK-DS] 统一连接 {linked} 个殿试 ParticipationEvent -> {getattr(survivor,'name','Exam')}")
                break
        if survivor is None:
            template = exams[0]
            survivor = clone_exam_for_person(template, chosen, p, canonical, provs, level_text="殿试")
            rewire_person_palace_events_to_exam(p, survivor)

        try: dp_exam_time[survivor] = [chosen]
        except Exception: dp_exam_time[survivor] = [str(chosen)]
        rebuild_canonical_time_block(survivor, canonical, provs)
        purge_time_assertions_to_canonical(survivor, canonical)
        print(f"  [WRITE] 幸存者 {getattr(survivor,'name','Exam')}：考试时间 <- {chosen}")

        for loser in sorted([x for x in exams if not same_individual(x, survivor)], key=lambda x: iri_of(x)):
            if exam_is_shared_with_others(loser, p):
                moved = 0
                for evt in op_list(p, participatesIn):
                    if not is_palace_event(evt): continue
                    if list_has_individual(op_list(evt, hasExam), loser):
                        if event_is_shared_with_others(evt, p):
                            evt = clone_event_for_person(evt, p)
                        try: hasExam[evt] = [survivor]
                        except Exception:
                            try: hasExam[evt] = []
                            except Exception: pass
                            op_add_unique(evt, hasExam, survivor)
                        moved += 1
                if moved:
                    print(f"  [REWIRE-SHARED] {getattr(loser,'name','Exam')}（共享）仅重挂此人殿试事件 {moved} 个")
            else:
                migrate_non_time_assertions(loser, survivor)
                move_exam_edges(loser, survivor)
                print(f"  [MERGE] 非共享 {getattr(loser,'name','Exam')} → {getattr(survivor,'name','Exam')}")
        linked = rewire_person_palace_events_to_exam(p, survivor)
        if linked:
            print(f"  [LINK-DS] 二次收敛：{linked} 个殿试 ParticipationEvent -> {getattr(survivor,'name','Exam')}")

        PERSON_AUTHORITY[key_of(p)] = {"year": chosen, "provs": provs, "survivor": survivor}
        print(f"  [DONE] 人物【{pname}】殿试幸存者：{getattr(survivor,'name','Exam')} | 年={dp_get_one(survivor, dp_exam_time)}")
    merge_palace_exams_by_year()
    postprocess_rewrite_to_authoritative_safe()

def postprocess_rewrite_to_authoritative_safe():
    print("\n—— 殿试后处理：按权威年份校准仍在用殿试 ——")
    changed = 0
    for p in Person.instances():
        auth = PERSON_AUTHORITY.get(key_of(p))
        if not auth: continue
        auth_year = int(auth["year"])
        provs = auth.get("provs", []) or []
        canonical = canonical_for_year(auth_year)
        pname = dp_get_one(p, dp_name) or getattr(p, "name", "")
        evts = list(op_list(p, participatesIn))
        for evt in evts:
            if not is_palace_event(evt): continue
            if event_is_shared_with_others(evt, p):
                evt = clone_event_for_person(evt, p)
            exs = [x for x in op_list(evt, hasExam) if is_palace_exam(x)]
            if len(exs)==1 and str(dp_get_one(exs[0], dp_exam_time)) == str(auth_year):
                continue
            target_exam = auth.get("survivor")
            if not target_exam or str(dp_get_one(target_exam, dp_exam_time)) != str(auth_year):
                target_exam = None
                for ex in ImperialExam.instances():
                    if is_palace_exam(ex) and str(dp_get_one(ex, dp_exam_time)) == str(auth_year):
                        target_exam = ex; break
                if not target_exam:
                    template = exs[0] if exs else ImperialExam(f"ImperialExam_{auth_year}_{uuid.uuid4().hex[:8]}")
                    target_exam = clone_exam_for_person(template, auth_year, p, canonical, provs, level_text="殿试")
            try: hasExam[evt] = [target_exam]
            except Exception:
                try: hasExam[evt] = []
                except Exception: pass
                op_add_unique(evt, hasExam, target_exam)
            changed += 1
            print(f"  [POST-FIX-DS] 人物={pname} 事件={getattr(evt,'name','evt')}：仅连殿试权威 -> {getattr(target_exam,'name','Exam')}({auth_year})")
    if not changed:
        print("[POST] 殿试后处理：无需改写")

def process_provincial_by_person():
    print("\n—— 乡试筛选：逐人以殿试权威时间为锚点（差值≤16） ——")
    handled = 0
    for p in Person.instances():
        pkey = key_of(p)
        auth = PERSON_AUTHORITY.get(pkey)
        if not auth:
            continue
        auth_year = int(auth["year"])
        pname = dp_get_one(p, dp_name) or getattr(p, "name", "")

        palace_exams = set()
        for evt in op_list(p, participatesIn):
            for ex in op_list(evt, hasExam):
                if is_palace_exam(ex):
                    palace_exams.add(ex)
        if len(palace_exams) != 1:
            if len(palace_exams) > 1:
                print(f"  [SKIP-PROV] {pname}：殿试实例数={len(palace_exams)} > 1，放弃乡试处理")
            continue

        prov_exams: Set[Any] = set()
        for evt in op_list(p, participatesIn):
            if not is_provincial_event(evt): continue
            for ex in op_list(evt, hasExam):
                if is_provincial_exam(ex):
                    prov_exams.add(ex)
        if not prov_exams:
            continue

        earlier_diffs = []
        for ex in prov_exams:
            y = normalized_year_for_exam(ex)
            if y is not None and y < auth_year:
                earlier_diffs.append(auth_year - y)

        cands = []
        for ex in prov_exams:
            y = normalized_year_for_exam(ex)
            if y is None: continue
            diff = auth_year - y
            if diff <= 0 or diff > MAX_PROV_GAP:
                continue
            cands.append((diff, y, info_score_for_exam(ex), ex))

        if not cands:
            if earlier_diffs:
                md = min(earlier_diffs)
                if md > MAX_PROV_GAP:
                    print(f"  [SKIP-PROV] {pname}：最近乡试距殿试 {md} 年 > {MAX_PROV_GAP}，跳过乡试处理")
                else:
                    print(f"  [SKIP-PROV] {pname}：虽有早于殿试的乡试，但无法解析或数据异常，跳过")
            else:
                print(f"  [SKIP-PROV] {pname}：无早于殿试的乡试，跳过")
            continue

        cands.sort(key=lambda t: (t[0], -t[2]))
        diff_best, prov_year, _, chosen_exam = cands[0]
        canonical = canonical_for_year(prov_year)
        provs = collect_time_provs_for_exam(chosen_exam)

        if exam_is_shared_with_others(chosen_exam, p):
            survivor = clone_exam_for_person(chosen_exam, prov_year, p, canonical, provs, level_text="乡试")
        else:
            survivor = chosen_exam

        linked = rewire_person_provincial_events_to_exam(p, survivor)
        if linked:
            print(f"  [LINK-XS] 人物={pname} 统一连接 {linked} 个乡试事件 -> {getattr(survivor,'name','Exam')}（差 {diff_best} 年）")

        try: dp_exam_time[survivor] = [prov_year]
        except Exception: dp_exam_time[survivor] = [str(prov_year)]
        rebuild_canonical_time_block(survivor, canonical, provs)
        purge_time_assertions_to_canonical(survivor, canonical)

        for loser in sorted([x for x in prov_exams if not same_individual(x, survivor)], key=lambda x: iri_of(x)):
            if exam_is_shared_with_others(loser, p):
                moved = 0
                for evt in op_list(p, participatesIn):
                    if not is_provincial_event(evt): continue
                    if list_has_individual(op_list(evt, hasExam), loser):
                        if event_is_shared_with_others(evt, p):
                            evt = clone_event_for_person(evt, p)
                        try: hasExam[evt] = [survivor]
                        except Exception:
                            try: hasExam[evt] = []
                            except Exception: pass
                            op_add_unique(evt, hasExam, survivor)
                        moved += 1
                if moved:
                    print(f"  [REWIRE-XS] 共享 {getattr(loser,'name','Exam')}：仅重挂此人乡试事件 {moved} 个 -> {getattr(survivor,'name','Exam')}")
            else:
                migrate_non_time_assertions(loser, survivor)
                move_exam_edges(loser, survivor)
                print(f"  [MERGE-XS] 非共享 {getattr(loser,'name','Exam')} → {getattr(survivor,'name','Exam')}")

        print(f"  [DONE-XS] 人物【{pname}】乡试幸存者：{getattr(survivor,'name','Exam')} | 年={dp_get_one(survivor, dp_exam_time)}")
        handled += 1

    if handled == 0:
        print("[XS] 本轮无可处理对象（条件不满足或无候选≤16年）")
    else:
        print(f"[XS] 乡试筛选完成：处理 {handled} 人（差值≤{MAX_PROV_GAP}）")

def save_all():
    try:
        onto.save(file=OUT_OWL, format="rdfxml"); print(f"\n[OK] OWL -> {OUT_OWL}")
    except Exception as e:
        print(f"[WARN] OWL 保存失败：{e}")
    try:
        g = world.as_rdflib_graph()
        ttl = g.serialize(format="turtle")
        with open(OUT_TTL,"wb") as f: f.write(ttl if isinstance(ttl,(bytes,bytearray)) else ttl.encode("utf-8"))
        nt  = g.serialize(format="nt")
        with open(OUT_NT, "wb") as f: f.write(nt  if isinstance(nt,(bytes,bytearray))  else nt.encode("utf-8"))
        print(f"[OK] TTL/NT -> {OUT_TTL} / {OUT_NT}")
    except Exception as e:
        print(f"[WARN] TTL/NT 导出失败：{e}")

if __name__ == "__main__":
    print("[START] 殿试规范化（共享安全） + 乡试筛选（与殿试对齐，差值≤16）")
    process_palace_all()
    process_provincial_by_person()
    gc_orphan_exams()
    save_all()
    print("[DONE]")
