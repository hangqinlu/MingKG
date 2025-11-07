# -*- coding: utf-8 -*-
"""
自动主控脚本（20步 · 自动发现脚本路径 · 全局 OUT_DIR 复用 · 自动链路承接）
- 仅在开头输入一次必要参数；之后全自动执行，无需逐步确认
- 自动递归搜索 “脚本1.py … 脚本20.py” 并按编号执行
- 链路承接：
  #1 JSON_OUT -> #2 --json
  #3 在全局 OUT_DIR 产出的最新 .nt -> #4 --src
  #4 的最新 .nt -> #5 --src -> … -> #20 --src
"""

import os
import sys
import subprocess
import datetime
from pathlib import Path
from typing import Optional, List, Dict

# ========== 基础工具 ==========
def shlex_join_win(args: List[str]) -> str:
    def q(a: str) -> str:
        a = str(a)
        if " " in a or "\\" in a:
            return f'"{a}"'
        return a
    return " ".join(q(x) for x in args)

def run(cmd: List[str], cwd: Optional[Path] = None) -> int:
    print(f"\n[RUN] {shlex_join_win(cmd)}")
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode

def newest_nt_in(dirpath: Path) -> Optional[Path]:
    if not dirpath.exists():
        return None
    nts = sorted(dirpath.glob("*.nt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return nts[0] if nts else None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# ========== 简易输入 ==========
def prompt_str(msg: str, default: Optional[str] = None, allow_empty: bool = False) -> str:
    tip = f" [{default}]" if default not in (None, "") else ""
    while True:
        val = input(f"{msg}{tip}: ").strip()
        if not val and default is not None:
            return str(default)
        if val or allow_empty:
            return val
        print("不能为空，请重试。")

def prompt_existing_path(msg: str, default: Optional[str] = None) -> str:
    while True:
        p = prompt_str(msg, default=default)
        if Path(p).exists():
            return p
        print("路径不存在，请重试。")

def prompt_dir_create(msg: str, default: Optional[str]) -> str:
    p = prompt_str(msg, default=default)
    ensure_dir(Path(p))
    return p

def prompt_yes_no(msg: str, default: bool = True) -> bool:
    d = "Y/n" if default else "y/N"
    s = input(f"{msg} ({d}): ").strip().lower()
    if not s:
        return default
    return s in ("y", "yes", "true", "1")

# ========== 自动发现脚本 ==========
def find_scripts(root: Path, count: int = 20) -> Dict[int, Path]:
    """
    递归在 root 下寻找 '脚本{i}.py'（i=1..count）
    返回 {i: Path}，缺失的编号不包含在字典中
    """
    found: Dict[int, Path] = {}
    for i in range(1, count + 1):
        name = f"脚本{i}.py"
        for p in root.rglob(name):
            found[i] = p
            break
    return found

# ========== 主流程 ==========
def main():
    print(f"[{now_ts()}] 自动主控启动（20步 · 无需人工监督）")

    pyexe = sys.executable

    # 一次性获取必要参数
    scripts_root = Path(prompt_existing_path("脚本根目录（将递归搜索 脚本1.py~脚本20.py）",
                                             default=str(Path.cwd())))

    out_dir_global = Path(prompt_dir_create("全局 OUT_DIR", default=str(Path.cwd() / "本体结构")))
    ensure_dir(out_dir_global)

    excel_path = prompt_existing_path("脚本1 · Excel 路径",
                                      default=os.getenv("EXCEL_PATH", r"C:\Users\卢航青\Desktop\地点实例规范.xlsx"))
    json_in    = prompt_existing_path("脚本1 · 输入 JSON 路径",
                                      default=os.getenv("JSON_IN_PATH", r"C:\Users\卢航青\PycharmProjects\pythonProject11\OSPLR-main\data\group2.json"))
    json_out   = str(out_dir_global / Path(os.getenv("JSON_OUT_NAME", "group2.json")).name)

    neo4j_uri  = prompt_str("脚本2/3 · Neo4j URI",  default=os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user = prompt_str("脚本2/3 · Neo4j 用户", default=os.getenv("NEO4J_USER", "neo4j"))
    neo4j_pwd  = prompt_str("脚本2/3 · Neo4j 密码（明文，可留空）", default=os.getenv("NEO4J_PWD", ""), allow_empty=True)
    neo4j_db   = prompt_str("脚本2 · Neo4j 数据库名（留空=默认）", default=os.getenv("NEO4J_DB", ""), allow_empty=True)
    docx_path  = prompt_str("脚本2 · Word 溯源 DOCX（可留空）", default=os.getenv("DOCX_PATH", ""), allow_empty=True)

    include_sm = prompt_yes_no("脚本4 · 是否包含南明年号？", default=True)

    # 自动发现脚本路径
    scripts = find_scripts(scripts_root, count=20)
    missing = [i for i in range(1, 21) if i not in scripts]
    if missing:
        raise FileNotFoundError(f"未找到以下脚本：{missing}\n请确认它们位于：{scripts_root}（或其子目录）")

    # 运行上下文（自动承接）
    last_json: Optional[Path] = None
    last_nt:   Optional[Path] = None

    # ====== 执行 ======
    for sid in range(1, 21):
        spath = scripts[sid]
        cmd = [pyexe, str(spath)]

        if sid == 1:
            # 脚本1：--excel --json-in --json-out （JSON_OUT 写入全局 OUT_DIR）
            cmd += ["--excel", excel_path, "--json-in", json_in, "--json-out", json_out]
            print(f"\n[STEP 1] 将要执行：{shlex_join_win(cmd)}")
            rc = run(cmd, cwd=spath.parent)
            if rc != 0:
                raise SystemExit(f"[FAILED] 脚本1 退出码={rc}")
            last_json = Path(json_out)
            print(f"[STEP 1] JSON_OUT = {last_json}")

        elif sid == 2:
            # 脚本2：承接 #1 的 JSON；写入 Neo4j
            if not last_json or not last_json.exists():
                raise SystemExit("[ERROR] 未获取到脚本1产出的 JSON_OUT")
            cmd += ["--json", str(last_json),
                    "--neo4j-uri", neo4j_uri, "--neo4j-user", neo4j_user, "--neo4j-pwd", neo4j_pwd]
            if docx_path:
                cmd += ["--docx", docx_path]
            if neo4j_db:
                cmd += ["--neo4j-db", neo4j_db]
            print(f"\n[STEP 2] 将要执行：{shlex_join_win(cmd)}")
            rc = run(cmd, cwd=spath.parent)
            if rc != 0:
                raise SystemExit(f"[FAILED] 脚本2 退出码={rc}")

        elif sid == 3:
            # 脚本3：Neo4j→OWL/NT，产物写全局 OUT_DIR（随后自动承接最新 .nt）
            cmd += ["--neo4j-uri", neo4j_uri, "--neo4j-user", neo4j_user, "--neo4j-pwd", neo4j_pwd,
                    "--out-dir", str(out_dir_global), "--reasoner-before"]
            print(f"\n[STEP 3] 将要执行：{shlex_join_win(cmd)}")
            rc = run(cmd, cwd=spath.parent)
            if rc != 0:
                raise SystemExit(f"[FAILED] 脚本3 退出码={rc}")
            last_nt = newest_nt_in(out_dir_global)
            if not last_nt:
                raise SystemExit("[ERROR] 脚本3结束后未发现 .nt 文件")
            print(f"[STEP 3] 最新 NT：{last_nt}")

        elif sid == 4:
            # 脚本4：承接 #3 的 .nt；产物仍写全局 OUT_DIR
            if not last_nt or not last_nt.exists():
                raise SystemExit("[ERROR] 未获取到脚本3产出的 NT")
            cmd += ["--src", str(last_nt), "--out-dir", str(out_dir_global)]
            if include_sm:
                cmd += ["--include-southern-ming"]
            print(f"\n[STEP 4] 将要执行：{shlex_join_win(cmd)}")
            rc = run(cmd, cwd=spath.parent)
            if rc != 0:
                raise SystemExit(f"[FAILED] 脚本4 退出码={rc}")
            last_nt = newest_nt_in(out_dir_global)
            if not last_nt:
                raise SystemExit("[ERROR] 脚本4结束后未发现 .nt 文件")
            print(f"[STEP 4] 最新 NT：{last_nt}")

        else:
            # 脚本5~20：统一承接 latest NT，产物仍写全局 OUT_DIR
            if not last_nt or not last_nt.exists():
                raise SystemExit(f"[ERROR] 未获取到上一环节 NT（step={sid-1}）")
            cmd += ["--src", str(last_nt), "--out-dir", str(out_dir_global)]
            print(f"\n[STEP {sid}] 将要执行：{shlex_join_win(cmd)}")
            rc = run(cmd, cwd=spath.parent)
            if rc != 0:
                raise SystemExit(f"[FAILED] 脚本{sid} 退出码={rc}")
            nt = newest_nt_in(out_dir_global)
            if nt:
                last_nt = nt
                print(f"[STEP {sid}] 最新 NT：{last_nt}")

    print(f"\n[{now_ts()}] ✅ 全流程完成（1→20）")
    if last_nt:
        print(f"最后产物（latest NT）: {last_nt}")
    print(f"全局输出目录: {out_dir_global}")

if __name__ == "__main__":
    main()
