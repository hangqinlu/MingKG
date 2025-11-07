# apps/run_pipeline.py
# -*- coding: utf-8 -*-
import os
import sys
import time
import datetime
import subprocess
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import streamlit as st

# =============== 工具函数 ===============
def shlex_join_win(args: List[str]) -> str:
    def q(a: str) -> str:
        a = str(a)
        if " " in a or "\\" in a:
            return f'"{a}"'
        return a
    return " ".join(q(x) for x in args)

def newest_nt_in(dirpath: Path) -> Optional[Path]:
    if not dirpath.exists():
        return None
    nts = sorted(dirpath.glob("*.nt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return nts[0] if nts else None

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def now_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def find_scripts(root: Path, count: int = 20) -> Dict[int, Path]:
    """递归在 root 下寻找 '脚本{i}.py'（i=1..count）"""
    found: Dict[int, Path] = {}
    for i in range(1, count + 1):
        name = f"脚本{i}.py"
        for p in root.rglob(name):
            found[i] = p
            break
    return found

# 运行子进程并把输出实时写到前端（强制 UTF-8 / 无缓冲 / 心跳；无超时）
def run_and_stream(
    cmd: List[str],
    workdir: Optional[Path],
    out_area,
) -> Tuple[int, str]:
    # 环境：强制 Python 子进程 UTF-8 & 无缓冲；若有 Java 也设编码
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"              # 3.7+ 启用 UTF-8 模式
    env["PYTHONIOENCODING"] = "utf-8"    # stdout/stderr 明确 UTF-8
    env["PYTHONUNBUFFERED"] = "1"        # 无缓冲
    env.setdefault("JAVA_TOOL_OPTIONS", "-Dfile.encoding=UTF-8")

    proc = subprocess.Popen(
        cmd,
        cwd=str(workdir) if workdir else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,                  # 行缓冲（配合 -u 更稳）
        universal_newlines=True,
        env=env,
    )

    full: List[str] = []
    last_line_ts = time.time()
    HEARTBEAT_SEC = 10  # 每 10 秒打一条心跳

    while True:
        line = proc.stdout.readline()
        if line:
            full.append(line)
            out_area.write(line.rstrip("\n"))
            last_line_ts = time.time()
        else:
            if proc.poll() is not None:
                break
            now = time.time()
            if now - last_line_ts > HEARTBEAT_SEC:
                out_area.write("…(仍在运行，可能处于推理/导出阶段)…")
                last_line_ts = now
            time.sleep(0.2)

    rc = proc.wait()
    try:
        proc.stdout.close()
    except Exception:
        pass
    return rc, "".join(full)

# =============== 页面入口 ===============
def run(configure_page: bool = False):
    st.caption("• 自动递归搜索脚本 • 全局 OUT_DIR 复用 • 自动链路承接 • 实时日志（1→20）")

    with st.sidebar:
        st.header("基础配置")
        default_root = Path(__file__).resolve().parents[1] / "pipeline"
        scripts_root = Path(
            st.text_input(
                "脚本根目录（递归搜索 脚本1.py~脚本20.py）",
                value=str(default_root),
                key="rp_scripts_root",
            )
        )

        out_dir_global = Path(
            st.text_input(
                "全局 OUT_DIR（所有产物归档到此）",
                value=str(Path.home() / "Desktop" / "本体结构"),
                key="rp_out_dir",
            )
        )
        ensure_dir(out_dir_global)

        st.header("脚本1 · Excel/JSON")
        excel_path = st.text_input(
            "Excel 路径",
            value=os.getenv("EXCEL_PATH", r"C:\Users\卢航青\Desktop\地点实例规范.xlsx"),
            key="rp_excel",
        )
        json_in = st.text_input(
            "输入 JSON 路径",
            value=os.getenv("JSON_IN_PATH", r"C:\Users\卢航青\PycharmProjects\pythonProject11\OSPLR-main\data\group2.json"),
            key="rp_json_in",
        )
        json_out_name = st.text_input(
            "输出 JSON 文件名",
            value=os.getenv("JSON_OUT_NAME", "group2.json"),
            key="rp_json_out_name",
        )

        st.header("脚本2/3 · Neo4j")
        neo4j_uri = st.text_input("Neo4j URI", value=os.getenv("NEO4J_URI", "bolt://localhost:7687"), key="rp_uri")
        neo4j_user = st.text_input("Neo4j 用户", value=os.getenv("NEO4J_USER", "neo4j"), key="rp_user")
        neo4j_pwd = st.text_input("Neo4j 密码（明文，可留空）", value=os.getenv("NEO4J_PWD", ""), key="rp_pwd")
        neo4j_db = st.text_input("Neo4j 数据库名（留空=默认）", value=os.getenv("NEO4J_DB", ""), key="rp_db")
        docx_path = st.text_input("脚本2 · Word 溯源 DOCX（可留空）", value=os.getenv("DOCX_PATH", ""), key="rp_docx")

        st.header("脚本4 · 选项")
        include_sm = st.checkbox("包含南明年号", value=True, key="rp_include_sm")

        st.header("其它")
        python_exe = st.text_input("Python 解释器", value=sys.executable, key="rp_py")
        run_btn = st.button("🚀 一键运行（1→20）", type="primary", use_container_width=True, key="rp_run_btn")

    log_tab, summary_tab = st.tabs(["🖨️ 实时日志", "📦 结果摘要"])

    if run_btn:
        with log_tab:
            st.write(f"[{now_ts()}] 开始执行")
            st.write(f"搜索目录：`{scripts_root}`")
            st.write(f"全局 OUT_DIR：`{out_dir_global}`")
            scripts = find_scripts(scripts_root, count=20)
            missing = [i for i in range(1, 21) if i not in scripts]
            if missing:
                st.error(f"未找到以下脚本：{missing}（请确认位于 {scripts_root} 或子目录）")
                st.stop()

            progress = st.progress(0.0)
            last_json: Optional[Path] = None
            last_nt: Optional[Path] = None
            json_out = out_dir_global / json_out_name

            # 预检查
            if not Path(excel_path).exists():
                st.error(f"Excel 不存在：{excel_path}"); st.stop()
            if not Path(json_in).exists():
                p = Path(json_in)
                if p.suffix.lower() == ".jso" and p.with_suffix(".json").exists():
                    json_in = str(p.with_suffix(".json"))
                    st.warning(f"[更正] 自动将输入 JSON 修正为：{json_in}")
                else:
                    st.error(f"输入 JSON 不存在：{json_in}"); st.stop()

            # 逐步执行
            for sid in range(1, 21):
                spath = scripts[sid]
                # 子进程 Python 处于 UTF-8 + 无缓冲模式
                cmd = [python_exe, "-X", "utf8", "-u", str(spath)]

                if sid == 1:
                    cmd += ["--excel", excel_path, "--json-in", json_in, "--json-out", str(json_out)]
                    st.code(shlex_join_win(cmd), language="bash")
                    rc, _ = run_and_stream(cmd, spath.parent, st)
                    if rc != 0:
                        st.error(f"[FAILED] 脚本1 退出码={rc}"); st.stop()
                    last_json = json_out
                    st.success(f"[STEP 1] JSON_OUT = {last_json}")

                elif sid == 2:
                    if not last_json or not last_json.exists():
                        st.error("[ERROR] 未获取脚本1产出的 JSON_OUT"); st.stop()
                    cmd += ["--json", str(last_json),
                            "--neo4j-uri", neo4j_uri, "--neo4j-user", neo4j_user, "--neo4j-pwd", neo4j_pwd]
                    if docx_path:
                        cmd += ["--docx", docx_path]
                    if neo4j_db:
                        cmd += ["--neo4j-db", neo4j_db]
                    st.code(shlex_join_win(cmd), language="bash")
                    rc, _ = run_and_stream(cmd, spath.parent, st)
                    if rc != 0:
                        st.error(f"[FAILED] 脚本2 退出码={rc}"); st.stop()

                elif sid == 3:
                    cmd += ["--neo4j-uri", neo4j_uri, "--neo4j-user", neo4j_user, "--neo4j-pwd", neo4j_pwd,
                            "--out-dir", str(out_dir_global), "--reasoner-before"]
                    st.code(shlex_join_win(cmd), language="bash")
                    rc, _ = run_and_stream(cmd, spath.parent, st)
                    if rc != 0:
                        st.error(f"[FAILED] 脚本3 退出码={rc}"); st.stop()
                    last_nt = newest_nt_in(out_dir_global)
                    if not last_nt:
                        st.error("[ERROR] 脚本3结束后未发现 .nt 文件"); st.stop()
                    st.success(f"[STEP 3] 最新 NT：{last_nt}")

                elif sid == 4:
                    if not last_nt or not last_nt.exists():
                        st.error("[ERROR] 未获取脚本3产出的 NT"); st.stop()
                    cmd += ["--src", str(last_nt), "--out-dir", str(out_dir_global)]
                    if include_sm:
                        cmd += ["--include-southern-ming"]
                    st.code(shlex_join_win(cmd), language="bash")
                    rc, _ = run_and_stream(cmd, spath.parent, st)
                    if rc != 0:
                        st.error(f"[FAILED] 脚本4 退出码={rc}"); st.stop()
                    last_nt = newest_nt_in(out_dir_global)
                    if not last_nt:
                        st.error("[ERROR] 脚本4结束后未发现 .nt 文件"); st.stop()
                    st.success(f"[STEP 4] 最新 NT：{last_nt}")

                else:
                    if not last_nt or not last_nt.exists():
                        st.error(f"[ERROR] 未获取到上一环节 NT（step={sid-1}）"); st.stop()
                    cmd += ["--src", str(last_nt), "--out-dir", str(out_dir_global)]
                    st.code(shlex_join_win(cmd), language="bash")
                    rc, _ = run_and_stream(cmd, spath.parent, st)
                    if rc != 0:
                        st.error(f"[FAILED] 脚本{sid} 退出码={rc}"); st.stop()
                    nt = newest_nt_in(out_dir_global)
                    if nt:
                        last_nt = nt
                        st.success(f"[STEP {sid}] 最新 NT：{last_nt}")

                progress.progress(sid / 20.0)
                time.sleep(0.05)

            st.success(f"[{now_ts()}] ✅ 全流程完成（1→20）")
            if last_nt:
                st.write(f"最后产物（latest NT）: `{last_nt}`")
            st.write(f"全局输出目录: `{out_dir_global}`")

    with summary_tab:
        st.info("在“实时日志”里可查看每步命令与输出；所有产物写入全局 OUT_DIR。")
