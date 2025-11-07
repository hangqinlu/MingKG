# -*- coding: utf-8 -*-
# run_force_merge.py
import os
import sys
import subprocess
from pathlib import Path

# ======= 按需修改这三项 =======
SCRIPT_PATH = r"C:\Users\卢航青\PycharmProjects\pythonProject11\Database\pipeline\脚本5.py"  # 脚本路径
SRC         = r"C:\Users\卢航青\Desktop\本体结构\ontology_dedup_updated.nt"       # 输入 .nt/.ttl/.owl/.rdf 文件
OUT_DIR     = r"C:\Users\卢航青\Desktop\本体结构"                       # 输出目录
# =================================

def main():
    sp = Path(SCRIPT_PATH)
    src = Path(SRC)
    out = Path(OUT_DIR)

    if not sp.exists():
        raise FileNotFoundError(f"找不到脚本：{sp}")
    if not src.exists() or src.is_dir():
        raise FileNotFoundError(f"输入文件不存在或是目录：{src}")
    out.mkdir(parents=True, exist_ok=True)

    # 该脚本支持 --src/--onto 和 --out-dir，这里统一用 --src 与 --out-dir
    cmd = [
        sys.executable,
        str(sp),
        "--src", str(src),
        "--out-dir", str(out),
    ]

    # 同步一份环境变量（脚本内部也会从这些变量兜底读取）
    env = os.environ.copy()
    env["ONTO_FILE"] = str(src)
    env["OUT_DIR"]   = str(out)

    print("="*88)
    print("即将执行 force_merge_events_by_level.py")
    print("="*88)
    print(f"Python   : {sys.executable}")
    print(f"脚本     : {sp}")
    print(f"SRC      : {src}")
    print(f"OUT_DIR  : {out}")
    print("-"*88)

    try:
        subprocess.run(cmd, check=True, env=env, cwd=str(sp.parent))
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] 脚本运行失败：{e}")
        sys.exit(e.returncode)

    print("\n[OK] 合并完成，已在输出目录写出 NT/TTL。")

if __name__ == "__main__":
    main()
