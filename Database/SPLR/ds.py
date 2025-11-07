from openai import OpenAI
import re
import json
import yaml
import os

def load_llm_config(path=None):
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, 'configs', 'config.yaml')
    else:
        config_path = path
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config['llm']
llm_config = load_llm_config()
MODEL_NAME = llm_config.get('model', 'deepseek-reasoner')
TIMEOUT = llm_config.get('timeout', 1500)
def is_valid_jsonl(line):
    try:
        json.loads(line)
        return True
    except Exception:
        return False
def clean_stream_response(raw: str) -> str:
    no_think = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    return re.sub(r"^[^{]*", "", no_think, count=1).strip()

def extract_json_objs(text):
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return [data]
    except Exception:
        pass

    lines = [ln for ln in text.splitlines() if is_valid_jsonl(ln)]
    if len(lines) > 0 and len(lines) >= text.count('{') // 2:
        return [json.loads(ln) for ln in lines]

    pattern = r'(\{[\s\S]*?\})(?=(?:\s*\{)|\s*$)'
    matches = re.finditer(pattern, text)
    objs = []
    for m in matches:
        try:
            obj = json.loads(m.group(1))
            objs.append(obj)
        except Exception:
            continue
    return objs


client = OpenAI(
    base_url=llm_config['api_base_url'],
    api_key=llm_config['api_key']
)


def call_openai(full_prompt: str) -> list:
    full_response = ""
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个帮助生成明朝科举-人物-职官-文献-地点知识图谱的助手，"
                        "需要从古籍文本中提取结构化数据并生成图数据库格式，"
                        "输出只能包含严格的 JSONL 数据，不能有任何额外信息。注意按行输出，一行为一条数据"
                    )
                },
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
            stream=True,
            timeout=TIMEOUT
        )
        for chunk in stream:
            if hasattr(chunk, "choices"):
                for choice in chunk.choices:
                    delta = getattr(choice, "delta", None)
                    if delta and getattr(delta, "content", None):
                        content = delta.content
                        if content.strip():
                            full_response += content
    except Exception as e:
        print(f"[ERROR] OpenAI 调用失败: {e}")
        return []
    cleaned = clean_stream_response(full_response)
    return extract_json_objs(cleaned)
