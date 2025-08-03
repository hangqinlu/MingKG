import streamlit as st
from py2neo import Graph
import pandas as pd
import json
from openai import OpenAI
from pyvis.network import Network
import streamlit.components.v1 as components

llm_config = {
    "api_base_url": "https://tbnx.plus7.plus/v1",
    "api_key": "sk-e8DdamFXsM6jBn1MA5NTyUAvMDdsQLJnKLKfgItEz75GUj1Q"
}
MODEL_NAME = "deepseek-reasoner"
TIMEOUT = 1500
client = OpenAI(base_url=llm_config['api_base_url'], api_key=llm_config['api_key'])

@st.cache_resource
def get_graph():
    return Graph("bolt://localhost:7687", auth=("neo4j", "lhq18385458795"))
graph = get_graph()

@st.cache_data
def load_social_data():
    query = '''
        MATCH (p1:ImperialPerson)-[r:社会关系]->(p2:ImperialPerson)
        RETURN p1.姓名 AS source, p2.姓名 AS target, r.关系类型 AS relation_type
    '''
    data = [dict(record) for record in graph.run(query)]
    return pd.DataFrame(data)

def build_relation_prompt(df):
    prompt = (
        "以下为明清科举人物间的社会关系三元组（source, target, relation_type）。"
        "请你用社会网络分析的学术视角，进行如下任务：\n"
        "1. 自动归纳所有关系类型（如师生、亲属、同乡等），并分层说明其学术意义；\n"
        "2. 识别并输出所有主要社会团体（如核心家族、同年、同乡群体等）及其成员名单；\n"
        "3. 统计每类关系出现频次；\n"
        "4. 总结网络的核心结构特征（如中心人物、核心圈、重要桥梁等，按人名标注）；\n"
        "5. 输出可视化网络数据，节点为人物姓名，边为关系（含关系类型，建议输出JSON格式如下：{'nodes':[{'id':'A'},{'id':'B'},...],'edges':[{'source':'A','target':'B','relation':'同年'},...]})\n"
        "以下为数据：\n"
    )
    for _, row in df.iterrows():
        prompt += f"{row['source']}\t{row['relation_type']}\t{row['target']}\n"
    prompt += (
        "\n请按以上要求输出。网络部分务必用严格JSON结构输出，便于后续可视化处理。"
        "其余部分用条理分明的分点说明。"
    )
    return prompt

def call_openai(full_prompt: str) -> str:
    full_response = ""
    try:
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是中国明清社会网络知识归纳与结构分析专家，能够将输入的历史人物社会关系三元组自动分层归纳、团体发现、节点排名、"
                        "并输出便于网络可视化的标准JSON。"
                    )
                },
                {"role": "user", "content": full_prompt}
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
        st.error(f"[ERROR] OpenAI 调用失败: {e}")
        return ""
    return full_response

def run():
    st.header("科举人物社会关系网络可视化")
    df = load_social_data()
    if df.empty:
        st.error("数据库未返回有效社会关系数据！")
        st.stop()

    relation_stats = df['relation_type'].value_counts()
    person_stats = pd.concat([df['source'], df['target']]).value_counts()

    st.subheader("原始社会关系分布（关系类型Top10）")
    st.bar_chart(relation_stats.head(10))

    st.subheader("社交网络核心节点（出现在社会关系中的Top10人物）")
    st.bar_chart(person_stats.head(10))

    if st.button("一键归纳与社会网络可视化"):
        with st.spinner("正在智能归纳与网络分析，请稍候..."):
            prompt = build_relation_prompt(df)
            llm_response = call_openai(prompt)
            # 提取网络JSON（假定网络部分在 {...} 块中）
            try:
                network_json_str = llm_response.split("```json")[-1].split("```")[0].strip()
                network_data = json.loads(network_json_str)
                nodes = network_data["nodes"]
                edges = network_data["edges"]
            except Exception:
                st.error("未成功解析 LLM 返回的网络结构化数据。请检查LLM输出格式。")
                st.text_area("LLM原始返回内容", value=llm_response, height=300)
                st.stop()

            # 用 pyvis 可视化网络
            net = Network(height="800px", width="1200px", notebook=False, directed=True)
            for node in nodes:
                node_id = str(node["id"])
                freq = int(person_stats.get(node_id, 1))
                net.add_node(node_id, label=node_id, size=int(min(40, 10 + freq * 2)),
                             color="#FFD700" if freq > 5 else "#90caf9")
            for edge in edges:
                source = str(edge["source"])
                target = str(edge["target"])
                relation = str(edge.get("relation", ""))
                net.add_edge(source, target, label=relation, color="#bdbdbd")

            net.save_graph("social_network_llm.html")
            HtmlFile = open("social_network_llm.html", 'r', encoding='utf-8')
            components.html(HtmlFile.read(), height=820)

            st.subheader("LLM归纳内容（原文）")
            try:
                analysis_text = llm_response.split("```json")[0]
                st.text_area("社会网络学术归纳与结构点评", value=analysis_text.strip(), height=300)
            except:
                st.text_area("LLM原始返回内容", value=llm_response, height=300)
