# prompt_templates.py
def build_prompt(text, ner_knowledge):
    return f"""
            请记住以下本体结构和示例格式，然后从给定的<文本>中提取对应信息（隐性属性如果推理不出则置为空），并将结果转换为图数据 JSONL 格式（按行输出，一行为一条数据）。
            文本：{text}。部分嵌套实体信息如下：{ner_knowledge}
            ---  

            ## 一、本体结构节点（Node） 
            0. **输入文本（Text）**
             - 示例格式：
                 ```jsonl
                 "Text": "这里填充的是输入的文本，这不允许为空"
            1. **人物（Person）**  
               - 属性：姓名、字、生年、户籍类型、户籍地、学籍、家庭排行、学术专长  
               - 示例格式：
                 ```jsonl
                 {{
                   "type": "node",
                   "label": "Person",
                   "properties": {{
                     "姓名": "",
                     "字": "",
                     "生年": "",
                     "户籍类型": "",
                     "户籍地": "",
                     "学籍": "",
                     "家庭排行": "",
                     "学术专长": ""
                   }}
                 }}
                 ```

            2. **职官（Official Position）**  
               - 属性：官职名称、官阶  
               - 示例格式：
                 ```jsonl
                 {{
                   "type": "node",
                   "label": "Official Position",
                   "properties": {{
                     "官职名称": "",
                     "官阶": ""
                   }}
                 }}
                 ```

            3. **科举（ImperialExam）**  
               - 属性：考试等级（童试／乡试／会试／殿试）、考试时间、主考官、科举政策  
               - 示例格式：
                 ```jsonl
                 {{
                   "type": "node",
                   "label": "ImperialExam",
                   "properties": {{
                     "考试等级": "",
                     "考试时间": "",
                     "考试地点": ""
                   }}
                 }}
                 ```

            4. **地点（Place）**  
               - 属性：历史名称、现代名称、区划层级、经纬坐标  
               - 示例格式：
                 ```jsonl
                 {{
                   "type": "node",
                   "label": "Place",
                   "properties": {{
                     "历史名称": "",
                     "现代名称": "",
                     "现代区划层级": "",
                     "经纬坐标": ""
                   }},
                   "text": ""
                 }}
                 ```

            ---  

            ## 二、对象属性关系（Relationship）  

            1. **参与（Person→ImperialExam）**  
               - 参与某场科举考试  
               - 属性：甲第等级、考中年龄、名次、功名  
               - 示例格式：
                 ```jsonl
                 {{
                   "type": "relationship",
                   "label": "参与",
                   "start": {{ "label": "Person", "姓名": "" }},
                   "end":   {{ "label": "ImperialExam", "考试等级": "" }},
                   "properties": {{
                     "甲第等级": "",
                     "考中年龄": "",
                     "名次": "",
                     "功名": ""
                   }},
                   "text": ""
                 }}
                 ```

            2. **担任（Person→Official Position）**  
               - 担任某个官职  
               - 属性：授予类型、迁转、授予时间  
               - 示例格式：
                 ```jsonl
                 {{
                   "type": "relationship",
                   "label": "担任",
                   "start": {{ "label": "Person", "姓名": "" }},
                   "end":   {{ "label": "Official Position", "官职名称": "" }},
                   "properties": {{
                     "授予类型": "一授／二授/...",
                     "迁转": "升／贬／调",
                     "授予时间": ""
                   }},
                   "text": ""
                 }}
                 ```


            3. **社会关系（Person↔Person）**  
               - 属性：关系类型  
               - 示例格式：
                 ```jsonl
                 {{
                   "type": "relationship",
                   "label": "社会关系",
                   "start": {{ "label": "Person", "姓名": "" }},
                   "end":   {{ "label": "Person", "姓名": "" }},
                   "properties": {{
                     "关系类型": ""
                   }},
                   "text": ""
                 }}
                 ```

            4. **生（Person→Place）**  
               - 出生与某地（一般户籍地通常为出生地）  
               - 示例格式：
                 ```jsonl
                 {{
                   "type": "relationship",
                   "label": "生",
                   "start": {{ "label": "Person", "姓名": "" }},
                   "end":   {{ "label": "Place", "历史名称": "" }},
                   "text": ""
                 }}
                 ```


            5. **任务执行（Person→Place）**  
               - 去某地执行任务  
               - 属性：职事名目、任务时间  
               - 示例格式：
                 ```jsonl
                 {{
                   "type": "relationship",
                   "label": "任务执行",
                   "start": {{ "label": "Person", "姓名": "" }},
                   "end":   {{ "label": "Place", "历史名称": "" }},
                   "properties": {{
                     "职事名目": "",
                     "任务时间": ""
                   }},
                   "text": ""
                 }}
                 ```

            6. **职任（Person→Place）**  
               - 在某地当官  
               - 属性：任职时间  
               - 示例格式：
                 ```jsonl
                 {{
                   "type": "relationship",
                   "label": "职任",
                   "start": {{ "label": "Person", "姓名": "" }},
                   "end":   {{ "label": "Place", "历史名称": "" }},
                   "properties":{{
                     "任职时间": ""
                   }},
                   "text": ""
                 }}
                 ```

            7.. **隶属（Place→Place）**  
               - 某地隶属某地   
               - 示例格式：
                 ```jsonl
                 {{
                   "type": "relationship",
                   "label": "隶属",
                   "start": {{ "label": "Person", "历史名称": "" }},
                   "end":   {{ "label": "Place", "历史名称": "" }},
                   "properties": {{
                   }},
                   "text": ""
                 }}
                 ```
            """
