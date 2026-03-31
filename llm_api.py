import requests
import re

# 你的配置信息
API_KEY = "sk-focjlzypldvwdzxqjaguwjxmhptctdnoevxtkgtnrpzzllgd"
API_BASE = "https://api.siliconflow.cn/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V3"

def generate_new_heuristic(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8, 
        "max_tokens": 2048 # 【改动1】增大token上限，因为我们要让它写大段的思考过程
    }
    
    try:
        response = requests.post(API_BASE, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # 【改动2】提取代码，同时返回完整的回复（包含 Thought）
        match = re.search(r'```python\n(.*?)\n```', content, re.DOTALL)
        code = match.group(1) if match else None
        
        return content, code
    except Exception as e:
        print(f"LLM API Error: {e}")
        return None, None