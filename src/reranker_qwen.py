"""Qwen-turbo重排序模块"""

import json
import logging
import os
from typing import List, Dict, Optional, Any
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

# 重试配置（模块级别，供装饰器使用）
retry_config = {
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=1, min=4, max=10)
}

class QwenTurboReranker:
    """Qwen-turbo重排序器"""

    def __init__(self, api_key: Optional[str] = None, model: str = "qwen-turbo"):
        self.model = model
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

        if not self.api_key:
            logger.warning("未提供API密钥，将使用模拟重排序")
            self.use_mock = True
        else:
            self.use_mock = False

    @retry(**retry_config)
    def rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int = 5) -> List[Dict[str, Any]]:
        """重排序文档"""
        if not documents:
            return []

        if len(documents) <= top_n:
            logger.debug(f"文档数量({len(documents)}) <= top_n({top_n})，无需重排序")
            return documents[:top_n]

        if self.use_mock:
            return self._mock_rerank(query, documents, top_n)

        try:
            # 构建重排序提示
            prompt = self._build_rerank_prompt(query, documents)

            # 调用API
            response = self._call_qwen_api(prompt)

            # 解析重排序结果
            ranked_indices = self._parse_rerank_response(response, len(documents))

            # 重排序文档
            reranked_docs = []
            for rank, idx in enumerate(ranked_indices[:top_n]):
                if 0 <= idx < len(documents):
                    doc = documents[idx].copy()
                    doc["rerank_score"] = len(ranked_indices) - rank  # 分数递减
                    doc["rerank_rank"] = rank + 1
                    doc["rerank_reason"] = response.get("reason", "")
                    reranked_docs.append(doc)

            logger.info(f"重排序完成: {len(documents)} -> {len(reranked_docs)}")
            return reranked_docs

        except Exception as e:
            logger.error(f"重排序失败: {e}")
            # 降级：返回原始顺序的前top_n个
            logger.warning("降级到原始顺序")
            return documents[:top_n]

    def _build_rerank_prompt(self, query: str, documents: List[Dict[str, Any]]) -> str:
        """构建重排序提示"""
        prompt = f"""你是一个专业的文档重排序助手。请根据以下查询的相关性对文档进行排序。

查询：{query}

文档列表：
"""

        for i, doc in enumerate(documents):
            # 提取文档内容（截取前500字符）
            content = doc.get("content", "")
            if len(content) > 500:
                content = content[:500] + "..."

            # 添加文档类型信息
            doc_type = doc.get("type", "text")
            source = doc.get("metadata", {}).get("source", "unknown")
            page_no = doc.get("page_no", "unknown")

            prompt += f"\n文档 {i}: [{doc_type} from {source} page {page_no}]\n{content}\n"

        prompt += """
排序要求：
1. 根据文档与查询的相关性从高到低排序
2. 考虑文档内容、类型和来源
3. 返回最相关的文档编号

请按以下JSON格式返回结果：
{
  "ranking": [doc_index_1, doc_index_2, ...],
  "reason": "排序理由说明"
}

注意：只返回文档编号，如 [2, 0, 1, 3] 表示文档2最相关，文档3最不相关。"""

        return prompt

    def _call_qwen_api(self, prompt: str) -> Dict[str, Any]:
        """调用Qwen-turbo API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        data = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个专业的文档相关性评估助手。请严格按照要求的JSON格式返回结果。"
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code != 200:
            raise Exception(f"API调用失败: {response.status_code} - {response.text}")

        result = response.json()
        content = result['choices'][0]['message']['content']

        # 解析JSON响应
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # 如果JSON解析失败，尝试修复
            logger.warning("JSON解析失败，尝试修复")
            return self._repair_json_response(content)

    def _parse_rerank_response(self, response: Dict[str, Any], total_docs: int) -> List[int]:
        """解析重排序响应"""
        ranking = response.get("ranking", [])

        if not ranking:
            logger.warning("未获取到排序结果，使用原始顺序")
            return list(range(total_docs))

        # 验证排序结果
        valid_ranking = []
        seen_indices = set()

        for idx in ranking:
            if isinstance(idx, int) and 0 <= idx < total_docs and idx not in seen_indices:
                valid_ranking.append(idx)
                seen_indices.add(idx)
            else:
                logger.warning(f"无效索引: {idx}")

        # 补充缺失的索引
        for i in range(total_docs):
            if i not in seen_indices:
                valid_ranking.append(i)

        return valid_ranking

    def _repair_json_response(self, content: str) -> Dict[str, Any]:
        """修复JSON响应"""
        # 简单的JSON修复逻辑
        try:
            # 尝试提取数组
            import re
            array_match = re.search(r'\[([\d,\s]+)\]', content)
            if array_match:
                array_str = array_match.group(1)
                indices = [int(x.strip()) for x in array_str.split(',') if x.strip().isdigit()]
                return {"ranking": indices}
        except Exception:
            pass

        # 如果无法修复，返回空排序
        logger.error("无法修复JSON响应")
        return {"ranking": []}

    def _mock_rerank(self, query: str, documents: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        """模拟重排序（用于测试）"""
        logger.info(f"使用模拟重排序: query='{query[:20]}...', docs={len(documents)}->{top_n}")

        # 简单的相关性评分模拟
        scored_docs = []
        for i, doc in enumerate(documents):
            content = doc.get("content", "")

            # 计算简单相关性分数
            score = 0.0
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())

            # 词匹配分数
            word_match = len(query_words.intersection(content_words))
            score += word_match * 10

            # 位置分数（标题更高）
            if doc.get("type") == "title":
                score += 5

            # 长度惩罚（避免太长或太短）
            content_len = len(content)
            if 100 < content_len < 1000:
                score += 2

            # 添加随机因素
            score += random.uniform(0, 5)

            doc_copy = doc.copy()
            doc_copy["rerank_score"] = score
            doc_copy["rerank_rank"] = i + 1
            doc_copy["rerank_reason"] = "模拟重排序"
            scored_docs.append(doc_copy)

        # 按分数排序
        scored_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored_docs[:top_n]

    def get_cost_estimate(self, document_count: int) -> Dict[str, Any]:
        """估算API调用成本"""
        # Qwen-turbo定价（需要更新为实际价格）
        input_tokens_per_doc = 500  # 估算每个文档的token数
        total_input_tokens = document_count * input_tokens_per_doc
        output_tokens = 200  # 估算输出token数

        # 估算价格（单位：元）
        input_cost = total_input_tokens * 0.002 / 1000  # 假设0.002元/1K tokens
        output_cost = output_tokens * 0.002 / 1000
        total_cost = input_cost + output_cost

        return {
            "input_tokens": total_input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost,
            "currency": "CNY"
        }

# 测试函数
def test_reranker():
    """测试重排序器"""
    reranker = QwenTurboReranker()  # 使用模拟模式

    # 测试文档
    documents = [
        {
            "id": "doc1",
            "type": "text",
            "content": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "page_no": 1,
            "metadata": {"source": "test.pdf"}
        },
        {
            "id": "doc2",
            "type": "title",
            "content": "第三章 人工智能的发展历程",
            "page_no": 2,
            "metadata": {"source": "test.pdf"}
        },
        {
            "id": "doc3",
            "type": "text",
            "content": "机器学习是人工智能的核心技术之一，它使计算机能够从数据中学习。",
            "page_no": 3,
            "metadata": {"source": "test.pdf"}
        },
        {
            "id": "doc4",
            "type": "table",
            "content": "| 年份 | AI发展里程碑 |\n|------|-------------|\n| 1956 | AI术语提出 |\n| 1997 | 深蓝击败卡斯帕罗夫 |",
            "page_no": 4,
            "metadata": {"source": "test.pdf"}
        }
    ]

    query = "人工智能的发展历程"

    print(f"查询: {query}")
    print(f"文档数量: {len(documents)}")

    # 重排序
    reranked_docs = reranker.rerank(query, documents, top_n=3)

    print("\n重排序结果:")
    for i, doc in enumerate(reranked_docs):
        print(f"{i+1}. [{doc['type']}] {doc['content'][:60]}...")
        print(f"   分数: {doc.get('rerank_score', 'N/A')}, 原排名: {doc.get('original_rank', 'N/A')}")

    # 成本估算
    cost = reranker.get_cost_estimate(len(documents))
    print(f"\n成本估算: {cost}")

    return reranked_docs

if __name__ == "__main__":
    test_reranker()
