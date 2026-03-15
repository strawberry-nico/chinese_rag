"""父文档检索器 - 提供子文档检索+父文档上下文的混合检索能力"""

import logging
from typing import List, Dict, Optional, Any
from collections import OrderedDict

logger = logging.getLogger(__name__)

class ParentDocumentRetriever:
    """父文档检索器"""

    def __init__(self, vector_store, cache_size: int = 1000):
        """
        初始化父文档检索器

        Args:
            vector_store: 向量存储实例
            cache_size: LRU缓存大小
        """
        self.vector_store = vector_store
        self.cache_size = cache_size
        self.parent_cache = OrderedDict()  # LRU缓存
        self.logger = logging.getLogger(__name__)

    def retrieve_with_parent_context(self, query: str, query_embedding: List[float],
                                   top_k: int = 10, parent_top_k: int = 3,
                                   min_parent_context: int = 1) -> List[Dict[str, Any]]:
        """
        检索子文档并包含父文档上下文

        Args:
            query: 查询文本
            query_embedding: 查询向量
            top_k: 返回的子文档数量
            parent_top_k: 最多获取多少个父文档
            min_parent_context: 最少需要多少个父文档上下文

        Returns:
            包含父子文档信息的结果列表
        """
        self.logger.info(f"父文档检索: query='{query[:50]}...', top_k={top_k}")

        # 1. 先检索子文档（小块）
        initial_limit = max(top_k * 2, 20)  # 获取更多结果用于后续处理
        child_results = self.vector_store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            limit=initial_limit
        )

        if not child_results:
            self.logger.warning("未检索到任何子文档")
            return []

        self.logger.info(f"初始检索到 {len(child_results)} 个子文档")

        # 2. 收集相关的父文档
        parent_docs_map = {}
        child_with_parent = []

        for child in child_results:
            parent_id = child['metadata'].get('parent_id')

            if parent_id and parent_id not in parent_docs_map:
                # 获取父文档
                parent_doc = self._get_parent_document(parent_id)
                if parent_doc:
                    parent_docs_map[parent_id] = parent_doc
                    self.logger.debug(f"获取父文档: {parent_id}")

            # 构建包含父上下文的结果
            enriched_child = {
                **child,
                'parent_context': parent_docs_map.get(parent_id, {}),
                'retrieval_type': 'child_with_parent',
                'has_parent_context': bool(parent_id and parent_id in parent_docs_map)
            }
            child_with_parent.append(enriched_child)

        # 3. 如果父文档不足，直接检索父文档
        unique_parents = len(parent_docs_map)
        if unique_parents < min_parent_context:
            self.logger.info(f"父文档不足({unique_parents} < {min_parent_context})，补充检索父文档")
            additional_parents = self._search_parent_documents_directly(
                query, query_embedding, min_parent_context - unique_parents
            )

            # 将额外的父文档添加到映射中
            for parent in additional_parents:
                parent_id = parent['id']
                if parent_id not in parent_docs_map:
                    parent_docs_map[parent_id] = parent

        # 4. 重新排序和评分
        final_results = self._rerank_with_parent_context(child_with_parent, parent_docs_map)

        # 5. 返回最终结果
        final_results = final_results[:top_k]

        self.logger.info(f"返回最终结果: {len(final_results)} 个")

        # 记录统计信息
        stats = {
            'total_children': len(final_results),
            'unique_parents': len(set(r['parent_context'].get('id', '') for r in final_results if r['has_parent_context'])),
            'with_parent_context': sum(1 for r in final_results if r['has_parent_context'])
        }
        self.logger.info(f"检索统计: {stats}")

        return final_results

    def _get_parent_document(self, parent_id: str) -> Optional[Dict[str, Any]]:
        """获取父文档（带LRU缓存）"""
        if not parent_id:
            return None

        # 检查缓存
        if parent_id in self.parent_cache:
            # 移到末尾（最近使用）
            self.parent_cache.move_to_end(parent_id)
            self.logger.debug(f"从缓存获取父文档: {parent_id}")
            return self.parent_cache[parent_id]

        # 从数据库获取
        try:
            # 构建查询获取父文档
            # 注意：这里假设可以通过parent_id字段查询
            parent_results = self._search_by_field("id", parent_id)

            if parent_results:
                parent_doc = parent_results[0]

                # 添加到缓存
                self.parent_cache[parent_id] = parent_doc
                self.parent_cache.move_to_end(parent_id)

                # 清理最老的缓存
                if len(self.parent_cache) > self.cache_size:
                    removed = self.parent_cache.popitem(last=False)
                    self.logger.debug(f"清理缓存: {removed[0]}")

                self.logger.debug(f"从数据库获取并缓存父文档: {parent_id}")
                return parent_doc

        except Exception as e:
            self.logger.error(f"获取父文档失败: {e}")

        return None

    def _search_by_field(self, field: str, value: str) -> List[Dict[str, Any]]:
        """按字段搜索"""
        # 调用 vector_store 的 search_by_field 方法
        if hasattr(self.vector_store, 'search_by_field'):
            return self.vector_store.search_by_field(field, value)
        else:
            logger.warning(f"vector_store 不支持 search_by_field 方法")
            return []

    def _search_parent_documents_directly(self, query: str, query_embedding: List[float],
                                        limit: int) -> List[Dict[str, Any]]:
        """直接搜索父文档"""
        # 搜索类型为parent的文档
        # 在实际实现中，应该使用Weaviate的过滤器
        # 这里简化处理：搜索所有文档然后过滤

        all_results = self.vector_store.hybrid_search(
            query=query,
            query_embedding=query_embedding,
            limit=limit * 3  # 获取更多结果用于过滤
        )

        parent_results = []
        for result in all_results:
            if result.get("doc_type") == "parent":
                enriched_result = {
                    **result,
                    "retrieval_type": "parent_direct",
                    "parent_context": result  # 自己就是父文档
                }
                parent_results.append(enriched_result)

            if len(parent_results) >= limit:
                break

        self.logger.info(f"直接检索到 {len(parent_results)} 个父文档")
        return parent_results

    def _rerank_with_parent_context(self, child_results: List[Dict[str, Any]],
                                  parent_docs_map: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """使用父上下文重新评分"""
        scored_results = []

        for child in child_results:
            base_score = child.get("score", 0.0)
            parent_context = child.get("parent_context", {})

            # 计算增强分数
            enhanced_score = base_score

            # 如果有父上下文，增加分数
            if parent_context:
                # 简单的增强逻辑：基础分数 + 父文档分数的加权平均
                parent_score = parent_context.get("score", 0.0)
                parent_weight = 0.3  # 父文档权重
                enhanced_score = base_score * (1 - parent_weight) + parent_score * parent_weight

                # 添加父文档内容摘要
                parent_summary = self._summarize_parent_context(parent_context)
                child["parent_summary"] = parent_summary

            # 更新分数
            child["enhanced_score"] = enhanced_score
            child["score_boost"] = enhanced_score - base_score

            scored_results.append(child)

        # 按增强后的分数排序
        scored_results.sort(key=lambda x: x["enhanced_score"], reverse=True)

        return scored_results

    def _summarize_parent_context(self, parent_context: Dict[str, Any]) -> str:
        """摘要父文档上下文"""
        if not parent_context:
            return ""

        content = parent_context.get("content", "")
        source = parent_context.get("source", "")
        page_no = parent_context.get("page_no", "")

        # 截取前200字符作为摘要
        if len(content) > 200:
            summary = content[:200] + "..."
        else:
            summary = content

        return f"[来自{source}第{page_no}页] {summary}"

    def warm_cache(self, parent_ids: List[str]) -> None:
        """预热缓存"""
        self.logger.info(f"预热缓存: {len(parent_ids)} 个父文档")
        for parent_id in parent_ids:
            if parent_id and parent_id not in self.parent_cache:
                parent_doc = self._get_parent_document(parent_id)
                if parent_doc:
                    self.logger.debug(f"预热缓存: {parent_id}")

    def clear_cache(self) -> None:
        """清空缓存"""
        self.parent_cache.clear()
        self.logger.info("父文档缓存已清空")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return {
            "cache_size": len(self.parent_cache),
            "cache_capacity": self.cache_size,
            "cache_hit_rate": "N/A",  # 可以扩展实现
            "cache_entries": list(self.parent_cache.keys())
        }

# 测试函数
def test_parent_document_retriever():
    """测试父文档检索器"""
    # 模拟向量存储
    class MockVectorStore:
        def hybrid_search(self, query, query_embedding, limit):
            # 返回模拟的子文档结果
            return [
                {
                    "id": "child1",
                    "content": "子文档1内容",
                    "type": "text",
                    "score": 0.9,
                    "page_no": 1,
                    "metadata": {"parent_id": "parent1", "source": "test.pdf"}
                },
                {
                    "id": "child2",
                    "content": "子文档2内容",
                    "type": "table",
                    "score": 0.8,
                    "page_no": 2,
                    "metadata": {"parent_id": "parent1", "source": "test.pdf"}
                },
                {
                    "id": "child3",
                    "content": "子文档3内容",
                    "type": "text",
                    "score": 0.7,
                    "page_no": 3,
                    "metadata": {"parent_id": "parent2", "source": "test.pdf"}
                }
            ]

    # 创建检索器
    mock_store = MockVectorStore()
    retriever = ParentDocumentRetriever(mock_store)

    # 模拟父文档数据
    retriever.parent_cache = {
        "parent1": {
            "id": "parent1",
            "content": "第1页完整内容，包含多个段落和表格",
            "type": "parent",
            "page_no": 1,
            "score": 0.85,
            "metadata": {"source": "test.pdf", "page_range": [1, 1]}
        },
        "parent2": {
            "id": "parent2",
            "content": "第2页完整内容",
            "type": "parent",
            "page_no": 2,
            "score": 0.75,
            "metadata": {"source": "test.pdf", "page_range": [2, 2]}
        }
    }

    # 测试检索
    query = "测试查询"
    query_embedding = [0.1] * 1024  # 模拟向量

    results = retriever.retrieve_with_parent_context(
        query, query_embedding, top_k=5
    )

    print(f"检索结果数量: {len(results)}")
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['content']}")
        print(f"   类型: {result['type']}")
        print(f"   分数: {result.get('score', 0):.4f}")
        print(f"   增强分数: {result.get('enhanced_score', 0):.4f}")
        print(f"   有父上下文: {result.get('has_parent_context', False)}")
        if result.get('has_parent_context'):
            print(f"   父摘要: {result.get('parent_summary', 'N/A')}")

    # 测试缓存统计
    stats = retriever.get_cache_stats()
    print(f"\n缓存统计: {stats}")

    return results

if __name__ == "__main__":
    test_parent_document_retriever()""""file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/parent_document_retriever.py