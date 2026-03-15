"""中文RAG系统主流程"""

import logging
import os
from pathlib import Path
from typing import List, Dict, Optional, Any
import json
from datetime import datetime

# 导入各个模块
from src.config import config
from src.pdf_parser_mineru import MinerUPDFParser, ParsedDocument
from src.embedding_bge import BGEM3Embedding
from src.vector_store_milvus import MilvusVectorStore
from src.auto_tuner import HybridSearchAutoTuner
from src.reranker_qwen import QwenTurboReranker
from src.parent_document_retriever import ParentDocumentRetriever

logger = logging.getLogger(__name__)

class ChineseRAGPipeline:
    """中文RAG主流程"""

    def __init__(self, config_override: Optional[Dict] = None):
        """初始化RAG管道"""
        self.logger = logging.getLogger(__name__)
        self.logger.info("初始化中文RAG管道...")

        # 合并配置
        self.config = self._merge_config(config_override)

        # 初始化各个组件
        self._initialize_components()

        self.logger.info("中文RAG管道初始化完成")

    def _merge_config(self, config_override: Optional[Dict]) -> Dict:
        """合并配置"""
        base_config = {
            'pdf_parser': config.pdf_parser,
            'embedding': config.embedding,
            'vector_store': config.vector_store,
            'retrieval': config.retrieval,
            'reranker': config.reranker,
            'auto_tuner': config.auto_tuner,
            'gpu': config.gpu
        }

        if config_override:
            # 深度合并配置
            base_config = self._deep_merge(base_config, config_override)

        return base_config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """深度合并字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def _initialize_components(self):
        """初始化各个组件"""
        # 1. PDF解析器
        self.pdf_parser = MinerUPDFParser(
            device=self.config['pdf_parser']['device']
        )
        self.logger.info("✓ PDF解析器初始化完成")

        # 2. 向量化模型
        self.embedder = BGEM3Embedding(
            model_name=self.config['embedding']['model_name'],
            device=self.config['embedding']['device']
        )
        self.logger.info("✓ 向量化模型初始化完成")

        # 3. 向量数据库
        self.vector_store = MilvusVectorStore(
            collection_name=self.config['vector_store']['collection_name'],
            db_path=self.config['vector_store']['db_path'],
            dim=self.config['vector_store'].get('dim', 1024)
        )
        self.logger.info("✓ 向量数据库初始化完成")

        # 4. 自动调节器
        self.auto_tuner = HybridSearchAutoTuner(
            window_size=self.config['auto_tuner']['window_size'],
            initial_alpha=self.config['vector_store']['initial_alpha'],
            variance_threshold_high=self.config['auto_tuner']['variance_threshold_high'],
            variance_threshold_low=self.config['auto_tuner']['variance_threshold_low'],
            min_alpha=self.config['auto_tuner']['min_alpha'],
            max_alpha=self.config['auto_tuner']['max_alpha'],
            adjust_cooldown=self.config['auto_tuner']['adjust_cooldown']
        )
        self.logger.info("✓ 自动调节器初始化完成")

        # 5. 重排序器
        self.reranker = QwenTurboReranker(
            api_key=self.config['reranker']['api_key'],
            model=self.config['reranker']['model']
        )
        self.logger.info("✓ 重排序器初始化完成")

        # 6. 父文档检索器
        self.parent_retriever = ParentDocumentRetriever(
            vector_store=self.vector_store,
            cache_size=1000
        )
        self.logger.info("✓ 父文档检索器初始化完成")

    def index_documents(self, pdf_dir: Path, batch_size: int = 32) -> Dict[str, Any]:
        """索引文档目录"""
        self.logger.info(f"开始索引文档目录: {pdf_dir}")

        stats = {
            'total_pdfs': 0,
            'total_chunks': 0,
            'total_parents': 0,
            'errors': [],
            'start_time': datetime.now().isoformat()
        }

        if not pdf_dir.exists():
            raise ValueError(f"PDF目录不存在: {pdf_dir}")

        pdf_files = list(pdf_dir.glob("*.pdf"))
        self.logger.info(f"找到 {len(pdf_files)} 个PDF文件")

        for i, pdf_file in enumerate(pdf_files):
            try:
                self.logger.info(f"处理PDF {i+1}/{len(pdf_files)}: {pdf_file.name}")

                # 1. 解析PDF
                parsed_doc = self.pdf_parser.parse_pdf(pdf_file)
                stats['total_pdfs'] += 1

                # 2. 生成embedding
                chunks = parsed_doc.all_chunks
                if not chunks:
                    self.logger.warning(f"PDF {pdf_file.name} 没有解析到任何内容")
                    continue

                texts = [chunk["content"] for chunk in chunks]
                embeddings = self.embedder.encode(texts, batch_size=batch_size)

                # 3. 存储到向量数据库
                self.vector_store.add_documents(chunks, embeddings)

                stats['total_chunks'] += len(chunks)
                stats['total_parents'] += sum(1 for c in chunks if c["type"] == "parent")

                self.logger.info(f"完成 {pdf_file.name}: {len(chunks)} 个chunks")

            except Exception as e:
                error_msg = f"处理PDF失败 {pdf_file.name}: {e}"
                self.logger.error(error_msg)
                stats['errors'].append(error_msg)

        stats['end_time'] = datetime.now().isoformat()
        stats['duration'] = str(datetime.fromisoformat(stats['end_time']) -
                               datetime.fromisoformat(stats['start_time']))

        self.logger.info(f"索引完成: {stats}")
        return stats

    def search(self, query: str, top_k: int = 10, enable_auto_tune: bool = True,
               enable_parent_retrieval: bool = True) -> Dict[str, Any]:
        """搜索文档"""
        self.logger.info(f"搜索: query='{query[:50]}...', top_k={top_k}")

        start_time = datetime.now()

        # 1. 生成查询embedding
        query_embedding = self.embedder.encode_queries([query])[0]

        # 2. 获取当前alpha值
        current_alpha = self.auto_tuner.get_current_alpha()
        self.vector_store.update_alpha(current_alpha)

        # 3. 检索
        if enable_parent_retrieval:
            # 使用父文档检索
            retrieval_results = self.parent_retriever.retrieve_with_parent_context(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k * 2,  # 获取更多结果用于重排序
                parent_top_k=3
            )
        else:
            # 普通混合检索
            retrieval_results = self.vector_store.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                limit=top_k * 2
            )

        if not retrieval_results:
            self.logger.warning("未检索到任何结果")
            return {
                "query": query,
                "results": [],
                "total": 0,
                "alpha": current_alpha,
                "auto_tuned": False,
                "parent_retrieval": enable_parent_retrieval
            }

        self.logger.info(f"检索到 {len(retrieval_results)} 个初始结果")

        # 4. 记录性能（用于自动调节）
        self.auto_tuner.record_performance(retrieval_results)

        # 5. 重排序
        reranked_results = self.reranker.rerank(
            query=query,
            documents=retrieval_results,
            top_n=top_k
        )

        # 6. 自动调节alpha
        if enable_auto_tune:
            new_alpha = self.auto_tuner.adjust_alpha()
            if new_alpha != current_alpha:
                self.logger.info(f"alpha已调节: {current_alpha} -> {new_alpha}")

        # 7. 构建最终响应
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        response = {
            "query": query,
            "results": reranked_results,
            "total": len(reranked_results),
            "alpha": self.auto_tuner.get_current_alpha(),
            "auto_tuned": enable_auto_tune,
            "parent_retrieval": enable_parent_retrieval,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }

        # 添加统计信息
        response["stats"] = self._calculate_search_stats(reranked_results)

        self.logger.info(f"搜索完成: {len(reranked_results)} 个结果, 耗时: {duration:.2f}s")

        return response

    def _calculate_search_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算搜索统计信息"""
        if not results:
            return {}

        # 基本统计
        doc_types = {}
        sources = set()
        pages = set()

        for result in results:
            doc_type = result.get("type", "unknown")
            doc_types[doc_type] = doc_types.get(doc_type, 0) + 1

            source = result.get("metadata", {}).get("source", "unknown")
            sources.add(source)

            page_no = result.get("page_no")
            if page_no is not None:
                pages.add(page_no)

        return {
            "doc_type_distribution": doc_types,
            "unique_sources": len(sources),
            "source_list": list(sources),
            "page_range": [min(pages), max(pages)] if pages else [],
            "with_parent_context": sum(1 for r in results if r.get("has_parent_context", False))
        }

    def answer_question(self, question: str, context_docs: List[Dict[str, Any]],
                       max_context_length: int = 3000) -> str:
        """基于上下文回答问题"""
        self.logger.info(f"生成答案: question='{question[:50]}...'")

        if not context_docs:
            return "抱歉，没有找到相关的上下文信息来回答这个问题。"

        # 构建上下文
        context_parts = []
        total_length = 0

        for doc in context_docs[:3]:  # 最多使用3个文档
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "unknown")
            page_no = doc.get("page_no", "unknown")

            # 截取内容
            if len(content) > 1000:
                content = content[:1000] + "..."

            part = f"[来自{source}第{page_no}页]\n{content}"

            if total_length + len(part) > max_context_length:
                break

            context_parts.append(part)
            total_length += len(part)

        context = "\n\n".join(context_parts)

        # 构建提示
        prompt = f"""基于以下上下文回答问题：

上下文：
{context}

问题：{question}

要求：
1. 基于提供的上下文准确回答问题
2. 如果上下文信息不足以回答问题，请明确说明
3. 回答要简洁明了，避免冗余信息
4. 可以引用具体的页码和文档来源

答案："""

        # 调用Qwen-turbo生成答案
        try:
            # 使用重排序器的API客户端
            response = self.reranker.client.chat.completions.create(
                model="qwen-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "你是一个专业的问答助手。请基于提供的上下文准确回答问题。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )

            answer = response.choices[0].message.content
            self.logger.info("答案生成完成")
            return answer

        except Exception as e:
            self.logger.error(f"生成答案失败: {e}")
            return "抱歉，生成答案时出现了错误。"

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        return {
            "vector_store": self.vector_store.get_stats(),
            "auto_tuner": {
                "current_alpha": self.auto_tuner.get_current_alpha(),
                "performance_summary": self.auto_tuner.get_performance_summary()
            },
            "parent_retriever": self.parent_retriever.get_cache_stats()
        }

    def export_search_history(self, filepath: str) -> None:
        """导出搜索历史（包括自动调节历史）"""
        history = self.auto_tuner.export_history()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        self.logger.info(f"搜索历史已导出到: {filepath}")

    def close(self):
        """关闭系统"""
        self.logger.info("正在关闭RAG系统...")
        if hasattr(self, 'vector_store'):
            self.vector_store.close()
        self.logger.info("RAG系统已关闭")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# 测试函数
def test_chinese_rag_pipeline():
    """测试完整的RAG管道"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("测试中文RAG管道...")

    # 创建RAG管道（使用模拟模式）
    with ChineseRAGPipeline() as rag:
        # 测试搜索
        query = "什么是人工智能？"
        results = rag.search(query, top_k=3)

        print(f"\n查询: {query}")
        print(f"检索到 {results['total']} 个结果")
        print(f"alpha值: {results['alpha']}")
        print(f"耗时: {results['duration']:.2f}s")

        for i, result in enumerate(results['results']):
            print(f"\n{i+1}. [{result['type']}] {result['content'][:100]}...")
            print(f"   分数: {result.get('score', 0):.4f}")
            print(f"   页码: {result.get('page_no', 'N/A')}")
            if result.get('has_parent_context'):
                print(f"   有父上下文: ✓")

        # 测试问答
        if results['results']:
            answer = rag.answer_question(query, results['results'])
            print(f"\n生成的答案:\n{answer}")

        # 获取统计信息
        stats = rag.get_stats()
        print(f"\n系统统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")

if __name__ == "__main__":
    test_chinese_rag_pipeline()