"""Milvus Lite 向量数据库存储和检索模块"""

import logging
from typing import List, Dict, Optional, Any
import numpy as np
from pathlib import Path

# Milvus Lite
from milvus import default_server
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection, utility
)

logger = logging.getLogger(__name__)


class MilvusVectorStore:
    """Milvus Lite 向量存储 - 纯向量检索版"""

    def __init__(self, collection_name: str = "ChineseDocuments",
                 db_path: str = "./milvus_data/milvus.db",
                 dim: int = 1024):
        """
        初始化 Milvus Lite 存储

        Args:
            collection_name: 集合名称
            db_path: 本地数据库文件路径
            dim: 向量维度 (BGE-M3 是 1024)
        """
        self.collection_name = collection_name
        self.db_path = Path(db_path)
        self.dim = dim

        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 启动 Milvus Lite 服务器
        self._start_server()

        # 创建或获取集合
        self._setup_collection()

        logger.info(f"Milvus Lite 初始化完成: {db_path}")

    def _start_server(self):
        """启动 Milvus Lite 本地服务器"""
        try:
            # 设置本地存储路径
            default_server.set_base_dir(str(self.db_path.parent))

            # 启动服务器（如果还没启动）
            if not default_server.running:
                default_server.start()
                logger.info("Milvus Lite 服务器已启动")

            # 连接
            connections.connect(host="127.0.0.1", port=default_server.listen_port)
            logger.info(f"已连接到 Milvus Lite: 127.0.0.1:{default_server.listen_port}")

        except Exception as e:
            logger.error(f"启动 Milvus Lite 失败: {e}")
            raise

    def _setup_collection(self):
        """设置文档集合"""
        try:
            # 检查集合是否存在
            if utility.has_collection(self.collection_name):
                logger.info(f"集合 {self.collection_name} 已存在")
                self.collection = Collection(self.collection_name)
                self.collection.load()
            else:
                logger.info(f"创建新集合: {self.collection_name}")
                self.collection = self._create_collection()

        except Exception as e:
            logger.error(f"设置集合失败: {e}")
            raise

    def _create_collection(self):
        """创建新集合"""
        try:
            # 定义字段
            # 注意：添加 doc_id 字段用于存储原始 chunk id，便于父文档检索
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=128),  # 原始 chunk id
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="content_type", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
                FieldSchema(name="page_no", dtype=DataType.INT64),
                FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
                FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=128),
                FieldSchema(name="child_count", dtype=DataType.INT64),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]

            schema = CollectionSchema(fields, description="中文文档向量库")

            collection = Collection(self.collection_name, schema)

            # 创建 IVF_FLAT 索引（适合中等数据量，查询快）
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="embedding", index_params=index_params)
            collection.load()

            logger.info(f"成功创建集合并建立索引: {self.collection_name}")
            return collection

        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """添加文档到向量数据库"""
        if not chunks or not embeddings:
            logger.warning("没有文档或 embedding 需要添加")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks数量({len(chunks)})与embeddings数量({len(embeddings)})不匹配")

        logger.info(f"添加 {len(chunks)} 个文档到 Milvus")

        try:
            # 准备数据
            entities = []
            for chunk, embedding in zip(chunks, embeddings):
                # 生成 doc_id（如果 chunk 没有 id）
                doc_id = chunk.get("id", f"{chunk['metadata']['source']}_{chunk['page_no']}_{chunk['type']}")

                entities.append({
                    "doc_id": doc_id,
                    "content": chunk["content"][:65535],  # 截断避免超长
                    "content_type": chunk["type"],
                    "doc_type": "parent" if chunk["type"] == "parent" else "child",
                    "page_no": chunk["page_no"],
                    "source": chunk["metadata"]["source"],
                    "parent_id": chunk["metadata"].get("parent_id", ""),
                    "child_count": chunk["metadata"].get("child_count", 0),
                    "embedding": embedding
                })

            # 批量插入 - 字段顺序必须与 schema 一致
            insert_data = [
                [e["doc_id"] for e in entities],
                [e["content"] for e in entities],
                [e["content_type"] for e in entities],
                [e["doc_type"] for e in entities],
                [e["page_no"] for e in entities],
                [e["source"] for e in entities],
                [e["parent_id"] for e in entities],
                [e["child_count"] for e in entities],
                [e["embedding"] for e in entities]
            ]

            self.collection.insert(insert_data)
            self.collection.flush()  # 确保持久化

            logger.info(f"成功添加 {len(chunks)} 个文档")

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def search(self, query_embedding: List[float], limit: int = 10,
               filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """向量检索"""
        logger.debug(f"向量检索: limit={limit}")

        try:
            # 搜索参数
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": 16}  # 搜索的聚类数
            }

            # 构建过滤表达式
            expr = None
            if filters:
                filter_parts = []
                if "source" in filters:
                    filter_parts.append(f'source == "{filters["source"]}"')
                if "doc_type" in filters:
                    filter_parts.append(f'doc_type == "{filters["doc_type"]}"')
                if filter_parts:
                    expr = " and ".join(filter_parts)

            # 执行搜索
            results = self.collection.search(
                data=[query_embedding],
                anns_field="embedding",
                param=search_params,
                limit=limit,
                expr=expr,
                output_fields=["doc_id", "content", "content_type", "doc_type", "page_no",
                               "source", "parent_id", "child_count"]
            )

            # 处理结果
            formatted_results = []
            for hits in results:
                for hit in hits:
                    result = {
                        "id": hit.entity.get("doc_id") or str(hit.id),
                        "content": hit.entity.get("content"),
                        "type": hit.entity.get("content_type"),
                        "doc_type": hit.entity.get("doc_type"),
                        "page_no": hit.entity.get("page_no"),
                        "source": hit.entity.get("source"),
                        "parent_id": hit.entity.get("parent_id"),
                        "child_count": hit.entity.get("child_count"),
                        "score": float(hit.distance),
                        "metadata": {
                            "score": float(hit.distance),
                            "source": hit.entity.get("source"),
                            "parent_id": hit.entity.get("parent_id"),
                            "retrieval_type": "vector_only"
                        }
                    }
                    formatted_results.append(result)

            logger.debug(f"检索到 {len(formatted_results)} 个结果")
            return formatted_results

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def hybrid_search(self, query: str, query_embedding: List[float], limit: int = 10,
                      alpha: Optional[float] = None, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """
        混合检索（目前退化为纯向量检索）

        注意：Milvus Lite 不支持内置全文检索，如需 BM25 需要额外实现
        """
        logger.debug(f"混合检索（当前退化为向量检索）: query='{query[:50]}...', limit={limit}")

        # 目前只做向量检索，后续可以在这里加 BM25 融合
        return self.search(query_embedding, limit=limit, filters=filters)

    def search_by_field(self, field: str, value: str) -> List[Dict[str, Any]]:
        """按字段精确搜索

        支持字段：id (int64), doc_id (varchar), parent_id (varchar) 等
        """
        try:
            # 构建查询表达式
            if field == "id":
                # 如果是 doc_id，用字符串匹配
                expr = f'doc_id == "{value}"'
            elif field in ["doc_id", "parent_id", "source", "content_type", "doc_type"]:
                # 字符串字段需要加引号
                expr = f'{field} == "{value}"'
            elif field in ["page_no", "child_count"]:
                # 数值字段
                expr = f'{field} == {value}'
            else:
                # 默认作为字符串处理
                expr = f'{field} == "{value}"'

            results = self.collection.query(
                expr=expr,
                output_fields=["doc_id", "content", "content_type", "doc_type", "page_no",
                               "source", "parent_id", "child_count"]
            )

            formatted_results = []
            for hit in results:
                result = {
                    "id": hit.get("doc_id") or str(hit.get("id")),
                    "content": hit.get("content"),
                    "type": hit.get("content_type"),
                    "doc_type": hit.get("doc_type"),
                    "page_no": hit.get("page_no"),
                    "source": hit.get("source"),
                    "parent_id": hit.get("parent_id"),
                    "child_count": hit.get("child_count"),
                    "score": 1.0,  # 精确匹配默认分数
                    "metadata": {
                        "score": 1.0,
                        "source": hit.get("source"),
                        "parent_id": hit.get("parent_id"),
                        "retrieval_type": "exact_match"
                    }
                }
                formatted_results.append(result)

            return formatted_results

        except Exception as e:
            logger.error(f"按字段搜索失败: {e}")
            return []

    def update_alpha(self, alpha: float) -> None:
        """更新混合检索权重（当前版本无实际作用）"""
        logger.info(f"设置 alpha 为 {alpha}（当前版本仅记录，不影响检索）")

    def get_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            stats = self.collection.num_entities
            return {
                "total_objects": stats,
                "collection_name": self.collection_name,
                "db_path": str(self.db_path)
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    def close(self):
        """关闭连接"""
        try:
            if hasattr(self, 'collection'):
                self.collection.release()
            connections.disconnect("default")
            logger.info("已关闭 Milvus 连接")
        except Exception as e:
            logger.warning(f"关闭连接时出错: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# 测试函数
def test_vector_store():
    """测试 Milvus 向量数据库"""
    import numpy as np

    # 创建测试数据
    test_chunks = [
        {
            "id": "parent_1",
            "type": "parent",
            "content": "第1页完整内容包含机器学习算法和深度学习技术",
            "page_no": 1,
            "metadata": {
                "source": "test.pdf",
                "parent_id": "",
                "child_count": 2
            }
        },
        {
            "id": "child_1",
            "type": "text",
            "content": "人工智能是计算机科学的一个分支",
            "page_no": 1,
            "metadata": {
                "source": "test.pdf",
                "parent_id": "parent_1",  # 指向父文档
                "child_count": 0
            }
        },
        {
            "id": "child_2",
            "type": "text",
            "content": "深度学习是机器学习的一种方法",
            "page_no": 1,
            "metadata": {
                "source": "test.pdf",
                "parent_id": "parent_1",  # 指向父文档
                "child_count": 0
            }
        }
    ]

    test_embeddings = [
        np.random.randn(1024).tolist(),  # parent
        np.random.randn(1024).tolist(),  # child_1
        np.random.randn(1024).tolist()   # child_2
    ]

    # 测试存储和检索
    store = MilvusVectorStore(
        collection_name="TestCollection",
        db_path="./test_milvus/milvus.db"
    )

    try:
        # 添加文档
        store.add_documents(test_chunks, test_embeddings)

        # 测试检索
        query_embedding = np.random.randn(1024).tolist()
        results = store.search(query_embedding, limit=5)

        print(f"检索到 {len(results)} 个结果")
        for i, result in enumerate(results):
            print(f"  {i+1}. {result['content'][:50]}... (score: {result['score']:.4f})")

        # 获取统计信息
        stats = store.get_stats()
        print(f"统计信息: {stats}")

    finally:
        store.close()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    test_vector_store()
