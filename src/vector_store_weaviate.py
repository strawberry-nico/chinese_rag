"""Weaviate向量数据库存储和检索模块"""

import logging
from typing import List, Dict, Optional, Any
import weaviate
from weaviate.classes.config import Configure, Property, DataType
from weaviate.classes.query import MetadataQuery
import numpy as np

logger = logging.getLogger(__name__)

class WeaviateHybridStore:
    """Weaviate混合检索存储"""

    def __init__(self, collection_name: str = "ChineseDocuments", alpha: float = 0.5,
                 host: str = "localhost", port: int = 8080, grpc_port: int = 50051):
        self.collection_name = collection_name
        self.alpha = alpha  # 混合检索权重
        self.host = host
        self.port = port
        self.grpc_port = grpc_port

        # 连接到Weaviate
        self._connect()

        # 创建或获取集合
        self._setup_collection()

    def _connect(self):
        """连接到Weaviate"""
        try:
            self.client = weaviate.connect_to_local(
                host=self.host,
                port=self.port,
                grpc_port=self.grpc_port,
                headers={}  # 可以添加认证头
            )

            # 测试连接
            if self.client.is_ready():
                logger.info(f"成功连接到Weaviate: {self.host}:{self.port}")
            else:
                raise ConnectionError("Weaviate服务未就绪")

        except Exception as e:
            logger.error(f"连接Weaviate失败: {e}")
            raise

    def _setup_collection(self):
        """设置文档集合"""
        try:
            # 检查集合是否存在
            if self.client.collections.exists(self.collection_name):
                logger.info(f"集合 {self.collection_name} 已存在")
                self.collection = self.client.collections.get(self.collection_name)
            else:
                logger.info(f"创建新集合: {self.collection_name}")
                self.collection = self._create_collection()

        except Exception as e:
            logger.error(f"设置集合失败: {e}")
            raise

    def _create_collection(self):
        """创建新集合"""
        try:
            return self.client.collections.create(
                name=self.collection_name,
                vectorizer_config=Configure.Vectorizer.none(),  # 手动提供向量
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="content_type", data_type=DataType.TEXT),  # text, table, title, parent
                    Property(name="doc_type", data_type=DataType.TEXT),     # child, parent
                    Property(name="page_no", data_type=DataType.INT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="parent_id", data_type=DataType.TEXT),    # 父文档ID
                    Property(name="child_count", data_type=DataType.INT),   # 子文档数量
                    Property(name="bbox", data_type=DataType.TEXT_ARRAY),   # 边界框
                ],
                # 向量索引配置
                vector_index_config=Configure.VectorIndex.hnsw(
                    distance_metric="cosine",
                    ef_construction=128,
                    max_connections=64
                ),
                # 倒排索引配置（用于BM25）
                inverted_index_config=Configure.inverted_index(
                    bm25_b=0.7,
                    bm25_k1=1.25
                )
            )
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            raise

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """添加文档到向量数据库"""
        if not chunks or not embeddings:
            logger.warning("没有文档或embedding需要添加")
            return

        if len(chunks) != len(embeddings):
            raise ValueError(f"chunks数量({len(chunks)})与embeddings数量({len(embeddings)})不匹配")

        logger.info(f"添加 {len(chunks)} 个文档到向量数据库")

        try:
            with self.collection.batch.dynamic() as batch:
                for chunk, embedding in zip(chunks, embeddings):
                    # 准备属性数据
                    properties = {
                        "content": chunk["content"],
                        "content_type": chunk["type"],
                        "doc_type": "parent" if chunk["type"] == "parent" else "child",
                        "page_no": chunk["page_no"],
                        "source": chunk["metadata"]["source"],
                        "parent_id": chunk["metadata"].get("parent_id", ""),
                        "child_count": chunk["metadata"].get("child_count", 0),
                        "bbox": [str(x) for x in chunk["metadata"].get("bbox", [])]
                    }

                    # 添加对象
                    batch.add_object(
                        properties=properties,
                        vector=embedding
                    )

            logger.info(f"成功添加 {len(chunks)} 个文档")

        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            # 检查是否有失败的批次
            if self.collection.batch.failed_objects:
                logger.error(f"失败对象数量: {len(self.collection.batch.failed_objects)}")
            raise

    def hybrid_search(self, query: str, query_embedding: List[float], limit: int = 10,
                     alpha: Optional[float] = None, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """混合检索"""
        if alpha is None:
            alpha = self.alpha

        logger.debug(f"混合检索: query='{query[:50]}...', alpha={alpha}, limit={limit}")

        try:
            # 构建查询
            query_obj = self.collection.query.hybrid(
                query=query,
                vector=query_embedding,
                alpha=alpha,  # 0=纯BM25, 1=纯向量
                limit=limit,
                return_metadata=MetadataQuery(score=True, explain_score=True)
            )

            # 如果有过滤器，需要添加
            if filters:
                # TODO: 添加过滤器支持
                pass

            # 处理结果
            results = []
            for obj in query_obj.objects:
                result = {
                    "id": str(obj.uuid),
                    "content": obj.properties["content"],
                    "type": obj.properties["content_type"],
                    "doc_type": obj.properties["doc_type"],
                    "page_no": obj.properties["page_no"],
                    "source": obj.properties["source"],
                    "parent_id": obj.properties["parent_id"],
                    "child_count": obj.properties["child_count"],
                    "bbox": [float(x) for x in obj.properties.get("bbox", [])],
                    "score": obj.metadata.score,
                    "metadata": {
                        "score": obj.metadata.score,
                        "explain_score": getattr(obj.metadata, "explain_score", ""),
                        "vector_distance": getattr(obj.metadata, "distance", None)
                    }
                }
                results.append(result)

            logger.debug(f"检索到 {len(results)} 个结果")
            return results

        except Exception as e:
            logger.error(f"混合检索失败: {e}")
            return []

    def vector_search_only(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """纯向量检索（降级方案）"""
        logger.debug(f"纯向量检索: limit={limit}")

        try:
            query_obj = self.collection.query.near_vector(
                near_vector=query_embedding,
                limit=limit,
                return_metadata=MetadataQuery(score=True)
            )

            results = []
            for obj in query_obj.objects:
                result = {
                    "id": str(obj.uuid),
                    "content": obj.properties["content"],
                    "type": obj.properties["content_type"],
                    "doc_type": obj.properties["doc_type"],
                    "page_no": obj.properties["page_no"],
                    "source": obj.properties["source"],
                    "parent_id": obj.properties["parent_id"],
                    "child_count": obj.properties["child_count"],
                    "bbox": [float(x) for x in obj.properties.get("bbox", [])],
                    "score": obj.metadata.score,
                    "metadata": {
                        "score": obj.metadata.score,
                        "retrieval_type": "vector_only"
                    }
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            return []

    def search_by_id(self, doc_id: str) -> Optional[List[Dict[str, Any]]]:
        """根据ID搜索文档"""
        try:
            # 注意：Weaviate使用UUID，这里假设doc_id存储在某个字段中
            # 实际实现可能需要调整
            query_obj = self.collection.query.fetch_objects(
                filters={
                    "operator": "Equal",
                    "path": ["id"],
                    "valueText": doc_id
                }
            )

            results = []
            for obj in query_obj.objects:
                result = {
                    "id": str(obj.uuid),
                    "content": obj.properties["content"],
                    "type": obj.properties["content_type"],
                    "doc_type": obj.properties["doc_type"],
                    "page_no": obj.properties["page_no"],
                    "source": obj.properties["source"],
                    "parent_id": obj.properties["parent_id"],
                    "child_count": obj.properties["child_count"],
                    "bbox": [float(x) for x in obj.properties.get("bbox", [])]
                }
                results.append(result)

            return results if results else None

        except Exception as e:
            logger.error(f"按ID搜索失败: {e}")
            return None

    def update_alpha(self, alpha: float) -> None:
        """更新混合检索权重"""
        self.alpha = alpha
        logger.info(f"更新alpha值为: {alpha}")

    def get_stats(self) -> Dict[str, Any]:
        """获取集合统计信息"""
        try:
            stats = self.collection.aggregate.over_all(
                total_count=True
            )

            return {
                "total_objects": stats.total_count,
                "collection_name": self.collection_name,
                "alpha": self.alpha
            }

        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {}

    def close(self):
        """关闭连接"""
        if self.client:
            self.client.close()
            logger.info("已关闭Weaviate连接")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# 测试函数
def test_vector_store():
    """测试向量数据库"""
    import numpy as np

    # 创建测试数据
    test_chunks = [
        {
            "id": "test_1",
            "type": "text",
            "content": "人工智能是计算机科学的一个分支",
            "page_no": 1,
            "metadata": {
                "source": "test.pdf",
                "parent_id": "",
                "child_count": 0,
                "bbox": []
            }
        },
        {
            "id": "test_2",
            "type": "parent",
            "content": "第1页完整内容",
            "page_no": 1,
            "metadata": {
                "source": "test.pdf",
                "parent_id": "",
                "child_count": 2,
                "bbox": []
            }
        }
    ]

    test_embeddings = [
        np.random.randn(1024).tolist(),
        np.random.randn(1024).tolist()
    ]

    # 测试存储和检索
    with WeaviateHybridStore(collection_name="TestCollection") as store:
        # 添加文档
        store.add_documents(test_chunks, test_embeddings)

        # 测试检索
        query = "人工智能"
        query_embedding = np.random.randn(1024).tolist()

        results = store.hybrid_search(query, query_embedding, limit=5)
        print(f"检索到 {len(results)} 个结果")

        for i, result in enumerate(results):
            print(f"  {i+1}. {result['content'][:50]}... (score: {result['score']:.4f})")

        # 获取统计信息
        stats = store.get_stats()
        print(f"统计信息: {stats}")

if __name__ == "__main__":
    test_vector_store()""""file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''"file_path":"/Users/nico/Downloads/8-项目实战：企业知识库/RAG-Challenge-2-main/chinese_rag/src/vector_store_weaviate.py"}''}''