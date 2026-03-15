#!/usr/bin/env python3
"""中文RAG系统主入口"""

import argparse
import logging
import sys
from pathlib import Path

# 将src目录添加到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from chinese_rag_pipeline import ChineseRAGPipeline

def setup_logging(level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/chinese_rag.log", encoding="utf-8")
        ]
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="中文RAG系统")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 索引命令
    index_parser = subparsers.add_parser("index", help="索引PDF文档")
    index_parser.add_argument("--pdf-dir", type=Path, default=Path("data/pdfs"),
                            help="PDF文档目录")
    index_parser.add_argument("--batch-size", type=int, default=32,
                            help="批处理大小")

    # 搜索命令
    search_parser = subparsers.add_parser("search", help="搜索文档")
    search_parser.add_argument("query", help="搜索查询")
    search_parser.add_argument("--top-k", type=int, default=10,
                             help="返回结果数量")
    search_parser.add_argument("--no-auto-tune", action="store_true",
                             help="禁用自动调节")
    search_parser.add_argument("--no-parent-retrieval", action="store_true",
                             help="禁用父文档检索")

    # 问答命令
    qa_parser = subparsers.add_parser("qa", help="问答模式")
    qa_parser.add_argument("question", help="问题")
    qa_parser.add_argument("--top-k", type=int, default=5,
                         help="检索结果数量")

    # 测试命令
    test_parser = subparsers.add_parser("test", help="运行测试")

    # 通用参数
    parser.add_argument("--log-level", default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                      help="日志级别")

    args = parser.parse_args()

    # 设置日志
    setup_logging(args.log_level)

    # 执行命令
    if args.command == "index":
        index_documents(args)
    elif args.command == "search":
        search_documents(args)
    elif args.command == "qa":
        question_answer(args)
    elif args.command == "test":
        run_tests()
    else:
        parser.print_help()

def index_documents(args):
    """索引文档"""
    print(f"开始索引文档目录: {args.pdf_dir}")

    with ChineseRAGPipeline() as rag:
        stats = rag.index_documents(args.pdf_dir, args.batch_size)

        print("\n索引完成!")
        print(f"处理PDF数量: {stats['total_pdfs']}")
        print(f"生成chunks数量: {stats['total_chunks']}")
        print(f"父文档数量: {stats['total_parents']}")
        if stats['errors']:
            print(f"错误数量: {len(stats['errors'])}")
            for error in stats['errors'][:5]:  # 显示前5个错误
                print(f"  - {error}")
        print(f"耗时: {stats['duration']}")

def search_documents(args):
    """搜索文档"""
    print(f"搜索: {args.query}")

    with ChineseRAGPipeline() as rag:
        results = rag.search(
            query=args.query,
            top_k=args.top_k,
            enable_auto_tune=not args.no_auto_tune,
            enable_parent_retrieval=not args.no_parent_retrieval
        )

        print(f"\n找到 {results['total']} 个结果")
        print(f"alpha值: {results['alpha']}")
        print(f"耗时: {results['duration']:.2f}秒")

        for i, result in enumerate(results['results']):
            print(f"\n{i+1}. [{result['type']}] {result['content'][:150]}...")
            print(f"   分数: {result.get('score', 0):.4f}")
            print(f"   来源: {result['metadata']['source']} 第{result['page_no']}页")
            if result.get('has_parent_context'):
                print(f"   父上下文: ✓")

        # 显示统计信息
        if 'stats' in results:
            stats = results['stats']
            print(f"\n统计信息:")
            print(f"  文档类型分布: {stats['doc_type_distribution']}")
            print(f"  来源数量: {stats['unique_sources']}")
            print(f"  页面范围: {stats['page_range']}")

def question_answer(args):
    """问答模式"""
    print(f"问题: {args.question}")

    with ChineseRAGPipeline() as rag:
        # 先搜索相关文档
        search_results = rag.search(args.question, top_k=args.top_k)

        if not search_results['results']:
            print("没有找到相关的文档来回答这个问题。")
            return

        # 生成答案
        answer = rag.answer_question(args.question, search_results['results'])

        print(f"\n答案:\n{answer}")

        # 显示参考的文档
        print("\n参考文档:")
        for i, doc in enumerate(search_results['results'][:3]):
            print(f"{i+1}. {doc['metadata']['source']} 第{doc['page_no']}页")

def run_tests():
    """运行测试"""
    print("运行系统测试...")

    # 运行各个模块的测试
    from src.pdf_parser_mineru import test_pdf_parser
    from src.embedding_bge import test_embedding
    from src.vector_store_milvus import test_vector_store
    from src.auto_tuner import test_auto_tuner
    from src.reranker_qwen import test_reranker
    from src.parent_document_retriever import test_parent_document_retriever

    print("\n1. 测试PDF解析器...")
    test_pdf_parser()

    print("\n2. 测试向量化模型...")
    test_embedding()

    print("\n3. 测试向量数据库...")
    test_vector_store()

    print("\n4. 测试自动调节器...")
    test_auto_tuner()

    print("\n5. 测试重排序器...")
    test_reranker()

    print("\n6. 测试父文档检索器...")
    test_parent_document_retriever()

    print("\n所有测试完成!")

if __name__ == "__main__":
    main()