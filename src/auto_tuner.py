"""混合检索自动调节器"""

import logging
import numpy as np
from typing import List, Dict, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)

class HybridSearchAutoTuner:
    """基于检索结果质量自动调节混合检索权重"""

    def __init__(self, window_size: int = 10, initial_alpha: float = 0.5,
                 variance_threshold_high: float = 0.3, variance_threshold_low: float = 0.1,
                 min_alpha: float = 0.2, max_alpha: float = 0.8, adjust_cooldown: int = 5):
        """
        初始化自动调节器

        Args:
            window_size: 性能历史窗口大小
            initial_alpha: 初始alpha值（向量权重）
            variance_threshold_high: 高方差阈值
            variance_threshold_low: 低方差阈值
            min_alpha: alpha最小值
            max_alpha: alpha最大值
            adjust_cooldown: 调节冷却期（查询次数）
        """
        self.window_size = window_size
        self.alpha = initial_alpha
        self.variance_threshold_high = variance_threshold_high
        self.variance_threshold_low = variance_threshold_low
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha
        self.adjust_cooldown = adjust_cooldown

        # 性能历史记录
        self.performance_history = deque(maxlen=window_size)
        self.alpha_history = deque(maxlen=window_size)

        # 调节状态
        self.query_count = 0
        self.last_adjust_query = 0

        self.logger = logging.getLogger(__name__)

        # 记录初始alpha
        self.alpha_history.append({
            'query_count': 0,
            'alpha': self.alpha,
            'reason': 'initial'
        })

    def record_performance(self, results: List[Dict[str, Any]]) -> None:
        """记录检索性能"""
        if not results:
            self.performance_history.append({
                'query_count': self.query_count,
                'variance': 0.0,
                'mean_score': 0.0,
                'result_count': 0,
                'score_range': 0.0,
                'timestamp': np.datetime64('now')
            })
            return

        # 提取分数
        scores = []
        for result in results:
            score = result.get('score', result.get('metadata', {}).get('score', 0.0))
            scores.append(float(score))

        if not scores:
            scores = [0.0] * len(results)

        # 计算统计指标
        scores_array = np.array(scores)
        variance = np.var(scores_array)
        mean_score = np.mean(scores_array)
        score_range = np.max(scores_array) - np.min(scores_array)

        performance = {
            'query_count': self.query_count,
            'variance': variance,
            'mean_score': mean_score,
            'result_count': len(results),
            'score_range': score_range,
            'timestamp': np.datetime64('now')
        }

        self.performance_history.append(performance)

        self.logger.debug(f"记录性能: variance={variance:.4f}, mean_score={mean_score:.4f}, "
                         f"count={len(results)}, range={score_range:.4f}")

    def should_adjust_alpha(self) -> bool:
        """判断是否应该调节alpha"""
        self.query_count += 1

        # 基础检查
        if len(self.performance_history) < 3:
            return False

        # 冷却期检查
        if self.query_count - self.last_adjust_query < self.adjust_cooldown:
            return False

        # 获取最近几次的性能
        recent_performances = list(self.performance_history)[-3:]
        recent_variances = [p['variance'] for p in recent_performances]

        # 检查方差的稳定性
        variance_of_variance = np.var(recent_variances)
        if variance_of_variance > 0.1:  # 方差本身变化很大，系统不稳定
            self.logger.debug(f"方差不稳定，不调参: variance_of_variance={variance_of_variance:.4f}")
            return False

        # 检查是否有足够的有效结果
        recent_counts = [p['result_count'] for p in recent_performances]
        if min(recent_counts) < 3:  # 结果太少，不稳定
            return False

        return True

    def adjust_alpha(self) -> float:
        """调节alpha值"""
        if not self.should_adjust_alpha():
            return self.alpha

        # 获取最近性能
        recent_performances = list(self.performance_history)[-3:]
        avg_variance = np.mean([p['variance'] for p in recent_performances])
        avg_mean_score = np.mean([p['mean_score'] for p in recent_performances])

        # 调节逻辑
        old_alpha = self.alpha
        adjustment_reason = ""

        # 方差过大：结果太分散，增加向量权重（更依赖语义）
        if avg_variance > self.variance_threshold_high:
            self.alpha = min(self.max_alpha, self.alpha + 0.1)
            adjustment_reason = f"high_variance({avg_variance:.3f})"

        # 方差过小：结果太集中，增加关键词权重
        elif avg_variance < self.variance_threshold_low:
            self.alpha = max(self.min_alpha, self.alpha - 0.1)
            adjustment_reason = f"low_variance({avg_variance:.3f})"

        # 记录调节历史
        if old_alpha != self.alpha:
            self.last_adjust_query = self.query_count
            self.alpha_history.append({
                'query_count': self.query_count,
                'old_alpha': old_alpha,
                'new_alpha': self.alpha,
                'reason': adjustment_reason,
                'avg_variance': avg_variance,
                'avg_mean_score': avg_mean_score
            })

            self.logger.info(f"调节alpha: {old_alpha} -> {self.alpha} ({adjustment_reason})")
        else:
            self.logger.debug(f"alpha保持不变: {self.alpha}")

        return self.alpha

    def get_recommendation(self) -> Dict[str, Any]:
        """获取调节建议"""
        if len(self.performance_history) < 5:
            return {
                'should_adjust': False,
                'reason': 'insufficient_data',
                'current_alpha': self.alpha
            }

        # 分析历史趋势
        recent_alphas = [h['new_alpha'] for h in list(self.alpha_history)[-5:]]
        alpha_trend = np.diff(recent_alphas).mean()

        # 分析性能趋势
        recent_scores = [p['mean_score'] for p in list(self.performance_history)[-5:]]
        score_trend = np.diff(recent_scores).mean()

        recommendation = {
            'current_alpha': self.alpha,
            'alpha_trend': alpha_trend,
            'score_trend': score_trend,
            'performance_summary': self.get_performance_summary()
        }

        # 提供建议
        if alpha_trend > 0.1 and score_trend < 0:
            recommendation['suggestion'] = "alpha持续增加但分数下降，可能需要反向调节"
            recommendation['recommended_alpha'] = max(self.min_alpha, self.alpha - 0.1)
        elif alpha_trend < -0.1 and score_trend < 0:
            recommendation['suggestion'] = "alpha持续减少但分数下降，可能需要增加向量权重"
            recommendation['recommended_alpha'] = min(self.max_alpha, self.alpha + 0.1)
        else:
            recommendation['suggestion'] = "当前参数设置合理"
            recommendation['recommended_alpha'] = self.alpha

        return recommendation

    def get_performance_summary(self) -> Dict[str, float]:
        """获取性能摘要"""
        if not self.performance_history:
            return {}

        variances = [p['variance'] for p in self.performance_history]
        mean_scores = [p['mean_score'] for p in self.performance_history]
        result_counts = [p['result_count'] for p in self.performance_history]

        return {
            'avg_variance': np.mean(variances),
            'avg_mean_score': np.mean(mean_scores),
            'avg_result_count': np.mean(result_counts),
            'variance_std': np.std(variances),
            'total_queries': self.query_count,
            'adjustment_count': len([h for h in self.alpha_history if 'old_alpha' in h])
        }

    def reset(self):
        """重置调节器状态"""
        self.performance_history.clear()
        self.alpha_history.clear()
        self.query_count = 0
        self.last_adjust_query = 0
        self.logger.info("自动调节器已重置")

    def get_current_alpha(self) -> float:
        """获取当前alpha值"""
        return self.alpha

    def export_history(self) -> Dict[str, List]:
        """导出历史数据用于分析"""
        return {
            'performance_history': list(self.performance_history),
            'alpha_history': list(self.alpha_history),
            'current_alpha': self.alpha,
            'total_queries': self.query_count
        }

# 测试函数
def test_auto_tuner():
    """测试自动调节器"""
    import random

    tuner = HybridSearchAutoTuner(
        window_size=5,
        initial_alpha=0.5,
        adjust_cooldown=2  # 测试时缩短冷却期
    )

    print("测试自动调节器...")

    # 模拟不同的检索结果
    test_cases = [
        # 高方差情况（结果分散）
        [{'score': 0.9}, {'score': 0.6}, {'score': 0.3}, {'score': 0.1}],
        # 低方差情况（结果集中）
        [{'score': 0.8}, {'score': 0.79}, {'score': 0.81}, {'score': 0.78}],
        # 中等方差
        [{'score': 0.85}, {'score': 0.7}, {'score': 0.6}, {'score': 0.5}],
    ]

    for i, results in enumerate(test_cases * 3):  # 重复测试
        print(f"\n查询 {i+1}:")

        # 记录性能
        tuner.record_performance(results)

        # 调节alpha
        new_alpha = tuner.adjust_alpha()

        # 获取建议
        if i >= 5:
            recommendation = tuner.get_recommendation()
            print(f"  当前alpha: {new_alpha}")
            print(f"  建议: {recommendation.get('suggestion', 'N/A')}")

    # 最终统计
    summary = tuner.get_performance_summary()
    print(f"\n最终统计: {summary}")

    return tuner

if __name__ == "__main__":
    test_auto_tuner()
