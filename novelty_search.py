"""
Novelty Search 和相关算法实现
=============================
基于 "Why Greatness Cannot Be Planned?" 的思想
核心：探索不同的解决方法，而非只追求最优
"""

import ast
import re
from typing import List, Tuple, Dict, Set
from collections import Counter


class NoveltyCalculator:
    """新颖性计算器 - 衡量代码的行为多样性和结构差异"""
    
    def __init__(self):
        self.evaluated_programs: List[dict] = []  # 存储每个评估过的程序
        self.population_features: List[set] = []   # 存储特征集合
        
    def _extract_behavior_features(self, code: str) -> set:
        """提取代码行为特征 - 用于衡量新颖性"""
        features = set()
        
        # 1. 检测代码结构特征
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if hasattr(node.func, 'attr'):
                        features.add(f'call:{node.func.attr}')
                    elif hasattr(node.func, 'id'):
                        features.add(f'call:{node.func.id}')
                elif isinstance(node, ast.BinOp):
                    features.add(f'op:{type(node.op).__name__}')
                elif isinstance(node, ast.Compare):
                    features.add('compare')
                elif isinstance(node, ast.IfExp):
                    features.add('ternary')
        except:
            pass
        
        # 2. 检测关键概念/模式
        code_lower = code.lower()
        concept_patterns = [
            ('centroid', r'centroid'),
            ('norm', r'norm\(|np\.linalg'),
            ('mean', r'mean\(|average'),
            ('min', r'min\(|argmin'),
            ('max', r'max\(|argmax'),
            ('loop', r'for\s+\w+\s+in'),
            ('comprehension', r'\[.*for.*in'),
            ('lambda', r'lambda'),
            ('sort', r'sort\(|sorted\('),
            ('zip', r'zip\('),
        ]
        
        for concept, pattern in concept_patterns:
            if re.search(pattern, code_lower):
                features.add(concept)
        
        return features
    
    def _compute_novelty_score(self, code: str) -> float:
        """
        计算新颖性分数：与已有程序的距离
        距离越远（特征重叠越少），新颖性越高
        """
        if not self.evaluated_programs:
            return 1.0  # 第一个程序默认高新颖性
        
        current_features = self._extract_behavior_features(code)
        
        if not current_features:
            return 0.5  # 无法提取特征时返回中等新颖性
        
        # 计算与所有已有程序的最大距离（新颖性 = 最小距离的最大值）
        min_overlap = float('inf')
        
        for existing in self.evaluated_programs:
            existing_features = existing['features']
            if not existing_features:
                continue
            
            # Jaccard距离：1 - intersection/union
            intersection = current_features & existing_features
            union = current_features | existing_features
            jaccard_sim = len(intersection) / len(union) if union else 0
            jaccard_dist = 1 - jaccard_sim
            
            min_overlap = min(min_overlap, jaccard_sim)
        
        # 新颖性 = 1 - 最小相似度（与最相似已有程序的距离）
        novelty = 1 - min_overlap if min_overlap != float('inf') else 1.0
        return novelty
    
    def add_program(self, code: str, score_ratio: float):
        """添加已评估的程序到种群"""
        features = self._extract_behavior_features(code)
        self.evaluated_programs.append({
            'code': code,
            'score_ratio': score_ratio,
            'features': features,
            'novelty': self._compute_novelty_score(code)
        })
        self.population_features.append(features)
    
    def get_novelty_scores(self) -> List[float]:
        """获取所有程序的新颖性分数"""
        return [p['novelty'] for p in self.evaluated_programs]
    
    def get_best_novel_individuals(self, n: int = 5) -> List[dict]:
        """获取最佳新颖性个体（按新颖性排序）"""
        sorted_programs = sorted(
            self.evaluated_programs, 
            key=lambda x: x['novelty'], 
            reverse=True
        )
        return sorted_programs[:n]


class ParetoOptimizer:
    """Pareto优化器 - 多目标优化（性能和多样性）"""
    
    def __init__(self, novelty_weight: float = 0.3):
        """
        Args:
            novelty_weight: 新颖性权重 (0-1)，1表示完全按新颖性选择
        """
        self.novelty_weight = novelty_weight
        self.performance_weight = 1 - novelty_weight
        
    def _normalize(self, values: List[float]) -> List[float]:
        """Min-Max归一化"""
        min_val, max_val = min(values), max(values)
        if max_val == min_val:
            return [0.5] * len(values)
        return [(v - min_val) / (max_val - min_val) for v in values]
    
    def select_candidates(
        self, 
        programs: List[dict],
        top_k: int = 5
    ) -> List[dict]:
        """
        基于Pareto优化选择候选程序
        考虑性能和多样性的平衡
        
        Args:
            programs: 程序列表，每个包含 score_ratio 和 novelty
            top_k: 返回前k个
            
        Returns:
            选中的程序列表
        """
        if not programs:
            return []
        
        # 分离性能和多样性分数
        performances = [p['score_ratio'] for p in programs]
        novelties = [p['novelty'] for p in programs]
        
        # 归一化（注意：score_ratio越小越好，novelty越大越好）
        norm_perf = self._normalize(performances)
        norm_nov = self._normalize(novelties)
        
        # 性能归一化后需要反转（因为越小越好）
        # fitness = (1 - norm_perf) * performance_weight + norm_nov * novelty_weight
        fitness_scores = []
        for i, p in enumerate(programs):
            # 性能分数：1 - 归一化后的性能（这样越小性能越好，fitness越高）
            perf_score = (1 - norm_perf[i]) * self.performance_weight
            nov_score = norm_nov[i] * self.novelty_weight
            fitness_scores.append(perf_score + nov_score)
        
        # 按fitness排序，选择前k个
        indexed_scores = list(enumerate(fitness_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        for idx, score in indexed_scores[:top_k]:
            selected.append({
                'program': programs[idx],
                'fitness': score
            })
        
        return selected


class NoveltyDrivenSearch:
    """Novelty-driven搜索主类"""
    
    def __init__(
        self,
        novelty_weight: float = 0.4,
        archive_threshold: float = 0.6
    ):
        """
        Args:
            novelty_weight: 新颖性在选择中的权重
            archive_threshold: 存档阈值，低于此分数的程序也会被考虑存档
        """
        self.calculator = NoveltyCalculator()
        self.optimizer = ParetoOptimizer(novelty_weight=novelty_weight)
        self.archive_threshold = archive_threshold
        self.elite_archive: List[dict] = []  # 精英存档（好但不一定最新）
        
    def evaluate_and_select(
        self, 
        new_code: str, 
        score_ratio: float,
        top_k: int = 3
    ) -> Tuple[bool, dict]:
        """
        评估新程序并决定是否添加到种群
        
        Returns:
            (is_selected, selection_info)
        """
        # 1. 计算新颖性
        novelty = self.calculator._compute_novelty_score(new_code)
        
        program_info = {
            'code': new_code,
            'score_ratio': score_ratio,
            'novelty': novelty
        }
        
        # 2. Pareto选择
        all_programs = self.calculator.evaluated_programs + [program_info]
        candidates = self.optimizer.select_candidates(all_programs, top_k=top_k)
        
        # 3. 检查是否被选中
        selected_programs = [c['program'] for c in candidates] if candidates else []
        is_selected = program_info in selected_programs
        
        # 4. 即使没被选中，如果足够新颖也存入elite_archive
        if novelty > self.archive_threshold and score_ratio < 1.1:
            already_archived = False
            for archived in self.elite_archive:
                if self._codes_similar(archived['code'], new_code):
                    already_archived = True
                    break
            if not already_archived:
                self.elite_archive.append(program_info)
        
        selection_info = {
            'novelty': novelty,
            'fitness': candidates[0]['fitness'] if is_selected else 0,
            'candidates_count': len(candidates),
            'is_in_elite_archive': novelty > self.archive_threshold
        }
        
        # 5. 添加到calculator
        self.calculator.add_program(new_code, score_ratio)
        
        return is_selected, selection_info
    
    def _codes_similar(self, code1: str, code2: str, threshold: float = 0.8) -> bool:
        """检查两个代码是否相似"""
        feat1 = self.calculator._extract_behavior_features(code1)
        feat2 = self.calculator._extract_behavior_features(code2)
        
        if not feat1 or not feat2:
            return False
            
        intersection = feat1 & feat2
        union = feat1 | feat2
        similarity = len(intersection) / len(union) if union else 0
        
        return similarity >= threshold
    
    def get_diverse_population(self, n: int = 10) -> List[dict]:
        """获取多样化的种群（用于选择父亲程序）"""
        all_programs = self.calculator.evaluated_programs
        
        if len(all_programs) <= n:
            return all_programs
        
        # 按新颖性排序，选择最不同的组合
        sorted_by_novelty = sorted(all_programs, key=lambda x: x['novelty'], reverse=True)
        
        selected = [sorted_by_novelty[0]]
        remaining = sorted_by_novelty[1:]
        
        while len(selected) < n and remaining:
            # 选择与已选个体最不同的下一个
            best_candidate = None
            best_min_similarity = -1
            
            for candidate in remaining:
                min_sim_to_selected = float('inf')
                for selected_item in selected:
                    feat1 = candidate['features']
                    feat2 = selected_item['features']
                    if feat1 and feat2:
                        sim = len(feat1 & feat2) / len(feat1 | feat2)
                        min_sim_to_selected = min(min_sim_to_selected, sim)
                
                if min_sim_to_selected > best_min_similarity:
                    best_min_similarity = min_sim_to_selected
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return selected
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        programs = self.calculator.evaluated_programs
        return {
            'total_evaluated': len(programs),
            'unique_novelties': len(set(p['novelty'] for p in programs)),
            'elite_archive_size': len(self.elite_archive),
            'novelty_range': (
                min(p['novelty'] for p in programs) if programs else 0,
                max(p['novelty'] for p in programs) if programs else 0
            ),
            'performance_range': (
                min(p['score_ratio'] for p in programs) if programs else 0,
                max(p['score_ratio'] for p in programs) if programs else 0
            )
        }
