"""
Population管理增强模块
=======================
功能：
1. Top-K候选池 - 维护表现最好的K个程序
2. 岛屿模型 - 多个子种群独立进化，定期交换信息
3. 允许不可行解短暂存活 - 增加多样性
"""

import random
from typing import List, Tuple, Dict, Set, Optional
from collections import deque


class TopKCandidatePool:
    """Top-K候选池 - 维护表现最好的程序"""
    
    def __init__(self, k: int = 10):
        """
        Args:
            k: 保留的候选数量上限
        """
        self.k = k
        self.candidates: List[dict] = []  # 每个元素: {code, score_ratio, novelty, island_id}
        self.history: List[dict] = []     # 完整历史
    
    def add(self, code: str, score_ratio: float, novelty: float = 0.5, island_id: int = 0) -> bool:
        """
        添加程序到候选池
        Returns:
            是否成功添加（即使不是top-K，只要通过多样性检查也会保留到历史）
        """
        program = {
            'code': code,
            'score_ratio': score_ratio,
            'novelty': novelty,
            'island_id': island_id
        }
        
        self.history.append(program)
        
        # 检查是否应该加入top-K
        if len(self.candidates) < self.k:
            self.candidates.append(program)
            self.candidates.sort(key=lambda x: x['score_ratio'])
            return True
        else:
            # 如果比最差的更好，则替换
            worst = max(self.candidates, key=lambda x: x['score_ratio'])
            if score_ratio < worst['score_ratio']:
                self.candidates.remove(worst)
                self.candidates.append(program)
                self.candidates.sort(key=lambda x: x['score_ratio'])
                return True
        
        return False
    
    def get_best(self) -> Optional[dict]:
        """获取最佳程序"""
        return self.candidates[0] if self.candidates else None
    
    def get_diverse_sample(self, n: int = 3) -> List[dict]:
        """从候选池中获取多样化的样本"""
        if len(self.candidates) <= n:
            return self.candidates.copy()
        
        # 按新颖性分布选择，而非只选最好的
        sample = []
        remaining = self.candidates.copy()
        
        # 先选最好的
        best = remaining.pop(0)
        sample.append(best)
        
        while len(sample) < n and remaining:
            # 选与已有最不相似的
            best_candidate = None
            best_min_similarity = -1
            
            for candidate in remaining:
                min_sim = float('inf')
                for selected in sample:
                    sim = self._compute_similarity(candidate, selected)
                    min_sim = min(min_sim, sim)
                
                if min_sim > best_min_similarity:
                    best_min_similarity = min_sim
                    best_candidate = candidate
            
            if best_candidate:
                sample.append(best_candidate)
                remaining.remove(best_candidate)
            else:
                break
        
        return sample
    
    def _compute_similarity(self, prog1: dict, prog2: dict) -> float:
        """计算两个程序的相似度"""
        # 简单基于score_ratio的差异
        return 1 - abs(prog1['score_ratio'] - prog2['score_ratio'])
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            'pool_size': len(self.candidates),
            'history_size': len(self.history),
            'best_score': self.candidates[0]['score_ratio'] if self.candidates else None,
            'worst_in_pool': max(c['score_ratio'] for c in self.candidates) if self.candidates else None,
        }


class IslandModel:
    """岛屿模型 - 多个子种群独立进化，定期交换信息"""
    
    def __init__(
        self,
        num_islands: int = 3,
        migration_interval: int = 5,
        migration_size: int = 1
    ):
        """
        Args:
            num_islands: 岛屿数量
            migration_interval: 多少代迁移一次
            migration_size: 每次迁移的程序数量
        """
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_size = migration_size
        
        # 每个岛屿独立的候选池
        self.islands: List[TopKCandidatePool] = [
            TopKCandidatePool(k=10) for _ in range(num_islands)
        ]
        
        # 岛屿间的迁移历史
        self.migration_log: List[dict] = []
    
    def add_to_island(
        self, 
        island_id: int, 
        code: str, 
        score_ratio: float, 
        novelty: float = 0.5
    ) -> bool:
        """添加程序到指定岛屿"""
        if 0 <= island_id < self.num_islands:
            return self.islands[island_id].add(code, score_ratio, novelty, island_id)
        return False
    
    def migrate(self, iteration: int) -> List[dict]:
        """
        执行迁移操作
        在migration_interval代时，随机选择程序迁移到其他岛屿
        """
        if iteration > 0 and iteration % self.migration_interval != 0:
            return []
        
        migrations = []
        
        for source_id in range(self.num_islands):
            source_pool = self.islands[source_id]
            if len(source_pool.candidates) < 2:
                continue
            
            # 随机选择目标岛屿
            target_id = (source_id + random.randint(1, self.num_islands - 1)) % self.num_islands
            target_pool = self.islands[target_id]
            
            # 选择migrate_size个最"不同"的程序迁移
            candidates = source_pool.candidates[-self.migration_size:]  # 选较差的
            
            for candidate in candidates:
                # 添加到目标岛屿
                target_pool.add(
                    candidate['code'],
                    candidate['score_ratio'],
                    candidate['novelty'],
                    target_id
                )
                
                migrations.append({
                    'from_island': source_id,
                    'to_island': target_id,
                    'code': candidate['code'],
                    'score_ratio': candidate['score_ratio']
                })
        
        self.migration_log.extend(migrations)
        return migrations
    
    def get_best_overall(self) -> Optional[dict]:
        """获取全局最佳程序"""
        all_best = []
        for island in self.islands:
            if island.candidates:
                all_best.append(island.get_best())
        
        if not all_best:
            return None
        
        return min(all_best, key=lambda x: x['score_ratio'])
    
    def get_all_best_per_island(self) -> List[dict]:
        """获取每个岛屿的最佳程序"""
        results = []
        for i, island in enumerate(self.islands):
            best = island.get_best()
            if best:
                results.append({
                    'island_id': i,
                    'best_code': best['code'],
                    'best_score': best['score_ratio']
                })
        return results
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        island_stats = [island.get_statistics() for island in self.islands]
        return {
            'num_islands': self.num_islands,
            'migration_count': len(self.migration_log),
            'islands': island_stats
        }


class InfeasibleTracker:
    """不可行解追踪器 - 允许不可行解短暂存活以增加多样性"""
    
    def __init__(
        self,
        max_infeasible_age: int = 3,
        infeasible_threshold: float = 1.15
    ):
        """
        Args:
            max_infeasible_age: 不可行解最大存活代数
            infeasible_threshold: 超过此score_ratio认为不可行
        """
        self.max_infeasible_age = max_infeasible_age
        self.infeasible_threshold = infeasible_threshold
        
        # 不可行解队列：(program, age)
        self.infeasible_programs: deque = deque(maxlen=20)
        
        # 统计
        self.total_infeasible = 0
        self.recovered_count = 0  # 不可行解后来变可行的次数
    
    def add(self, code: str, score_ratio: float, novelty: float) -> bool:
        """
        添加程序
        Returns:
            是否应该保留（可行或年轻不可行）
        """
        is_infeasible = score_ratio > self.infeasible_threshold
        
        if is_infeasible:
            self.total_infeasible += 1
            self.infeasible_programs.append({
                'code': code,
                'score_ratio': score_ratio,
                'novelty': novelty,
                'age': 0
            })
            return False
        else:
            # 检查是否能恢复之前的不可行解
            self._try_recover(code)
            return True
    
    def _try_recover(self, new_code: str) -> bool:
        """尝试将之前的不可行解恢复为可行"""
        if not self.infeasible_programs:
            return False
        
        # 找最老的不可行解
        oldest_idx = 0
        max_age = -1
        
        for i, prog in enumerate(self.infeasible_programs):
            if prog['age'] > max_age:
                max_age = prog['age']
                oldest_idx = i
        
        oldest = self.infeasible_programs[oldest_idx]
        
        # 增加所有不可行解的年龄
        for prog in self.infeasible_programs:
            prog['age'] += 1
        
        # 如果最老的超过阈值，移除并标记为恢复
        if max_age >= self.max_infeasible_age:
            self.infeasible_programs.remove(oldest)
            self.recovered_count += 1
            return True
        
        return False
    
    def get_infeasible_programs(self) -> List[dict]:
        """获取仍然年轻的不可行解"""
        return [p for p in self.infeasible_programs if p['age'] < self.max_infeasible_age]
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            'current_infeasible_count': len(self.infeasible_programs),
            'total_infeasible': self.total_infeasible,
            'recovered_count': self.recovered_count,
            'recovery_rate': self.recovered_count / self.total_infeasible if self.total_infeasible > 0 else 0
        }


class EnhancedPopulationManager:
    """增强版种群管理器 - 整合上述所有功能"""
    
    def __init__(
        self,
        num_islands: int = 3,
        island_pool_size: int = 10,
        migration_interval: int = 5,
        novelty_weight: float = 0.4,
        infeasible_threshold: float = 1.15
    ):
        """
        Args:
            num_islands: 岛屿数量
            island_pool_size: 每个岛屿的池大小
            migration_interval: 迁移间隔代数
            novelty_weight: 新颖性权重
            infeasible_threshold: 不可行阈值
        """
        self.num_islands = num_islands
        self.novelty_weight = novelty_weight
        
        # 组件初始化
        self.island_model = IslandModel(
            num_islands=num_islands,
            migration_interval=migration_interval
        )
        self.infeasible_tracker = InfeasibleTracker(
            infeasible_threshold=infeasible_threshold
        )
        
        # 全局最佳
        self.global_best: Optional[dict] = None
    
    def evaluate_and_update(
        self,
        code: str,
        score_ratio: float,
        novelty: float,
        iteration: int
    ) -> dict:
        """
        评估程序并更新种群
        """
        result = {
            'added_to_pool': False,
            'is_global_best': False,
            'migrations': [],
            'info': ''
        }
        
        # 1. 检查是否应该添加（可行或年轻不可行）
        should_keep = self.infeasible_tracker.add(code, score_ratio, novelty)
        
        if not should_keep:
            result['info'] = f"Skip (infeasible score {score_ratio:.4f} > {self.infeasible_threshold})"
            return result
        
        # 2. 选择目标岛屿（基于新颖性选择岛屿）
        island_id = self._select_island_for_program(novelty)
        
        # 3. 添加到岛屿
        added = self.island_model.add_to_island(island_id, code, score_ratio, novelty)
        result['added_to_pool'] = added
        
        # 4. 检查是否是全局最佳
        if self.global_best is None or score_ratio < self.global_best['score_ratio']:
            self.global_best = {
                'code': code,
                'score_ratio': score_ratio,
                'novelty': novelty,
                'island_id': island_id,
                'iteration': iteration
            }
            result['is_global_best'] = True
            result['info'] = f"New global best: {score_ratio:.4f}"
        else:
            result['info'] = f"Pool updated (island {island_id})"
        
        # 5. 执行迁移
        migrations = self.island_model.migrate(iteration)
        if migrations:
            result['migrations'] = migrations
        
        return result
    
    def _select_island_for_program(self, novelty: float) -> int:
        """基于新颖性选择岛屿（鼓励岛屿间多样性）"""
        # 更高的新颖性倾向于分配到较少程序的岛屿
        island_sizes = [len(island.candidates) for island in self.island_model.islands]
        
        if novelty > 0.6:
            # 高新颖性：选最小的岛屿
            return island_sizes.index(min(island_sizes))
        else:
            # 正常：随机选
            return random.randint(0, self.num_islands - 1)
    
    def get_parent_candidates(self, n: int = 3) -> List[dict]:
        """获取用于生成新程序的候选父亲"""
        candidates = []
        
        # 1. 全局最佳
        if self.global_best:
            candidates.append(self.global_best)
        
        # 2. 从各岛屿最佳中选多样化样本
        for island in self.island_model.islands:
            pool_sample = island.get_diverse_sample(min(2, n))
            candidates.extend(pool_sample)
        
        # 3. 不可行解中选年轻且高新颖性的
        infeasible = self.infeasible_tracker.get_infeasible_programs()
        high_novel_infeasible = [p for p in infeasible if p['novelty'] > 0.6]
        candidates.extend(high_novel_infeasible[:2])
        
        # 去重并返回
        unique = []
        seen_codes = set()
        for c in candidates:
            if c['code'] not in seen_codes:
                unique.append(c)
                seen_codes.add(c['code'])
        
        return unique[:n]
    
    def get_statistics(self) -> dict:
        """获取完整统计信息"""
        return {
            'global_best': self.global_best['score_ratio'] if self.global_best else None,
            'island_stats': self.island_model.get_statistics(),
            'infeasible_stats': self.infeasible_tracker.get_statistics()
        }
