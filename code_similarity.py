import ast
import re
from typing import List, Tuple


class CodeSimilarityChecker:
    """检测代码相似度，基于AST结构比较"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        """
        Args:
            similarity_threshold: 相似度阈值，超过这个值认为代码重复
        """
        self.similarity_threshold = similarity_threshold
        self.evaluated_codes: List[str] = []
        
    def _normalize_code(self, code: str) -> str:
        """标准化代码：去除空格、注释，统一格式"""
        # 去除注释
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        # 去除多余空白
        code = re.sub(r'\s+', ' ', code)
        code = code.strip()
        return code
    
    def _extract_features(self, code: str) -> dict:
        """提取代码特征用于比较"""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return {}
        
        features = {
                'function_names': set(),
                'loop_types': [],
                'call_count': 0,
                'numpy_usage': False,
                'has_if_statements': False,
                'has_lambda': False,
                'variable_assignments': set(),
                'line_count': len(code.splitlines()),
            }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                features['function_names'].add(node.name)
            elif isinstance(node, ast.For):
                features['loop_types'].append('for')
            elif isinstance(node, ast.While):
                features['loop_types'].append('while')
            elif isinstance(node, ast.Call):
                features['call_count'] += 1
            elif isinstance(node, ast.Name):
                features['variable_assignments'].add(node.id)
            elif isinstance(node, ast.If):
                features['has_if_statements'] = True
            elif isinstance(node, ast.Lambda):
                features['has_lambda'] = True
        
        # 检查numpy使用
        features['numpy_usage'] = 'numpy' in code or 'np.' in code
        
        return features
    
    def _compute_similarity(self, code1: str, code2: str) -> float:
        """计算两个代码段的相似度"""
        norm1 = self._normalize_code(code1)
        norm2 = self._normalize_code(code2)
        
        if norm1 == norm2:
            return 1.0
        
        try:
            tree1 = ast.parse(norm1)
            tree2 = ast.parse(norm2)
        except SyntaxError:
            # 如果解析失败，使用文本相似度
            return self._text_similarity(norm1, norm2)
        
        # 特征相似度（基于代码结构特征）
        feat1 = self._extract_features(code1)
        feat2 = self._extract_features(code2)
        
        feature_sim = self._feature_similarity(feat1, feat2)
        
        # 结构相似度：直接比较AST dump的token集合
        dump1 = ast.dump(tree1)
        dump2 = ast.dump(tree2)
        structural_sim = self._text_similarity(dump1, dump2)
        
        # 综合相似度：特征占50%，结构占50%
        return 0.5 * structural_sim + 0.5 * feature_sim
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度（Jaccard word-level）"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _feature_similarity(self, feat1: dict, feat2: dict) -> float:
        """计算特征相似度"""
        score = 0.0
        total_weight = 0.0
        
        # 函数名相同
        weight = 2.0
        if feat1.get('function_names') and feat2.get('function_names'):
            score += len(feat1['function_names'] & feat2['function_names']) / max(len(feat1['function_names'] | feat2['function_names']), 1)
        total_weight += weight
        
        # 循环类型相同
        weight = 1.5
        loops1 = set(feat1.get('loop_types', []))
        loops2 = set(feat2.get('loop_types', []))
        if loops1 and loops2:
            score += len(loops1 & loops2) / max(len(loops1 | loops2), 1)
        total_weight += weight
        
        # 行数接近度
        weight = 0.5
        if feat1.get('line_count') and feat2.get('line_count'):
            line_diff = abs(feat1['line_count'] - feat2['line_count']) / max(feat1['line_count'], feat2['line_count'], 1)
            score += max(0, 1 - line_diff)
        total_weight += weight
        
        # numpy使用
        weight = 1.0
        if feat1.get('numpy_usage') == feat2.get('numpy_usage'):
            score += 1
        total_weight += weight
        
        # 调用次数接近度
        weight = 0.5
        if feat1.get('call_count') and feat2.get('call_count'):
            call_diff = abs(feat1['call_count'] - feat2['call_count']) / max(feat1['call_count'], feat2['call_count'], 1)
            score += max(0, 1 - call_diff)
        total_weight += weight
        
        return score / total_weight if total_weight > 0 else 0.0
    
    def is_duplicate(self, new_code: str) -> Tuple[bool, float, str]:
        """
        检查新代码是否与已评估代码重复
        
        Returns:
            (is_duplicate, similarity_score, most_similar_code)
        """
        if not new_code or not new_code.strip():
            return False, 0.0, ""
        
        max_similarity = 0.0
        most_similar_code = ""
        
        for existing_code in self.evaluated_codes:
            sim = self._compute_similarity(new_code, existing_code)
            if sim > max_similarity:
                max_similarity = sim
                most_similar_code = existing_code
        
        is_duplicate = max_similarity >= self.similarity_threshold
        return is_duplicate, max_similarity, most_similar_code
    
    def add_evaluated_code(self, code: str):
        """将已评估的代码添加到历史记录"""
        if code and code.strip():
            self.evaluated_codes.append(code)
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'total_evaluated': len(self.evaluated_codes),
            'unique_codes': len(set(self.evaluated_codes)),
        }


class ThoughtAnalyzer:
    """分析历史Thoughts，提取成功模式"""
    
    def __init__(self):
        self.thought_history: List[dict] = []
        
    def add_thought(self, thought: str, code: str, score_ratio: float, improved: bool):
        """记录一次思考结果"""
        self.thought_history.append({
            'thought': thought,
            'code': code,
            'score_ratio': score_ratio,
            'improved': improved,
        })
    
    def get_successful_patterns(self) -> List[str]:
        """提取成功的思考模式"""
        successful = [h['thought'] for h in self.thought_history if h['improved']]
        return successful
    
    def get_failed_patterns(self) -> List[str]:
        """提取失败的思考模式"""
        failed = [h['thought'] for h in self.thought_history if not h['improved']]
        return failed
    
    def analyze_success_keywords(self) -> dict:
        """分析成功思考中的关键词"""
        from collections import Counter
        
        success_thoughts = self.get_successful_patterns()
        if not success_thoughts:
            return {}
        
        # 提取关键短语
        keywords = []
        for thought in success_thoughts:
            # 提取包含特定关键词的短语
            patterns = [
                r'penalty\s+term',
                r'hybrid\s+weight',
                r'centroid',
                r'connectivity',
                r'adaptive',
                r'dynamic',
                r'alpha\s*\(',
                r'outlier',
                r'spread',
            ]
            for pattern in patterns:
                matches = re.findall(pattern, thought.lower())
                keywords.extend(matches)
        
        return dict(Counter(keywords).most_common(10))
    
    def suggest_improvement_focus(self) -> str:
        """基于历史分析建议改进方向"""
        success_keywords = self.analyze_success_keywords()
        
        if not success_keywords:
            return "继续保持多样化的探索策略"
        
        top_keywords = list(success_keywords.keys())[:3]
        return f"基于历史成功经验，建议关注: {', '.join(top_keywords)}"
    
    def get_recent_improved_thoughts(self, n: int = 3) -> List[str]:
        """获取最近n次成功的思考内容摘要"""
        improved = [h['thought'] for h in self.thought_history if h['improved']]
        return improved[-n:] if len(improved) > n else improved
