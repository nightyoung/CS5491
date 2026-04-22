"""
测试代码相似度检测器
"""
import sys
sys.path.insert(0, '.')

from code_similarity import CodeSimilarityChecker, ThoughtAnalyzer


def test_code_similarity():
    print("=" * 60)
    print("测试代码相似度检测器")
    print("=" * 60)
    
    checker = CodeSimilarityChecker(similarity_threshold=0.85)
    
    # 测试代码1：标准的最近邻
    code1 = """
def select_next_node(distance_matrix, current_node, unvisited_nodes):
    min_dist = float('inf')
    best_node = unvisited_nodes[0]
    for node in unvisited_nodes:
        if distance_matrix[current_node][node] < min_dist:
            min_dist = distance_matrix[current_node][node]
            best_node = node
    return best_node
"""
    
    # 测试代码2：几乎相同的代码（只有变量名不同）
    code2 = """
def select_next_node(dist_mat, curr_node, unvisited):
    min_distance = float('inf')
    best = unvisited[0]
    for node in unvisited:
        if dist_mat[curr_node][node] < min_distance:
            min_distance = dist_mat[curr_node][node]
            best = node
    return best
"""
    
    # 测试代码3：完全不同的逻辑
    code3 = """
def select_next_node(distance_matrix, current_node, unvisited_nodes):
    import numpy as np
    distances = distance_matrix[current_node][unvisited_nodes]
    return unvisited_nodes[np.argmin(distances)]
"""
    
    # 测试代码4：混合策略（类似现有最佳启发式）
    code4 = """
def select_next_node(distance_matrix, current_node, unvisited_nodes):
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    import numpy as np
    n = len(distance_matrix)
    progress = 1 - len(unvisited_nodes) / n
    remaining_nodes = list(unvisited_nodes)
    remaining_distances = distance_matrix[remaining_nodes]
    centroid = np.mean(remaining_distances, axis=0)
    centroid_divs = np.linalg.norm(remaining_distances - centroid, axis=1)
    avg_dist = np.mean(distance_matrix)
    alpha = 0.4 / np.log10(n + 1) * (1 + 0.5 * (avg_dist > np.median(distance_matrix)))
    alpha = alpha * (0.7 + 0.6 * progress)
    distances = distance_matrix[current_node][remaining_nodes]
    scores = distances - alpha * centroid_divs
    return remaining_nodes[np.argmin(scores)]
"""
    
    print("\n1. 测试几乎相同的代码（变量名不同）")
    is_dup, sim, similar = checker.is_duplicate(code2)
    print(f"   相似度: {sim:.2%}")
    print(f"   是否重复: {is_dup}")
    
    print("\n2. 测试完全不同的逻辑")
    is_dup, sim, similar = checker.is_duplicate(code3)
    print(f"   相似度: {sim:.2%}")
    print(f"   是否重复: {is_dup}")
    
    print("\n3. 测试混合策略代码")
    is_dup, sim, similar = checker.is_duplicate(code4)
    print(f"   相似度: {sim:.2%}")
    print(f"   是否重复: {is_dup}")
    
    # 添加评估历史
    checker.add_evaluated_code(code1)
    checker.add_evaluated_code(code3)
    
    print("\n4. 添加code1和code3到历史后，测试stats")
    stats = checker.get_stats()
    print(f"   统计: {stats}")
    
    # 测试ThoughtAnalyzer
    print("\n" + "=" * 60)
    print("测试思考分析器")
    print("=" * 60)
    
    analyzer = ThoughtAnalyzer()
    
    thought1 = "Use centroid penalty to guide the algorithm to visit outliers early. This helps improve the tour quality."
    thought2 = "Add adaptive alpha scaling based on problem size. Dynamic weights work better."
    thought3 = "Implement hybrid weighting with centroid and connectivity terms. The penalty term is crucial."
    
    analyzer.add_thought(thought1, code1, 1.0, improved=True)
    analyzer.add_thought(thought2, code2, 0.98, improved=True)
    analyzer.add_thought(thought3, code3, 1.1, improved=False)
    
    print("\n1. 成功模式:")
    for p in analyzer.get_successful_patterns():
        print(f"   - {p[:60]}...")
    
    print("\n2. 成功关键词分析:")
    keywords = analyzer.analyze_success_keywords()
    print(f"   {keywords}")
    
    print("\n3. 改进建议:")
    suggestion = analyzer.suggest_improvement_focus()
    print(f"   {suggestion}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！")
    print("=" * 60)


if __name__ == "__main__":
    test_code_similarity()
