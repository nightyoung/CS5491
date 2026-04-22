"""
测试Novelty-driven搜索功能
"""
import sys
sys.path.insert(0, '.')

from novelty_search import NoveltyCalculator, ParetoOptimizer, NoveltyDrivenSearch


def test_novelty_calculator():
    print("=" * 60)
    print("测试 NoveltyCalculator")
    print("=" * 60)
    
    calc = NoveltyCalculator()
    
    # 代码1：最近邻
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
    
    # 代码2：带质心惩罚
    code2 = """
def select_next_node(distance_matrix, current_node, unvisited_nodes):
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    import numpy as np
    n = len(distance_matrix)
    remaining_nodes = list(unvisited_nodes)
    remaining_distances = distance_matrix[remaining_nodes]
    centroid = np.mean(remaining_distances, axis=0)
    centroid_divs = np.linalg.norm(remaining_distances - centroid, axis=1)
    avg_dist = np.mean(distance_matrix)
    alpha = 0.4 / np.log10(n + 1)
    distances = distance_matrix[current_node][remaining_nodes]
    scores = distances - alpha * centroid_divs
    return remaining_nodes[np.argmin(scores)]
"""
    
    # 代码3：纯numpy版本
    code3 = """
def select_next_node(distance_matrix, current_node, unvisited_nodes):
    import numpy as np
    distances = distance_matrix[current_node][unvisited_nodes]
    return unvisited_nodes[np.argmin(distances)]
"""
    
    # 添加程序并获取新颖性
    calc.add_program(code1, 1.0)
    calc.add_program(code2, 0.95)
    calc.add_program(code3, 1.05)
    
    print("\n1. 测试特征提取:")
    features1 = calc._extract_behavior_features(code1)
    features2 = calc._extract_behavior_features(code2)
    print(f"   代码1特征: {features1}")
    print(f"   代码2特征: {features2}")
    print(f"   共同特征: {features1 & features2}")
    
    print("\n2. 测试新颖性分数:")
    for i, prog in enumerate(calc.evaluated_programs):
        print(f"   程序{i+1}新颖性: {prog['novelty']:.4f}")
    
    print("\n3. 获取最佳新颖个体:")
    best = calc.get_best_novel_individuals(2)
    for b in best:
        print(f"   score={b['score_ratio']:.4f}, novelty={b['novelty']:.4f}")


def test_pareto_optimizer():
    print("\n" + "=" * 60)
    print("测试 ParetoOptimizer")
    print("=" * 60)
    
    optimizer = ParetoOptimizer(novelty_weight=0.3)
    
    programs = [
        {'code': 'A', 'score_ratio': 0.9, 'novelty': 0.3},
        {'code': 'B', 'score_ratio': 1.0, 'novelty': 0.8},
        {'code': 'C', 'score_ratio': 0.95, 'novelty': 0.5},
        {'code': 'D', 'score_ratio': 0.85, 'novelty': 0.2},
    ]
    
    print("\n输入程序:")
    for p in programs:
        print(f"   {p['code']}: perf={p['score_ratio']:.2f}, nov={p['novelty']:.2f}")
    
    selected = optimizer.select_candidates(programs, top_k=3)
    print("\n选中程序 (top-3):")
    for s in selected:
        print(f"   {s['program']['code']}: fitness={s['fitness']:.4f}")


def test_novelty_driven_search():
    print("\n" + "=" * 60)
    print("测试 NoveltyDrivenSearch")
    print("=" * 60)
    
    search = NoveltyDrivenSearch(novelty_weight=0.4)
    
    test_codes = [
        ("最近邻", """
def select_next_node(distance_matrix, current_node, unvisited_nodes):
    min_dist = float('inf')
    best_node = unvisited_nodes[0]
    for node in unvisited_nodes:
        if distance_matrix[current_node][node] < min_dist:
            min_dist = distance_matrix[current_node][node]
            best_node = node
    return best_node
""", 1.0),
        ("质心惩罚", """
def select_next_node(distance_matrix, current_node, unvisited_nodes):
    import numpy as np
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    remaining_nodes = list(unvisited_nodes)
    remaining_distances = distance_matrix[remaining_nodes]
    centroid = np.mean(remaining_distances, axis=0)
    centroid_divs = np.linalg.norm(remaining_distances - centroid, axis=1)
    alpha = 0.3
    distances = distance_matrix[current_node][remaining_nodes]
    scores = distances - alpha * centroid_divs
    return remaining_nodes[np.argmin(scores)]
""", 0.95),
        ("自适应alpha", """
def select_next_node(distance_matrix, current_node, unvisited_nodes):
    import numpy as np
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    n = len(distance_matrix)
    progress = 1 - len(unvisited_nodes)/n
    remaining_nodes = list(unvisited_nodes)
    remaining_distances = distance_matrix[remaining_nodes]
    centroid = np.mean(remaining_distances, axis=0)
    centroid_divs = np.linalg.norm(remaining_distances - centroid, axis=1)
    avg_dist = np.mean(distance_matrix)
    alpha_base = 0.4 / np.log10(n + 1)
    alpha = alpha_base * (0.7 + 0.6 * progress)
    distances = distance_matrix[current_node][remaining_nodes]
    scores = distances - alpha * centroid_divs
    return remaining_nodes[np.argmin(scores)]
""", 0.92),
    ]
    
    print("\n评估过程:")
    for name, code, score in test_codes:
        is_selected, info = search.evaluate_and_select(code, score)
        print(f"  {name}: score={score:.4f}, novelty={info['novelty']:.2f}, selected={is_selected}")
    
    print("\n统计信息:")
    stats = search.get_statistics()
    for k, v in stats.items():
        print(f"   {k}: {v}")
    
    print("\n多样化种群 (top-3):")
    diverse = search.get_diverse_population(3)
    for d in diverse:
        print(f"   perf={d['score_ratio']:.4f}, nov={d['novelty']:.4f}")


if __name__ == "__main__":
    test_novelty_calculator()
    test_pareto_optimizer()
    test_novelty_driven_search()
    print("\n" + "=" * 60)
    print("[OK] All Novelty tests passed!")
    print("=" * 60)
