"""
增强版 FunSearch 主脚本 (Novelty-driven版本)
============================================
集成功能：
1. Thoughts-augmented 增强：分析历史成功模式，给LLM提供上下文
2. Sample-efficient 改进：重复代码检测，避免重复评估相似代码
3. Novelty-driven 搜索：Pareto优化，保持多样性和性能的平衡
"""

from llm_api import generate_new_heuristic
from tsp_sandbox import load_tsp_data_from_csv, evaluate_heuristic
from code_similarity import CodeSimilarityChecker, ThoughtAnalyzer
from novelty_search import NoveltyDrivenSearch


# --- 初始启发式算法（最近邻，作为基线） ---
INITIAL_CODE = """
def select_next_node(distance_matrix, current_node, unvisited_nodes):
    min_dist = float('inf')
    best_node = unvisited_nodes[0]
    for node in unvisited_nodes:
        if distance_matrix[current_node][node] < min_dist:
            min_dist = distance_matrix[current_node][node]
            best_node = node
    return best_node
"""


def main():
    print("🚀 启动增强版 Novelty-driven FunSearch...")
    print("📦 集成功能：重复代码检测 + 历史模式分析 + 多样性搜索\n")
    
    # 初始化各个检测器
    similarity_checker = CodeSimilarityChecker(similarity_threshold=0.85)
    thought_analyzer = ThoughtAnalyzer()
    novelty_search = NoveltyDrivenSearch(novelty_weight=0.4)
    
    # 加载数据
    dataset = load_tsp_data_from_csv("tsp_instances_dataset.csv", num_instances=25)
    
    if not dataset:
        return
        
    # 评估初始基线
    best_avg_raw_distance, baseline_distances, best_paths = evaluate_heuristic(INITIAL_CODE, dataset)
    best_code = INITIAL_CODE
    
    print(f"\n📍 初始基线算法在 {len(dataset)} 个地图上的成绩如下 (平均距离: {best_avg_raw_distance:.2f}):")
    for i, dist in enumerate(baseline_distances):
        print(f"   - 地图 {i+1} ({dataset[i]['num_cities']}城): {dist:.2f}")
    print("-" * 50 + "\n")
    
    best_score_ratio = 1.0
    iterations = 30
    
    # 多样性搜索的候选池
    candidate_pool_size = 5
    all_candidates: list[dict] = []
    
    for i in range(iterations):
        print(f"\n{'='*10} 迭代 {i+1}/{iterations} {'='*10}")
        
        # ====== 增强1: 构建包含历史上下文的高级Prompt ======
        successful_patterns = thought_analyzer.get_successful_patterns()
        improvement_suggestion = thought_analyzer.suggest_improvement_focus()
        
        context_additions = []
        if successful_patterns:
            context_additions.append(f"\n[SUCCESSFUL PATTERNS]")
            for idx, pattern in enumerate(successful_patterns[-3:], 1):
                truncated = pattern[:200] + "..." if len(pattern) > 200 else pattern
                context_additions.append(f"  {idx}. {truncated}")
        
        if improvement_suggestion:
            context_additions.append(f"\n[建议方向] {improvement_suggestion}")
        
        context_block = "\n".join(context_additions) if context_additions else ""
        
        # ====== 增强3: 选择多样化的父亲代码 ======
        diverse_pop = novelty_search.get_diverse_population(n=3)
        if diverse_pop and len(diverse_pop) >= 2:
            # 随机选择一个不太一样但性能也不错的作为父亲
            import random
            parent = random.choice(diverse_pop[:3])
            prompt_code = parent['code']
            parent_info = f"\n[父亲代码性能: {parent['score_ratio']:.4f}, 新颖性: {parent['novelty']:.2f}]"
        else:
            prompt_code = best_code
            parent_info = ""
        
        prompt = (
            "You are an expert algorithm scientist optimizing a heuristic for the Traveling Salesman Problem (TSP).\n"
            f"Current best Relative Score Ratio: {best_score_ratio:.4f} (lower is better. 1.0 is the Nearest Neighbor baseline).\n"
            f"Iteration: {i+1}/{iterations}\n"
            f"Total codes evaluated: {similarity_checker.get_stats()['total_evaluated']}\n"
            f"{parent_info}{context_block}\n\n"
            "CURRENT CODE TO IMPROVE:\n"
            f"```python\n{prompt_code}\n```\n\n"
            "INSTRUCTIONS:\n"
            "1. [Design Thought]: Analyze how to improve the current code.\n"
            "   - DO NOT use '1-step Look-ahead' (it fails on these maps).\n"
            "   - Implement a 'Hybrid Weighting' strategy: score = distance_to_candidate + alpha * penalty_term.\n"
            "   - Keep alpha small (0.1 to 0.5) so it gently guides the greedy choice.\n"
            "   - Try to explore DIFFERENT approaches from previous solutions for diversity.\n"
            "2. [Python Code]: Implement your adjusted scoring mechanism.\n"
            "   - CRITICAL: Handle `len(unvisited_nodes) == 1` properly.\n"
            "   - Return ONLY valid Python code inside ```python ``` blocks.\n"
        )
        
        print("🧠 正在请求 DeepSeek 进行思考与编码...")
        full_response, new_code = generate_new_heuristic(prompt)
        
        if not new_code:
            print("❌ 未能生成有效代码，跳过...")
            continue
            
        print("\n--- 💡 DeepSeek 的设计思路 (Design Thought) ---")
        thought = full_response.split("```python")[0].strip()
        print(thought[:500] + "...\n(思路已截断)")
        print("----------------------------------------------\n")
        
        # ====== 增强2: 重复代码检测 ======
        print("🔍 检查代码重复度...")
        is_duplicate, similarity, similar_code = similarity_checker.is_duplicate(new_code)
        
        if is_duplicate:
            print(f"⚠️ 检测到重复代码！相似度: {similarity:.2%}")
            print("   跳过评估（节省API调用）")
            thought_analyzer.add_thought(thought, new_code, best_score_ratio, improved=False)
            continue
        else:
            print(f"✅ 代码相似度: {similarity:.2%}，通过检测")
        
        print("⚙️ 评估新代码...")
        
        score_ratio, new_distances, new_paths = evaluate_heuristic(new_code, dataset, baseline_distances=baseline_distances)
    
        if score_ratio == float('inf') or not new_distances:
            print("💥 新代码运行失败或逻辑不合法")
            thought_analyzer.add_thought(thought, new_code, float('inf'), improved=False)
        else:
            # ====== 增强3: Novelty-driven选择 ======
            is_selected, selection_info = novelty_search.evaluate_and_select(new_code, score_ratio)
            
            print(f"📊 新颖性分数: {selection_info['novelty']:.2f}")
            
            if score_ratio < best_score_ratio:
                print(f"🎉 进化成功！发现了更优的逻辑！")
                print(f"📉 综合提升比例: {(1 - score_ratio)*100:.2f}% (综合得分: {score_ratio:.4f})")
                
                print("📊 详细战报：")
                for idx, ndist in enumerate(new_distances):
                    bdist = baseline_distances[idx]
                    print(f"   - 地图 {idx+1}: {bdist:.2f} -> {ndist:.2f} (比率: {(ndist/bdist):.3f})")
                
                best_score_ratio = score_ratio
                best_code = new_code
                best_paths = new_paths
                
                with open("best_tsp_heuristic.py", "w", encoding="utf-8") as f:
                    f.write(best_code)
                
                import json
                with open("best_paths.json", "w", encoding="utf-8") as f:
                    json.dump(best_paths, f)
                
                print(f"💾 成果已保存")
                thought_analyzer.add_thought(thought, new_code, score_ratio, improved=True)
                similarity_checker.add_evaluated_code(new_code)
            elif is_selected:
                print(f"📈 新代码被选入候选池（性能:{score_ratio:.4f}, 新颖性:{selection_info['novelty']:.2f}）")
                thought_analyzer.add_thought(thought, new_code, score_ratio, improved=True)
                similarity_checker.add_evaluated_code(new_code)
            else:
                print(f"📉 进化失败。新算法没有改进。当前综合得分: {score_ratio:.4f} (历史最佳: {best_score_ratio:.4f})")
                thought_analyzer.add_thought(thought, new_code, score_ratio, improved=False)
                similarity_checker.add_evaluated_code(new_code)
        
        print("\n")
    
    print("\n🏁 ========== 搜索结束 ==========")
    print(f"🏆 发现的最优综合得分 (Ratio): {best_score_ratio:.4f}")
    
    # 输出统计信息
    stats = novelty_search.get_statistics()
    print(f"\n📊 Novelty Search 统计:")
    print(f"   - 总共评估: {stats['total_evaluated']} 个程序")
    print(f"   - 精英存档大小: {stats['elite_archive_size']}")
    print(f"   - 新颖性范围: {stats['novelty_range'][0]:.2f} - {stats['novelty_range'][1]:.2f}")
    
    success_keywords = thought_analyzer.analyze_success_keywords()
    if success_keywords:
        print("\n🔑 成功模式关键词分析:")
        for kw, count in success_keywords.items():
            print(f"   - '{kw}': {count}次出现")
    
    print("\n📜 最优代码已保存在 best_tsp_heuristic.py 中。最后版本如下:")
    print(best_code)


if __name__ == "__main__":
    main()
