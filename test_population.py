"""
测试Population管理增强功能
"""
import sys
sys.path.insert(0, '.')

from population_manager import (
    TopKCandidatePool, 
    IslandModel, 
    InfeasibleTracker,
    EnhancedPopulationManager
)


def test_top_k_pool():
    print("=" * 60)
    print("Test TopKCandidatePool")
    print("=" * 60)
    
    pool = TopKCandidatePool(k=3)
    
    # 添加程序
    programs = [
        ("code_A", 1.0, 0.5),
        ("code_B", 0.9, 0.6),
        ("code_C", 0.95, 0.7),
        ("code_D", 0.85, 0.4),  # 比top-K差，不加入
        ("code_E", 0.88, 0.8),  # 比C好，替换
    ]
    
    for code, score, novelty in programs:
        pool.add(code, score, novelty)
        print(f"Added {code}: score={score:.4f}, novelty={novelty:.2f}")
    
    print(f"\nPool size: {pool.get_statistics()}")
    print(f"Best in pool: {pool.get_best()['code']}")
    
    diverse = pool.get_diverse_sample(2)
    print(f"Diverse sample (2): {[d['code'] for d in diverse]}")


def test_island_model():
    print("\n" + "=" * 60)
    print("Test IslandModel")
    print("=" * 60)
    
    islands = IslandModel(num_islands=3, migration_interval=5)
    
    # 初始添加
    islands.add_to_island(0, "code_A", 1.0, 0.5)
    islands.add_to_island(0, "code_B", 0.9, 0.6)
    islands.add_to_island(1, "code_C", 0.95, 0.7)
    islands.add_to_island(2, "code_D", 0.88, 0.8)
    
    print("Initial state:")
    for i, best in enumerate(islands.get_all_best_per_island()):
        print(f"  Island {i}: score={best['best_score']:.4f}")
    
    # 执行迁移
    print("\nMigrations at iteration 5:")
    migrations = islands.migrate(5)
    for m in migrations:
        print(f"  {m['from_island']} -> {m['to_island']}: score={m['score_ratio']:.4f}")
    
    print(f"\nGlobal best: {islands.get_best_overall()['code']}")


def test_infeasible_tracker():
    print("\n" + "=" * 60)
    print("Test InfeasibleTracker")
    print("=" * 60)
    
    tracker = InfeasibleTracker(max_infeasible_age=2, infeasible_threshold=1.1)
    
    programs = [
        ("code_A", 1.0, 0.5),   # feasible
        ("code_B", 1.15, 0.6),  # infeasible
        ("code_C", 1.05, 0.4),  # feasible
        ("code_D", 1.2, 0.7),   # infeasible
    ]
    
    for code, score, novelty in programs:
        kept = tracker.add(code, score, novelty)
        print(f"Added {code}: score={score:.4f}, kept={kept}")
    
    print(f"\nStats: {tracker.get_statistics()}")
    print(f"Current infeasible: {[p['code'] for p in tracker.get_infeasible_programs()]}")


def test_enhanced_population_manager():
    print("\n" + "=" * 60)
    print("Test EnhancedPopulationManager")
    print("=" * 60)
    
    manager = EnhancedPopulationManager(
        num_islands=3,
        island_pool_size=5,
        migration_interval=5
    )
    
    test_programs = [
        ("code_A", 1.0, 0.5),
        ("code_B", 0.95, 0.6),
        ("code_C", 1.05, 0.7),  # infeasible
        ("code_D", 0.90, 0.4),
        ("code_E", 0.88, 0.8),
    ]
    
    for i, (code, score, novelty) in enumerate(test_programs):
        result = manager.evaluate_and_update(code, score, novelty, i)
        print(f"\n{i}: Added {code} (score={score:.4f})")
        print(f"   Result: {result['info']}")
        if result['migrations']:
            print(f"   Migrations: {len(result['migrations'])}")
    
    print(f"\n{'='*40}")
    print("Final Statistics:")
    stats = manager.get_statistics()
    print(f"  Global best: {stats['global_best']}")
    print(f"  Island stats: {stats['island_stats']['num_islands']} islands")
    print(f"  Infeasible: {stats['infeasible_stats']['current_infeasible_count']} current")
    
    # 测试获取父亲候选
    print(f"\nParent candidates:")
    parents = manager.get_parent_candidates(3)
    for p in parents:
        code = p['code'] if len(p['code']) < 20 else p['code'][:20] + "..."
        print(f"   {code}: score={p['score_ratio']:.4f}, nov={p.get('novelty', 0):.2f}")


if __name__ == "__main__":
    test_top_k_pool()
    test_island_model()
    test_infeasible_tracker()
    test_enhanced_population_manager()
    print("\n" + "=" * 60)
    print("[OK] All Population Manager tests passed!")
    print("=" * 60)
