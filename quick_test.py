import sys
import pandas as pd
import numpy as np

print('Step 1: importing pandas...')
df = pd.read_csv('tsp_instances_dataset.csv')
print(f'Step 2: loaded {len(df)} rows')

row = df.iloc[0]
print(f'Step 3: first instance: {row["TSP_Instance"]}')

num_cities = int(row['Num_Cities'])
print(f'Step 4: num_cities = {num_cities}')

coords = []
for i in range(1, num_cities + 1):
    x_col, y_col = f'City_{i}_X', f'City_{i}_Y'
    if x_col in row and y_col in row:
        coords.append([row[x_col], row[y_col]])
        
print(f'Step 5: got {len(coords)} coords')

coords = np.array(coords)
dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
print(f'Step 6: dist_matrix shape = {dist_matrix.shape}')

print('SUCCESS: TSP data loading works!')

# Now test the full tsp_sandbox loading
from tsp_sandbox import load_tsp_data_from_csv, evaluate_heuristic

print('\n--- Testing tsp_sandbox ---')
dataset = load_tsp_data_from_csv('tsp_instances_dataset.csv', num_instances=3)
print(f'Loaded {len(dataset)} instances')

if dataset:
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
    avg_dist, distances, paths = evaluate_heuristic(INITIAL_CODE, dataset)
    print(f'Average distance: {avg_dist:.2f}')
    print(f'Per-instance distances: {[f"{d:.2f}" for d in distances]}')
