import numpy as np
import pandas as pd

def load_tsp_data_from_csv(file_path, num_instances=5):
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"找不到文件: {file_path}，请确保路径正确。")
        return []

    instances = []
    
    for index, row in df.head(num_instances).iterrows():
        num_cities = int(row['Num_Cities'])
        coords = []
        
        for i in range(1, num_cities + 1):
            x_col, y_col = f'City_{i}_X', f'City_{i}_Y'
            if x_col in row and y_col in row:
                coords.append([row[x_col], row[y_col]])
            else:
                break
                
        if len(coords) < num_cities:
            continue
            
        coords = np.array(coords)
        dist_matrix = np.linalg.norm(coords[:, np.newaxis] - coords, axis=2)
        
        instances.append({
            'id': row['TSP_Instance'],
            'num_cities': num_cities,
            'dist_matrix': dist_matrix,
            'coords': coords # 把坐标也存下来，以后画图用得着
        })
        
    return instances

def evaluate_heuristic(heuristic_code, instances, baseline_distances=None):
    local_scope = {}
    try:
        exec(heuristic_code, globals(), local_scope)
        select_next_node = local_scope.get('select_next_node')
        if not select_next_node:
            return float('inf'), [], []
    except Exception as e:
        return float('inf'), [], []
    
    current_distances = []
    current_paths = [] # 【新增】用来保存每个地图的具体行走路线
    total_ratio = 0.0
    
    for i, instance in enumerate(instances):
        dist_matrix = instance['dist_matrix']
        num_cities = instance['num_cities']
        
        unvisited = list(range(1, num_cities)) # 从城市 0 出发
        current_node = 0
        tour_distance = 0
        tour_path = [0] # 【新增】路线起点是 0
        
        try:
            while unvisited:
                next_node = select_next_node(dist_matrix, current_node, unvisited.copy())
                
                if next_node not in unvisited:
                    return float('inf'), [], [] # 违规选择
                
                tour_distance += dist_matrix[current_node][next_node]
                current_node = next_node
                unvisited.remove(next_node)
                tour_path.append(current_node) # 【新增】记录走过的节点
                
            # 回到起点
            tour_distance += dist_matrix[current_node][0]
            tour_path.append(0) # 【新增】终点回到 0
            
            current_distances.append(tour_distance)
            current_paths.append(tour_path) # 【新增】把这条路线保存到总列表里
            
            if baseline_distances:
                ratio = tour_distance / baseline_distances[i]
                total_ratio += ratio
            
        except Exception:
            return float('inf'), [], []
            
    if not baseline_distances:
        avg_raw_distance = sum(current_distances) / len(instances)
        return avg_raw_distance, current_distances, current_paths # 【修改】返回路径
    else:
        avg_ratio = total_ratio / len(instances)
        return avg_ratio, current_distances, current_paths # 【修改】返回路径