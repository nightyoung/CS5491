import numpy as np

def select_next_node(distance_matrix, current_node, unvisited_nodes):
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
    
    n = len(distance_matrix)
    progress = 1 - len(unvisited_nodes)/n
    
    # Maintain the adaptive alpha scaling (proven effective)
    avg_dist = np.mean(distance_matrix)
    alpha_base = 0.4 / np.log10(n + 1) * (1 + 0.5 * (avg_dist > np.median(distance_matrix)))
    alpha = alpha_base * (0.7 + 0.6 * progress)
    
    # Precompute values more efficiently
    remaining_nodes = list(unvisited_nodes)
    remaining_distances = distance_matrix[remaining_nodes]
    centroid = np.mean(remaining_distances, axis=0)
    
    # Compute all penalty components vectorized
    centroid_divs = np.linalg.norm(remaining_distances - centroid, axis=1)
    
    # Mean distance to other unvisited nodes for each candidate
    conn_scores = np.mean(remaining_distances, axis=1)
    
    # Hub scores (how much each node helps others connect)
    hub_scores = []
    for i, node in enumerate(remaining_nodes):
        others = [n for n in remaining_nodes if n != node]
        hub_scores.append(np.mean([np.min(distance_matrix[other]) for other in others]))
    hub_scores = np.array(hub_scores)
    
    # Adaptive weights based on progress
    centroid_weight = 0.4 * (1.0 - 0.5 * progress)  # Favor centroid more early
    conn_weight = 0.4 * (0.3 + 0.7 * progress)      # Connectivity matters more later
    hub_weight = 0.2                                # Constant hub importance
    
    # Combine penalty terms with adaptive weights
    penalty_terms = (centroid_weight * centroid_divs + 
                     conn_weight * conn_scores + 
                     hub_weight * hub_scores)
    
    # Get distances from current node
    distances = distance_matrix[current_node][remaining_nodes]
    
    # Compute final scores
    scores = distances - alpha * penalty_terms
    
    # Return node with minimum score
    return remaining_nodes[np.argmin(scores)]