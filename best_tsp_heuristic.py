def select_next_node(distance_matrix, current_node, unvisited_nodes):
    if len(unvisited_nodes) == 1:
        return unvisited_nodes[0]
        
    # Enhanced dynamic weighting based on solution progress
    progress = 1 - len(unvisited_nodes)/len(distance_matrix)
    alpha = 0.15 + 0.25 * (1 - progress)**2  # More aggressive early, gentle later
    
    # Precompute node importance metrics
    centrality = {}
    connectivity = {}
    for node in unvisited_nodes:
        # Improved centrality measure (weighted reciprocal)
        centrality[node] = sum(1/(distance_matrix[node][other] + 1e-5)
                             for other in unvisited_nodes if other != node)
        
        # Enhanced connectivity measure (incorporates both global and local structure)
        distances = [distance_matrix[node][other] for other in unvisited_nodes if other != node]
        avg_dist = sum(distances) / len(distances)
        std_dist = (sum((d - avg_dist)**2 for d in distances)/len(distances))**0.5
        connectivity[node] = (1/(avg_dist + 1e-5)) * (1 - 0.2 * std_dist/(avg_dist + 1e-5))
    
    min_score = float('inf')
    best_node = unvisited_nodes[0]
    
    for candidate in unvisited_nodes:
        base_distance = distance_matrix[current_node][candidate]
        
        # Dynamic penalty components based on progress
        centrality_weight = 0.3 + 0.4 * progress  # More important later
        connectivity_weight = 0.7 - 0.4 * progress  # More important early
        
        # Normalize metrics for the current candidate set
        max_centrality = max(centrality.values())
        min_centrality = min(centrality.values())
        norm_centrality = (centrality[candidate] - min_centrality) / (max_centrality - min_centrality + 1e-5)
        
        max_connectivity = max(connectivity.values())
        min_connectivity = min(connectivity.values())
        norm_connectivity = (connectivity[candidate] - min_connectivity) / (max_connectivity - min_connectivity + 1e-5)
        
        # Combined penalty term (weighted harmonic mean)
        penalty = 1 / (centrality_weight/(norm_centrality + 1e-5) + connectivity_weight/(norm_connectivity + 1e-5))
        
        # Final score with dynamic weighting
        score = base_distance + alpha * penalty
        
        if score < min_score:
            min_score = score
            best_node = candidate
            
    return best_node