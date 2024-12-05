import numpy as np

def make_periodic(x:np.array, L:float) -> np.array:
    x[x > L] -= 2 * L 
    x[x < -L] += 2 * L
    return x


def normalize_angle(x:np.array) -> np.array:
    return ((x + np.pi) % (2 * np.pi)) - np.pi


def get_sizes(size_e, n_e):
    size = np.concatenate((
        np.full(n_e, size_e),
    ))
    sizes = np.tile(size.reshape(n_e, 1), (1, n_e))
    sizes = sizes + sizes.T
    np.fill_diagonal(sizes, 0)
    return size, sizes


def get_mass( m_e, n_e):
    masses = np.concatenate((
        np.full(n_e, m_e),
    ))
    return masses


def get_focused(Pos, Vel, Inf, norm_threshold, width, remove_self):
    norms = np.sqrt( Pos[0,:]**2 + Pos[1,:]**2 )
    sorted_seq = np.argsort(norms)    
    Pos = Pos[:, sorted_seq]   
    norms = norms[sorted_seq] 
    Pos = Pos[:, norms < norm_threshold] 
    sorted_seq = sorted_seq[norms < norm_threshold]   
    if remove_self == True:
        Pos = Pos[:,1:]  
        sorted_seq = sorted_seq[1:]                    
    Vel = Vel[:, sorted_seq]
    Infected = Inf[sorted_seq].copy()
    target_Pos = np.zeros( (2, width) )
    target_Vel = np.zeros( (2, width) )
    target_Infect = np.zeros( (width) )
    until_idx = np.min( [Pos.shape[1], width] )
    target_Pos[:, :until_idx] = Pos[:, :until_idx] 
    target_Vel[:, :until_idx] = Vel[:, :until_idx]
    target_Infect[:until_idx] = Infected[:until_idx]
    return target_Pos, target_Vel, target_Infect

def get_dist_b2b(positions, L, is_periodic, sizes):
    """
    Calculate distances between agents and detect collisions.
    
    Args:
        positions: Array of shape (2, n_agents) with x,y coordinates
        sizes: Array of shape (n_agents, n_agents) with interaction distances
        is_periodic: Boolean for periodic boundary conditions
        L: Box size for periodic boundaries
        
    Returns:
        d_b2b_center: Distances between agent centers
        d_b2b_edge: Distances between agent edges
        is_collide_b2b: Boolean array indicating collisions
    """
    n_agents = positions.shape[1]
    d_b2b_center = np.zeros((n_agents, n_agents))
    d_b2b_edge = np.zeros((n_agents, n_agents))
    is_collide_b2b = np.zeros((n_agents, n_agents), dtype=bool)

    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            # Calculate vector between agents
            delta = positions[:, j] - positions[:, i]
            
            # Apply periodic boundary conditions if needed
            if is_periodic:
                delta = make_periodic(delta, L)
                
            # Calculate distances and detect collisions
            dist_center = np.linalg.norm(delta)
            dist_edge = dist_center - sizes[i, j]
            
            # Fill symmetric matrices
            d_b2b_center[i, j] = d_b2b_center[j, i] = dist_center
            d_b2b_edge[i, j] = d_b2b_edge[j, i] = dist_edge
            is_collide_b2b[i, j] = is_collide_b2b[j, i] = dist_edge < 0

    return d_b2b_center, -d_b2b_edge, is_collide_b2b

def get_dist_b2w(positions, sizes, L):
    """
    Calculate distances between agents and walls and apply repulsion if needed.
    
    Args:
        positions: Array of shape (2, n_agents) with x,y coordinates
        sizes: Array of shape (n_agents,) with agent sizes
        L: Box size for periodic boundaries
        
    Returns:
        d_b2w: Distances between agents and walls
               Shape: (4, n_agents), order: [left, right, bottom, top]
        is_collide_b2w: Boolean array indicating collisions
                        Shape: (4, n_agents)
    """
    n_agents = positions.shape[1]
    d_b2w = np.zeros((4, n_agents))  # 4 walls
    is_collide_b2w = np.zeros((4, n_agents), dtype=bool)
    # positions are in the range [-L, L] !!!
    for i in range(n_agents):
        # Calculate distances to walls, again from 
        left_dist = positions[0, i] - sizes[i] + L
        right_dist = L - positions[0, i] - sizes[i]
        top_dist = L - positions[1, i] - sizes[i]
        bottom_dist = positions[1, i] - sizes[i] + L
        
        d_b2w[0, i] = left_dist
        d_b2w[2, i] = right_dist
        d_b2w[1, i] = top_dist
        d_b2w[3, i] = bottom_dist
        
        is_collide_b2w[0, i] = left_dist < 0
        is_collide_b2w[2, i] = right_dist < 0
        is_collide_b2w[1, i] = top_dist < 0
        is_collide_b2w[3, i] = bottom_dist < 0
    
    return -d_b2w, is_collide_b2w