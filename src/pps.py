from gymnasium import spaces
from putils import *
from gym.utils import seeding
import numpy as np
from pettingzoo import ParallelEnv
import networkx as nx

class PredatorPreySwarmEnv(ParallelEnv):
    
    metadata = {"render.modes": ["human", "rgb_array", "pygame"], "video.frames_per_second": 30}
    
    def __init__(self, config=None, render_mode=None, metrics_in_info=False):
        # Setting up the environment parameters
        if config is None:
            print('Using default config')
            # agent parameters
            self._n_e = 250
            self._escaper_strategy = 'input'
            self._act_dim_escaper = 2
            self._m_e = 1
            self._size_e = 0.035 
            self._topo_n_e2e = 6
            self._FoV_e = 5
            self._sensitivity = 1
            self._linVel_e_max = 0
            self._linAcc_e_max = 1
            self._angle_e_max = 2
            self._ep_len = 200
            self._part_infected = 0.1
            
            # environment parameters
            self._is_periodic = True
            self._L = 1
            self._k_ball = 50 
            self._c_aero = 2  
            self._k_wall = 100
            self._c_wall = 5
            self._dt = 0.1
            self._n_frames = 1

            # rendering parameters
            self._render_traj = True
            self._traj_len = 15

        else:
            print('Using custom config')
            for key in config:
                setattr(self, f'_{key}', config[key])
        self._linAcc_e_min = 0
        self.metrics = None
        
        self.viewer = None
        self.render_mode = render_mode
        self.seed()
        
        self.terminateds = {agent:False for agent in list(range(self._n_e))}
        self.truncateds = {agent:False for agent in list(range(self._n_e))}
        self.possible_agents = list(range(self._n_e))
        self.observation_spaces = {agent: self._get_observation_space() for agent in self.possible_agents}
        self.action_spaces = {agent: self._get_action_space() for agent in self.possible_agents}
        self._m = get_mass(self._m_e, self._n_e)  
        self._size, self._sizes = get_sizes(self._size_e, self._n_e)  

        self.infected = {agent:False for agent in list(range(self._n_e))}
        for i in np.random.choice(self.possible_agents, int(self._n_e*self._part_infected)):
            self.infected[i] = True

        self.metrics_in_info = metrics_in_info
        self.interactions = np.zeros((self._n_e, self._n_e), dtype=int)

        self.grid_size = 20  # Size of the grid (2000x2000)
        self.X, self.Y = np.meshgrid(np.linspace(-self._L, self._L, self.grid_size), np.linspace(-self._L, self._L, self.grid_size))
        self.pos_phero = np.zeros_like(self.X)  # Initialize pheromone map
        self.pheromone_intensity = 1  # Intensity of the pheromone
        self.pheromone_sigma = 0.15  # Spread of the pheromone
        self.sense_phero_threshold = 0.9
        self.pheromone_decay = 0.25
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self, seed=None, options=None):
        
        if seed is not None:
            self.seed(seed)
            
        self.agents = self.possible_agents[:]
        self.timesteps_left = self._ep_len

        max_size = np.max(self._size)
        max_respawn_times = 100
        for respawn_time in range (max_respawn_times):
            self._p = np.random.uniform(-1+max_size, 1-max_size, (2, self._n_e))   # Initialize self._p
            self._d_b2b_center, _, _is_collide_b2b = get_dist_b2b(self._p, self._L, self._is_periodic, self._sizes)
            if _is_collide_b2b.sum() == 0:
                break
            if respawn_time == max_respawn_times-1:
                print('Some particles are overlapped at the initial time !')
        if self._render_traj == True:
            self._p_traj = np.zeros((self._traj_len, 2, self._n_e))
            self._p_traj[0,:,:] = self._p
        self._dp = np.zeros((2, self._n_e))                  
        self._ddp = np.zeros((2, self._n_e))                                     
        self._theta = np.pi * np.random.uniform(-1,1, (1, self._n_e))
        self._heading = np.concatenate((np.cos(self._theta), np.sin(self._theta)), axis=0)
        self.obs = self._get_obs()

        self.infected = {agent:False for agent in list(range(self._n_e))}
        for i in np.random.choice(self.possible_agents, int(self._n_e*self._part_infected)):
            self.infected[i] = True
        self.interactions = np.zeros((self._n_e, self._n_e), dtype=int)
        
        return self.obs, self._get_info()
        
    def _dissipate_pheromones(self):
        """
        Simplified pheromone dissipation function where the whole grid's value increases at the agent's position.
        """
        # Decay the pheromone map (apply simple decay factor)
        self.pos_phero *= (1-self.pheromone_decay)  # Applying a simple decay to all cells in the grid
        
        # Assume _p holds agent positions (2D: [x, y] for each agent)
        for i in range(self._n_e):  # Loop over agents (assuming self._n_e is the number of agents)
            agent_x, agent_y = self._p[0, i], self._p[1, i]
            
            # Map agent positions from [-1, 1] range to [0, grid_size] range (0-10)
            shift_x = int((agent_x + 1) / 2 * self.grid_size)
            shift_y = int((agent_y + 1) / 2 * self.grid_size)
            
            # Ensure the position is within bounds (for edge cases)
            shift_x = np.clip(shift_x, 0, self.grid_size - 1)
            shift_y = np.clip(shift_y, 0, self.grid_size - 1)
            
            # Increase the pheromone value at the agent's position
            self.pos_phero[shift_x, shift_y] += self.pheromone_intensity

        # Optionally, set small pheromone values to zero for sparsity
        self.pos_phero[self.pos_phero < 1e-4] = 0

    def _sense_phero(self, pos, positive=True):
        agent_x = pos[0]
        agent_y = pos[1]
        
        # Map agent positions from [-1, 1] range to [0, grid_size] range (0-10)
        shift_x = int((agent_x + 1) / 2 * self.grid_size)
        shift_y = int((agent_y + 1) / 2 * self.grid_size)

        # Ensure indices are within the valid range (0 to grid_size - 1)
        shift_x = np.clip(shift_x, 0, self.grid_size - 1)
        shift_y = np.clip(shift_y, 0, self.grid_size - 1)

        if positive:
            val = self.pos_phero[shift_y, shift_x]
            if val >= self.sense_phero_threshold:
                return 1
            else:
                return 0
        else:
            return 0

    def _get_obs(self):

        observations = {}
        for i in range(self._n_e):
            self.obs = np.zeros(self.observation_space(i).shape)   
            relPos_e2e = self._p - self._p[:,[i]]
            if self._is_periodic: relPos_e2e = make_periodic(relPos_e2e, self._L)
            relVel_e2e = self._heading - self._heading[:,[i]]
            infected = np.array([self.infected[j] for j in range(self._n_e)])
            relPos_e2e, relVel_e2e, infected = get_focused(relPos_e2e, relVel_e2e, infected, self._FoV_e, self._topo_n_e2e, True)  

            obs_escaper_pos = np.concatenate((self._p[:, [i]], relPos_e2e), axis=1)
            obs_escaper_vel = np.concatenate((self._dp[:, [i]], relVel_e2e), axis=1)
            obs_escaper_infected = np.concatenate((np.array([self.infected[i]]), infected), axis=0).reshape(1,self._topo_n_e2e+1)
            obs_escaper = np.concatenate((obs_escaper_pos, obs_escaper_vel, obs_escaper_infected), axis=0) 
            #print(f"post concat: {obs_escaper.shape} : {obs_escaper_infected}")
            self.obs[:self.obs_dim_escaper-3] = obs_escaper.T.reshape(-1)
            self.obs[-3] = self._sense_phero(self._p[:, i])
            #print(self.obs[-3])
            self.obs[self.obs_dim_escaper-2:] = self._heading[:,i] # Own heading
            #print(f"all:{self.obs.shape}")
            
            observations[i] = self.obs
            
        return observations

    def _get_reward(self, a):        
        reward = {agent:0 for agent in list(range(self._n_e))}
        # check if the agents are colliding, if so, give a negative reward to both agents
        for i in range(self._n_e):
            for j in range(i):
                if self._is_collide_b2b[i,j]:
                    if self.infected[i] != self.infected[j]:
                        reward[i] -= 10
                        reward[j] -= 10
                    else:
                        reward[i] += 2  
                        reward[j] += 2
                # if one is sick and they collide give both -1
                # if both are healthy give bot +1

        return reward
    
    def _get_reward_new(self, a):
        rewards = {agent:0 for agent in list(range(self._n_e))}
        for agent in range(self._n_e):
            for j in range(agent):
                if self._is_collide_b2b[agent,j]: #TODO: simply elif by checking whether agents i or j are sick
                    rewards[agent] = +2
                    rewards[j] = +2
            # Agent position in physical coordinates
            agent_x = self._p[0, agent]  # Agent's x-coordinate
            agent_y = self._p[1, agent]  # Agent's y-coordinate
            
            # Map agent positions from [-1, 1] range to [0, grid_size] range (0-10)
            shift_x = int((agent_x + 1) / 2 * self.grid_size)
            shift_y = int((agent_y + 1) / 2 * self.grid_size)

            # Ensure indices are within the valid range (0 to grid_size - 1)
            shift_x = np.clip(shift_x, 0, self.grid_size - 1)
            shift_y = np.clip(shift_y, 0, self.grid_size - 1)

            # Sample pheromone concentration at the agent's position
            phero_val = self.pos_phero[shift_y, shift_x]
            if phero_val <= self.sense_phero_threshold:
                rewards[agent] = -5
            else:
                rewards[agent] = +2
        #print(rewards)
        
        return rewards
    
    def _get_info(self):

        if self.metrics_in_info:
            metrics = self.compute_metrics()
            return {agent:metrics for agent in list(range(self._n_e))}
        else:
            return {agent:{} for agent in list(range(self._n_e))}
    
    def step(self, a):        
        # a is {agent: [a1, a2]}
        # but need it to be shape (2, n_e)    
        a = np.array([a[agent] for agent in self.agents]).T
        for _ in range(self._n_frames): 
            a[0] *= self._angle_e_max
            a[1] = (self._linAcc_e_max-self._linAcc_e_min)/2 * a[1] + (self._linAcc_e_max+self._linAcc_e_min)/2 

            self._d_b2b_center, self.d_b2b_edge, self._is_collide_b2b = get_dist_b2b(self._p, self._L, self._is_periodic, self._sizes)
            sf_b2b_all = np.zeros((2*self._n_e, self._n_e)) 
            for i in range(self._n_e):
                for j in range(i):
                    delta = self._p[:,j]-self._p[:,i]
                    if self._is_periodic:
                        delta = make_periodic(delta, self._L)
                    dir = delta / self._d_b2b_center[i,j]
                    sf_b2b_all[2*i:2*(i+1),j] = self._is_collide_b2b[i,j] * self.d_b2b_edge[i,j] * self._k_ball * (-dir)
                    sf_b2b_all[2*j:2*(j+1),i] = - sf_b2b_all[2*i:2*(i+1),j]  

            sf_b2b = np.sum(sf_b2b_all, axis=1, keepdims=True).reshape(self._n_e,2).T 

            if self._is_periodic == False:
                self.d_b2w, self.is_collide_b2w = get_dist_b2w(self._p, self._size, self._L)
                sf_b2w = np.array([[1, 0, -1, 0], [0, -1, 0, 1]]).dot(self.is_collide_b2w * self.d_b2w) * self._k_wall   
                df_b2w = np.array([[-1, 0, -1, 0], [0, -1, 0, -1]]).dot(self.is_collide_b2w*np.concatenate((self._dp, self._dp), axis=0))  *  self._c_wall 

            if self._escaper_strategy == 'input':
                pass
            elif self._escaper_strategy == 'static':
                a = np.zeros((self._act_dim_escaper, self._n_e))
            elif self._escaper_strategy == 'random':
                a = np.random.uniform(-1,1, (self._act_dim_escaper, self._n_e)) 
                a[0] *= self._angle_e_max
                a[1] = (self._linAcc_e_max-self._linAcc_e_min)/2 * a[1] + (self._linAcc_e_max+self._linAcc_e_min)/2 
            else:
                print('Wrong in Step function')
            self._theta += a[0]
            self._theta = normalize_angle(self._theta)
            self._heading = np.concatenate((np.cos(self._theta), np.sin(self._theta)), axis=0) 
            u = a[1] * self._heading 
            if self._is_periodic:
                F = self._sensitivity * u  + sf_b2b - self._c_aero*self._dp
            else:
                F = self._sensitivity * u  + sf_b2b - self._c_aero*self._dp + sf_b2w + df_b2w 
            self._ddp = F/self._m
            self._dp += self._ddp * self._dt
            self._dp = np.clip(self._dp, -self._linVel_e_max, self._linVel_e_max)
            self._p += self._dp * self._dt
            if self._is_periodic:
                self._p = make_periodic(self._p, self._L)
            if self._render_traj == True:
                self._p_traj = np.concatenate( (self._p_traj[1:,:,:], self._p.reshape(1, 2, self._n_e)), axis=0 )
        
        if self.timesteps_left <= 0:
            self.agents = []
            truncateds = {agent:True for agent in list(range(self._n_e))}
        else:
            self.timesteps_left -= 1      
            truncateds = {agent:False for agent in list(range(self._n_e))}

        self.obs = self._get_obs()

        min_dist = 0.07
        for i in range(self._n_e):
            for j in range(i):
                if self._d_b2b_center[i,j] < min_dist:
                    self.interactions[i, j] += 1

        return self.obs, self._get_reward(a), self.terminateds, truncateds, self._get_info()
    
    def step_new(self, a):
        # Convert actions dictionary to a matrix (2, n_e)
        a = np.array([a[agent] for agent in self.agents]).T
        for _ in range(self._n_frames): 
            # Apply heading changes
            a[0] *= self._angle_e_max  # Scale heading changes to the maximum allowed
            self._theta += a[0]        # Update heading angles
            self._theta = normalize_angle(self._theta)  # Keep angles within [-pi, pi]
            self._heading = np.concatenate((np.cos(self._theta), np.sin(self._theta)), axis=0)  # Compute new headings

            # Apply movement force
            a[1] = (self._linAcc_e_max - self._linAcc_e_min) / 2 * a[1] + (self._linAcc_e_max + self._linAcc_e_min) / 2
            u = a[1] * self._heading * self._sensitivity  # Compute acceleration based on heading and scaled force
            F = self._sensitivity * u #  + sf_b2b - self._c_aero*self._dp
            self._ddp = F/self._m
            #print(a[1])
            # Update velocity (_dp) and clip it to max velocity
            self._dp = self._ddp * self._dt
            self._dp = np.clip(self._dp, -self._linVel_e_max, self._linVel_e_max)

            # Update positions
            self._p += self._dp * self._dt  # Simple update without acceleration or drag
            self._p = make_periodic(self._p, self._L)  # Keep positions within bounds (periodic boundary)
            #print(self._p)

            # Detect collisions
            self._d_b2b_center, self.d_b2b_edge, self._is_collide_b2b = get_dist_b2b(self._p, self._L, self._is_periodic, self._sizes)

            # Handle collisions (simplified)
            for i in range(self._n_e):
                for j in range(i):
                    if self._is_collide_b2b[i, j]:  # If collision detected
                        # Compute displacement vector to separate agents
                        delta = self._p[:, j] - self._p[:, i]
                        delta = make_periodic(delta, self._L)  # Account for periodic boundaries
                        dir = delta / self._d_b2b_center[i, j]  # Unit vector for displacement
                        overlap = self.d_b2b_edge[i, j]  # Overlap distance

                        # Adjust positions to resolve collision
                        self._p[:, i] -= dir * overlap / 2  # Push agent i away
                        self._p[:, j] += dir * overlap / 2  # Push agent j away

            #self._dissipate_pheromones() # update pheromone concentration
            #print(self.pos_phero)

            # Render trajectories if enabled
            if self._render_traj:
                self._p_traj = np.concatenate((self._p_traj[1:, :, :], self._p.reshape(1, 2, self._n_e)), axis=0)

        # Check episode termination conditions
        if self.timesteps_left <= 0:
            self.agents = []
            truncateds = {agent: True for agent in list(range(self._n_e))}
        else:
            self.timesteps_left -= 1
            truncateds = {agent: False for agent in list(range(self._n_e))}

        # Update observations
        self.obs = self._get_obs()
        return self.obs, self._get_reward(a), self.terminateds, truncateds, self._get_info()


    def render(self, mode="rgb_array"): 
        
        if not self.viewer and mode in ["human", "rgb_array"]:
            import rendering
            self.viewer = rendering.Viewer(1000, 1000)
            self.viewer.set_bounds(-1, 1, -1, 1.)
            agents = []
            self.tf = []
            if self._render_traj: self.trajrender = []
            for i in range(self._n_e):
                if self._render_traj: self.trajrender.append( rendering.Traj( list(zip(self._p_traj[:,0,i], self._p_traj[:,1,i])),  False) )
                #agents.append( rendering.make_unicycle(self._size_e) )
                agents.append( rendering.make_ant(self._size_e) )
                # change color of sick agents
                if self.infected[i]:
                    agents[i].set_color_alpha(0.778, 0.333, 0, 1)
                    if self._render_traj: self.trajrender[i].set_color_alpha(0.778, 0.333, 0, 0.5)
                else:
                    agents[i].set_color_alpha(0, 0, 0, 1)
                    if self._render_traj: self.trajrender[i].set_color_alpha(0, 0, 0, 0.5) #0, 0.333, 0.778, 0.5
                self.tf.append( rendering.Transform() )
                agents[i].add_attr(self.tf[i])
                self.viewer.add_geom(agents[i])
                if self._render_traj: self.viewer.add_geom(self.trajrender[i])
                
        elif self.viewer and mode in ["human", "rgb_array"]:
            for i in range(self._n_e):
                self.tf[i].set_rotation(self._theta[0,i])
                self.tf[i].set_translation(self._p[0,i], self._p[1,i])
                if self._render_traj: self.trajrender[i].set_traj(list(zip(self._p_traj[:,0,i], self._p_traj[:,1,i])))
            return self.viewer.render(return_rgb_array=mode=="rgb_array")
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def _get_observation_space(self):
        self.obs_dim_escaper = ( 5 * self._topo_n_e2e ) + 8
        observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim_escaper, ), dtype=np.float32)
        return observation_space

    def _get_action_space(self):
        action_space = spaces.Box(low=-1, high=1, shape=(self._act_dim_escaper, ), dtype=np.float32)
        return action_space
    
    def env2nx(self):
        G = nx.Graph()
        G.add_nodes_from(list(range(self._n_e)))
        for i in range(self._n_e):
            for j in range(i):
                G.add_edge(i, j, distance=self._d_b2b_center[i,j])
        return G
    
    def compute_metrics(self):
        G = self.env2nx()
        min_dist = 0.1
        G_ = G.copy().to_undirected()
        norm_interactions = self.interactions / np.max((np.max(self.interactions), 1))
        for edge in G.edges(data=True):
            if edge[2]['distance'] > min_dist:
                G_.remove_edge(edge[0], edge[1])
        
        # METRICS CONSIDERED IN PAPER ########################################
        # md = ... # modularity between infected and non infected nodes
        ac = nx.average_clustering(G_)
        dens = nx.density(G_)
        md = nx.algorithms.community.quality.modularity(G_, [{i for i in range(self._n_e) if self.infected[i]}, {i for i in range(self._n_e) if not self.infected[i]}])
        # diam = ... # network diameter, but network is not connected, adjust
        ne = nx.global_efficiency(G_)
        dc = nx.degree_centrality(G_)
        avg_dc = np.mean(list(dc.values()))
        # dc = nx.betweenness_centrality(G_) # or some other centrality measure, but for whole network
        # as = ... # assortativity, node specific
        ######################################################################
        
        return {'average_clustering': ac, 'network_efficiency': ne, 'density': dens, 'modularity': md, 'average_degree_centrality': avg_dc, 'G': G_}

if __name__ == '__main__':
    # parse json file
    import json
    with open('config/env_params.json') as f:
        config = json.load(f)


    # ENV TESTING ############################################################
    env = PredatorPreySwarmEnv(config)
    obs, info = env.reset()
    max_ep_len = env._ep_len
    # check for all truncated flags
    for _ in range(max_ep_len):
        img = env.render(mode='rgb_array')
        # print(img.shape)
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        #print(actions)
        obs, rew, term, trunc, info = env.step(actions)
    env.close()
    
    # API test from pettingzoo ############################################################
    # import pettingzoo
    # from pettingzoo.test import parallel_api_test
    # env = PredatorPreySwarmEnv(config)
    # print(isinstance(env, pettingzoo.ParallelEnv)) # => True
    # parallel_api_test(env, num_cycles=5000)
    
    # API test from stable_baselines3 ############################################################
    # from stable_baselines3.common.env_checker import check_env
    # import supersuit as ss
    # env = PredatorPreySwarmEnv()
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 1, base_class='stable_baselines3')
    # check_env(env) # => known BUG in the check_env function (it fails, but the env still functions)
    
    # Testing compute_metrics ############################################################
    # import supersuit as ss
    
    # print("Pettingzoo env:")
    # env = PredatorPreySwarmEnv(config, metrics_in_info=True)
    # obs, info = env.reset()
    # print(info[0])
    
    # print("Stable Baselines env:")
    # env = PredatorPreySwarmEnv(config, metrics_in_info=True)
    # env = ss.pettingzoo_env_to_vec_env_v1(env)
    # env = ss.concat_vec_envs_v1(env, 1, base_class="stable_baselines3")
    # obs = env.reset()
    # action = [env.action_space.sample() for _ in range(env.num_envs)]
    # obs, rewards, dones, info = env.step(action)
    # print(info[0])