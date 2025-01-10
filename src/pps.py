from gymnasium import spaces
from putils import *
from gym.utils import seeding
import numpy as np
from pettingzoo import ParallelEnv
import networkx as nx

class PredatorPreySwarmEnv(ParallelEnv):
    
    metadata = {"render.modes": ["human", "rgb_array", "pygame"], "video.frames_per_second": 30}
    
    def __init__(self, config=None, render_mode=None, metrics_in_info=False, infect=True):
        # Setting up the environment parameters
        if config is None:
            print('Using default config')
            # agent parameters
            self._n_e = 50
            self._escaper_strategy = 'input'
            self._act_dim_escaper = 2
            self._m_e = 1
            self._size_e = 0.035 
            self._topo_n_e2e = 6
            self._FoV_e = 5
            self._sensitivity = 1 
            self._linVel_e_max = 0.5
            self._linAcc_e_max = 1
            self._angle_e_max = 0.5
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

        to_infect = np.random.choice(self.possible_agents, int(self._n_e*self._part_infected))
        self.to_infect = {agent:False for agent in list(range(self._n_e))}
        for i in to_infect:
            self.to_infect[i] = True

        self.is_infect = infect
        self.infected = {agent:False for agent in list(range(self._n_e))}
        if self.is_infect:
            self.infected = self.to_infect.copy()

        self.metrics_in_info = metrics_in_info
        self.interactions = np.zeros((self._n_e, self._n_e), dtype=int)
        self.recent_interactions = np.zeros((self._n_e, self._n_e), dtype=float)
        
    def observation_space(self, agent):
        return self.observation_spaces[agent]
    
    def action_space(self, agent):
        return self.action_spaces[agent]

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def infect(self):
        self.infected = self.to_infect.copy()

    def clear_network(self):
        self.interactions = np.zeros((self._n_e, self._n_e), dtype=int)

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
                        
        self.task_state = {agent: np.random.choice([0, 1], p=[0.5, 0.5]) for agent in list(range(self._n_e))}
        # x position of the agent
        self.prev_position = {agent: self._p[0, agent] for agent in list(range(self._n_e))}

        if self.is_infect:
            self.infected = {agent:False for agent in list(range(self._n_e))}
        if self.is_infect:
            self.infected = self.to_infect.copy()
        self.interactions = np.zeros((self._n_e, self._n_e), dtype=int)
        self.recent_interactions = np.zeros((self._n_e, self._n_e), dtype=float)
        
        self.obs = self._get_obs()
        return self.obs, self._get_info()


    def _get_obs(self):

        observations = {}
        for i in range(self._n_e):
            self.obs = np.zeros(self.observation_space(i).shape)   
            relPos_e2e = self._p - self._p[:,[i]]
            if self._is_periodic: relPos_e2e = make_periodic(relPos_e2e, self._L)
            relVel_e2e = self._heading - self._heading[:,[i]]
            infected = np.array([self.infected[j] for j in range(self._n_e)])
            relPos_e2e, relVel_e2e, infected = get_focused(relPos_e2e, relVel_e2e, infected, self._FoV_e, self._topo_n_e2e, True)  

            obs_escaper = []
            obs_escaper_pos = np.concatenate((self._p[:, [i]], relPos_e2e), axis=1)
            obs_escaper_vel = np.concatenate((self._dp[:, [i]], relVel_e2e), axis=1)
            obs_escaper_infected = np.concatenate((np.array([self.infected[i]]), infected), axis=0).reshape(1,self._topo_n_e2e+1)
            obs_escaper = np.concatenate((obs_escaper_pos, obs_escaper_vel, obs_escaper_infected), axis=0) 
            
            obs_escaper = np.append(obs_escaper, self.task_state[i])
            
            self.obs[:self.obs_dim_escaper-2] = obs_escaper.T.reshape(-1)        
            self.obs[self.obs_dim_escaper-2:] = self._heading[:,i] # Own heading
            
            observations[i] = self.obs

        return observations

    def _get_reward(self, a):        
        
        
        reward = {agent:0 for agent in list(range(self._n_e))}
        
        # ADDED TASK ############################################################ 
        # loop through all agents and update tasks:
        # if agent task_state is 0 and is colliding with left wall, set task_state to 1
        # if agent task_state is 1 and is colliding with right wall, set task_state to 0
        for i in range(self._n_e):
            
            # TASK REWARDS 
            colliding_left = self.is_collide_b2w[0, i]
            colliding_right = self.is_collide_b2w[2, i]
            if self.task_state[i] == 0:
                if colliding_left:
                    self.task_state[i] = 1
                    reward[i] = 1
                elif self._p[0, i] < self.prev_position[i]:
                    reward[i] = 0.1                   
                
            elif self.task_state[i] == 1:
                if colliding_right:
                    self.task_state[i] = 0
                    reward[i] = 1
                elif self._p[0, i] > self.prev_position[i]:
                    reward[i] = 0.1
                
            # UPDATE POSITION
            self.prev_position[i] = self._p[0, i]
            
            # INFECTION REWARDS
            for j in range(i):
                if self._is_collide_b2b[i,j]:
                    if self.infected[i] != self.infected[j]:
                        reward[i] -= 1
                        reward[j] -= 1
                    else:
                        reward[i] += 0.1
                        reward[j] += 0.1
                        
                    # else:
                    #     reward[i] += 1 * (1 - self.recent_interactions[i, j])
                    #     reward[j] += 1 * (1 - self.recent_interactions[i, j])
            #         self.recent_interactions[i, j] = 1
        
            # if np.mean([1 if x != 0 else 0 for x in self.recent_interactions[i, :]]) < 0.2:
            #     reward[i] -= 1
        
        # print(reward)
        return reward
                
        
        
        
        # check if the agents are colliding, if so, give a negative ¸ to both agents
        # for i in range(self._n_e):
        #     for j in range(i):
        #         if self._is_collide_b2b[i,j]:
        #             if self.infected[i] != self.infected[j]:
        #                 reward[i] -= 1
        #                 reward[j] -= 1
        #             else:
        #                 reward[i] += 1 * (1 - self.recent_interactions[i, j])
        #                 reward[j] += 1 * (1 - self.recent_interactions[i, j])
        #             self.recent_interactions[i, j] = 1
        
        #     if np.mean([1 if x != 0 else 0 for x in self.recent_interactions[i, :]]) < 0.2:
        #         reward[i] -= 1

        # return reward
    
    # Healthy clustering with infected avoiding them
    # if self.infected[i] != self.infected[j]:
    #     reward[i] -= 1
    #     reward[j] -= 1
    # else:
    #     reward[i] += 1
    #     reward[j] += 1

    # Added decreasing penalty for repeated interactions
    # for j in range(i):
    #     if self._is_collide_b2b[i,j]:
    #         if self.infected[i] != self.infected[j]:
    #             reward[i] -= 1
    #             reward[j] -= 1
    #         else:
    #             reward[i] += 1 * (1 - self.recent_interactions[i, j])
    #             reward[j] += 1 * (1 - self.recent_interactions[i, j])
    #         self.recent_interactions[i, j] = 1
    # if np.mean([1 if x != 0 else 0 for x in self.recent_interactions[i, :]]) < 0.2:
    #     reward[i] -= 1
    
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

        self.recent_interactions *= 0.9
        self.obs = self._get_obs()


        min_dist = 0.07 #self._size_e # TODO find reasonable value
        for i in range(self._n_e):
            for j in range(i):
                if self._d_b2b_center[i,j] < min_dist:
                    self.interactions[i, j] += 1

        return self.obs, self._get_reward(a), self.terminateds, truncateds, self._get_info()

    def render(self, mode="rgb_array"): 
        
        if not self.viewer and mode in ["human", "rgb_array"]:
            import rendering
            self.viewer = rendering.Viewer(700, 700)
            self.viewer.set_bounds(-1, 1, -1, 1.)
            agents = []
            self.tf = []
            if self._render_traj: self.trajrender = []
            for i in range(self._n_e):
                if self._render_traj: self.trajrender.append( rendering.Traj( list(zip(self._p_traj[:,0,i], self._p_traj[:,1,i])),  False) )
                #agents.append( rendering.make_unicycle(self._size_e) )
                agents.append( rendering.make_ant(self._size_e) )
                if self.infected[i]:
                    agents[i].set_color_alpha(0.778, 0.333, 0, 1)
                    if self._render_traj: self.trajrender[i].set_color_alpha(0.778, 0.333, 0, 0.5)
                else:
                    agents[i].set_color_alpha(0, 0, 0, 1)
                    if self._render_traj: self.trajrender[i].set_color_alpha(0, 0, 0, 0.5)
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
        # self.obs_dim_escaper = ( 5 * self._topo_n_e2e ) + 7
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
            if self.to_infect[i]:
                G.nodes[i]['infected'] = True
            else:
                G.nodes[i]['infected'] = False
        # for i in range(self._n_e):
        #     for j in range(i):
        #         G.add_edge(i, j, distance=self._d_b2b_center[i,j])
        return G
    
    def compute_metrics(self):
        G = self.env2nx()
        G_ = G.copy().to_undirected()

        norm_interactions = self.interactions / np.max((np.max(self.interactions), 1))
        # G_ = nx.Graph().to_undirected()
        # G_.add_nodes_from(list(range(self._n_e)))
        for i in range(self._n_e):
            for j in range(i):
                if norm_interactions[i, j] > 0.01:
                    G_.add_edge(i, j, weight=norm_interactions[i, j])
        
        # METRICS CONSIDERED IN PAPER ########################################
        # md = ... # modularity between infected and non infected nodes
        ac = nx.average_clustering(G_, weight='weight')

        # edges_to_remove = [(i, j) for i, j, w in G_.edges(data=True) if w['weight'] < 0.05]
        # G_.remove_edges_from(edges_to_remove)
        dens = nx.density(G_)
        md = nx.algorithms.community.quality.modularity(G_, [{i for i in range(self._n_e) if self.infected[i]}, {i for i in range(self._n_e) if not self.infected[i]}])
        # diam = ... # network diameter, but network is not connected, adjust
        ne = nx.global_efficiency(G_)
        # dc = nx.betweenness_centrality(G_) # or some other centrality measure, but for whole network
        dc = nx.degree_centrality(G_)
        avg_dc = np.mean(list(dc.values()))
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