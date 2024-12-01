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
            
            # environment parameters
            self._is_periodic = True
            self._L = 1
            self._k_ball = 50 
            self._c_aero = 2        
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

        self.infected = np.zeros(self._n_e)

        self.metrics_in_info = metrics_in_info
        
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

        self.infected = np.zeros(self._n_e)
        random_indices = np.random.choice(self._n_e, int(self._n_e/10), replace=False)
        self.infected[random_indices] = 1

        return self.obs, self._get_info()


    def _get_obs(self):

        observations = {}
        for i in range(self._n_e):
            self.obs = np.zeros(self.observation_space(i).shape)   
            relPos_e2e = self._p - self._p[:,[i]]
            am_infected = self.infected[i]
            if self._is_periodic: relPos_e2e = make_periodic(relPos_e2e, self._L)
            relVel_e2e = self._heading - self._heading[:,[i]] #relative velocity of all other agents
            relPos_e2e, relVel_e2e, rel_infected = get_focused(relPos_e2e, relVel_e2e, self.infected, self._FoV_e, self._topo_n_e2e, True) #relative position of 6 nearest

            obs_escaper_pos = np.concatenate((self._p[:, [i]], relPos_e2e), axis=1)
            obs_escaper_vel = np.concatenate((self._dp[:, [i]], relVel_e2e), axis=1)
            #print(self.infected.shape)
            #print(rel_infected.shape)
            obs_escaper_infect = np.append(self.infected[i], rel_infected)#np.concatenate((self.infected[i], rel_infected), axis=1)
            #print(obs_escaper_pos.shape)
            #print(obs_escaper_vel.shape)
            #print(obs_escaper_infect.shape)
            #print(self.obs.shape)

            obs_escaper = np.concatenate((obs_escaper_pos, obs_escaper_vel), axis=0)

            self.obs[:self.obs_dim_escaper-2] = np.concatenate((obs_escaper.T.reshape(-1), obs_escaper_infect))
            self.obs[self.obs_dim_escaper-2:] = self._heading[:,i] # Own heading
            
            observations[i] = self.obs
            
        return observations

    def _get_reward(self, a):        
        reward = {agent:0 for agent in list(range(self._n_e))}
        # check if the agents are colliding, if so give them a reward
        #if the agents are coliding, and one is infected, give the other a big negative reward
        for i in range(self._n_e):
            for j in range(self._n_e):
                if self._is_collide_b2b[i,j]:
                    if self.infected[j] == 1 and self.infected[i] == 0: #i is infected, j is not
                        transmission = np.random.choice([0,1], p=[0.1,0.9])
                        if transmission == 1: #give negative reward to other agent
                            #self.infected[j] = 1
                            reward[i] -= 100
                            #print("INFECTION")

                        else:
                            reward[i] += 1
                    elif self.infected[j] == 1 and self.infected[i] == 1:
                        reward[i] += 1
                    elif self.infected[j] == 0 and self.infected[i] == 0:
                        #still give reward for collisions
                        reward[i] += 1
        #reward_sum = sum(list(reward.values()))
        #print(list(reward.values()))
        #print(reward_sum)
        #if reward_sum != 0:
        #    reward_norm = {agent: reward[agent]/reward_sum for agent in reward.keys()}
        #    return reward_norm
        return reward
    
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
                    delta = make_periodic(delta, self._L)
                    dir = delta / self._d_b2b_center[i,j]
                    sf_b2b_all[2*i:2*(i+1),j] = self._is_collide_b2b[i,j] * self.d_b2b_edge[i,j] * self._k_ball * (-dir)
                    sf_b2b_all[2*j:2*(j+1),i] = - sf_b2b_all[2*i:2*(i+1),j]  

            sf_b2b = np.sum(sf_b2b_all, axis=1, keepdims=True).reshape(self._n_e,2).T 

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
            F = self._sensitivity * u  + sf_b2b - self._c_aero*self._dp
            self._ddp = F/self._m
            self._dp += self._ddp * self._dt
            self._dp = np.clip(self._dp, -self._linVel_e_max, self._linVel_e_max)
            self._p += self._dp * self._dt
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
                agents.append( rendering.make_unicycle(self._size_e) )
                if self.infected[i] == 1:
                    agents[i].set_color_alpha(0.778, 0.333, 0, 1)
                else:
                    agents[i].set_color_alpha(0, 0.333, 0.778, 1)
                if self._render_traj: self.trajrender[i].set_color_alpha(0, 0.333, 0.778, 0.5)
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
        self.obs_dim_escaper = ( 5 * self._topo_n_e2e ) + 7
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
        min_dist = 0.2 # TODO find reasonable value
        G_ = G.copy()
        for edge in G.edges(data=True):
            if edge[2]['distance'] > min_dist:
                G_.remove_edge(edge[0], edge[1])
        
        # METRICS CONSIDERED IN PAPER ########################################
        # md = ... # modularity between infected and non infected nodes
        ac = nx.average_clustering(G_)
        dens = nx.density(G_)
        # diam = ... # network diameter, but network is not connected, adjust
        ne = nx.global_efficiency(G_)
        # dc = nx.betweenness_centrality(G_) # or some other centrality measure, but for whole network
        # as = ... # assortativity, node specific
        ######################################################################
        
        return {'average_clustering': ac, 'network_efficiency': ne, 'density': dens}

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