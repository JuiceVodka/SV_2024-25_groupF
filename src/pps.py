from gym import spaces
from putils import *
from prop import *
from gym.utils import *
import numpy as np
from customize_pps import PredatorPreySwarmCustomizer
import os

class PredatorPreySwarmEnv(PredatorPreySwarmEnvProp):
    param_list = params

    def __init__(self, n_e=10):
        self.n_e = n_e
        self.viewer = None
        self.seed()
    
    def __reinit__(self):
        
        self.observation_space = self._get_observation_space()  
        self.action_space = self._get_action_space()   
        self._m = get_mass(self._m_e, self._n_e)  
        self._size, self._sizes = get_sizes(self._size_e, self._n_e)  

        self._linAcc_e_min = 0


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def reset(self):
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
            self._p_traj = np.zeros((self._traj_len, 2, self.n_e))
            self._p_traj[0,:,:] = self._p
        self._dp = np.zeros((2, self.n_e))                  
        self._ddp = np.zeros((2, self.n_e))                                     
        self._theta = np.pi * np.random.uniform(-1,1, (1, self.n_e))
        self._heading = np.concatenate((np.cos(self._theta), np.sin(self._theta)), axis=0)  
        return self._get_obs()


    def _get_obs(self):
        self.obs = np.zeros(self.observation_space.shape)   
        return self.obs

    def _get_reward(self, a):        
        reward = np.zeros((1, self.n_e))
        return reward

    def _get_done(self):
        all_done = np.zeros((1, self.n_e)).astype(bool)
        return all_done

    def _get_info(self):
        return np.zeros((1, self.n_e))

    def step(self, a):  
        for _ in range(self._n_frames): 
            a[0, :] *= self._angle_e_max
            a[1, :] = (self._linAcc_e_max-self._linAcc_e_min)/2 * a[1, :] + (self._linAcc_e_max+self._linAcc_e_min)/2 

            self._d_b2b_center, self.d_b2b_edge, self._is_collide_b2b = get_dist_b2b(self._p, self._L, self._is_periodic, self._sizes)
            sf_b2b_all = np.zeros((2*self.n_e, self.n_e)) 
            for i in range(self.n_e):
                for j in range(i):
                    delta = self._p[:,j]-self._p[:,i]
                    delta = make_periodic(delta, self._L)
                    dir = delta / self._d_b2b_center[i,j]
                    sf_b2b_all[2*i:2*(i+1),j] = self._is_collide_b2b[i,j] * self.d_b2b_edge[i,j] * self._k_ball * (-dir)
                    sf_b2b_all[2*j:2*(j+1),i] = - sf_b2b_all[2*i:2*(i+1),j]  

            sf_b2b = np.sum(sf_b2b_all, axis=1, keepdims=True).reshape(self.n_e,2).T 
    
            if self.escaper_strategy == 'input':
                pass
            elif self.escaper_strategy == 'static':
                a = np.zeros((self._act_dim_escaper, self.n_e))
            elif self.escaper_strategy == 'random':
                a = np.random.uniform(-1,1, (self._act_dim_escaper, self.n_e)) 
                a[0, :] *= self._angle_e_max
                a[1, :] = (self._linAcc_e_max-self._linAcc_e_min)/2 * a[1, :] + (self._linAcc_e_max+self._linAcc_e_min)/2 
            else:
                print('Wrong in Step function')    

            self._theta += a[[0],:]
            self._theta = normalize_angle(self._theta)
            self._heading = np.concatenate((np.cos(self._theta), np.sin(self._theta)), axis=0) 
            u = a[[1], :] * self._heading 
            F = self._sensitivity * u  + sf_b2b - self._c_aero*self._dp
            self._ddp = F/self._m
            self._dp += self._ddp * self._dt
            self._dp = np.clip(self._dp, -self._linVel_e_max, self._linVel_e_max)
            self._p += self._dp * self._dt
            self._p = make_periodic(self._p, self._L)
            if self._render_traj == True:
                self._p_traj = np.concatenate( (self._p_traj[1:,:,:], self._p.reshape(1, 2, self.n_e)), axis=0 )
        return self._get_obs(), self._get_reward(a), self._get_done(), self._get_info()

    def render(self, mode="human"): 
    
        if self.viewer is None:
            import rendering
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(-1, 1, -1, 1.)
            
            agents = []
            self.tf = []
            if self._render_traj: self.trajrender = []
            for i in range(self.n_e):
                if self._render_traj: self.trajrender.append( rendering.Traj( list(zip(self._p_traj[:,0,i], self._p_traj[:,1,i])),  False) )
                agents.append( rendering.make_unicycle(self._size_e) )
                agents[i].set_color_alpha(0, 0.333, 0.778, 1)
                if self._render_traj: self.trajrender[i].set_color_alpha(0, 0.333, 0.778, 0.5)
                self.tf.append( rendering.Transform() )
                agents[i].add_attr(self.tf[i])
                self.viewer.add_geom(agents[i])
                if self._render_traj: self.viewer.add_geom(self.trajrender[i])

        for i in range(self.n_e):
            self.tf[i].set_rotation(self._theta[0,i])
            self.tf[i].set_translation(self._p[0,i], self._p[1,i])
            if self._render_traj: self.trajrender[i].set_traj(list(zip(self._p_traj[:,0,i], self._p_traj[:,1,i])))

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
    
    def _get_observation_space(self):
        self.obs_dim_escaper = ( 2 + 2 * 6 ) * 2  + 3   
        observation_space = spaces.Box(low=-np.inf, high=+np.inf, shape=(self.obs_dim_escaper, self.n_e), dtype=np.float32)
        return observation_space

    def _get_action_space(self):
        action_space = spaces.Box(low=-1, high=1, shape=(self._act_dim_escaper, self.n_e), dtype=np.float32)
        return action_space


if __name__ == '__main__':
    env = PredatorPreySwarmEnv()
    custom_param = 'env_test_custom_param.json'
    custom_param = os.path.dirname(os.path.realpath(__file__)) + '/' + custom_param
    env = PredatorPreySwarmCustomizer(env, custom_param)
    s = env.reset()   # (obs_dim, n_e)
    for _ in range(1):
        for step in range(1000):
            img = env.render( mode='rgb_array')
            a = env.action_space.sample()
            s_, r, done, info = env.step(a)
            s = s_.copy()