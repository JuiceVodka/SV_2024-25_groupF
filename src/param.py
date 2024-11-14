import gym

class PredatorPreySwarmEnvParam(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 30}

    _n_e = 50
    _is_periodic = True
    _escaper_strategy = 'input'

    _act_dim_escaper = 2
    _m_e = 1
    _size_e = 0.035 
    _linVel_e_max = 0.5
    _linAcc_e_max = 1
    _angle_e_max = 0.5
    
    # Environment
    _L = 1
    _k_ball = 50       # sphere-sphere contact stiffness  N/m 
    _c_aero = 2        # sphere aerodynamic drag coefficient N/m/s

    ## Simulation Steps
    _dt = 0.1
    _n_frames = 1  
    _sensitivity = 1 

    ## Rendering
    _render_traj = True
    _traj_len = 15


def get_param():
    params = PredatorPreySwarmEnvParam.__dict__.keys()
    params = [param for param in params if param.startswith('_') and not param.startswith('__')]
    params = [param[1:] for param in params]
    return params + ['p']

params = get_param()