from param import *
import numpy as np

class PredatorPreySwarmEnvProp(PredatorPreySwarmEnvParam):
    
    ## Useful parameters to customize observations and reward functions

    @property
    def p(self):
        return self._p

    @p.setter
    def p(self, value):
        self._assert_2X_ndarray('p', value)
        self._p = value

    @property
    def dp(self):
        return self._dp

    @dp.setter
    def dp(self, value):
        self._assert_2X_ndarray('dp', value)
        self._dp = value

    @property
    def ddp(self):
        return self._ddp

    @ddp.setter
    def ddp(self, value):
        self._assert_2X_ndarray('ddp', value)
        self._ddp = value

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, value):
        self._assert_1X_ndarray('theta', value)
        self._theta = value

    @property
    def heading(self):
        return self._heading

    @heading.setter
    def heading(self, value):
        self._assert_2X_ndarray('heading', value)
        self._heading = value

    @property
    def d_b2b_center(self):
        return self._d_b2b_center

    @d_b2b_center.setter
    def d_b2b_center(self, value):
        self._d_b2b_center = value

    @property
    def is_collide_b2b(self):
        return self._is_collide_b2b

    @is_collide_b2b.setter
    def is_collide_b2b(self, value):
        self._is_collide_b2b = value

    @property
    def n_e(self):
        return self._n_e

    @n_e.setter
    def n_e(self, value:int):
        self._n_e = value
    
    @property
    def escaper_strategy(self):
        return self._escaper_strategy

    @escaper_strategy.setter
    def escaper_strategy(self, value:str):
        domain = ['input', 'static', 'random', 'nearest']
        if value not in domain:
            raise ValueError(f"reward_sharing_mode must be '{domain}'.")
        self._escaper_strategy = value

    @property
    def m_e(self):
        return self._m_e

    @m_e.setter
    def m_e(self, new_m_e):
        self._m_e = new_m_e

    @property
    def size_e(self):
        return self._size_e

    @size_e.setter
    def size_e(self, value):
        self._size_e = value

    @property
    def linVel_e_max(self):
        return self._linVel_e_max

    @linVel_e_max.setter
    def linVel_e_max(self, value):
        self._linVel_e_max = value


    @property
    def linAcc_e_max(self):
        return self._linAcc_e_max

    @linAcc_e_max.setter
    def linAcc_e_max(self, value):
        self._linAcc_e_max = value
    
    @property
    def angle_e_max(self):
        return self._angle_e_max

    @angle_e_max.setter
    def angle_e_max(self, value):
        self._angle_e_max = value


    @property
    def L(self):
        return self._L

    @L.setter
    def L(self, value):
        self._L = value

    @property
    def k_ball(self):
        return self._k_ball

    @k_ball.setter
    def k_ball(self, value):
        self._k_ball = value

    @property
    def c_aero(self):
        return self._c_aero

    @c_aero.setter
    def c_aero(self, value):
        self._c_aero = value
    
    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        if value > 0.5:
            print("Note: Please exercise caution as the chosen time step may potentially lead to unstable behaviors.")
        self._dt = value
    
    @property
    def render_traj(self):
        return self._render_traj

    @render_traj.setter
    def render_traj(self, value:bool):
        self._render_traj = value

    @property
    def traj_len(self):
        return self._traj_len

    @traj_len.setter
    def traj_len(self, value):
        self._assert_nonnegative_int('traj_len', value)
        self._traj_len = value

    @classmethod
    def _assert_nonnegative_int(cls, name, value):
        if not isinstance(value, int) or value < 0:
            raise TypeError(f" '{name}' must be a non-negative integer ")

    def _assert_2X_ndarray(cls, name, value):
        if not isinstance(value, np.ndarray) or value.shape[0] != 2:
            raise TypeError(f" '{name}' must be a 2-D np.ndarray with shape (2, x)")

    def _assert_1X_ndarray(cls, name, value):
        if not isinstance(value, np.ndarray) or value.shape[0] != 1:
            raise TypeError(f" '{name}' must be a 2-D np.ndarray with shape (1, x)")