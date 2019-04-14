# **********************************************************************
#
# Copyright (c) 2019, Autonomous Systems Lab
# Author: Dongho Kang <eastsky.kang@gmail.com>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in
#    the documentation and/or other materials provided with the
#    distribution.
# 3. Neither the name PX4 nor the names of its contributors may be
#    used to endorse or promote products derived from this software
#    without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# *************************************************************************
import numpy as np
import os

from gym_reinmav.envs.mujoco import MujocoQuadEnv


class MujocoQuadQuaternionEnv(MujocoQuadEnv):
    def __init__(self):
        super(MujocoQuadQuaternionEnv, self).__init__(xml_name="quadrotor_quat.xml")

        # reward weights
        self.position_error_penalty_weight = -10.
        self.velocity_error_penalty_weight = -0.1
        self.action_penalty = -0.1
        self.alive_bonus = 10.

        # goal position 
        self.target_position = np.array([0., 0., 1.])

        # terminate condition
        self.terminate_height = np.zeros(2)     # low, high
        self.terminate_height[0] = -1.0
        self.terminate_height[1] = 2.0

    def step(self, a):
        self.do_simulation(self.clip_action(a), self.frame_skip)
        ob = self._get_obs()

        reward = + np.linalg.norm(ob[0:3] - self.target_position) * self.position_error_penalty_weight \
                 + np.linalg.norm(ob[7:] - np.zeros(6)) * self.velocity_error_penalty_weight \
                 + np.linalg.norm(a) * self.action_penalty \
                 + self.alive_bonus

        notdone = np.isfinite(ob).all() \
                  and ob[2] > self.terminate_height[0] \
                  and ob[2] < self.terminate_height[1] \
                  and abs(ob[0]) < 2.0 \
                  and abs(ob[1]) < 2.0

        done = not notdone
        return ob, reward, done, {}
