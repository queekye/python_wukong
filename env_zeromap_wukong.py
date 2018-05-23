from numpy import *

# 任务介绍：
# 小行星探测器在小行星上跳跃行走，目标点始终设定为原点。
# 输入网络的地图为32*32，每像素代表4m*4m的区域，即总共128m*128m的区域
# 为方便网络对地图的理解，设定达到原点附近的半径为8的区域就视为到达目标
# agent-env交互方案：
# agent接收探测器所有角点均离开地面时的状态和相应的global_map,local_map；输出期望的姿态角速度
# env接收agent的action后，无控仿真至碰撞，然后按此时期望姿态下是否有角点在地面以下，向前或向后搜索碰前状态，


# 地图参数
MAP_DIM = 64
PIXEL_METER = 16
DONE_R = 20

# 小行星与探测器的相关参数
M_wheel_max = 0.01
g = array([0, 0, -0.001])
I_star = array([[0.055, 0, 0], [0, 0.055, 0], [0, 0, 0.055]])
I_star_inv = array([[1 / 0.055, 0, 0], [0, 1 / 0.055, 0], [0, 0, 1 / 0.055]])
U = eye(3)
J_wheel = array([[0.002, 0, 0], [0, 0.002, 0], [0, 0, 0.002]])
UJ_inv = array([[500, 0, 0], [0, 500, 0], [0, 0, 500]])
m = 5
l_robot = array([0.2, 0.2, 0.2])
mu = 0.45
# k:地面刚度数据，有待考证
k = 4.7384 * 10 ** 4
recovery_co = 0.95
# mu0:静摩擦系数
mu0 = 0.48
lx = l_robot[0]
ly = l_robot[1]
lz = l_robot[2]
# vertex_b : 体坐标系下初始各角点相对质心的位置，dim:3*8
vertex_b = array([[lx, lx, lx, lx, -lx, -lx, -lx, -lx],
                  [-ly, -ly, ly, ly, -ly, -ly, ly, ly],
                  [-lz, lz, lz, -lz, -lz, lz, lz, -lz]])
MIN_ENERGY = 0.001
MAX_VXY = 1.5
MAX_VZ = 0.2
# RK4算法参数
STEP_LENGTH = 0.001


# check: OK
def mat_q(q):
    mq = array([[q[0], -q[1], -q[2], -q[3]],
                [q[1], q[0], -q[3], q[2]],
                [q[2], q[3], q[0], -q[1]],
                [q[3], -q[2], q[1], q[0]]])
    return mq


# check:OK
def crossMatrix(w):
    wx = array([[0, -w[2], w[1]],
                [w[2], 0, -w[0]],
                [-w[1], w[0], 0]])
    return wx


# check:OK
# 四元数转换为姿态余弦矩阵
def q_to_DCM(q):
    a = q[0] * eye(3) - crossMatrix(q[1:4])
    DCM = dot(reshape(q[1:4], [-1, 1]), reshape(q[1:4], [1, -1])) + dot(a, a)
    return DCM


# check:OK
# 计算惯性系下的各角点参数
def vertex(state):
    XYZc, v, q, w = state[0:3], state[3:6], state[6:10], state[10:13]
    q /= sqrt(q.dot(q))
    DCM = q_to_DCM(q)
    vertex_s = dot(DCM.T, vertex_b) + reshape(XYZc, [-1, 1])
    vertex_high = vertex_s[2, :]
    vertex_v = dot(DCM.T, dot(crossMatrix(w), vertex_b)) + reshape(v, [-1, 1])
    return vertex_s, vertex_high, vertex_v


# check:OK
# 动力学微分方程，flag标记是否发生碰撞，true为在碰撞，
def dynamic(state, flag=ones([8]) < 0, v0=zeros(8)):
    XYZc, v, q, w, w_wheel = state[0:3], state[3:6], state[6:10], state[10:13], state[13:16]
    q /= sqrt(q.dot(q))
    d_w_wheel = array([0, 0, 0])
    F = zeros([3, 8])
    T = zeros([3, 8])
    DCM = q_to_DCM(q)
    vertex_s, vertex_high, vertex_v = vertex(state)
    Teq = cross(w, I_star.dot(w) + U.dot(J_wheel).dot(w_wheel)) + array([0, 0, 0])
    for i in range(0, 8):
        if flag[i]:
            r_one = vertex_b[:, i]
            high = vertex_high[i]
            normal_one = array([0, 0, 1])
            invade_s = -dot(array([0, 0, high]), normal_one)
            if invade_s < 0:
                continue
            vertex_v_one = vertex_v[:, i]
            invade_v = -dot(vertex_v_one, normal_one)
            vt = vertex_v_one - dot(vertex_v_one, normal_one) * normal_one
            vt_value = sqrt(dot(vt, vt))
            if abs(v0[i]) < 0.0001:
                v0[i] = 0.0001 * sign(v0[i])
            c = 0.75 * (1 - recovery_co ** 2) * k * (invade_s ** 1.5) / v0[i]
            Fn_value = k * (invade_s ** 1.5) + c * invade_v
            # if Fn_value < 0:
            #    Fn_value = 1e-8
            Fn = Fn_value * normal_one
            if vt_value >= 0.0001:
                Ft = -mu * Fn_value * vt / vt_value
            else:
                # 具体方程及求解过程见 笔记

                A = eye(3) / m - linalg.multi_dot([crossMatrix(r_one), I_star_inv, crossMatrix(r_one), DCM])
                A_inv = linalg.inv(A)
                b = crossMatrix(r_one).dot(I_star_inv).dot(Teq) + dot(w, r_one) * w - dot(w, w) * r_one
                alpha = (Fn.dot(Fn) + A_inv.dot(b).dot(Fn)) / (A_inv.dot(Fn).dot(Fn))
                Ft = -A_inv.dot(dot(A - alpha * eye(3), Fn) + b)
                Ft_value = sqrt(Ft.dot(Ft))
                if Ft_value >= mu0 * Fn_value:
                    Ft = mu0 * Fn_value * Ft / Ft_value
            F[:, i] = Ft + Fn
            T[:, i] = cross(r_one, DCM.dot(Ft + Fn))
    F = sum(F, 1) + m * g
    T = sum(T, 1)
    M_star = T - Teq
    d_XYZc = v
    d_v = F / m
    d_q = 0.5 * dot(mat_q(q), concatenate((zeros(1), w)))
    d_w = I_star_inv.dot(M_star)
    d_state = concatenate((d_XYZc, d_v, d_q, d_w, d_w_wheel))
    return d_state


# check:OK
def RK4(t, state, step_length=STEP_LENGTH, flag=ones([8]) < 0, v0=zeros(8)):
    h = step_length
    k1 = dynamic(state, flag, v0)
    k2 = dynamic(state + h * k1 / 2, flag, v0)
    k3 = dynamic(state + h * k2 / 2, flag, v0)
    k4 = dynamic(state + h * k3, flag, v0)
    state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    state[6:10] /= linalg.norm(state[6:10])
    t += h
    if state[3:13].dot(state[3:13]) >= 1000:
        raise Exception("Invalid State!")
    return t, state


class Env:
    s_dim = 6
    a_dim = array([4, 3])
    a_bound = array([1, 1, 1, 1, 2, 2, 2])

    def __init__(self):
        self.t = 0
        self.state = array([0, 0, 3, 0.12, -0.08, 0, 1, 0, 0, 0, 0.2, -0.1, 0.15, -1.9, 1.5, -1.2])
        self.state0 = array([0, 0, 3, 0.12, -0.08, 0, 1, 0, 0, 0, 0.2, -0.1, 0.15, -1.9, 1.5, -1.2])

    # check:
    # 设定初始状态，即探测器与地面的撞前状态
    # 暂时不设定难度，（根据初始xy坐标与原点（目标）的距离，分10个难度，默认为最高难度）
    def set_state_seed(self, sd=1):
        random.seed(sd)
        minXY = 3 * PIXEL_METER
        maxXY = MAP_DIM * PIXEL_METER / 2 - 4 * PIXEL_METER
        minVxy = 0.05
        maxVxy = 0.2
        XY_theta = random.random() * 2 * pi
        XY = ((maxXY - minXY) * random.random() + minXY) * array([cos(XY_theta), sin(XY_theta)])
        v_theta = random.random() * 2 * pi
        v_xy = ((maxVxy - minVxy) * random.random() + minVxy) * array([cos(v_theta), sin(v_theta)])
        vz = 0.07 * random.random() + 0.03
        q = random.rand(4)
        q /= linalg.norm(q)
        w = random.rand(3) * 2 - 1
        w_wheel = random.rand(3) * 2 - 1
        zf = -min(dot(q_to_DCM(q).T, vertex_b)[2, ...]) + 0.0001

        self.state = concatenate([XY, array([zf]), v_xy, array([vz]), q, w, w_wheel])
        self.t = 0
        self.state0 = self.state.copy()
        return self.observe_state()

    # check:
    # step env接收agent的action后的状态转移
    # 输入action即碰前角速度与姿态，输出新的state，reward以及标志该次仿真是否达到目的地的done，是否速度过小导致停止的stop
    # 先按假想的无控进行仿真，若探测器在空中时间小于MIN_FLY_TIME,按无控进行仿真
    # ?探测器在空中角动量守恒，应该可以计算出与action对应的飞轮转速？
    def step(self, action):
        pre_t = self.t
        pre_state = self.state.copy()
        v0 = zeros(8)
        vertex_s, vertex_high, vertex_v = vertex(self.state)
        if (vertex_high <= 0).any():
            raise Exception("Invalid pre_state!")
        qf = action[0:4].copy()
        wf = action[4:7].copy()
        # 根据目标姿态求碰撞时间
        zf = -min(dot(q_to_DCM(qf).T, vertex_b)[2, ...])
        t_up = self.state[5] / -g[2]
        z_max = self.state[2] + 0.5 * g[2] * t_up * 2 + self.state[5] * t_up
        if zf >= z_max:
            # 无法到达目标姿态，直接进行无控仿真
            controlled = False
        else:
            # 计算空中飞行时间t_fly
            t_down = sqrt(2 * (zf - z_max) / g[2])
            t_fly = t_down + t_up
            # 按预定状态更新state到碰撞前
            self.t += t_fly
            self.state[0:2] += self.state[3:5] * t_fly
            self.state[2] = zf + 0.000001  # 避免计算误差导致最低点位于地面以下,多加0.000001
            self.state[5] = t_down * g[2]
            # q, w, w_wheel = self.state[6:10], self.state[10:13], self.state[13:16]
            w_wheel_target = random.rand(3) * 2 - 1
            # UJ_inv.dot(q_to_DCM(qf).dot(q_to_DCM(q).T).dot(dot(I_star, w)+U.dot(J_wheel).dot(w_wheel))-I_star.dot(wf))
            self.state[6:10], self.state[10:13], self.state[13:16] = qf, wf, w_wheel_target

            vertex_s, vertex_high, vertex_v = vertex(self.state)
            index = argmin(vertex_high)
            if vertex_v[2, index] > 0:
                # 最低点速度向上，导致最低点不是碰撞点，碰撞姿态会有很大不同
                controlled = False
            else:
                controlled = True

        self.flySim()
        vertex_s, vertex_high, vertex_v = vertex(self.state)
        # 至此，进入碰前状态，最低点刚刚越过地面
        flag = vertex_high <= 0
        if not flag.any():
            raise Exception("Invalid pre_coll_state!")
        v0[flag] = -vertex_v[2, flag]
        action = self.state[6:13].copy()
        pre_energy = self.energy()
        self.collisionSim(flag, v0)
        after_energy = self.energy()
        loss_energy = pre_energy - after_energy
        if loss_energy <= 0:
            raise Exception("Energy improved!")
        stop_bool = self.energy() < MIN_ENERGY
        over_map = linalg.norm(self.state[0:2]) > (MAP_DIM * PIXEL_METER)
        over_speed = linalg.norm(self.state[3:5]) > MAX_VXY or linalg.norm(self.state[5]) > MAX_VZ
        done_bool = linalg.norm(self.state[0:2]) < DONE_R
        reward_value = self.reward(done_bool, stop_bool, pre_state, pre_t, over_speed, over_map)
        return self.observe_state(), reward_value, done_bool or stop_bool or over_map or over_speed, controlled, action

    def observe_state(self):
        o_s = self.state[0:6].copy()
        o_s[0:2] /= (MAP_DIM*PIXEL_METER/2)
        return o_s

    # check:
    # 碰撞仿真，直至所有顶点均与地面脱离
    # 更新t,state
    def collisionSim(self, flag, v0):
        if not flag.any():
            raise Exception("Invalid collisionSim!")
        while flag.any():
            self.t, self.state = RK4(self.t, self.state, STEP_LENGTH, flag, v0)
            vertex_s, vertex_high, vertex_v = vertex(self.state)
            slc1 = logical_and(flag, vertex_high > 0)
            flag[slc1] = False
            v0[slc1] = 0
            slc2 = logical_and(logical_not(flag), vertex_high <= 0)
            flag[slc2] = True
            v0[slc2] = -vertex_v[2, slc2]

    def flySim(self):
        vertex_s, vertex_high, vertex_v = vertex(self.state)
        if (vertex_high <= 0).any():
            raise Exception("Invalid pre_sim_state!")
        pre_t, pre_state = self.t, self.state.copy()
        while (vertex_high > 0).all():
            pre_t, pre_state = self.t, self.state.copy()
            self.t, self.state = RK4(self.t, self.state, STEP_LENGTH * 100)
            vertex_s, vertex_high, vertex_v = vertex(self.state)
        self.t, self.state = pre_t, pre_state.copy()
        vertex_s, vertex_high, vertex_v = vertex(self.state)
        while (vertex_high > 0).all():
            self.t, self.state = RK4(self.t, self.state, STEP_LENGTH)
            vertex_s, vertex_high, vertex_v = vertex(self.state)

    def reward(self, done_bool, stop_bool, pre_state, pre_t, over_speed, over_map):
        def _cos_vec(a, b):
            f = dot(a, b) / (linalg.norm(a) * linalg.norm(b))
            return f

        if done_bool:
            reward_value = 10
        elif over_map:
            reward_value = -3
        elif stop_bool:
            reward_value = (linalg.norm(self.state0[0:2]) - linalg.norm(self.state[0:2])) / \
                           max(linalg.norm(self.state0[0:2]), linalg.norm(self.state[0:2]))
        elif over_speed:
            reward_value = -2
        else:
            d = (linalg.norm(pre_state[0:2]) - linalg.norm(self.state[0:2])) / \
                max(linalg.norm(pre_state[0:2]), linalg.norm(self.state[0:2]))
            c_pre = _cos_vec(-pre_state[0:2], pre_state[3:5])
            c = _cos_vec(-self.state[0:2], self.state[3:5])
            v_xy = linalg.norm(self.state[3:5])
            reward_value = (c - c_pre) + (v_xy*c - v_xy*sqrt(1-c**2)) + d - 0.0001 * (self.t - pre_t)
        return reward_value

    def energy(self):
        v, w = self.state[3:6], self.state[10:13]
        eg = 0.5 * m * dot(v, v) + 0.5 * reshape(w, [1, 3]).dot(I_star).dot(w)
        return eg


if __name__ == '__main__':
    env = Env()
    for i in range(100):
        ep_reward = 0
        ave_w = 0
        r = 0
        step = 0
        sed = random.randint(1, 10000)
        env.set_state_seed(sed)
        for step in range(200):
            act = random.rand(7) * env.a_bound * 2 - env.a_bound
            act[0:4] /= linalg.norm(act[0:4])
            ave_w += linalg.norm(act[4:7])
            next_s, r, done, controlled, real_action = env.step(act)
            ep_reward += r
            if done:
                break
        print("episode: %10d   ep_reward:%10.5f   last_reward:%10.5f  ave_w:%10.5f" % (i, ep_reward, r, ave_w/(step+1)))

