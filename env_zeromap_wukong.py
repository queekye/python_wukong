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
PIXEL_METER = 32

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
MAX_VZ = 0.3
# RK4算法参数
STEP_LENGTH = 0.001


def wheel_model(M, w):
    flag = abs(w) >= 100
    w[flag] = 0
    M = minimum(0.1, maximum(-0.1, M))
    return M, w


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
    q_norm = q / max(array([sqrt(q.dot(q)), 1e-8]))
    DCM = q_to_DCM(q_norm)
    vertex_s = dot(DCM.T, vertex_b) + reshape(XYZc, [-1, 1])
    vertex_high = vertex_s[2, :]
    vertex_v = dot(DCM.T, dot(crossMatrix(w), vertex_b)) + reshape(v, [-1, 1])
    return vertex_s, vertex_high, vertex_v


# check:OK
# 动力学微分方程，flag标记是否发生碰撞，true为在碰撞，
def dynamic(state, v0=zeros(8)):
    XYZc, v, q, w, w_wheel = state[0:3], state[3:6], state[6:10], state[10:13], state[13:16]
    q /= max(array([sqrt(q.dot(q)), 1e-8]))
    M_wheel, w_wheel = wheel_model(array([0, 0, 0]), w_wheel)
    d_w_wheel = UJ_inv.dot(M_wheel)
    F = zeros([3, 8])
    T = zeros([3, 8])
    DCM = q_to_DCM(q)
    vertex_s, vertex_high, vertex_v = vertex(state)
    normal = zeros([3, 8])
    normal[2, :] = 1
    vn = vertex_v[2, :]
    vt = vertex_v.copy()
    vt[2, :] = 0
    Teq = cross(w, I_star.dot(w) + U.dot(J_wheel).dot(w_wheel))
    for j in range(0, 8):
        if vertex_high[j] < 0:
            if v0[j] < 0.1:
                # print("small v")
                v0[j] = 0.1

            r_one = vertex_b[:, j]
            high = vertex_high[j]
            normal_one = array([0, 0, 1])
            invade_s = -high
            invade_v = -vn[j]
            vt_one = vt[:, j]
            vt_value = sqrt(dot(vt_one, vt_one))
            c = 0.75 * (1 - recovery_co ** 2) * k * (invade_s ** 1.5) / v0[j]
            Fn_value = k * (invade_s ** 1.5) + c * invade_v
            Fn = Fn_value * normal_one
            if vt_value >= 0.0006:
                Ft = -mu * Fn_value * vt_one / vt_value
            else:
                # 具体方程及求解过程见 笔记

                A = eye(3) / m - linalg.multi_dot([crossMatrix(r_one), I_star_inv, crossMatrix(r_one), DCM])
                A_inv = linalg.inv(A)
                b = crossMatrix(r_one).dot(I_star_inv).dot(Teq) + dot(w, r_one) * w - dot(w, w) * r_one
                alpha = (Fn.dot(Fn) + A_inv.dot(b).dot(Fn)) / (A_inv.dot(Fn).dot(Fn))
                Ft = -A_inv.dot(dot(A - alpha * eye(3), Fn) + b)
                Ft_value = sqrt(Ft.dot(Ft))
                if Ft_value >= mu0 * abs(Fn_value):
                    Ft = mu0 * abs(Fn_value) * vt_one / vt_value
                Ft[2] = 0
            F[:, j] = Ft + Fn
            T[:, j] = cross(r_one, DCM.dot(Ft + Fn))
        else:
            v0[j] = -vn[j]
    F = sum(F, 1) + m * g
    T = sum(T, 1)
    M_star = T - Teq
    d_XYZc = v
    d_v = F / m
    d_q = 0.5 * dot(mat_q(q), concatenate((zeros(1), w)))
    d_w = I_star_inv.dot(M_star)
    d_state = concatenate((d_XYZc, d_v, d_q, d_w, d_w_wheel))
    return d_state, v0


# check:OK
def RK4(t, state, step_length=STEP_LENGTH, v0=zeros(8)):
    h = step_length
    k1, v1 = dynamic(state.copy(), v0.copy())
    k2, v2 = dynamic(state.copy() + h * k1 / 2, v1.copy())
    k3, v3 = dynamic(state.copy() + h * k2 / 2, v2.copy())
    k4, v4 = dynamic(state.copy() + h * k3, v3.copy())
    state += h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    state[6:10] /= max(array([linalg.norm(state[6:10]), 1e-8]))
    t += h
    if state[3:13].dot(state[3:13]) >= 10000:
        print("Invalid State!")
    return t, state, v4.copy()


# check:OK
def Euler(t, state, step_length, v0):
    h = step_length
    k1, v0 = dynamic(state, v0.copy())
    state += h * k1
    state[6:10] /= linalg.norm(state[6:10])
    t += h
    return t, state, v0


class Env:
    s_dim = 6
    a_dim = array([4, 3])
    a_bound = array([1, 1, 1, 1, 2, 2, 2])

    def __init__(self):
        self.t = 0
        self.state = array([0, 0, 3, 0.12, -0.08, 0, 1, 0, 0, 0, 0.2, -0.1, 0.15, -1.9, 1.5, -1.2])
        self.state0 = array([0, 0, 3, 0.12, -0.08, 0, 1, 0, 0, 0, 0.2, -0.1, 0.15, -1.9, 1.5, -1.2])
        self.r_obj = 20

    def cut_r_obj(self):
        self.r_obj /= 2

    # check:
    # 设定初始状态，即探测器最高点状态
    # 暂时不设定难度，（根据初始xy坐标与原点（目标）的距离，分10个难度，默认为最高难度）
    def set_state_seed(self, sd=1):
        random.seed(sd)
        minXY = 3 * PIXEL_METER
        maxXY = MAP_DIM * PIXEL_METER * 3 / 8
        minVxy = 0.05
        maxVxy = 0.2
        XY_theta = random.random() * 2 * pi
        XY = ((maxXY - minXY) * random.random() + minXY) * array([cos(XY_theta), sin(XY_theta)])
        v_theta = random.random() * 2 * pi
        v_xy = ((maxVxy - minVxy) * random.random() + minVxy) * array([cos(v_theta), sin(v_theta)])
        vz = 0  # -0.07 * random.random() - 0.03
        q = random.rand(4)
        q /= max(array([sqrt(q.dot(q)), 1e-8]))
        w = random.rand(3) * 2 - 1
        w_wheel = random.rand(3) * 2 - 1

        DCM = q_to_DCM(q)
        s_dot = dot(DCM.T, vertex_b)
        minZ = 2
        maxZ = 10
        # zf = -min(s_dot[2, :])
        zf = (maxZ - minZ) * random.random() + minZ

        self.state = concatenate([XY, array([zf]), v_xy, array([vz]), q, w, w_wheel])
        self.t = 0
        self.state0 = self.state.copy()
        return self.observe_state()

    def step(self, action):
        overtime = False
        pre_t = self.t
        pre_state = self.state.copy()
        v0 = self.relocation(action)
        validated = self.validate(action)
        successed, stop_bool, over_speed, over_map = False, False, False, False
        if validated:
            pre_energy = self.energy()
            overtime = self.collisionSim(v0)
            after_energy = self.energy()
            loss_energy = after_energy - pre_energy
            if loss_energy > 0:
                print("energy improved", loss_energy)
            if not overtime:
                t_fly = abs(self.state[5] / g[2])
                self.state[0:2] += self.state[3:5] * t_fly
                self.state[2] += 0.5*abs(g[2])*t_fly**2
                self.state[5] = 0
                XY_luo = self.state[0:2] + self.state[3:5] * t_fly
                stop_bool = self.energy() < MIN_ENERGY or linalg.norm(self.state[3:5]) < 1e-2 or abs(
                    g[2]*t_fly) < 1e-2
                over_map = linalg.norm(XY_luo) > (MAP_DIM * PIXEL_METER / 2)
                over_speed = linalg.norm(self.state[3:5]) > MAX_VXY or self.state[2] > 20
                successed = linalg.norm(self.state[0:2]) < self.r_obj

        reward_value = self.reward(successed, stop_bool, pre_state, pre_t, over_speed, over_map, validated, overtime)
        done = successed or stop_bool or over_map or over_speed or overtime or (not validated)
        return self.observe_state(), reward_value, done

    def observe_state(self):
        o_s = self.state[0:5].copy()
        o_s[0:2] /= 100
        o_s[2] /= 5
        o_s[3:5] /= 0.1
        return o_s

    def validate(self, action):
        q, w = action[0:4].copy(), action[4:7].copy()
        v = self.state[3:6].copy()
        q_norm = q / max(array([sqrt(q.dot(q)), 1e-8]))
        DCM = q_to_DCM(q_norm)
        vertex_v = dot(DCM.T, dot(crossMatrix(w), vertex_b)) + reshape(v, [-1, 1])
        rdot = dot(DCM.T, vertex_b)[2, :]
        index = argmin(rdot)
        validated = vertex_v[2, index] < 0
        # 最低点速度向上，导致最低点不是碰撞点，碰撞姿态会有很大不同
        return validated

    def collisionSim(self, v0):
        t0 = self.t
        t_max = 20
        while (self.state[2] <= 0.2 * sqrt(3) or self.state[5] <= 0) and self.t - t0 < t_max:
            self.t, self.state, v0 = RK4(self.t, self.state, STEP_LENGTH * 10, v0)
        overtime = self.t - t0 >= t_max
        return overtime

    def relocation(self, action):
        qf, wf = action[0:4].copy(), action[4:7].copy()
        zf = -min(dot(q_to_DCM(qf).T, vertex_b)[2, :])
        t_d2 = sqrt(2 * (zf - self.state[2]) / g[2])

        self.t += t_d2
        self.state[0:2] += self.state[3:5] * t_d2
        self.state[2] = zf
        self.state[5] = t_d2 * g[2]
        self.state[6:13] = action.copy()
        self.state[13:16] = random.rand(3) * 2 - 1
        vertex_v = dot(q_to_DCM(qf).T, dot(crossMatrix(wf), vertex_b)) + reshape(self.state[3:6].copy(), [-1, 1])
        v0 = -vertex_v[2, :]
        return v0

    def reward(self, successed, stop_bool, pre_state, pre_t, over_speed, over_map, validated, overtime):
        def _cos_vec(a, b):
            f = dot(a, b) / max(array([linalg.norm(a) * linalg.norm(b), 1e-8]))
            return f

        if not validated:
            reward_value = -3
        elif successed:
            reward_value = 10
        elif over_speed:
            reward_value = -2.5
        elif over_map:
            reward_value = -2
        elif stop_bool or overtime:
            reward_value = (linalg.norm(self.state0[0:2]) - linalg.norm(self.state[0:2])) / \
                           max(linalg.norm(self.state0[0:2]), linalg.norm(self.state[0:2]))
        else:
            d = (linalg.norm(pre_state[0:2]) - linalg.norm(self.state[0:2])) / \
                max(linalg.norm(pre_state[0:2]), linalg.norm(self.state[0:2]))
            c_pre = _cos_vec(-pre_state[0:2], pre_state[3:5])
            c = _cos_vec(-self.state[0:2], self.state[3:5])
            v_xy = linalg.norm(self.state[3:5])
            reward_value = (c - c_pre) + (v_xy * c - v_xy * sqrt(1 - c ** 2)) + d  # - 0.0001 * (self.t - pre_t)
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
        env.set_state_seed(i)
        for step in range(200):
            act = random.rand(7) * env.a_bound * 2 - env.a_bound
            act[0:4] /= linalg.norm(act[0:4])
            act[4:7] /= 10
            ave_w += linalg.norm(act[4:7])
            next_s, r, d = env.step(act)
            ep_reward += r
            if d:
                break
        print("episode: %10d   step_num:%10d    ep_reward:%10.5f   last_reward:%10.5f  ave_w:%10.5f"
              % (i, step + 1, ep_reward, r, ave_w / (step + 1)))

    env = Env()
    act = array([0.64, 0.48, 0.36, 0.48, -0.6, -0.25, 0.85])
    act *= env.a_bound
    s_, r, d = env.step(act)
    print(s_, r, d)
