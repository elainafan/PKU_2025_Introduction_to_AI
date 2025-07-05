from typing import List
import numpy as np
from utils import Particle

### 可以在这里写下一些你需要的变量和函数 ###
COLLISION_DISTANCE = 1
MAX_ERROR = 50000
K=1.08
### 可以在这里写下一些你需要的变量和函数 ###


def generate_uniform_particles(walls, N):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    N: int, 采样点数量
    输出：
    particles: List[Particle], 返回在空地上均匀采样出的N个采样点的列表，每个点的权重都是1/N
    """
    all_particles: List[Particle] = []
    for _ in range(N):
        all_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    x_max=np.max(walls,axis=0)[0]
    y_max=np.max(walls,axis=0)[1]
    for i in range(N):
        x_random=np.random.uniform(0,x_max)
        y_random=np.random.uniform(0,y_max)
        theta_random=np.random.uniform(-np.pi,np.pi)
        all_particles[i]=Particle(x_random,y_random,theta_random,1/N)
    ### 你的代码 ###
    return all_particles


def calculate_particle_weight(estimated, gt):
    """
    输入：
    estimated: np.array, 该采样点的距离传感器数据
    gt: np.array, Pacman实际位置的距离传感器数据
    输出：
    weight, float, 该采样点的权重
    """
    weight = 1.0
    ### 你的代码 ###
    weight=np.exp(-K*np.linalg.norm(estimated-gt))
    ### 你的代码 ###
    return weight


def resample_particles(walls, particles: List[Particle]):
    """
    输入：
    walls: 维度为(xxx, 2)的np.array, 地图的墙壁信息，具体设定请看README关于地图的部分
    particles: List[Particle], 上一次采样得到的粒子，注意是按权重从大到小排列的
    输出：
    particles: List[Particle], 返回重采样后的N个采样点的列表
    """
    resampled_particles: List[Particle] = []
    for _ in range(len(particles)):
        resampled_particles.append(Particle(1.0, 1.0, 1.0, 0.0))
    ### 你的代码 ###
    noise_dd=0.09
    x_max=np.max(walls,axis=0)[0]
    y_max=np.max(walls,axis=0)[1]
    resample_num_list=[(int)(particl.weight*len(particles)) for particl in particles]
    temp_sum=0
    for i in range(len(particles)):
        for j in range(resample_num_list[i]):
            ddx=np.random.normal(0,noise_dd)
            ddy=np.random.normal(0,noise_dd)
            ddtheta=np.random.normal(0,noise_dd)
            resampled_particles[temp_sum]=Particle(particles[i].position[0]+ddx,particles[i].position[1]+ddy,particles[i].theta+ddtheta,1.0/len(particles))
            temp_sum+=1
    for i in range(temp_sum,len(particles)):
        x_random=np.random.uniform(0,x_max)
        y_random=np.random.uniform(0,y_max)
        theta_random=np.random.uniform(-np.pi,np.pi)
        resampled_particles[i]=Particle(x_random,y_random,theta_random,1.0/len(particles))
    ### 你的代码 ###
    return resampled_particles

def apply_state_transition(p: Particle, traveled_distance, dtheta):
    """
    输入：
    p: 采样的粒子
    traveled_distance, dtheta: ground truth的Pacman这一步相对于上一步运动方向改变了dtheta，并移动了traveled_distance的距离
    particle: 按照相同方式进行移动后的粒子
    """
    ### 你的代码 ###
    temp_theta=p.theta+dtheta
    p.position[0]=p.position[0]+np.cos(temp_theta)*traveled_distance
    p.position[1]=p.position[1]+np.sin(temp_theta)*traveled_distance
    p.theta=temp_theta
    ### 你的代码 ###
    return p

def get_estimate_result(particles: List[Particle]):
    """
    输入：
    particles: List[Particle], 全部采样粒子
    输出：
    final_result: Particle, 最终的猜测结果
    """
    final_result = Particle()
    ### 你的代码 ###
    particle_weight=[particle.weight for particle in particles]
    final_result=particles[particle_weight.index(max(particle_weight))]
    ### 你的代码 ###
    return final_result