import argparse
import numpy as np
import torch

from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers

np.set_printoptions(precision=4)
if not torch.cuda.is_available():
    raise Exception("CUDA not available to be used for the program.")

gpu_cuda = torch.device("cuda")


def main():
    n, l, t, r, v, nu, kappa = parse_args()
    # print(f"""Hyperparameters:-
    #     Number of Particles: {n}
    #     Periodic Spatial Domain: {l}
    #     Total No. of Iterations: {t}
    #     Interaction Radius: {r}
    #     Initial Particle velocity: {v}
    #     Jump Rate: {nu}
    #     Concentration Parameter: {kappa}""")

    fig, ax = plt.subplots(dpi=200)

    writer = writers['ffmpeg'](fps=15, metadata=dict(artist="Jawad"), bitrate=1800)
    ani = FuncAnimation(fig, update_quiver_frame, frames=process_particles(n, l, t, r, v, nu, kappa),
                        fargs=(ax, l), interval=30, save_count=int(100 * t * nu) + 1, repeat=False)
    ani.save("quiver_basic.mp4", writer=writer)
    # plt.show()


def update_quiver_frame(frame_data, ax, l):
    ax.clear()
    sep = l / 10
    ax.set_xticks(np.arange(0, l + sep, sep))
    ax.set_yticks(np.arange(0, l + sep, sep))
    ax.set_xlim(0, l)
    ax.set_ylim(0, l)

    pos, vel = frame_data
    scale = l / 60

    q = ax.quiver(pos[:, 0].transpose(), pos[:, 1].transpose(),
                  (scale * np.cos(vel)).flatten(), (scale * np.sin(vel)).flatten())
    ax.quiverkey(q, X=0.3, Y=1.1, U=0.05, label='Quiver key, length = 0.05', labelpos='E')


def process_particles(n, l, t, r, v, nu, kappa):
    dt = 0.01 / nu
    max_iter = np.floor(t / dt).astype(int)
    scaled_velocity = l * v
    rr = l / np.floor(l / r)
    pos = torch.mul(torch.rand(n, 2, device=gpu_cuda), l)
    vel = torch.mul(torch.rand(n, 1, device=gpu_cuda), 2 * np.pi)

    # print(f"""Calculated Parameters:-
    #     Time Discretisation Step: {dt}
    #     Max Iteration: {max_iter}
    #     Scaled Velocity of Particles: {scaled_velocity}
    #     Scale: {scale}
    #     Scaled Interaction Radius: {rr}
    #     Positions of the Particles:
    #     {pos}
    #     Direction of the Motion of Particles:
    #     {vel}""")

    particle_map = np.full((int(l / rr), int(l / rr)), np.nan).astype(np.object)
    index = index_map(pos, rr)
    particle_map = fill_map(particle_map, index)

    for t in range(max_iter):
        jump = torch.rand(n, 1, device=gpu_cuda)
        who = torch.where(torch.gt(jump, torch.exp(-nu * dt)), tensor(1, torch.int64), tensor(0, torch.int64))
        target = deepcopy(vel)
        target[torch.where(torch.eq(who[:, 0], 1))] = \
            average_orientation(pos, target, index[torch.where(torch.eq(who[:, 0], 1))], particle_map, r)
        vel[torch.where(torch.eq(who[:, 0], 1))] = \
            torch.remainder(target[torch.where(torch.eq(who[:, 0], 1))] + circ_vmrnd(0, kappa, torch.sum(who).item()),
                            tensor(2 * np.pi, torch.float))
        pos = torch.remainder(pos + torch.mul(torch.cat((torch.cos(vel), torch.sin(vel)), 1), dt * scaled_velocity), l)

        if t % 10 == 0:
            yield pos, vel

        index = index_map(pos, rr)
        particle_map = np.full((int(l / rr), int(l / rr)), np.nan).astype(np.object)
        particle_map = fill_map(particle_map, index)


def circ_vmrnd(theta, kappa, n):
    if kappa < 1e-6:
        return torch.mul(torch.rand(n, 1, device=gpu_cuda), 2 * np.pi).sub(np.pi)

    a = 1 + np.sqrt(1 + 4 * kappa ** 2)
    b = (a - np.sqrt(2 * a)) / (2 * kappa)
    r = (1 + b ** 2) / (2 * b)

    alpha = torch.zeros(n, 1, device=gpu_cuda)
    for j in range(n):
        while True:
            u = np.random.random(size=3)

            z = np.cos(np.pi * u[0])
            f = (1 + r * z) / (r + z)
            c = kappa * (r - f)

            if u[1] < c * (2 - c) or not (np.log(c) - np.log(u[1]) + 1 - c < 0):
                break

        temp = np.mod(theta + np.sign(u[2] - 0.5) * np.arccos(f), 2 * np.pi)
        alpha[j, 0] = tensor(temp - 2 * np.pi if np.pi < temp <= 2 * np.pi else temp, torch.float)

    return alpha


def average_orientation(pos, vel, index, particle_map, r):
    k = particle_map.shape[0]
    n = index.size()[0]
    ao = torch.zeros(n, 1, device=gpu_cuda)
    for i in range(n):
        first_indexes = [(index[i, 1].item() + j) % k for j in range(-1, 2)]
        second_indexes = [(index[i, 2].item() + j) % k for j in range(-1, 2)]

        neighbours = flatten([[particle_map[x, y] for x in first_indexes] for y in second_indexes])
        result = torch.norm(pos[neighbours, :] - pos[index[i, 0], :], p=2, dim=1, keepdim=True)
        true_neighbours = neighbours[torch.where(torch.lt(result, r))[0]]

        target = torch.sum(torch.cat((torch.sin(vel[true_neighbours]), torch.cos(vel[true_neighbours])), 1), 0)
        ao[i, 0] = torch.atan(torch.div(target[1], target[0]))  # angle
    return ao


def flatten(array):
    result = []
    for x in range(len(array)):
        for y in range(len(array[0])):
            try:
                result += list(array[x][y])
            except TypeError:
                result += [array[x][y]]
    return tensor([e for e in result if not np.isnan(e)], torch.int64)


def fill_map(particle_map, index):
    for i in range(len(index)):
        if np.all(np.isnan(particle_map[index[i, 1], index[i, 2]])):
            particle_map[index[i, 1], index[i, 2]] = [index[i, 0]]
        else:
            particle_map[index[i, 1], index[i, 2]] = np.r_[index[i, 0], particle_map[index[i, 1], index[i, 2]]]
    return particle_map


def index_map(pos, r):
    indexes = torch.arange(pos.size()[0], device=gpu_cuda).reshape(pos.size()[0], 1)
    return torch.cat((indexes, torch.floor(pos / r).to(torch.int64)), 1)


def tensor(value, data_type):
    return torch.tensor(value, dtype=data_type, device=gpu_cuda)


def parse_args():
    parser = argparse.ArgumentParser(description="Depicting the movement of several quaternions in a 3D space")

    parser.add_argument("-n", "--agents_num", type=int, default=50000, help="The Number of Agents")
    parser.add_argument("-l", "--box_size", type=int, default=5, help="The Size of the Box (Periodic Spatial Domain)")
    parser.add_argument("-t", "--max_iter", type=int, default=100, help="The Total Number of Iterations")
    parser.add_argument("-r", "--interact_radius", type=float, default=0.07, help="The Radius of Interaction")
    parser.add_argument("-v", "--particle_velocity", type=float, default=0.02, help="The Velocity of the Particles")
    parser.add_argument("-nu", "--jump_rate", type=float, default=0.3, help="The Jump Rate")
    parser.add_argument("-k", "--concentration", type=float, default=20.0, help="The Concentration Parameter")

    args = parser.parse_args()

    return args.agents_num, args.box_size, args.max_iter, args.interact_radius, \
           args.particle_velocity, args.jump_rate, args.concentration


if __name__ == '__main__':
    main()
