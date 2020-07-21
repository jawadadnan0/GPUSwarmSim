import argparse
import numpy as np

from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers

np.set_printoptions(precision=4)


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
                        fargs=(ax, l), interval=10, save_count=t)
    # ani.save("quiver.mp4", writer=writer)
    plt.show()


def update_quiver_frame(frame_data, ax, l):
    ax.clear()
    sep = l / 10
    ax.set_xticks(np.arange(0, l + sep, sep))
    ax.set_yticks(np.arange(0, l + sep, sep))
    ax.set_xlim(0, l)
    ax.set_ylim(0, 1)

    pos, vel = frame_data
    scale = l / 60

    q = ax.quiver(pos[:, 0].transpose(), pos[:, 1].transpose(), scale * np.cos(vel), scale * np.sin(vel))
    ax.quiverkey(q, X=0.3, Y=1.1, U=0.05, label='Quiver key, length = 0.05', labelpos='E')


def process_particles(n, l, t, r, v, nu, kappa):
    random = np.random.RandomState(seed=0)

    dt = 0.01 / nu
    max_iter = np.floor(t / dt).astype(int)
    scaled_velocity = l * v
    rr = l / np.floor(l / r)
    pos = l * random.uniform(size=(n, 2))
    vel = 2 * np.pi * random.uniform(size=(n, 1))

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
        jump = random.uniform(size=(n, 1))
        who = np.where(jump > np.exp(-nu * dt), 1, 0)
        target = deepcopy(vel)
        target[np.where(who[:, 0] == 1)] = average_orientation(pos, target,
                                                               index[np.where(who[:, 0] == 1)],
                                                               particle_map, r)
        vel[np.where(who[:, 0] == 1)] = np.mod(target[np.where(who[:, 0] == 1)] +
                                               circ_vmrnd(0, kappa, who.sum()), 2 * np.pi)
        pos = np.mod(pos + dt * scaled_velocity * np.c_[np.cos(vel), np.sin(vel)], l)

        if t % 10 == 0:
            yield pos, vel

        index = index_map(pos, rr)
        particle_map = np.full((int(l / rr), int(l / rr)), np.nan).astype(np.object)
        particle_map = fill_map(particle_map, index)


def circ_vmrnd(theta=0, kappa=1, n=10):
    if kappa < 1e-6:
        return 2 * np.pi * np.random.random(size=(n, 1)) - np.pi

    a = 1 + np.sqrt((1 + 4 * kappa ** 2))
    b = (a - np.sqrt(2 * a)) / (2 * kappa)
    r = (1 + b ** 2) / (2 * b)

    alpha = np.zeros(shape=(n, 1))
    for j in range(n):
        while True:
            u = np.random.random(size=3)

            z = np.cos(np.pi * u[0])
            f = (1 + r * z) / (r + z)
            c = kappa * (r - f)

            if u[1] < c * (2 - c) or not (np.log(c) - np.log(u[1]) + 1 - c < 0):
                break

        alpha[j, 0] = np.mod(theta + np.sign(u[2] - 0.5) * np.arccos(f), 2 * np.pi)
        alpha[j, 0] = alpha[j, 0] - 2 * np.pi if np.pi < alpha[j, 0] <= 2 * np.pi else alpha[j, 0]

    return alpha


def average_orientation(pos, vel, index, particle_map, r):
    k = particle_map.shape[0]
    n = index.shape[0]
    ao = np.zeros(shape=(n, 1))
    for i in range(n):
        first_indexes = [(index[i, 1] - 1) % k, index[i, 1] % k, (index[i, 1] + 1) % k]
        second_indexes = [(index[i, 2] - 1) % k, index[i, 2] % k, (index[i, 2] + 1) % k]

        neighbours_map = np.array([[particle_map[x, y] for x in first_indexes] for y in second_indexes], dtype=np.object)
        neighbours = flatten(neighbours_map)

        result = np.linalg.norm(pos[neighbours, :] - pos[index[i, 0], :], ord=2, axis=1, keepdims=True)
        true_neighbours = neighbours[np.where(result < r)[0]]

        target = np.sum(np.c_[np.sin(vel[true_neighbours]), np.cos(vel[true_neighbours])], axis=0)
        ao[i, 0] = np.arctan(target[1] / target[0])
    return ao


def flatten(array):
    result = []
    rows, cols = array.shape
    for x in range(rows):
        for y in range(cols):
            try:
                result += list(array[x, y])
            except TypeError:
                result += [array[x, y]]
    return np.array(list(filter(lambda e: not np.isnan(e), result)))


def fill_map(particle_map, index):
    for i in range(len(index)):
        if np.all(np.isnan(particle_map[index[i, 1], index[i, 2]])):
            particle_map[index[i, 1], index[i, 2]] = [index[i, 0]]
        else:
            particle_map[index[i, 1], index[i, 2]] = np.r_[index[i, 0], particle_map[index[i, 1], index[i, 2]]]
    return particle_map


def index_map(pos, r):
    return np.c_[np.arange(len(pos)), (np.floor(pos / r))].astype(int)


def parse_args():
    parser = argparse.ArgumentParser(description="Depicting the movement of several quaternions in a 3D space")

    parser.add_argument("-n", "--agents_num", type=int, default=5000, help="The Number of Agents")
    parser.add_argument("-l", "--box_size", type=int, default=1, help="The Size of the Box (Periodic Spatial Domain)")
    parser.add_argument("-t", "--max_iter", type=int, default=5000, help="The Total Number of Iterations")
    parser.add_argument("-r", "--interact_radius", type=float, default=0.07, help="The Radius of Interaction")
    parser.add_argument("-v", "--particle_velocity", type=float, default=0.02, help="The Velocity of the Particles")
    parser.add_argument("-nu", "--jump_rate", type=float, default=0.3, help="The Jump Rate")
    parser.add_argument("-k", "--concentration", type=float, default=20.0, help="The Concentration Parameter")

    args = parser.parse_args()

    return args.agents_num, args.box_size, args.max_iter, args.interact_radius, \
           args.particle_velocity, args.jump_rate, args.concentration


if __name__ == '__main__':
    main()
