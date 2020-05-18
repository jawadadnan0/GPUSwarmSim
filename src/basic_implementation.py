import argparse
import numpy as np

from copy import deepcopy

np.set_printoptions(precision=4)


def main():
    n, l, t, r, v, nu, kappa = parse_args()
#     print(f"""Number of Particles: {n}
# Periodic Spatial Domain: {l}
# Total No. of Iterations: {t}
# Interaction Radius: {r}
# Initial Particle velocity: {v}
# Jump Rate: {nu}
# Concentration Parameter: {kappa}""")

    dt = 0.01 / nu
    max_iter = np.floor(t / dt).astype(int)
    scaled_velocity = l * v
    rr = l / np.floor(l / r)
    pos = l * np.random.random((n, 2))
    vel = 2 * np.pi * np.random.random((n, 1))
    scale = l / 60

#     print(f"""Time Discretisation Step: {dt}
# Max Iteration: {max_iter}
# Scaled Velocity of Particles: {scaled_velocity}
# Scaled Interaction Radius: {rr}
# Positions of the Particles:
# {pos}
# Direction of the Motion of Particles:
# {vel}""")

    particle_map = np.full((int(l / rr), int(l / rr)), np.nan).astype(np.object)
    index = index_map(pos, rr)
    particle_map = fill_map(particle_map, index)

    for t in range(max_iter):

        particle_map_old = deepcopy(particle_map)
        index_old = deepcopy(index)
        pos_old = deepcopy(pos)
        vel_old = deepcopy(vel)

        jump = np.random.uniform(size=(n, 1))
        who = np.where(jump > np.exp(-nu * dt), 1, 0)
        target = vel_old
        target[np.where(who[:, 0] == 1)] = average_orientation(pos_old, vel_old,
                                                               index_old[np.where(who[:, 0] == 1)],
                                                               particle_map_old, r)


def average_orientation(pos, vel, index, particle_map, r):
    K = particle_map.shape[0]
    n = index.shape[0]
    ao = np.zeros(shape=(n, 1))
    for i in range(n):
        I1 = [(index[i, 1] - 1) % K, index[i, 1] % K, (index[i, 1] + 1) % K]
        I2 = [(index[i, 2] - 1) % K, index[i, 2] % K, (index[i, 2] + 1) % K]

        neighbours_map = np.asarray([[particle_map[x, y] for x in I1] for y in I2])
        neighbours = flatten(neighbours_map)
        result = np.linalg.norm(pos[neighbours, :] - pos[index[i, 0], :], ord=2, axis=1, keepdims=True)
        true_neighbours = neighbours[np.where(result < r)[0]]
        target = np.sum(np.c_[np.sin(vel[true_neighbours]), np.cos(vel[true_neighbours])], axis=0)
        ao[i, 0] = np.angle(target[0] + 1.0j * target[1])
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
        # print("Before: ", particle_map[index[i, 1], index[i, 2]])
        if np.all(np.isnan(particle_map[index[i, 1], index[i, 2]])):
            particle_map[index[i, 1], index[i, 2]] = [index[i, 0]]
        else:
            particle_map[index[i, 1], index[i, 2]] = np.r_[index[i, 0], particle_map[index[i, 1], index[i, 2]]]
        # print("After: ", particle_map[index[i, 1], index[i, 2]])
    return particle_map


def index_map(pos, r):
    return np.c_[np.arange(len(pos)), np.floor(pos / r)].astype(int)


def parse_args():
    parser = argparse.ArgumentParser(description="Depicting the movement of several quaternions in a 3D space")

    parser.add_argument("-n", "--agents_num", type=int, default=500, help="The Number of Agents")
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
