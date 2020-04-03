import argparse
import numpy as np

np.set_printoptions(precision=3)


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
    max_iter = np.floor(t / dt)
    scaled_velocity = l * v
    rr = l / np.floor(l / r)
    pos = l * np.random.uniform(size=(n, 2))
    vel = 2 * np.pi * np.random.uniform(size=(n, 1))
    scale = l / 60

#     print(f"""Time Discretisation Step: {dt}
# Max Iteration: {max_iter}
# Scaled Velocity of Particles: {scaled_velocity}
# Scaled Interaction Radius: {rr}
# Positions of the Particles:
# {pos}
# Direction of the Motion of Particles:
# {vel}""")

    particle_map = np.zeros((int(l / rr), int(l / rr))).astype(int)
    index = index_map(pos, rr)
    particle_map = fill_map(particle_map, index)

    print(particle_map)


def fill_map(particle_map, index):
    for i in range(len(index)):
        if particle_map[index[i, 1], index[i, 2]] == 0:
            particle_map[index[i, 1], index[i, 2]] = [index[i, 0]]
        else:
            particle_map[index[i, 1], index[i, 2]] = np.c_[index[i, 0], particle_map[index[i, 1], index[i, 2]]]
    return particle_map


def index_map(pos, r):
    return np.c_[np.arange(len(pos)), np.ceil(pos / r)].astype(int)


def parse_args():
    parser = argparse.ArgumentParser(description="Depicting the movement of several quaternions in a 3D space")

    parser.add_argument("-n", "--agents_num", type=int, default=500, help="The Number of Agents")
    parser.add_argument("-l", "--box_size", type=int, default=5, help="The Size of the Box (Periodic Spatial Domain)")
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
