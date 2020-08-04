import argparse
import numpy as np

from copy import deepcopy
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.axes import Axes
from typing import Any, Generator, List, Tuple

np.set_printoptions(precision=4)


def main() -> None:
    """
    The main function that parses the program arguments using parse_args(),
    creates a plot for a moving quiver diagram, sets up a writer for the video
    file with ffmpeg format with metadata, creates the animation by calling
    process_particles() generator to recieve frame data, passes it along to
    update_quiver_frame() function to draw out a frame and this gets either
    shown in a window or saved to "quiver_basic.mp4" to create a video.

    Returns: None (void function)

    """
    save, file, n, l, t, r, v, nu, kappa = parse_args()
    # print(file"""Hyperparameters:-
    #     Save to File: {save}
    #     Save File Name: {file}
    #     Number of Particles: {n}
    #     Periodic Spatial Domain: {l}
    #     Total No. of Iterations: {t}
    #     Interaction Radius: {r}
    #     Initial Particle velocity: {v}
    #     Jump Rate: {nu}
    #     Concentration Parameter: {kappa}""")
    current_time = datetime.now()
    fig, ax = plt.subplots(dpi=300)

    writer = writers['ffmpeg'](fps=15, metadata=dict(artist="Jawad"), bitrate=1800)
    ani = FuncAnimation(fig, update_quiver_frame, frames=process_particles(n, l, t, r, v, nu, kappa),
                        fargs=(ax, l), interval=30, save_count=int(100 * t * nu) + 1, repeat=False)

    if save:
        ani.save(file, writer=writer)
        print("[100% Complete] Time taken:", (datetime.now() - current_time).seconds, "seconds")
    else:
        plt.show()


def update_quiver_frame(frame_data: Tuple[np.ndarray, np.ndarray], ax: Axes, l: int) -> None:
    """
    This function is executed every single time the frame needs to updated
    whether it is to view it in real-time or to save it into a video.

    Args:
        frame_data: A tuple of two numpy arrays, containing the positions and velocities of the particles.
        ax: The axis object of the plot in order to help set it up.
        l: The length of the square to be drawn that will contain the particles.

    Returns: None (void function)

    """
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
    ax.quiverkey(q, X=0.3, Y=1.1, U=0.05,
                 label=f"Quiver key, length = 0.05 - Particles: {pos.shape[0]:,}", labelpos='E')


def process_particles(n: int, l: int, t: int, r: float, v: float, nu: float, kappa: float) -> \
        Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """
    This function calculates the positions and velocities of each of the 'n' particles
    for '100 * t * nu' iterations where the positions and velocities calculated are
    sent back after every 10 iterations.

    Args:
        n: The number of particles in the simulation.
        l: The length of edge of the box (Periodic Spatial Domain).
        t: The total number of iterations/seconds the simulation is played for.
        r: The radius of interaction each particle has on others.
        v: The scale of the velocity of the particles.
        nu: The jump time for each of the particles.
        kappa: The concentration parameter for the simulation.

    Returns: A generator which yields a tuple of two numpy arrays for position and velocity.

    """
    random = np.random.RandomState(seed=0)

    dt = 0.01 / nu
    max_iter = np.floor(t / dt).astype(int) * 5
    scaled_velocity = l * v
    rr = l / np.floor(l / r)
    pos = l * random.uniform(size=(n, 2))
    vel = 2 * np.pi * random.uniform(size=(n, 1))

    # print(f"""Calculated Parameters:-
    #     Time Discretisation Step: {dt}
    #     Max Iteration: {max_iter}
    #     Scaled Velocity of Particles: {scaled_velocity}
    #     Scale: {scale}
    #     Scaled Interaction Radius: {rr}""")

    index = index_map(pos, rr)
    particle_map = fill_map(int(l / rr), index)

    for t in range(max_iter + 1):
        jump = random.uniform(size=(n, 1))
        who = np.where(jump > np.exp(-nu * dt), 1, 0)
        target = deepcopy(vel)
        target[np.where(who[:, 0] == 1)] = \
            average_orientation(pos, target, index[np.where(who[:, 0] == 1)], particle_map, r)
        vel[np.where(who[:, 0] == 1)] = \
            np.mod(target[np.where(who[:, 0] == 1)] + von_mises_dist(0, kappa, who.sum()), 2 * np.pi)
        pos = np.mod(pos + dt * scaled_velocity * np.c_[np.cos(vel), np.sin(vel)], l)

        if t % 10 == 0:
            print(f"Iteration number: {t} (out of {max_iter} iterations) [{(100 * t) // max_iter}% complete]")
            yield pos, vel

        index = index_map(pos, rr)
        particle_map = fill_map(int(l / rr), index)


def von_mises_dist(theta: float, kappa: float, n: int) -> np.ndarray:
    """
    Simulates 'n' random angles from a von Mises distribution, with preferred
    direction 'theta' and concentration parameter 'kappa'.

    Args:
        theta: The mean angle, i.e. the preferred rotation.
        kappa: The concentration of distribution (the higher the value, the
            more concentrated the data is around 'theta').
            [Note]: small kappa -> uniform distribution.
        n: The number of random angles to generate.

    Returns: A numpy array of size 'n' with random angle around 'theta'.

    """
    if kappa < 1e-6:
        return 2 * np.pi * np.random.random(size=(n, 1)) - np.pi

    a = 1 + np.sqrt(1 + 4 * kappa ** 2)
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


def average_orientation(pos: np.ndarray, vel: np.ndarray, index: np.ndarray,
                        particle_map: List[List[List[int]]], r: float) -> np.ndarray:
    """
    This function uses the velocities of all the particles, within the
    interaction radius 'r' for each and every particle, to calculate
    an average orientation (angle) that will then be used to calculate
    the new position for that particle.

    Args:
        pos: A numpy array containing the positions for all the particles.
        vel: A numpy array containing the velocities for all the particles.
        index: A map of the indexes alongside their previous positions that
            is jumping in this iteration.
        particle_map: A 2D list representing the map of the simulation square where each
        value is a list of all the indexes of the particles in that quadrant.
        r: The interaction radius for each particle.

    Returns: A numpy array of 'n' angles that will be used to update the positions
        of particles jumping in this iteration.

    """
    k = len(particle_map)
    n = index.shape[0]
    ao = np.zeros(shape=(n, 1))
    for i in range(n):
        first_indexes = [(index[i, 1].item() + j) % k for j in range(-1, 2)]
        second_indexes = [(index[i, 2].item() + j) % k for j in range(-1, 2)]

        neighbours = np.array(sum([particle_map[x][y] for x in first_indexes for y in second_indexes], []))
        result = np.linalg.norm(pos[neighbours, :] - pos[index[i, 0], :], ord=2, axis=1, keepdims=True)
        true_neighbours = neighbours[np.where(result < r)[0]]

        target = np.sum(np.c_[np.sin(vel[true_neighbours]), np.cos(vel[true_neighbours])], axis=0)
        ao[i, 0] = np.arctan(target[1] / target[0])
    return ao


def fill_map(size: int, index: np.ndarray) -> List[List[List[int]]]:
    """
    Breaks down the simulation square into quadrants (sub-squares)
    of equal length according to the interaction radius. Creates
    a 2D list of those quadrants and enters the indexes of each
    and every particle into their appropriate quadrants.

    Args:
        size: The length of each quadrant (sub-square) of the square.
        index: A numpy array containing the particles' indexes and their
            corresponding position.

    Returns: The 2D list representing the map of the simulation square where each
        value is a list of all the indexes of the particles in that quadrant.

    """
    particle_map = [[[] for _ in range(size)] for _ in range(size)]
    for i in range(index.shape[0]):
        particle_map[index[i, 1].item()][index[i, 2].item()].insert(0, index[i, 0].item())
    return particle_map


def index_map(pos: np.ndarray, r: float) -> np.ndarray:
    """
    Creates a new numpy array that holds each particle's index
    position and their scaled position value. The scaling is
    done according to the interaction radius (which itself is
    scaled according to the length of the box).

    Args:
        pos: A numpy array that holds the position data of each particle.
        r: The interaction radius scaled according tot length of the box.

    Returns: A new numpy array with particles' indexes as the first column
        and their positions for the next two.

    """
    return np.c_[np.arange(len(pos)), np.floor(pos / r)].astype(int)


def parse_args() -> Tuple[bool, str, int, int, int, float, float, float, float]:
    """
    This function handles the program arguments provided to the program
    when the program starts. It has default values in case of missing
    program arguments. Program runs without them because of this but
    can be tweaked according to simulation wanted.

    Returns: A tuple containing all the different program arguments
        needed to run the simulation.

    """
    parser = argparse.ArgumentParser(description="Depicting the movement of several particles in a 2D "
                                                 "space using only the CPU.")

    parser.add_argument("-s", "--save", action="store_true", default=False, help="Save in a File or not.")
    parser.add_argument("-f", "--video_file", type=str, default="quiver_basic.mp4", help="The Video File to Save in")
    parser.add_argument("-n", "--agents_num", type=int, default=10000, help="The Number of Agents")
    parser.add_argument("-l", "--box_size", type=int, default=10, help="The Size of the Box (Periodic Spatial Domain)")
    parser.add_argument("-t", "--max_iter", type=int, default=60, help="The Total Number of Iterations/Seconds")
    parser.add_argument("-r", "--interact_radius", type=float, default=0.07, help="The Radius of Interaction")
    parser.add_argument("-v", "--particle_velocity", type=float, default=0.02, help="The Velocity of the Particles")
    parser.add_argument("-nu", "--jump_rate", type=float, default=0.3, help="The Jump Rate")
    parser.add_argument("-k", "--concentration", type=float, default=20.0, help="The Concentration Parameter")

    args = parser.parse_args()

    return args.save, args.video_file, args.agents_num, args.box_size, args.max_iter, \
           args.interact_radius, args.particle_velocity, args.jump_rate, args.concentration


if __name__ == '__main__':
    main()
