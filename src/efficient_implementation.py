import argparse
import numpy as np
import torch

from copy import deepcopy
from datetime import datetime
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.axes import Axes
from torch import Tensor
from typing import Any, Generator, List, Tuple

np.set_printoptions(precision=4)
if not torch.cuda.is_available():
    raise Exception("CUDA not available to be used for the program.")

gpu_cuda = torch.device("cuda")


def main() -> None:
    """
    The main function that parses the program arguments using parse_args(),
    creates a plot for a moving quiver diagram, sets up a writer for the video
    file with ffmpeg format with metadata, creates the animation by calling
    process_particles() generator to recieve frame data, passes it along to
    update_quiver_frame() function to draw out a frame and this gets either
    shown in a window or saved to "quiver_efficient.mp4" to create a video.

    Returns: None (void function)

    """
    save, file, n, l, t, r, v, nu, kappa = parse_args()
    # print(f"""Hyperparameters:-
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
    fig, ax = plt.subplots(dpi=200)

    writer = writers['ffmpeg'](fps=15, metadata=dict(artist="Jawad"), bitrate=1800)
    ani = FuncAnimation(fig, update_quiver_frame, frames=process_particles(n, l, t, r, v, nu, kappa),
                        fargs=(ax, l), interval=30, save_count=int(100 * t * nu) + 1, repeat=False)

    if save:
        ani.save(file, writer=writer)
        print("[100% Complete] Time taken:", (datetime.now() - current_time).seconds, "seconds")
    else:
        plt.show()


def update_quiver_frame(frame_data: Tuple[Tensor, Tensor], ax: Axes, l: int) -> None:
    ax.clear()
    sep = l / 10

    ax.set_xticks(np.arange(0, l + sep, sep))
    ax.set_yticks(np.arange(0, l + sep, sep))

    ax.set_xlim(0, l)
    ax.set_ylim(0, l)

    pos, vel = frame_data
    scale = l / 60

    q = ax.quiver(pos[:, 0].tolist(), pos[:, 1].tolist(),
                  torch.mul(torch.cos(vel), scale).flatten().tolist(),
                  torch.mul(torch.sin(vel), scale).flatten().tolist())
    ax.quiverkey(q, X=0.3, Y=1.1, U=0.05,
                 label=f"Quiver key, length = 0.05 - Particles: {pos.size()[0]:,}", labelpos='E')


def process_particles(n: int, l: int, t: int, r: float, v: float, nu: float, kappa: float) -> \
        Generator[Tuple[Tensor, Tensor], None, None]:
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

    Returns: A generator which yields a tuple of two Tensors for position and velocity.

    """
    dt = 0.01 / nu
    max_iter = np.floor(t / dt).astype(int) * 5
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

    empty_particle_map = np.full((int(l / rr), int(l / rr)), np.nan).astype(np.object)

    index = index_map(pos, rr)
    particle_map = fill_map(deepcopy(empty_particle_map), index)

    for t in range(max_iter + 1):
        jump = torch.rand(n, 1, device=gpu_cuda)
        who = torch.where(torch.gt(jump, torch.exp(tensor(-nu * dt, torch.float))),
                          tensor(1, torch.int64),
                          tensor(0, torch.int64))
        target = deepcopy(vel)
        target[torch.where(torch.eq(who[:, 0], 1))] = \
            average_orientation(pos, target, index[torch.where(torch.eq(who[:, 0], 1))], particle_map, r)
        vel[torch.where(torch.eq(who[:, 0], 1))] = \
            torch.remainder(target[torch.where(torch.eq(who[:, 0], 1))] +
                                von_mises_dist(0, kappa, torch.sum(who).item()),
                            tensor(2 * np.pi, torch.float))
        pos = torch.remainder(pos + torch.mul(torch.cat((torch.cos(vel), torch.sin(vel)), 1), dt * scaled_velocity), l)

        if t % 10 == 0:
            print(f"Iteration number: {t} (out of {max_iter} iterations) [{(100 * t) // max_iter}% complete]")
            yield pos, vel

        index = index_map(pos, rr)
        particle_map = fill_map(deepcopy(empty_particle_map), index)


def von_mises_dist(theta: float, kappa: float, n: int) -> Tensor:
    """
    Simulates 'n' random angles from a von Mises distribution, with preferred
    direction 'theta' and concentration parameter 'kappa'.

    Args:
        theta: The mean angle, i.e. the preferred rotation.
        kappa: The concentration of distribution (the higher the value, the
            more concentrated the data is around 'theta').
            [Note]: small kappa -> uniform distribution.
        n: The number of random angles to generate.

    Returns: A Tensor of size 'n' with random angle around 'theta'.

    """
    if kappa < 1e-6:
        return torch.mul(torch.rand(n, 1, device=gpu_cuda), 2 * np.pi).sub(np.pi)

    a = 1 + np.sqrt(1 + 4 * kappa ** 2)
    b = (a - np.sqrt(2 * a)) / (2 * kappa)
    r = (1 + b ** 2) / (2 * b)

    alpha = torch.zeros(n, 1, device=gpu_cuda)
    for j in range(n):
        while True:
            u = np.random.uniform(size=3)

            z = np.cos(np.pi * u[0])
            f = (1 + r * z) / (r + z)
            c = kappa * (r - f)

            if u[1] < c * (2 - c) or not (np.log(c) - np.log(u[1]) + 1 - c < 0):
                break

        temp = np.mod(theta + np.sign(u[2] - 0.5) * np.arccos(f), 2 * np.pi)
        alpha[j, 0] = tensor(temp - 2 * np.pi if np.pi < temp <= 2 * np.pi else temp, torch.float)

    return alpha


def average_orientation(pos: Tensor, vel: Tensor, index: Tensor,
                        particle_map: np.ndarray, r: float) -> Tensor:
    """
    This function uses the velocities of all the particles, within the
    interaction radius 'r' for each and every particle, to calculate
    an average orientation (angle) that will then be used to calculate
    the new position for that particle.

    Args:
        pos: A Tensor containing the positions for all the particles.
        vel: A Tensor containing the velocities for all the particles.
        index: A Tensor containing a map of the indexes alongside their
            previous positions that is jumping in this iteration.
        particle_map: A numpy array that keeps a track of where the particles are and have been.
        r: The interaction radius for each particle.

    Returns: A Tensor of 'n' angles that will be used to update the positions
        of particles jumping in this iteration.

    """
    k = particle_map.shape[0]
    n = index.size()[0]
    ao = torch.zeros(n, 1, device=gpu_cuda)
    for i in range(n):
        first_indexes = [(index[i, 1].item() + j) % k for j in range(-1, 2)]
        second_indexes = [(index[i, 2].item() + j) % k for j in range(-1, 2)]

        neighbours = flatten([particle_map[x, y] for x in first_indexes for y in second_indexes])
        result = torch.norm(pos[neighbours, :] - pos[index[i, 0], :], p=2, dim=1, keepdim=True)
        true_neighbours = neighbours[torch.where(torch.lt(result, r))[0]]

        target = torch.sum(torch.cat((torch.sin(vel[true_neighbours]), torch.cos(vel[true_neighbours])), 1), 0)
        ao[i, 0] = torch.atan(torch.div(target[1], target[0]))  # Calculates the angle
    return ao


def flatten(array_matrix: List[Any]) -> Tensor:
    """
    Helper function that helps to convert a list of values and/or
    arrays of indexes into a Tensor where each array is concatenated
    to each other left-to-right.

    Args:
        array_matrix: A list of values and/or arrays of indexes.

    Returns: A Tensor that is a flattened version (no sub-lists) of 'array_matrix'.

    """
    result = []
    for i in range(len(array_matrix)):
        try:
            result += list(array_matrix[i])
        except TypeError:
            result += [array_matrix[i]]
    return tensor([e for e in result if not np.isnan(e)], torch.int64)


def fill_map(particle_map: np.ndarray, index: Tensor) -> np.ndarray:
    """
    Fills in a particle map with the index value of each particle.
    If the value at position in the map from 'index' is empty, a new
    array is created with that index as its only element. Otherwise,
    the index is inserted at the beginning of the exisiting array.

    Args:
        particle_map: A numpy 2D matrix map of the simulation box where each
            value is a location quadrant where a group of particles may exist.
        index: A Tensor containing the particles' indexes and their
            corresponding position.

    Returns: The given particle map with new indexes filled in.

    """
    for i in range(index.size()[0]):
        if np.all(np.isnan(particle_map[index[i, 1], index[i, 2]])):
            particle_map[index[i, 1], index[i, 2]] = np.array([index[i, 0].item()])
        else:
            particle_map[index[i, 1], index[i, 2]] = np.r_[index[i, 0].item(), particle_map[index[i, 1], index[i, 2]]]
    return particle_map


def index_map(pos: Tensor, r: float) -> Tensor:
    """
    Creates a new Tensor that holds each particle's index
    position and their scaled position value. The scaling is
    done according to the interaction radius (which itself is
    scaled according to the length of the box).

    Args:
        pos: A Tensor that holds the position data of each particle.
        r: The interaction radius scaled according tot length of the box.

    Returns: A new Tensor with particles' indexes as the first column
        and their positions for the next two.

    """
    indexes = torch.arange(pos.size()[0], device=gpu_cuda).reshape(pos.size()[0], 1)
    return torch.cat((indexes, torch.floor(torch.div(pos, r)).to(torch.int64)), 1)


def tensor(value: Any, data_type: Any) -> Tensor:
    """
    A helper function to easily build a Tensor for any given
    value and its data type that can be sent to a CUDA GPU.

    Args:
        value: Any value to be entered into Tensor
        data_type: The data type for the Tensor

    Returns: A new Tensor with given value and data type.

    """
    return torch.tensor(value, dtype=data_type, device=gpu_cuda)


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
                                                 "space using a combination of CPU and GPU.")

    parser.add_argument("-s", "--save", action="store_true", default=False, help="Save in a File or not.")
    parser.add_argument("-f", "--video_file", type=str, default="quiver_efficient.mp4", help="The Video File to Save in")
    parser.add_argument("-n", "--agents_num", type=int, default=1000000, help="The Number of Agents")
    parser.add_argument("-l", "--box_size", type=int, default=100, help="The Size of the Box (Periodic Spatial Domain)")
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
