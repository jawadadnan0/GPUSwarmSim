import argparse
import math
import torch

from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
from torch import Tensor
from torch.distributions.von_mises import VonMises
from typing import Any, Generator, List, Tuple

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
    shown in a window or saved to "quiver_3D.mp4" to create a video.

    Returns: None (void function)

    """
    save, file, n, l, t, r, v, nu, kappa = parse_args()
    # print(f"""Hyperparameters:-
    #     Save to File: {save}
    #     Save File Name: {file}
    #     Number of Particles: {n}
    #     Periodic Spatial Domain: {l}
    #     Simulation Length (in Seconds): {t}
    #     Interaction Radius: {r}
    #     Initial Particle velocity: {v}
    #     Jump Rate: {nu}
    #     Concentration Parameter: {kappa}""")

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record(None)
    fig, ax = plt.subplots(dpi=300, subplot_kw=dict(projection="3d"))

    writer = writers['ffmpeg'](fps=15, metadata=dict(artist="Jawad"), bitrate=1800)
    ani = FuncAnimation(fig, update_quiver_frame, frames=process_particles(n, l, t, r, v, nu, kappa),
                        fargs=(ax, l, r, v, nu, kappa), interval=30, save_count=int(100 * t * nu) + 1, repeat=False)

    if save:
        ani.save(file, writer=writer)
        end.record(None)
        torch.cuda.synchronize()
        print("[100% Complete] Time taken:", start.elapsed_time(end) // 1000, "seconds")
    else:
        plt.show()


def update_quiver_frame(frame_data: Tuple[Tensor, Tensor], ax: Axes3D, l: int,
                        r: float, v: float, nu: float, kappa: float) -> None:
    """
    This function is executed every single time the frame needs to updated
    whether it is to view it in real-time or to save it into a video.

    Args:
        frame_data: A tuple of two Tensors, containing the positions and velocities of the particles.
        ax: The 3D axis object of the plot in order to help set it up.
        l: The length of the square to be drawn that will contain the particles.
        r: The interaction radius of each particle.
        v: The max velocity of each particle.
        nu: The jump rate for each particle.
        kappa: The concentration parameter for von Mises Distribution.

    Returns: None (void function)

    """
    ax.clear()

    ticks = 10
    ax.set_xticks([value / ticks for value in range(ticks + 1)])
    ax.set_yticks([value / ticks for value in range(ticks + 1)])
    ax.set_zticks([value / ticks for value in range(ticks + 1)])

    ax.set_xlim(0, l)
    ax.set_ylim(0, l)
    ax.set_zlim(0, 1)

    pos, vel = frame_data
    scale = l / 60

    ax.quiver3D(pos[:, 0].tolist(), pos[:, 1].tolist(), pos[:, 2].tolist(),
                torch.mul(torch.sin(vel[:, 1]) * torch.cos(vel[:, 0]), scale).flatten().tolist(),
                torch.mul(torch.sin(vel[:, 1]) * torch.sin(vel[:, 0]), scale).flatten().tolist(),
                torch.mul(torch.cos(vel[:, 1]), scale).flatten().tolist())
    ax.set_title(f"Particles = {pos.size()[0]:,}, Interaction Radius = {r}, Velocity = {v},\n"
                 f"Jump Rate = {nu}, Concentration Parameter = {kappa}", fontsize="small")


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
    von_mises = VonMises(tensor(0, torch.float), tensor(kappa, torch.float))

    dt = 0.01 / nu
    max_iter = int(t / dt) * 5
    scaled_velocity = l * v
    rr = l / int(l / r)
    pos = torch.mul(torch.rand(n, 3, device=gpu_cuda), l)
    vel = torch.cat((torch.mul(torch.rand(n, 1, device=gpu_cuda), 2 * math.pi),
                     torch.mul(torch.rand(n, 1, device=gpu_cuda), math.pi)), 1)

    # print(f"""Calculated Parameters:-
    #     Time Discretisation Step: {dt}
    #     Max Iteration: {max_iter}
    #     Scaled Velocity of Particles: {scaled_velocity}
    #     Scaled Interaction Radius: {rr}""")

    index = index_map(pos, rr)
    particle_map = fill_map(int(l / rr), index)

    for t in range(max_iter + 1):
        jump = torch.rand(n, 1, device=gpu_cuda)
        who = torch.where(torch.gt(jump, torch.exp(tensor(-nu * dt, torch.float))),
                          tensor(1, torch.int64),
                          tensor(0, torch.int64))
        condition = torch.where(torch.eq(who[:, 0], 1))

        target = deepcopy(vel)
        target[condition] = average_orientation(pos, target, index[condition], particle_map, r)
        vel[condition] = target[condition] + von_mises.sample((who.sum(), 2))
        vel[:, 0][condition] = torch.remainder(vel[:, 0][condition], 2 * math.pi)
        vel[:, 1][condition] = torch.remainder(vel[:, 1][condition], math.pi)

        x = torch.sin(vel[:, 1]) * torch.cos(vel[:, 0])
        y = torch.sin(vel[:, 1]) * torch.sin(vel[:, 0])
        z = torch.cos(vel[:, 1])
        pos = torch.remainder(pos + torch.mul(torch.cat((x.reshape(x.size()[0], 1),
                                                         y.reshape(y.size()[0], 1),
                                                         z.reshape(z.size()[0], 1)), 1), dt * scaled_velocity), l)

        if t % 10 == 0:
            print(f"Iteration number: {t} (out of {max_iter} iterations) [{(100 * t) // max_iter}% complete]")
            yield pos, vel

        index = index_map(pos, rr)
        particle_map = fill_map(int(l / rr), index)


def average_orientation(pos: Tensor, vel: Tensor, index: Tensor,
                        particle_map: List[List[List[List[int]]]], r: float) -> Tensor:
    """
    This function uses the velocities of all the particles, within the
    interaction radius 'r' for each and every particle, to calculate
    an average orientation (angles) that will then be used to calculate
    the new position for that particle.

    Args:
        pos: A Tensor containing the positions for all the particles.
        vel: A Tensor containing the velocities for all the particles.
        index: A Tensor containing a map of the indexes alongside their
            previous positions that is jumping in this iteration.
        particle_map: The 3D list representing the map of the simulation box where each
            value is a list of all the indexes of the particles in that quadrant
        r: The interaction radius for each particle.

    Returns: A Tensor of ('n', 2) angles that will be used to update the positions
        of particles jumping in this iteration.

    """
    k = len(particle_map)
    n = index.size()[0]
    ao = torch.zeros(n, 2, device=gpu_cuda)
    for i in range(n):
        first_indexes = [(index[i, 1].item() + j) % k for j in range(-1, 2)]
        second_indexes = [(index[i, 2].item() + j) % k for j in range(-1, 2)]
        third_indexes = [(index[i, 3].item() + j) % k for j in range(-1, 2)]

        neighbours = tensor(sum([particle_map[x][y][z] for x in first_indexes
                                                       for y in second_indexes
                                                       for z in third_indexes], []), torch.int64)
        result = torch.norm(pos[neighbours, :] - pos[index[i, 0], :], p=2, dim=1)
        true_neighbours = neighbours[torch.where(torch.lt(result, r))]

        x = torch.sum(torch.sin(vel[true_neighbours, 1]) * torch.cos(vel[true_neighbours, 0]))
        y = torch.sum(torch.sin(vel[true_neighbours, 1]) * torch.sin(vel[true_neighbours, 0]))
        z = torch.sum(torch.cos(vel[true_neighbours, 1]))

        # Calculate the azimuth and inclinations.
        ao[i, 0] = torch.atan(torch.div(y, x))
        ao[i, 1] = torch.atan(torch.div(torch.sqrt(x * x + y * y), z))
    return ao


def fill_map(size: int, index: Tensor) -> List[List[List[List[int]]]]:
    """
    Breaks down the simulation box into quadrants (sub-boxes)
    of equal length according to the interaction radius. Creates
    a 3D list of those quadrants and enters the indexes of each
    and every particle into their appropriate quadrants.

    Args:
        size: The length of each quadrant (sub-boxes) of the box.
        index: A Tensor containing the particles' indexes and their
            corresponding position.

    Returns: The 3D list representing the map of the simulation box where each
        value is a list of all the indexes of the particles in that quadrant.

    """
    particle_map = [[[[] for _ in range(size)] for _ in range(size)] for _ in range(size)]
    for i in range(index.size()[0]):
        particle_map[index[i, 1].item()][index[i, 2].item()][index[i, 3].item()].append(index[i, 0].item())
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
    parser = argparse.ArgumentParser(description="Depicting the movement of several particles in a 3D "
                                                 "space using a combination of CPU and GPU.")

    parser.add_argument("-s", "--save", action="store_true", default=False, help="Save in a File or not.")
    parser.add_argument("-f", "--video_file", type=str, default="quiver_3D.mp4", help="The Video File to Save in")
    parser.add_argument("-n", "--agents_num", type=int, default=500, help="The Number of Agents")
    parser.add_argument("-l", "--box_size", type=int, default=1, help="The Size of the Box (Periodic Spatial Domain)")
    parser.add_argument("-t", "--seconds", type=int, default=60, help="Simulation Length in Seconds")
    parser.add_argument("-r", "--interact_radius", type=float, default=0.07, help="The Radius of Interaction")
    parser.add_argument("-v", "--particle_velocity", type=float, default=0.02, help="The Velocity of the Particles")
    parser.add_argument("-nu", "--jump_rate", type=float, default=0.3, help="The Jump Rate")
    parser.add_argument("-k", "--concentration", type=float, default=20.0, help="The Concentration Parameter")

    args = parser.parse_args()

    return args.save, args.video_file, args.agents_num, args.box_size, args.seconds, \
           args.interact_radius, args.particle_velocity, args.jump_rate, args.concentration


if __name__ == '__main__':
    main()
