import argparse
import numpy as np
import torch

from copy import deepcopy
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from matplotlib.axes import Axes
from torch import Tensor
from torch.distributions.von_mises import VonMises
from typing import Any, Generator, List, Tuple

np.set_printoptions(precision=4)
if not torch.cuda.is_available():
    raise Exception("CUDA not available to be used for the program.")

gpu_cuda = torch.device("cuda")


def main():
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
    fig, ax = plt.subplots(dpi=200)

    writer = writers['ffmpeg'](fps=15, metadata=dict(artist="Jawad"), bitrate=1800)
    ani = FuncAnimation(fig, update_quiver_frame, frames=process_particles(n, l, t, r, v, nu, kappa),
                        fargs=(ax, l.item(), r.item(), v.item(), nu.item(), kappa.item()),
                        interval=30, save_count=int(100 * (t * nu).item()) + 1, repeat=False)

    if save:
        ani.save(file, writer=writer)
        end.record(None)
        torch.cuda.synchronize()
        print("[100% Complete] Time taken:", start.elapsed_time(end) // 1000, "seconds")
    else:
        plt.show()


def update_quiver_frame(frame_data: Tuple[Tensor, Tensor], ax: Axes, l: int,
                        r: float, v: float, nu: float, kappa: float) -> None:
    ax.clear()
    sep = l / 10
    ax.set_xticks(np.arange(0, l + sep, sep))
    ax.set_yticks(np.arange(0, l + sep, sep))
    ax.set_xlim(0, l)
    ax.set_ylim(0, l)

    pos, vel = frame_data
    scale = l / 60

    ax.quiver(pos[:, 0].tolist(), pos[:, 1].tolist(),
              torch.mul(torch.cos(vel), scale).flatten().tolist(),
              torch.mul(torch.sin(vel), scale).flatten().tolist())
    ax.set_title(f"Particles = {pos.size()[0]:,}, Interaction Radius = {r}, Velocity = {v},\n"
                 f"Jump Rate = {nu}, Concentration Parameter = {kappa}", fontsize="small")


def process_particles(n: int, l: Tensor, t: Tensor, r: Tensor, v: Tensor, nu: Tensor, kappa: Tensor) -> \
        Generator[Tuple[Tensor, Tensor], None, None]:
    von_mises = VonMises(tensor(0, torch.float), tensor(kappa, torch.float))

    dt = tensor(0.01, torch.float) / nu
    max_iter = torch.floor(t / dt).to(torch.int64).item() * 5
    scaled_velocity = l * v
    rr = l / torch.floor(l / r)
    pos = l * torch.rand(n, 2, device=gpu_cuda)
    vel = 2 * np.pi * torch.rand(n, 1, device=gpu_cuda)

    # print(f"""Calculated Parameters:-
    #     Time Discretisation Step: {dt}
    #     Max Iteration: {max_iter}
    #     Scaled Velocity of Particles: {scaled_velocity}
    #     Scaled Interaction Radius: {rr}""")

    dim = torch.floor(l / rr).to(torch.int64).item()

    index = index_map(pos, rr)
    particle_map = fill_map(dim, index)

    for t in range(max_iter):
        jump = torch.rand(n, 1, device=gpu_cuda)
        who = torch.where(torch.gt(jump, torch.exp(-nu * dt)), tensor(1, torch.int64), tensor(0, torch.int64))
        condition = torch.where(torch.eq(who[:, 0], 1))

        target = deepcopy(vel)
        target[condition] = average_orientation(pos, target, index[condition], particle_map, r)
        vel[condition] = torch.remainder(target[condition] + von_mises.sample((torch.sum(who), 1)), 2 * np.pi)
        pos = torch.remainder(pos + dt * scaled_velocity * torch.cat((torch.cos(vel), torch.sin(vel)), 1), l)

        if t % 10 == 0:
            print(f"Iteration number: {t} (out of {max_iter} iterations) [{(100 * t) // max_iter}% complete]")
            yield pos, vel

        index = index_map(pos, rr)
        particle_map = fill_map(dim, index)


def average_orientation(pos: Tensor, vel: Tensor, index: Tensor,
                        particle_map: List[List[List[int]]], r: Tensor) -> Tensor:
    k = len(particle_map)
    n = index.size()[0]
    ao = torch.zeros(n, 1, device=gpu_cuda)
    for i in range(n):
        first_indexes = [(index[i, 1].item() + j) % k for j in range(-1, 2)]
        second_indexes = [(index[i, 2].item() + j) % k for j in range(-1, 2)]

        neighbours = tensor(sum([particle_map[x][y] for x in first_indexes for y in second_indexes], []), torch.int64)
        result = torch.norm(pos[neighbours, :] - pos[index[i, 0], :], p=2, dim=1)
        true_neighbours = neighbours[torch.where(torch.lt(result, r))]

        target = torch.sum(torch.cat((torch.sin(vel[true_neighbours]), torch.cos(vel[true_neighbours])), 1), 0)
        ao[i, 0] = torch.atan(target[1] / target[0])  # angle
    return ao


def fill_map(size: int, index: Tensor) -> List[List[List[int]]]:
    particle_map = [[[] for _ in range(size)] for _ in range(size)]
    for i in range(index.size()[0]):
        particle_map[index[i, 1].item() % size][index[i, 2].item() % size].insert(0, index[i, 0].item())
    return particle_map


def index_map(pos: Tensor, r: Tensor) -> Tensor:
    indexes = torch.arange(pos.size()[0], device=gpu_cuda).reshape(pos.size()[0], 1)
    return torch.cat((indexes, torch.floor(pos / r).to(torch.int64)), 1)


def tensor(value: Any, data_type: Any) -> Tensor:
    return torch.tensor(value, dtype=data_type, device=gpu_cuda)


def parse_args() -> Tuple[bool, str, int, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    parser = argparse.ArgumentParser(description="Depicting the movement of several particles in a 2D "
                                                 "space using only the GPU.")

    parser.add_argument("-s", "--save", action="store_true", default=False, help="Save in a File or not.")
    parser.add_argument("-f", "--video_file", type=str, default="quiver_pytorch.mp4", help="The Video File to Save in")
    parser.add_argument("-n", "--agents_num", type=int, default=5000, help="The Number of Agents")
    parser.add_argument("-l", "--box_size", type=int, default=1, help="The Size of the Box (Periodic Spatial Domain)")
    parser.add_argument("-t", "--seconds", type=int, default=60, help="Simulation Length in Seconds")
    parser.add_argument("-r", "--interact_radius", type=float, default=0.07, help="The Radius of Interaction")
    parser.add_argument("-v", "--particle_velocity", type=float, default=0.02, help="The Velocity of the Particles")
    parser.add_argument("-nu", "--jump_rate", type=float, default=0.3, help="The Jump Rate")
    parser.add_argument("-k", "--concentration", type=float, default=20.0, help="The Concentration Parameter")

    args = parser.parse_args()

    return args.save, args.video_file, args.agents_num, \
           tensor(args.box_size, torch.int64), tensor(args.seconds, torch.int64), \
           tensor(args.interact_radius, torch.float), tensor(args.particle_velocity, torch.float), \
           tensor(args.jump_rate, torch.float), tensor(args.concentration, torch.float)


if __name__ == '__main__':
    main()
