import argparse


def main():
    n, l, t, r, v, nu, kappa = parse_args()
    print(f"""Number of Particles: {n}
Periodic Spatial Domain: {l}
Total No. of Iterations: {t}
Interaction Radius: {r}
Initial Particle velocity: {v}
Jump Rate: {nu}
Concentration Parameter: {kappa}""")


def parse_args():
    parser = argparse.ArgumentParser(description="Depicting the movement of several quaternions ")

    parser.add_argument("-n", "--agents_num", type=int, default=500, help="The Number of Agents")
    parser.add_argument("-l", "--box_size", type=int, default=1, help="The Size of the Box (Periodic Spatial Domain)")
    parser.add_argument("-t", "--max_iter", type=int, default=5000, help="The Total Number of Iterations")
    parser.add_argument("-r", "--interact_radius", type=float, default=0.07, help="The Radius of Interaction")
    parser.add_argument("-v", "--particle_velocity", type=float, default=0.02, help="The Velocity of the Particles")
    parser.add_argument("-nu", "--jump_rate", type=float, default=0.3, help="The Jump Rate")
    parser.add_argument("-k", "--concentration", type=float, default=20, help="The Concentration Parameter")

    args = parser.parse_args()

    return args.n, args.l, args.t, args.r, args.v, args.nu, args.kappa


if __name__ == '__main__':
    main()
