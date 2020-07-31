# GPUSwarmSim
This is my 3rd Year Final Project on the Simulation of bio-inspired swarming drones on GPU.

## Summary
Technology becomes increasingly inspired by nature (from swimsuits imitating
shark skin to micro-drones copying insects). Here, we are interested in the
decentralized control of swarming drones through coordination rules inspired
from flocking birds or schooling fish. Specifically, we have developed a
computational model where the drones are described as solid objects moving
at constant speed in three dimensional space and coordinating their body
attitude with their neighbors. Although the coordination rules are
simple, the overal behavior of a swarm consisting of milions of such objects
exhibits fascinating features such as clustering, bands, traveling waves and
other kinds of intringuing patterns.

In order to characterize the behaviour of such systems we need to scan a
large space of parameters. For each values of the parameters, simulations
may include milions of individuals or more. This makes the task extremely
computationally intensive and requires up-to-date computational techniques.
So far, a demonstrator code has been written in matlab. It has allowed
us to have a glimpse into the fascinating behavior of the model but the code
is far too slow for a systematic study of the system. In order to carry this
project further, we need to implement the model in a computationally
efficient language making use of massively parallel architectures.

Particle models to which this body attitude coordination model belongs have
been shown to be extremely adapted to implementation on GPU’s. This is
because the displacement phase of each object is independent of the other
ones and is very similar to ray tracing algorithms used in image synthesis.
The interaction phase, where the objects coordinate their body attitude with
their neighbors, requires local communications between neighboring objects,
and consequently, between the processors, and this is where a smart
implementation within GPU’s is needed. This would be the task to be carried
out in the present project.

## Installation

Make sure to install CUDA Toolkit Version 10.2 (or above) for your appropriate
Operating System through the link below:

https://developer.nvidia.com/cuda-downloads

Third-party library files used in this Simulation:

i) Numpy 1.19.0 (or above): https://numpy.org/install/  
ii) PyTorch 1.6.0 (or above): https://pytorch.org/  
iii) Matplotlib 3.30 (or above): https://matplotlib.org/users/installing.html

Or you can run the following command in Terminal in main directory:

		pip install -r requirements.txt
