import nengo
import nengo.helpers
import matplotlib.pyplot as plt

import nengo_rt_backend

import numpy as np

import os

# enable debug spew
import logging
logging.basicConfig(level=logging.DEBUG)

hardware = True

model = nengo.Model(label='Integrator')

input = nengo.Node(nengo.helpers.piecewise({0:0,0.2:1,1:0,2:-2,3:0,4:1,5:0}), label='Piecewise input')
#input = nengo.Node(nengo.helpers.piecewise({0:0.5}), label='Constant input')

A = []
tau = 0.1

for i in range(1):
    Ai = nengo.Ensemble(nengo.LIF(100, tau_rc=0.02, tau_ref=0.002),
                        dimensions=1, label='Integrator A' + str(i))
    A.append(Ai)
    nengo.Connection(A[i], A[i], transform=[[1]], filter=tau)
    nengo.Connection(input, A[i], transform=[[tau]], filter=tau)

p1 = nengo.Probe(input, 'output')
p2 = nengo.Probe(A[0], 'decoded_output', filter=0.01)

def fn(t, x):
    return -1.0 * x

#N = nengo.Node(nengo.helpers.piecewise({0:0,0.2:-1,1:0,2:2,3:0,4:-1,5:0}), label='Piecewise input')

N = nengo.Node(label='Software function', output=fn, size_in=1)
nengo.Connection(A[0], N, transform=[[1]])

B = nengo.Ensemble(nengo.LIF(100, tau_rc=0.02, tau_ref=0.002),
                   dimensions=1, label='Integrator B')
nengo.Connection(B, B, transform=[[1]], filter=tau)
nengo.Connection(N, B, transform=[[tau]], filter=tau)

p3 = nengo.Probe(N, 'output')
p4 = nengo.Probe(B, 'decoded_output', filter=0.01)

if hardware:
    sim = nengo_rt_backend.Simulator(model, targetFile='target_1d2d.xml')
else:
    sim = nengo.Simulator(model)

sim.run(6)
t = sim.trange()
fh = plt.figure()
plt.plot(t, sim.data(p1), label="Input")
plt.plot(t, sim.data(p2), label="Integrator A output")
plt.plot(t, sim.data(p3), label="Node output")
plt.plot(t, sim.data(p4), label="Integrator B output")
plt.legend()
fh.suptitle('Hardware simulation output')
plt.show()


