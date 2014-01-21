import nengo
import nengo.helpers
import matplotlib.pyplot as plt

import nengo_rt_backend

import numpy as np

# enable debug spew
import logging
logging.basicConfig(level=logging.DEBUG)

model = nengo.Model(label='Integrator')

A = nengo.Ensemble(nengo.LIFSurrogate(100), dimensions=1, label='Integrator')

input = nengo.Node(nengo.helpers.piecewise({0:0,0.2:1,1:0,2:-2,3:0,4:1,5:0}), label='Piecewise input')

tau = 0.1
nengo.Connection(A, A, transform=[[1]], filter=tau)
nengo.Connection(input, A, transform=[[tau]], filter=tau)

p1 = nengo.Probe(input, 'output')
p2 = nengo.Probe(A, 'decoded_output', filter=0.01)

# software simulation
sim = nengo.Simulator(model)
sim.run(6)

# cheating for a minute
A_built = sim.model.objs[0]
#eval_points = A_built.eval_points
eval_points = np.matrix([np.linspace(-1.0, 1.0, num=500)]).transpose()
activities = A_built.activities(eval_points)

print("Eval points: " + str(eval_points.shape))
print("Activities: " + str(activities.shape))

# print(activities[:,0]) # this looks right for one neuron

print("Plotting activities...")
plt.plot(eval_points, activities)
plt.show()

u, s, v = np.linalg.svd(activities.transpose())

# check PCs, which should be in v
npc = 7
print("Plotting principal components...")
plt.plot(eval_points, v[0:npc,:].transpose())
plt.show()

print((u.shape, s.shape, v.shape))
S = np.zeros((u.shape[0], v.shape[0]), dtype=complex)
S[:s.shape[0], :s.shape[0]] = np.diag(s)

# as a test, find optimal linear firing-rate decoders and see if the estimate makes sense
gamma = np.dot(activities.transpose(), activities)
gamma = gamma + np.dot(np.mean(np.diag(gamma)), np.identity(gamma.shape[0]))
upsilon = np.dot(activities.transpose(), eval_points)
decoders = np.linalg.solve(gamma, upsilon)
estimate = np.dot(activities, decoders)

print("Plotting decoded estimate from rates...")
plt.plot(eval_points, estimate)
plt.show()

# calculate PCs over saturation range
xExtended = np.matrix([np.linspace(-2.0, 2.0, num=len(eval_points))]).transpose()
ratesExtended = A_built.activities(xExtended)
usi = np.linalg.pinv(np.dot(u,S))
print(usi.shape)
print(ratesExtended.shape)
PCsExtended = np.dot(usi[0:npc, :], ratesExtended.transpose())

print(xExtended.shape)
print(PCsExtended.shape)

t1 = xExtended
t2 = PCsExtended
print(t1.shape)
print(t2.shape)
print("Plotting extended principal components...")
plt.plot(t1,t2.transpose())
plt.show()

# calculate approximate decoders
approxDecoders = np.dot(S[0:npc, 0:npc], np.dot(u[:,0:npc].transpose(), decoders))
print("Decoders: " + str(approxDecoders))
print((PCsExtended.shape, approxDecoders.shape))
approxEstimate = np.dot(PCsExtended.transpose(), approxDecoders)
print("Plotting decoded estimate from principal components...")
plt.plot(xExtended, approxEstimate)
plt.show()

# cheating is over

print("Plotting simulation output...")
t = sim.trange()
plt.plot(t, sim.data(p1), label="Input")
plt.plot(t, sim.data(p2), 'k', label="Integrator output")
plt.legend()
plt.show()

# now attempt this in hardware
sim = nengo_rt_backend.Simulator(model)
sim.run(6)

# FIXME get input

