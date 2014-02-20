import nengo
import nengo.helpers
import matplotlib.pyplot as plt

import nengo_rt_backend

import numpy as np

import os

# enable debug spew
import logging
logging.basicConfig(level=logging.DEBUG)

# set to True to plot intermediate results, which will block until closed
makePlots = False

model = nengo.Model(label='Integrator')

input = nengo.Node(nengo.helpers.piecewise({0:0,0.2:1,1:0,2:-2,3:0,4:1,5:0}), label='Piecewise input')

A = []
tau = 0.1

for i in range(64):
    Ai = nengo.Ensemble(nengo.LIF(100, tau_rc=0.02, tau_ref=0.002),
                        dimensions=1, label='Integrator ' + str(i))
    A.append(Ai)
    nengo.Connection(A[i], A[i], transform=[[1]], filter=tau)
    nengo.Connection(input, A[i], transform=[[tau]], filter=tau)

p1 = nengo.Probe(input, 'output')
p2 = nengo.Probe(A[0], 'decoded_output', filter=0.01)

# cheating for a minute
#A_built = sim.model.objs[0]
#eval_points = A_built.eval_points
#eval_points = np.matrix([np.linspace(-1.0, 1.0, num=500)]).transpose()
#activities = A_built.activities(eval_points)

#print("Eval points: " + str(eval_points.shape))
#print("Activities: " + str(activities.shape))

# print(activities[:,0]) # this looks right for one neuron

#if makePlots:
#    print("Plotting activities...")
#    f1 = plt.figure()
#    plt.plot(eval_points, activities)
#    f1.suptitle('Neural activities')

#u, s, v = np.linalg.svd(activities.transpose())

# check PCs, which should be in v
#npc = 7
#if makePlots:
#    print("Plotting principal components...")
#    f2 = plt.figure()
#    plt.plot(eval_points, v[0:npc,:].transpose())
#    f2.suptitle('Principal components')

#S = np.zeros((u.shape[0], v.shape[0]), dtype=complex)
#S[:s.shape[0], :s.shape[0]] = np.diag(s)
# find optimal linear firing-rate decoders and see if the estimate makes sense
#gamma = np.dot(activities.transpose(), activities)
#gamma = gamma + np.dot(np.mean(np.diag(gamma)), np.identity(gamma.shape[0]))
#upsilon = np.dot(activities.transpose(), eval_points)
#decoders = np.linalg.solve(gamma, upsilon)
#c0_built = sim.model.connections[0]
#decoders = c0_built.decoders * (1.0 / 1000.0) # multiply by timestep to get activities per timestep instead of activities per second
#estimate = np.dot(activities, decoders)

#if makePlots:
#    print("Plotting decoded estimate from rates...")
#    f3 = plt.figure()
#    plt.plot(eval_points, estimate)
#    f3.suptitle('Decoded estimate from rates')

# calculate PCs over saturation range
#xExtended = np.matrix([np.linspace(-2.0, 2.0, num=len(eval_points))]).transpose()
#ratesExtended = A_built.activities(xExtended)
#usi = np.linalg.pinv(np.dot(u,S))
#PCsExtended = np.dot(usi[0:npc, :], ratesExtended.transpose())

#if makePlots:
#    print("Plotting extended principal components...")
#    f4 = plt.figure()
#    plt.plot(xExtended,PCsExtended.transpose())
#    f4.suptitle('Extended principal components')

# calculate approximate decoders
#approxDecoders = np.dot(S[0:npc, 0:npc], np.dot(u[:,0:npc].transpose(), decoders))
#print("Decoders: " + os.linesep + str(approxDecoders))
#print((PCsExtended.shape, approxDecoders.shape))
#approxEstimate = np.dot(PCsExtended.transpose(), approxDecoders)

#if makePlots:
#    print("Plotting decoded estimate from principal components...")
#    f5 = plt.figure()
#    plt.plot(xExtended, approxEstimate)
#    f5.suptitle('Decoded estimate from principal components')

# cheating is over

if makePlots:
    # software simulation
    sim = nengo.Simulator(model)
    sim.run(6)
    print("Plotting simulation output...")
    t = sim.trange()
    f6 = plt.figure()
    plt.plot(t, sim.data(p1), label="Input")
    plt.plot(t, sim.data(p2), 'k', label="Integrator output")
    plt.legend()
    f6.suptitle('Software simulation output')
    plt.show()

# now attempt this in hardware
sim = nengo_rt_backend.Simulator(model, targetFile='target_basicboard.xml')
sim.run(6)

# FIXME get input

