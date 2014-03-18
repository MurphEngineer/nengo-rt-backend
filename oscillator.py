import nengo
import nengo.helpers
import matplotlib.pyplot as plt

import nengo_rt_backend

hardware = True

# enable debug spew
import logging
logging.basicConfig(level=logging.DEBUG)

model = nengo.Model('Oscillator')
neurons = nengo.Ensemble(nengo.LIF(200), dimensions=2, label='van der Pol Oscillator')
input = nengo.Node(output=nengo.helpers.piecewise({0:[1,0], 0.1: [0,0]}), label='Initial condition')

tau = 0.1
freq = 5.0
i2o = nengo.Connection(input, neurons)
o2o = nengo.Connection(neurons, neurons, transform=[[1,-freq*tau], [freq*tau,1]], filter=tau)

input_probe = nengo.Probe(input, 'output')
neuron_probe = nengo.Probe(neurons, 'decoded_output', filter=tau)

if hardware:
    # hardware simulation
    sim = nengo_rt_backend.Simulator(model, targetFile='target_1d2d.xml')
    sim.run(5)
else:
    # software simulation
    sim = nengo.Simulator(model)
    sim.run(5)
    plt.figure(1)
    plt.subplot(2, 1, 1)
    plt.plot(sim.trange(), sim.data(neuron_probe))
    plt.xlabel('Time (s)', fontsize='large')
    plt.legend(['$x_0$', '$x_1$'])
    plt.subplot(2, 1, 2)
    data = sim.data(neuron_probe)
    plt.plot(data[:,0], data[:,1], label='Decoded output')
    plt.xlabel('$x_0$', fontsize=20)
    plt.ylabel('$x_1$', fontsize=20)
    plt.legend()
    plt.show()
