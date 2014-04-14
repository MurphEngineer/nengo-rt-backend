import nengo
import nengo.helpers
import matplotlib.pyplot as plt

import nengo_rt_backend

hardware = True

# enable debug spew
import logging
logging.basicConfig(level=logging.DEBUG)

model = nengo.Model('Multiplier')
A = nengo.Ensemble(nengo.LIF(100), dimensions=1, radius=1.4, label='A')
B = nengo.Ensemble(nengo.LIF(100), dimensions=1, radius=1.4, label='B')
combined = nengo.Ensemble(nengo.LIF(224), dimensions=2, radius=2, label='combined')
prod = nengo.Ensemble(nengo.LIF(100), dimensions=1, radius=2, label='prod')

inputA = nengo.Node(nengo.helpers.piecewise({0: 0, 2.5: 0.5, 4: -0.5}))
inputB = nengo.Node(nengo.helpers.piecewise({0: 1.0, 1.5: 0.8, 3: 0.0, 4.5: 0.8}))

nengo.Connection(inputA, A)
nengo.Connection(inputB, B)

nengo.Connection(A, combined, transform=[[1], [0]])
nengo.Connection(B, combined, transform=[[0], [1]])

def product(x):
    return x[0] * x[1]

nengo.Connection(combined, prod, function=product)

#inputA_probe = nengo.Probe(inputA, 'output')
#inputB_probe = nengo.Probe(inputB, 'output')
A_probe = nengo.Probe(A, 'decoded_output', filter=0.01)
B_probe = nengo.Probe(B, 'decoded_output', filter=0.01)
#combined_probe = nengo.Probe(combined, 'decoded_output', filter=0.01)
prod_probe = nengo.Probe(prod, 'decoded_output', filter=0.01)

if hardware:
    # hardware simulation
    sim = nengo_rt_backend.Simulator(model, targetFile='target_1d2d.xml')
    sim.run(5)
else:
    # software simulation
    sim = nengo.Simulator(model)
    sim.run(5)
    plt.figure(1)
    plt.plot(sim.trange(), sim.data(A_probe), label='Decoded A')
    plt.plot(sim.trange(), sim.data(B_probe), label='Decoded B')
    plt.plot(sim.trange(), sim.data(prod_probe), label='Decoded product')
    plt.xlabel('Time (s)', fontsize='large')
    plt.legend(loc='best')
    plt.show()
