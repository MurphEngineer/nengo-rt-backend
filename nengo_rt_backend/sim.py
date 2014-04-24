# just putting some code here temporarily

import nengo
import numpy as np

def short_simulation(ensemble):
    steps = 500
    dt = .001

    assert type(ensemble.neurons == nengo.nonlinearities.LIF)
    bias = ensemble.neurons.bias

    spikes = np.zeros((len(bias), steps))
    J = bias
    
    spiked = np.zeros_like(J)
    voltage = np.zeros_like(J)
    refractory_time = np.zeros_like(J)
        
    for i in range(steps):
        ensemble.neurons.step_math0(dt, J, voltage, refractory_time, spiked)
#         ensemble.neurons.step_math(dt, J, spiked, voltage, refractory_time) #TODO: it looks like this in HEAD
        spikes[:,i] = spiked
    
    spikes = spikes.T
    return spikes
    
def decoded_std(connection, spikes):
    d = connection._decoders
    decoded = spikes.dot(d)
    return np.std(decoded)

tau = .005
model = nengo.Model(label='Communication Channel')
A1 = nengo.Ensemble(nengo.LIF(100, tau_rc=0.02, tau_ref=0.002), dimensions=1, label='Input')
A2 = nengo.Ensemble(nengo.LIF(100, tau_rc=0.02, tau_ref=0.002), dimensions=1, label='Output')
connection = nengo.Connection(A1, A2, transform=[[1]], filter=tau)
bdr = nengo.builder.Builder()
bdr.copy = False 
dt = .001
model = bdr(model, dt)

spikes = short_simulation(A1)    
print decoded_std(connection, spikes)

