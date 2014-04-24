"""
Simulator.py

Connection to programming and I/O interface on Nengo-RT hardware.
"""

import logging

import numpy as np

import os

import nengo
from .builder import Builder
from .target import Target
from .controller import EthernetController
from .io import EthernetIOController

import time

log = logging.getLogger(__name__)

class Simulator(object):
    
    def __init__(self, model, dt=0.001, seed=None, builder=None, targetFile=None, keep=False):
        self.dt = dt
        self.keep = keep # whether to keep loadfile
        if builder is None:
            self.builder = Builder(targetFile)
        else:
            self.builder = builder
            
        self.model = self.builder(model, dt)
        self.controller = None
        self.ioctrl = None

        self.setUpControllers()
        self.elapsedTime = 0.0

    def __del__(self):
        if not self.keep:
            os.unlink(self.builder.filename)

    def setUpControllers(self):
        # Attach a programming and I/O controller
        target = self.builder.target
        board = target.boards[0] # FIXME multi-board
        # choose the first controller we recognize and use that for programming
        for ctrl in board.controls:
            if ctrl.type == 'ethernet':
                self.controller = EthernetController(self.builder.filename,
                                                ctrl.mac_address, ctrl.device)
                break
            else:
                log.warn("ignoring unknown control type '" + ctrl.type + "'")

        if self.controller is None:
            raise ValueError("no suitable control interface found for this board")

        for io in board.ios:
            if io.type == 'ethernet':
                self.ioctrl = EthernetIOController(io.mac_address, io.device)
                break
            else:
                log.warn("ignoring unknown I/O type '" + io.type + "'")

        if self.ioctrl is None:
            raise ValueError("no suitable I/O interface found for this board")

    def program(self):
        log.info("programming...")
        self.controller.program()

    def run(self, time_in_seconds):
        """Simulate (free-run) for the given length of time."""
        self.program() # FIXME we may want to run more than once; have a "is_programmed" flag
        
        # FIXME handle nodes

        # initialize probe data collection arrays
        self.probe_data = {}
        for probe in self.builder.probes:
            self.probe_data[probe] = []

        # build a buffer for probe data received during each timestep
        probe_buffer = {}
        for probe in self.builder.probes:
            probe_buffer[probe] = [0] * probe.dimensions

        log.info("starting free-running hardware simulation")
        self.controller.start()
        
        simulationStartTime = time.perf_counter(); # relative time
        run = True
        currentSimulationTime = self.elapsedTime # absolute time
        stepStartTime = simulationStartTime
        while run:
            lastStepTime = stepStartTime
            stepStartTime = time.perf_counter() # relative time
            currentSimulationTime += stepStartTime - lastStepTime
            if stepStartTime - simulationStartTime + self.dt >= time_in_seconds:
                run = False # finish after this step
            # receive output from the board
            receivedValues = self.ioctrl.recv() # this blocks for free synchronization
            for pair in zip(self.builder.probe_sequence, receivedValues):
                address = pair[0]
                data = pair[1]
                for probe in self.builder.probe_address_table[address]:
                    dim = probe.dimension_address_table[address]
                    probe_buffer[probe][dim] = data
            # now copy and append each probe buffer to the data array for that probe
            for probe in self.builder.probes:
                self.probe_data[probe].append( list(probe_buffer[probe]) )
            # send input to the board
            sentValues = []
            for node in self.builder.nodes:
                # the nodes are assigned addresses in this (ascending) order,
                # so iterating over them should give us consecutive outputs
                if node.optimized_out:
                    continue
                if len(node.inputs) == 0:
                    # output is only a function of simulation time
                    output = node.output(currentSimulationTime)
                else:
                    raise NotImplementedError("state-dependent nodes not yet supported")
                if output.ndim == 0:
                    sentValues.append(output)
                else:
                    for val in output:
                        sentValues.append(val)
            self.ioctrl.send(sentValues)
            
        log.info("done, pausing")
        self.controller.pause()
        self.elapsedTime += time_in_seconds
        
    def data(self, probe):
        vals = self.probe_data[probe]
        # if the number of values is larger than the number of timesteps,
        # throw away the first however many values, they are usually wrong
        nsteps = self.trange().shape[0]
        extraSamples = len(vals) - nsteps
        if extraSamples > 0:
            vals = vals[extraSamples:]
        return np.asarray(vals)

    def trange(self, dt=None):
        # parameter dt is ignored, it's only there for interface reasons.
        return np.arange(0.0, self.elapsedTime, self.dt)
