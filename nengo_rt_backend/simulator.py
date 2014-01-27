"""
Simulator.py

Connection to programming and I/O interface on Nengo-RT hardware.
"""

import logging

import numpy as np

import nengo
from .builder import Builder

log = logging.getLogger(__name__)

class Simulator(object):
    
    def __init__(self, model, dt=0.001, seed=None, builder=None, targetFile=None):
        if builder is None:
            builder = Builder(targetFile)
            
        self.model = builder(model, dt)

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        # FIXME reset, program, and control the board
        pass
