"""
Simulator.py

Connection to programming and I/O interface on Nengo-RT hardware.
"""

import logging

import numpy as np

import nengo
from .builder import Builder
from .target import Target
from .programmer import EthernetProgrammer

log = logging.getLogger(__name__)

class Simulator(object):
    
    def __init__(self, model, dt=0.001, seed=None, builder=None, targetFile=None):
        if builder is None:
            self.builder = Builder(targetFile)
        else:
            self.builder = builder
            
        self.model = self.builder(model, dt)

    def run(self, time_in_seconds):
        """Simulate for the given length of time."""
        target = self.builder.target
        board = target.boards[0] # FIXME multi-board
        # choose the first controller we recognize and use that for programming
        programmer = None
        for ctrl in board.controls:
            if ctrl.type == 'ethernet':
                programmer = EthernetProgrammer(self.builder.filename,
                                                ctrl.mac_address, ctrl.device)
                break
            else:
                log.warn("ignoring unknown control type '" + ctrl.type + "'")

        if programmer is None:
            raise ValueError("no suitable control interface found for this board")
        
        log.info("programming...")
        programmer.program();
        
