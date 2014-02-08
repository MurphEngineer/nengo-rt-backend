import math
import logging
import copy

log = logging.getLogger(__name__)

class ScheduledRead(object):
    def __init__(self, readInfo):
        # associated (DV address, weight) info
        self.readInfo = readInfo
        # targeted DV buffer
        self.target = self.readInfo[0] >> 11 # 2^11 = 2048 addresses in each DV buffer
        # initial scheduled time
        self.time = 0

class EncoderScheduler(object):
    def __init__(self, encoderLists, optimizer=GeneticOptimizer):
        # encoderLists is a list whose elements are lists of (DV address, weight) data for each encoder
        self.encoderLists = encoderLists
        # scheduler object used to search for a solution
        self.optimizer = optimizer
        self.encoderSchedule = []
        # create the initial list of scheduled reads
        for eList in self.encoderLists:
            schedule = []
            for Rd in eList:
                schedule.append(ScheduledRead(Rd))
            self.encoderSchedule.append(schedule)

# Objective function for schedules:
# Given a schedule, return a value that is proportional
# to the quality of the schedule. 
# To simplify things, the objective function returns larger values
# for "worse" schedules, so we want to be minimizing this function.
def objective(schedule):
    badness = 0
    # we want to be minimizing the time at which the last read is made,
    # so the largest scheduled time contributes directly to the badness
    max_t = 0
    for encoder in schedule:
        for op in encoder:
            if op.time > max_t:
                max_t = op.time

    badness += max_t
    # we also want to be minimizing the number of read conflicts
    # (in fact, we'd like it to be 0) so we need to collect all reads made in each timestep
    # and see if any three of them hit the same bank
    collisionPenalty = 1000

    reads = [] * (max_t + 1)
    for encoder in schedule:
        for op in encoder:
            t = op.time
            reads[t].append(op)
    for ops in reads:
        bank_accesses = {}
        for op in ops:
            if op.time in ops:
                ops[op.time].append(op)
            else:
                ops[op.time] = [op]
        # count number of conflicting bank accesses; if this is >2, this is bad
        for accesses in bank_accesses.values():
            if len(accesses) > 2:
                badness += collisionPenalty * ( len(accesses) - 2)
    return badness

class BaseOptimizer(object):
    def __init__(self):
        pass
        
    def __call__(self, initialSchedule):
        raise NotImplementedException("Optimizer must implement __call__")
    


class GeneticOptimizer(BaseOptimizer):
    def __init__(self):
        BaseScheduler.__init__(self)
    
    def __call__(self, initialSchedule):
        pass 
