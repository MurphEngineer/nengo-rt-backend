import math
import logging
import copy
import numpy as np

log = logging.getLogger(__name__)

class ScheduledRead(object):
    def __init__(self, readInfo):
        # associated (DV address, weight) info
        self.readInfo = readInfo
        # targeted DV buffer
        self.target = self.readInfo[0] >> 11 # 2^11 = 2048 addresses in each DV buffer
        # initial scheduled time
        self.time = 0
    def __str__(self):
        return str(self.readInfo) + "@" + str(self.time)

class EncoderScheduler(object):
    def __init__(self, encoderLists, optimizer):
        # encoderLists is a list whose elements are lists of (DV address, weight) data for each encoder
        self.encoderLists = encoderLists
        # scheduler object used to search for a solution
        self.optimizer = optimizer()
        self.encoderSchedule = []
        # create the initial list of scheduled reads
        for eList in self.encoderLists:
            schedule = []
            for Rd in eList:
                schedule.append(ScheduledRead(Rd))
            self.encoderSchedule.append(schedule)

    def __call__(self):
        return self.optimizer(self.encoderSchedule)

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

    reads = [ [] for i in range(max_t + 1) ]
    # collect encoding operations in the list "reads", indexed by operation time
    for encoder in schedule:
        for op in encoder:
            t = op.time
            reads[t].append(op)
    # now iterate over the list of reads, and for every set of operations taking place
    # during the same cycle, check for target conflicts by inserting each operation
    # into a dictionary indexed by target address
    for ops in reads:
        bank_accesses = {}
        for op in ops:
            if op.target in bank_accesses.keys():
                bank_accesses[op.target].append(op)
            else:
                bank_accesses[op.target] = [op]
        # count number of conflicting bank accesses; if this is >2, this is bad
        for accesses in bank_accesses.values():
            if len(accesses) > 2:
                badness += collisionPenalty * (len(accesses) - 2)
    return badness

class BaseOptimizer(object):
    def __init__(self):
        self.rng = np.random.RandomState()
        
    def __call__(self, initialSchedule):
        raise NotImplementedException("Optimizer must implement __call__")
    


class GeneticOptimizer(BaseOptimizer):
    def __init__(self):
        BaseOptimizer.__init__(self)
        # number of solutions to maintain in each generation
        self.solutions_per_generation = 100
        # threshold above which solutions are unacceptable
        self.threshold = 128
        # maximum number of generations before giving up
        self.max_generations = 1000
        # maximum number of consecutive generations with no progress before giving up, 
        # assuming we are already below the threshold
        self.max_run_without_progress = 50
        # percentage (between 0 and 1) of new generations drawn directly from 
        # individuals in previous generation (elitist replacement)
        self.pct_elitist = 0.10
        # number of competitors in tournament selection
        self.selection_pressure = 10
        # probability (0-1) of a mutation affecting any given location after crossover
        self.mutation_rate = 0.01
        # relative probability of a "stronger" mutation, i.e. taking the mutated value further from its initial one
        self.mutation_strength = 0.10

        self.previous_generation = []
        self.current_generation = []
        self.best_feasible_candidate = None
        self.best_feasible_candidate_badness = None
        self.no_progress_run = 0 
    
    def __call__(self, initialSchedule):
        log.info("using GeneticOptimizer")
        # create the initial generation by randomly creating (solutions_per_generation) schedules
        log.debug("creating initial group of " + str(self.solutions_per_generation) + " candidate schedules")
        for i in range(self.solutions_per_generation):
            # start by copying the old schedule...
            candidate = copy.deepcopy(initialSchedule)
            for encoder in candidate:
                if len(encoder) == 0:
                    continue
                else:
                    # choose a random start time for the first operation
                    t = self.rng.randint(0, self.threshold)
                    encoder[0].time = t
                    for op in range(1, len(encoder)):
                        # increase t by at least 2 for each consecutive operation
                        t += self.rng.randint(2, self.threshold)
                        encoder[op].time = t
            # now put the candidate into our current generation
            self.current_generation.append(candidate)
        # now the GA can begin
        for generation in range(1, self.max_generations + 1):
            log.info("generation " + str(generation))
            best = objective(self.current_generation[0])
            bestIdx = 0
            # list of objective values of (soon-to-be) previous generation
            ranks = []
            for i in range(1, len(self.current_generation)):
                candidate = self.current_generation[i]
                challenge = objective(candidate)
                ranks.append(challenge)
                if challenge < best:
                    best = challenge
                    bestIdx = i
            log.info("best schedule in this generation (#" + str(bestIdx) + ") has badness of " + str(best))

            # keep a copy of this solution if it is the best seen so far
            if self.best_feasible_candidate is None or best < self.best_feasible_candidate_badness:
                self.best_feasible_candidate = copy.deepcopy(self.current_generation[bestIdx])
                self.best_feasible_candidate_badness = best
                self.no_progress_run = 0
                log.debug("new best feasible candidate")
            # otherwise, we have made no progress
            elif (self.best_feasible_candidate_badness is not None
                  and self.best_feasible_candidate_badness < self.threshold) :
                self.no_progress_run += 1
                if self.no_progress_run >= self.max_run_without_progress:
                    log.info("no progress for " + str(self.no_progress_run) + " generations and threshold reached")
                    log.info("returning best schedule, with badness of " + str(self.best_feasible_candidate_badness))
                    return self.best_feasible_candidate

            self.previous_generation = self.current_generation.copy()
            self.current_generation = []
            # draw some proportion of individuals directly from the previous generation 
            # to continue to the new generation
            elitist_ranks = ranks[:] # copy array to allow modification, in order to prevent multiple selection
            for i in range( int(self.solutions_per_generation * self.pct_elitist)):
                idx = self.tournament(elitist_ranks)
                elitist_ranks[idx] = None
                individual = self.previous_generation[idx]
                self.current_generation.append(individual)
            while len(self.current_generation) < self.solutions_per_generation:
                # choose two parent solutions and perform crossover to get offspring
                # again, start by copying the array, so we can prevent multiple selection
                parent_ranks = ranks[:]
                idxParent1 = self.tournament(parent_ranks)
                parent_ranks[idxParent1] = None
                parent1 = self.previous_generation[idx]
                idxParent2 = self.tournament(parent_ranks)
                parent_ranks[idxParent2] = None
                parent2 = self.previous_generation[idx]
                # now combine parent1 and parent2 to generate new solutions
                offspring1 = copy.deepcopy(parent1) # in practice, which parent is copied is arbitrary
                offspring2 = copy.deepcopy(parent2) # since they both have the same instructions
                # iterate over encoders...
                for geneIdx in range(len(parent1)):
                    # don't waste time if this is an empty encoder
                    if len(parent1[geneIdx]) == 0:
                        continue
                    # encode both parents' instruction timings for this encoder
                    # as a sequence of the form
                    # [start time, next time-2, next time-2, ...]
                    gene_parent1 = self.phenotypeToGenotype(parent1[geneIdx])
                    gene_parent2 = self.phenotypeToGenotype(parent2[geneIdx])
                    # for genes of length n, there are (n-1) crossover points that result in a
                    # non-trivial crossover
                    crossoverPoint = self.rng.randint(0, len(gene_parent1) - 1)
                    (gene_offspring1, gene_offspring2) = self.crossover(gene_parent1, gene_parent2, crossoverPoint)
                    # perform mutation on the resulting genes
                    gene_offspring1 = self.mutation(gene_offspring1)
                    gene_offspring2 = self.mutation(gene_offspring2)
                    # now translate back to a phenotype and modify the instruction timings for the offspring
                    times_offspring1 = self.genotypeToPhenotype(gene_offspring1)
                    times_offspring2 = self.genotypeToPhenotype(gene_offspring2)
                    for i in range(len(times_offspring1)):
                        offspring1[geneIdx][i].time = times_offspring1[i]
                        offspring2[geneIdx][i].time = times_offspring2[i]
                    # add to list and repeat
                    self.current_generation.append(offspring1)
                    self.current_generation.append(offspring2)
        log.warn("generation limit reached")
        if self.best_feasible_candidate_badness > self.threshold:
            log.warn("solution does not meet threshold")
        log.info("returning best schedule, with badness of " + str(self.best_feasible_candidate_badness))
        return self.best_feasible_candidate

    # k-way tournament selection
    def tournament(self, ranks):
        k = self.selection_pressure
        best = None
        for i in range(k):
            idx = -1
            # don't count choosing a rank of "None" as a tournament
            while idx == -1 or ranks[idx] is None:
                idx = self.rng.randint(0, len(ranks))

            if best is None or ranks[idx] < ranks[best]:
                best = idx
        return best

    # Encode a list of ScheduledReads as a gene suitable for crossover,
    # of the form [first time, next time-2, next time-2, ...].
    # The (-2) ensures that all non-negative values result in a feasible solution,
    # since read times are separated by at least two clock cycles.
    def phenotypeToGenotype(self, insnlist):
        gene = []
        if len(insnlist) > 0:
            gene.append(insnlist[0].time)
            for i in range(1, len(insnlist)):
                previousTime = gene[-1]
                gene.append(insnlist[i].time - previousTime - 2)
        return gene

    # Decode a genotype into a list of *TIMES* for ScheduledReads.
    def genotypeToPhenotype(self, gene):
        times = []
        if len(gene) > 0:
            times.append(gene[0])
            for i in range(1, len(gene)):
                previousTime = times[-1]
                times.append(2 + previousTime + gene[i])
        return times

    def crossover(self, gene1, gene2, crossPt):
        new1 = []
        new2 = []
        for i in range( len(gene1) ):
            if i <= crossPt:
                new1.append(gene1[i])
                new2.append(gene2[i])
            else:
                new1.append(gene2[i])
                new2.append(gene1[i])
        return (new1, new2)

    def mutation(self, gene):
        mutatedGene = gene[:]
        condition = self.rng.random_sample(len(gene))
        for i in range(len(gene)):
            if condition[i] < self.mutation_rate:
                delta = 1
                # sample self.mutation_strength in the same way to (potentially) increase this
                while self.mutation_strength < self.rng.random_sample():
                    delta += 1
                # flip a coin to choose the sign of delta
                if self.rng.random_integers(0, 1) == 1:
                    delta *= -1
                # check that this does not cause the location to go below zero
                if mutatedGene[i] + delta < 0:
                    mutatedGene[i] = 0
                else:
                    mutatedGene[i] += delta
        return mutatedGene
