import numpy as np
import scipy
import scipy.cluster
import math

#from .target import Target, Board, Output, Input, Control, IO
from .target import Target

from .scheduler import EncoderScheduler, GeneticOptimizer

import os
import logging
import xml.etree.ElementTree as ET
import datetime

import nengo
import nengo.builder
import nengo.decoders
import nengo.nonlinearities

# for visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

log = logging.getLogger(__name__)
  
def principal_component_distance(P1, P2):
    assert(P1.principal_components.shape == P2.principal_components.shape)
    result = 0
    # zip together corresponding principal components
    for pc in zip(P1.principal_components, P2.principal_components):
        # try both the original PCs and the negative of one of the PCs,
        # and take the smaller distance
        # FIXME normalization
        r1 = scipy.spatial.distance.euclidean(pc[0], pc[1])
        r2 = scipy.spatial.distance.euclidean(pc[0], -1 * pc[1])
        result += min(r1, r2)
    
    return result

def filter_distance(P1, P2):
    # FIXME
    return 0.0

def object_pdist(objs, metric):
    # Like scipy.spatial.distance.pdist, but can take an ARBITRARY sequence
    # of objects, provided the 2-arity metric can be called on them.
    X = np.asarray(objs, order='c')
    s = X.shape
    m = s[0]
    dm = np.zeros((m * (m-1)) // 2, dtype=np.double)
    k = 0
    for i in range(0, m-1):
        for j in range(i+1, m):
            dm[k] = metric(X[i], X[j])
            k += 1
    
    return dm

# Prepend p to the string s until its length is at least n.
def pad(s, p, n):
    x = s
    while len(x) < n:
        x = p + x
    return x

def float2sfixed(f):
    if f < 0:
        sgn = "1"
        f = f * -1
    else:
        sgn = "0"
    if f > 2047.0 / 1024.0:
        log.warn("clamped unrepresentable value " + str(f))
        f = 2047.0 / 1024.0 # largest representable value
    ipart = int(math.floor(f))
    fpart = int(round((f - ipart) * 2**10))
    istr = pad(bin(ipart)[2:], "0", 1) # remove '0b' prefix and pad out
    fstr = pad(bin(fpart)[2:], "0", 10)
    # if the string is negative, take the 2's complement
    if sgn == "1":
        # take two's complement
        n = int(istr + fstr, 2)
        nc = 2**12 - n
#        return sgn + pad(bin(nc)[2:], "1", 11)
        return pad(bin(nc)[2:], "1", 12)
    else:
        return sgn + istr + fstr

# Returns an array of dimensions (nX * nY, 2)
# where nX = (1 + (x2-x1)/xStep) and similar for nY.
# Values in the first dimension of the returned array run faster than
# values in the second dimension.
def samplepoints2d(x1, x2, xStep, y1, y2, yStep):
    xRange = np.arange(x1, x2, xStep)
    yRange = np.arange(y1, y2, yStep)
    result = []
    for y in yRange:
        for x in xRange:
            result.append([x, y])
    return np.array(result)

class ShapeMismatch(ValueError):
    pass

class NotFeasibleError(ValueError):
    pass

class Builder(object):
    def __init__(self, targetFile=None, weightPCDistance=0.5, weightFilterDistance=0.5):
        self.targetFile = targetFile
        self.weightPCDistance = weightPCDistance
        self.weightFilterDistance = weightFilterDistance
        # a useful constant
        self.max_12bit_value = 2047.0 / 1024.0 # about 1.999, or "0111 1111 1111" as sfixed(1 downto -10)

    def __call__(self, model, dt):
        if dt != 0.001:
            raise ValueError("Timestep must be 0.001 for this backend")

        # attempt to call the default Nengo builder on the model.
        builder = nengo.builder.Builder(copy=False)
        self.model = builder(model, dt)
        self.model.dt = dt
        if self.model.seed is None:
            self.model.seed = np.random.randint(np.iinfo(np.int32).max)

        self.populations = []
        self.connections = []
        self.nodes = []
        self.probes = []

        # This is a list of decoded value addresses corresponding to the order in which
        # DVs come back from the board (over Ethernet or whatever).
        # So, the first value received goes to the first address in this list,
        # the second goes to the next, and so on.
        self.probe_sequence = []

        log.info("Collecting populations and nodes")
        for obj in self.model.objs:
            if isinstance(obj, nengo.Ensemble):
                self.populations.append(obj)
            elif isinstance(obj, nengo.Node):
                self.nodes.append(obj)
            else:
                raise TypeError("Don't know how to build object " + str(obj))

        log.debug("Collected " + str(len(self.populations)) + " populations and " + str(len(self.nodes)) + " nodes")

        log.info("Separating 1D and 2D populations")
        self.populations_1d = []
        self.populations_2d = []
        for population in self.populations:
            if population.dimensions == 1:
                self.populations_1d.append(population)
            elif population.dimensions == 2:
                self.populations_2d.append(population)
        log.debug("Separated into " + str(len(self.populations_1d)) + " 1D populations" + 
                  " and " + str(len(self.populations_2d)) + " 2D populations")

        log.info("Collecting probes")
        for probe in self.model.probed.values():
            self.probes.append(probe)

        log.debug("Collected " + str(len(self.probes)) + " probes")

        log.info("Collecting connections")
        for conn in self.model.connections:
            self.connections.append(conn)

        log.debug("Collected " + str(len(self.connections)) + " connections")

        log.info("Building populations")
        for p in self.populations:
            self.build_ensemble(p)

        log.info("Building nodes")
        for n in self.nodes:
            self.build_node(n)

        log.info("Building probes")
        for p in self.probes:
            self.build_probe(p)

        log.info("Building connections")
        for c in self.connections:
            self.build_connection(c)

        # VERIFY: among all connections into an ensemble, there are at most two distinct non-empty filter constants
        log.info("Verifying feasibility of incoming connection filters")
        for population in self.populations:
            filters = []
            for conn in population.inputs:
                if conn.filter is not None and conn.filter not in filters:
                    filters.append(conn.filter)
            
            log.debug("Filter time constants for " + population.label + ": " 
                      + str(filters))

            if len(filters) > 2:
                raise ValueError(
                    "Ensemble " + population.label + " has a total of " + str(len(filters)) + 
                    " distinct filter time constants among all incoming connections."
                    " The maximum number supported by this backend is 2.")

            population.filters = list(set(filters)) # eliminate duplicates.

        if self.targetFile is None:
            raise ValueError("targetFile not specified")
        else:
            # parse target file
            self.target = Target(self.targetFile)
            if len(self.target.boards) > 1:
                raise NotFeasibleError(
                    "Multi-board targets not yet supported")
            log.info("Targeting device with " + str(self.target.total_population_1d_count) + 
                     " one-dimensional populations"
                     " and " + str(self.target.total_population_2d_count) + 
                     " two-dimensional populations")

        # FIXME: before we can do this, check the number of decoded values from each population
        # and if there are more than four
        # (counting EACH DIMENSION as one)
        # split the population into copies until there are at most
        # four decoded values coming from each one

        # perform clustering of 1D populations
        if len(self.populations_1d) == 0:
            log.info("No 1D populations")
            self.population_clusters_1d = []
        elif len(self.populations_1d) == 1:
            log.info("Only one 1D population, clustering is trivial")
            # fake the outcome of clustering
            cluster_assignments = [1]
            square_dist = np.array([[0.0]])
        else:
            log.info("Clustering 1D population groups")
            dist = object_pdist(self.populations_1d, self.population_distance)
            nonzero_dist = dist[dist > 0.0]
            square_dist = scipy.spatial.distance.squareform(dist)
            log.debug("Maximum distance is " + str(max(dist)))
            log.debug("Minimum non-zero distance is " + str(min(nonzero_dist)))
            log.debug("Average non-zero distance is " + str(np.mean(nonzero_dist)))        
 
        # visualize
#        log.debug("Plotting distance matrix")
#        imgplot = plt.matshow(square_dist)
#        plt.show()
        
            linkage = scipy.cluster.hierarchy.linkage(dist, method='single')
        # visualize
#        log.debug("Plotting clustered dendrogram")
#        scipy.cluster.hierarchy.dendrogram(linkage)
#        plt.show()
            cluster_assignments = scipy.cluster.hierarchy.fcluster(linkage, criterion='maxclust',
                                                               t=self.target.total_population_1d_count)

        if len(self.populations_1d) > 0:
            log.info(str(max(cluster_assignments) - min(cluster_assignments) + 1) 
                     + " clusters formed")
            cluster_sizes = np.bincount(cluster_assignments)[1:] # the first element of the original array is always 0
            log.debug("1D population cluster sizes: " + os.linesep + 
                          str(cluster_sizes))

            # assign populations to clusters
            self.population_clusters_1d = [ [] for i in range(max(cluster_assignments)) ]
            self.cluster_principal_components_1d = []
            self.cluster_pc_scale_factor_1d = []
            self.cluster_representatives_1d = []

            for i in range(len(cluster_assignments)):
                population = self.populations_1d[i]
                population.population_list_idx = i # used to index into the distance matrix
                cluster_idx = cluster_assignments[i] - 1 # our clusters start at 0
                self.population_clusters_1d[cluster_idx].append(population)
                population.cluster_idx = cluster_idx

            # calculate the representative principal components of each cluster
            log.info("Calculating clustered 1D principal components")
            for cluster in self.population_clusters_1d:
                # Calculate representative principal components by finding a representative population in each cluster.
                # To simplify things, this will be the population with the least maximum distance
                # to another population in the same cluster.
                # Use the precomputed distance matrix square_dist to help us
                candidate_idx = -1
                candidate_minimum = np.inf
                for challenger in cluster:
                    # find maximum distance between challenger and another population
                    max_distance = 0
                    for matched in cluster:
                        new_distance = square_dist[challenger.population_list_idx, matched.population_list_idx]
                        if new_distance > max_distance:
                            max_distance = new_distance
                    if max_distance < candidate_minimum:
                        candidate_idx = challenger.population_list_idx
                        candidate_minimum = max_distance
                candidate = self.populations_1d[candidate_idx]
                log.debug("Selected population #" + str(candidate_idx) + 
                          "( " + candidate.label + ") as candidate")
                self.cluster_representatives_1d.append(candidate)
                pc_rep = candidate.principal_components
                # scale to +/-max_12bit_value
                pc_max = np.absolute(pc_rep).max(axis=1)

                scale_factors = []

                for i in range(pc_rep.shape[0]):
                    scale_factor = self.max_12bit_value / pc_max[i]
                    # save this for when we recompute decoders
                    scale_factors.append(scale_factor)
                    for j in range(pc_rep.shape[1]):
                        pc_rep[i, j] *= scale_factor
                    log.debug("Scale factor for candidate's " + str(i) 
                              + "-order component: " + str(scale_factor))
                self.cluster_pc_scale_factor_1d.append(scale_factors)
                self.cluster_principal_components_1d.append(pc_rep)

            # calculate representative filter coefficients for each cluster
            log.info("Collecting filter coefficients for 1D clusters")
            self.cluster_filters_1d = []
            for cluster in self.population_clusters_1d:
                n = len(cluster)
                filters = set()
                for population in cluster:
                    for coefficient in population.filters:
                        filters.add(coefficient)
                # add zeros until we have at least two
                filters = list(set(filters))
                while len(filters) < 2:
                    filters.append(0.0)
                # if there are two now, we're done
                if len(filters) == 2:
                    self.cluster_filters_1d.append(filters)
                else:
                    raise NotFeasibleError(
                        "FIXME: cluster has >2 distinct filters, deal with this case")

            # VERIFY: at most 1024 populations per cluster; if this is violated,
            # move populations from over-populated clusters one at a time to the
            # nearest (by metric) cluster with room left
            for cluster in self.population_clusters_1d:
                if len(cluster) > 1024:
                    raise NotFeasibleError(
                        "FIXME: cluster has >1024 populations, implement reassignment")

            # FIXME do statistics on populations vs. cluster principal components

            # so at this point, we have clusters of at most 1024 populations,
            # each with at most four outgoing connections,
            # and for each cluster, we know its filter coefficients and its principal components.
            # the next step is to go through each outgoing connection,
            # recompute its decoders with respect to the "correct" principal components,
            # and assign it an address in the decoded value address space.
            # Assume that population unit N connects to DV buffers (2N) and (2N+1);
            # then, population #A running on population unit #N writes decoded value #D
            # to address [NNNNNNN][DD][AAAAAAAAAA].
            log.info("Calculating decoders for connections leaving 1D populations")
            for N in range(len(self.population_clusters_1d)):
                cluster = self.population_clusters_1d[N]
                # cluster[N] runs on population unit N for 1D clusters            
                scale_factor = self.cluster_pc_scale_factor_1d[N]
                log.debug("Scaling populations on unit " + str(N) + " with scale factor " +
                          str(scale_factor))
                pop_idx = 0
                for population in cluster:
                    decoder_idx = 0
                    for conn in population.outputs:
                        # recompute decoders wrt. representative population
                        rep = self.cluster_representatives_1d[N]
                        self.recompute_decoders(rep, conn)
                        conn._decoders = np.real(conn._decoders)

                        # (PC * decoder) = (PC * scale factor * 1/scale factor * decoder)
                        # = (PC * scale_factor) * (1/scale factor * decoder);
                        # adjust conn._decoders by 1/scale factor
                        log.debug("recalculated decoders for " + conn.label + ": ")
                        log.debug(str(conn._decoders))
                        conn._decoders /= scale_factor
                        log.debug("scaled decoders for " + conn.label + ": ")
                        log.debug(str(conn._decoders))
                        decoders_max = np.absolute(conn._decoders).max()
                        if decoders_max > self.max_12bit_value:
                            log.warn("decoders for " + conn.label + " out of range (" + str(decoders_max) +
                                     "), rescaling")
                            decoder_scale_factor = self.max_12bit_value / decoders_max
                            conn._decoders *= decoder_scale_factor
                            conn.decoder_inverse_scale_factor = 1.0 / decoder_scale_factor
                        else:
                            # decoders don't need rescaling
                            conn.decoder_inverse_scale_factor = 1.0
                        # VERIFY: N < 96, D < 4, A < 1024
                        if N >= 96:
                            raise NotFeasibleError("Inconsistency detected: more than 96 clusters")
                        if decoder_idx >= 4 or decoder_idx + conn.dimensions > 4 :
                            raise NotFeasibleError("Inconsistency detected: population decodes more than 4 values")
                        if pop_idx >= 1024:
                            raise NotFeasibleError("Inconsistency detected: more than 1024 populations in cluster")
                        # at this point, we can correctly assign addresses to this connection
                        # TODO this assumes single-board targets; use different address spaces for multi-board
                        conn.decoded_value_addrs = []                
                        for i in range(conn.dimensions):
                            conn.decoded_value_addrs.append( (N << 12) + (decoder_idx << 10) + (pop_idx) )
                            decoder_idx += 1
                        log.debug("addresses for " + conn.label + " are " 
                                  + str(conn.decoded_value_addrs))
                    pop_idx += 1

        # perform clustering of 2D populations
        if len(self.populations_2d) == 0:
            log.info("No 2D populations")
            self.population_clusters_2d = []
        elif len(self.populations_2d) == 1:
            log.info("Only one 2D population, clustering is trivial")
            # fake the outcome of clustering
            cluster_assignments = [1]
            square_dist = np.array([[0.0]])
        else:
            log.info("Clustering 2D population groups")
            dist = object_pdist(self.populations_2d, self.population_distance)
            nonzero_dist = dist[dist > 0.0]
            square_dist = scipy.spatial.distance.squareform(dist)
            log.debug("Maximum distance is " + str(max(dist)))
            log.debug("Minimum non-zero distance is " + str(min(nonzero_dist)))
            log.debug("Average non-zero distance is " + str(np.mean(nonzero_dist)))
            linkage = scipy.cluster.hierarchy.linkage(dist, method='single')
            cluster_assignments = scipy.cluster.hierarchy.fcluster(linkage, criterion='maxclust',
                                                                   t=self.target.total_population_2d_count)
        if len(self.populations_2d) > 0:
            log.info(str(max(cluster_assignments)-min(cluster_assignments)+1) + " clusters formed")
            cluster_sizes = np.bincount(cluster_assignments)[1:]
            log.debug("2D population cluster sizes: " + os.linesep + str(cluster_sizes))
            # assign populations to clusters
            self.population_clusters_2d = [ [] for i in range(max(cluster_assignments)) ]
            self.cluster_principal_components_2d = []
            self.cluster_pc_scale_factor_2d = []
            self.cluster_representatives_2d = []

            for i in range(len(cluster_assignments)):
                population = self.populations_2d[i]
                population.population_list_idx = i # used to index into the distance matrix
                cluster_idx = cluster_assignments[i] - 1 # our clusters start at 0
                self.population_clusters_2d[cluster_idx].append(population)
                population.cluster_idx = cluster_idx
            # calculate the representative principal components of each cluster
            log.info("Calculating clustered 2D principal components")
            for cluster in self.population_clusters_2d:
                # As before, calculate representative principal components by finding
                # a representative population in each cluster.
                candidate_idx = -1
                candidate_minimum = np.inf
                for challenger in cluster:
                    max_distance = 0
                    for matched in cluster:
                        new_distance = square_dist[challenger.population_list_idx,
                                                   matched.population_list_idx]
                        if new_distance > max_distance:
                            max_distance = new_distance
                    if max_distance < candidate_minimum:
                        candidate_idx = challenger.population_list_idx
                        candidate_minimum = max_distance
                candidate = self.populations_2d[candidate_idx]
                log.debug("Selected population #" + str(candidate_idx) + 
                          " (" + candidate.label + ") as candidate")
                self.cluster_representatives_2d.append(candidate)
                pc_rep = candidate.principal_components
                # scale to +/-max_12bit_value
                pc_max = np.absolute(pc_rep).max(axis=1)
                scale_factors = []
                for i in range(pc_rep.shape[0]):
                    scale_factor = self.max_12bit_value / pc_max[i]
                    scale_factors.append(scale_factor)
                    for j in range(pc_rep.shape[1]):
                        pc_rep[i, j] *= scale_factor
                    log.debug("Scale factor for candidate's " + str(i)
                              + "-order component: " + str(scale_factor))
                self.cluster_pc_scale_factor_2d.append(scale_factors)
                self.cluster_principal_components_2d.append(pc_rep)
            # calculate representative filter coefficients for each cluster
            log.info("Collecting filter coefficients for 2D clusters")
            self.cluster_filters_2d = []
            for cluster in self.population_clusters_2d:
                n = len(cluster)
                filters = set()
                for population in cluster:
                    for coefficient in population.filters:
                        filters.add(coefficient)
                # add zeros until we have at least two
                filters = list(set(filters))
                while len(filters) < 2:
                    filters.append(0.0)
                if len(filters) == 2:
                    self.cluster_filters_2d.append(filters)
                else:
                    raise NotFeasibleError(
                        "FIXME: cluster has >2 distinct filters, deal with this case")
            # VERIFY: as before, at most 1024 populations per cluster.
            for cluster in self.population_clusters_2d:
                if len(cluster) > 1024:
                    raise NotFeasibleError(
                        "FIXME: cluster has >1024 populations, implement reassignment")
            # FIXME do statistics on populations vs. cluster principal components
            # as before, recompute decoders on all outgoing connections
            # and assign addresses in DV address space.
            # Even for 2D populations, population unit N connects to DV buffers 2N and 2N+1.
            # So population #A running on population unit #N writes decoded value #D
            # to address [NNNNNNN][DD][AAAAAAAAAA].
            log.info("Calculating decoders for connections leaving 2D populations")
            for clusterN in range(len(self.population_clusters_2d)):
                cluster = self.population_clusters_2d[clusterN]
                # cluster[clusterN] runs on population unit (95-clusterN) for 2D clusters
                N = 95 - clusterN
                scale_factor = self.cluster_pc_scale_factor_2d[clusterN]
                log.debug("Scaling populations on unit " + str(N) + " with scale factor " +
                          str(scale_factor))
                pop_idx = 0
                for population in cluster:
                    decoder_idx = 0
                    for conn in population.outputs:
                        rep = self.cluster_representatives_1d[clusterN]
                        self.recompute_decoders(rep, conn)
                        conn._decoders = np.real(conn._decoders)

                        log.debug("old decoders for " + conn.label + ": ")
                        log.debug(str(conn._decoders))
                        conn._decoders /= scale_factor
                        log.debug("new decoders for " + conn.label + ": ")
                        log.debug(str(conn._decoders))
                        decoders_max = np.absolute(conn._decoders).max()
                        if decoders_max > self.max_12bit_value:
                            log.warn("decoders for " + conn.label + " out of range (" +
                                     str(decoders_max) + "), rescaling")
                            decoder_scale_factor = self.max_12bit_value / decoders_max
                            conn._decoders *= decoder_scale_factor
                            conn.decoder_inverse_scale_factor = 1.0 / decoder_scale_factor
                        else:
                            conn.decoder_inverse_scale_factor = 1.0
                        # VERIFY: N >= 0, D < 4, A < 1024
                        if N < 0:
                            raise NotFeasibleError("Inconsistency detected: more than 96 clusters")
                        if decoder_idx >= 4 or decoder_idx + conn.dimensions > 4:
                            raise NotFeasibleError("Inconsistency detected: population decodes more than 4 values")
                        if pop_idx >= 1024:
                            raise NotFeasibleError("Inconsistency detected: more than 1024 populations in cluster")
                        # at this point, we can correctly assign addresses to this connection
                        # TODO this also assumes single-board targets
                        conn.decoded_value_addrs = []
                        for i in range(conn.dimensions):
                            conn.decoded_value_addrs.append( (N<<12) + (decoder_idx<<10) + (pop_idx))
                            decoder_idx += 1
                        log.debug("addresses for " + conn.label + " are " + 
                                  str(conn.decoded_value_addrs))
                    pop_idx += 1

        # we have now clustered all 1D and 2D populations onto population units,
        # assigned filter coefficients, and calculated decoders and DV addresses for
        # all connections leaving 1D and 2D populations.
        # we should now be ready to deal with connections leaving nodes,
        # which are INPUTS with respect to the hardware (because the values come from the host simulation)
        # and are OUTPUTS with respect to the nodes (because they are values provided from the nodes)

        log.info("Assigning input addresses to nodes")

        maximum_input_count = self.target.boards[0].input_dv_count * 2048 # TODO this assumes single-board

        first_input_address = (self.target.boards[0].first_input_dv_index << 11) # TODO this assumes single-board
        input_address = first_input_address

        log.debug("Input space begins at address " + str(first_input_address) + 
                  ", maximum of " + str(maximum_input_count) + " inputs available")

        for node in self.nodes:
            if len(node.outputs) == 0:
                # no outputs, so nothing to do
                continue

            if len(node.inputs) == 0:
                # node is output-only, which means output is sourced from the host side.
                # allocate it an address in input-space
                if isinstance(node.output, (int, float)): # FIXME extend this to arrays of these
                    # OPTIMIZATION: output is a constant.
                    output_signal = [node.output]
                    log.debug("optimizing out node " + node.label + ", constant " + str(output_signal))
                    node.optimized_out = True
                    node.initial_value = output_signal
                    node.output_dimensions = 1
                    node.output_addrs = [input_address]
                    log.debug("Node " + node.label + " input addresses: " + str(node.output_addrs))
                    input_address += node.output_dimensions
                elif isinstance(node.output, complex):
                    # OPTIMIZATION: output is a constant.
                    # FIXME check this against what Nengo does internally.
                    # Assuming we just treat this as a 2-D output in rectangular form.
                    output_signal = [node.output.real, node.output.imag]
                    log.debug("optimizing out node " + node.label + ", constant " + str(output_signal))
                    node.optimized_out = True
                    node.initial_value = output_signal
                    node.output_dimensions = 2
                    node.output_addrs = [input_address, input_address+1]
                    log.debug("Node " + node.label + " input addresses: " + str(node.output_addrs))
                    input_address += node.output_dimensions
                else:
                    # output is a function (of simulation time)
                    # do what the default builder does to guess the dimensionality of the output:
                    # assume the function takes 1 argument (simulation time) and call it with 0.0
                    node.initial_value = node.output(0.0)
                    if node.initial_value.ndim == 0:
                        node.initial_value = np.asarray([node.initial_value])
                    node.output_dimensions = node.initial_value.size
                    log.debug("node " + node.label + " is a state-invariant " + 
                              str(node.output_dimensions) + "-dimensional function")
                    node.output_addrs = []
                    for i in range(node.output_dimensions):
                        node.output_addrs.append(input_address)
                        input_address += 1
                    log.debug("Node " + node.label + " input addresses: " + str(node.output_addrs))
            else:
                if node.output is None:
                    # in this special case, Nengo says that the output is the identity function
                    log.debug("optimizing out node " + node.label + ", identity function")
                    node.optimized_out = True
                    # if the transform is also identity, this should be as easy as
                    # re-mapping the addresses of incoming connections onto the outgoing ones...
                    # FIXME
                    raise NotImplementedError("identity-function node encountered, but optimization of this case "
                                              "is not yet supported")
                else:
                    # node has both inputs and outputs
                    # do what the default builder does to guess the dimensionality of the output:
                    # assume the function takes 2 arguments.
                    # the first is simulation time (try 0.0)
                    # the second is the input state (try an array of all zeroes whose length is the input dim.)
                    node.initial_value = node.output(0.0, np.zeros(node.dimensions))
                    if node.initial_value.ndim == 0:
                        node.initial_value = np.asarray([node.initial_value])
                    node.output_dimensions = node.initial_value.size
                    log.debug("node " + node.label + " is a " + 
                              str(node.dimensions) + "-to-" + str(node.output_dimensions) +
                              " host-side function")
                    node.output_addrs = []
                    for i in range(node.output_dimensions):
                        node.output_addrs.append(input_address)
                        input_address += 1
                    log.debug("Node " + node.label + " input addresses: " + str(node.output_addrs))
            # now use the node's output addresses to give DV addresses to each outgoing connection
            for conn in node.outputs:
                conn.decoded_value_addrs = node.output_addrs

        input_count = input_address - first_input_address
        log.info("total of " + str(input_count) + " inputs in this model")

        if input_count == 0:
            log.warn("no inputs detected in this model, possible model inconsistency or optimization error")
        elif input_count > maximum_input_count:
            raise NotFeasibleError("model has too many inputs; attempted to assign " + str(input_count) +
                                   " but only " + str(maximum_input_count) + " available on this target")
        else:
            log.debug("input address range is " + str(first_input_address) + " to " + str(input_address - 1))

        # VERIFY: every connection has *some* non-empty list of decoded value addresses
        for conn in self.connections:
            if not hasattr(conn, "decoded_value_addrs") or len(conn.decoded_value_addrs) == 0:
                raise NotFeasibleError("connection " + conn.label + " was not assigned decoded value addresses")

        # now we can start calculating encoders.
        # for each population in each cluster, group its incoming connections by filter time constant.
        # assign each group to the encoder on that population unit with the closest time constant.

        log.info("Calculating encoders for 1-D populations")
        self.cluster_encoders_1d = []
        for N in range(len(self.population_clusters_1d)):
            cluster = self.population_clusters_1d[N]
            filters = self.cluster_filters_1d[N] 
            # len(filters) == 2
            filter0 = filters[0]
            filter1 = filters[1]
            # each element of these arrays is an array (one per population) of tuples of the form
            # (DV address, weight)
            filter0_encoders = []
            filter1_encoders = []
            for population in cluster:
                population_encoders_0 = []
                population_encoders_1 = []
                # generate encoders
                for conn in population.inputs:
                    connection_encoders = self.generate_connection_encoders(conn)
                    # len(connection_encoders) == 1
                    if abs(conn.filter - filter0) <= abs(conn.filter - filter1):
                        for encoder in connection_encoders[0]:
                            population_encoders_0.append(encoder)
                    else:
                        for encoder in connection_encoders[0]:
                            population_encoders_1.append(encoder)

                filter0_encoders.append(population_encoders_0)
                filter1_encoders.append(population_encoders_1)
                log.debug("population " + population.label + " encoders by filter:")
                log.debug("encoder 0 (filter=" + str(filter0) + "):")
                log.debug(population_encoders_0)
                log.debug("encoder 1 (filter=" + str(filter1) + "):")
                log.debug(population_encoders_1)
                
            self.cluster_encoders_1d.append( [filter0_encoders, filter1_encoders] )
        
        # calculate encoders for 2-D populations
        log.info("Calculating encoders for 2-D populations")
        self.cluster_encoders_2d = []
        for N in range(len(self.population_clusters_2d)):
            cluster = self.population_clusters_2d[N]
            filters = self.cluster_filters_2d[N]
            # len(filters) == 2
            filter0 = filters[0]
            filter1 = filters[1]
            # Each element of these arrays is an array (one per population) of tuples of the form
            # (DV address, weight).
            # However, for 2D populations, there are TWO encoders associated with each filter
            # (one in each dimension).
            filter0_encoders = [] # X filter 0
            filter1_encoders = [] # X filter 1
            filter2_encoders = [] # Y filter 0
            filter3_encoders = [] # Y filter 1
            for population in cluster:
                population_encoders_0 = []
                population_encoders_1 = []
                population_encoders_2 = []
                population_encoders_3 = []
                # generate encoders
                for conn in population.inputs:
                    connection_encoders = self.generate_connection_encoders(conn)
                    # len(connection_encoders) == 2
                    if abs(conn.filter - filter0) <= abs(conn.filter - filter1):
                        for encoder in connection_encoders[0]:
                            population_encoders_0.append(encoder)
                        for encoder in connection_encoders[1]:
                            population_encoders_2.append(encoder)
                    else:
                        for encoder in connection_encoders[0]:
                            population_encoders_1.append(encoder)
                        for encoder in connection_encoders[1]:
                            population_encoders_3.append(encoder)
                filter0_encoders.append(population_encoders_0)
                filter1_encoders.append(population_encoders_1)
                filter2_encoders.append(population_encoders_2)
                filter3_encoders.append(population_encoders_3)
                log.debug("population " + population.label + " encoders by filter:")
                log.debug("encoder 0 (filter=" + str(filter0) + "):")
                log.debug(population_encoders_0)
                log.debug("encoder 1 (filter=" + str(filter1) + "):")
                log.debug(population_encoders_1)
                log.debug("encoder 2 (filter=" + str(filter0) + "):")
                log.debug(population_encoders_2)
                log.debug("encoder 3 (filter=" + str(filter1) + "):")
                log.debug(population_encoders_3)

            self.cluster_encoders_2d.append( [filter0_encoders, filter1_encoders,
                                              filter2_encoders, filter3_encoders] )
        
        # so, at this point, each element of self.cluster_encoders_(1|2)d
        # is a list E that has two elements for a 1-D population unit and four elements for a 2-D population unit.
        # these lists (E_0, E_1, E_2, E_3) contain all of the actual encoders for each population.
        # so, up to this point, self.cluster_encoders_1d[3][1][4] is a list of encoders
        # for population #4, using encoder #1, running on (1-D) population unit #3.
        # the key is this: self.cluster_encoders_1d[:][:][0] gets us
        # all encoders over all population units for the #0 population on each population unit,
        
        # we want to list the encoders in board address order; 
        # if an encoder has no instructions for a given population,
        # use a blank list. the outer loop therefore should iterate over 
        # population units, and the inner loop over encoders

        # FIXME feasibility check: there can't be more instructions in one encoder than memory to hold them
        log.info("starting optimization of encoder schedules")
        self.encoder_schedules = []
        for i in range(1024):
            # for each timeslice (0-1023), we schedule 128 (max) population units 
            # and 4 (max) encoders per population unit
            schedule = [ [] for i in range(128 * 4) ]
            emptySchedule = True
            # first count 1-D populations, which start at 0 and go up
            for cluster in range(len(self.cluster_encoders_1d)):
                for encoder in range(4):
                    # if this cluster doesn't use this encoder, or this cluster 
                    # doesn't have an i-th population, don't schedule anything here
                    if ( len(self.cluster_encoders_1d[cluster]) < encoder+1 
                         or len(self.cluster_encoders_1d[cluster][encoder]) < i+1 ):
                        schedule[cluster * 4 + encoder] = []
                    else:
                        schedule[cluster * 4 + encoder] = self.cluster_encoders_1d[cluster][encoder][i]
                        if len(self.cluster_encoders_1d[cluster][encoder][i]) > 0:
                            emptySchedule = False
            # now count 2-D populations, which start at 95 and go down
            for cluster in range(len(self.cluster_encoders_2d)):
                for encoder in range(4):
                    # if this cluster doesn't use this encoder, or this cluster
                    # doesn't have an i-th population, don't schedule anything
                    if ( len(self.cluster_encoders_2d[cluster]) < encoder+1
                         or len(self.cluster_encoders_2d[cluster][encoder]) < i+1 ):
                        schedule[(95 - cluster) * 4 + encoder] = []
                    else:
                        schedule[(95 - cluster) * 4 + encoder] = self.cluster_encoders_2d[cluster][encoder][i]
                        if len(self.cluster_encoders_2d[cluster][encoder][i]) > 0:
                            emptySchedule = False
            # perform scheduling and add the result to self.encoder_schedules
            if emptySchedule:
#               log.debug("ignoring empty encoder schedule in timeslice #" + str(i))
                self.encoder_schedules.append( [] ) # FIXME check that this is okay
            else:
                log.debug("optimizing over timeslice #" + str(i) + "...")
                scheduler = EncoderScheduler(schedule, GeneticOptimizer)
                self.encoder_schedules.append( scheduler() )

        # At this point we should have almost everything we need to write out a loadfile.

        # FIXME this sequence of operations should probably go into a subroutine
        # in case there are multiple formats that can be written (possibly depending on control interface type)
        timestamp = datetime.datetime.now().isoformat(sep='_')
        self.filename = self.model.label + '-' + timestamp + '.nengo-rt'
        log.info("writing loadfile " + self.filename + " for target")
        loadfile = open("./" + self.filename, 'w')
        # FIXME we assume single-board targets again 

        # 0x0: Decoded Value buffers
        # We can do this one by looping over all connections, which have all been assigned some
        # list of decoded_value_addrs, and taking either their initial_value or a vector of 0s
        # to be the starting values for the DV buffers.
        # program at address [000 00 HHHHHHHH LLLLLLLLLLL]
        # where H is the high (<<11) part of the DV address and L is the low (&2^11-1) part of the DV address;
        # the data is simply the 12-bit float2sfixed() representation of each initial value
        log.info("writing decoded value buffer initial values...")
        for conn in self.connections:
            pre = conn.pre
            if pre.dv_addrs_done:
                continue            
            if hasattr(pre, "initial_value"):
                # write the corresponding initial value for each address
                for Q in zip(conn.decoded_value_addrs, pre.initial_value):
                    addr = Q[0]
                    H = addr>>11
                    L = addr & (2**11 - 1)
                    Hstr = pad(bin(H)[2:], '0', 8)
                    Lstr = pad(bin(L)[2:], '0', 11)
                    addrStr = "000" + "00" + Hstr + Lstr
                    data = Q[1]
                    dataStr = pad(float2sfixed(data), '0', 40)
                    print(addrStr + ' ' + dataStr, file=loadfile)
            else:
                # write zero for each address
                for addr in conn.decoded_value_addrs:
                    H = addr>>11
                    L = addr & (2**11 - 1)
                    Hstr = pad(bin(H)[2:], '0', 8)
                    Lstr = pad(bin(L)[2:], '0', 11)
                    addrStr = "000" + "00" + Hstr + Lstr
                    data = "0"*40
                    print(addrStr + ' ' + data, file=loadfile)
            pre.dv_addrs_done = True
        
        # 0x1: Encoder instruction buffers
        # Program 40-bit instructions at "001 000000000000 NNNNNNN EE"
        # where N is the population unit index and E is the encoder index on that population unit
        # Instruction format is as follows:
        # |L|TTTTTTT|P|AAAAAAAAAAAAAAAAAAA|WWWWWWWWWWWW|
        # where L is the 'last flag', T is the time delay, P is the port number,
        # A is the decoded value address, and W is the weight (sfixed, 1 downto -10)

        # The only really "difficult" one.
        # There are 1024 elements in the list self.encoder_schedules
        # Each one is either the empty list [], which means "no encoders in this timeslice",
        # or a list T of more lists (128*4; PUs then encoders) E. 
        # Each list E corresponds to the schedule for one encoder
        # in this timeslice. Now, E can be empty, in which case we need to write a no-op instruction.
        # Our no-op is "1 0000000 1 1111111111111111111 000000000000" which should almost always be
        # an illegal address. This makes sure the encoder finishes as quickly as possible and does not
        # produce a result that affects the operation of other encoders or the population unit.
        # If E is not empty, then it will contain one or more ScheduledRead objects (scheduler.py).
        # We sort these reads by the .time attribute and then walk across the sorted list to write
        # the instructions. The first n-1 instructions have last flag 0, and the final one has it at 1.
        # The time delay is just .time for the first instruction, and for subsequent instructions,
        # it is two less than the time difference between the current and previous instruction.
        # Then we transcribe the .readInfo (DV address, weight) and .port to make an instruction
        # and output it to the loadfile.

        # By the way, how to program a no-op:
        #         # calculate write address
        #         Nstr = pad(bin(N)[2:], '0', 7)
        #         Estr = pad(bin(E)[2:], '0', 2)
        #         addrStr = "001" + "000000000000" + Nstr + Estr
        #         insnStr = "1000000011111111111111111111000000000000"
        #         print(addrStr + ' ' + insnStr, file=loadfile)

        log.info("writing encoder instruction buffers...")

        for N in range(96):
            # if this population unit isn't used, program a single no-op on all encoders and continue
            if N >= len(self.population_clusters_1d) and N <= 95 - len(self.population_clusters_2d):
                # writing zero instructions to an encoder optimizes it out completely
#               log.debug("not programming encoders on unused population unit " + str(N))
                continue
            # so, this population unit has been assigned at least one population.
            # however, not all encoders are necessarily going to be used;
            # for each encoder, first check if any schedule is non-empty,
            # and if all are empty, program a single no-op
            for E in range(4):
                disableEncoder = True
                # check all schedules for this encoder
                for T in range(1024):
                    schedules = self.encoder_schedules[T]
                    if schedules is []:
                        continue
                    if len(schedules) <= N*4+E:
                        continue 
                    schedule = schedules[N*4+E]
                    if len(schedule) == 0:
                        continue
                    # found a non-empty schedule for this encoder
                    disableEncoder = False
                    break
                if disableEncoder:
#                   log.debug("not programming unused encoder " + str(E) +
#                             " on population unit " + str(N))
                    continue                

                for T in range(1024):
                    schedules = self.encoder_schedules[T]
                    if schedules is []:
                        # no instructions for any encoder in this timeslice;
                        # program a no-op
                        Nstr = pad(bin(N)[2:], '0', 7)
                        Estr = pad(bin(E)[2:], '0', 2)
                        addrStr = "001" + "000000000000" + Nstr + Estr
                        insnStr = "1000000011111111111111111111000000000000"
                        print(addrStr + ' ' + insnStr, file=loadfile)
                    else:
                        # FIXME skip programming encoders 3 and 4 on 1D population units
                        # calculate write address
                        Nstr = pad(bin(N)[2:], '0', 7)
                        Estr = pad(bin(E)[2:], '0', 2)
                        addrStr = "001" + "000000000000" + Nstr + Estr
                        if len(schedules) > N*4+E:
                            schedule = schedules[N * 4 + E]
                        else:
                            schedule = []
                        if len(schedule) == 0:                            
                            # program a no-op on this encoder
                            insnStr = "1000000011111111111111111111000000000000"
                            print(addrStr + ' ' + insnStr, file=loadfile)
                        else:
                            # sort the schedule by .time
                            ordered_schedule = sorted(schedule, key=lambda op: op.time)
                            for op_idx in range(len(ordered_schedule)):
                                op = ordered_schedule[op_idx]
                                # set (L)ast flag bit
                                if op_idx == len(ordered_schedule)-1:
                                    Lstr = '1'
                                else:
                                    Lstr = '0'
                                # calculate relative offset only if this is not the first instruction
                                if op_idx == 0:
                                    T = op.time
                                else:
                                    T = op.time - ordered_schedule[op_idx-1].time - 2
                                # FIXME if T > 127:
                                Tstr = pad(bin(T)[2:], '0', 7)
                                if op.port == 1:
                                    Pstr = '1'
                                else:
                                    Pstr = '0'
                                Astr = pad(bin(op.readInfo[0])[2:], '0', 19)
                                Wstr = float2sfixed(op.readInfo[1])
                                # sanity check
                                if len(Wstr) != 12:
                                    raise NotFeasibleError("sanity check failed: incorrect conversion of float " 
                                                               + str(op.readInfo[1]) + " to sfixed '" + Wstr + "', wrong length " + str(len(Wstr)))
                                insnStr = Lstr + Tstr + Pstr + Astr + Wstr
                                print(addrStr + ' ' + insnStr, file=loadfile)

        # 0x2: Principal Component filter characteristics
        # call filter_coefs(pstc, dt) with the appropriate PSTC for each encoder;
        # get this list from self.cluster_filters_1d which will be a list of length 2
        # or self.cluster_filters_2d which will be a list of length 4
        # the tuple returned is (A, B) and since we need C and D also, we set C=A and D=B for the filter
        # program at address [010 00000000 00 NNNNNNNFFXX]
        # where N is the population unit index (0-127)
        # F is the filter index
        # and X is the coefficient offset (A=0, B=1, C=2, D=3)
        log.info("writing filter coefficients...")
        for N in range(self.target.total_population_1d_count):
            if N < len(self.population_clusters_1d):
                for F in range(2):
                    pstc = self.cluster_filters_1d[N][F]
                    # fix for filters whose time constant is 0
                    if pstc == 0.0:
#                        log.debug("zeroing unused filter " + str(F) + " on population unit " + str(N))
                        A = 0.0
                        B = 0.0
                    else:
                        (A, B) = self.filter_coefs(pstc, self.model.dt)
                    C = A
                    D = B
                    Nstr = pad(bin(N)[2:], '0', 7)
                    Fstr = pad(bin(F)[2:], '0', 2)
                    Astr = pad(float2sfixed(A), '0', 40)
                    Bstr = pad(float2sfixed(B), '0', 40)
                    Cstr = pad(float2sfixed(C), '0', 40)
                    Dstr = pad(float2sfixed(D), '0', 40)
                    addr = "010" + "00000000" + "00" + Nstr + Fstr # + CC
                    print(addr + "00" + ' ' + Astr, file=loadfile)
                    print(addr + "01" + ' ' + Bstr, file=loadfile)
                    print(addr + "10" + ' ' + Cstr, file=loadfile)
                    print(addr + "11" + ' ' + Dstr, file=loadfile)
            else:
                # program bogus filters for unused population unit
#                log.debug("zeroing filters on unused 1D population unit #" + str(N))
                # FIXME is this necessary?
                for F in range(2):
                    Nstr = pad(bin(N)[2:], '0', 7)
                    Fstr = pad(bin(F)[2:], '0', 2)
                    data = pad('0', '0', 40)
                    addr = "010" + "00000000" + "00" + Nstr + Fstr # + CC
                    print(addr + "00" + ' ' + data, file=loadfile)
                    print(addr + "01" + ' ' + data, file=loadfile)
                    print(addr + "10" + ' ' + data, file=loadfile)
                    print(addr + "11" + ' ' + data, file=loadfile)

        for cN in range(self.target.total_population_2d_count):
            N = 95 - cN
            if cN < len(self.population_clusters_2d):
                for F in range(4):
                    pstc = self.cluster_filters_2d[cN][F%2]
                    if pstc == 0.0:
#                        log.debug("zeroing unused filter " + str(F) + " on population unit " + str(N))
                        A = 0.0
                        B = 0.0
                    else:
                        (A, B) = self.filter_coefs(pstc, self.model.dt)
                    C = A
                    D = B
                    Nstr = pad(bin(N)[2:], '0', 7)
                    Fstr = pad(bin(F)[2:], '0', 2)
                    Astr = pad(float2sfixed(A), '0', 40)
                    Bstr = pad(float2sfixed(B), '0', 40)
                    Cstr = pad(float2sfixed(C), '0', 40)
                    Dstr = pad(float2sfixed(D), '0', 40)
                    addr = "010" + "00000000" + "00" + Nstr + Fstr # + CC
                    print(addr + "00" + ' ' + Astr, file=loadfile)
                    print(addr + "01" + ' ' + Bstr, file=loadfile)
                    print(addr + "10" + ' ' + Cstr, file=loadfile)
                    print(addr + "11" + ' ' + Dstr, file=loadfile)
            else:
                # program bogus filters for unused population unit
#                log.debug("zeroing filters on unused 2D population unit #" + str(N))
                # FIXME is this necessary?
                for F in range(4):
                    Nstr = pad(bin(N)[2:], '0', 7)
                    Fstr = pad(bin(F)[2:], '0', 2)
                    data = pad('0', '0', 40)
                    addr = "010" + "00000000" + "00" + Nstr + Fstr # + CC
                    print(addr + "00" + ' ' + data, file=loadfile)
                    print(addr + "01" + ' ' + data, file=loadfile)
                    print(addr + "10" + ' ' + data, file=loadfile)
                    print(addr + "11" + ' ' + data, file=loadfile)

        # 0x3: Principal Component LFSRs
        # this is an easy one. we have 4 LFSRs for each population unit,
        # so loop over their addresses and initialize them all with random 32-bit values
        # program at address [011 00000 0000000N NNNNNNLL]
        # where N is the population unit index (0-127)
        # and L is the LFSR index (0-3)
        log.info("writing LFSR seeds...")
        for N in range(self.target.total_population_1d_count):
            for L in range(4):
                Nstr = pad(bin(N)[2:], '0', 7)
                Lstr = pad(bin(L)[2:], '0', 2)
                addr = "011" + "00000" + "0000000" + Nstr + Lstr
                seed = np.random.random_integers(0, 2**32 - 1)
                seedstr = pad(bin(seed)[2:], '0', 40)
                print(addr + ' ' + seedstr, file=loadfile)
        for N in range(self.target.total_population_2d_count):
            for L in range(4):
                Nstr = pad(bin(95-N)[2:], '0', 7)
                Lstr = pad(bin(L)[2:], '0', 2)
                addr = "011" + "00000" + "0000000" + Nstr + Lstr
                seed = np.random.random_integers(0, 2**32-1)
                seedstr = pad(bin(seed)[2:], '0', 40)
                print(addr + ' ' + seedstr, file=loadfile)

        # 0x4: Principal Component lookup tables
        # this is an easy one too although there's one place where we have to be careful
        # for 1D population units, there are 7 PCs to program.
        # we write to 1024 addresses, converting corresponding PC values to 12-bit signed fixed point (float2sfixed).
        # the trick is figuring out which address corresponds to which value.
        # since the highest bit is the sign bit, addresses 0-511 correspond to values at 0.0 - 1.999,
        # and addresses 512-1023 correspond to values at -1.999 - -0.001 (effectively).
        # now, since we have the 1D principal components as vectors over that range,
        # starting at the low end, really all we have to do is swap the first 512 and last 512 samples,
        # and write that out in ascending order.
        # for 2D population units, there are 15 PCs to program and addressing is harder.
        # The 10 address bits we want are AAAAAAAAAA = XXXXX YYYYY
        # where each of X and Y go from 0 to 1.875 then from -2 to -0.125
        # (equivalent to running from 00000 to 11111, or 0 to 31)
        # program at address [100 NNNNNNN PPPPAAA AAAAAAA]
        # where N is the population unit index,
        # P is the principal component index (0-6 for 1D, 0-14 for 2D),
        # and A is the LUT address (0-1023)
        log.info("writing principal component lookup tables...")
        for N in range(len(self.population_clusters_1d)):
            for P in range(7):
                pc = self.cluster_principal_components_1d[N][P]
                for A in range(1024):
                    if A >= 512:
                        sample = pc[A - 512] # sample from (x < 0) side of PC
                    else:
                        sample = pc[A + 512] # sample from (x >= 0) side of PC
                    Nstr = pad(bin(N)[2:], '0', 7)
                    Pstr = pad(bin(P)[2:], '0', 4)
                    Astr = pad(bin(A)[2:], '0', 10)
                    addr = "100" + Nstr + Pstr + Astr
                    sampleStr = pad(float2sfixed(sample), '0', 40)
                    print(addr + ' ' + sampleStr, file=loadfile)
        for cN in range(len(self.population_clusters_2d)):
           N = 95 - cN
           for P in range(15):
               pc = self.cluster_principal_components_2d[cN][P]
               for X in range(32):
                   for Y in range(32):
                       if X >= 16:
                           sampleX = X - 16
                       else:
                           sampleX = X + 16
                       if Y >= 16:
                           sampleY = Y - 16
                       else:
                           sampleY = Y + 16
                       sample = pc[(sampleX<<5) + sampleY]
                       Nstr = pad(bin(N)[2:], '0', 7)
                       Pstr = pad(bin(P)[2:], '0', 4)
                       Xstr = pad(bin(X)[2:], '0', 5)
                       Ystr = pad(bin(Y)[2:], '0', 5)
                       Astr = Xstr + Ystr
                       addr = "100" + Nstr + Pstr + Astr
                       sampleStr = pad(float2sfixed(sample), '0', 40)
                       print(addr + ' ' + sampleStr, file=loadfile)

        # 0x5: Decoder circular buffers
        # loop over clusters, then populations, then outgoing connections.
        # each connection should have a ._decoders of length either 7 (1D) or 15 (2D).
        # for the last one, corresponding to the artificial noise, send "000000010100" for now
        # FIXME better choice of noise gain term
        # If there are fewer than 1024 populations in any given cluster, 
        # or if for any reason a population has fewer than the requisite number of decoders,
        # program the remaining decoders to be all zeroes.
        # program at address [101 00000000 NNNNNNNVV DDDD]
        # where N is the cluster index (0-127), V is the index of the decoded value (0-3),
        # and D is the index of the decoder (0-15).
        # Note that there is no field for population; this is because the target is a circular buffer.
        # FIXME on population units that don't use all four decoders, program a single zero
        # on each of the unused decoders
        log.info("writing decoder circular buffers...")
        for N in range(len(self.population_clusters_1d)):
            Nstr = pad(bin(N)[2:], '0', 7)
            cluster = self.population_clusters_1d[N]
            pop_idx = 0
            for population in cluster:
                decoder_idx = 0
                for conn in population.outputs:
                    for i in range(conn.dimensions):
                        Vstr = pad(bin(decoder_idx)[2:], '0', 2)
                        # new decoders: [[0], [1], [2], [3],...,[7]]
                        decoders = conn._decoders
                        for D in range(7):
                            Dstr = pad(bin(D)[2:], '0', 4)                            
                            decoder = decoders[0, D]
                            decoderStr = pad(float2sfixed(decoder), '0', 40)
                            addrStr = "101" + "00000000" + Nstr + Vstr + Dstr
                            print(addrStr + ' ' + decoderStr, file=loadfile)
                        # write decoder 7 = "000000010100"
                        print("101" + "00000000" + Nstr + Vstr + "0111" + ' ' + 
                              pad("000000010100", '0', 40), file=loadfile)
                        decoder_idx += 1
                while decoder_idx < 4:
                    Vstr = pad(bin(decoder_idx)[2:], '0', 2)
                    # program a fake decoded value with all-zeroes decoders
                    for D in range(8):
                        Dstr = pad(bin(D)[2:], '0', 4)
                        decoderStr = pad('0', '0', 40)
                        addrStr = "101" + "00000000" + Nstr + Vstr + Dstr
                        print(addrStr + ' ' + decoderStr, file=loadfile)
                    decoder_idx += 1
                pop_idx += 1
            while pop_idx < 1024:
                # program a fake population with all-zeroes decoders
                for decoder_idx in range(4):
                    Vstr = pad(bin(decoder_idx)[2:], '0', 2)
                    for D in range(8):
                        Dstr = pad(bin(D)[2:], '0', 4)
                        decoderStr = pad('0', '0', 40)
                        addrStr = "101" + "00000000" + Nstr + Vstr + Dstr
                        print(addrStr + ' ' + decoderStr, file=loadfile)
                pop_idx += 1

        for cN in range(len(self.population_clusters_2d)):
            N = 95 - cN
            Nstr = pad(bin(N)[2:], '0', 7)
            cluster = self.population_clusters_2d[cN]
            pop_idx = 0
            for population in cluster:
                decoder_idx = 0
                for conn in population.outputs:
                    for i in range(conn.dimensions):
                        Vstr = pad(bin(decoder_idx)[2:], '0', 2)
                        # new decoders: [[0], [1], [2], [3],...,[15]]
                        decoders = conn._decoders
                        for D in range(15):
                            Dstr = pad(bin(D)[2:], '0', 4)                            
                            decoder = decoders[(conn.dimensions-1) - i, D] # need to investigate WHY these are backwards, and whether that's true for e.g. a 3D output
                            decoderStr = pad(float2sfixed(decoder), '0', 40)
                            addrStr = "101" + "00000000" + Nstr + Vstr + Dstr
                            print(addrStr + ' ' + decoderStr, file=loadfile)
                        # write decoder 15 = "000000010100"
                        print("101" + "00000000" + Nstr + Vstr + "1111" + ' ' + 
                              pad("000000010100", '0', 40), file=loadfile)
                        decoder_idx += 1
                while decoder_idx < 4:
                    Vstr = pad(bin(decoder_idx)[2:], '0', 2)
                    # program a fake decoded value with all-zeroes decoders
                    for D in range(16):
                        Dstr = pad(bin(D)[2:], '0', 4)
                        decoderStr = pad('0', '0', 40)
                        addrStr = "101" + "00000000" + Nstr + Vstr + Dstr
                        print(addrStr + ' ' + decoderStr, file=loadfile)
                    decoder_idx += 1
                pop_idx += 1
            while pop_idx < 1024:
                # program a fake population with all-zeroes decoders
                for decoder_idx in range(4):
                    Vstr = pad(bin(decoder_idx)[2:], '0', 2)
                    for D in range(16):
                        Dstr = pad(bin(D)[2:], '0', 4)
                        decoderStr = pad('0', '0', 40)
                        addrStr = "101" + "00000000" + Nstr + Vstr + Dstr
                        print(addrStr + ' ' + decoderStr, file=loadfile)
                pop_idx += 1
        
        # 0x6: Output channel instruction buffers
        # program at address "110 00 00000000 00000000 CCCCCCC"
        # where C is the output channel index
        # FIXME lots of work to be done here, wrt. which decoded values go to which output channels.
        # For now we assume that all probed signals go to output channel 0, and nothing else is output.
        # When there are multiple output channels, scheduling will become necessary
        # (in fact, scheduling might become more difficult because we can split reads across
        # multiple instructions to free up a bank)
        # The instruction word is this:
        # |L|P|DDDDDDDD|AAAAAAAAAAA|NNNNNNNNNNN|TTTT|
        # where L is the "last instruction" flag, P is the port,
        # D is the DV bank address, A is the first address to read inside that DV bank,
        # N is one less than the number of words to read (starting at A),
        # T is one less than the number of cycles to delay (for concurrency reasons,
        # similar to what is done with the encoders)
        # The maximum number of instructions we can issue for each output channel is 512.
        # We may have to schedule some "unnecessary" reads if we run out of instructions,
        # i.e. if we care about values at addresses 0, 2, 4, 6 in the same bank,
        # we may just issue the instruction "read addresses 0-6 in bank N" and do it in 
        # one instruction instead of four.
        # We'll also need to write down where to expect values we care about in the incoming frames.
        log.info("writing output channel instruction buffers...")
        # all probes go to channel 0 automatically
        addrStr = "110000000000000000000000"
        probed_addresses = set()
        # keep a dictionary mapping addresses to the probes that care about them
        self.probe_address_table = {}
        for probe in self.probes:
            # keep a dictionary mapping addresses to input dimension for each probe
            probe.dimension_address_table = {}
            dimension = 0
            for conn in probe.inputs:
                for addr in conn.decoded_value_addrs:
                    probed_addresses.add(addr)
                    if addr in self.probe_address_table:
                        self.probe_address_table[addr].append(probe)
                    else:
                        self.probe_address_table[addr] = [probe]
                    probe.dimension_address_table[addr] = dimension
                    dimension += 1

        # now do something similar for nodes which have inputs
        for node in self.nodes:
            if len(node.inputs) == 0 or node.optimized_out:
                continue
            # node performs I/O
            node.dimension_address_table = {}
            dimension = 0
            for conn in node.inputs:
                for addr in conn.decoded_value_addrs:
                    probed_addresses.add(addr)
                    if addr in self.probe_address_table:
                        self.probe_address_table[addr].append(node)
                    else:
                        self.probe_address_table[addr] = [node]
                    node.dimension_address_table[addr] = dimension
                    dimension += 1

        log.debug("Probes at addresses " + str(probed_addresses))
        if len(probed_addresses) <= 512:
            probed_addresses = list(probed_addresses)
            # easy case: one-to-one correspondence between addresses and instructions
            for i in range(len(probed_addresses)):
                addr = probed_addresses[i]
                if i == len(probed_addresses) - 1:
                    Lstr = '1'
                else:
                    Lstr = '0'
                Pstr = '0'
                DAstr = pad(bin(addr)[2:], '0', 19)
                Nstr = "00000000000"
                Tstr = "0000"
                insnStr = pad(Lstr + Pstr + DAstr + Nstr + Tstr, '0', 40)
                print(addrStr + ' ' + insnStr, file=loadfile)
                self.probe_sequence.append(addr)
        else:
            # hard case: we have to combine some reads into single instructions
            raise NotImplementedError("more than 512 probed addresses, implement instruction selection for this case")

        # 0x7: not used
        log.info("finished writing loadfile")
        loadfile.close()
        return self.model

    # The encoder performs [DV address * transform weight] * connection inverse scale factor
    # (where the inverse scale factor is 1.0 if it does not exist;
    #  the purpose of that is to recover the correct decoded values when the decoders
    #  would have been out of range)
    def generate_connection_encoders(self, conn):
        log.debug("Generating connection encoders for " + conn.label)
        connection_encoders = []
        for i in range(conn.post.dimensions):
            encoders = []
            decoded_value_addresses = conn.decoded_value_addrs
            if conn.transform.ndim == 0:
                # if conn.transform is a 0-D array, it is a scalar,
                # so use that scalar for the appropriate dimension and 0s elsewhere
                transform_vector = []
                for j in range(len(decoded_value_addresses)):
                    if i == j: # dimensions match
                        transform_vector.append(conn.transform[()])
                    else:
                        transform_vector.append(0.0)
            else:
                # transform[i, :] is the transform vector for the i-th dimension
                transform_vector = conn.transform[i, :]
            # make sure these are the same size...
            if len(decoded_value_addresses) != len(transform_vector):
                raise NotFeasibleError("connection " + conn.label + " associated with " + 
                                       str(len(decoded_value_addresses)) + " decoded values and " +
                                       str(len(transform_vector)) + " transforms; these must be equal")
            for j in range(len(decoded_value_addresses)):
                dv_addr = decoded_value_addresses[j]
                weight = transform_vector[j]
                if hasattr(conn, "decoder_inverse_scale_factor"):
                    weight *= conn.decoder_inverse_scale_factor
                # otherwise, don't touch the weight;
                # this attribute won't be set for, e.g., connections leaving nodes
                # because these connections don't have decoders
                if weight != 0.0:
                    encoders.append( (dv_addr, weight) )
            connection_encoders.append(encoders)
        return connection_encoders

    def population_distance(self, P1, P2):
        return (self.weightPCDistance * principal_component_distance(P1, P2) 
                + self.weightFilterDistance * filter_distance(P1, P2))

    def build_ensemble(self, ens):
        log.debug("Building ensemble " + str(ens.label))
        # first half is similar to the default builder's operation
        if ens.dimensions <= 0:
            raise ValueError(
                "Number of dimensions (%d) must be positive" % ens.dimensions)
        if ens.dimensions >= 3:
            raise ValueError(
                "Ensemble " + ens.label + " has more than 2 dimensions; not supported by this backend")
        
        if ens.seed is None:
            ens.seed = self.model._get_new_seed()
        rng = np.random.RandomState(ens.seed)
            
#        if ens.eval_points is None:
#            ens.eval_points = nengo.decoders.sample_hypersphere(
#                ens.dimensions, ens.EVAL_POINTS, rng) * ens.radius
#        else:
#            ens.eval_points = np.array(ens.eval_points, dtype=np.float64)
#            if ens.eval_points.ndim == 1:
#                ens.eval_points.shape = (-1, 1)

#        if ens.neurons.gain is None or ens.neurons.bias is None:
            # if max_rates and intercepts are distributions,
            # turn them into fixed samples
#            if hasattr(ens.max_rates, 'sample'):
#                ens.max_rates = ens.max_rates.sample(
#                    ens.neurons.n_neurons, rng=rng)
#            if hasattr(ens.intercepts, 'sample'):
#                ens.intercepts = ens.intercepts.sample(
#                    ens.neurons.n_neurons, rng=rng)
#            ens.neurons.set_gain_bias(ens.max_rates, ens.intercepts)

        # build ens.neurons
        if ens.neurons.n_neurons <= 0:
            raise ValueError(
                'Number of neurons (%d) must be non-negative' % ens.neurons.n_neurons)

        # Set up encoders
#        if ens.encoders is None:
#            ens.encoders = ens.neurons.default_encoders(ens.dimensions, rng)
#        else:
#            ens.encoders = np.array(ens.encoders, dtype=np.float64)
#            enc_shape = (ens.neurons.n_neurons, ens.dimensions)
#            if ens.encoders.shape != enc_shape:
#                raise ShapeMismatch(
#                    "Encoder shape is %s. Should be (n_neurons, dimensions);"
#                    "in this case %s." % (ens.encoders.shape, enc_shape))

#            norm = np.sum(ens.encoders * ens.encoders, axis=1)[:, np.newaxis]
#            ens.encoders /= np.sqrt(norm)

#        if isinstance(ens.neurons, nengo.Direct):
#            ens._scaled_encoders = ens.encoders
#        else:
#            ens._scaled_encoders = ens.encoders * (
#                ens.neurons.gain / ens.radius)[:, np.newaxis]
        
        # the second half calculates hardware-specific things
        # outside the usual build process, which include
        # principal components and (unscaled) decoders
        log.debug("Calculating principal components")
        # we need 1024 eval points because that's how many samples we can store in hardware
        if ens.dimensions == 1:
            eval_points = np.array([np.linspace(-1.0, 1.0, 
                                                 num=1024)]).transpose()
        elif ens.dimensions == 2: 
            # FIXME number of samples
            # FIXME do we want (-2,2) here?
            eval_points = samplepoints2d(-1.0, 1.0, 0.1, -1.0, 1.0, 0.1)

        activities = ens.activities(eval_points)
        u, s, v = np.linalg.svd(activities.transpose())
        if ens.dimensions == 1:
            npc = 7
        elif ens.dimensions == 2:
            npc = 15

        # ens.principal_components = v[0:npc, :].transpose()
        S = np.zeros((u.shape[0], v.shape[0]), dtype=complex)
        S[:s.shape[0], :s.shape[0]] = np.diag(s)
        # extend principal components to full representable range
        if ens.dimensions == 1:
            eval_points_extended = np.array([np.linspace(-2.0, self.max_12bit_value, 
                                                          num=len(eval_points))]).transpose()
        elif ens.dimensions == 2:
            eval_points_extended = samplepoints2d(-2.0, self.max_12bit_value, 0.125,
                                                   -2.0, self.max_12bit_value, 0.125)

        activities_extended = ens.activities(eval_points_extended)
        usi = np.linalg.pinv(np.dot(u,S))
        ens.principal_components = np.real(np.dot(usi[0:npc, :], activities_extended.transpose()))
        # visualize
        # for n in range(npc):
        #      pc = ens.principal_components[n]
        #      pc2D = np.reshape(pc, (32, 32))
        #      x2D = np.reshape(eval_points_extended[:,0], (32, 32))
        #      y2D = np.reshape(eval_points_extended[:,1], (32, 32))
        #      fig = plt.figure()
        #      ax = fig.add_subplot(111, projection='3d')
        #      ax.plot_surface(x2D, y2D, pc2D, cmap=plt.get_cmap('jet'))
        #      plt.title('Principal Component ' + str(n))
        # plt.show()            

        # we have to save a few values in order to calculate approximate decoders later on
        ens.pc_u = u
        ens.pc_S = S
        ens.npc = npc

        # set up input and output arrays to be filled with connections
        ens.inputs = []
        ens.outputs = []
        ens.dv_addrs_done = False

    def build_node(self, node):
        # set up input and output arrays to be filled with connections
        node.inputs = []
        node.outputs = []
        node.dv_addrs_done = False
        node.optimized_out = False
        node.dimensions = node._size_in

    def build_probe(self, probe):
        # set up input array to be filled with connections
        probe.inputs = []

    def recompute_decoders(self, population, conn):
        rng = np.random.RandomState(self.model._get_new_seed())
        dt = self.model.dt
        activities = population.activities(conn.eval_points) * dt
        if conn.function is None:
            targets = conn.eval_points
        else:
            targets = np.array(
                [conn.function(ep) for ep in conn.eval_points])
            if targets.ndim < 2:
                targets.shape = targets.shape[0], 1
                    
        conn.base_decoders = conn.decoder_solver(activities, targets, rng) * dt
           
        # to solve for approximate decoders wrt. principal components:
        conn._decoders = np.dot(population.pc_S[0:population.npc, 0:population.npc],
                                np.dot(population.pc_u[:,0:population.npc].transpose(), 
                                       conn.base_decoders)).transpose()

    def build_connection(self, conn):
        log.debug("Building connection " + conn.label)
        dt = self.model.dt
        # find out what we're connecting from
        if isinstance(conn.pre, nengo.Ensemble): # FIXME direct mode?        
            self.recompute_decoders(conn.pre, conn)
            log.debug("Decoders for " + conn.label + ": " + os.linesep + str(conn._decoders))
            if conn.filter is not None and conn.filter > dt:
                conn.o_coef, conn.n_coef = self.filter_coefs(pstc=conn.filter, dt=dt)
            else:
                conn.filter = 0.0 # FIXME check this
                conn.o_coef, conn.n_coef = 0.0, 1.0 # FIXME check these
        elif isinstance(conn.pre, nengo.Node):
            pass # this is not a decoded connection
        else:
            # FIXME
            raise NotImplementedError("attempt to connect from unsupported object '" 
                                      + type(conn.pre).__name__ + "'")
        
        conn.transform = np.asarray(conn.transform, dtype=np.float64)
        if isinstance(conn.post, nengo.nonlinearities.Neurons):
            conn.transform *= conn.post.gain[:, np.newaxis]

        # add this connection to input/output lists of post/pre
        conn.pre.outputs.append(conn)
        conn.post.inputs.append(conn)

    @staticmethod
    def filter_coefs(pstc, dt):
        pstc = max(pstc, dt)
        decay = np.exp(-dt / pstc)
        return decay, (1.0 - decay)
