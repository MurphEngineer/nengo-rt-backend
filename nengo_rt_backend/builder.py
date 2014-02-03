import numpy as np
import scipy
import scipy.cluster
import math

#from .target import Target, Board, Output, Input, Control, IO
from .target import Target

import os
import logging
import xml.etree.ElementTree as ET

import nengo
import nengo.decoders
import nengo.nonlinearities

# for visualization
import matplotlib.pyplot as plt

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

class ShapeMismatch(ValueError):
    pass

class NotFeasible(ValueError):
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

        self.model = model
        self.model.dt = dt
        if self.model.seed is None:
            self.model.seed = np.random.randint(np.iinfo(np.int32).max)

        self.populations = []
        self.connections = []
        self.nodes = []
        self.probes = []
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

            # use sets of filters instead of lists of filters in order to allow hashing,
            # since these become the keys to a dictionary in a later step
            population.filters = frozenset(filters)

        if self.targetFile is None:
            raise ValueError("targetFile not specified")
        else:
            # parse target file
            self.target = Target(self.targetFile)
            log.info("Targeting device with " + str(self.target.total_population_1d_count) + 
                     " one-dimensional populations"
                     " and " + str(self.target.total_population_2d_count) + 
                     " two-dimensional populations")

        # FIXME: before we can do this, check the number of decoded values from each population
        # and if there are more than four, split the population into copies until there are at most
        # four decoded values coming from each one

        # perform clustering of 1D populations
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
        log.info(str(max(cluster_assignments) - min(cluster_assignments) + 1) + " clusters formed")
        cluster_sizes = np.bincount(cluster_assignments)[1:] # the first element of the original array is always 0
        log.debug("1D population cluster sizes: " + os.linesep + 
                  str(cluster_sizes))

        # assign populations to clusters
        self.population_clusters_1d = [ [] for i in range(max(cluster_assignments)) ]
        self.cluster_principal_components_1d = []
        self.cluster_pc_scale_factor_1d = []

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
            pc_rep = candidate.principal_components
            # scale to +/-max_12bit_value
            pc_max = np.absolute(pc_rep).max(axis=1)
            for i in range(pc_rep.shape[0]):
                scale_factor = self.max_12bit_value / pc_max[i]
                # save this for when we recompute decoders
                self.cluster_pc_scale_factor_1d.append(scale_factor)
                for j in range(pc_rep.shape[1]):
                    pc_rep[i, j] *= scale_factor
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
            while len(filters) < 2:
                filters.add(0.0)
            # if there are two now, we're done
            if len(filters) == 2:
                self.cluster_filters_1d.append(frozenset(filters))
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
            pop_idx = 0
            for population in cluster:
                decoder_idx = 0
                for conn in population.outputs:
                    # (PC * decoder) = (PC * scale factor * 1/scale factor * decoder)
                    # = (PC * scale_factor) * (1/scale factor * decoder);
                    # adjust conn._decoders by 1/scale factor
                    conn._decoders /= scale_factor
                    log.debug("new decoders for " + conn.label + ": " + str(conn.decoders))
                    decoders_max = np.absolute(conn.decoders).max()
                    if decoders_max > self.max_12bit_value:
                        log.warn("decoders for " + conn.label + " out of range (" + str(decoders_max) +
                                 "), rescaling")
                        decoder_scale_factor = self.max_12bit_value / decoders_max
                        conn._decoders *= decoder_scale_factor
                        conn.decoder_inverse_scale_factor = 1.0 / decoder_scale_factor
                    else:
                        # decoders don't need rescaling in encoders
                        conn.decoder_inverse_scale_factor = 1.0
                    # at this point, we can correctly assign an address to this connection
                    conn.decoded_value_addr = (N << 12) + (decoder_idx << 10) + (pop_idx)
                    log.debug("address for cluster " + str(N) + " population " + str(pop_idx) + 
                              " decoder " + str(decoder_idx) + 
                              " is " + str(bin(conn.decoded_value_addr)))
                    decoder_idx += 1
                pop_idx += 1

        # perform clustering of 2D populations

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
            
        if ens.eval_points is None:
            ens.eval_points = nengo.decoders.sample_hypersphere(
                ens.dimensions, ens.EVAL_POINTS, rng) * ens.radius
        else:
            ens.eval_points = np.array(ens.eval_points, dtype=np.float64)
            if ens.eval_points.ndim == 1:
                ens.eval_points.shape = (-1, 1)

        if ens.neurons.gain is None or ens.neurons.bias is None:
            # if max_rates and intercepts are distributions,
            # turn them into fixed samples
            if hasattr(ens.max_rates, 'sample'):
                ens.max_rates = ens.max_rates.sample(
                    ens.neurons.n_neurons, rng=rng)
            if hasattr(ens.intercepts, 'sample'):
                ens.intercepts = ens.intercepts.sample(
                    ens.neurons.n_neurons, rng=rng)
            ens.neurons.set_gain_bias(ens.max_rates, ens.intercepts)

        # build ens.neurons
        if ens.neurons.n_neurons <= 0:
            raise ValueError(
                'Number of neurons (%d) must be non-negative' % ens.neurons.n_neurons)

        # Set up encoders
        if ens.encoders is None:
            ens.encoders = ens.neurons.default_encoders(ens.dimensions, rng)
        else:
            ens.encoders = np.array(ens.encoders, dtype=np.float64)
            enc_shape = (ens.neurons.n_neurons, ens.dimensions)
            if ens.encoders.shape != enc_shape:
                raise ShapeMismatch(
                    "Encoder shape is %s. Should be (n_neurons, dimensions);"
                    "in this case %s." % (ens.encoders.shape, enc_shape))

            norm = np.sum(ens.encoders * ens.encoders, axis=1)[:, np.newaxis]
            ens.encoders /= np.sqrt(norm)

        if isinstance(ens.neurons, nengo.Direct):
            ens._scaled_encoders = ens.encoders
        else:
            ens._scaled_encoders = ens.encoders * (
                ens.neurons.gain / ens.radius)[:, np.newaxis]
        
        # the second half calculates hardware-specific things
        # outside the usual build process, which include
        # principal components and (unscaled) decoders
        log.debug("Calculating principal components")
        # we need 1024 eval points because that's how many samples we can store in hardware
        # FIXME eval_points is slightly different for 2-dimensional populations
        eval_points = np.matrix([np.linspace(-1.0, 1.0, num=1024)]).transpose()
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
        # FIXME this is also different for 2-dimensional populations
        # FIXME use the *actual* representable range as a signed 12-bit fixed point value instead of +/-2.0
        eval_points_extended = np.matrix([np.linspace(-2.0, 2.0, num=len(eval_points))]).transpose()
        activities_extended = ens.activities(eval_points_extended)
        usi = np.linalg.pinv(np.dot(u,S))
        ens.principal_components = np.real(np.dot(usi[0:npc, :], activities_extended.transpose()))
        # we have to save a few values in order to calculate approximate decoders later on
        ens.pc_u = u
        ens.pc_S = S
        ens.npc = npc

        # set up input and output arrays to be filled with connections
        ens.inputs = []
        ens.outputs = []

    def build_node(self, node):
        # set up input and output arrays to be filled with connections
        node.inputs = []
        node.outputs = []

    def build_probe(self, probe):
        # set up input array to be filled with connections
        probe.inputs = []

    def build_connection(self, conn):
        log.debug("Building connection " + conn.label)
        rng = np.random.RandomState(self.model._get_new_seed())
        dt = self.model.dt
        # find out what we're connecting from
        if isinstance(conn.pre, nengo.Ensemble): # FIXME direct mode?
            if conn._decoders is None:
                activities = conn.pre.activities(conn.eval_points) * dt
                if conn.function is None:
                    targets = conn.eval_points
                else:
                    targets = np.array(
                        [conn.function(ep) for ep in conn.eval_points])
                    if targets.ndim < 2:
                        targets.shape = targets.shape[0], 1

                conn.base_decoders = conn.decoder_solver(activities, targets, rng) * dt

                # to solve for approximate decoders wrt. principal components:
                conn._decoders = np.dot(conn.pre.pc_S[0:conn.pre.npc, 0:conn.pre.npc],
                                        np.dot(conn.pre.pc_u[:,0:conn.pre.npc].transpose(), 
                                               conn.base_decoders.transpose())).transpose()
                log.debug("Decoders for " + conn.label + ": " + os.linesep + str(conn.decoders))
            if conn.filter is not None and conn.filter > dt:
                conn.o_coef, conn.n_coef = self.filter_coefs(pstc=conn.filter, dt=dt)
            else:
                conn.filter = 0.0 # FIXME check this
                conn.o_coef, conn.n_coef = 0.0, 1.0 # FIXME check these
        else:
            pass # FIXME
        
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
