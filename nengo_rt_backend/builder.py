import numpy as np

import logging

import nengo
import nengo.decoders
import nengo.nonlinearities

import os

log = logging.getLogger(__name__)

class ShapeMismatch(ValueError):
    pass

class Builder(object):
    def __init__(self):
        pass

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
        # FIXME eval_points is slightly different for 2-dimensional populations
        eval_points = np.matrix([np.linspace(-1.0, 1.0, num=ens.EVAL_POINTS)]).transpose()
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
        eval_points_extended = np.matrix([np.linspace(-2.0, 2.0, num=len(eval_points))]).transpose()
        activities_extended = ens.activities(eval_points_extended)
        usi = np.linalg.pinv(np.dot(u,S))
        ens.principal_components = np.dot(usi[0:npc, :], activities_extended.transpose())
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

                base_decoders = conn.decoder_solver(activities, targets, rng) * dt

                # now solve for approximate decoders wrt. principal components
                conn._decoders = np.dot(conn.pre.pc_S[0:conn.pre.npc, 0:conn.pre.npc],
                                        np.dot(conn.pre.pc_u[:,0:conn.pre.npc].transpose(), 
                                               base_decoders.transpose())).transpose()
                log.debug("Decoders for " + conn.label + ": " + os.linesep + str(conn.decoders))
            if conn.filter is not None and conn.filter > dt:
                conn.o_coef, conn.n_coef = self.filter_coefs(pstc=conn.filter, dt=dt)
            else:
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
