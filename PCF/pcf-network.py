'''
Created on Oct 14, 2019
Predictive Coding Framework (PCF) Network 
This module contains several class definitions that implement
a spiking neural network that represents an arbitrary linear
dynamical system according to the Predictive Coding Framework 
see (https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003258)
@author: Chris Fritz 
@date: 14 OCT 2019
'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate
import queue

class LinearDynamicalSystem():
    '''
    A LinearDynamicalSystem (LDS) object instantiates a specified dynamical 
    system. The user provides a LDS specification via initial arguments. The system begins at 
    time = 0, and is advanced  by calling the update() method. Any field listed below can be 
    directly accessed via dot notation.  
    
    ---------------- Initial Arguments: ----------------------------------------------------------------------------------------
    init_state: (M numpy array) - an initial configuration of the dynamical system
    
    A: state transition matrix (M x M numpy array) - determines how one state maps to the next 
    
    init_input: (L numpy array) - an initial configuration of the input to the dynamical system 
    
    B: input matrix ( M x L numpy array ) - maps the dependence of the state on the input vector 
    
    dt: time step (miliseconds) - the time elapsed in the dynamical system between successive updates - defaults to 0.001 ms
        

    --------------- Fields -------------------------------------------------------------------------------------------------------
    t: (T scalar) the elapsed time of the simulation
    
    A: state_transition matrix (see above)
    
    B: input matrix (see above)
    
    x: (M numpy array ) state vector at current time
    
    u: (L numpy array) input vector current time 
    
    u: (L numpy array) input vector at current time 
    
    '''
    
    def __init__(self, init_state, A, init_input, B, dt = .001):
        '''
        Initialize the Linear Dynamical System. If there are any defaults, ensure they have appropriate
        dimensions. 
        '''
        self.x = init_state  
        assert(self.x.ndim == 1), "Initial State is expected to be a vector, but has %i dimensions" %self.x.ndim
        assert(self.x.shape[0] == A.shape[0]), ("State vector size (%i) does not match"\
                                                " Number of State Transition Matrix Rows (%i)" %(self.x.shape[0], A.shape[0]) )
        self.A = A
            
            
        self.u = init_input
        assert(self.u.ndim == 1), "Initial Input is expected to be a vector, but has %i dimensions" %self.u.ndim
        self._null_input = np.zeros(self.u.shape)

    
        assert(B.shape[0] == A.shape[0]), "Input matrix rows (%i) should match state transition matrix rows (%i)" %(B.shape[0], A.shape[0])
        assert(B.shape[1] == len(self.u)), "Input matrix columns (%i) should match length of provided input vector (%i)" %(B.shape[1], self.u.shape[0])
        self.B = B
            
        self.dt = dt
        self.t = 0

    def get_derivative(self, t, x):
        '''
        Return the derivative of the state (x-dot) at the current time
        '''
        return self.A @ x + self.B @ self.u
    
    def update(self, u = None):
        '''
        Update the LDS to the next timestep using the provided input vector:
        compute next state using RK45 integration (via scipy)
        If input vector is not correct length, then it will be discarded and 
        treated as 0.  
        '''
        
        # if input valid, change input field, otherwise
        # leave as 0
        if (u is not None and len(u) == len(self.u)):
            self.u   = u
        else:
            self.u   = self._null_input
        self.x = scipy.integrate.solve_ivp(self.get_derivative, (self.t, self.t + self.dt), self.x).y[:,-1]
        self.t += self.dt
    
class Decoder():
    ''' 
    The decoder class defines a neural decoder, which is used to read out the state estimate from
    a given neural network. The decoder must first be instantiated, then attached to a network.
    To attach a decoder to a network, instantiate a network with the desired coder as an argument
    and the network will attach itself.
    An error will be thrown if readout is attempted and no network is attached. 
    
    ---------------- Initial Arguments: ----------------------------------------------------------------------------------------class Network(object):
    G : (M x N numpy array) the decoder matrix mapping N neural activities to M state variables
    lambda_d: (scalar) the decay rate of the readout
    '''


    def __init__(self, G, lambda_d = 1):
        ''' Initialize the Decoder, No network is attached'''
        self.lambda_d = lambda_d
        self.G = G
        self._network = None
    
    def attach_network(self, net, ):
        '''
        Attach the decoder to a network. Checks to ensure decoder matrix has  
        N columns (number of neurons in net)
        '''
        assert(net.N == self.G.shape[1]), "Decoder matrix number of columns (%i)"\
        " does not match number of neurons in network (%i) " %(self.G.shape[1], net.N)
        print("network successfully attached to decoder")
        self._network = net
    
    def readout(self):
        ''' Return the estimate of the state variable from the network activity firing rate r '''
        assert(self._network is not None), "Cannot readout from decoder because it is not attached to a network."
        return self.G @ self._network.r

class NoiseSource():
    ''' 
    Creates a Noise generator object with properties specified by user. To get the noise
    contribution at a particular timestep, call draw_noise() method. 
    Currently only implements gaussian white noise. 
    A dedicated class exists so that the model may be updated or changed independently of the neural network.
    
    
    ---------------- Initial Arguments: ----------------------------------------------------------------------------------------class Network(object):
    mu: (scalar) mean of Gaussian distribution 
    
    sigma: (scalar) standard deviation of the Gaussian distribution (sqrt(Variance))
    
    N: (scalar) the number of random samples to return in vector form
    ''' 
    def __init__(self, mu = 0, sigma = 1, N = 1):
        '''Create a noise generator object) Assumes inputs are both scalars. '''
        self._mu = mu
        self._sigma = sigma
        self._N = N
    
    def draw_noise(self):
        ''' Draw a sample from the specified noise distribution, currently gaussian '''

        return np.random.normal(self._mu, self._sigma, self._N)
    
class Network():
    '''
    A network implements a PCF neural network. The network takes a linear dynamical system, a decoder, and
    a noise source, and implements the LSD in a spiking neural network which can be read out by the decoder.  
    
        
    ---------------- Initial Arguments: ----------------------------------------------------------------------------------------
    dec: (Decoder Object) instantiated decoder matrix to be attached to the network
    
    lds: (LinearDynamicalSystem Object) instantiated linear dynamical system that the network will implement
    
    noise_src: (NoiseSource Object) instantiated noise source that the network draws from.
    
    N: (scalar) number of neurons comprising the network
    
    v: (scalar) linear regularization parameter
    
    m: (scalar) quadratic regularization parameter
    
    lambda_v: (scalar) leak voltage specifying membrane potential decay rate
    
    sigma_v: (scalar) noise gain factor that impacts how noise impacts the  change in membrane potential for each neuron

    thresh_v: (N numpy array) threshold voltage in miliVolts specifying the membrane potential above which each of N neurons spike
    
    reset_v: (N numpy array) reset voltage specifying the membrane potential each of N neurons is set to when the neuron spikes
    
    
    --------------- Fields -------------------------------------------------------------------------------------------------------
    N: (scalar) number of neurons in network 
    
    W: (N x N numpy array) connectivity matrix between neurons derived from decoder and LDS
    
    V: (N numpy array) vector of voltages for each neuron
    
    r: (N numpy array) vector of instantaneous firing rates for each neuron
    
    S: (N numpy array) spike raster containing time of last spike for each neuron (initially 0 vector)
    '''
    
    ABS_REFRAC_PERIOD = .001  #Minimum refractory period in S between consecutive spikes of a neuron


    def __init__(self, decoder, lds, noise_src, N,  v, m, lambda_v, sigma_v, thres_v, reset_v):
        ''' 
        Initialize the neural network according to the PCF equations specified in the paper listed at top of document.
        Check to ensure that the decoder matrix agrees in shape with the state vector, then attach to decoder matrix
        '''
        
        self.N = N
        self.v = v
        self.m = m
        self._lambda_v = lambda_v
        self._sigma_v = sigma_v
        
        assert(len(thresh_v) == N), "Threshold voltages supplied should be a vector of %i elements, but contained %i" %(N, len(thresh_v))
        assert(len(reset_v) == N), "Reset voltages supplied should be a vector of %i elements, but contained %i" %(N, len(reset_v))
        self._thresh_v = thres_v
        self._reset_v = reset_v
        
        self._dec = decoder
        assert(decoder.G.shape[0] == lds.x.size), "Could not attach decoder to network:"\
        " the number of decoder matrix rows (%i) does not match the number of LDS state variables (%i)" %(decoder.G.shape[0], lds.x.size)
        self._dec.attach_network(self)

        self._lds = lds
        self._noise_src = noise_src
        
        self._derive_kernels()
                
        self.V = .01*np.ones((N,))
        self.r = np.zeros((N,))
        self.S = [[0 for i in np.arange(1)] for j in np.arange(N)]
            
    
    def _derive_kernels(self):
        '''
        Derive the fast and slow network kernels and 
        use these to compute the spiking thresholds
        '''
        
        ld = self._dec.lambda_d
        nu = self.v
        mu = self.m
        G  = self._dec.G
        A  = self._lds.A
        # solve for neuron scaling parameters
        self._w_fast = G.T @ G + mu * ld**2 * np.eye(self.N)
        self._w_slow = G.T @ (A + ld * np.eye(A.shape[0])) @ G
        
        # calculate thresholds 

        for i in np.arange(self.N):
            self._thresh_v[i] = (
                (nu * ld) + (mu * ld**2) + (G[:,i].T @ G[:,i])
                ) / 2  
        
    
    # helper functions for getting time-dependent connectivity kernel
    def _h_d(self, tau):
        ''' given a time tau, return the decay kernel h_d = exp(-lamba_d * t) '''
        if tau >= 0:
            return np.exp(-self._dec.lambda_d * tau)
        else:
            return 0    
        
    def get_w_kernel(self, tau):
        ''' compute the time-dependent connectivity kernel given time tau'''

        
        
        return slow_term - fast_term
    
    def _r_dot(self):
        '''
        Get the derivative of the firing rate
        '''
        return -self._dec.lambda_d * self.r
        
 
    def _V_dot(self):
        '''
        Get the derivative of the voltage (membrane potential). 
        '''

        # noise makes function -very- slow
        
        return (
            -self._lambda_v * self.V  
            + self._dec.G.T @ (self._lds.B @ self._lds.u) 
            + self._sigma_v * self._noise_src.draw_noise()
            + self._w_slow @ self.r # slow kernel contribution  
            # fast kernel is implemented as instantaneous voltage drop inside _spike method
            )
        
    def _spike(self):
        ''' 
        update spike rasters.
        If the given neurons voltage threshold is reached, it is set to
        spike in the next time frame (i.e. when update is called next), 
        so we add dt. A spike can only occur if after the absolute refractory
        period, defined as a class constant, has elapsed'''
        
        # helper function that returns a list of indices of cells that have spiked
        def get_spiked_indices(self):
            ''' 
            Helper function: retrieve a list of indices of cells that
            have spiked.
            '''
            
            
        spikes = self.V >= self._thresh_v
        curr_time = self._lds.t
        while (sum(spikes)):
            spike_idxs = [i for i in np.arange(N) if spikes[i]]
            i = np.random.choice(spike_idxs)    
            self.S[i].append(curr_time) 
            self.r[i] += self._dec.lambda_d
            self.V += -self._w_fast[:,i]
            spikes = self.V >= self._thresh_v

        
    def update(self, u = None):
        ''' Update the neural network to the next time step '''

        # advance network in time then check for spikes
        self.r += self._lds.dt * self._r_dot()
        self.V += self._lds.dt * self._V_dot()
        self._spike()
        # update underlying lds which defines time for the simulation
        self._lds.update(u) 
     
         
# configure linear dynamical system
A = np.zeros((2,2))
A[0,1] = -1
A[1,0] = 1 
x0 = np.asarray([.2, .4])
u0 = np.asarray([0])
B  = np.zeros((2,1))
B[0] = -1
dt = .001 # time step (S)
lds = LinearDynamicalSystem(x0, A, u0, B, dt)

# Network Parameters
N = 400         # number of neurons
v = 10**-5    # linear regularization parameter 
m = 10**-6     # quadratic regularizatino parameter
lambda_v = .020   # leak voltage rate (Hz)
sigma_v = .1     # voltage noise gain (Hz)
thresh_v = -30 * np.ones((N,)) *10**-3  # threshold potential of N neurons (V)
reset_v  = -80 * np.ones((N,)) *10**-3  # reset potential of N neurons (V)

# configure decoder
G = np.ones((2, N)) * .1
G[0,:] = G[0,:] = -.1
G[:,int(N/2):] = - G[:,int(N/2):]
G += .001*(np.random.random(G.shape)-.5)
lambda_d = 10 
dec = Decoder(G, lambda_d)

# initialize noise source
noise_src = NoiseSource(N = N)
    

# initialize network
net = Network(dec, lds, noise_src, N, v, m, lambda_v, sigma_v, thresh_v, reset_v)

ts = np.arange(1000)*dt
readout = np.zeros((A.shape[1],len(ts)))
actual   = np.zeros((A.shape[1],len(ts)))
rs = np.zeros((len(ts),N))
vs = np.zeros((len(ts),N))
for idx,t in enumerate(ts):
    rs[idx, :] = net.r
    vs[idx, :] = net.V
    readout[:,idx] = dec.readout()
    actual[:, idx] = lds.x
    net.update()

    print("%i : %f"%(t/dt, t))
plt.figure()
plt.imshow(rs)
plt.show(block=False)
plt.figure()
plt.imshow(vs)
#plt.scatter(readout[0,:],readout[1,:], label='decoded')
#plt.scatter(actual[0,:],actual[1,:], label='actual')
#plt.legend()
plt.show()

