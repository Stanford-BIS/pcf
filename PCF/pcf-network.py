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

class LinearDynamicalSystem():
    '''
    A LinearDynamicalSystem (LDS) object instantiates a specified dynamical 
    system. The user provides a LDS specification via initial arguments. 
    
    ---------------- Initial Arguments: ----------------------------------------------------------------------------------------
    init_state: (M numpy array) - an initial configuration of the dynamical system - defaults to 0
    
    A: state transition matrix (M x M numpy array) - determines how one state maps to the next - defaults to Identity matrix
    
    init_input: (L numpy array) - an initial configuration of the input to the dynamical system - defaults to 0
    
    B: input matrix ( M x L numpy array ) - maps the dependence of the state on the input vector - defaults to Identity matrix
    
    dt: time step (miliseconds) - the time elapsed in the dynamical system between successive updates - defaults to 0.001 ms
        

    --------------- Fields -------------------------------------------------------------------------------------------------------
    t: (T numpy array) the elapsed time of the simulation
    
    A: state_transition matrix (see above)
    
    B: input matrix (see above)
    
    x: (M x T numpy array ) state vector over time
    
    u: (L x T numpy array) input vector over time 
    
    '''
    def __init__(self, init_state = np.asarray([0]), A = np.asarray([1]), init_input = np.asarray([0]), B = np.asarray([1]), dt = .001):
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
        
        # if input valid, append to end of input field, otherwise
        # append 0
        if (u is not None and len(u) == len(self.u)):
            self.u   = u
        else:
            self.u   = self._null_input
        self.x = scipy.integrate.solve_ivp(self.get_derivative, (self.t, self.t + self.dt), self.x).y[:,-1]
        self.t += self.dt
                
A = np.zeros((2,2))
A[0, 1] = -1
A[1, 0] = 1
A = A * 6
x0 = np.asarray([.2, .2])

B = np.zeros((2,2))
B[0,0] = 1
B[1,1] = 3
u0 = np.asarray([0, 0])

dt = .01

lds = LinearDynamicalSystem(
    init_state = x0,
    A = A,
    init_input = u0,
    B = B,
    dt = dt
    )
plt.figure()

x = []
while lds.t < 10:
    lds.update(np.asarray([10, lds.t]))
    x.append(lds.x)
  
x = np.asarray(x).T
plt.scatter(x[0,:], x[1,:])
plt.show()  
#      
# for atr in dir(lds):
#     if (atr[0:1] != '_' ):
#         print(atr)
#         print(getattr(lds, atr))
# print(lds.u.shape)
class Network(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        