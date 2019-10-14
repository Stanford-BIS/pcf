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
    def __init__(self, init_state = np.asarray([0]), A = np.asarray([1]), init_input = 0, B = 1, dt = .001):
        '''
        Initialize the Linear Dynamical System. If there are any defaults, ensure they have appropriate
        dimensions. 
        '''
        self.x = init_state
        if (len(self.x) is not np.asarray(A).shape[0]):
            self.A = np.eye(len(self.x))
        else:
            self.A = A


    def get_derivative(self):
        '''
        Return the derivative of the state (x-dot) at the current time
        '''
        return self.A @ self.x[:,-1] + self.B @ self.u[:-1]
    
    
lds = LinearDynamicalSystem([1, 2 ,4], np.eye(3) * 2)

class Network(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        