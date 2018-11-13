
# coding: utf-8

# # Assignment 12

# In[1]:


import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt


# In[2]:


class OneDimReservoir():
    
    def __init__(self, inputs):
        '''
            Class for solving one-dimensional reservoir problems with
            finite differences.
        '''
        
        #stores input dictionary as class attribute
        self.inputs = inputs
        
        #assigns class attributes from input data
        self.parse_inputs()
        
        #calls fill matrix method (must be completely implemented to work)
        self.fill_matrices()
        
        #applies the initial reservoir pressues to self.p
        self.apply_initial_conditions()
        
        #create an empty list for storing data if plots are requested
        if 'plots' in self.inputs:
            self.p_plot = []
            
        return
    
    def parse_inputs(self):
        '''
            Stores inputs as data attributes
        '''
        
        self.viscosity = self.inputs['fluid']['viscosity']
        self.formation_volume_factor = self.inputs['fluid']['formation volume factor']
        self.compressibility = self.inputs['fluid']['compressibility'] 
        self.ngrids = self.inputs['numerical']['number of grids']
        self.delta_t = self.inputs['numerical']['time step']
        
        #Read in 'unit conversion factor' if it exists in the input deck, 
        #otherwise set it to 1.0
        if 'conversion factor' in self.inputs:
            self.conversion_factor = self.inputs['conversion factor']
        else:
            self.conversion_factor = 1.0
            
        
        phi = self.inputs['reservoir']['porosity']
        k = self.inputs['reservoir']['permeability']
        A = self.inputs['reservoir']['cross sectional area']
        
        self.permeability = self.check_input_and_return_data(k)
        self.area = self.check_input_and_return_data(A)
        self.porosity = self.check_input_and_return_data(phi)
        
        #computes delta_x
        self.delta_x = self.assign_delta_x_array()
        
    
    def assign_delta_x_array(self):
        """
           Used to assign grid block widths (dx values) after pereability
           and porosity has been assigned.

           Can also accept user defined list of dx values.

           TODO: Add ability to read dx values from file.
        """
        
        ngrids = self.ngrids

        #If dx is not defined by user, compute a uniform dx
        if 'delta x' not in self.inputs['numerical']:
            length = self.inputs['reservoir']['length']
            delta_x = np.float(length) / ngrids
            delta_x_arr = np.ones(ngrids) * delta_x
        else:
            #Convert to numpy array and ensure that the length of 
            #dx matches ngrids
            delta_x_arr = np.array(self.inputs['numerical']['delta x'], 
                              dtype=np.double)

            length_delta_x_arr = delta_x_arr.shape[0]
            
            #For user input 'delta x' array, we need to ensure that its size
            #agress with ngrids as determined from permeability/porosity values
            assert length_delta_x_arr == ngrids, ("User defined 'delta x' array                                                    doesn't match 'number of grids'")

        return delta_x_arr
    
    
    def check_input_and_return_data(self, input_name):
        '''
           Used to parse data from the inputs 
           depending on whether they are to be read from file, given by user
           input lists or constants.
        '''

        #Check to see if data is given by a file
        if isinstance(input_name, str):
            #Get filename
            filename = input_name
            #Load data 
            data = np.loadtxt(filename, dtype=np.double)
            
        #Check to see if data is given by a list
        elif isinstance(input_name, (list, tuple)):
            #Turn the list into numpy array
            data = np.array(input_name, 
                            dtype=np.double)

        #data is a constant array (homogeneous)
        else:
            ngrids = self.inputs['numerical']['number of grids']
            data = (input_name *  np.ones(ngrids))
            
        return data
    
    
    def compute_transmissibility(self, i, j):
        '''
            Computes the transmissibility.
        '''
        
        mu = self.viscosity
        k = self.permeability
        A = self.area
        B_alpha = self.formation_volume_factor
        dx = self.delta_x
        
        kA_half = 2 * k[i] * A[i] * k[j] * A[j] / (k[i] * A[i] * dx[j] + k[j] * A[j] * dx[i]) 
        
        return kA_half / mu / B_alpha
    
    
    
    def compute_accumulation(self, i):
        '''
            Computes the accumulation.
        '''
        
        c_t = self.compressibility
        phi = self.porosity
        B_alpha = self.formation_volume_factor
        
        A = self.area
        dx = self.delta_x
        
        volume = A[i] * dx[i]
        
        return volume * phi[i] * c_t / B_alpha
    
    
    def fill_matrices(self):
        '''
           Assemble the transmisibility, accumulation matrices, and the flux
           vector.  Returns sparse data-structures
        '''
    
        
        #Pointer reassignment for convenience
        N = self.ngrids
        factor = self.conversion_factor

        #Begin with a linked-list data structure for the transmissibilities,
        #and one-dimenstional arrays for the diagonal of B and the flux vector
        T = scipy.sparse.lil_matrix((N, N), dtype=np.double)
        B = np.zeros(N, dtype=np.double)
        Q = np.zeros(N, dtype=np.double)

        #Read in boundary condition types and values
        bcs = self.inputs['boundary conditions']
        bc_type_1 = bcs['left']['type'].lower()
        bc_type_2 = bcs['right']['type'].lower()
        bc_value_1 = bcs['left']['value']
        bc_value_2 = bcs['right']['value']
      
        #Loop over all grid cells
        for i in range(N):

            #Apply left BC
            if i == 0:
                T[i, i+1] = -self.compute_transmissibility(i, i + 1)

                if bc_type_1 == 'prescribed flux':
                    T[i, i] = T[i,i] - T[i, i+1]
                elif bc_type_1 == 'prescribed pressure':
                    #Computes the transmissibility of the ith block
                    T0 = self.compute_transmissibility(i, i)
                    T[i, i] = T[i,i] - T[i, i+1] + 2.0 * T0
                    Q[i] = 2.0 * T0 * bc_value_1 * factor
                else:
                    pass #TODO: Add error checking here if no bc is specified

            #Apply right BC
            elif i == (N - 1):
                T[i, i-1] = -self.compute_transmissibility(i, i - 1)

                if bc_type_2 == 'prescribed flux':
                    T[i, i] = T[i,i] - T[i, i-1]
                elif bc_type_2 == 'prescribed pressure':
                    #Computes the transmissibility of the ith block
                    T0 = self.compute_transmissibility(i, i)
                    T[i, i] = T[i, i] - T[i, i-1] + 2.0 * T0
                    Q[i] = 2.0 * T0 * bc_value_2 * factor
                else:
                    pass #TODO:Add error checking here if no bc is specified

            #If there is no boundary condition compute interblock transmissibilties
            else:
                T[i, i-1] = -self.compute_transmissibility(i, i-1)
                T[i, i+1] = -self.compute_transmissibility(i, i+1)
                T[i, i] = (self.compute_transmissibility(i, i-1) +
                           self.compute_transmissibility(i, i+1))

            #Compute accumulations
            B[i] = self.compute_accumulation(i)

        
        #Return sparse data-structures
        self.T = T.tocsr() * factor
        self.B = scipy.sparse.csr_matrix((B, (np.arange(N), np.arange(N))), shape=(N,N))
        self.Q = Q
        
        return
        
            
                
    def apply_initial_conditions(self):
        '''
            Applies initial pressures to self.p
        '''
        
        N = self.inputs['numerical']['number of grids']
        
        self.p = np.ones(N) * self.inputs['initial conditions']['pressure']
        
        return
                
                
    def solve_one_step(self):
        '''
            Solve one time step using either the implicit or explicit method
        '''
        
        B = self.B
        T = self.T
        Q = self.Q
        
        dt = self.delta_t
        
        if self.inputs['numerical']['solver'] == 'explicit':
            self.p = self.p + dt * 1. / B.diagonal() * (Q - T.dot(self.p)) 
        elif self.inputs['numerical']['solver'] == 'implicit':
            self.p = scipy.sparse.linalg.cg(T + B / dt, B.dot(self.p) / dt + Q, atol='legacy')[0]
        elif 'mixed method' in self.inputs['numerical']['solver']:
            
            theta = self.inputs['numerical']['solver']['mixed method']['theta']
            
            A = (1 - theta) * T + B / dt
            b = (B / dt - theta * T).dot(self.p) + Q
            
            self.p = scipy.sparse.linalg.cg(A, b, atol='legacy')[0]
            
        return
            
            
    def solve(self):
        '''
            Solves until "number of time steps"
        '''
        
        for i in range(self.inputs['numerical']['number of time steps']):
            self.solve_one_step()
            
            if i % self.inputs['plots']['frequency'] == 0:
                self.p_plot += [self.get_solution()]
                
        return
                
    def plot(self):
        '''
           Crude plotting function.  Plots pressure as a function of grid block #
        '''
        
        if self.p_plot is not None:
            for i in range(len(self.p_plot)):
                plt.plot(self.p_plot[i])
        
        return
            
    def get_solution(self):
        '''
            Returns solution vector
        '''
        return self.p

