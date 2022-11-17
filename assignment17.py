#!/usr/bin/env python
# coding: utf-8

# # Homework Assignment 17
# 
# 
# ## Instructions
# 
# Consider the reservoir shown below with the given properties that has been discretized into equal grid blocks.
# 
# ![image](images/grid.png)
# 
# To be clear, there is a constant-rate injector of 1000 ft$^3$/day at $x$ = 5000 ft, $y$ = 5000 ft and a constant BHP well (producer) with $p_w$ = 800 psi at $x$ = 9000 ft, $y$ = 9000 ft. Both wells have a radius of 0.25 ft and no skin factor.
# 
# Use the code you wrote in [Assignment 15](https://github.com/PGE323M-Students/assignment15) and add additional functionality to incorporate the wells.  The wells section of the inputs will look something like:
# 
# ```yml
# 'wells':
#     'rate':
#         'locations': 
#             - [0.0, 1.0]
#             - [9999.0, 2.0]
#         'values': [1000, 1000]
#         'radii': [0.25, 0.25]
#     'bhp':
#         'locations': 
#             - [6250.0, 1.0]
#         'values': [800]
#         'radii': [0.25]
#         'skin factor': 0.0
# ```
# 
# notice that all the values are Python lists so that multiple wells of each type can be included.  The `'locations'` keyword has a value that is a list of lists.  Each tuple contains the $x,y$ Cartesian coordinate pair that gives the location of the well.  You must write some code that can take this $x,y$-pair and return the grid block number that the well resides in.  This should be general enough that changing the number of grids in the $x$ and $y$ directions still gives the correct grid block.  Once you know the grid block numbers for the wells, the changes to `fill_matrices()` should be relatively easy.
# 
# All of the old tests from the last few assignments are still in place, so your code must run in the absence of any well section in your inputs.

# In[1]:


import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import yaml

from assignment13 import OneDimReservoir


# In[6]:


class TwoDimReservoir(OneDimReservoir):
    
    def __init__(self, inputs):
        '''
            Class for solving one-dimensional reservoir problems with
            finite differences.
        '''
        
        #stores input dictionary as class attribute
        if isinstance(inputs, str):
            with open(inputs) as f:
                self.inputs = yaml.load(f)
        else:
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
        
        self.viscosity = self.inputs['fluid']['water']['viscosity']
        self.formation_volume_factor = self.inputs['fluid']['water']['formation volume factor']
        self.compressibility = self.inputs['fluid']['water']['compressibility'] 
        self.Nx = self.inputs['numerical']['number of grids']['x']
        self.Ny = self.inputs['numerical']['number of grids']['y']
        self.N = self.Nx * self.Ny
        self.delta_t = self.inputs['numerical']['time step']
        
        #Read in 'unit conversion factor' if it exists in the input deck, 
        #otherwise set it to 1.0
        if 'conversion factor' in self.inputs:
            self.conversion_factor = self.inputs['conversion factor']
        else:
            self.conversion_factor = 1.0
            
        
        phi = self.inputs['reservoir']['porosity']
        k = self.inputs['reservoir']['permeability']
        d = self.inputs['reservoir']['depth']
        
        self.permeability = self.check_input_and_return_data(k)
        self.depth = self.check_input_and_return_data(d)
        self.porosity = self.check_input_and_return_data(phi)
        
        #computes delta_x and delta_y
        delta_x = self.assign_delta_x_array()
        delta_y = self.assign_delta_y_array()
        
        self.delta_x, self.delta_y = np.meshgrid(delta_x, delta_y)
        
        self.area = self.delta_x * self.delta_y
        
        #If wells are present, find their grid indices, and compute productivity
        #index
        if 'wells' in self.inputs:
            if 'rate' in self.inputs['wells']:
                self.rate_well_grids = self.compute_well_index_locations('rate')
                self.rate_well_values = np.array(self.inputs['wells']['rate']['values'], 
                                                 dtype=np.double)
                self.rate_well_prod_ind = self.compute_productivity_index('rate')
            else:
                self.rate_well_grids = None

            if 'bhp' in self.inputs['wells']:
                self.bhp_well_grids = self.compute_well_index_locations('bhp')
                self.bhp_well_values = np.array(self.inputs['wells']['bhp']['values'], 
                                           dtype=np.double)
                self.bhp_well_prod_ind = self.compute_productivity_index('bhp')
            else:
                self.bhp_well_grids = None
        else:
            self.rate_well_grids = None
            self.bhp_well_grids = None
            
            
    def compute_well_index_locations(self, well_type='rate'):
        """
           Used to find well index locations from given coordinate positions.
        """
        
        #Reassignment for convenience, not a deep-copy
        dx = self.delta_x
        dy = self.delta_y

        #Compute grid centers
        grid_centers_x = np.cumsum(dx, axis=1) - dx[:,0,None] / 2.0
        grid_centers_y = np.cumsum(dy, axis=0) - dy[None, 0,:] / 2.0
        

        #Coordinate locations of wells
        total_bool_arr = []
        for loc_x, loc_y in self.inputs['wells'][well_type]['locations']:
            bool_arr_1 = grid_centers_x - dx[:,0,None] / 2.0 <= loc_x
            bool_arr_2 = grid_centers_x + dx[:,0,None] / 2.0 >  loc_x
            bool_arr_3 = grid_centers_y - dy[None,0,:] / 2.0 <= loc_y 
            bool_arr_4 = grid_centers_y + dy[None,0,:] / 2.0 >  loc_y
            total_bool_arr += [np.all([bool_arr_1, bool_arr_2, bool_arr_3, bool_arr_4], axis=0)]
        
        grid_numbers = np.arange(self.N, dtype=np.int64).reshape(-1, self.Nx)
        
        return grid_numbers[np.any(total_bool_arr, axis=0)]
    
    
    def compute_productivity_index(self, well_type='rate'):
        """
           Used to compute productivity indices of wells.  All indices for
           a 'well_type' are computed and returned at once (vectorized)
        """

        #Pointer reassignment for convenience
        k = self.permeability
        mu = self.viscosity
        dx = self.delta_x.flatten()
        dy = self.delta_y.flatten()
        d = self.depth.flatten()
        factor = self.conversion_factor
        Balpha = self.formation_volume_factor
        
        if 'skin factor' in self.inputs['wells'][well_type]:
            skin_factor = self.inputs['wells'][well_type]['skin factor']
        else:
            skin_factor = 0.0

        #Get grid indices for 'well_type' wells
        if well_type == 'rate':
            grids = self.rate_well_grids
        elif well_type == 'bhp':
            grids = self.bhp_well_grids
        
        #Read in well radius from inputs
        r_w = np.array(self.inputs['wells'][well_type]['radii'], 
                       dtype=np.double)
        
        #Compute equivalent radius with Peaceman correction
        r_eq = 0.14 * np.sqrt(dx[grids] ** 2. + dy[grids] ** 2.)

        #Return array of productivity indices for 'well_type' wells
        return (2.0 * np.pi * k[grids] * d[grids]) /  (mu * Balpha * np.log(r_eq / r_w) + skin_factor)
        
    
    def assign_delta_x_array(self):
        """
           Used to assign grid block widths (dx values) after permeability
           and porosity has been assigned.

           Can also accept user defined list of dx values.

           TODO: Add ability to read dx values from file.
        """
        
        nxgrids = self.Nx

        #If dx is not defined by user, compute a uniform dx
        if 'delta x' not in self.inputs['numerical']:
            length = self.inputs['reservoir']['length']
            delta_x = np.float64(length) / nxgrids
            delta_x_arr = np.ones(nxgrids) * delta_x
        else:
            #Convert to numpy array and ensure that the length of 
            #dx matches ngrids
            delta_x_arr = np.array(self.inputs['numerical']['delta x'], 
                              dtype=np.double)

        return delta_x_arr
    
    
    def assign_delta_y_array(self):
        """
           Used to assign grid block widths (dx values) after pereability
           and porosity has been assigned.

           Can also accept user defined list of dx values.

           TODO: A ability to read dx values from file.
        """
        
        nygrids = self.Ny

        #If dx is not defined by user, compute a uniform dx
        if 'delta y' not in self.inputs['numerical']:
            height = self.inputs['reservoir']['height']
            delta_y = np.float64(height) / nygrids
            delta_y_arr = np.ones(nygrids) * delta_y
        else:
            #Convert to numpy array and ensure that the length of 
            #dx matches ngrids
            delta_y_arr = np.array(self.inputs['numerical']['delta y'], 
                              dtype=np.double)

        return delta_y_arr
    
    
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
            ngrids = self.N
            data = (input_name *  np.ones(ngrids))
            
        return data
    
    
    def compute_transmissibility(self, i, j):
        '''
            Computes the transmissibility.
        '''
        
        mu = self.viscosity
        k = self.permeability
        d = self.depth
        B_alpha = self.formation_volume_factor
        dx = self.delta_x.flatten()
        dy = self.delta_y.flatten()
        
        if k[i] <= 0.0 and k[j] <= 0:
            return 0.0
        else:
            if abs(i - j) <= 1:
                k_half = k[i] * k[j] * (dx[i] + dx[j]) / (dx[i] * k[j] + dx[j] * k[i])
                dx_half = (dx[i] + dx[j]) / 2.
                return k_half * d[i] * dy[i] / mu / B_alpha / dx_half
            else:
                k_half = k[i] * k[j] * (dy[i] + dy[j]) / (dy[i] * k[j] + dy[j] * k[i])
                dx_half = (dy[i] + dy[j]) / 2.
                return k_half * d[i] * dx[i] / mu / B_alpha / dx_half
    
    
    
    def compute_accumulation(self, i):
        '''
            Computes the accumulation.
        '''
        
        c_t = self.compressibility
        phi = self.porosity
        B_alpha = self.formation_volume_factor
        
        d = self.depth
        dx = self.delta_x.flatten()
        dy = self.delta_y.flatten()
        
        volume = d[i] * dx[i] * dy[i]
        
        return volume * phi[i] * c_t / B_alpha
    
    
    def fill_matrices(self):
        '''
           Assemble the transmisibility, accumulation matrices, and the flux
           vector.  Returns sparse data-structures
        '''
    
        
        #Pointer reassignment for convenience
        N = self.N
        Nx = self.Nx
        Ny = self.Ny
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
        bc_type_3 = bcs['top']['type'].lower()
        bc_type_4 = bcs['bottom']['type'].lower()
        bc_value_1 = bcs['left']['value']
        bc_value_2 = bcs['right']['value']
        bc_value_3 = bcs['top']['value']
        bc_value_4 = bcs['bottom']['value']
      
        #Loop over all grid cells
        for i in range(N):
            

            #Check to make sure problem is truly 2D
            if Nx > 1:
                #Apply left BC
                if i % Nx == 0:
                    T[i, i + 1] = -self.compute_transmissibility(i, i + 1)

                    if bc_type_1 == 'prescribed flux':
                        T[i, i] += 0
                    elif bc_type_1 == 'prescribed pressure':
                        #Computes the transmissibility of the ith block
                        T0 = self.compute_transmissibility(i, i)
                        T[i, i] -= 2.0 * T0
                        Q[i] = 2.0 * T0 * bc_value_1 * factor
                    else:
                        pass #TODO: Add error checking here if no bc is specified

                #Apply right BC
                elif (i+1) % Nx == 0:
                    T[i, i - 1] = -self.compute_transmissibility(i, i - 1)

                    if bc_type_2 == 'prescribed flux':
                        T[i, i] += 0
                    elif bc_type_2 == 'prescribed pressure':
                        #Computes the transmissibility of the ith block
                        T0 = self.compute_transmissibility(i, i)
                        T[i, i] -= 2.0 * T0
                        Q[i] = 2.0 * T0 * bc_value_2 * factor
                    else:
                        pass #TODO:Add error checking here if no bc is specified
                else:
                    T[i, i + 1] = -self.compute_transmissibility(i, i + 1)
                    T[i, i - 1] = -self.compute_transmissibility(i, i - 1)
                    
                
            #Check to make sure problem is truly 2D
            if Ny > 1:
                #Apply top boundary condition
                if i > (N-1) - Nx:

                    T[i, i - Nx] = -self.compute_transmissibility(i, i - Nx)

                    if bc_type_3 == 'prescribed flux':
                        T[i, i] += 0
                    elif bc_type_3 == 'prescribed pressure':
                        #Computes the transmissibility of the ith block
                        T0 = self.compute_transmissibility(i, i)
                        T[i, i] -= 2.0 * T0
                        Q[i] = 2.0 * T0 * bc_value_3 * factor
                    else:
                        pass #TODO: Add error checking here if no bc is specified

                #Apply bottom boundary condition
                elif i < Nx:
                    T[i, i + Nx] = -self.compute_transmissibility(i, i + Nx)

                    if bc_type_4 == 'prescribed flux':
                        T[i, i] += 0
                    elif bc_type_4 == 'prescribed pressure':
                        #Computes the transmissibility of the ith block
                        T0 = self.compute_transmissibility(i, i)
                        T[i, i] -= 2.0 * T0
                        Q[i] = 2.0 * T0 * bc_value_4 * factor
                    else:
                        pass #TODO: Add error checking here if no bc is specified
                else:
                    T[i, i - Nx] = -self.compute_transmissibility(i, i - Nx)
                    T[i, i + Nx] = -self.compute_transmissibility(i, i + Nx)
                    

            #Sum off diagonal entries into diagonal
            T[i, i] = -np.sum(T[i])
            
            #Compute accumulations
            B[i] = self.compute_accumulation(i)
        
            
        #If constant-rate wells are present, add them to the flux vector
        if self.rate_well_grids is not None:
            Q[self.rate_well_grids] += self.rate_well_values
            
        #If bhp wells are present, add productivity index to the flux vector and T matrix
        if self.bhp_well_grids is not None:
            Q[self.bhp_well_grids] += self.bhp_well_prod_ind * self.bhp_well_values * factor
            T[self.bhp_well_grids, self.bhp_well_grids] += self.bhp_well_prod_ind 
        
        #Return sparse data-structures
        self.T = T.tocsr() * factor
        self.B = scipy.sparse.csr_matrix((B, (np.arange(N), np.arange(N))), shape=(N,N))
        self.Q = Q
        
        return


# In[14]:


def test_compute_productivity_index():
    
    from test import TestSolution
    
    t = TestSolution()
    t.setUp()
    
    
    parameters = t.inputs
    
    parameters['wells'] = {
            'rate': {
                'locations': [[0.0, 1.0]],
                'values': [1000],
                'radii': [0.25]
            },
            'bhp': {
                'locations': [[6250.0, 1.0]],
                'values': [800],
                'radii': [0.25]
            }
        }
    
    parameters['reservoir'] = {
            'permeability': 50, #mD
            'porosity': 0.2,
            'length': 10000, #ft
            'height': 2500, #ft
            'depth': 80 #ft
        }
    
    
    parameters['boundary conditions']['left']['type'] = 'prescribed flux'
    parameters['boundary conditions']['left']['value'] = 0.0
    parameters['boundary conditions']['right']['type'] = 'prescribed pressure'
    parameters['boundary conditions']['right']['value'] = 2000.0
    
    problem = TwoDimReservoir(parameters)
    
    np.testing.assert_allclose(problem.compute_productivity_index('bhp'), 3310.9, atol=0.5)
    
    return


# In[16]:


#test_compute_productivity_index()


# In[9]:


#problem = TwoDimReservoir('inputs.yml')
#problem.solve()
#problem.plot()

