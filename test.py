#/usr/bin/env python

import os
import numpy as np

from project2 import Project2

def secret_project2_test_1(BHP=float(os.environ['BHP1'])):
    
    test = Project2('inputs.yml')
    
    test.inputs['wells']['bhp']['values'] = [BHP, BHP, BHP, BHP]
    
    test.parse_inputs()
    
    test.fill_matrices()
    
    test.solve_one_step()
    
    np.testing.assert_allclose(test.get_solution()[605:609], 
                               np.array([ 3309.0,  3287.8,  3262.7,  3237.4]),
                               atol=0.5)
    
    return

def secret_project2_test_2(BHP=float(os.environ['BHP2'])):
    
    test = Project2('inputs.yml')
    
    test.inputs['wells']['bhp']['values'] = [BHP, BHP, BHP, BHP]
    
    test.parse_inputs()
    
    test.fill_matrices()
    
    test.solve()
    
    np.testing.assert_allclose(test.get_solution()[500:508], 
                               np.array([ 2228.2,  2228.2,  2228.1,  2228.0,
                                          2228.0,  2227.9,  2227.9,  2227.9]),
                               atol=0.1)
    
    return 
