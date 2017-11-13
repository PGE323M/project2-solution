#/usr/bin/env python

import os
import numpy as np
import yaml

from project2 import Project2

def secret_project2_test_1(BHP=float(os.environ['BHP1'])):

    with open('inputs.yml') as f:
        inputs = yaml.load(f)

    inputs['wells']['bhp']['values'] = [BHP, BHP, BHP, BHP]
    
    test = Project2(inputs)
    
    test.inputs['wells']['bhp']['values'] = [BHP, BHP, BHP, BHP]
    
    test.solve_one_step()
    
    np.testing.assert_allclose(test.get_solution()[605:609], 
                               np.array([ 3309.0,  3287.8,  3262.7,  3237.4]),
                               atol=0.5)
    
    return

def secret_project2_test_2(BHP=float(os.environ['BHP2'])):

    with open('inputs.yml') as f:
        inputs = yaml.load(f)

    inputs['wells']['bhp']['values'] = [BHP, BHP, BHP, BHP]
    
    test = Project2('inputs.yml')
    
    test.solve()
    
    np.testing.assert_allclose(test.get_solution()[500:508], 
                               np.array([ 2228.2,  2228.2,  2228.1,  2228.0,
                                          2228.0,  2227.9,  2227.9,  2227.9]),
                               atol=0.1)
    
    return 
