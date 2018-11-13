#/usr/bin/env python

import os
import numpy as np
import yaml
import unittest

from project2 import Project2

class TestSolution(unittest.TestCase):

    def setUp(self):

        with open('inputs.yml') as f:
            self.inputs = yaml.load(f)

    def test_project2_test_1(self, BHP=2514.0):

        self.inputs['wells']['bhp']['values'] = [BHP, BHP, BHP, BHP]
        
        test = Project2(self.inputs)
        
        test.inputs['wells']['bhp']['values'] = [BHP, BHP, BHP, BHP]
        
        test.solve_one_step()
        
        np.testing.assert_allclose(test.get_solution()[605:609], 
                                   np.array([ 3309.0,  3287.8,  3262.7,  3237.4]),
                                   atol=0.5)
        
        return

    def test_project2_test_2(self, BHP=2221.0):


        self.inputs['wells']['bhp']['values'] = [BHP, BHP, BHP, BHP]
        
        test = Project2(self.inputs)
        
        test.solve()
        
        np.testing.assert_allclose(test.get_solution()[500:508], 
                                   np.array([ 2228.2,  2228.2,  2228.1,  2228.0,
                                              2228.0,  2227.9,  2227.9,  2227.9]),
                                   atol=0.1)
        
        return 

if __name__ == '__main__':
    unittest.main()
