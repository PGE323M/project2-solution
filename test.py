#!/usr/bin/env python

# Copyright 2018-2020 John T. Foster
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import unittest
import nbconvert
import numpy as np
import yaml

with open("project2.ipynb") as f:
    exporter = nbconvert.PythonExporter()
    python_file, _ = exporter.from_file(f)


with open("project2.py", "w") as f:
    f.write(python_file)

from project2 import Project2

class TestSolution(unittest.TestCase):

    def setUp(self):

        with open('inputs.yml') as f:
            self.inputs = yaml.load(f, Loader=yaml.FullLoader)

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
