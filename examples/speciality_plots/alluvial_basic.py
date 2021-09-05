"""
===============================
Alluvial Demo - Basic Flow Demo
===============================

Draw a simple alluvial diagram based on flows.
"""
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.alluvial import Alluvial
from pyalluv import Alluvial

ext = np.array([1, 2, 1])
flows = [[[0, 1, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 0, 0]],
         [[0, 1, 0, 0],
          [1, 0, 1, 1]]]
# memberships = [[0, 1, 1, 2],  # t1
#                [3, 0, 1, 2],  # t2
#                [1, 0, 1, 1]]  # t3

# Create an alluvial diagram based on the memberships
Alluvial(flows=flows, ext=ext, layout='top')
# label block

plt.title("The default settings produce a diagram like this.")
# plt.show()
#############################################################################
# =====================================
# Alluvial Demo - Multiple sub-diagrams
# =====================================
#
# Draw Alluvial using flow matrices
#
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.alluvial import Alluvial
from pyalluv import Alluvial

ext = np.array([1, 2, 1])
flows = [[[0, 1, 0],
          [0, 1, 0],
          [0, 0, 1],
          [1, 0, 0]],
         [[0, 1, 0, 0],
          [1, 0, 1, 1]]]
# memberships = [[0, 1, 1, 2],  # t1
#                [3, 0, 1, 2],  # t2
#                [1, 0, 1, 1]]  # t3

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[],
                     title="Multiple Subdiagrams")
# Create an alluvial diagram based on the memberships
alluv = Alluvial(x=['t1', 't2', 't3'], ax=ax)
alluv.add(flows=flows, ext=ext, width=0.2, yoff=0,
          layout=['top', 'centered', 'bottom'])
# alluv.add(flows, ext, layout='bottom', yoff=3)
alluv.finish()
# label block

plt.show()

#############################################################################
#
# ------------
#
# References
# """"""""""
#
# The use of the following functions, methods, classes and modules is shown
# in this example:

# TODO: remove #'s
# import matplotlib
# matplotlib.alluvial
# matplotlib.alluvial.Alluvial
# matplotlib.alluvial.Alluvial.from_memberships
# matplotlib.alluvial.Alluvial.finish
# matplotlib.alluvial.Alluvial.add_from_memberships
