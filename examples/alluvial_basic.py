"""
================================
Alluvial Demo - Membership based
================================

Draw Alluvial diagram based on membership lists to reflect changes in network
structure.
"""
import numpy as np
# from matplotlib.alluvial import Alluvial
from pyalluv import Alluvial
import matplotlib.pyplot as plt

# Consider a small network of 4 nodes with membership lists at 3 point in time:
memberships = [[0, 1, 1, 2],  # t1
               [3, 0, 1, 2],  # t2
               [1, 0, 1, 1]]  # t3


# Create an alluvial diagram based on the memberships
Alluvial.from_memberships(memberships, layout='top', x=['t1', 't2', 't3'],
                          width=0.2)

plt.show()

#############################################################################
# ==========================
# Alluvial Demo - Flow based
# ==========================
#
# Draw Alluvial using flow matrices
#
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.alluvial import Alluvial
from pyalluv import Alluvial

ext = np.array([1, 2, 1])
flows = np.array([[[0, 1, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]],
                  [[0, 1, 0, 0],
                   [1, 0, 1, 1]]])
# memberships = [[0, 1, 1, 2],  # t1
#                [3, 0, 1, 2],  # t2
#                [1, 0, 1, 1]]  # t3

# Create an alluvial diagram based on the memberships
alluv = Alluvial(x=['t1', 't2', 't3'], flows=flows, ext=ext, width=0.2, yoff=0,
                 layout=['top', 'optimized', 'top'])

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

import matplotlib
matplotlib.alluvial
matplotlib.alluvial.Alluvial
matplotlib.alluvial.Alluvial.from_memberships
matplotlib.alluvial.Alluvial.finish
matplotlib.alluvial.Alluvial.add_from_memberships
