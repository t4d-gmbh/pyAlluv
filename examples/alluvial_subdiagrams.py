"""
==========================
Alluvial Demo - Flow based
==========================

Draw Alluvial using flow matrices

"""
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.alluvial import Alluvial
from pyalluv import Alluvial

ext = np.array([1, 3, 1])
flows = np.array([[[0, 1, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]],
                  [[0, 1, 0, 0],
                   [1, 0, 1, 1]]])
extouts = [[(1, (1, 1, 2), 1)], []]

# Start with an empty alluvial plot
alluv = Alluvial(x=['t1', 't2', 't3'], blockprops=dict(width=0.2))
# Now we add the first subdiagram
alluv.add(flows=flows, ext=ext, extout=extouts, yoff=0,
          layout=['top', 'top', 'top'])
# Adding a new subdiagram, using the same flows
alluv.add(flows=flows, ext=ext, yoff=7,
          layout=['top', 'optimized', 'bottom'])

alluv.finish()
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
