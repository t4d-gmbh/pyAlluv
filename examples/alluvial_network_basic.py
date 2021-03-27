"""
=================================
Alluvial Demo - Network structure
=================================

Draw Alluvial diagram to reflect changes in network structure.
"""
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.alluvial import Alluvial
from pyalluv import Alluvial

# Consider a small network of 4 nodes with membership lists at 3 point in time:
memberships = [[0, 1, 1, 2],  # t1
               [3, 0, 1, 2],  # t2
               [1, 0, 1, 1]]  # t3


# Create an alluvial diagram based on the memberships
alluv = Alluvial.from_memberships(memberships, layout='top',
                                  x=['t1', 't2', 't3'], width=0.2)

plt.show()

#############################################################################
# =======================
# Alluvial Demo - Layouts
# =======================
#
# Draw Alluvial diagrams with different layouts
#
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.alluvial import Alluvial
from pyalluv import Alluvial

memberships = [[0, 1, 1, 2],  # t1
               [3, 0, 1, 2],  # t2
               [1, 0, 1, 1]]  # t3

layouts = ['centered', 'bottom', 'top']
y_offsets = [0, 3, -3]
alluv = Alluvial(x=['t1', 't2', 't3'], width=0.4)
for layout, yoff in zip(layouts, y_offsets):
    # Create an alluvial diagram based on the memberships using a centered layout
    alluv.add_from_memberships(memberships, layout=layout, yoff=yoff)
# TODO: Adding labels for sub-diagrams
# Tell alluvial to determine the layout and draw the diagrams
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
