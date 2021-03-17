"""
==================================
Alluvial Demo - Dynamic Structures
==================================

Use Alluvial diagrams to draw dynamic community structures in temporal networks
"""
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
# from matplotlib.alluvial import Alluvial
from pyalluv import Alluvial

# for reproducibility
np.random.seed(19680801)

# Consider a network of 14 nodes with membership lists at 5 point in time.
# To construct a dynamic community structure we define the first 4 nodes as
# references and associate the remaining nodes randomly to one of the reference
# nodes at each time point. As dynamic communities we then identify the groups
# of the 4 reference nodes.
time_points = 5
nbr_refnodes, nbr_varnodes = 4, 10
ref_nodes = np.arange(nbr_refnodes)

# construct the x axis data
start = datetime(2021, 3, 17)
dt = timedelta(days=7)
x = np.array([start + i * dt for i in range(time_points)])

memberships = []
for i in range(time_points):
    memberships.append(
        np.hstack((ref_nodes,
                   np.random.randint(0, nbr_refnodes, nbr_varnodes)))
    )

alluvial = Alluvial(x=x, layout='centered', hspace_combine='divide',
                    blockprops=dict(width=timedelta(days=1)))
alluvial.add_from_memberships(memberships)
# TODO: edgecolor is not set in this case > no borders on blocks
# a.style_tag('dc0', cmap='cool', mappable='x')
alluvial.finish()

plt.xticks(rotation=35)
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
matplotlib.alluvial.Alluvial.finish
matplotlib.alluvial.Alluvial.add_from_memberships
