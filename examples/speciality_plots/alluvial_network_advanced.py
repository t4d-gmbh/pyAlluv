"""
==================================
Alluvial Demo - Dynamic Structures
==================================

Use Alluvial diagrams to draw dynamic community structures in temporal networks
"""
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.alluvial import Alluvial
from pyalluv import Alluvial

# for reproducibility
np.random.seed(19680802)

# Consider a network of 14 nodes with membership lists at 5 point in time.
# To construct a dynamic community structure we define the first 4 nodes as
# references and associate the remaining nodes randomly to one of the reference
# nodes at each time point. As dynamic communities we then identify the groups
# of the 4 reference nodes.
time_points = 5
nbr_refnodes, nbr_varnodes = 4, 30
ref_nodes = np.arange(nbr_refnodes)


# construct the x axis data
# start = datetime(2021, 3, 17)
# dt = timedelta(days=7)
# x = np.array([start + i * dt for i in range(time_points)])
x = np.array([i for i in range(time_points)])


def new_affiliation(affiliation, pchange=0.2):
    """
    Redraw the affiliation with a certain probability.
    """
    return np.where(np.random.rand(*affiliation.shape) > pchange, affiliation,
                    np.random.randint(0, nbr_refnodes, nbr_varnodes))


memberships = []
initial_affiliation = np.random.randint(0, nbr_refnodes, nbr_varnodes)
dcs = []
for i in range(time_points):
    memberships.append(
        np.hstack((ref_nodes, new_affiliation(initial_affiliation)))
    )
    dcs.append(ref_nodes)

# TODO:
# Multiple plots
# # layout bottom
# # layout centered
# # layout optimized
# # as subd's

fig, axs = plt.subplots(2, 2)

# layout bottom
alluvial = Alluvial.from_memberships(memberships, dcs=dcs, ax=axs[0][0], x=x,
                                     layout='bottom', hspace_combine='divide',
                                     blockprops=dict(width=0.2))

# layout centered
alluvial = Alluvial.from_memberships(memberships, dcs=dcs, ax=axs[0][1], x=x,
                                     layout='centered', hspace_combine='divide',
                                     blockprops=dict(width=0.2))

# layout optimized
alluvial = Alluvial.from_memberships(memberships, dcs=dcs, ax=axs[1][0], x=x,
                                     layout='optimized', hspace_combine='divide',
                                     blockprops=dict(width=0.2))


# separated dcs into subds
# TODO

plt.xticks(rotation=35)
plt.show()

#############################################################################
# =======================
# Alluvial Demo - Layouts
# =======================
#
# Draw Alluvial diagrams with different layouts
#

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
