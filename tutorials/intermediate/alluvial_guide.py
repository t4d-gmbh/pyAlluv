"""
=========================
A guide to alluvial plots
=========================

This guide is an extension of the documentation available at
:func:`~matplotlib.alluvial.Alluvial` and covers additional styling and layout
functionalities. Before you continue with this guide, please get familiar with
the documentation.

.. glossary::

   selecting elements
         an alluvial diagram consists of various elements that can be selected
         and styled individually or in custom ensembles.

.. _selections_in_alluvial:

Making selections in an alluvial diagram
========================================

When creating it might not be possible to set the desired styling of all
elements in an alluvial plot at once. In this case it is possible to select
single or multiple elements after initiation and apply modifications to them.
Selections are always made via an existing instance of
:obj:`.matplotlib.alluvial.Alluvial` and can be made on the level of
sub-diagrams, columns, flows or blocks.
Note modifications in an alluvial diagram must occur prior to the calling
:meth:`.matplotlib.alluvial.Alluvial.finish`.
"""
from matplotlib import pyplot as plt
# from matplotlib.alluvial import Alluvial
from pyalluv import Alluvial
import numpy as np

ext = np.array([1, 2, 1])
flows = np.array([[[0, 1, 0],
                   [0, 1, 0],
                   [0, 0, 1],
                   [1, 0, 0]],
                  [[0, 1, 0, 0],
                   [1, 0, 1, 1]]])

alluv = Alluvial(x=['t1', 't2', 't3'])
alluv.add(flows=flows, ext=ext, width=0.2, yoff=0, layout='top')

for block in alluv.select_blocks(0, 1, -1):
    block.set_facecolor('green')
    # set the label
    block.set_label('my block')
