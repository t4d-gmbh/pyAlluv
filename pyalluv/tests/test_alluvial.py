import pytest
import numpy as np
from pyalluv import Alluvial
from matplotlib.testing.decorators import check_figures_equal
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
import pandas as pd


def _test_block_ordering(alluvial, ref_columns):
    # test whether the resulting block ordering in each column reflects
    # the chosen layout (only 'top' and 'bottom')
    columns = alluvial.get_diagrams()[0].get_columns()
    block_heights = [[b.get_height() for b in c]
                     for c in columns]
    assert block_heights == ref_columns


flows = [[[0.8, 0], [0, 0.7], [0, 0.3]], [[0, 1, 0], [0.5, 0, 1]]]
test_data = [
    # (flows, ext, fractionflow, layout, layout, (resulting) columns
    (flows, [10, 10], True, 'top', [[10., 10.], [8., 7., 3.][::-1], [7., 7.]]),
    (flows, [[10, 10], [1, 1, 1], [2, 0.5]], True,
     ['bottom', 'top', 'bottom'], [[10., 10.], [9., 8., 4.][::-1], [10., 9.]]),
    pytest.param(flows, None, True, 'top',
                 [[10., 10.], [8., 7., 3.], [7., 7.]],
                 marks=pytest.mark.xfail),
    (np.atleast_2d(*flows), np.array([10, 10]), True, 'bottom',
     [[10., 10.], [8., 7., 3.], [7., 7.]]),
    (np.atleast_2d(*flows), np.atleast_1d([10, 10], [1, 1, 1], [2, 0.5]), True,
     'top', [[10., 10.], [9., 8., 4.][::-1], [10., 9.][::-1]]),
    pytest.param(np.atleast_2d(flows), None, True, 'bottom',
                 [[10., 10.], [8., 7., 3.], [7., 7.]], marks=pytest.mark.xfail)
]
test_ids = ['lists-fractionflows', 'lists-fractionflows-extInitOnly',
            'lists-fractionflows-extMissing', 'arrays-fractionflows',
            'arrays-fractionflows-extInitOnly',
            'arrays-fractionflows-extMissing']


@pytest.mark.parametrize(
    'flows, ext, fractionflow, layout, ref_columns', test_data, ids=test_ids
)
class TestAlluvialFlows:
    def test_simple_alluvial(self, ext, flows, fractionflow, layout,
                             ref_columns):
        # test creation of alluvial via __init__ directly.
        alluvial = Alluvial(flows=flows, ext=ext, fractionflow=fractionflow,
                            layout=layout, width=1)
        _test_block_ordering(alluvial, ref_columns=ref_columns)

    # @pytest.mark.skip(reason="later")
    def test_alluvial_creation(self, ext, flows, fractionflow, layout,
                               ref_columns):
        # test creation of alluvial diagram with add and finish
        alluvial = Alluvial()
        alluvial.add(flows=flows, ext=ext, fractionflow=fractionflow,
                     layout=layout, width=1)
        alluvial.finish()
        # TODO: ordering might not really what is to test here
        _test_block_ordering(alluvial, ref_columns=ref_columns)

    def test_multiple_subdiagrams(self, ext, flows, fractionflow, layout,
                                  ref_columns):
        # dev-test
        # several sub diagrams with add and finish
        alluvial = Alluvial()
        alluvial.add(flows=flows, ext=ext, fractionflow=fractionflow,
                     layout=layout, width=1)
        alluvial.add(flows=flows, ext=ext, fractionflow=fractionflow, yoff=2,
                     layout=layout, width=1)
        alluvial.finish()
        # TODO: ordering might not really what is to test here
        _test_block_ordering(alluvial, ref_columns=ref_columns)
        _test_block_ordering(alluvial, ref_columns=ref_columns)


memberships = [[0, 1, 1, 2], [3, 0, 1, 2], [1, 1, 1, 0]]
ref_columns = [[1, 2, 1], [1, 1, 1, 1], [3, 1]]
ref_flows = [[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
             [[1, 1, 1, 0], [0, 0, 0, 1]]]
test_data = [
    # (memberships, ref_columns, ref_flows)
    (memberships, ref_columns, ref_flows),
    # passing pandas df
    (pd.DataFrame(memberships, index=[1, 2, 3]),
     ref_columns, ref_flows),
]
test_ids = ['memberships-conversion', 'memberships-df-conversion']


@pytest.mark.parametrize(
    'memberships, ref_columns, ref_flows', test_data, ids=test_ids
)
class TestAlluvialMemberships:
    def test_memberships_conversion(self, memberships, ref_columns, ref_flows):
        alluvial = Alluvial.from_memberships(memberships, width=0.3, layout='centered')
        alluvial.finish()
        _test_block_ordering(alluvial, ref_columns)


flows = [[[0, 3, 2], [4, 0, 0]], [[0, 4], [2, 0], [1, 0], [1, 0], [1, 0]]]
test_data = [
    # (flows, ext, fractionflow, layout, layout, (resulting) columns
    (flows, [4, 3, 2], False, ['bottom', 'top', 'centered'],
     [[4., 3., 2.], [4., 5.], [1., 1., 4., 2., 1.]]),
]
test_ids = ['block-sorting']


@pytest.mark.parametrize(
    'flows, ext, fractf, layout, ref_columns', test_data, ids=test_ids
)
class TestAlluvialLayout:

    @pytest.mark.devtest
    def test_vertical_ordering(self, flows, ext, fractf, layout, ref_columns):
        alluvial = Alluvial(flows=flows, ext=ext, fractionflow=fractf,
                            layout=layout, width=0.2)
        _test_block_ordering(alluvial, ref_columns=ref_columns)
        # dev-test
        # Make sure the ordering is as expected for 'top', 'bottom', 'centered'
        # and 'optimized'

    @pytest.mark.devtest
    def test_axis_position(self, flows, ext, fractf, layout, ref_columns):
        raise NotImplementedError()

    @pytest.mark.devtest
    def test_yoff(self, flows, ext, fractf, layout, ref_columns):
        raise NotImplementedError()
        from pyalluv import Alluvial
        from matplotlib import pyplot as plt
        a = Alluvial.from_memberships(
            [[0, 1, 1, 2], [3, 0, 1, 2], [1, 0, 1, 1]], layout='centered',
            width=0.2, hspace_combine='divide'
        )
        a.add_memberships([[0, 1, 1, 2], [3, 0, 1, 2], [1, 0, 1, 1]],
                          layout='top', width=0.2, hspace_combine='divide',
                          yoff=-2)
        a.add_memberships([[0, 1, 1, 2], [3, 0, 1, 2], [1, 0, 1, 1]],
                          layout='bottom', width=0.2, hspace_combine='divide',
                          yoff=2)
        a.finish()
        a.ax.set_xlim(-1, 4)
        a.ax.set_ylim(-4, 8)
        plt.show()


class TestAlluvialStyling:
    @check_figures_equal()
    def test_Block_styling(self, fig_test, fig_ref):
        # dev-test
        # Check individual styling of Rectangles.
        style = dict(ec='green', lw=2, clip_on=True)
        # create the two figures
        tesax = fig_test.subplots()
        refax = fig_ref.subplots()
        # ###
        # refax
        # draw a simple Recangle on refax
        pc = []
        pc.append(Rectangle((0, 0), width=1, height=3, **style))
        pc.append(Rectangle((0, 4), width=1, height=1, **style))
        pc.append(Rectangle((2, 0), width=1, height=2, fc='red', **style))
        refax.add_collection(PatchCollection(pc, match_original=True, zorder=4))
        # ###
        # tesax
        # draw an alluvial with 1 diagram 1 col and a single block on tesax
        alluvial = Alluvial(x=[0, 2], ax=tesax, width=1)
        diagram0 = alluvial.add(flows=None, ext=[[3, 1], [2]], layout='bottom',
                                **style)
        # set the styling of a single block
        block = diagram0.get_block((1, 0))  # column 1, block 0
        block.set_property('facecolor', 'red')
        alluvial.finish()
        # ###

        # set common limits and axis styling
        refax.set_xlim(*tesax.get_xlim())
        refax.set_ylim(*tesax.get_ylim())

    @check_figures_equal()
    def test_styling_hierarchy(self, fig_test, fig_ref):
        # dev-test
        # Test styling hierarchy: Block/Flow > SubDiagram > Alluvial
        # set styling
        style = dict(ec='green', lw=2)
        # create the two figures
        tesax = fig_test.subplots()
        refax = fig_ref.subplots()
        # draw a simple Recangle on refax
        pc = []
        yoff = 4
        pc.append(Rectangle((0, yoff), width=1, height=1, fc='blue', **style))
        pc.append(Rectangle((0, 0), width=1, height=3, fc='yellow', **style))
        pc.append(Rectangle((2, 0), width=1, height=2, fc='red', **style))
        refax.add_collection(PatchCollection(pc, match_original=True))
        # set fc to blue for the entire alluvial plot
        alluvial = Alluvial(x=[0, 2], ax=tesax, width=1, fc='blue', **style)
        # set fc to yellow for first subdiagram
        diagram0 = alluvial.add(flows=None, ext=[[3], [2]], layout='bottom',
                                fc='yellow', **style)
        # do not set fc for the second fc -> inherit from alluvial
        alluvial.add(flows=None, ext=[1], yoff=yoff, layout='bottom',
                     **style)
        # set the styling of a single block in an already styled subdiagram
        block = diagram0.get_block((1, 0))  # column 1, block 0
        block.set_property('facecolor', 'red')
        alluvial.finish()

        # set common limits and axis styling
        refax.set_xlim(-1, 4)
        tesax.set_xlim(-1, 4)
        refax.set_ylim(-1, 6)
        tesax.set_ylim(-1, 6)
