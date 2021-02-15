import pytest
import numpy as np
from pyalluv import Alluvial
from matplotlib.testing.decorators import check_figures_equal


flows = [[[0.8, 0], [0, 0.7], [0, 0.3]], [[0, 1, 0], [0.5, 0, 1]]]
test_data = [
    (flows, [10, 10], True, [[10., 10.], [8., 7., 3.], [7., 7.]], 'top'),
    (
        flows, [[10, 10], [1, 1, 1], [2, 0.5]], True,
        [[10., 10.], [9., 8., 4.], [10., 9.]], ['bottom', 'top', 'bottom']
    ),
    pytest.param(
        flows, None, True, [[10., 10.], [8., 7., 3.], [7., 7.]], 'top',
        marks=pytest.mark.xfail
    ),
    (
        np.asarray(flows), np.array([10, 10]), True,
        [[10., 10.], [8., 7., 3.], [7., 7.]], 'top'
    ),
    (
        np.asarray(flows), np.asarray([[10, 10], [1, 1, 1], [2, 0.5]]), True,
        [[10., 10.], [9., 8., 4.], [10., 9.]], 'top'
    ),
    pytest.param(
        np.asarray(flows), None, True, [[10., 10.], [8., 7., 3.], [7., 7.]],
        'bottom', marks=pytest.mark.xfail
    )
]
test_ids = ['lists-fractionflows', 'lists-fractionflows-extInitOnly',
            'lists-fractionflows-extMissing', 'arrays-fractionflows',
            'arrays-fractionflows-extInitOnly',
            'arrays-fractionflows-extMissing']


@pytest.mark.parametrize(
    'flows, ext, fractionflow, columns, layout', test_data, ids=test_ids
)
class TestAlluivial:
    def _test_block_ordering(self, alluvial_collumns, ref_columns, layout):
        # test whether the resulting block ordering in each column reflects
        # the chosen layout (only 'top' and 'bottom')
        if isinstance(layout, str):
            if layout == 'top':
                _rev = False
            elif layout == 'bottom':
                _rev = True
            rev = [_rev for _ in ref_columns]
        else:
            rev = [False if _l == 'top' else True for _l in layout]
        block_heights = [[b.get_height() for b in c]
                         for c in alluvial_collumns]
        assert block_heights == [
            sorted(c, reverse=rev[i])
            for i, c in enumerate(ref_columns)
        ]

    def test_simple_alluvial(self, ext, flows, fractionflow, columns, layout):
        # test creation of alluvial via __init__ directly.
        alluvial = Alluvial(flows=flows, ext=ext, fractionflow=fractionflow,
                            layout=layout)
        self._test_block_ordering(alluvial.get_diagrams()[0].get_columns(),
                                  columns, layout=layout)

    # @pytest.mark.skip(reason="later")
    def test_alluvial_creation(self, ext, flows, fractionflow, columns,
                               layout):
        # test creation of alluvial diagram with add and finish
        alluvial = Alluvial()
        alluvial.add(flows=flows, ext=ext, fractionflow=fractionflow,
                     layout=layout)
        alluvial.finish()
        self._test_block_ordering(alluvial.get_diagrams()[0].get_columns(),
                                  columns, layout=layout)


class TestAlluvialStyling:
    @check_figures_equal()
    def test_Block_styling(self, fig_test, fig_ref):
        # check if the styling parameter are passed along correctly to the
        # creation of Rectangles.
        from matplotlib.patches import Rectangle
        tesax = fig_test.subplots()
        refax = fig_ref.subplots()
        # draw a simple Recangle on refax
        refax.add_patch(Rectangle((0, 0), width=1, height=3))
        # draw an alluvial with 1 diagram 1 col and a single block on tesax
        tesax.add_patch(Rectangle((0, 0), width=1, height=3))

        # set common limits and axis styling
        refax.set_xlim(-1, 2)
        tesax.set_xlim(-1, 2)
        refax.set_ylim(-1, 4)
        tesax.set_ylim(-1, 4)

    # def test_Node(self):
    #     node = pyalluv.Cluster(
    #         height=10,
    #         anchor=(0, 0),
    #         widht=4,
    #         x_anchor='left',
    #         label='test node',
    #         label_margin=(1, 2)
    #     )
    #     # make sure x_anchor works fine
    #     assert node.x_pos == 0
