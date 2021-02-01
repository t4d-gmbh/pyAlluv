import pytest
import numpy as np
from pyalluv import Alluvial


flows = [[[0.8, 0], [0, 0.7], [0, 0.3]], [[0, 1, 0], [0.5, 0, 1]]]
test_data = [
    (flows, [10, 10], True, [[10., 10.], [8., 7., 3.], [7., 7.]]),
    (
        flows, [[10, 10], [1, 1, 1], [2, 0.5]], True,
        [[10., 10.], [9., 8., 4.], [10., 10.]]
    ),
    pytest.param(
        flows, None, True, [[10., 10.], [8., 7., 3.], [7., 7.]],
        marks=pytest.mark.xfail
    ),
    (
        np.asarray(flows), np.array([10, 10]), True,
        [[10., 10.], [8., 7., 3.], [7., 7.]]
    ),
    (
        np.asarray(flows), np.asarray([[10, 10], [1, 1, 1], [2, 0.5]]), True,
        [[10., 10.], [9., 8., 4.], [10., 10.]]
    ),
    pytest.param(
        np.asarray(flows), None, True, [[10., 10.], [8., 7., 3.], [7., 7.]],
        marks=pytest.mark.xfail
    )
]
test_ids = [
    'lists-fractionflows',
    'lists-fractionflows-extInitOnly',
    'lists-fractionflows-extMissing',
    'arrays-fractionflows',
    'arrays-fractionflows-extInitOnly',
    'arrays-fractionflows-extMissing',
]


@pytest.mark.parametrize(
    'flows, ext, fractionflow, columns',
    test_data,
    ids=test_ids
)
class TestAllivial:
    def test_simple_alluvial(self, ext, flows, fractionflow, columns):
        # test creation of alluvial via __init__ directly.
        alluvial = Alluvial(flows=flows, ext=ext, fractionflow=fractionflow)
        # TODO: check columns
        assert alluvial.columns == columns

    def test_alluvial_creation(self, ext, flows, fractionflow, columns):
        # test creation of alluvial diagram with add and finish
        alluvial = Alluvial()
        alluvial.add(flows=flows, ext=ext, fractionflow=fractionflow)
        alluvial.finish()
        # TODO: check columns
        assert alluvial.columns == columns

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
