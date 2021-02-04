import logging
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib import docstring
from matplotlib import cbook
from maptlotlib import _api
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.path import Path
from matplotlib.patches import Patch
import matplotlib.patches as patches
from datetime import datetime
from bisect import bisect_left

# TODO: unused so far
_log = logging.getLogger(__name__)

__author__ = 'Jonas I. Liechti'


@cbook._define_aliases({
    "horizontalalignment": ["ha"],
})
class _Block(Patch):
    """
    A patch describing a block in an Alluvial diagram.

    Blocks in an Alluvial diagram get their vertical position assigned by a
    layout algorithm and thus after creation. This is the rational to why
    *_Block* inherits directly from `matplotlib.patches.Patch`, rather than
    `matplotlib.patches.PathPatch` or `matplotlib.patches.Rectangle`.

    """
    @docstring.depend_interpd
    def __init__(self, height, anchor=None, width=1.0, label=None,
                 horizontalalignment='center', label_margin=(0, 0),
                 pathprops=None, **kwargs):
        """
        Parameters
        -----------
        height : float
          Size of the block.
        anchor : float or (float, float), optional
          Set the anchor point. Either only the x coordinate, both
          x and y, or nothing can be provided.
        width : float,  default: 1.0
          Block width.
        label : str, optional
          Block label that can be displayed in the diagram.
        horizontalalignment : {'center', 'left', 'right'}, default: 'center'
          Determine the location of the anchor point of the block.
        label_margin: (float, float), default: (0., 0.)
            x and y margin in target coordinates of ``self.get_transform()``
            and added to the *anchor* point to set the point to draw the label.

        Other Parameters
        ----------------
        **kwargs : `.Patch` properties

          %(Patch_kwdoc)s

          linewidth: float (default=0.0)
            Set the width of the line surrounding a cluster.
        pathprops : None
          TODO: This might be not needed, it's rather unlikely to be used.
          Keyword arguments that are passed to `matplotlib.path.Path`.


        Attributes
        -----------
        x: float
          Horizontal position of the cluster anchor.
        y: float
          Vertical position of the cluster center.
        height: float
          Size of the cluster that will determine its height in the diagram.
        width: float
          Width of the cluster. In the same units as ``x``.
        label: str
          Label, id or name of the cluster.
        out_fluxes: list[:class:`~pyalluv.fluxes.Flux`]
          All outgoing fluxes of this cluster.
        """
        # TODO: only keep what's in else:
        if isinstance(height, (list, tuple)):
            self.height = len(height)
        else:
            self.height = height
        self.width = width

        self.label = label or ''
        self.label_margin = label_margin
        self.pathprops = pathprops or dict()
        self.set_horizontalalignment(horizontalalignment)
        self.anchor = anchor
        _x, _y = self._get_xy_from_anchor(anchor)
        self.set_x(_x)
        self.set_y(_y)  # this also sets *_mid_height*
        self.set_outloc()
        self.set_inloc()

        # init the in and out flows:
        self.set_outflows([])
        self.set_inflows([])
        self.in_margin = {
            'bottom': 0,
            'top': 0
        }
        self.out_margin = {
            'bottom': 0,
            'top': 0
        }

        # TODO: this is not necessarily the place to run this:
        super().__init__(**kwargs)

    def set_outflows(self, outflows):
        self._outflows = outflows

    def get_outflows(self):
        return self._outflows

    def add_outflow(self, outflow):
        self._outflows.append(outflow)

    outflows = property(get_outflows, set_outflows, doc="List of `._Flux`"
                                                        "objects leaving the"
                                                        "block.")

    def set_inflows(self, inflows):
        self._inflows = inflows

    def get_inflows(self):
        return self._inflows

    def add_inflow(self, inflow):
        self._inflows.append(inflow)

    inflows = property(get_inflows, set_inflows, doc="List of `._Flux` objects"
                                                     "entering theblock.")

    def set_horizontalalignment(self, align):
        _api.check_in_list(['center', 'left', 'right'], align=align)
        self._horizontalalignment = align

    def _get_xy_from_anchor(self, anchor):
        # set x and y (if possible)
        if isinstance(anchor, (list, tuple)):
            xa, ya = anchor
        else:
            xa, ya = anchor, None
        if xa is not None:
            xa -= 0.5 * self.width
            if self._horizontalalignment == 'left':
                xa += 0.5 * self.width
            elif self._horizontalalignment == 'right':
                xa -= 0.5 * self.width
        return xa, ya

    def set_x(self, x):
        """
        Set the horizontal position of the block.
        """
        self._x = x
        self.stale = True

    def get_x(self):
        return self._x

    x = property(get_x, set_x, doc="The x coordinate of the block")

    def set_y(self, y):
        self._y = y
        if self._y is not None:
            self._mid_height = self._y + 0.5 * self.height
            self.set_inloc()
            self.set_outloc()
        else:
            self._mid_height = None
        self.stale = True

    def get_y(self):
        if self._y is None:
            _log.warning(
                "Before accessing vertical coordinates of a block (i.e. *y* or"
                " *mid_heigth*) the block need to be distributed vertically."
            )
        else:
            return self._y

    def set_mid_height(self, mid_height):
        self._mid_height = mid_height
        if self._mid_height is not None:
            self._y = self._mid_height - 0.5 * self.height
            self.set_inloc()
            self.set_outloc()
        else:
            self._y = None
        self.stale = True

    def get_mid_height(self):
        if self._mid_height is None:
            _log.warning(
                "Before accessing vertical coordinates of a block (i.e. *y* or"
                " *mid_heigth*) the block need to be distributed vertically."
            )
        return self._mid_height

    mid_height = property(
        get_mid_height, set_mid_height,
        doc="y coordinate of the block's center."
    )

    def get_path(self):
        vertices = [
            (self.x, self._y),
            (self.x, self._y + self.height),
            (self.x + self.width, self._y + self.height),
            (self.x + self.width, self._y),
            (self.x, self._y)  # ignored as codes[-1] is CLOSEPOLY
        ]
        codes = [
            Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY
        ]
        return Path(
            vertices,
            codes,
            **self.pathprops
        )

    def get_patch(self, **kwargs):
        # TODO: This method is essentially useless if _Block is a Patch
        # NOTE: however, that Alluvial passes 'cluster_kwargs' down into this
        # when calling Alluvial.get_patchcollection, and since Alluvial should
        # define the default, 'cluster_kwargs' should be used when initiating
        # _Blocks. alternatively a Patch could be creates as 'template_block'
        # and then call slef.update_from(tempate_block) to set the styling,
        # however this would overwrite specific block styles
        _kwargs = dict(kwargs)
        _kwargs.update(self.patch_kwargs)

        self.set_outloc()
        self.set_inloc()

        # self.set_path(self.create_path())
        # return patches.PathPatch(
        #     self.get_path(),
        #     **_kwargs
        # )
        return self

    def set_loc_out_fluxes(self,):
        for out_flux in self._outflows:
            in_loc = None
            out_loc = None
            if out_flux.target_cluster is not None:
                if self.mid_height > out_flux.target_cluster.mid_height:
                    # draw to top
                    if self.mid_height >= \
                            out_flux.target_cluster.inloc['top'][1]:
                        # draw from bottom to in top
                        out_loc = 'bottom'
                        in_loc = 'top'
                    else:
                        # draw from top to top
                        out_loc = 'top'
                        in_loc = 'top'
                else:
                    # draw to bottom
                    if self.mid_height <= \
                            out_flux.target_cluster.inloc['bottom'][1]:
                        # draw from top to bottom
                        out_loc = 'top'
                        in_loc = 'bottom'
                    else:
                        # draw form bottom to bottom
                        out_loc = 'bottom'
                        in_loc = 'bottom'
            else:
                out_flux.out_loc = out_flux.out_flux_vanish
            out_flux.in_loc = in_loc
            out_flux.out_loc = out_loc

    def sort_out_fluxes(self,):
        _top_fluxes = [
            (i, self._outflows[i])
            for i in range(len(self._outflows))
            if self._outflows[i].out_loc == 'top'
        ]
        _bottom_fluxes = [
            (i, self._outflows[i])
            for i in range(len(self._outflows))
            if self._outflows[i].out_loc == 'bottom'
        ]
        if _top_fluxes:
            sorted_top_idx, _fluxes_top = zip(*sorted(
                _top_fluxes,
                key=lambda x: x[1].target_cluster.mid_height
                if x[1].target_cluster
                else -10000,
                reverse=True
            ))
        else:
            sorted_top_idx = []
        if _bottom_fluxes:
            sorted_bottom_idx, _fluxes_bottom = zip(*sorted(
                _bottom_fluxes,
                key=lambda x: x[1].target_cluster.mid_height
                if x[1].target_cluster
                else -10000,
                reverse=False
            ))
        else:
            sorted_bottom_idx = []
        sorted_idx = list(sorted_top_idx) + list(sorted_bottom_idx)
        self._outflows = [self._outflows[i] for i in sorted_idx]

    def sort_in_fluxes(self,):
        _top_fluxes = [
            (i, self._inflows[i])
            for i in range(len(self._inflows))
            if self._inflows[i].in_loc == 'top'
        ]
        _bottom_fluxes = [
            (i, self._inflows[i])
            for i in range(len(self._inflows))
            if self._inflows[i].in_loc == 'bottom'
        ]
        if _top_fluxes:
            sorted_top_idx, _fluxes_top = zip(*sorted(
                _top_fluxes,
                key=lambda x: x[1].source_cluster.mid_height
                if x[1].source_cluster
                else -10000,
                reverse=True
            ))
        else:
            sorted_top_idx = []
        if _bottom_fluxes:
            sorted_bottom_idx, _fluxes_bottom = zip(*sorted(
                _bottom_fluxes,
                key=lambda x: x[1].source_cluster.mid_height
                if x[1].source_cluster
                else -10000,
                reverse=False
            ))
        else:
            sorted_bottom_idx = []
        sorted_idx = list(sorted_top_idx) + list(sorted_bottom_idx)
        self._inflows = [self._inflows[i] for i in sorted_idx]

    def get_loc_out_flux(self, flux_width, out_loc, in_loc):
        anchor_out = (
            self.outloc[out_loc][0],
            self.outloc[out_loc][1] + self.out_margin[out_loc] + (flux_width if in_loc == 'bottom' else 0)
        )
        top_out = (
            self.outloc[out_loc][0],
            self.outloc[out_loc][1] + self.out_margin[out_loc] + (flux_width if in_loc == 'top' else 0)
        )
        self.out_margin[out_loc] += flux_width
        return anchor_out, top_out

    def set_anchor_out_fluxes(self,):
        for out_flux in self._outflows:
            out_width = out_flux.flux_width \
                if out_flux.out_loc == 'bottom' else - out_flux.flux_width
            out_flux.anchor_out, out_flux.top_out = self.get_loc_out_flux(
                out_width, out_flux.out_loc, out_flux.in_loc
            )

    def set_anchor_in_fluxes(self,):
        for in_flux in self._inflows:
            in_width = in_flux.flux_width \
                if in_flux.in_loc == 'bottom' else - in_flux.flux_width
            in_flux.anchor_in, in_flux.top_in = self.get_loc_in_flux(
                in_width, in_flux.out_loc, in_flux.in_loc
            )

    def get_loc_in_flux(self, flux_width, out_loc, in_loc):
        anchor_in = (
            self.inloc[in_loc][0],
            self.inloc[in_loc][1] + self.in_margin[in_loc] + (flux_width if out_loc == 'bottom' else 0)
        )
        top_in = (
            self.inloc[in_loc][0],
            self.inloc[in_loc][1] + self.in_margin[in_loc] + (flux_width if out_loc == 'top' else 0)
        )
        self.in_margin[in_loc] += flux_width
        return anchor_in, top_in

    def set_inloc(self,):
        if self._y is None:
            self.inloc = None
        else:
            self.inloc = {
                'bottom': (self._x, self._y),  # left, bottom
                'top': (self._x, self._y + self.height)  # left, top
            }

    def set_outloc(self,):
        if self._y is None:
            self.outloc = None
        else:
            self.outloc = {
                # right, top
                'top': (self._x + self.width, self._y + self.height),
                'bottom': (self._x + self.width, self._y)  # right, bottom
            }


class _Flux(object):
    r"""

    Parameters
    -----------
    relative_flux: bool
      If ``True`` the fraction of the height of parameter `source_cluster`
      is taken, if the source_cluster is none, then the
      relative height form the target_cluster is taken.
    source_cluster: :class:`pyalluv.clusters.Cluster` (default=None)
      Cluster from which the flux originates.
    target_cluster: :class:`pyalluv.clusters.Cluster` (default=None)
      Cluster into which the flux leads.
    \**kwargs optional parameter:
      interpolation_steps:

      out_flux_vanish: str (default='top')

      default_fc: (default='gray')

      default_ec: (default='gray')

      default_alpha: int (default=0.3)

      closed
      readonly
      facecolors
      edgecolors
      linewidths
      linestyles
      antialiaseds

    Attributes
    -----------

    flux: float
      The size of the flux which will translate to the height of the flux in
      the Alluvial diagram.
    source_cluster: :class:`pyalluv.clusters.Cluster` (default=None)
      Cluster from which the flux originates.
    target_cluster: :class:`pyalluv.clusters.Cluster` (default=None)
      Cluster into which the flux leads.
    """
    def __init__(
            self, flux,
            source_cluster=None, target_cluster=None,
            relative_flux=False,
            **kwargs):
        self._interp_steps = kwargs.pop('interpolation_steps', 1)
        self.out_flux_vanish = kwargs.pop('out_flux_vanish', 'top')
        self.default_fc = kwargs.pop('default_fc', 'gray')
        self.default_ec = kwargs.pop('default_ec', 'gray')
        self.default_alpha = kwargs.pop('default_alpha', 0.3)
        # self.default_alpha = kwargs.pop(
        #         'default_alpha',
        #         kwargs.get('alpha', {}).pop('default', 0.3)
        #         )
        self.closed = kwargs.pop('closed', False)
        self.readonly = kwargs.pop('readonly', False)
        self.patch_kwargs = kwargs
        self.patch_kwargs['lw'] = self.patch_kwargs.pop(
            'linewidth', self.patch_kwargs.pop('lw', 0.0)
        )

        if isinstance(flux, (list, tuple)):
            self.flux = len(flux)
        else:
            self.flux = flux
        self.relative_flux = relative_flux
        self.source_cluster = source_cluster
        self.target_cluster = target_cluster
        if self.source_cluster is not None:
            if self.relative_flux:
                self.flux_width = self.flux * self.source_cluster.height
            else:
                self.flux_width = self.flux
        else:
            if self.target_cluster is not None:
                if self.relative_flux:
                    self.flux_width = self.flux * self.target_cluster.height
                else:
                    self.flux_width = self.flux
        # append the flux to the clusters
        if self.source_cluster is not None:
            self.source_cluster.add_outflow(self)
        if self.target_cluster is not None:
            self.target_cluster.add_inflow(self)

    def get_patch(self, **kwargs):
        _kwargs = dict(kwargs)
        _to_in_kwargs = {}
        _to_out_kwargs = {}
        for kw in _kwargs:
            if kw.startswith('in_'):
                _to_in_kwargs[kw[3:]] = _kwargs.pop(kw)
            elif kw.startswith('out_'):
                _to_out_kwargs[kw[3:]] = _kwargs.pop(kw)
        # update with Flux specific styling
        _kwargs.update(self.patch_kwargs)
        for _color in ['facecolor', 'edgecolor']:
            _set_color = _kwargs.pop(_color, None)
            _set_alpha = _kwargs.pop('alpha', None)
            if isinstance(_set_alpha, (int, float)):
                _kwargs['alpha'] = _set_alpha
                _set_alpha = None
            color_is_set = False
            if _set_color == 'source_cluster' or _set_color == 'cluster':
                from_cluster = self.source_cluster
                color_is_set = True
            elif _set_color == 'target_cluster':
                from_cluster = self.target_cluster
                color_is_set = True
            elif isinstance(_set_color, str) and '__' in _set_color:
                which_cluster, flow_type = _set_color.split('__')
                if which_cluster == 'target_cluster':
                    from_cluster = self.target_cluster
                else:
                    from_cluster = self.source_cluster
                if flow_type == 'migration' \
                        and self.source_cluster.patch_kwargs.get(_color) \
                        != self.target_cluster.patch_kwargs.get(_color):
                    color_is_set = True
                    if _set_alpha:
                        _kwargs['alpha'] = _set_alpha.get(
                            'migration', _set_alpha.get('default', self.default_alpha)
                        )
                elif flow_type == 'reside'  \
                        and self.source_cluster.patch_kwargs.get(_color) \
                        == self.target_cluster.patch_kwargs.get(_color):
                    color_is_set = True
                    if _set_alpha:
                        _kwargs['alpha'] = _set_alpha.get(
                            'reside', _set_alpha.get('default', self.default_alpha)
                        )
                else:
                    _set_color = None
            if color_is_set:
                _kwargs[_color] = from_cluster.patch_kwargs.get(
                    _color, None
                )

            # set it back
            else:
                _kwargs[_color] = _set_color
                if _set_color is None:
                    if _color == 'facecolor':
                        _kwargs[_color] = self.default_fc
                    elif _color == 'edgecolor':
                        _kwargs[_color] = self.default_ec
                if _set_alpha:
                    _kwargs['alpha'] = _set_alpha.get('default', self.default_alpha)
        # line below is probably not needed as alpha is set with the color
        _kwargs['alpha'] = _kwargs.get('alpha', self.default_alpha)
        # set in/out only flux styling
        _in_kwargs = dict(_kwargs)
        _in_kwargs.update(_to_in_kwargs)
        _out_kwargs = dict(_kwargs)
        _out_kwargs.update(_to_out_kwargs)

        _dist = None
        if self.out_loc is not None:
            if self.in_loc is not None:
                _dist = 2 / 3 * (
                    self.target_cluster.in_['bottom'][0] - self.source_cluster.outloc['bottom'][0]
                )
            else:
                _dist = 2 * self.source_cluster.width
                _kwargs = _out_kwargs
        else:
            if self.in_loc is not None:
                _kwargs = _in_kwargs
            else:
                raise Exception('Flux with neither source nor target cluster')

        # now complete the path points
        if self.anchor_out is not None:
            anchor_out_inner = (
                self.anchor_out[0] - 0.5 * self.source_cluster.width,
                self.anchor_out[1]
            )
            dir_out_anchor = (self.anchor_out[0] + _dist, self.anchor_out[1])
        else:
            # TODO set to form vanishing flux
            # anchor_out = anchor_out_inner =
            # dir_out_anchor =
            pass
        if self.top_out is not None:
            top_out_inner = (
                self.top_out[0] - 0.5 * self.source_cluster.width,
                self.top_out[1]
            )
            # 2nd point 2/3 of distance between clusters
            dir_out_top = (self.top_out[0] + _dist, self.top_out[1])
        else:
            # TODO set to form vanishing flux
            # top_out = top_out_inner =
            # dir_out_top =
            pass
        if self.anchor_in is not None:
            anchor_in_inner = (
                self.anchor_in[0] + 0.5 * self.target_cluster.width,
                self.anchor_in[1]
            )
            dir_in_anchor = (self.anchor_in[0] - _dist, self.anchor_in[1])
        else:
            # TODO set to form new in flux
            # anchor_in = anchor_in_inner =
            # dir_in_anchor =
            pass
        if self.top_in is not None:
            top_in_inner = (
                self.top_in[0] + 0.5 * self.target_cluster.width,
                self.top_in[1]
            )
            dir_in_top = (self.top_in[0] - _dist, self.top_in[1])
        else:
            # TODO set to form new in flux
            # top_in = top_in_inner =
            # dir_in_top =
            pass

        vertices = [
            self.anchor_out,
            dir_out_anchor, dir_in_anchor, self.anchor_in,
            anchor_in_inner, top_in_inner, self.top_in,
            dir_in_top, dir_out_top, self.top_out,
            top_out_inner, anchor_out_inner,
            self.anchor_out
        ]
        codes = [
            Path.MOVETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.LINETO, Path.LINETO, Path.LINETO,
            Path.CURVE4, Path.CURVE4, Path.CURVE4,
            Path.LINETO, Path.LINETO,
            Path.CLOSEPOLY
        ]
        _path = Path(vertices, codes, self._interp_steps, self.closed, self.readonly)

        flux_patch = patches.PathPatch(_path, **_kwargs)
        return flux_patch


class Alluvial:
    """
    Alluvial diagram.

        Alluvial diagrams are a variant of flow diagram designed to represent
        changes in classifications, in particular changes in network
        structure over time.
        `Wikipedia (23/1/2021) <https://en.wikipedia.org/wiki/Alluvial_diagram>`_
    """
    @docstring.dedent_interpd
    def __init__(
        self, clusters, ax=None, y_pos='overwrite', cluster_w_spacing=1,
        blockprops=None, flux_kwargs={}, label_kwargs={},
        **kwargs
    ):
        """
        Create a new Alluvial instance.


        Parameters
        ===========

        clusters: dict[str, dict], dict[float, list] or list[list]
          You have 2 options to create an Alluvial diagram::

          raw data: dict[str, dict]
            *NOT IMPLEMENTED YET*

            Provide for each cluster (`key`) a dictionary specifying the
            out-fluxes in the form of a dictionary (`key`: cluster, `value`: flux).

            .. note::

              The `key` ``None`` can hold a dictionary specifying fluxes from/to
              outside the system. If is present in the provided dictionary it
              allows to specify in-fluxes, i.e. data source that were not present
              at the previous slice.

              If it is present in the out-fluxes of a cluster, the specified amount
              simply vanishes and will not lead to a flux.

          collections of :obj:`.Cluster`: dict[float, list] and list[list]
            If a `list` is provided each element must be a `list`
            of :obj:`.Cluster` objects. A `dictionary` must provide a `list` of
            :obj:`.Cluster` (*value*) for a horizontal position (*key*), e.g.
            ``{1.0: [c11, c12, ...], 2.0: [c21, c22, ...], ...}``.

        ax: `~.axes.Axes`
          Axes onto which the Alluvial diagram should be drawn.
          If *ax* is not provided a new Axes instance will be created.
        y_pos: str
          **options:** ``'overwrite'``, ``'keep'``, ``'complement'``, ``'sorted'``

          'overwrite':
             Ignore existing y coordinates for a cluster and set the vertical
             position to minimize the vertical displacements of all fluxes.
          'keep':
            use the cluster's :attr:`~pyalluv.clusters.Cluster.y`. If a
            cluster has no y position set this raises an exception.
          'complement':
            use the cluster's :attr:`~pyalluv.clusters.Cluster.y` if
            set. Cluster without y position are positioned relative to the other
            clusters by minimizing the vertical displacements of all fluxes.
          'sorted':
            NOT IMPLEMENTED YET
        cluster_w_spacing: float, int (default=1)
          Vertical spacing between clusters
        blockprops : dict, optional
          The properties used to draw the blocks. *blockprops* accepts the
          following specific keyword arguments:

          Any further arguments provided are passed to
          `matplotlib.patches.PathPatch`.

          Note that *blockprops* sets the properties of all sub-diagrams,
          unless specific properties are provided when a sub-diagram is added
          (see :meth:`add` for details), or :meth:`set_blockprops` is called
          before adding further sub-diagrams.

          TODO: specify particular blockprops kw's
          `facecolor`, `edgecolor`, `alpha`, `linewidth`, ...
          ==========   ======================================================
          Key          Description
          ==========   ======================================================
          width        The width of the arrow in points
          headwidth    The width of the base of the arrow head in points
          headlength   The length of the arrow head in points
          shrink       Fraction of total length to shrink from both ends
          ?            Any key to :class:`matplotlib.patches.FancyArrowPatch`
          ==========   ======================================================

          For a list of available options see
          :class:`matplotlib.patches.PathPatch`
          TODO: or this:
          %(Patch_kwdoc)s

        flux_kwargs: dict (default={})
          dictionary styling the :obj:`~matplotlib.patches.PathPatch` of fluxes.

          for a list of available options see
          :class:`~matplotlib.patches.PathPatch`

          Note
          -----

            Passing a string to `facecolor` and/or `edgecolor` allows to color
            fluxes relative to the color of their source or target clusters.

            ``'source_cluster'`` or ``'target_cluster'``:
              will set the facecolor equal to the color of the respective cluster.

              ``'cluster'`` *and* ``'source_cluster'`` *are equivalent.*

            ``'<cluster>_reside'`` or ``'<cluster>_migration'``:
              set the color based on whether source and target cluster have the
              same color or not. ``'<cluster>'`` should be either
              ``'source_cluster'`` or ``'target_cluster'`` and determines the
              cluster from which the color is taken.

              **Examples:**

              ``facecolor='cluster_reside'``
                set `facecolor` to the color of the source cluster if both source
                and target cluster are of the same color.

              ``edgecolor='cluster_migration'``
                set `edgecolor` to the color of the source cluster if source and
                target cluster are of different colors.

        **kwargs optional parameter:
            x_lim: tuple
              the horizontal limit values for the :class:`~matplotlib.axes.Axes`.
            y_lim: tuple
              the vertical limit values for the :class:`~matplotlib.axes.Axes`.
            set_x_pos: bool
              if clusters is a dict then the key is set for all clusters
            cluster_width: float
              (NOT IMPLEMENTED) overwrites width of all clusters
            format_xaxis: bool (default=True)
              If set to `True` the axes is formatted according to the data
              provided. For now, this is only relevant if the horizontal positions
              are :class:`~datetime.datetime` objects.
              See :meth:`~.Alluvial.set_dates_xaxis` for further informations.
            x_axis_offset: float
              how much space (relative to total height)
              should be reserved for the x_axis. If set to 0.0, then
              the x labels will not be visible.
            fill_figure: bool
              indicating whether or not set the
              axis dimension to fill up the entire figure
            invisible_x/invisible_y: bool
              whether or not to draw these axis.
            y_fix: dict
              with x_pos as keys and a list of tuples
              (cluster labels) as values. The position of clusters (tuples)
              are swapped.
            redistribute_vertically: int (default=4)
              how often the vertical pairwise swapping of clusters at a given time
              point should be performed.
            y_offset: float
              offsets the vertical position of each cluster by this amount.

              .. note::

                This ca be used to draw multiple alluvial diagrams on the same
                :obj:`~matplotlib.axes.Axes` by simply calling
                :class:`~.Alluvial` repeatedly with changing offset value, thus
                stacking alluvial diagrams.

        Attributes
        ===========

        clusters: dict
          Holds for each vertical position a list of :obj:`.Cluster` objects.
        """
        self._diagrams = []
        self._extouts = []
        self._diagc = 0
        self._dlabels = []
        self._dirty = False  # indicate if between diagram flows exist

        # create axes if not provided
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            # TODO: not sure if specifying the ticks is necessary
            ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        self.ax = ax

        # store the inputs
        self.y_pos = y_pos
        self.cluster_w_spacing = cluster_w_spacing
        self._cluster_kwargs = blockprops
        self._flux_kwargs = flux_kwargs
        self._label_kwargs = label_kwargs

        # if args are provided, call add and finish()
        if kwargs:
            self.add(**kwargs)
            self.finish()

        # if clusters are given in a list of lists (each list is a x position)
        self._set_x_pos = kwargs.get('set_x_pos', True)
        self._redistribute_vertically = kwargs.get(
            'redistribute_vertically',
            4
        )
        self.with_cluster_labels = kwargs.get('with_cluster_labels', True)
        self.format_xaxis = kwargs.get('format_xaxis', True)
        self._x_axis_offset = kwargs.get('x_axis_offset', 0.0)
        self._fill_figure = kwargs.get('fill_figure', False)
        self._invisible_y = kwargs.get('invisible_y', True)
        self._invisible_x = kwargs.get('invisible_x', False)
        self.y_offset = kwargs.get('y_offset', 0)
        self.y_fix = kwargs.get('y_fix', None)
        if isinstance(clusters, dict):
            self.clusters = clusters
        else:
            self.clusters = {}
            for cluster in clusters:
                try:
                    self.clusters[cluster.x].append(cluster)
                except KeyError:
                    self.clusters[cluster.x] = [cluster]
        self.x_positions = sorted(self.clusters.keys())
        # set the x positions correctly for the clusters
        if self._set_x_pos:
            for x_pos in self.x_positions:
                for cluster in self.clusters[x_pos]:
                    cluster = cluster.set_x_pos(x_pos)
        self._x_dates = False
        _minor_tick = 'months'
        cluster_widths = []
        if isinstance(self.x_positions[0], datetime):
            # assign date locator/formatter to the x-axis to get proper labels
            if self.format_xaxis:
                locator = mdates.AutoDateLocator(minticks=3)
                formatter = mdates.AutoDateFormatter(locator)
                self.ax.xaxis.set_major_locator(locator)
                self.ax.xaxis.set_major_formatter(formatter)
            self._x_dates = True
            if (self.x_positions[-1] - self.x_positions[0]).days < 2 * 30:
                _minor_tick = 'weeks'
            self.clusters = {
                mdates.date2num(x_pos): self.clusters[x_pos]
                for x_pos in self.x_positions
            }
            self.x_positions = sorted(self.clusters.keys())
            for x_pos in self.x_positions:
                for cluster in self.clusters[x_pos]:
                    # in days (same as mdates.date2num)
                    cluster.width = cluster.width.total_seconds() / 60 / 60 / 24
                    cluster_widths.append(cluster.width)
                    if cluster.label_margin is not None:
                        _h_margin = cluster.label_margin[0].total_seconds() / 60 / 60 / 24
                        cluster.label_margin = (
                            _h_margin, cluster.label_margin[1]
                        )
                    cluster.set_x_pos(mdates.date2num(cluster.x))

        # TODO: set the cluster.width property
        else:
            for x_pos in self.x_positions:
                for cluster in self.clusters[x_pos]:
                    cluster_widths.append(cluster.width)
        self.cluster_width = kwargs.get('cluster_width', None)
        self.x_lim = kwargs.get(
            'x_lim',
            (
                self.x_positions[0] - 2 * min(cluster_widths),
                # - 2 * self.clusters[self.x_positions[0]][0].width,
                self.x_positions[-1] + 2 * min(cluster_widths),
                # + 2 * self.clusters[self.x_positions[-1]][0].width,
            )
        )
        self.y_min, self.y_max = None, None
        if self.y_pos == 'overwrite':
            # reset the vertical positions for each row
            for x_pos in self.x_positions:
                self.distribute_clusters(x_pos)
            for x_pos in self.x_positions:
                self._move_new_clusters(x_pos)
            for x_pos in self.x_positions:
                nbr_clusters = len(self.clusters[x_pos])
                for _ in range(nbr_clusters):
                    for i in range(1, nbr_clusters):
                        n1 = self.clusters[x_pos][nbr_clusters - i - 1]
                        n2 = self.clusters[x_pos][nbr_clusters - i]
                        if self._swap_clusters(n1, n2, 'forwards'):
                            n2.y = n1.y
                            n1.y = n2.y + n2.height + self.cluster_w_spacing
                            self.clusters[x_pos][nbr_clusters - i] = n1
                            self.clusters[x_pos][nbr_clusters - i - 1] = n2
        else:
            # TODO: keep and complement
            pass
        if isinstance(self.y_fix, dict):
            # TODO: allow to directly get the index given the cluster label
            for x_pos in self.y_fix:
                for st in self.y_fix[x_pos]:
                    n1_idx, n2_idx = (
                        i for i, l in enumerate(
                            map(lambda x: x.label, self.clusters[x_pos])
                        )
                        if l in st
                    )
                    self.clusters[x_pos][n1_idx], self.clusters[x_pos][n2_idx] = self.clusters[
                        x_pos
                    ][n2_idx], self.clusters[x_pos][n1_idx]
                    self._distribute_column(x_pos, self.cluster_w_spacing)

        # positions are set
        self.y_lim = kwargs.get('y_lim', (self.y_min, self.y_max))
        # set the colors
        # TODO

        # now draw
        patch_collection = self.get_patchcollection(
            cluster_kwargs=self._cluster_kwargs,
            flux_kwargs=self._flux_kwargs
        )
        self.ax.add_collection(patch_collection)
        if self.with_cluster_labels:
            label_collection = self.get_labelcollection(**self._label_kwargs)
            if label_collection:
                for label in label_collection:
                    self.ax.annotate(**label)
        self.ax.set_xlim(
            *self.x_lim
        )
        self.ax.set_ylim(
            *self.y_lim
        )
        if self._fill_figure:
            self.ax.set_position(
                [
                    0.0,
                    self._x_axis_offset,
                    0.99,
                    1.0 - self._x_axis_offset
                ]
            )
        if self._invisible_y:
            self.ax.get_yaxis().set_visible(False)
        if self._invisible_x:
            self.ax.get_xaxis().set_visible(False)
        self.ax.spines['right'].set_color('none')
        self.ax.spines['left'].set_color('none')
        self.ax.spines['top'].set_color('none')
        self.ax.spines['bottom'].set_color('none')
        if isinstance(self.x_positions[0], datetime) and self.format_xaxis:
            self.set_dates_xaxis(_minor_tick)

    @property
    def clusters(self):
        # TODO create clusters dict based on _diagrams, dlabels and cluster
        # labels
        _clusters = None
        return _clusters

    def _to_valid_sequence(self, data, attribute):
        try:
            data = np.asfarray(data)
        except ValueError:
            try:
                _data = iter(data)
            except TypeError:
                raise TypeError("'{attr}' must be an iterable sequence and"
                                " should be of type list or numpy.ndarray,"
                                " '{ftype}' is not supported."
                                .format(attr=attribute, ftype=type(data)))
            else:
                data = []
                for i, d in enumerate(_data):
                    try:
                        data.append(np.asfarray(d))
                    except ValueError:
                        raise ValueError("{attr} can only contain array-like"
                                         " objects, which is not the case"
                                         " for entry {entry} in the provided"
                                         " argument."
                                         .format(attr=attribute, entry=i))
        finally:
            return data

    def add(self, flows, ext=None, extout=None, label=None, yoff=0,
            fractionflow=False, **kwargs):
        r"""
        Add an Alluvial diagram with a vertical offset.
        The offset must be provided in the same units as the block sizes.

        Parameters
        ----------
        flows : sequence of array-like objects
            The flows between columns of the Alluvial diagram.

            *flows[i]* determines the flow matrix from blocks in column
            *i* to the blocks in column *i+1*.

            Note that an Alluvial diagram with M columns needs *flows* to be
            a sequence of M-1 array-like objects.
        ext : sequence, optional
            External inflow to the Alluvial diagram. Supported formats are:

            - sequence of M array-like objects: Specify for each of the M
              columns in the diagram the inflows to the blocks.
            - sequence of floats: Set the block sizes in the initial column.

            If *ext* is not provided, the block sizes for the initial columns
            are inferred from the first entry in *flows* as the column-wise sum
            of *flows[0]*.
        extout : iterable, optional
            The outflows to blocks belonging to another Alluvial diagram.
            The values provided in *extout* must be of the shape (M-1, N, P),
            with N the number of blocks in the diagram that is added and P the
            number of blocks in the destination diagram to which the flows will
            be directed.

            If a dictionary is provided a key must specify the destination
            diagram. Allowed is either the label, *dlabel*, or the index of a
            diagram.

            If a list is provided the entries are mapped to diagrams by index.
        label : string, optional
            The label of the diagram to add.
        fractionflow : bool, default: False
            When set to *False* (the default) the values in *flows* are
            considered to be absolute values.

            If set to *True* the values in *flows* are considered to be
            fractions of block sizes, and the actual flow between columns *i*
            and *i+1* is given by the dot product of *flows[i]* and the array
            of block sizes in column *i*.

            If fractions are provided,  you must set *ext* to provide at least
            the block sizes for the initial column of the Alluvial diagram.
        yoff : int or float, default: 0
            A constant vertical offset applied to the added diagram.


        Notes
        -----
        The procedure to set the block sizes of column *i+1* changes depending
        on whether *flows* provides fractions of block sizes (if *fractionflow*
        is set to True) or absolute values (default, *fractionflow* is False).
        For a column *i* with N blocks and the column *i+1* with P blocks, the
        relation is defined as as follows:

        - *fractionflow* is False:

          The block sizes in column *i+1* are given by:

          .. math::
              \textbf{c}_{i+1} = \mathbf{F}_i\cdot\textbf{1}+\textbf{e}_{i+1},

          where :math:`\mathbf{F}_i` is the flow matrix of shape (P, N), given
          by *flux[i]*, :math:`\textbf{1}` is a vector of ones of shape (N) and
          :math:`\textbf{e}_{i+1}` is the external influx vector of shape (P),
          given by *e[i+1]*.
        - *fractionflow* is True:

          The block sizes in column *i+1* depend directly on the block sizes of
          column *i*, :math:`\textbf{c}_{i}`, and are given by:

          .. math::
              \textbf{c}_{i+1}=\mathbf{F}_i\cdot\textbf{c}_i+\textbf{e}_{i+1},

          where :math:`\mathbf{F}_i` is the flow matrix of shape (P, N), given
          by *flux[i]*, :math:`\textbf{c}_i` the vector of N block sizes in
          column *i* and :math:`\textbf{e}_{i+1}` the external influx vector of
          shape (P) given by *e[i+1]*.
        """
        # check the provided arguments
        flows = self._to_valid_sequence(flows, 'flows')
        nbr_cols = len(flows) + 1
        # check ext and set initial column
        if ext is None:
            if fractionflow:
                raise TypeError("'ext' cannot be None if 'fractionflow' is"
                                " True: You need to provide at least the block"
                                " sizes for the first column of the Alluvial"
                                " diagram if the flows are given as"
                                " fractions.")
            ext = np.zeros(nbr_cols)
            _cinit = flows[0].sum(0)
        else:
            ext = self._to_valid_sequence(ext, 'ext')
            if isinstance(ext[0], np.ndarray):
                _cinit = ext[0]
            else:
                _cinit = ext[:]
                ext = np.zeros(nbr_cols)

        # create the columns
        _columns = [_cinit]
        for flow, e in zip(flows, ext[1:]):
            if not fractionflow:
                _col = flow.sum(1) + e
            else:
                _col = flow.dot(_columns[-1]) + e
            _columns.append(_col)

        if extout is not None:
            # check extout format
            pass

        # add the new diagram
        self._diagrams.append(_columns)
        self._dlabels.append(label or f'diagram-{self._diagc}')
        self._extouts.append(extout)
        self._diagc += 1

        # Create the sequence of clusterings
        time_points = [0, 4, 9, 14, 18.2]
        # Define the cluster sizes per snapshot
        # at each time point {cluster_id: cluster_size})
        cluster_sizes = [{0: 3}, {0: 5}, {0: 3, 1: 2}, {0: 5}, {0: 4}]
        # Define the membership fluxes between neighbouring clusterings
        between_fluxes = [
            {(0, 0): 3},  # key: (from cluster, to cluster), value: size
            {(0, 0): 3, (0, 1): 2},
            {(0, 0): 3, (1, 0): 2},
            {(0, 0): 4}
        ]
        # set the colors
        cluster_color = {0: "C1", 1: "C2"}
        # create a dictionary with the time points as keys and a list of clusters
        # as values
        clustering_sequence = {}
        for tp, clustering in enumerate(cluster_sizes):
            clustering_sequence[time_points[tp]] = [
                _Cluster(
                    height=clustering[cid],
                    label="{0}".format(cid),
                    facecolor=cluster_color[cid],
                ) for cid in clustering
            ]
        # now create the fluxes between the clusters
        for tidx, tp in enumerate(time_points[1:]):
            fluxes = between_fluxes[tidx]
            for from_csid, to_csid in fluxes:
                _Flux(
                    flux=fluxes[(from_csid, to_csid)],
                    source_cluster=clustering_sequence[time_points[tidx]][from_csid],
                    target_cluster=clustering_sequence[tp][to_csid],
                    facecolor='source_cluster'
                )

    def distribute_clusters(self, x_pos):
        r"""
        Distribute the clusters for a given x_position vertically

        Parameters
        -----------
        x_pos: float
          The horizontal position at which the clusters should be distributed.
          This must be a `key` of the :attr:`~.Alluvial.clusters`
          attribute.
        """
        nbr_clusters = len(self.clusters[x_pos])
        if nbr_clusters:
            # sort clusters according to height
            _clusters = sorted(self.clusters[x_pos], key=lambda x: x.height)
            # sort so to put biggest height in the middle
            self.clusters[x_pos] = _clusters[::-2][::-1] + \
                _clusters[nbr_clusters % 2::2][::-1]
            # set positioning
            self._distribute_column(x_pos, self.cluster_w_spacing)
            # now sort again considering the fluxes.
            old_mid_heights = [
                cluster.mid_height for cluster in self.clusters[x_pos]
            ]
            # do the redistribution 4 times
            _redistribute = False
            for _ in range(self._redistribute_vertically):
                for cluster in self.clusters[x_pos]:
                    weights = []
                    positions = []
                    for in_flux in cluster.inflows:
                        if in_flux.source_cluster is not None:
                            weights.append(in_flux.flux_width)
                            positions.append(in_flux.source_cluster.mid_height)
                    if sum(weights) > 0.0:
                        _redistribute = True
                        cluster.mid_height = sum([
                            weights[i] * positions[i]
                            for i in range(len(weights))
                        ]) / sum(weights)
                if _redistribute:
                    sort_key = [
                        bisect_left(
                            old_mid_heights, self.clusters[x_pos][i].mid_height
                        ) for i in range(nbr_clusters)
                    ]
                    cs, _sort_key = zip(
                        *sorted(
                            zip(
                                list(range(nbr_clusters)),
                                sort_key,
                            ),
                            key=lambda x: x[1]
                        )
                    )
                    self.clusters[x_pos] = [
                        self.clusters[x_pos][_k] for _k in cs
                    ]
                    # redistribute them
                    self._distribute_column(x_pos, self.cluster_w_spacing)
                    old_mid_heights = [
                        cluster.mid_height for cluster in self.clusters[x_pos]
                    ]
                else:
                    break
            # perform pairwise swapping for backwards fluxes
            for _ in range(int(0.5 * nbr_clusters)):
                for i in range(1, nbr_clusters):
                    n1 = self.clusters[x_pos][i - 1]
                    n2 = self.clusters[x_pos][i]
                    if self._swap_clusters(n1, n2, 'backwards'):
                        n2.y = n1.y
                        n1.y = n2.y + n2.height + self.cluster_w_spacing
                        self.clusters[x_pos][i - 1] = n2
                        self.clusters[x_pos][i] = n1
            for _ in range(int(0.5 * nbr_clusters)):
                for i in range(1, nbr_clusters):
                    n1 = self.clusters[x_pos][nbr_clusters - i - 1]
                    n2 = self.clusters[x_pos][nbr_clusters - i]
                    if self._swap_clusters(n1, n2, 'backwards'):
                        n2.y = n1.y
                        n1.y = n2.y + n2.height + self.cluster_w_spacing
                        self.clusters[x_pos][nbr_clusters - i - 1] = n2
                        self.clusters[x_pos][nbr_clusters - i] = n1

            _min_y = min(
                self.clusters[x_pos], key=lambda x: x.y
            ).y - 2 * self.cluster_w_spacing
            _max_y_cluster = max(
                self.clusters[x_pos],
                key=lambda x: x.y + x.height
            )
            _max_y = _max_y_cluster.y + \
                _max_y_cluster.height + 2 * self.cluster_w_spacing
            self.y_min = min(
                self.y_min,
                _min_y
            ) if self.y_min is not None else _min_y
            self.y_max = max(
                self.y_max,
                _max_y
            ) if self.y_max is not None else _max_y

    def set_dates_xaxis(self, resolution='months'):
        r"""
        Format the x axis in case :class:`~datetime.datetime` objects are
        provide for the horizontal placement of clusters.

        Parameters
        -----------
        resolution: str (default='months')
          Possible values are ``'months'`` and ``'weeks'``.
          This determines the resolution of the minor ticks via
          :obj:`~matplotlib.axis.Axis.set_minor_formatter`.
          The major tick is then either given in years or months.

          .. todo::

            Include further options or allow passing parameters directly to
            :meth:`~matplotlib.axis.Axis.set_minor_formatter` and
            :meth:`~matplotlib.axis.Axis.set_major_formatter`.

        """
        import matplotlib.dates as mdates
        years = mdates.YearLocator()
        months = mdates.MonthLocator()
        weeks = mdates.WeekdayLocator(mdates.MONDAY)
        if resolution == 'months':
            monthsFmt = mdates.DateFormatter('%b')
            yearsFmt = mdates.DateFormatter('\n%Y')  # add space
            self.ax.xaxis.set_minor_locator(months)
            self.ax.xaxis.set_minor_formatter(monthsFmt)
            self.ax.xaxis.set_major_locator(years)
            self.ax.xaxis.set_major_formatter(yearsFmt)
        elif resolution == 'weeks':
            monthsFmt = mdates.DateFormatter('\n%b')
            weeksFmt = mdates.DateFormatter('%b %d')
            self.ax.xaxis.set_minor_locator(weeks)
            self.ax.xaxis.set_minor_formatter(weeksFmt)
            self.ax.xaxis.set_major_locator(months)
            self.ax.xaxis.set_major_formatter(monthsFmt)

    def _swap_clusters(self, n1, n2, direction='backwards'):
        squared_diff = {}
        for cluster in [n1, n2]:
            weights = []
            sqdiff = []
            if direction in ['both', 'backwards']:
                for in_flux in cluster.inflows:
                    if in_flux.source_cluster is not None:
                        weights.append(in_flux.flux_width)
                        sqdiff.append(abs(
                            cluster.mid_height - in_flux.source_cluster.mid_height
                        ))
            if direction in ['both', 'forwards']:
                for out_flux in cluster.outflows:
                    if out_flux.target_cluster is not None:
                        weights.append(out_flux.flux_width)
                        sqdiff.append(abs(
                            cluster.mid_height - out_flux.target_cluster.mid_height
                        ))
            if sum(weights) > 0.0:
                squared_diff[cluster] = sum(
                    [weights[i] * sqdiff[i]
                        for i in range(len(weights))]
                ) / sum(weights)
        # inverse order and check again
        assert n1.y < n2.y
        inv_mid_height = {
            n1: n2.y + n2.height + self.cluster_w_spacing + 0.5 * n1.height,
            n2: n1.y + 0.5 * n2.height
        }
        squared_diff_inf = {}
        for cluster in [n1, n2]:
            weights = []
            sqdiff = []
            if direction in ['both', 'backwards']:
                for in_flux in cluster.inflows:
                    if in_flux.source_cluster is not None:
                        weights.append(in_flux.flux_width)
                        sqdiff.append(abs(
                            inv_mid_height[cluster] - in_flux.source_cluster.mid_height
                        ))
            if direction in ['both', 'forwards']:
                for out_flux in cluster.outflows:
                    if out_flux.target_cluster is not None:
                        weights.append(out_flux.flux_width)
                        sqdiff.append(abs(
                            inv_mid_height[cluster] - out_flux.target_cluster.mid_height
                        ))
            if sum(weights) > 0.0:
                squared_diff_inf[cluster] = sum([
                    weights[i] * sqdiff[i]
                    for i in range(len(weights))
                ]) / sum(weights)
        if sum(squared_diff.values()) > sum(squared_diff_inf.values()):
            return True
        else:
            return False

    def _move_new_clusters(self, x_pos):
        r"""
        This method redistributes fluxes without in-flux so to minimize the
        vertical displacement of out-fluxes.

        Parameters
        -----------
        x_pos: float
          The horizontal position where new clusters without in-flux should be
          distributed. This must be a `key` of the
          :attr:`~.Alluvial.clusters` attribute.

        Once the clusters are distributed for all x positions this method
        redistributes within a given x_positions the clusters that have no
        influx but out fluxes. The clusters are moved closer (vertically) to
        the target clusters of the out flux(es).
        """
        old_mid_heights = [
            cluster.mid_height for cluster in self.clusters[x_pos]
        ]
        _redistribute = False
        for cluster in self.clusters[x_pos]:
            if sum([_flux.flux_width for _flux in cluster.inflows]) == 0.0:
                weights = []
                positions = []
                for out_flux in cluster.outflows:
                    if out_flux.target_cluster is not None:
                        weights.append(out_flux.flux_width)
                        positions.append(out_flux.target_cluster.mid_height)
                if sum(weights) > 0.0:
                    _redistribute = True
                    cluster.mid_height = sum(
                        [weights[i] * positions[i] for i in range(len(weights))]
                    ) / sum(weights)
        if _redistribute:
            sort_key = [
                bisect_left(
                    old_mid_heights, self.clusters[x_pos][i].mid_height
                ) for i in range(len(self.clusters[x_pos]))
            ]
            cs, _sort_key = zip(
                *sorted(
                    zip(
                        list(range(len(self.clusters[x_pos]))),
                        sort_key,
                    ),
                    key=lambda x: x[1]
                )
            )
            self.clusters[x_pos] = [self.clusters[x_pos][_k] for _k in cs]
            # redistribute them
            self._distribute_column(x_pos, self.cluster_w_spacing)

    def get_patchcollection(
        self, match_original=True,
        cluster_kwargs={},
        flux_kwargs={},
        *args, **kwargs
    ):
        """
        Gather the patchcollection to add to the axes

        Parameter:
        ----------
        :param kwargs:
            Options:
        """
        cluster_patches = []
        fluxes = []
        if cluster_kwargs is None:
            cluster_kwargs = dict()
        for x_pos in self.x_positions:
            out_fluxes = []
            for cluster in self.clusters[x_pos]:
                # TODO: set color
                # _cluster_color
                cluster.y = cluster.y + self.y_offset
                cluster_patches.append(
                    cluster.get_patch(
                        **cluster_kwargs
                    )
                )
                # sort the fluxes for minimal overlap
                cluster.set_loc_out_fluxes()
                cluster.sort_in_fluxes()
                cluster.sort_out_fluxes()
                cluster.set_anchor_in_fluxes()
                cluster.set_anchor_out_fluxes()
                out_fluxes.extend(cluster.outflows)
            fluxes.append(out_fluxes)
        flux_patches = []
        for out_fluxes in fluxes:
            for out_flux in out_fluxes:
                flux_patches.append(out_flux.get_patch(**flux_kwargs))
        all_patches = []
        all_patches.extend(flux_patches)
        all_patches.extend(cluster_patches)
        return PatchCollection(
            all_patches,
            match_original=match_original,
            *args, **kwargs
        )

    def get_labelcollection(self, *args, **kwargs):
        h_margin = kwargs.pop('h_margin', None)
        v_margin = kwargs.pop('v_margin', None)
        if 'horizontalalignment' not in kwargs:
            kwargs['horizontalalignment'] = 'right'
        if 'verticalalignment' not in kwargs:
            kwargs['verticalalignment'] = 'bottom'
        cluster_labels = []
        for x_pos in self.x_positions:
            for cluster in self.clusters[x_pos]:
                _h_margin = h_margin
                _v_margin = v_margin
                if cluster.label_margin:
                    _h_margin, _v_margin = cluster.label_margin
                if cluster.label is not None:
                    # # Options (example):
                    # 'a polar annotation',
                    # xy=(thistheta, thisr),  # theta, radius
                    # xytext=(0.05, 0.05),    # fraction, fraction
                    # textcoords='figure fraction',
                    # arrowprops=dict(facecolor='black', shrink=0.05),
                    cluster_label = {
                        's': cluster.label,
                        'xy': (
                            cluster.x - _h_margin,
                            cluster.y + _v_margin
                        )
                    }
                    cluster_label.update(kwargs)
                    cluster_labels.append(cluster_label)
        return cluster_labels

    def _distribute_column(self, x_pos, cluster_w_spacing):
        displace = 0.0
        for cluster in self.clusters[x_pos]:
            cluster.y = displace
            displace += cluster.height + cluster_w_spacing
        # now offset to center
        low = self.clusters[x_pos][0].y
        high = self.clusters[x_pos][-1].y + self.clusters[x_pos][-1].height
        cent_offset = low + 0.5 * (high - low)
        # _h_clusters = 0.5 * len(clusters)
        # cent_idx = int(_h_clusters) - 1 \
        #     if _h_clusters.is_integer() \
        #     else int(_h_clusters)
        # cent_offest = clusters[cent_idx].mid_height
        for cluster in self.clusters[x_pos]:
            cluster.y = cluster.y - cent_offset

    def color_clusters(self, patches, colormap=plt.cm.rainbow):
        r"""
        *unused*

        Parameters
        -----------
        patches: list[:class:`~matplotlib.patches.PathPatch`]
          Cluster patches to color.
        colormap: :obj:`matplotlib.cm` (default='rainbow')
          See the matplotlib tutorial for colormaps
          (`link <https://matplotlib.org/tutorials/colors/colormaps.html>`_)
          for details.

        """
        nbr_clusters = len(patches)
        c_iter = iter(colormap([i / nbr_clusters for i in range(nbr_clusters)]))
        for i in range(nbr_clusters):
            _color = next(c_iter)
            patches[i].set_facecolor(_color)
            patches[i].set_edgecolor(_color)
        return None
