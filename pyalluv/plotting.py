from __future__ import division, absolute_import, unicode_literals
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.path import Path
import matplotlib.patches as patches
from datetime import datetime
from bisect import bisect_left


class _Cluster(object):
    r"""
    This class defines the cluster objects for an alluvial diagram.

    Note
    -----

    The vertical position of a cluster will be set when creating a
    :class:`~pyalluv.plotting.Alluvial`.

    Parameters
    -----------

    height: float, int
      The cluster size which will translate into the height of this cluster.
    anchor: float (default=None)
      Set the anchor position. Either only the horizontal position, both
      :math:`(x, y)` or nothing can be provided.
    width: float (default=1.0)
      Set the cluster width.
    label: str (default=None)
      The label for this cluster, that can be shown in the diagram
    \**kwargs optional parameter:
      x_anchor: ``'center'``, ``'left'`` or ``'right'`` (default='center')
        Determine where the anchor position is relative to the rectangle
        that will represent this cluster. Options are either the left or
        right corner or centered:
      linewidth: float (default=0.0)
        Set the width of the line surrounding a cluster.
      label_margin: tuple(horizontal, vertical)
        Sets horizontal and vertical margins for the label of a cluster.

    Attributes
    -----------
    x_pos: float
      Horizontal position of the cluster anchor.
    y_pos: float
      Vertical position of the cluster center.
    x_anchor: str
      Anchor position relative to the rectangle representing the cluster.
      Possible values are: ``'center'``, ``'left'`` or ``'right'``.
    height: float
      Size of the cluster that will determine its height in the diagram.
    width: float
      Width of the cluster. In the same units as ``x_pos``.
    label: str
      Label, id or name of the cluster.
    in_fluxes: list[:class:`~pyalluv.fluxes.Flux`]
      All incoming fluxes of this cluster.
    out_fluxes: list[:class:`~pyalluv.fluxes.Flux`]
      All outgoing fluxes of this cluster.

    """
    def __init__(self, height, anchor=None, width=1.0, label=None, **kwargs):
        self._interp_steps = kwargs.pop('_interpolation_steps', 1)
        self.x_anchor = kwargs.pop('x_anchor', 'center')
        self.label = label
        self.label_margin = kwargs.pop('label_margin', None)
        self._closed = kwargs.pop('closed', False)
        self._readonly = kwargs.pop('readonly', False)
        self.patch_kwargs = kwargs
        self.patch_kwargs['lw'] = self.patch_kwargs.pop(
            'linewidth', self.patch_kwargs.pop('lw', 0.0)
        )
        if isinstance(height, (list, tuple)):
            self.height = len(height)
        else:
            self.height = height
        self.width = width
        if isinstance(anchor, (list, tuple)):
            x_coord, y_coord = anchor
        else:
            x_coord, y_coord = anchor, None
        self = self.set_x_pos(x_coord).set_y_pos(y_coord)

        # init the in and out fluxes:
        self.out_fluxes = []
        self.in_fluxes = []
        self.in_margin = {
            'bottom': 0,
            'top': 0
        }
        self.out_margin = {
            'bottom': 0,
            'top': 0
        }
        # ref points to add fluxes
        self.in_ = None
        self.out_ = None

    def set_x_pos(self, x_pos):
        r"""
        Set the horizontal position of a cluster.

        The position is set according to the value provided in ``x_pos`` and
        ``self.x_anchor``.

        Parameters
        -----------
        x_pos: float
          Horizontal position of the anchor for the cluster.

        Returns
        --------
        self: :class:`.Cluster`
          with new property ``x_pos``.

        """
        self.x_pos = x_pos
        if self.x_pos is not None:
            self.x_pos -= 0.5 * self.width
            if self.x_anchor == 'left':
                self.x_pos += 0.5 * self.width
            elif self.x_anchor == 'right':
                self.x_pos -= 0.5 * self.width

        return self

    def get_patch(self, **kwargs):
        _kwargs = dict(kwargs)
        _kwargs.update(self.patch_kwargs)
        self.set_in_out_anchors()

        vertices = [
            (self.x_pos, self.y_pos),
            (self.x_pos, self.y_pos + self.height),
            (self.x_pos + self.width, self.y_pos + self.height),
            (self.x_pos + self.width, self.y_pos),
            # this is just ignored as the code is CLOSEPOLY
            (self.x_pos, self.y_pos)
        ]
        codes = [
            Path.MOVETO,
            Path.LINETO,
            Path.LINETO,
            Path.LINETO,
            Path.CLOSEPOLY
        ]

        return patches.PathPatch(
            Path(
                vertices,
                codes,
                self._interp_steps,
                self._closed,
                self._readonly
            ),
            **_kwargs
        )

    def set_loc_out_fluxes(self,):
        for out_flux in self.out_fluxes:
            in_loc = None
            out_loc = None
            if out_flux.target_cluster is not None:
                if self.mid_height > out_flux.target_cluster.mid_height:
                    # draw to top
                    if self.mid_height >= \
                            out_flux.target_cluster.in_['top'][1]:
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
                            out_flux.target_cluster.in_['bottom'][1]:
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
            (i, self.out_fluxes[i])
            for i in range(len(self.out_fluxes))
            if self.out_fluxes[i].out_loc == 'top'
        ]
        _bottom_fluxes = [
            (i, self.out_fluxes[i])
            for i in range(len(self.out_fluxes))
            if self.out_fluxes[i].out_loc == 'bottom'
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
        self.out_fluxes = [self.out_fluxes[i] for i in sorted_idx]

    def sort_in_fluxes(self,):
        _top_fluxes = [
            (i, self.in_fluxes[i])
            for i in range(len(self.in_fluxes))
            if self.in_fluxes[i].in_loc == 'top'
        ]
        _bottom_fluxes = [
            (i, self.in_fluxes[i])
            for i in range(len(self.in_fluxes))
            if self.in_fluxes[i].in_loc == 'bottom'
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
        self.in_fluxes = [self.in_fluxes[i] for i in sorted_idx]

    def get_loc_out_flux(self, flux_width, out_loc, in_loc):
        anchor_out = (
            self.out_[out_loc][0],
            self.out_[out_loc][1] + self.out_margin[out_loc] + (flux_width if in_loc == 'bottom' else 0)
        )
        top_out = (
            self.out_[out_loc][0],
            self.out_[out_loc][1] + self.out_margin[out_loc] + (flux_width if in_loc == 'top' else 0)
        )
        self.out_margin[out_loc] += flux_width
        return anchor_out, top_out

    def set_anchor_out_fluxes(self,):
        for out_flux in self.out_fluxes:
            out_width = out_flux.flux_width \
                if out_flux.out_loc == 'bottom' else - out_flux.flux_width
            out_flux.anchor_out, out_flux.top_out = self.get_loc_out_flux(
                out_width, out_flux.out_loc, out_flux.in_loc
            )

    def set_anchor_in_fluxes(self,):
        for in_flux in self.in_fluxes:
            in_width = in_flux.flux_width \
                if in_flux.in_loc == 'bottom' else - in_flux.flux_width
            in_flux.anchor_in, in_flux.top_in = self.get_loc_in_flux(
                in_width, in_flux.out_loc, in_flux.in_loc
            )

    def get_loc_in_flux(self, flux_width, out_loc, in_loc):
        anchor_in = (
            self.in_[in_loc][0],
            self.in_[in_loc][1] + self.in_margin[in_loc] + (flux_width if out_loc == 'bottom' else 0)
        )
        top_in = (
            self.in_[in_loc][0],
            self.in_[in_loc][1] + self.in_margin[in_loc] + (flux_width if out_loc == 'top' else 0)
        )
        self.in_margin[in_loc] += flux_width
        return anchor_in, top_in

    def set_mid_height(self, mid_height):
        self.mid_height = mid_height
        if self.mid_height is not None:
            self.y_pos = self.mid_height - 0.5 * self.height
            self.set_in_out_anchors()
        else:
            self.y_pos = None

    def set_y_pos(self, y_pos):
        self.y_pos = y_pos
        if self.y_pos is not None:
            self.mid_height = self.y_pos + 0.5 * self.height
            self.set_in_out_anchors()
        else:
            self.mid_height = None

        return self

    def set_in_out_anchors(self,):
        """
        This sets the proper anchor points for fluxes to enter/leave
        """
        # if self.y_pos is None or self.mid_height is None:
        #     self.set_y_pos()

        self.in_ = {
            'bottom': (self.x_pos, self.y_pos),  # left, bottom
            'top': (self.x_pos, self.y_pos + self.height)  # left, top
        }
        self.out_ = {
            # right, top
            'top': (self.x_pos + self.width, self.y_pos + self.height),
            'bottom': (self.x_pos + self.width, self.y_pos)  # right,bottom
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
            self.source_cluster.out_fluxes.append(self)
        if self.target_cluster is not None:
            self.target_cluster.in_fluxes.append(self)

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
                    self.target_cluster.in_['bottom'][0] - self.source_cluster.out_['bottom'][0]
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


class Alluvial(object):
    """
    Alluvial diagram.

        Alluvial diagrams are a variant of flow diagram designed to represent
        changes in classifications, in particular changes in network
        structure over time.
        `Wikipedia (23/1/2021) <https://en.wikipedia.org/wiki/Alluvial_diagram>`_
    """
    def __init__(
        self, clusters, ax=None, y_pos='overwrite', cluster_w_spacing=1,
        cluster_kwargs={}, flux_kwargs={}, label_kwargs={},
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
            use the cluster's :attr:`~pyalluv.clusters.Cluster.y_pos`. If a
            cluster has no y position set this raises an exception.
          'complement':
            use the cluster's :attr:`~pyalluv.clusters.Cluster.y_pos` if
            set. Cluster without y position are positioned relative to the other
            clusters by minimizing the vertical displacements of all fluxes.
          'sorted':
            NOT IMPLEMENTED YET
        cluster_w_spacing: float, int (default=1)
          Vertical spacing between clusters
        cluster_kwargs: dict (default={})
          dictionary styling the Path elements of clusters.

          Keys:
            `facecolor`, `edgecolor`, `alpha`, `linewidth`, ...

        cluster_kwargs: dict (default={})
          dictionary styling the :obj:`~matplotlib.patches.PathPatch` of clusters.

          for a list of available options see
          :class:`~matplotlib.patches.PathPatch`

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

              **Examples\:**

              ``facecolor='cluster_reside'``
                set `facecolor` to the color of the source cluster if both source
                and target cluster are of the same color.

              ``edgecolor='cluster_migration'``
                set `edgecolor` to the color of the source cluster if source and
                target cluster are of different colors.

        \**kwargs optional parameter:
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
        # create axes if not provided
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            # TODO: not sure if specifying the ticks is necessary
            ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])

        # if clusters are given in a list of lists (each list is a x position)
        self._set_x_pos = kwargs.get('set_x_pos', True)
        self._redistribute_vertically = kwargs.get(
            'redistribute_vertically',
            4
        )
        self.with_cluster_labels = kwargs.get('with_cluster_labels', True)
        self.format_xaxis = kwargs.get('format_xaxis', True)
        self._cluster_kwargs = cluster_kwargs
        self._flux_kwargs = flux_kwargs
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
                    self.clusters[cluster.x_pos].append(cluster)
                except KeyError:
                    self.clusters[cluster.x_pos] = [cluster]
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
                ax.xaxis.set_major_locator(locator)
                ax.xaxis.set_major_formatter(formatter)
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
                    cluster.set_x_pos(mdates.date2num(cluster.x_pos))

        # TODO: set the cluster.width property
        else:
            for x_pos in self.x_positions:
                for cluster in self.clusters[x_pos]:
                    cluster_widths.append(cluster.width)
        self.cluster_width = kwargs.get('cluster_width', None)
        self.cluster_w_spacing = cluster_w_spacing
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
        if y_pos == 'overwrite':
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
                            n2.set_y_pos(n1.y_pos)
                            n1.set_y_pos(
                                n2.y_pos + n2.height + self.cluster_w_spacing
                            )
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
        ax.add_collection(patch_collection)
        if self.with_cluster_labels:
            label_collection = self.get_labelcollection(**label_kwargs)
            if label_collection:
                for label in label_collection:
                    ax.annotate(**label)
        ax.set_xlim(
            *self.x_lim
        )
        ax.set_ylim(
            *self.y_lim
        )
        if self._fill_figure:
            ax.set_position(
                [
                    0.0,
                    self._x_axis_offset,
                    0.99,
                    1.0 - self._x_axis_offset
                ]
            )
        if self._invisible_y:
            ax.get_yaxis().set_visible(False)
        if self._invisible_x:
            ax.get_xaxis().set_visible(False)
        ax.spines['right'].set_color('none')
        ax.spines['left'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        if isinstance(self.x_positions[0], datetime) and self.format_xaxis:
            self.set_dates_xaxis(ax, _minor_tick)

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
                    for in_flux in cluster.in_fluxes:
                        if in_flux.source_cluster is not None:
                            weights.append(in_flux.flux_width)
                            positions.append(in_flux.source_cluster.mid_height)
                    if sum(weights) > 0.0:
                        _redistribute = True
                        cluster.set_mid_height(
                            sum([
                                weights[i] * positions[i]
                                for i in range(len(weights))
                            ]) / sum(weights)
                        )
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
                        n2.set_y_pos(n1.y_pos)
                        n1.set_y_pos(
                            n2.y_pos + n2.height + self.cluster_w_spacing
                        )
                        self.clusters[x_pos][i - 1] = n2
                        self.clusters[x_pos][i] = n1
            for _ in range(int(0.5 * nbr_clusters)):
                for i in range(1, nbr_clusters):
                    n1 = self.clusters[x_pos][nbr_clusters - i - 1]
                    n2 = self.clusters[x_pos][nbr_clusters - i]
                    if self._swap_clusters(n1, n2, 'backwards'):
                        n2.set_y_pos(n1.y_pos)
                        n1.set_y_pos(
                            n2.y_pos + n2.height + self.cluster_w_spacing
                        )
                        self.clusters[x_pos][nbr_clusters - i - 1] = n2
                        self.clusters[x_pos][nbr_clusters - i] = n1

            _min_y = min(
                self.clusters[x_pos], key=lambda x: x.y_pos
            ).y_pos - 2 * self.cluster_w_spacing
            _max_y_cluster = max(
                self.clusters[x_pos],
                key=lambda x: x.y_pos + x.height
            )
            _max_y = _max_y_cluster.y_pos + \
                _max_y_cluster.height + 2 * self.cluster_w_spacing
            self.y_min = min(
                self.y_min,
                _min_y
            ) if self.y_min is not None else _min_y
            self.y_max = max(
                self.y_max,
                _max_y
            ) if self.y_max is not None else _max_y

    def set_dates_xaxis(self, ax, resolution='months'):
        r"""
        Format the x axis in case :class:`~datetime.datetime` objects are
        provide for the horizontal placement of clusters.

        Parameters
        -----------
        ax: :class:`~matplotlib.axes.Axes`
          Object to plot the alluvial diagram on.
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
            ax.xaxis.set_minor_locator(months)
            ax.xaxis.set_minor_formatter(monthsFmt)
            ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(yearsFmt)
        elif resolution == 'weeks':
            monthsFmt = mdates.DateFormatter('\n%b')
            weeksFmt = mdates.DateFormatter('%b %d')
            ax.xaxis.set_minor_locator(weeks)
            ax.xaxis.set_minor_formatter(weeksFmt)
            ax.xaxis.set_major_locator(months)
            ax.xaxis.set_major_formatter(monthsFmt)

    def _swap_clusters(self, n1, n2, direction='backwards'):
        squared_diff = {}
        for cluster in [n1, n2]:
            weights = []
            sqdiff = []
            if direction in ['both', 'backwards']:
                for in_flux in cluster.in_fluxes:
                    if in_flux.source_cluster is not None:
                        weights.append(in_flux.flux_width)
                        sqdiff.append(abs(
                            cluster.mid_height - in_flux.source_cluster.mid_height
                        ))
            if direction in ['both', 'forwards']:
                for out_flux in cluster.out_fluxes:
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
        assert n1.y_pos < n2.y_pos
        inv_mid_height = {
            n1: n2.y_pos + n2.height + self.cluster_w_spacing + 0.5 * n1.height,
            n2: n1.y_pos + 0.5 * n2.height
        }
        squared_diff_inf = {}
        for cluster in [n1, n2]:
            weights = []
            sqdiff = []
            if direction in ['both', 'backwards']:
                for in_flux in cluster.in_fluxes:
                    if in_flux.source_cluster is not None:
                        weights.append(in_flux.flux_width)
                        sqdiff.append(abs(
                            inv_mid_height[cluster] - in_flux.source_cluster.mid_height
                        ))
            if direction in ['both', 'forwards']:
                for out_flux in cluster.out_fluxes:
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
            if sum([_flux.flux_width for _flux in cluster.in_fluxes]) == 0.0:
                weights = []
                positions = []
                for out_flux in cluster.out_fluxes:
                    if out_flux.target_cluster is not None:
                        weights.append(out_flux.flux_width)
                        positions.append(out_flux.target_cluster.mid_height)
                if sum(weights) > 0.0:
                    _redistribute = True
                    cluster.set_mid_height(
                        sum(
                            [weights[i] * positions[i] for i in range(len(weights))]
                        ) / sum(weights)
                    )
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
        for x_pos in self.x_positions:
            out_fluxes = []
            for cluster in self.clusters[x_pos]:
                # TODO: set color
                # _cluster_color
                cluster.set_y_pos(cluster.y_pos + self.y_offset)
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
                out_fluxes.extend(cluster.out_fluxes)
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
                            cluster.x_pos - _h_margin,
                            cluster.y_pos + _v_margin
                        )
                    }
                    cluster_label.update(kwargs)
                    cluster_labels.append(cluster_label)
        return cluster_labels

    def _distribute_column(self, x_pos, cluster_w_spacing):
        displace = 0.0
        for cluster in self.clusters[x_pos]:
            cluster.set_y_pos(displace)
            displace += cluster.height + cluster_w_spacing
        # now offset to center
        low = self.clusters[x_pos][0].y_pos
        high = self.clusters[x_pos][-1].y_pos + self.clusters[x_pos][-1].height
        cent_offset = low + 0.5 * (high - low)
        # _h_clusters = 0.5 * len(clusters)
        # cent_idx = int(_h_clusters) - 1 \
        #     if _h_clusters.is_integer() \
        #     else int(_h_clusters)
        # cent_offest = clusters[cent_idx].mid_height
        for cluster in self.clusters[x_pos]:
            cluster.set_y_pos(cluster.y_pos - cent_offset)

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
