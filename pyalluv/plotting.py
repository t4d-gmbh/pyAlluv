import logging
import numpy as np
from matplotlib.collections import PatchCollection
# from matplotlib import docstring
from matplotlib import cbook
# from matplotlib import transforms
# from matplotlib import _api
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
from matplotlib.path import Path
from matplotlib.patches import Rectangle
import matplotlib.patches as patches
from matplotlib.legend import Legend
# from datetime import datetime
from bisect import bisect_left

# TODO: unused so far
_log = logging.getLogger(__name__)

__author__ = 'Jonas I. Liechti'


def _to_valid_sequence(data, attribute):
    if data is None:
        return None
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


# Note: could also be AlluvialArtistProxy
class _ArtistProxy:
    """
    Proxy class for `Artist` subclasses used to draw the various element in an
    alluvial diagram.

    This class assembles the common properties and relation to :class:`Patch`
    of the elements present in an Alluvial diagram.
    """
    _artistcls = None  # derived must overwrite
    _alias_map = getattr(_artistcls, "_alias_map", {})
    _sty_reduce_alias = {alias: p for p, aliases in _alias_map.items()
                         for alias in aliases}

    def __init__(self, label=None, **kwargs):
        """
        """
        self.stale = True
        self._tags = []
        self._artist = None
        self._kwargs = kwargs

    def set_tags(self, tags: list):
        """Set a tag for the block."""
        self._tags = tags

    def get_tags(self):
        return self._tags

    def add_tag(self, tag):
        if tag not in self._tags:
            self._tags.append(tag)

    def remove_tag(self, tag):
        if tag in self._tags:
            self._tags.remove(tag)

    # TODO: not sure if needed
    tags = property(get_tags, set_tags)

    @property
    def is_tagged(self):
        """Indicate whether a block belongs to a tag or not."""
        return True if len(self._tags) else False

    def set_styling(self, props):
        """Set the styling of the element."""

        self._original_kwargs = dict(props)
        self._kwargs = {self._sty_reduce_alias.get(k, k): v
                        for k, v in props.items()}

    def get_styling(self, original=False):
        """Return the custom styling properties of this element."""
        if original:
            return self._original_kwargs
        else:
            return self._kwargs

    def get_styling_prop(self, prop: str, altval=None):
        """
        Return the value of a specific styling property.

        If the property is not set *altval* is returned instead.

        Parameters
        ----------
        prop : str
            The name of the property to fetch.
        altval : Any (default: None)
            The value to return in case the element has no styling property
            with the name provided in *prop*.

        """
        # OLD:
        # return self._kwargs.get(prop, self._original_kwargs.get(prop, altval))
        # if prop in self._kwargs get it
        _prop = self._sty_reduce_alias.get(prop, prop)
        if _prop in self._kwargs:
            return self._kwarg[_prop]
        # if not but there are tags, get it from the tags
        # TODO: check if the prop is present in any of self._tags
        # if not in tags or no tags return altval
        return altval

    @property
    def is_styled(self,):
        """Indicate if this element has custom styling."""
        return True if self.is_tagged or self._kwargs else False

    def _create_artist(self, **kwargs):
        """Initiate the patch."""
        raise NotImplementedError('Derived must override')

    def create_artist(self, **kwargs):
        """
        Create the patch (or collection) for this element and return whether
        the element has individual styling or not.
        """
        _kwargs = self._update_styling(kwargs)
        self._create_artist(**_kwargs)
        self.stale = False

    def get_artist(self,):
        # self.set_path(self.create_path())
        # return patches.PathPatch(
        #     self.get_path(),
        #     **_kwargs
        # )
        # TODO: set stale=True in setter fcts
        if self.stale:
            raise ValueError("The artist has not been created.")
        return self._artist

    def _update_styling(self, props: dict):
        """
        Update styling properties with the styling properties attached to this
        element.

        Parameters
        ----------
        props : dict
            The dictionary that will be updated in place with *self._kwargs*.

        Returns
        -------
        updated_props : dict
            A copy of *props* updated with *self._kwargs*.
        """
        # get rid of aliases in the provided props
        _props = {self._sty_reduce_alias.get(k, k): v for k, v
                  in props.items()}
        # new_props = {self._sty_reduce_alias.get(k, k): v
        #              for k, v in self._kwargs.items() if v is not None}
        _props.update(self._kwargs)
        return _props


@cbook._define_aliases({
    "verticalalignment": ["va"],
    "horizontalalignment": ["ha"],
    "width": ["w"],
    "height": ["h"]
})
class _Block(_ArtistProxy):
    """
    A Block in an Alluvial diagram.

    Blocks in an Alluvial diagram get their vertical position assigned by a
    layout algorithm and thus after creation. This is the rational to why
    *_Block* inherits directly from `matplotlib.patches.Patch`, rather than
    `matplotlib.patches.PathPatch` or `matplotlib.patches.Rectangle`.

    """
    # set the artist subclass to use
    _artistcls = Rectangle

    # TODO uncomment once in mpl
    # @docstring.dedent_interpd
    def __init__(self, height, xa=None, ya=None, width=1.0, label=None,
                 tag=None, horizontalalignment='left',
                 verticalalignment='bottom', label_margin=(0, 0),
                 pathprops=None, **kwargs):
        """
        Parameters
        -----------
        height : float
          Height of the block.
        xa: scalar, optional
          The x coordinate of the block's anchor point.
        ya: scalar, optional
          The y coordinate of the block's anchor point.
        width : float,  default: 1.0
          Block width.
        label : str, optional
          Block label that can be displayed in the diagram.
        horizontalalignment : {'center', 'left', 'right'}, default: 'center'
          The horizontal location of the anchor point of the block.
        verticalalignment: {'center', 'top', 'bottom'}, default: 'center'
          The vertical location of the anchor point of the block.
        label_margin: (float, float), default: (0., 0.)
            x and y margin in target coordinates of ``self.get_transform()``
            and added to the *anchor* point to set the point to draw the label.
        pathprops : None
          TODO: This might be not needed, it's rather unlikely to be used.
          Keyword arguments that are passed to `matplotlib.path.Path`.

        Other Parameters
        ----------------
        **kwargs : Allowed are all `.Patch` properties:

          %(Patch_kwdoc)s

        """
        super().__init__(label=label, **kwargs)

        # TODO: only keep what's in else:
        if isinstance(height, (list, tuple)):
            self._height = len(height)
        else:
            self._height = height
        self._xa = xa
        self._ya = ya
        self._width = width
        self._set_horizontalalignment(horizontalalignment)
        self._set_verticalalignment(verticalalignment)
        # init the in and out flows:
        self._outflows = []
        self._inflows = []
        # self.label = label or ''
        self._label = label

        self.label_margin = label_margin
        self.pathprops = pathprops or dict()
        self.in_margin = {'bottom': 0, 'top': 0}
        self.out_margin = {'bottom': 0, 'top': 0}

    def get_xa(self):
        """Return the x coordinate of the anchor point."""
        return self._xa

    def get_ya(self):
        """Return the y coordinate of the anchor point."""
        return self._ya

    def get_height(self):
        """Return the height of the block."""
        return self._height

    def get_width(self):
        """Return the width of the block."""
        return self._width

    def get_anchor(self,):
        """Return the anchor point of the block."""
        return self._anchor

    def get_x(self):
        """Return the left coordinate of the block."""
        x0 = self._xa
        if self._horizontalalignment == 'center':
            x0 -= 0.5 * self._width
        elif self._horizontalalignment == 'right':
            x0 -= self._width
        return x0

    def get_y(self):
        """Return the bottom coordinate of the block."""
        y0 = self._ya
        if self._verticalalignment == 'center':
            y0 -= 0.5 * self._height
        elif self._verticalalignment == 'top':
            y0 -= self._height
        return y0

    def get_xy(self):
        """Return the left and bottom coords of the block as a tuple."""
        return self.get_x(), self.get_y()

    def get_xc(self, ):
        """Return the y coordinate of the block's center."""
        return self.get_x() + 0.5 * self._height

    def get_yc(self, ):
        """Return the y coordinate of the block's center."""
        return self.get_y() + 0.5 * self._height

    def get_center(self,):
        """Return the center point of the block."""
        return (self.get_xc(),
                self.get_yc())

    def get_outflows(self):
        """Return a list of outgoing `._Flows`."""
        return self._outflows

    def get_inflows(self):
        """Return a list of incoming `._Flows`."""
        return self._inflows

    def set_xa(self, xa):
        """Set the x coordinate of the anchor point."""
        _update_locs = xa is not None and self._xa != xa
        self._xa = xa
        if _update_locs:
            self._set_inloc()
            self._set_outloc()
        self.stale = True

    def set_ya(self, ya):
        """Set the y coordinate of the anchor point."""
        _update_locs = ya is not None and self._ya != ya
        self._ya = ya
        if _update_locs:
            self._set_inloc()
            self._set_outloc()
        self.stale = True

    def set_y(self, y):
        """
        Set the y coordinate of the block's bottom.

        Note that this method alters the y coordinate of the anchor point.
        """
        self._ya = y
        if self._verticalalignment == 'center':
            self._ya += 0.5 * self._height
        elif self._verticalalignment == 'top':
            self._ya += self._height

    def set_yc(self, yc):
        """
        Set the y coordinate of the block center.

        Note that this method alters the y coordinate of the anchor point.
        """
        self._ya = yc
        if self._verticalalignment == 'bottom':
            self._ya -= 0.5 * self._height
        elif self._verticalalignment == 'top':
            self._ya += 0.5 * self._height

    def set_width(self, width):
        """Set the width of the block."""
        self._width = width
        self.stale = True

    def set_height(self, height):
        """Set the height of the block."""
        self._height = height
        self.stale = True

    # TODO: inloc and outloc should be set.
    # def set_y(self, y):
    #     self._y0 = y
    #     if self._y0 is not None:
    #         self._mid_height = self._y0 + 0.5 * self._height
    #         self._set_inloc()
    #         self._set_outloc()
    #     else:
    #         self._mid_height = None
    #     self.stale = True

    def _set_horizontalalignment(self, align):
        # TODO: uncomment once in mpl
        # _api.check_in_list(['center', 'left', 'right'], align=align)
        self._horizontalalignment = align
        self.stale = True

    def _set_verticalalignment(self, align):
        # TODO: uncomment once in mpl
        # _api.check_in_list(['center', 'top', 'bottom'], align=align)
        self._verticalalignment = align
        self.stale = True

    def set_horizontalalignment(self, align):
        """Set the horizontal alignment of the anchor point and the block."""
        _ha = self._horizontalalignment
        self._set_horizontalalignment(self, align)
        if _ha != self._horizontalalignment:
            self._set_inloc()
            self._set_outloc()

    def set_verticalalignment(self, align):
        """Set the vertical alignment of the anchor point and the block."""
        _va = self._verticalalignment
        self._set_verticalalignment(align)
        if _va != self._verticalalignment:
            self._set_inloc()
            self._set_outloc()

    def set_outflows(self, outflows):
        self._outflows = outflows

    def set_inflows(self, inflows):
        self._inflows = inflows

    # when it was a Patch
    # def get_bbox(self):
    #     """Return the `.Bbox`."""
    #     x0, y0, x1, y1 = self._convert_units()
    #     return transforms.Bbox.from_extents(x0, y0, x1, y1)

    xa = property(get_xa, set_xa, doc="The block anchor's x coordinate")
    ya = property(get_ya, set_ya, doc="The block anchor's y coordinate")
    y = property(get_y, set_y, doc="The y coordinate of the block bottom")
    x = property(get_x, None, doc="The x coordinate of the block bottom")
    inflows = property(get_inflows, set_inflows, doc="List of `._Flow` objects"
                                                     "entering the block.")
    outflows = property(get_outflows, set_outflows, doc="List of `._Flow`"
                                                        "objects leaving the"
                                                        "block.")

    def add_outflow(self, outflow):
        self._outflows.append(outflow)

    def add_inflow(self, inflow):
        self._inflows.append(inflow)

    # when it was a Patch
    # def get_path(self):
    #     """Return the vertices of the block."""
    #     # vertices = [
    #     #     (self._x0, self._y0),
    #     #     (self._x0, self._y0 + self._height),
    #     #     (self._x0 + self._width, self._y0 + self._height),
    #     #     (self._x0 + self._width, self._y0),
    #     #     (self._x0, self._y0)  # ignored as codes[-1] is CLOSEPOLY
    #     # ]
    #     # codes = [
    #     #     Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY
    #     # ]
    #     # return Path(
    #     #     vertices,
    #     #     codes,
    #     #     **self.pathprops
    #     # )
    #     return Path.unit_rectangle()

    # when it was a Patch
    # def _convert_units(self):
    #     """Convert bounds of the rectangle."""
    #     x0, y0 = self.get_xy()
    #     x0 = self.convert_xunits(x0)
    #     y0 = self.convert_yunits(y0)
    #     x1 = self.convert_xunits(x0 + self._width)
    #     y1 = self.convert_yunits(y0 + self._height)
    #     return x0, y0, x1, y1

    # when it was a Patch
    # def get_patch_transform(self):
    #     # Note: This cannot be called until after this has been added to
    #     # an Axes, otherwise unit conversion will fail. This makes it very
    #     # important to call the accessor method and not directly access the
    #     # transformation member variable.
    #     bbox = self.get_bbox()
    #     # return (transforms.BboxTransformTo(bbox)
    #     #         + transforms.Affine2D().rotate_deg_around(
    #     #             bbox.x0, bbox.y0, self.angle))
    #     return transforms.BboxTransformTo(bbox)
    # def set_mid_height(self, mid_height):
    #     self._mid_height = mid_height
    #     if self._mid_height is not None:
    #         self._y0 = self._mid_height - 0.5 * self._height
    #         self._set_inloc()
    #         self._set_outloc()
    #     else:
    #         self._y0 = None
    #     self.stale = True

    # old
    # def get_mid_height(self):
    #     if self._mid_height is None:
    #         _log.warning(
    #             "Before accessing vertical coordinates of a block (i.e. *y* or"
    #             " *mid_heigth*) the block need to be distributed vertically."
    #         )
    #     return self._mid_height
    # def _get_xy_from_anchor(self, anchor):
    #     # set x and y (if possible)
    #     if isinstance(anchor, (list, tuple)):
    #         xa, ya = anchor
    #     else:
    #         xa, ya = anchor, None
    #     if xa is not None:
    #         xa -= 0.5 * self._width
    #         if self._horizontalalignment == 'left':
    #             xa += 0.5 * self._width
    #         elif self._horizontalalignment == 'right':
    #             xa -= 0.5 * self._width
    #     return xa, ya
    # mid_height = property(
    #     get_mid_height, set_mid_height,
    #     doc="y coordinate of the block's center."
    # )

    def _create_artist(self, **kwargs):
        """Blocks use :class:`patches.Rectangle` as their patch."""
        # TODO: maybe use the more generic _artist?
        self._artist = self._artistcls(self.get_xy(), self._width,
                                       self._height, **kwargs)

    def update_locations(self,):
        # TODO: this needs to be called AFTER self._artist has been attached to
        # an axis
        x0, y0, x1, y1 = self._artist._convert_units()
        # Use x0, y0, x1, y1 to set outloc and inloc
        self._set_outloc()
        self._set_inloc()

    def set_loc_out_flows(self,):
        yc = self.get_yc()
        for out_flow in self._outflows:
            in_loc = None
            out_loc = None
            if out_flow.target is not None:
                if yc > out_flow.target.get_yc():
                    # draw to top
                    if yc >= out_flow.target.inloc['top'][1]:
                        # draw from bottom to in top
                        out_loc = 'bottom'
                        in_loc = 'top'
                    else:
                        # draw from top to top
                        out_loc = 'top'
                        in_loc = 'top'
                else:
                    # draw to bottom
                    if yc <= out_flow.target.inloc['bottom'][1]:
                        # draw from top to bottom
                        out_loc = 'top'
                        in_loc = 'bottom'
                    else:
                        # draw form bottom to bottom
                        out_loc = 'bottom'
                        in_loc = 'bottom'
            else:
                out_flow.out_loc = out_flow.out_flow_vanish
            out_flow.in_loc = in_loc
            out_flow.out_loc = out_loc

    def sort_out_flows(self,):
        _top_flows = [
            (i, self._outflows[i])
            for i in range(len(self._outflows))
            if self._outflows[i].out_loc == 'top'
        ]
        _bottom_flows = [
            (i, self._outflows[i])
            for i in range(len(self._outflows))
            if self._outflows[i].out_loc == 'bottom'
        ]
        if _top_flows:
            sorted_top_idx, _flows_top = zip(*sorted(
                _top_flows,
                key=lambda x: x[1].target.get_yc()
                if x[1].target
                # TODO: this should not simply be -10000
                else -10000,
                reverse=True
            ))
        else:
            sorted_top_idx = []
        if _bottom_flows:
            sorted_bottom_idx, _flows_bottom = zip(*sorted(
                _bottom_flows,
                key=lambda x: x[1].target.get_yc()
                if x[1].target
                # TODO: this should not simply be -10000
                else -10000,
                reverse=False
            ))
        else:
            sorted_bottom_idx = []
        sorted_idx = list(sorted_top_idx) + list(sorted_bottom_idx)
        self._outflows = [self._outflows[i] for i in sorted_idx]

    def sort_in_flows(self,):
        _top_flows = [
            (i, self._inflows[i])
            for i in range(len(self._inflows))
            if self._inflows[i].in_loc == 'top'
        ]
        _bottom_flows = [
            (i, self._inflows[i])
            for i in range(len(self._inflows))
            if self._inflows[i].in_loc == 'bottom'
        ]
        if _top_flows:
            sorted_top_idx, _flows_top = zip(*sorted(
                _top_flows,
                key=lambda x: x[1].source.get_yc()
                if x[1].source
                # TODO: this should not simply be -10000
                else -10000,
                reverse=True
            ))
        else:
            sorted_top_idx = []
        if _bottom_flows:
            sorted_bottom_idx, _flows_bottom = zip(*sorted(
                _bottom_flows,
                key=lambda x: x[1].source.get_yc()
                if x[1].source
                # TODO: this should not simply be -10000
                else -10000,
                reverse=False
            ))
        else:
            sorted_bottom_idx = []
        sorted_idx = list(sorted_top_idx) + list(sorted_bottom_idx)
        self._inflows = [self._inflows[i] for i in sorted_idx]

    def get_loc_out_flow(self, flow_width, out_loc, in_loc):
        anchor_out = (
            self.outloc[out_loc][0],
            self.outloc[out_loc][1] + self.out_margin[out_loc] + (flow_width if in_loc == 'bottom' else 0)
        )
        top_out = (
            self.outloc[out_loc][0],
            self.outloc[out_loc][1] + self.out_margin[out_loc] + (flow_width if in_loc == 'top' else 0)
        )
        self.out_margin[out_loc] += flow_width
        return anchor_out, top_out

    def set_anchor_out_flows(self,):
        for out_flow in self._outflows:
            out_width = out_flow.flow_width \
                if out_flow.out_loc == 'bottom' else - out_flow.flow_width
            out_flow.anchor_out, out_flow.top_out = self.get_loc_out_flow(
                out_width, out_flow.out_loc, out_flow.in_loc
            )

    def set_anchor_in_flows(self,):
        for in_flow in self._inflows:
            in_width = in_flow.flow_width \
                if in_flow.in_loc == 'bottom' else - in_flow.flow_width
            in_flow.anchor_in, in_flow.top_in = self.get_loc_in_flow(
                in_width, in_flow.out_loc, in_flow.in_loc
            )

    def get_loc_in_flow(self, flow_width, out_loc, in_loc):
        anchor_in = (
            self.inloc[in_loc][0],
            self.inloc[in_loc][1] + self.in_margin[in_loc] + (flow_width if out_loc == 'bottom' else 0)
        )
        top_in = (
            self.inloc[in_loc][0],
            self.inloc[in_loc][1] + self.in_margin[in_loc] + (flow_width if out_loc == 'top' else 0)
        )
        self.in_margin[in_loc] += flow_width
        return anchor_in, top_in

    def _set_inloc(self,):
        x0, y0 = self.get_xy()
        self.inloc = {
            'bottom': (x0, y0),  # left, bottom
            'top': (x0, y0 + self._height)  # left, top
        }

    def _set_outloc(self,):
        x0, y0 = self.get_xy()
        self.outloc = {
            # right, top
            'top': (x0 + self._width, y0 + self._height),
            'bottom': (x0 + self._width, y0)  # right, bottom
        }


@cbook._define_aliases({
    "edgecolor": ["ec"],
    "facecolor": ["fc"],
})
class _Flow:
    """
    A connection between two blocks from adjacent columns.
    """
    def __init__(
            self, flow,
            source=None, target=None,
            relative_flow=False,
            **kwargs):
        """

        Parameters
        -----------
        relative_flow: bool
          If ``True`` the fraction of the height of parameter `source`
          is taken, if the source is none, then the
          relative height form the target is taken.
        source: :class:`pyalluv.clusters.Cluster` (default=None)
          Cluster from which the flow originates.
        target: :class:`pyalluv.clusters.Cluster` (default=None)
          Cluster into which the flow leads.

        Other Parameters
        ----------------
        **kwargs : Allowed are all `.Patch` properties:

          %(Patch_kwdoc)s

        Note that `color`, `edgecolor` and `facecolor` also accept the
        particular values `'source'` (or `'s'`), `'target'` (or `'t'`) and
        `'interpolate'`.

        kwargs from old implementation

          interpolation_steps:

          out_flow_vanish: str (default='top')

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

        flow: float
          The size of the flow which will translate to the height of the flow in
          the Alluvial diagram.
        source: :class:`pyalluv.clusters.Cluster` (default=None)
          Cluster from which the flow originates.
        target: :class:`pyalluv.clusters.Cluster` (default=None)
          Cluster into which the flow leads.
        """
        self._interp_steps = kwargs.pop('interpolation_steps', 1)
        self.out_flow_vanish = kwargs.pop('out_flow_vanish', 'top')
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

        if isinstance(flow, (list, tuple)):
            self.flow = len(flow)
        else:
            self.flow = flow
        self.relative_flow = relative_flow
        self.source = source
        self.target = target
        if self.source is not None:
            if self.relative_flow:
                self.flow_width = self.flow * self.source.get_height()
            else:
                self.flow_width = self.flow
        else:
            if self.target is not None:
                if self.relative_flow:
                    self.flow_width = self.flow * self.target.get_height()
                else:
                    self.flow_width = self.flow
        # append the flow to the clusters
        if self.source is not None:
            self.source.add_outflow(self)
        if self.target is not None:
            self.target.add_inflow(self)
        self.stale = True

    def create_patch(self, **kwargs):
        _kwargs = dict(kwargs)
        _to_in_kwargs = {}
        _to_out_kwargs = {}
        for kw in _kwargs:
            if kw.startswith('in_'):
                _to_in_kwargs[kw[3:]] = _kwargs.pop(kw)
            elif kw.startswith('out_'):
                _to_out_kwargs[kw[3:]] = _kwargs.pop(kw)
        # update with flow specific styling
        _kwargs.update(self.patch_kwargs)
        for _color in ['facecolor', 'edgecolor']:
            _set_color = _kwargs.pop(_color, None)
            _set_alpha = _kwargs.pop('alpha', None)
            if isinstance(_set_alpha, (int, float)):
                _kwargs['alpha'] = _set_alpha
                _set_alpha = None
            color_is_set = False
            if _set_color == 'source' or _set_color == 'cluster':
                from_cluster = self.source
                color_is_set = True
            elif _set_color == 'target':
                from_cluster = self.target
                color_is_set = True
            elif isinstance(_set_color, str) and '__' in _set_color:
                which_cluster, flow_type = _set_color.split('__')
                if which_cluster == 'target':
                    from_cluster = self.target
                else:
                    from_cluster = self.source
                if flow_type == 'migration' \
                        and self.source.patch_kwargs.get(_color) \
                        != self.target.patch_kwargs.get(_color):
                    color_is_set = True
                    if _set_alpha:
                        _kwargs['alpha'] = _set_alpha.get(
                            'migration', _set_alpha.get('default', self.default_alpha)
                        )
                elif flow_type == 'reside'  \
                        and self.source.patch_kwargs.get(_color) \
                        == self.target.patch_kwargs.get(_color):
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
        # set in/out only flow styling
        _in_kwargs = dict(_kwargs)
        _in_kwargs.update(_to_in_kwargs)
        _out_kwargs = dict(_kwargs)
        _out_kwargs.update(_to_out_kwargs)

        _dist = None
        if self.out_loc is not None:
            if self.in_loc is not None:
                _dist = 2 / 3 * (
                    self.target.in_['bottom'][0] - self.source.outloc['bottom'][0]
                )
            else:
                _dist = 2 * self.source.get_width()
                _kwargs = _out_kwargs
        else:
            if self.in_loc is not None:
                _kwargs = _in_kwargs
            else:
                raise Exception('flow with neither source nor target cluster')

        # now complete the path points
        if self.anchor_out is not None:
            anchor_out_inner = (
                self.anchor_out[0] - 0.5 * self.source.get_width(),
                self.anchor_out[1]
            )
            dir_out_anchor = (self.anchor_out[0] + _dist, self.anchor_out[1])
        else:
            # TODO set to form vanishing flow
            # anchor_out = anchor_out_inner =
            # dir_out_anchor =
            pass
        if self.top_out is not None:
            top_out_inner = (
                self.top_out[0] - 0.5 * self.source.get_width(),
                self.top_out[1]
            )
            # 2nd point 2/3 of distance between clusters
            dir_out_top = (self.top_out[0] + _dist, self.top_out[1])
        else:
            # TODO set to form vanishing flow
            # top_out = top_out_inner =
            # dir_out_top =
            pass
        if self.anchor_in is not None:
            anchor_in_inner = (
                self.anchor_in[0] + 0.5 * self.target.get_width(),
                self.anchor_in[1]
            )
            dir_in_anchor = (self.anchor_in[0] - _dist, self.anchor_in[1])
        else:
            # TODO set to form new in flow
            # anchor_in = anchor_in_inner =
            # dir_in_anchor =
            pass
        if self.top_in is not None:
            top_in_inner = (
                self.top_in[0] + 0.5 * self.target.get_width(),
                self.top_in[1]
            )
            dir_in_top = (self.top_in[0] - _dist, self.top_in[1])
        else:
            # TODO set to form new in flow
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

        # flow_patch = patches.PathPatch(_path, **_kwargs)
        # return flow_patch
        self._patch = patches.PathPatch(_path, **_kwargs)

    # TODO: ideally setting ec and fc should not happen in this class, only
    # special colors should be converted
    def set_edgecolor(self, color):
        if color in ['source', 'target', 's', 't', 'interpolate']:
            # get the color form source or target
            # or
            # make this flux use a patchcollection with a linearsegment cmap
            pass
        # TODO: replace the same keyword in self._kwargs with an actual color
        # that can be passed to PathPatch
        pass

    def set_facecolor(self, color):
        pass

    def get_patch(self):
        if self.stale:
            self.create_patch()
            self.stale = False
        return self._patch


class BlockCollection(_ArtistProxy):
    """
    A collection of Blocks with common styling properties.
    """
    _artistcls = PatchCollection

    def __init__(self, blocks, label=None, **kwargs):
        """
        Parameters
        ----------
        blocks : sequence of :obj:`_Block`
            The blocks in this collection.
        label : str, optional
            Label of the collection.
        """
        # get cmap data:
        self._cmap_data = kwargs.pop('cmap_array', None)
        if self._cmap_data is not None:
            # TODO: allow either a block property, x, or custom array
            self._cmap_data = 'x'

        super().__init__(label=label, **kwargs)

        # TODO: make sure that blocks is correct
        self._blocks = blocks

        # ###
        # Below is from SubDiagram.__init__

        # # TODO: This is back from when it was a subclass of collection > used??
        # if match_original:
        #     def determine_facecolor(block):
        #         # TODO: Block does not have a get_fill, right?
        #         if block.get_fill():
        #             return block.get_facecolor()
        #         return [0, 0, 0, 0]
        #     kwargs['facecolors'] = [determine_facecolor(b) for b in self._blocks]
        #     kwargs['edgecolors'] = [b.get_edgecolor() for b in self._blocks]
        #     kwargs['linewidths'] = [b.get_linewidth() for b in self._blocks]
        #     kwargs['linestyles'] = [b.get_linestyle() for b in self._blocks]
        #     kwargs['antialiaseds'] = [b.get_antialiased() for b in self._blocks]

        # TODO: below attributes need to be handled
        self._redistribute_vertically = 4
        self.y_min, self.y_max = None, None

        # ###

    def to_element_styling(self, styleprops: dict):
        """
        Convert the styling properties to lists matching self._blocks.
        """
        indiv_props = dict()
        for k, v in styleprops.items():
            indiv_props[k] = [b.get_styling_prop(k, v) for b in self._blocks]
        return indiv_props

    def _create_artist(self, **kwargs):
        """
        Creates `.PatchCollections`s for the blocks in this collection.

        Parameters
        ----------

        """
        # remove the cmap as this will not fly with block.create_artist
        cmap = kwargs.pop('cmap', None)
        for block in self._blocks:
            # do not allow to pass kws to patch here.
            # block.create_artist(**_kwargs)
            # TODO: if block has no other tag
            block.create_artist(**kwargs)
        match_original = False
        if any(block.is_styled for block in self._blocks):
            match_original = True
        if match_original:
            kwargs = self.to_element_styling(kwargs)

        self._artist = self._artistcls(
            [block.get_artist()
             for block in self._blocks],
            # for column in self._columns
            # for block in column],
            cmap=cmap,
            match_original=match_original,
            **kwargs
        )
        if self._cmap_data is not None:
            self._artist.set_array(
                np.asarray([getattr(block, self._cmap_data)
                            for block in self._blocks]))
            # np.asarray([getattr(block, self._cmap_data)
            #             for column in self._columns
            #             for block in column]))

        # ###
        # TODO: Draw the flows
        # ###
        # now the flows
        # self._flow_collection = PatchCollection([flow.get_patch()
        #                                          for flow in self._flows])

    # NOTE: this is in _AlluvialElement
    # def create_patch(self, match_original=False, **kwargs):
    #     _kwargs = dict(kwargs)
    #     _kwargs.update(self._kwargs)
    #     # for col_id in range(self._nbr_columns):
    #     #     for block in self._columns[col_id]:

    def add_block(self, block):
        """Add a Block."""
        self._blocks.append(block)


# TODO: maybe Tag is not even needed, but can just be BlockCollection
class Tag(BlockCollection):
    """
    A collection of `Blocks`
    """
    def __init__(self, label=None, **kwargs):
        """
        Parameters
        ----------
        label : str, optional
            The label of this collection.
        """
        super().__init__(label=label, **kwargs)

    # def set_paths(self, blocks):
    #     """Set the paths for all blocks in this collection."""
    #     self._paths = []
    #     for col in blocks:
    #         self._paths.extend(
    #             [p.get_transform().transform_path(p.get_path())
    #              for p in col
    #              if not p.is_tagged])

    def get_paths(self,):
        return [b.get_transform().transform_path(b.get_path())
                for b in self._blocks]

    def add_block(self, block):
        self._blocks.append(block)
        # update the styling


class SubDiagram(BlockCollection):
    """
    A collection of Blocks and Flows belonging to a diagram.

    """
    # def __init__(self, patches, match_original=False, **kwargs):
    def __init__(self, x, columns, flows, fractionflow,
                 label=None, yoff=0, hspace=1, hspace_combine='add',
                 label_margin=(0, 0), layout='centered', blockprops=None,
                 **kwargs):
        """
        Parameters
        ----------
        x : sequence of scalars
            A sequence of M scalars that determine the x coordinates of columns
            provided in *columns*.
        columns : sequence of array_like objects
            Sequence of M array-like objects each containing the blocks of a
            column.
            Allowed are `_Block` objects or floats that will be interpreted as
            the size of a block.
        flows : sequence of array_like objects
            ... *TODO*
        label : str, optional
            Label of the diagram.
        yoff : int or float, default: 0
            A constant vertical offset applied to the added diagram.
        hspace : float, (default=1)
            The height reserved for space between blocks expressed as a
            float in the same unit as the block heights.
        hspace_combine : {'add', 'divide'}, default: 'add'
            Set how the vertical space between blocks should be combined.
            If set to 'add' (default) the space between two blocks takes
            the value provided by *hspace*. If set to 'divide' then the sum of
            all spaces between the blocks in a column is set to be equal to
            *hspace*.
        label_margin : tuple, optional
            determine the offset in points for the label.

            .. TODO:
                This should be in points.
        layout : sequence or str, default: 'centered'
            The type of layout used to display the diagram.
            Allowed layout modes are: {'centered', 'bottom', 'top'}.

            If as sequence is provided, the M elements must specify the layout
            for each of the M columns in the diagram.

            The following options are available:

            - 'centered' (default): The bigger the block (in terms of height)
              the more it is moved towards the center.
            - 'bottom': Blocks are sorted according to their height with the
              biggest blocks at the bottom.
            - 'top': Blocks are sorted according to their height with the
              biggest blocks at the top.

        Other Parameters
        ----------------
        **kwargs : Allowed are all `.Collection` properties
            Define the styling to apply to all elements in this subdiagram:

            %(Collection_kwdoc)s

        Note that *x* and *columns* must be sequences of the same length.
        """
        if x is not None:
            self._x = _to_valid_sequence(x, 'x')
        else:
            self._x = None
        self._yoff = yoff
        self._blockprops = blockprops or dict()

        # create the columns of Blocks
        columns = list(columns)
        _provided_blocks = False
        for col in columns:
            if len(col):
                if isinstance(col[0], _Block):
                    _provided_blocks = True
                break
        _blocks = []
        self._columns = []
        if _provided_blocks:
            for col in columns:
                # self._columns.append([])
                # self._columns.extend([i for i in
                #                       range(b_id, newb_id)])
                column = list(col)
                _blocks.extend(column)
                self._columns.append(column)
            # self._columns = [list(col) for col in columns]
        else:
            for xi, col in zip(x, columns):
                column = [_Block(size, xa=xi, **self._blockprops)
                          for size in col]
                self._columns.append(column)
                _blocks.extend(column)
                # self._columns.append([_Block(size, xa=xi, **self._blockprops)
                #                       for size in column])
                # self._columns.append([_Block(size, xa=xi,
                #                      **kwargs) for size in column])
        self._nbr_columns = len(self._columns)

        # TODO: determine what other arguments need to go to super
        super().__init__(blocks=_blocks, label=label, **kwargs)

        # create the Flows is only based on *flows* and *extout*'s
        self._flows = []
        # connect source and target:
        for m, flowM in enumerate(flows):
            # m is the source column, m+1 the target column
            s_col = self._columns[m]
            t_col = self._columns[m + 1]
            _flows = []
            for i, row in enumerate(flowM):
                # i is the index of the target block
                for j, f in enumerate(row):
                    # j is the index of the source block
                    # TODO: pass kwargs?
                    _flows.append(_Flow(flow=f, source=s_col[j],
                                        target=t_col[i],
                                        relative_flow=fractionflow))
            self._flows.append(_flows)

        self._hspace = hspace

        # TODO: create set_... and process like set_hspace
        self._hspace_combine = hspace_combine
        self.set_layout(layout)

        self._label_margin = label_margin

    def get_layout(self):
        """Get the layout of this diagram"""
        return self._layout

    def get_columns(self,):
        """Get all columns of this subdiagram"""
        return self._columns

    def set_column_layout(self, col_id, layout):
        """Set the layout for a single column"""
        # TODO: uncomment once in mpl
        # _api.check_in_list(['centered', 'top', 'bottom'], layout=layout)
        self._layout[col_id] = layout

    def get_column_layout(self, col_id):
        """Get the layout of a single column."""
        return self._layout[col_id]

    def set_layout(self, layout):
        """Set the layout for this diagram"""
        if isinstance(layout, str):
            # TODO: uncomment once in mpl
            # _api.check_in_list(['centered', 'top', 'bottom'], layout=layout)
            self._layout = [layout for _ in range(self._nbr_columns)]
        else:
            self._layout = []
            for _layout in layout:
                # TODO: uncomment once in mpl
                # _api.check_in_list(['centered', 'top', 'bottom'],
                #                     layout=_layout)
                self._layout.append(_layout)

    def determine_layout(self, ):
        for col_id in range(self._nbr_columns):
            # TODO: handle the layout parameter
            self.distribute_blocks(col_id)

    # TODO: This is probably not used, right?!
    def set_paths(self, columns):
        """Set the paths for untagged blocks of the subdiagram."""
        self._paths = []
        for col in columns:
            self._paths.extend(
                [p.get_transform().transform_path(p.get_path())
                 for p in col
                 if not p.is_tagged])

    def add_block(self, column: int, block):
        """Add a Block to a column."""
        # add the block to the blocks
        super().add_block(block)
        # add the block to _columns
        self._columns[column].append(block)

    def add_flow(self, column, flow):
        # TODO: _columns can only contain indices for blocks
        # self._columns[column].append(flow)
        pass

    def get_column_hspace(self, col_id):
        if self._hspace_combine == 'add':
            return self._hspace
        else:
            nbr_blocks = len(self._columns[col_id])
            if nbr_blocks > 1:
                return self._hspace / (nbr_blocks - 1)
            else:
                return 0

    def distribute_blocks(self, col_id):
        """
        Distribute the blocks in a column.

        Parameters
        -----------
        x_pos: float
          The horizontal position at which the clusters should be distributed.
          This must be a `key` of the :attr:`~.Alluvial.clusters`
          attribute.
        """
        # TODO: not sure why this was used
        # x_pos = self._x[col_id]
        nbr_blocks = len(self._columns[col_id])
        layout = self.get_column_layout(col_id)
        col_hspace = self.get_column_hspace(col_id)
        if nbr_blocks:
            # sort clusters according to height
            ordering, _column = zip(
                *sorted(enumerate(self._columns[col_id]),
                        key=lambda x: x[1].get_height())
            )
            if layout == 'top':
                # TODO: do the reordering outside if/elif/else (after)
                self._reorder_column(col_id, ordering)
                self._update_ycoords(col_id, col_hspace, layout)
            elif layout == 'bottom':
                ordering = ordering[::-1]
                self._reorder_column(col_id, ordering)
                self._update_ycoords(col_id, col_hspace, layout)
            # in both cases no further sorting is needed
            if layout == 'centered':
                # sort so to put biggest height in the middle
                # self._columns[col_id] = _column[::-2][::-1] + \
                #     _column[nbr_blocks % 2::2][::-1]
                ordering = ordering[::-2][::-1] + \
                    ordering[nbr_blocks % 2::2][::-1]
                # update the ordering
                self._reorder_column(col_id, ordering)
                # set positioning
                self._update_ycoords(col_id, col_hspace, layout)
                # ###
                # TODO: both methods below need to be checked
                # # now sort again considering the flows.
                # self._decrease_flow_distances(col_id)
                # # perform pairwise swapping for backwards flows
                # self._pairwise_swapping(col_id)

                # TODO: bad implementation, avoid duplicated get_y call
                _min_y = min(
                    self._columns[col_id],
                    key=lambda x: x.get_y()
                ).get_y() - 2 * col_hspace
                _max_y_cluster = max(
                    self._columns[col_id],
                    key=lambda x: x.get_y() + x.get_height()
                )
                _max_y = _max_y_cluster.get_y() + \
                    _max_y_cluster.get_height() + 2 * col_hspace
                self.y_min = min(
                    self.y_min,
                    _min_y
                ) if self.y_min is not None else _min_y
                self.y_max = max(
                    self.y_max,
                    _max_y
                ) if self.y_max is not None else _max_y

    def _decrease_flow_distances(self, col_id):
        _column = self._columns[col_id]
        # TODO: does not really make sense to recompute them here
        nbr_blocks = len(_column)
        layout = self.get_column_layout(col_id)
        col_hspace = self.get_column_hspace(col_id)
        old_mid_heights = [block.get_yc() for block in _column]
        # do the redistribution a certain amount of times
        _redistribute = False
        for _ in range(self._redistribute_vertically):
            # TODO: check this as soon as Flows are set-up correctly
            for block in _column:
                weights = []
                positions = []
                for in_flow in block.inflows:
                    if in_flow.source is not None:
                        weights.append(in_flow.flow_width)
                        positions.append(in_flow.source.get_yc())
                if sum(weights) > 0.0:
                    _redistribute = True
                    block.set_yc(sum([
                        weights[i] * positions[i]
                        for i in range(len(weights))
                    ]) / sum(weights))
            if _redistribute:
                sort_key = [bisect_left(old_mid_heights, col.get_yc())
                            for col in _column]
                cs, _sort_key = zip(
                    *sorted(
                        zip(
                            list(range(nbr_blocks)),
                            sort_key,
                        ),
                        key=lambda x: x[1]
                    )
                )
                self._reorder_column(col_id, ordering=cs)
                # self._columns[col_id] = [
                #     _column[_k] for _k in cs
                # ]

                # redistribute them
                self._update_ycoords(col_id, col_hspace, layout)
                old_mid_heights = [block.get_yc()
                                   for block in self._columns[col_id]]
            else:
                break

    def _pairwise_swapping(self, col_id):
        _column = self._columns[col_id]
        # TODO: does not really make sense to recompute them here
        nbr_blocks = len(_column)
        col_hspace = self.get_column_hspace(col_id)
        for _ in range(int(0.5 * nbr_blocks)):
            for i in range(1, nbr_blocks):
                b1 = _column[i - 1]
                b2 = _column[i]
                if self._swap_clusters((b1, b2), col_hspace,
                                       'backwards'):
                    b2.set_y(b1.get_y())
                    b1.set_y(
                        b2.get_y() + b2.get_height() + col_hspace
                    )
                    _column[i - 1] = b2
                    _column[i] = b1
        for _ in range(int(0.5 * nbr_blocks)):
            for i in range(1, nbr_blocks):
                b1 = _column[nbr_blocks - i - 1]
                b2 = _column[nbr_blocks - i]
                if self._swap_clusters((b1, b2), col_hspace,
                                       'backwards'):
                    b2.set_y(b1.get_y())
                    b1.set_y(
                        b2.get_y() + b2.get_height() + col_hspace
                    )
                    self._columns[col_id][nbr_blocks - i - 1] = b2
                    self._columns[col_id][nbr_blocks - i] = b1

    def _reorder_column(self, col_id, ordering):
        """Update the ordering of blocks in a column"""
        _column = self._columns[col_id]
        self._columns[col_id] = [_column[newid] for newid in ordering]

    # NOTE: almost indep on self
    def _update_ycoords(self, column: int, hspace, layout):
        """
        Update the y coordinate of the blocks in a column based on the
        diagrams vertical offset, the layout chosen for this column and the
        order of the blocks.

        Parameters
        ----------
        column : int
            Index of the column to reorder.
        """
        displace = self._yoff
        _column = self._columns[column]
        for block in _column:
            block.set_y(displace)
            displace += block.get_height() + hspace
        # now offset to center
        low = _column[0].get_y()  # this is just self._yoff
        # this is just `displace`:
        high = _column[-1].get_y() + _column[-1].get_height()
        if layout == 'centered':
            cent_offset = low + 0.5 * (high - low)
            # _h_clusters = 0.5 * len(clusters)
            # cent_idx = int(_h_clusters) - 1 \
            #     if _h_clusters.is_integer() \
            #     else int(_h_clusters)
            # cent_offest = clusters[cent_idx].mid_height
            for block in _column:
                block.set_y(block.get_y() - cent_offset)
        elif layout == 'top':
            for block in _column:
                block.set_y(block.get_y() - high)

    def _swap_clusters(self, blocks, hspace, direction='backwards'):
        """
        Check if swapping to blocks leads to shorter vertical flow distances.
        """
        # _blocks = [self._blocks[bid] for bid in (bid1, bid2)]
        squared_diff = {}

        for block in blocks:
            weights = []
            sqdiff = []
            if direction in ['both', 'backwards']:
                for flow in block.inflows:
                    if flow.source is not None:
                        weights.append(flow.flow_width)
                        sqdiff.append(abs(
                            block.get_yc() - flow.source.get_yc()
                        ))
            if direction in ['both', 'forwards']:
                for flow in block.outflows:
                    if flow.target is not None:
                        weights.append(flow.flow_width)
                        sqdiff.append(abs(
                            block.get_yc() - flow.target.get_yc()
                        ))
            if sum(weights) > 0.0:
                squared_diff[block] = sum(
                    [weights[i] * sqdiff[i]
                        for i in range(len(weights))]
                ) / sum(weights)
        # inverse order and check again
        # TODO: check why this assert statement fails
        # print(n1.get_y(), n2.get_y())
        # assert n1.get_y() < n2.get_y()
        # TODO: Cannot recreate the thought process behind this...
        inv_mid_height = [
            blocks[0].get_y() + blocks[1].get_height() + hspace + 0.5 * blocks[0].get_height(),
            blocks[0].get_y() + 0.5 * blocks[1].get_height()
        ]
        # inv_mid_height = {
        #     bid1: blocks[0].get_y() + blocks[1].get_height() + hspace + 0.5 * blocks[0].get_height(),
        #     bid2: blocks[0].get_y() + 0.5 * blocks[1].get_height()
        # }
        squared_diff_inf = {}
        for i, block in enumerate(blocks):
            weights = []
            sqdiff = []
            if direction in ['both', 'backwards']:
                for flow in block.inflows:
                    if flow.source is not None:
                        weights.append(flow.flow_width)
                        sqdiff.append(abs(
                            inv_mid_height[i] - flow.source.get_yc()
                        ))
            if direction in ['both', 'forwards']:
                for flow in block.outflows:
                    if flow.target is not None:
                        weights.append(flow.flow_width)
                        sqdiff.append(
                            abs(inv_mid_height[i] - flow.target.get_yc())
                        )
            if sum(weights) > 0.0:
                squared_diff_inf[block] = sum([
                    weights[i] * sqdiff[i]
                    for i in range(len(weights))
                ]) / sum(weights)
        if sum(squared_diff.values()) > sum(squared_diff_inf.values()):
            return True
        else:
            return False

    # TODO: automate the attribute separation or separate by construction
    @classmethod
    def separate_kwargs(cls, kwargs):
        """Separate all relevant kwargs for the init if a SubDiagram."""
        sdkwargs, other_kwargs = dict(), dict()
        sd_args = ['x', 'columns', 'match_original', 'yoff', 'layout',
                   'hspace_combine', 'label_margin', 'cmap', 'norm',
                   'cmap_array', 'blockprops']
        for k, v in kwargs.items():
            if k in sd_args:
                sdkwargs[k] = v
            else:
                other_kwargs[k] = v
        return sdkwargs, other_kwargs


class Alluvial:
    """
    Alluvial diagram.

        Alluvial diagrams are a variant of flow diagram designed to represent
        changes in classifications, in particular changes in network
        structure over time.
        `Wikipedia (23/1/2021) <https://en.wikipedia.org/wiki/Alluvial_diagram>`_
    """
    # @docstring.dedent_interpd
    def __init__(self, x=None, ax=None, y_pos='overwrite', tags=None,
                 cluster_w_spacing=1, blockprops=None,
                 flow_kwargs={},
                 label_kwargs={}, **kwargs):
        """
        Create a new Alluvial instance.


        Parameters
        ===========

        clusters: dict[str, dict], dict[float, list] or list[list]
          .. warning::
            to remove
          You have 2 options to create an Alluvial diagram::

          raw data: dict[str, dict]
            *NOT IMPLEMENTED YET*

            Provide for each cluster (`key`) a dictionary specifying the
            out-flows in the form of a dictionary (`key`: cluster, `value`: flow).

            .. note::

              The `key` ``None`` can hold a dictionary specifying flows from/to
              outside the system. If is present in the provided dictionary it
              allows to specify in-flows, i.e. data source that were not present
              at the previous slice.

              If it is present in the out-flows of a cluster, the specified amount
              simply vanishes and will not lead to a flow.

          collections of :obj:`.Cluster`: dict[float, list] and list[list]
            If a `list` is provided each element must be a `list`
            of :obj:`.Cluster` objects. A `dictionary` must provide a `list` of
            :obj:`.Cluster` (*value*) for a horizontal position (*key*), e.g.
            ``{1.0: [c11, c12, ...], 2.0: [c21, c22, ...], ...}``.

        x : array-like, optional
          The x coordinates for the columns in the Alluvial diagram.

          They will be used as default coordinates whenever a sub-diagram is
          added that does not specify it's own x coordinates.

          If not given, the x coordinates will be inferred as soon as the first
          diagram is added and default to the range of the number of columns in
          the diagram.

        ax: `~.axes.Axes`
          Axes onto which the Alluvial diagram should be drawn.
          If *ax* is not provided a new Axes instance will be created.
        y_pos: str
          **options:** ``'overwrite'``, ``'keep'``, ``'complement'``, ``'sorted'``

          'overwrite':
             Ignore existing y coordinates for a block and set the vertical
             position to minimize the vertical displacements of all flows.
          'keep':
            use the block's :meth:`_Block.get_y`. If a block has no y
            coordinate set this raises an exception.
          'complement':
            use the block's :meth:`_Block.get_y` if
            set. Blocks without y position are positioned relative to the other
            blocks by minimizing the vertical displacements of all flows.
          'sorted':
            NOT IMPLEMENTED YET
        cluster_w_spacing: float, int (default=1)
          Vertical spacing between blocks.
        blockprops : dict, optional
          The properties used to draw the blocks. *blockprops* accepts the
          following specific keyword arguments:

          - TODO

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

        flow_kwargs: dict (default={})
          dictionary styling the :obj:`~matplotlib.patches.PathPatch` of flows.

          for a list of available options see
          :class:`~matplotlib.patches.PathPatch`

          Note
          -----

            Passing a string to `facecolor` and/or `edgecolor` allows to color
            flows relative to the color of their source or target blocks.

            ``'source'`` or ``'target'``:
              will set the facecolor equal to the color of the respective block.

              ``'cluster'`` *and* ``'source'`` *are equivalent.*

            ``'<cluster>_reside'`` or ``'<cluster>_migration'``:
              set the color based on whether source and target block have the
              same color or not. ``'<cluster>'`` should be either
              ``'source'`` or ``'target'`` and determines the
              block from which the color is taken.

              **Examples:**

              ``facecolor='cluster_reside'``
                set `facecolor` to the color of the source block if both source
                and target block are of the same color.

              ``edgecolor='cluster_migration'``
                set `edgecolor` to the color of the source block if source and
                target block are of different colors.

        **kwargs optional parameter:
            x_lim: tuple
              the horizontal limit values for the :class:`~matplotlib.axes.Axes`.
            y_lim: tuple
              the vertical limit values for the :class:`~matplotlib.axes.Axes`.
            set_x_pos: bool
              if `clusters` is a dict then the key is set for all blocks
            cluster_width: float
              (NOT IMPLEMENTED) overwrites width of all blocks
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
              (block labels) as values. The position of blocks (tuples)
              are swapped.
            redistribute_vertically: int (default=4)
              how often the vertical pairwise swapping of blocks at a given time
              point should be performed.
            y_offset: float
              offsets the vertical position of each block by this amount.

              .. note::

                This ca be used to draw multiple alluvial diagrams on the same
                :obj:`~matplotlib.axes.Axes` by simply calling
                :class:`~.Alluvial` repeatedly with changing offset value, thus
                stacking alluvial diagrams.

        Attributes
        ===========

        clusters: dict
          Holds for each vertical position a list of :obj:`._Block` objects.
        """
        if x is not None:
            self._x = _to_valid_sequence(x, 'x')
        else:
            self._x = None
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
        self._blockprops = blockprops
        self._flow_kwargs = flow_kwargs
        self._label_kwargs = label_kwargs

        self._diagrams = []
        self._tags = []
        self._extouts = []
        self._diagc = 0
        self._dlabels = []
        self._kwargs = {}
        # TODO: unused for now
        self._dirty = False  # indicate if between diagram flows exist

        # TODO: if arguments for add are passed they cannot remain in kwargs
        if kwargs:
            _kwargs = kwargs.copy()
            # strap arguments for an add call
            flows = _kwargs.pop('flows', None)
            ext = _kwargs.pop('ext', None)
            label = _kwargs.pop('label', '')
            fractionflow = _kwargs.pop('fractionflow', False)
            tags = _kwargs.pop('tags', None)
            # keep styling related kwargs
            self._kwargs = _kwargs
            # draw a diagram if *flows* were provided
            if flows is not None or ext is not None:
                sdkw, self._kwargs = SubDiagram.separate_kwargs(
                    self._kwargs)
                self.add(flows=flows, ext=ext, extout=None, x=self._x,
                         label=label, yoff=0,
                         fractionflow=fractionflow, tags=tags,
                         **sdkw)
                self.finish()

        # # if blocks are given in a list of lists (each list is a x position)
        # self._set_x_pos = kwargs.get('set_x_pos', True)
        # self._redistribute_vertically = kwargs.get(
        #     'redistribute_vertically',
        #     4
        # )
        # self.with_cluster_labels = kwargs.get('with_cluster_labels', True)
        # self.format_xaxis = kwargs.get('format_xaxis', True)
        # self._x_axis_offset = kwargs.get('x_axis_offset', 0.0)
        # self._fill_figure = kwargs.get('fill_figure', False)
        # self._invisible_y = kwargs.get('invisible_y', True)
        # self._invisible_x = kwargs.get('invisible_x', False)
        # self.y_offset = kwargs.get('y_offset', 0)
        # self.y_fix = kwargs.get('y_fix', None)

        # # this goes to the add method anyways
        # if isinstance(clusters, dict):
        #     self.clusters = clusters
        # else:
        #     self.clusters = {}
        #     for cluster in clusters:
        #         try:
        #             self.clusters[cluster.x].append(cluster)
        #         except KeyError:
        #             self.clusters[cluster.x] = [cluster]
        # self.x_positions = sorted(self.clusters.keys())
        # set the x positions correctly for the clusters
        # if self._set_x_pos:
        #     for x_pos in self.x_positions:
        #         for cluster in self.clusters[x_pos]:
        #             cluster = cluster.set_x_pos(x_pos)
        # self._x_dates = False
        # _minor_tick = 'months'
        # cluster_widths = []
        # if isinstance(self.x_positions[0], datetime):
        #     # assign date locator/formatter to the x-axis to get proper labels
        #     if self.format_xaxis:
        #         locator = mdates.AutoDateLocator(minticks=3)
        #         formatter = mdates.AutoDateFormatter(locator)
        #         self.ax.xaxis.set_major_locator(locator)
        #         self.ax.xaxis.set_major_formatter(formatter)
        #     self._x_dates = True
        #     if (self.x_positions[-1] - self.x_positions[0]).days < 2 * 30:
        #         _minor_tick = 'weeks'
        #     self.clusters = {
        #         mdates.date2num(x_pos): self.clusters[x_pos]
        #         for x_pos in self.x_positions
        #     }
        #     self.x_positions = sorted(self.clusters.keys())
        #     for x_pos in self.x_positions:
        #         for cluster in self.clusters[x_pos]:
        #             # in days (same as mdates.date2num)
        #             cluster.set_width(cluster.width.total_seconds() / 60 / 60 / 24)
        #             cluster_widths.append(cluster.get_width())
        #             if cluster.label_margin is not None:
        #                 _h_margin = cluster.label_margin[0].total_seconds() / 60 / 60 / 24
        #                 cluster.label_margin = (
        #                     _h_margin, cluster.label_margin[1]
        #                 )
        #             cluster.set_x_pos(mdates.date2num(cluster.x))

        # # TODO: set the cluster.set_width property
        # else:
        #     for x_pos in self.x_positions:
        #         for cluster in self.clusters[x_pos]:
        #             cluster_widths.append(cluster.get_width())
        # self.cluster_width = kwargs.get('cluster_width', None)
        # self.x_lim = kwargs.get(
        #     'x_lim',
        #     (
        #         self.x_positions[0] - 2 * min(cluster_widths),
        #         # - 2 * self.clusters[self.x_positions[0]][0].get_width(),
        #         self.x_positions[-1] + 2 * min(cluster_widths),
        #         # + 2 * self.clusters[self.x_positions[-1]][0].get_width(),
        #     )
        # )
        # self.y_min, self.y_max = None, None
        # if self.y_pos == 'overwrite':
        #     # reset the vertical positions for each row
        #     for x_pos in self.x_positions:
        #         self.distribute_blocks(x_pos)
        #     for x_pos in self.x_positions:
        #         self._move_new_clusters(x_pos)
        #     for x_pos in self.x_positions:
        #         nbr_clusters = len(self.clusters[x_pos])
        #         for _ in range(nbr_clusters):
        #             for i in range(1, nbr_clusters):
        #                 n1 = self.clusters[x_pos][nbr_clusters - i - 1]
        #                 n2 = self.clusters[x_pos][nbr_clusters - i]
        #                 if self._swap_clusters(n1, n2, 'forwards'):
        #                     n2.set_y(n1.get_y())
        #                     n1.set_y(
        #                         n2.get_y() + n2.get_height() + self.cluster_w_spacing
        #                     )
        #                     self.clusters[x_pos][nbr_clusters - i] = n1
        #                     self.clusters[x_pos][nbr_clusters - i - 1] = n2
        # else:
        #     # TODO: keep and complement
        #     pass
        # if isinstance(self.y_fix, dict):
        #     # TODO: allow to directly get the index given the cluster label
        #     for x_pos in self.y_fix:
        #         for st in self.y_fix[x_pos]:
        #             n1_idx, n2_idx = (
        #                 i for i, l in enumerate(
        #                     map(lambda x: x.label, self.clusters[x_pos])
        #                 )
        #                 if l in st
        #             )
        #             self.clusters[x_pos][n1_idx], self.clusters[x_pos][n2_idx] = self.clusters[
        #                 x_pos
        #             ][n2_idx], self.clusters[x_pos][n1_idx]
        #             self._update_ycoords(x_pos, self.cluster_w_spacing)

        # # positions are set
        # self.y_lim = kwargs.get('y_lim', (self.y_min, self.y_max))
        # # set the colors
        # # TODO

        # # now draw
        # patch_collection = self.get_patchcollection(
        #     cluster_kwargs=self._cluster_kwargs,
        #     flow_kwargs=self._flow_kwargs
        # )
        # self.ax.add_collection(patch_collection)
        # if self.with_cluster_labels:
        #     label_collection = self.get_labelcollection(**self._label_kwargs)
        #     if label_collection:
        #         for label in label_collection:
        #             self.ax.annotate(**label)
        # self.ax.set_xlim(
        #     *self.x_lim
        # )
        # self.ax.set_ylim(
        #     *self.y_lim
        # )
        # if self._fill_figure:
        #     self.ax.set_position(
        #         [
        #             0.0,
        #             self._x_axis_offset,
        #             0.99,
        #             1.0 - self._x_axis_offset
        #         ]
        #     )
        # if self._invisible_y:
        #     self.ax.get_yaxis().set_visible(False)
        # if self._invisible_x:
        #     self.ax.get_xaxis().set_visible(False)
        # self.ax.spines['right'].set_color('none')
        # self.ax.spines['left'].set_color('none')
        # self.ax.spines['top'].set_color('none')
        # self.ax.spines['bottom'].set_color('none')
        # if isinstance(self.x_positions[0], datetime) and self.format_xaxis:
        #     self.set_dates_xaxis(_minor_tick)

    def get_x(self, ):
        """Return the sequence of x coordinates of the Alluvial diagram"""
        return self._x

    def set_x(self, x):
        """
        Set the sequence of x coordinates for all columns in the Alluvial
        diagram.

        Parameters
        ----------
        x : sequence of scalars
            Sequence of M scalars setting the x coordinates for all columns in
            all subdiagrams.

        Note that setting the coordinates will have no effect on subdiagrams
        that were already added. Only further calls of :meth:`add` will use the
        new x coordinates as default horizontal positioning.
        """
        if x is None:
            self._x = None
        else:
            self._x = _to_valid_sequence(x, 'x')

    def get_diagrams(self):
        """Get all sub-diagrams."""
        return self._diagrams

    def _create_columns(self, cinit, flows, ext, extout, fractionflow):
        """
        Create the columns of an alluvial diagram.
        """
        # create the columns
        columns = [cinit]
        if flows is not None:
            for flow, e in zip(flows, ext[1:]):
                _col = e
                if len(flow):
                    if fractionflow:
                        _flow = flow.dot(columns[-1])
                    else:
                        _flow = flow.sum(1)
                    _col = _flow + e
                columns.append(_col)

        if extout is not None:
            # TODO: check extout format
            pass
        return columns

    def add(self, flows, ext=None, extout=None, x=None, label=None, yoff=0,
            fractionflow=False, tags=None, **kwargs):
        r"""
        Add an Alluvial diagram with a vertical offset.
        The offset must be provided in the same units as the block sizes.

        Parameters
        ----------
        flows : sequence of array-like objects
            The flows between columns of the Alluvial diagram.

            *flows[i]* determines the flow matrix :math:`\mathbf{M}^i` from
            blocks in column *i* to the blocks in column *i+1*. The entry
            `\mathbf{M}^i_{k,l}` gives the amount that flows from block `l` in
            column *i* to block `k` in column *i+1*.

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
        x : array-like, optional
            The x coordinates of the columns.

            When provided, the added diagram ignores the x coordinates that might
            have been provided on initiation of the `.Alluvial` instance or any
            previous :meth:`.Alluvial.add` call.

            If the `.Alluvial` instance had no values set for the x coordinates
            *x* will be set to the new default.

            If not provided and no x coordinates have been set previously, then
            the x coordinates will default to the range defined by the number
            of columns in the diagram to add.
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
        tags : sequence or str, optional
            Tagging of the blocks. Tags can be provided in the following
            formats:

            - String, allowed are {'column', 'index'}.
              If *tags* is set to 'column', all blocks in a column get the same
              tag. If 'index' is used, in each column the blocks is tagged by
              their index in the column.
            - Sequence of M tags, providing for each column a separate tag.
            - Sequence of list of tags, providing fore each block in each
              column a tag.

            If a sequence is provided, the tags can be any hashable object.

            Note that *tags* should be used in combination with *tagprops* in
            order to specify the styling for each tag.
        tagprops : dict, optional
            Provide for each tag a dictionary that specifies the styling of
            blocks with this tag. See :meth:`
        Other Parameters
        ----------------
        **kwargs : `.SubDiagram` properties

            .. TODO:
                get doc from SubDiagram.__init__

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
          by *flow[i]*, :math:`\textbf{1}` is a vector of ones of shape (N) and
          :math:`\textbf{e}_{i+1}` is the external inflow vector of shape (P),
          given by *e[i+1]*.
        - *fractionflow* is True:

          The block sizes in column *i+1* depend directly on the block sizes of
          column *i*, :math:`\textbf{c}_{i}`, and are given by:

          .. math::
              \textbf{c}_{i+1}=\mathbf{F}_i\cdot\textbf{c}_i+\textbf{e}_{i+1},

          where :math:`\mathbf{F}_i` is the flow matrix of shape (P, N), given
          by *flow[i]*, :math:`\textbf{c}_i` the vector of N block sizes in
          column *i* and :math:`\textbf{e}_{i+1}` the external inflow vector of
          shape (P) given by *e[i+1]*.
        """
        _kwargs = dict(kwargs)
        # check the provided arguments
        # TODO: make sure empty flows are accepted
        flows = _to_valid_sequence(flows, 'flows')
        if flows is not None:
            nbr_cols = len(flows) + 1
        else:
            nbr_cols = None
        # check ext and set initial column
        if ext is None:
            if fractionflow:
                raise TypeError("'ext' cannot be None if 'fractionflow' is"
                                " True: You need to provide at least the block"
                                " sizes for the first column of the Alluvial"
                                " diagram if the flows are given as"
                                " fractions.")
            elif nbr_cols is None:
                raise TypeError("'ext' cannot be None if 'flows' is None too."
                                " You need to provide either `flows` or `ext`"
                                " to create an Alluvial diagram.")
            ext = np.zeros(nbr_cols)
            # Note: extout from the first column are ignored in the
            # construction of the first columns
            cinit = flows[0].sum(0)
        else:
            ext = _to_valid_sequence(ext, 'ext')
            if isinstance(ext[0], np.ndarray):
                cinit = ext[0]
                # if no flows were provided
                if nbr_cols is None:
                    nbr_cols = len(ext)
                    flows = [[] for _ in range(nbr_cols - 1)]
            else:
                cinit = ext[:]
                if nbr_cols is None:
                    nbr_cols = 1
                ext = np.zeros(nbr_cols)  # Note: we overwrite ext in this case

        columns = self._create_columns(cinit, flows, ext, extout,
                                       fractionflow)
        # here we can process flows further before passing it to SubDiagram
        # # but there is nothing further to do, it seems...

        if x is not None:
            x = _to_valid_sequence(x, 'x')
        else:
            # use default if not specified
            x = self._x or [i for i in range(len(columns))]
        # NOTE: add should anyways only accept kws for subdiagram
        # sdkw, otherkw = SubDiagram.separate_kwargs(_kwargs)
        _blockprops = _kwargs.pop('blockprops', self._blockprops)
        diagram = SubDiagram(x=x, columns=columns, flows=flows,
                             fractionflow=fractionflow, label=label,
                             yoff=yoff, blockprops=_blockprops, **_kwargs)
        # add the new subdiagram
        # get the x coordinates
        # TODO: cannot pass columns here. columns are a list of list[float]
        # and not list[dict] (as set for SubDiagram for now)
        self._add_diagram(diagram)
        self._dlabels.append(label or f'diagram-{self._diagc}')
        self._extouts.append(extout)
        self._diagc += 1

        # # Create the sequence of clusterings
        # time_points = [0, 4, 9, 14, 18.2]
        # # Define the cluster sizes per snapshot
        # # at each time point {cluster_id: cluster_size})
        # cluster_sizes = [{0: 3}, {0: 5}, {0: 3, 1: 2}, {0: 5}, {0: 4}]
        # # Define the membership flows between neighbouring clusterings
        # between_flows = [
        #     {(0, 0): 3},  # key: (from cluster, to cluster), value: size
        #     {(0, 0): 3, (0, 1): 2},
        #     {(0, 0): 3, (1, 0): 2},
        #     {(0, 0): 4}
        # ]
        # # set the colors
        # cluster_color = {0: "C1", 1: "C2"}
        # # create a dictionary with the time points as keys and a list of clusters
        # # as values
        # clustering_sequence = {}
        # for tp, clustering in enumerate(cluster_sizes):
        #     clustering_sequence[time_points[tp]] = [
        #         _Block(
        #             height=clustering[cid],
        #             label="{0}".format(cid),
        #             facecolor=cluster_color[cid],
        #         ) for cid in clustering
        #     ]
        # # now create the flows between the clusters
        # for tidx, tp in enumerate(time_points[1:]):
        #     flows = between_flows[tidx]
        #     for from_csid, to_csid in flows:
        #         _Flow(
        #             flow=flows[(from_csid, to_csid)],
        #             source=clustering_sequence[time_points[tidx]][from_csid],
        #             target=clustering_sequence[tp][to_csid],
        #             facecolor='source'
        #         )

    def _add_diagram(self, diagram):
        """
        Add a new subdiagram to the Alluvial diagram.
        """
        self._diagrams.append(diagram)

    def _create_collections(self):
        for diagram in self._diagrams:
            # create a PatchCollection out of all non-tagged blocks
            diag_zorder = 4
            diagram.determine_layout()
            diagram.create_artist(zorder=diag_zorder, **self._kwargs)
            self.ax.add_collection(diagram.get_artist())
        for tag in self._tags:
            # creat a PatchCollection for each tag
            # tag_zorder = 5
            tag_collection = None
            self.ax.add_collection(tag_collection)
        # create patches for regular flows
        # create the patches for extout

    def finish(self,):
        # TODO: distribute all blocks in all cols in all diagrams
        self._create_collections()
        pass

    # TODO uncomment once in mpl
    # @docstring.dedent_interpd
    def register_tag(self, tag, **kwargs):
        """
        Register a new tag.

        Parameters
        ----------
        tag : Any
            A hashable object used as identifier for the tag.

        Other Parameters
        ----------------
        **kwargs : `.Collection` properties
            Define the styling to apply to all blocks with this tag:

            %(Collection_kwdoc)s

        Note that if the tag has already been registered, a warning message is
        issued and the call will have no effect on the existing tag. If you
        want to update the styling of an existing tag, use :meth:`update_tag`
        instead.
        """
        if tag in self._tags:
            _log.warning(
                f"The tag '{tag}' has already been registered. Registering an"
                " existing tag again has no effect. You must use *update_tag*"
                "if you want to change the styling of an existing tag."
            )
            return None

    def update_tag(self, tag, **kwargs):
        pass

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

    def _move_new_clusters(self, x_pos):
        r"""
        This method redistributes flows without in-flow so to minimize the
        vertical displacement of out-flows.

        Parameters
        -----------
        x_pos: float
          The horizontal position where new clusters without in-flow should be
          distributed. This must be a `key` of the
          :attr:`~.Alluvial.clusters` attribute.

        Once the clusters are distributed for all x positions this method
        redistributes within a given x_positions the clusters that have no
        inflow but out flows. The clusters are moved closer (vertically) to
        the target clusters of the out flow(es).
        """
        old_mid_heights = [
            cluster.mid_height for cluster in self.clusters[x_pos]
        ]
        _redistribute = False
        for cluster in self.clusters[x_pos]:
            if sum([_flow.flow_width for _flow in cluster.inflows]) == 0.0:
                weights = []
                positions = []
                for out_flow in cluster.outflows:
                    if out_flow.target is not None:
                        weights.append(out_flow.flow_width)
                        positions.append(out_flow.target.mid_height)
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
            self._update_ycoords(x_pos, self.cluster_w_spacing)

    # unused
    def get_patchcollection(
        self, match_original=True,
        cluster_kwargs={},
        flow_kwargs={},
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
        flows = []
        if cluster_kwargs is None:
            cluster_kwargs = dict()
        for x_pos in self.x_positions:
            out_flows = []
            for cluster in self.clusters[x_pos]:
                # TODO: set color
                # _cluster_color
                cluster.set_y(cluster.get_y() + self.y_offset)
                cluster_patches.append(
                    cluster.get_artist(
                        **cluster_kwargs
                    )
                )
                # sort the flows for minimal overlap
                cluster.set_loc_out_flows()
                cluster.sort_in_flows()
                cluster.sort_out_flows()
                cluster.set_anchor_in_flows()
                cluster.set_anchor_out_flows()
                out_flows.extend(cluster.outflows)
            flows.append(out_flows)
        flow_patches = []
        for out_flows in flows:
            for out_flow in out_flows:
                flow_patches.append(out_flow.get_artist(**flow_kwargs))
        all_patches = []
        all_patches.extend(flow_patches)
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
                            cluster.get_y() + _v_margin
                        )
                    }
                    cluster_label.update(kwargs)
                    cluster_labels.append(cluster_label)
        return cluster_labels

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


class AlluvialHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        # TODO:
        # construct a simple alluvial diag
        patch = _Block(height=height, xa=x0, ya=y0, width=width, fc='red',
                       transform=handlebox.get_transform())
        # patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red',
        #                            edgecolor='black', hatch='xx', lw=3,
        #                            transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


# set the legend handler for an alluvial diagram
Legend.update_default_handler_map({SubDiagram: AlluvialHandler()})
