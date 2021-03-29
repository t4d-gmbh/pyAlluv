import functools
import inspect
import logging
import itertools
from copy import copy
from collections import defaultdict
from weakref import WeakValueDictionary
import numpy as np
import matplotlib as mpl
from datetime import datetime
# from bisect import bisect_left
from matplotlib.cbook import index_of
from matplotlib.cbook import normalize_kwargs as normed_kws
from matplotlib.collections import PatchCollection
from matplotlib import _api, cbook, cm
from matplotlib.artist import Artist
# from matplotlib import docstring
# from . import (_api, _path, artist, cbook, cm, colors as mcolors, docstring,
# from matplotlib import transforms
# from matplotlib import _api
# import matplotlib.dates as mdates
from matplotlib.path import Path
from matplotlib.patches import Rectangle
from matplotlib.rcsetup import cycler
from matplotlib.legend import Legend
from matplotlib.dates import date2num, AutoDateLocator, AutoDateFormatter
import matplotlib.ticker as mticker
import matplotlib.patches as patches

_log = logging.getLogger(__name__)

__author__ = 'Jonas I. Liechti'


# TODO: check if cbook has something for this
def _to_valid_arrays(data, attribute, dtype=None):
    """TODO: write docstring"""
    if data is None:
        return None
    if hasattr(data, 'index') and hasattr(data, 'values'):
        return data.values
    try:
        data = np.asarray(data, dtype=dtype)
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
                    data.append(np.asarray(d, dtype=dtype))
                except (TypeError, ValueError):
                    raise ValueError("{attr} can only contain array-like"
                                     " objects which is not the case at index"
                                     "{eindex}:\n"
                                     "{entry}".format(attr=attribute, eindex=i,
                                                      entry=d))
    return data


def memship_to_column(membership, absentval=None):
    """
    Convert a membership list into a column in an Alluvial diagram.

    Parameters
    ----------
    membership : array-like
        Associates individuals (typically a node in a network) to a group
        or class. The position in *membership* identifies an individual and
        the value the class or group it is associated to.
    absentval : scalar or None
        A specific value to indicate that a individual is not associated to any
        group or class.

    Returns
    -------
    nbr_blocks : int
        The number of groups or classes.
    column : array-like
        Column of individual count in each group or class.
    """
    if absentval in (None, np.nan):
        _mask = ~np.isnan(membership)
    else:
        _mask = membership != absentval
    nbr_blocks = int(np.amax(membership, where=_mask, initial=-1) + 1)
    block_ids, counts = np.unique(membership[_mask], return_counts=True)
    col = np.zeros(nbr_blocks)
    for bid, count in zip(block_ids.astype(int), counts):
        col[bid] = count
    return nbr_blocks, col


def _between_memships_flow(flow_dims, membership, membership_next,
                           absentval=None):
    """
    Return a flow matrix from *membership* to *membership_next* along with a
    potential external inflow to the latter membership list. As external inflow
    we consider individuals with *absentval* in the first membership list.

    Parameters
    ----------
    flow_dims : tuple[int, int]
        The number of groups in *membership_next* and *membership_last*.
    membership : array-like
        Group (or class) association of individual. The position identifies an
        individual and the value the class or group it is associated to.
    membership_next : array
        2nd membership list.
    absentval : scalar or None
        A specific value to indicate that a individual is not associated to any
        group or class.

    Returns
    -------
    flowmatrix : 2darray
        Flowmatrix between two membership lists.
    ext : array
        Count the number of individuals only present in *membership_next* per
        group or class.


    Note that the index must match between the two membership lists such that
    the first element in both lists corresponds to the same individual.
    """
    ext = np.zeros(flow_dims[0])
    flowmatrix = np.zeros(flow_dims, dtype=int)
    for m1, m2 in zip(membership.astype(int),
                      membership_next.astype(int)):
        if m1 == absentval:  # node was absent in last
            if m2 != absentval:  # node is present in next
                ext[m2] += 1
            else:  # node also absent in next:
                pass
        else:
            if m2 != absentval:  # node is present in both
                flowmatrix[m2, m1] += 1
            else:  # node is absent in next
                pass  # currently these generate no visual effects
    return flowmatrix, ext


def _separate_selector(*selector, max_selectors=3):
    """Returns slices based on the selectors."""
    nbr_args = len(selector)

    # convert all input to slices
    def _to_slice(arg):
        if isinstance(arg, int):
            return slice(arg, arg + 1 if arg != -1 else None)
        elif isinstance(arg, slice):
            return arg
        elif arg is None:
            return slice(None, None)
        else:
            return slice(*arg)
    selector = [_to_slice(arg) for arg in selector]
    return (max_selectors - nbr_args) * [slice(None, None)] + selector


def _expose_artist_getters_and_setters(cls):
    """
    Wrapper that exposes all setters and getters from a subclass of Artist in
    order to create a functional proxy class.
    """
    alias_mapping = getattr(cls._artistcls, "_alias_map", {})
    to_canonical = {alias: canonical for canonical, alias_list in
                    alias_mapping.items() for alias in alias_list}

    def make_proxy_getter(name):  # Enforce a closure over *name*.
        @functools.wraps(getattr(cls._artistcls, name))
        def method(self, *args, **kwargs):
            if self._artist is not None:
                return getattr(self._artist, name)(*args, **kwargs)
            else:
                # in principle we could return the attribute.
                # However, since getters and setters for layout relevant
                # properties need to be redefined in the particular proxy
                # class, an Error is raised here.
                # return self._kwargs.get(aname.replace("get_", ""), None)
                raise NotImplementedError(f"{cls} has no implementation"
                                          f" for '{name}', thus accessing"
                                          f" the property '{name[4:]}' is"
                                          " allowed only after the artist"
                                          " is initiated.")
        return method

    def make_proxy_setter(name, argnames):  # Enforce a closure over *name*.
        @functools.wraps(getattr(cls._artistcls, name))
        def method(self, *args, **kwargs):
            _prop = name.replace("set_", "")
            prop = to_canonical.get(_prop, _prop)
            if prop in ('facecolor', 'edgecolor', 'linewidth', 'linestyle',
                        'antialiased'):
                self._is_styled = True  # set a flag for individual styling
                setattr(self, f'own_{prop}', True)  # flag what is styled

            if self._artist is not None:
                return getattr(self._artist, name)(*args, **kwargs)
            else:
                # get the value either by position or by key
                if args:
                    value = args[0]
                else:
                    value = kwargs[argnames[0]]
                self._kwargs[prop] = value
        return method

    # get the getters and setters from _artist
    artistcls_fcts = vars(cls._artistcls)
    with _api.suppress_matplotlib_deprecation_warning():
        artistcls_fcts = inspect.getmembers(cls._artistcls,
                                            predicate=inspect.isfunction)
    for aname, attribute in artistcls_fcts:
        specs = inspect.getfullargspec(attribute)
        argnames = specs[0]  # raise error if argnames contains more than 1?
        # the getters
        if aname.startswith("get_") and callable(attribute) \
                and not hasattr(cls, aname):
            method = make_proxy_getter(aname)
            method.__name__ = aname
            method.__doc__ = "Proxy method for" \
                             f":meth:`{cls._artistcls.__name__}.{aname}'"
            setattr(cls, aname, method)
        # the setters
        elif aname.startswith("set_") and callable(attribute) \
                and not hasattr(cls, aname):
            method = make_proxy_setter(aname, argnames)
            method.__name__ = aname
            method.__doc__ = "Proxy method for" \
                             f":meth:`{cls._artistcls.__name__}.{aname}'"
            setattr(cls, aname, method)
    return cls


def init_defaults(callabs, cls=None):
    """Returns names of all parameter of callabs that have default values."""
    if cls is None:  # Return the actual class decorator.
        return functools.partial(init_defaults, callabs)

    layout_params = set()
    for callab in callabs:
        params = inspect.signature(callab).parameters
        layout_params.update((param for param in params
                              if params[param].default != inspect._empty))
    # remove all parameters present in cls.__init__
    layout_params = layout_params.difference(
        {param for param in inspect.signature(cls).parameters}
    )
    cls._init_defaults = layout_params
    return cls


class _ArtistProxy:
    """
    Proxy class for `Artist` subclasses used to draw the various element in an
    alluvial diagram.

    This class assembles common properties and relations to :class:`Patch`
    of the elements present in an Alluvial diagram.
    """
    _artistcls = Artist  # derived must overwrite

    def __init__(self, label=None, tags=None, **kwargs):
        """
        Parameters
        ----------
        label : str, optonal
           The label of this _ArtistProxy.
        tags : sequence, optional
           A set of `._Tag` instances.
        """
        # TODO: not sure if stale is needed for this
        self.stale = True   # This only tracks form/layout but no style changes
        self._tags = []     # list of tag identifiers (label of a tag)
        if tags is not None:
            for tag in tags:
                self.add_tag(tag)
        self._tag_props = dict()  # stores properties from tags to apply
        self._artist = None
        self._is_styled = False  # Indicates if styling properties were set
        self._kwargs = {}
        self.update(**kwargs)

    # def set_tags(self, tags: list):
    #     """Set a tag for the block."""
    #     for tag in tags:
    #         tag.register_proxy(self)
    #     self._tags = tags

    # def get_tags(self):
    #     """Return the list of tags."""
    #     return self._tags

    def add_tag(self, tag):
        """Adding a new tag to the proxy."""
        if tag not in self._tags:
            tag.register_proxy(self)
            self._tags.append(tag)

    def remove_tag(self, tag):
        """Removing a tag."""
        if tag in self._tags:
            tag.deregister_proxy(self)
            self._tags.remove(tag)
            self._set_tag_props()

    @property
    def is_tagged(self):
        """Indicate whether a block belongs to a tag or not."""
        return True if len(self._tags) else False

    def _set_tag_props(self):
        """
        Request properties from all of its tags and initiate tag to update
        if a tag is staled.
        """
        self._tag_props = dict()
        for tag in self._tags:
            _tag_props = normed_kws(tag.get_props(id(self)), self._artistcls)
            self._tag_props.update(_tag_props)

    def update(self, **props):
        # props = normed_kws(props, self._artistcls)
        for prop, value in props.items():
            setter = getattr(self, f"set_{prop}", None)
            if callable(setter):
                setter(value)
            else:
                raise AttributeError(f"{self} has no attribute named '{prop}'")
        return self.stale

    def get(self, prop: str, altval=None):
        """
        Return the value of a specific styling property attached to this proxy
        or to one of its Tags.
        If the property was not explicitly set, *altval* is returned.

        Parameters
        ----------
        prop : str
            The normalized name of the property to fetch.
        altval : Any (default: None)
            The value to return in case *prop* is not set.

        Note that aliases are not allowed for *prop*. This is because tags are
        included when trying to get the styling property and a single tag might
        be associated to proxies of various artist subclasses that might not
        all have the same aliases.
        """
        if getattr(self, f"own_{prop}", False):
            return getattr(self, f"get_{prop}")()
        else:
            if self.is_tagged:
                if not self._tag_props:
                    self._set_tag_props()
                return self._tag_props.get(prop, altval)
        return altval

    @property
    def is_styled(self,):
        """Indicate if this element has custom styling."""
        # NOTE: self.is_tagged might be too general as tags might not affect
        # fc, ec, lw, lw and aa
        return (self.is_tagged or self._is_styled)

    def _pre_creation(self, ax, **props):
        """Method handling properties foreign to the attached artist class."""
        self._set_tag_props()  # make sure to have the properties from all tags
        props.update(self._tag_props)  # complete/overwrite with tag porperties
        props.update(self._kwargs)  # complete/overwrite with onw properties
        return props

    def _init_artist(self, ax):
        """Initiate the artist."""
        raise NotImplementedError('Derived must override')

    def _update_artist(self, **kwargs):
        _kwargs = dict(kwargs)
        _kwargs.update(self._kwargs)
        self._artist.update(_kwargs)

    def _post_creation(self, ax=None):
        """Callback after the creation and init of artist."""
        pass

    def _applicable(self, props):
        """
        Separate *props* into applicable and non-applicable properties.
        Applicable are properties for which the proxy has a 'set_' method.
        """
        applicable = dict()
        nonapp = dict()
        for prop, v in props.items():
            if hasattr(self._artistcls, f"set_{prop}"):
                applicable[prop] = v
            else:
                nonapp[prop] = v
        return applicable, nonapp

    def create_artist(self, ax, **kwargs):
        """Create the artist of this proxy."""
        props = normed_kws(kwargs, self._artistcls)
        props = self._pre_creation(ax=ax, **props)  # run prepare callback
        props, _nonappl_props = self._applicable(props)
        self._init_artist(ax)                  # initiate artist with defaults
        self._update_artist(**props)           # apply collected properties
        self._post_creation(ax=ax)             # wrap-up callback

    def get_artist(self,):
        """Returns the artist associated to this proxy."""
        if self._artist is None:
            raise ValueError("The artist has not been created.")
        return self._artist


@_expose_artist_getters_and_setters
class _Block(_ArtistProxy):
    """
    A Block in an Alluvial diagram.

        Blocks in an Alluvial diagram get their vertical position assigned by a
        layout algorithm and thus after creation.
    """
    _artistcls = Rectangle

    # TODO uncomment once in mpl
    # @docstring.dedent_interpd
    def __init__(self, height, width=None, xa=None, ya=None, label=None,
                 tags=None, ha='center', va='bottom', label_margin=(0, 0),
                 **kwargs):
        """
        Parameters
        -----------
        height : float
          Height of the block.
        width : float,  optional
          Block width.
        xa: scalar, optional
          The x coordinate of the block's anchor point.
        ya: scalar, optional
          The y coordinate of the block's anchor point.
        label : str, optional
          Block label that can be displayed in the diagram.
        ha : {'center', 'left', 'right'}, default: 'center'
          The horizontal location of the anchor point of the block.
        va: {'center', 'top', 'bottom'}, default: 'center'
          The vertical location of the anchor point of the block.
        label_margin: (float, float), default: (0., 0.)
            x and y margin in target coordinates of ``self.get_transform()``
            and added to the *anchor* point to set the point to draw the label.

        Other Parameters
        ----------------
        **kwargs : Allowed are all `.Rectangle` properties:

          TODO: set to Rectangle (if it is registered)
          %(Patch_kwdoc)s

        """

        super().__init__(label=label, tags=tags, **kwargs)

        self._height = height
        self._width = width
        self._xa = xa
        self._ya = ya
        self.set_horizontalalignment(ha)
        self.set_verticalalignment(va)
        # init the in and out flows:
        self._outflows = []
        self._inflows = []
        self._label = label
        self.label_margin = label_margin
        # setting up the margin handling
        # _margins are encoded [[out-bottom, out-top], [in-bottom, in-top]]
        self._to_margin_index = lambda x: 0 if x <= 0 else 1  # out=0, in=1
        _yz = self._height - self._height  # create the neutral element
        self._margins = [[_yz, _yz], [_yz, _yz]]

    # getters and setters from attached artist
    def get_x(self):
        """Return the left coordinate of the block."""
        x = self._xa
        if self._horizontalalignment == 'center':
            x -= 0.5 * self.get_width()
        elif self._horizontalalignment == 'right':
            x -= self.get_width()
        return x

    def get_y(self):
        """Return the bottom coordinate of the block."""
        y = self._ya
        if self._verticalalignment == 'center':
            y -= 0.5 * self._height
        elif self._verticalalignment == 'top':
            y -= self._height
        return y

    def get_xy(self):
        """Return the left and bottom coords of the block as a tuple."""
        return self.get_x(), self.get_y()

    def get_width(self):
        """Return the width of the block."""
        if self._artist is not None:
            return self._artist.get_width()
        else:
            return self._width

    def get_height(self):
        """Return the height of the block."""
        if self._artist is not None:
            return self._artist.get_height()
        else:
            return self._height

    # def set_x(self, x):
    #     """Set the left coordinate of the rectangle."""
    #     # TODO: take into account ha and set xa then
    #     raise NotImplementedError('TODO')
    #     self.stale = True

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

    # def set_xy(self, xy):
    #     """
    #     Set the left and bottom coordinates of the rectangle.

    #     Parameters
    #     ----------
    #     xy : (float, float)
    #     """
    #     # TODO: use set_x and set_y here
    #     raise NotImplementedError('TODO')
    #     self.stale = True

    def set_width(self, width):
        """Set the width of the block."""
        self._width = width
        if self._artist is not None:
            self._artist.set_width(width)
        self.stale = True

    def set_height(self, height):
        """Set the height of the block."""
        self._height = height
        if self._artist is not None:
            self._artist.set_height(height)
        self.stale = True

    # Note: get_bbox is not overwritten as we want to avoid requesting the
    # bounding box before the artist is attached to an axis.

    # custom getters and setters for this proxy

    def get_xa(self):
        """Return the x coordinate of the anchor point."""
        return self._xa

    def get_ya(self):
        """Return the y coordinate of the anchor point."""
        return self._ya

    def get_xc(self, ):
        """Return the y coordinate of the block's center."""
        return self.get_x() + 0.5 * self.get_width()

    def get_yc(self, ):
        """Return the y coordinate of the block's center."""
        return self.get_y() + 0.5 * self._height

    def set_xa(self, x):
        """Set the x coordinate of the anchor point."""
        self._xa = x

    def set_horizontalalignment(self, align):
        """Set the horizontal alignment of the anchor point and the block."""
        # TODO: uncomment once in mpl
        # _api.check_in_list(['center', 'left', 'right'], align=align)
        self._horizontalalignment = align
        self.stale = True

    def set_verticalalignment(self, align):
        """Set the vertical alignment of the anchor point and the block."""
        # TODO: uncomment once in mpl
        # _api.check_in_list(['center', 'top', 'bottom'], align=align)
        self._verticalalignment = align
        self.stale = True

    def set_yc(self, yc):
        """Set the y coordinate of the block center."""
        self._ya = yc
        if self._verticalalignment == 'bottom':
            self._ya -= 0.5 * self._height
        elif self._verticalalignment == 'top':
            self._ya += 0.5 * self._height

    def get_flows(self, out=False):
        if out:
            return self._outflows
        else:
            return self._inflows

    def _set_flows(self, out, flows):
        if out:
            self._outflows = flows
        else:
            self._inflows = flows

    inflows = property(functools.partial(get_flows, out=False), None,
                       doc="List of `._Flow` objects entering the block.")
    outflows = property(functools.partial(get_flows, out=True), None,
                        doc="List of `._Flow` objects leaving the block.")

    def get_xlim(self,):
        """Returns the horizontal data limits as a tuple (x0, width)"""
        return self.get_x(), self.get_width()

    def get_ylim(self,):
        """Returns the vertical data limits as a tuple (y0, height)"""
        return self.get_y(), self.get_height()

    def get_datalim(self,):
        """Return the bounds (x0, y0, width, height) in data coordinates."""
        x0, width = self.get_xlim()
        y0, height = self.get_ylim()
        return x0, y0, width, height

    def add_flow(self, flow, out=False):
        """Add a flow either to the out- or the inflows."""
        if out:
            self._outflows.append(flow)
        else:
            self._inflows.append(flow)

    # ###
    # class specific methods for create_artist:
    def _pre_creation(self, ax=None, **props):
        props = super()._pre_creation(ax=ax, **props)

        # if facecolor was provided in a tag, complement the edgecolor
        if 'facecolor' in props:
            props['edgecolor'] = props.get('edgecolor', props['facecolor'])
        return props

    def _init_artist(self, ax):
        """Blocks use :class:`patches.Rectangle` as their patch."""
        self._artist = self._artistcls(self.get_xy(), width=self._width,
                                       height=self._height, axes=ax)

    def _post_creation(self, ax=None):
        # enforce setting the edgecolor to draw the border of the block.
        if (hasattr(self, 'own_facecolor') and not
                hasattr(self, 'own_edgecolor') and
                self.get_edgecolor()[3] == 0.0):
            fc = self.get_facecolor()
            self._artist.set_edgecolor(fc)
        self._handle_flows()
        x0, y0, width, height = self.get_bbox().bounds
    # ###

    def _request_loc(self, out: bool, width, out_pref, in_pref):
        """
        Get the anchor ant top for a flow a flow with a certain with and
        location preferences.

        Note that bottom is the lower corner while top is the upper corner.
        """
        h_loc = out_pref if out else in_pref  # preferred location here
        t_loc = in_pref if out else out_pref  # preferred location there
        loc = self.get_corner(out, h_loc)
        margin = self.get_margin(out, h_loc)
        bottom = (loc[0], loc[1] + margin + (width if t_loc < 0 else 0))
        top = (loc[0], loc[1] + margin + (width if t_loc > 0 else 0))
        self._update_margin(out, h_loc, width)
        return bottom, top

    def get_corner(self, out=False, preferred: int = 0) -> tuple:
        """
        Returns a corner of the block. If *out* is false (default) a corner
        from the left side is returned, otherwise the corner will be from the
        right side. Depending on the preference passed in *preferred* either
        the top or bottom corner is selected.

        Parameters
        ----------
        preferred : int
            Preferences are encoded as +/-: top/bottom; 2 is high priority, 1
            is low priority and 0 is no preference at all.
        """
        x0, y, width, height = self.get_bbox().bounds
        x = x0 + width if out else x0
        if preferred > 0:  # get the top
            y += height
        return (x, y)

    def get_margin(self, out=False, location=0):
        """
        Get the current vertical margin for a corner.
        The corner is selected based on *out* and *location* (see
        :meth:`get_corner` for details). The margin gives the space that is
        already occupied by attached flows.
        """
        _margin = self._margins[self._to_margin_index(out)]
        # _margin = self._out_margin if out else self._in_margin
        return _margin[self._to_margin_index(location)]

    def _update_margin(self, out: bool, location: int, change):
        """
        Increase a corner margin by the amount given in *change*.
        """
        _margin = self._margins[self._to_margin_index(out)]
        # _margin = self._out_margin if out else self._in_margin
        _margin[self._to_margin_index(location)] += change

    def _sort_flows(self, out: bool):
        # _inv_layout = True if layout == 'top' else False
        flows = self.get_flows(out)
        if not flows:
            return None
        other = 'target' if out else 'source'
        self_loc_idx = 1 if out else 2
        yc = self.get_yc()
        flow_infos = [(i, *of.get_prefs(), getattr(of, other).get_yc())
                      for i, of in enumerate(flows)]
        flow_infos = sorted(flow_infos, key=lambda x: (x[3] - yc))
        new_ordering = []
        for pref in [2, -2, 1, -1, 0]:  # prio. ordered (top bottom within)
            _order = (-1 if pref > 0 else 1)  # * (-1 if _inv_layout else 1)
            new_ordering.extend([tif[0]
                                 for tif in flow_infos
                                 if tif[self_loc_idx] == pref][::_order])
        self._set_flows(out, [flows[i] for i in new_ordering])

    def _set_flow_locations(self, out: bool):
        """
        For all flows (either out going - if *out* is True, or incoming) set
        the corner locations. The corner of a flow is either the left- or
        right-most horizontal line define by the top and bottom points.
        """
        for flow in self.get_flows(out):
            width = flow.flow * (1 if flow.get_pref(out) < 0 else -1)
            out_pref, in_pref = flow.get_prefs()
            flow.set_locations(out, *self._request_loc(out, width, out_pref,
                                                       in_pref))

    def _handle_flows(self,):
        """TODO: write docstring."""
        for out_flow in self._outflows:
            out_flow.update_prefs()
        for out in [True, False]:  # process both in and out flows
            self._sort_flows(out)
            self._set_flow_locations(out)


@_expose_artist_getters_and_setters
class _Flow(_ArtistProxy):
    """
    A connection between two blocks from adjacent columns.
    """
    _artistcls = patches.PathPatch

    def __init__(self, flow, source, target, label=None, tags=None,
                 **kwargs):
        """

        Parameters
        -----------
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

        By default *edgecolor* and *facecolor* are set to `lightgray`.
        """
        super().__init__(label=label, tags=tags, **kwargs)

        self.source = source
        self.target = target
        self.flow = flow
        self._out_pref = 0  # +/-: top/bottom,  2: high prio, 1: low prio
        self._in_pref = 0   # +/-: top/bottom,  2: high prio, 1: low prio
        self._xy1_in, self._xy1_out = None, None
        self._xy0_in, self._xy0_out = None, None
        # attach the flow to the source and target blocks
        if self.source is not None:
            self.source.add_flow(self, out=True)
        if self.target is not None:
            self.target.add_flow(self, out=False)
        self._path = None
        self.stale = True

    def get_prefs(self):
        """Get the preferred attach location on source and target block."""
        return self._out_pref, self._in_pref

    def get_pref(self, out: bool):
        """Get the preferred attach location on source (if *out*) or target."""
        return self._out_pref if out else self._in_pref

    def get_path(self,):
        """Return the :obj:`.path.Path` associated to this flow."""
        if self._artist is not None:
            return self._artist.get_path()
        else:
            return self._path

    def set_path(self, path):
        self._path = path
        if self._artist is not None:
            self._artist.set_path(path)
        self.stale = False

    def set_prefs(self, out_pref: int, in_pref: int):
        """
        Set the preferred attach locations. Preferences are encoded as follows:

        - Magnitude: 2 > high priority, 1 > low priority
        - Sign: + > Attaching at the top, - > attaching at the bottom
        """
        self._out_pref = out_pref
        self._in_pref = in_pref
        self.stale = True

    def set_locations(self, out: bool, bottom, top):
        """
        Set the bottom and top location. Note that the segment from *bottom* to
        *top* defines the area on a block where the flow exist perpendicular to
        the block surface. If *out* is set to `True` then the provided *bottom*
        and *top* define the area on the source block and if *out* is `False`
        on the target block.
        """
        if out:
            self._xy0_out = np.array(bottom)
            self._xy1_out = np.array(top)
        else:
            self._xy0_in = np.array(bottom)
            self._xy1_in = np.array(top)
        self.stale = True

    def update_prefs(self,):
        """Determine the preferred anchor positions on source and target"""
        # needed: s_y0, s_y1, s_yc, t_yc
        s_yc = self.source.get_yc()
        s_hh = 0.5 * self.source.get_height()
        s_y0, s_y1 = s_yc - s_hh, s_yc + s_hh
        t_yc = self.target.get_yc()
        if s_y1 < t_yc:
            out_pref, in_pref = 2, -2
        else:
            if s_y0 > t_yc:
                out_pref, in_pref = -2, 2
            else:  # low priority only
                if s_yc < t_yc:
                    out_pref, in_pref = 1, -1
                else:
                    out_pref, in_pref = -1, 1
        self.set_prefs(out_pref, in_pref)

    def update_path(self):
        """
        Creates a path based on the current state of the flow and attaches the
        path to the flow, un-staling the flow.
        """
        sx0, _, swidth, _ = self.source.get_bbox().bounds  # get the width in
        tx0, _, twidth, _ = self.target.get_bbox().bounds  # converted units
        out_hoff = np.array([0.5 * swidth, 0])  # horizontal offset on source
        in_hoff = np.array([0.5 * twidth, 0])  # horizontal offset on target
        gap_dist = tx0 - (sx0 + swidth)
        # place the control points at the golden ratio in the gap distance
        control_hoff = np.array([1 / 1.618 * gap_dist, 0])
        # set the external corner points for the flow.
        xy0_out_c = self._xy0_out - out_hoff
        xy0_in_c = self._xy0_in + in_hoff
        xy1_out_c = self._xy1_out - out_hoff
        xy1_in_c = self._xy1_in + in_hoff
        # set control points for the bezier curves
        dir_out_xy0 = self._xy0_out + control_hoff
        dir_out_xy1 = self._xy1_out + control_hoff
        dir_in_xy0 = self._xy0_in - control_hoff
        dir_in_xy1 = self._xy1_in - control_hoff
        # define the vertices
        vertices = [xy0_out_c, self._xy0_out, dir_out_xy0, dir_in_xy0,
                    self._xy0_in, xy0_in_c, xy1_in_c, self._xy1_in,
                    dir_in_xy1, dir_out_xy1, self._xy1_out, xy1_out_c]
        # encode the drawing
        codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4,
                 Path.CURVE4, Path.LINETO, Path.LINETO, Path.LINETO,
                 Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO]
        self.set_path(Path(vertices, codes))

    def _init_artist(self, ax):
        """TODO: write docstring."""
        # create the artist
        if self.stale:
            self.update_path()
        self._artist = self._artistcls(self.get_path(), axes=ax)

    def _update_artist(self, **kwargs):
        """Update the artist of a _Flow."""
        _kwargs = dict(kwargs)
        _kwargs.update(self._kwargs)
        # resolve properties that simply point to the source or target block
        kwargs = dict()
        for prop, value in _kwargs.items():
            # we use setters if value refers to blocks as this leads to an
            # individual styling
            if value == 'source':
                getattr(self,
                        f'set_{prop}')(getattr(self.source,
                                               f"get_{prop}")())
            elif value == 'target':
                # _kwargs[prop] = getattr(self.target, f"get_{prop}")()
                getattr(self,
                        f'set_{prop}')(getattr(self.target,
                                               f"get_{prop}")())
            elif value == 'interpolate':
                # TODO:
                raise NotImplementedError("The mode 'interpolate' for the"
                                          f" property '{prop}' of a _Flow is"
                                          "not yet implemented")
            else:
                kwargs[prop] = value
        # TODO: dont update the artist directly as this leads to discrepancy
        # between proxy and artist.
        self._artist.update(kwargs)


@_expose_artist_getters_and_setters
class _ProxyCollection(_ArtistProxy):
    """
    A collection of _ArtistProxy with common styling properties.
    """
    _artistcls = PatchCollection
    _singular_props = ['zorder', 'hatch', 'pickradius', 'capstyle',
                       'joinstyle', 'cmap', 'norm']

    def __init__(self, proxies, label=None, tags=None, **kwargs):
        """
        Parameters
        ----------
        proxies : sequence of :obj:`_ArtistProxy`
            The proxies in this collection.
        label : str, optional
            Label of the collection.
        """
        super().__init__(label=label, tags=tags, **kwargs)

        self._proxies = proxies
        self._match_original = None

    def __iter__(self):
        return iter(self._proxies)

    def __getitem__(self, key):
        return self._proxies[key]

    def __bool__(self,):
        return bool(self._proxies)

    def _applicable(self, props):
        """
        Separate *props* into applicable and non-applicable properties.
        Applicable are properties for which the proxy has a 'set_' method.
        """
        applicable, nonapp = super()._applicable(props)
        # since we use custom name for the mappable array we need to make it
        # applicable
        for to_applicable in ['mappable']:
            if to_applicable in nonapp:
                applicable[to_applicable] = nonapp.pop(to_applicable)
        return applicable, nonapp

    def to_element_styling(self, styleprops: dict):
        """Convert the styling properties to lists matching self._proxies."""
        _styprops = normed_kws(styleprops, self._artistcls)
        indiv_props = {sp: _styprops.pop(sp)
                       for sp in self._singular_props if sp in _styprops}
        for prop, altval in _styprops.items():
            indiv_props[prop] = [p.get(prop, altval) for p in self._proxies]
        return indiv_props

    def _pre_creation(self, ax=None, **props):
        """Ensure all proxies create their own artists."""
        props = super()._pre_creation(ax=ax, **props)
        self._match_original = False
        for proxy in self._proxies:
            proxy.create_artist(ax=ax, **props)
            if proxy.is_styled:
                self._match_original = True
        if self._match_original:
            self._proxy_props = self.to_element_styling(props)
        else:
            self._proxy_props = dict()
        return props

    def _init_artist(self, ax):
        """
        Creates `.PatchCollections`s for the blocks in this collection.
        Note that *ax* is unused for a ProxyCollection as the collection is
        attached to an axes in :meth:`._post_creation'
        """
        self._artist = self._artistcls(
            [proxy.get_artist() for proxy in self._proxies],
            match_original=self._match_original,
        )

    def _update_artist(self, **kwargs):
        _kwargs = dict(kwargs)
        _kwargs.update(self._kwargs)
        self._cmap_data = _kwargs.pop('mappable', None)
        if self._cmap_data is not None:
            _mappable_array = np.asarray([getattr(proxy,
                                                  f'get_{self._cmap_data}')()
                                          for proxy in self._proxies])
            # Note: cmap doesn't work with datetime objects, so we convert them
            _first_mappable = cbook.safe_first_element(_mappable_array)
            if isinstance(_first_mappable, datetime):
                _mappable_array = date2num(_mappable_array)
            self._artist.set_array(_mappable_array)

        if self._match_original:
            # make sure to not overwrite individual properties
            for props in ('facecolor', 'edgecolor', 'linewidth', 'linestyle',
                          'antialiased'):
                _kwargs.pop(props, None)
        self._artist.update(_kwargs)

    def _post_creation(self, ax=None):
        self._artist = ax.add_collection(self._artist)

    def add_proxy(self, proxy):
        """Add a Proxy."""
        self._proxies.append(proxy)


class _Tag(cm.ScalarMappable):
    """
    Class to create sets of styling properties that can be attached to various
    ArtistPropxies.
    """
    def __init__(self):
        """
        Parameters
        ----------
        """
        super().__init__()
        self._marked_obj = WeakValueDictionary()
        self._is_filled = True   # By default both facecolor and edgecolor will
        self._is_stroked = True  # be set by a tag.
        self._alpha = None
        self._mappable = None
        self._mappables = dict()
        self._props = dict()
        self.stale = True

    def _update_scalarmappable(self):
        """Update colors from the scalar mappable array, if it is not None."""
        marked_obj_ids = list(self._marked_obj.keys())
        self._proxy_props = {oid: {} for oid in marked_obj_ids}
        _mappable_array = np.asarray([getattr(self._marked_obj[oid],
                                              f'get_{self._mappable}')()
                                      for oid in marked_obj_ids])
        # Note: cmap doesn't work with datetime objects, so we convert the data
        _first_mappable = cbook.safe_first_element(_mappable_array)
        if isinstance(_first_mappable, datetime):
            _mappable_array = date2num(_mappable_array)
        self.set_array(_mappable_array)
        if self._A is None:
            return
        # QuadMesh can map 2d arrays (but pcolormesh supplies 1d array)
        if self._A.ndim > 1:
            raise ValueError('Collections can only map rank 1 arrays')
        if np.iterable(self._alpha):
            if self._alpha.size != self._A.size:
                raise ValueError(f'Data array shape, {self._A.shape} '
                                 'is incompatible with alpha array shape, '
                                 f'{self._alpha.shape}. '
                                 'This can occur with the deprecated '
                                 'behavior of the "flat" shading option, '
                                 'in which a row and/or column of the data '
                                 'array is dropped.')
            # pcolormesh, scatter, maybe others flatten their _A
            self._alpha = self._alpha.reshape(self._A.shape)

        _colors = self.to_rgba(self._A, self._alpha)
        if self._is_filled:
            for i, oid in enumerate(marked_obj_ids):
                self._proxy_props[oid]['facecolor'] = _colors[i]
        elif self._is_stroked:
            for i, oid in enumerate(marked_obj_ids):
                self._proxy_props[oid]['edgecolor'] = _colors[i]

    def register_proxy(self, proxy):
        """TODO: write docstring."""
        proxy_id = id(proxy)
        if proxy_id in self._marked_obj:
            return  # if the proxy was already registered, ignore it
        self._marked_obj[proxy_id] = proxy  # remember the proxy
        self.stale = True

    def deregister_proxy(self, proxy):
        proxy_id = id(proxy)
        self._marked_obj.pop(id(proxy))
        if self._mappable is not None:
            self.stale = True
            self._mappables.pop(proxy_id)

    def _prepare_props(self):
        self._proxy_props = {}
        if self._mappable is not None:
            for obj_id, proxy in self._marked_obj.items():
                self._mappables[obj_id] = getattr(proxy,
                                                  f'get_{self._mappable}')()
            self._update_scalarmappable()
        self.stale = False

    # TODO: uncomment once in mpl
    # @docstring.dedent_interpd
    def set(self, **props):
        """
        Setting the styling properties associated to this tag.

        Allowed are styling arguments of :class:`matplotlib.patches.PathPatch`
        (see below), as well as, *cmap* and *mappable*.

        Parameters:
        -----------
        cmap : `~.colors.Colormap`, optional
            Forwarded to `.ScalarMappable`. The default of
            ``None`` will result in :rc:`image.cmap` being used.

          %(Patch_kwdoc)s

        """
        props = dict(props)
        self.set_cmap(props.pop('cmap', None))
        self.set_norm(props.pop('norm', None))
        self._mappable = props.pop('mappable', None)
        if self._mappable is not None:
            self.stale = True
        self._props.update(props)

    def get_props(self, obj_id):
        """Return the properties specific to an object_id."""
        if self.stale:
            self._prepare_props()
        props = dict(self._props)
        props.update(self._proxy_props.get(obj_id, {}))
        return props


class _Initiator():
    _init_defaults = None

    @ classmethod
    def split_inits(cls, props):
        return {k: props.pop(k) for k in cls._init_defaults
                if props.get(k, None) is not None}


@init_defaults((_Block,))
class SubDiagram(_Initiator):
    """
    A collection of Blocks and Flows belonging to a diagram.

    """
    def __init__(self, columns, flows, x=None, label=None, yoff=0, hspace=1,
                 hspace_combine='divide', label_margin=(0, 0),
                 layout='centered', blockprops=None, flowprops=None,
                 tags=None):
        """
        Parameters
        ----------
        x : sequence of scalars
            The x coordinates of the columns.
            A sequence of M scalars that determine the x coordinates of columns
            provided in *columns*.

            When provided, the diagram ignores the x coordinates that might
            be provided by the `.Alluvial` instance it is attached to.

            If *x* is None and also the `.Alluvial` instance has no values set
            for the x coordinates they will  default to the range defined by
            the number of columns.
        columns : sequence of array_like objects
            Sequence of M array-like objects each containing the blocks of a
            column.
            Allowed are `_Block` objects or floats that will be interpreted as
            the size of a block.
        flows : sequence of array_like objects
            ... *TODO*
        label : str, optional
            Label of the diagram.
        yoff : int, float or sequence thereof, default: 0
            Vertical offset applied to the added diagram. A single value sets
            the offset for the first column, any sequential column determines
            the offset by minimizing the vertical displacement of the flows. If
            a sequence is provided it sets for each column the vertical offset
            explicitly and must be of the same length as *columns*.
        hspace : float, (default=1)
            The height reserved for space between blocks expressed as a
            float in the same unit as the block heights.
        hspace_combine : {'add', 'divide'}, default: 'divide'
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
            Allowed layouts are: {'centered', 'bottom', 'top', 'optimized'}.

            If as sequence is provided, the M elements must specify the layout
            for each of the M columns in the diagram.

            The following options are available:

            - 'centered' (default): The bigger the block (in terms of height)
              the more it is moved towards the center.
            - 'bottom': Blocks are sorted according to their height with the
              biggest blocks at the bottom.
            - 'top': Blocks are sorted according to their height with the
              biggest blocks at the top.
            - 'optimized': Starting from a centered layout the order of bocks
              in a column is iteratively changed to decrease the vertical
              displacement of all flows attached to the column.
        blockprops : dict, optional
            Styling parameter that are applied to all blocks in this diagram.
            For a list of possible parameter see the list for Collection below.
        flowprops : dict, optional
            Styling parameter that are applied to all flows in this diagram.
            For a list of possible parameter see the list for Collection below.
        tags : sequence or str, optional
            Tagging of the blocks. Tags can be provided in the following
            formats:

            - String, allowed are {'column', 'index'}.
              If *tags* is set to 'column', all blocks in a column get the same
              tag. If 'index' is used, in each column the blocks is tagged by
              their index in the column.

                .. warnging::
                not yet implemented

            - Sequence of M tags, providing for each column a separate tag.
            - Sequence of list of tags, providing fore each block in each
              column a tag.

            If a sequence is provided, the tags can be any hashable object.

            Note that *tags* should be used in combination with *tagprops* in
            order to specify the styling for each tag.

        Note that *x* and *columns* must be sequences of the same length.

        %(Collection_kwdoc)s
        """
        self.stale = True  # This only tracks form/layout and not style changes
        self._x = _to_valid_arrays(x, 'x')
        # Note: both _block-/_flowprops must be normalized already
        _blockprops = normed_kws(blockprops or dict(), _Block._artistcls)
        # separate block layout parameter from styling parameter:
        self._block_init = self.split_inits(_blockprops)
        self._blockprops = _blockprops
        # flows only have style parameter anyways
        self._flowprops = normed_kws(flowprops or dict(), _Flow._artistcls)

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
            for c_i, col in enumerate(columns):
                column = list(col)
                _blocks.extend(column)
                self._columns.append(column)
        else:
            for col in columns:
                column = [_Block(size, **self._block_init) for size in col]
                self._columns.append(column)
                _blocks.extend(column)
        self._nbr_columns = len(self._columns)
        # Tagging blocks if tags are provided
        if tags is not None:
            for c_i, col in enumerate(self._columns):
                column = list(col)
                col_tags = tags[c_i]
                if not np.iterable(col_tags):  # tagging by column
                    col_tags = len(column) * [col_tags]
                for i, block in enumerate(column):
                    block.add_tag(col_tags[i])

        # TODO: below attributes need to be handled
        self._redistribute_vertically = 4
        self._xlim = None
        self._ylim = None

        self._blocks = _ProxyCollection(_blocks, label=label)
        # create the Flows is only based on *flows* and *extout*'s
        _flows = []
        # connect source and target:
        for m, flowM in enumerate(flows):
            # m is the source column, m+1 the target column
            s_col = self._columns[m]
            t_col = self._columns[m + 1]
            for i, row in enumerate(flowM):
                # i is the index of the target block
                for j, flow in enumerate(row):
                    # j is the index of the source block
                    # TODO: pass kwargs?
                    if flow:
                        _flows.append(_Flow(flow=flow, source=s_col[j],
                                            target=t_col[i]))
        self._flows = _ProxyCollection(_flows, label=label)
        self._hspace = hspace
        self._hspace_combine = hspace_combine
        self.init_layout_yoff(layout, yoff)
        # TODO: setting label position is not implemented yet
        self._label_margin = label_margin
        self.generate_layout()

    def __iter__(self):
        return iter(self._columns)

    def __getitem__(self, key):
        return self._columns[key]

    def _update_datalim(self):
        """Return the limits of the block collection in data units."""
        # TODO: set x margin (for now just 1%)
        xmin, xmax = min(self._x), max(self._x)
        # setting some initial y limits
        for _col in self._columns:
            if _col:
                y0, height = _col[0].get_ylim()
                ymin, ymax = y0, y0 + height
                break
        # getting x limits
        for _block in self._columns[0]:
            x0, width = _block.get_xlim()
            xmin = min(xmin, x0)
        for _block in self._columns[-1]:
            x0, width = _block.get_xlim()
            xmax = max(xmax, x0 + width)
        # getting y limits
        for _col in self._columns:
            if _col:
                y0, height = _col[0].get_ylim()
                ymin = min(ymin, y0)
                y0, height = _col[-1].get_ylim()
                ymax = max(ymax, y0 + height)
        self._xlim = xmin, xmax
        self._ylim = ymin, ymax

    # def get_blocks(self):
    #     return self._blocks

    def get_ylim(self,):
        if self.stale:
            self.generate_layout()
        return self._ylim

    def get_minwidth(self,):
        """Returns the smallest block width."""
        mwidth = self._blocks[0].get_width()
        for col in self._columns:
            for block in col:
                mwidth = min(mwidth, block.get_width())
        return mwidth

    def get_datalim(self,):
        """Returns the limits in data units (x0, y0, x1, y1)."""
        if self.stale:
            self.generate_layout()
        x0, x1 = self._xlim
        y0, y1 = self._ylim
        return x0, y0, x1, y1

    def get_visuallim(self):
        """Returns the data limit with sensible margins added."""
        xmin, ymin, xmax, ymax = self.get_datalim()
        # TODO: set x margin (for now just 1%)
        minwidth = self.get_minwidth()
        xmargin = max(0.5 * minwidth, 0.01 * (max(self._x) - min(self._x)))
        if self._hspace_combine == 'add':
            ymargin = self._hspace
        else:
            ymargin = self._hspace / max(len(col) for col in self._columns)
        return xmin - xmargin, ymin - ymargin, xmax + xmargin, ymax + ymargin

    # def get_layout(self):
    #     """Get the layout of this diagram"""
    #     return self._layout

    def get_columns(self,):
        """Get all columns of this subdiagram"""
        if self.stale:
            self.generate_layout()
        return self._columns

    # def get_column(self, col_id):
    #     """TODO: write docstring."""
    #     return self._columns[col_id]

    def get_block(self, identifier):
        """TODO: write docstring."""
        if isinstance(identifier, int):
            return self._blocks[identifier]
        else:
            col_id, block_id = identifier
            return self._columns[col_id][block_id]

    def get_x(self):
        """Get the horizontal positioning of the columns"""
        return self._x

    def supplement_x(self, x):
        """Set the x coordinates."""
        if self._x is None:
            if x is None:
                self._x, self._columns = cbook.index_of(self._columns)
            else:
                # set the x values put only use the number of columns
                self._x = x[:len(self._columns)]
            for xi, col in zip(self._x, self._columns):
                for block in col:
                    block.set_xa(xi)

    def set_column_layout(self, col_id, layout):
        """Set the layout for a single column"""
        # TODO: uncomment once in mpl
        # _api.check_in_list(['centered', 'top', 'bottom', 'optimized'],
        #                    layout=layout)
        self._layout[col_id] = layout

    def init_layout_yoff(self, layout, yoff):
        """Set the layout and vertical offset for all columns."""
        if not np.iterable(yoff):
            yoff = self._nbr_columns * [yoff]
        self._yoff = yoff
        if isinstance(layout, str):
            # TODO: uncomment once in mpl
            # _api.check_in_list(['centered', 'top', 'bottom', 'optimized'],
            #                    layout=layout)
            self._layout = [layout for _ in range(self._nbr_columns)]
        else:
            self._layout = []
            for _layout in layout:
                # TODO: uncomment once in mpl
                # _api.check_in_list(['centered', 'top', 'bottom', 'optimized'],
                #                    layout=layout)
                self._layout.append(_layout)
        self.stale = True

    def generate_layout(self, ):
        """TODO: write docstring."""
        for col_id in range(self._nbr_columns):
            # TODO: handle the layout parameter
            self._distribute_blocks(col_id)
        # now check if some columns are optimized
        optimizing_col = [i for i, layout in enumerate(self._layout)
                          if layout == 'optimized']
        if len(optimizing_col):
            for col_id in optimizing_col:
                # # now sort again considering the flows.
                self._decrease_flow_distances(col_id)
            # for col_id in optimizing_col:
            #     # # perform pairwise swapping for backwards flows
            #     self._pairwise_swapping(col_id)
            # for col_id in optimizing_col[::-1]:
            #     # # perform pairwise swapping for backwards flows
            #     self._pairwise_swapping(col_id)
            #     # raise NotImplementedError("The optimized layout is not yet"
            #     #                          " implemented")
        self.stale = False

    def get_column_hspace(self, col_id):
        """TODO: write docstring."""
        if self._hspace_combine == 'add':
            return self._hspace
        else:
            nbr_blocks = len(self._columns[col_id])
            if nbr_blocks > 1:
                return self._hspace / (nbr_blocks - 1)
            else:
                return 0

    def update_blocks(self, cselector, bselector, **kwargs):
        """
        Update in all columns selected through *cselector* all blocks selected
        through *bselector*:
        """
        for column in self._columns[cselector]:
            for block in column:
                bstale = block.update(*kwargs)
            self.stale = self.stale or bstale
        return self.stale

    def _distribute_blocks(self, col_id: int):
        """
        Distribute the blocks in a column.

        Parameters
        -----------
        col_id: int
            The index of the column to recompute the distribution of blocks.

        Returns
        -------
        tuple
            (y_min, y_max) of the column in data units

        """
        nbr_blocks = len(self._columns[col_id])
        layout = self._layout[col_id]
        yoff = self._yoff[col_id]
        # NOTE: getting layout and yoff here but also passing col_id to _update_yorrd is not really clean
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
                self._update_ycoords(col_id, col_hspace, layout, yoff)
            elif layout == 'bottom':
                ordering = ordering[::-1]
                self._reorder_column(col_id, ordering)
                self._update_ycoords(col_id, col_hspace, layout, yoff)
            # in both cases no further sorting is needed
            else:
                # sort so to put biggest height in the middle
                ordering = ordering[::-2][::-1] + \
                    ordering[nbr_blocks % 2::2][::-1]
                # update the ordering the update the y coords
                self._reorder_column(col_id, ordering)
                self._update_ycoords(col_id, col_hspace, layout, yoff)

    def _decrease_flow_distances(self, col_id):
        """TODO: write docstring."""
        _column = self._columns[col_id]
        layout = self._layout[col_id]
        yoff = self._yoff[col_id]
        col_hspace = self.get_column_hspace(col_id)
        # do the redistribution a certain amount of times
        _redistribute = False
        for _ in range(self._redistribute_vertically):
            new_ycs = []
            for block in _column:
                weights = []
                difference = []
                b_yc = block.get_yc()
                for in_flow in block.inflows:
                    if in_flow.source is not None:
                        weights.append(in_flow.flow)
                        difference.append(b_yc - in_flow.source.get_yc())
                if weights:
                    _redistribute = True
                    new_ycs.append(
                        b_yc - sum(w * d for w, d in
                                   zip(weights, difference)) / sum(weights)
                    )
                else:
                    new_ycs.append(b_yc)
            if _redistribute:
                # set the new centers provisionally
                for block, nyc in zip(_column, new_ycs):
                    block.set_yc(nyc)
                # update the block order
                cs, _bc = zip(*sorted(((i, bc) for i, bc in enumerate(new_ycs)),
                                      key=lambda x: x[1]))
                # reorder blocks
                self._reorder_column(col_id, ordering=cs)
                # redistribute them
                self._update_ycoords(col_id, col_hspace, layout, yoff)
            else:
                break

    def _pairwise_swapping(self, col_id):
        """TODO: write docstring."""
        # TODO: this is broken: update the ordering then call _reorder_column
        _column = self._columns[col_id]
        nbr_blocks = len(_column)
        col_hspace = self.get_column_hspace(col_id)
        for _ in range(int(0.5 * nbr_blocks)):
            for i in range(1, nbr_blocks):
                b1, b2 = _column[i - 1], _column[i]
                if self._swap_blocks((b1, b2), col_hspace, 'backwards'):
                    b2.set_y(b1.get_y())
                    b1.set_y(b2.get_y() + b2.get_height() + col_hspace)
                    _column[i - 1], _column[i] = b2, b1
        for _ in range(int(0.5 * nbr_blocks)):
            for i in range(1, nbr_blocks):
                b1, b2 = _column[i - 1], _column[i]
                # b1 = _column[nbr_blocks - i - 1]
                # b2 = _column[nbr_blocks - i]
                if self._swap_blocks((b1, b2), col_hspace, 'forwards'):
                    b2.set_y(b1.get_y())
                    b1.set_y(b2.get_y() + b2.get_height() + col_hspace)
                    _column[i - 1], _column[i] = b2, b1
                    # _column[nbr_blocks - i - 1] = b2
                    # _column[nbr_blocks - i] = b1
        self.stale = True

    def _reorder_column(self, col_id, ordering):
        """Update the ordering of blocks in a column"""
        _column = self._columns[col_id]
        self._columns[col_id] = [_column[newid] for newid in ordering]
        self.stale = True

    def _best_offset(self, column):
        """
        Determine yoff of this column that minimizes vert. flows with the
        previous column.
        """
        weights = []
        difference = []
        for block in column:
            b_yc = block.get_yc()
            for in_flow in block.inflows:
                if in_flow.source is not None:
                    weights.append(in_flow.flow)
                    difference.append(b_yc - in_flow.source.get_yc())
        if sum(weights) == 0:
            return 0
        return -sum(w * d for w, d in zip(weights, difference)) / sum(weights)

    def set_column_y(self, column, y_start, hspace):
        for block in column:
            block.set_y(y_start)
            y_start += block.get_height() + hspace

    def _update_ycoords(self, col_id: int, hspace, layout, yoff):
        """
        Update the y coordinate of the blocks in a column based on the
        diagrams vertical offset, the layout chosen for this column and the
        order of the blocks.

        Parameters
        ----------
        col_id : int
            Index of the column to reorder.
        """
        _column = self._columns[col_id]
        if not _column:
            return
        # distribute block according to ordering
        if _column[0].get_y() is None:
            ystart = yoff
        elif layout == 'optimized' and col_id:
            ystart = self._columns[col_id - 1][0].get_y()
        else:
            ystart = _column[0].get_y()
        self.set_column_y(_column, ystart, hspace)
        # determine the y offset of the entire column
        if layout == 'optimized':
            yoff = self._best_offset(_column)
        else:
            low = _column[0].get_y()
            high = _column[-1].get_y() + _column[-1].get_height()

            if layout == 'centered':
                _offset = 0.5 * (high - low)
            elif layout == 'top':
                _offset = (high - low)
            else:
                _offset = 0
            yoff -= _offset
        # set the y position again including the offset
        y_position = _column[0].get_y() + yoff
        self.set_column_y(_column, y_position, hspace)
        self.stale = True

    def _swap_blocks(self, blocks, hspace, direction='backwards'):
        """
        Check if swapping to blocks leads to shorter vertical flow distances.
        """
        squared_diff = {}
        for block in blocks:
            weights, sqdiff = [], []
            if direction in ['both', 'backwards']:
                for flow in block.inflows:
                    if flow.source is not None:
                        weights.append(flow.flow)
                        sqdiff.append(abs(block.get_yc() - flow.source.get_yc()))
            if direction in ['both', 'forwards']:
                for flow in block.outflows:
                    if flow.target is not None:
                        weights.append(flow.flow)
                        sqdiff.append(abs(block.get_yc() - flow.target.get_yc()))
            if sum(weights) > 0.0:
                squared_diff[block] = sum(
                    [weights[i] * sqdiff[i] for i in range(len(weights))]
                ) / sum(weights)
        # assert n1.get_y() < n2.get_y()
        # TODO: Cannot recreate the thought process behind this...
        inv_mid_height = [blocks[0].get_y() + blocks[1].get_height() +
                          hspace + 0.5 * blocks[0].get_height(),
                          blocks[0].get_y() + 0.5 * blocks[1].get_height()]
        squared_diff_inf = {}
        for i, block in enumerate(blocks):
            weights = []
            sqdiff = []
            if direction in ['both', 'backwards']:
                for flow in block.inflows:
                    if flow.source is not None:
                        weights.append(flow.flow)
                        sqdiff.append(abs(
                            inv_mid_height[i] - flow.source.get_yc()
                        ))
            if direction in ['both', 'forwards']:
                for flow in block.outflows:
                    if flow.target is not None:
                        weights.append(flow.flow)
                        sqdiff.append(
                            abs(inv_mid_height[i] - flow.target.get_yc())
                        )
            if sum(weights) > 0.0:
                squared_diff_inf[block] = sum(
                    [weights[i] * sqdiff[i] for i in range(len(weights))]
                ) / sum(weights)
        if sum(squared_diff.values()) > sum(squared_diff_inf.values()):
            return True
        else:
            return False

    def create_block_artists(self, ax, **kwargs):
        if self._blocks:
            if self.stale:
                self.generate_layout()
            _kwargs = dict(kwargs)
            _kwargs.update(self._blockprops)
            _blockkws = normed_kws(_kwargs, self._blocks._artistcls)
            # blocks should have their edges drawn, so if only fc is provided
            # we complete it here. Note that normalize_kwargs is called only
            # to normalize 'edgecolor' & 'facecolor'
            _blockkws['edgecolor'] = _blockkws.get('edgecolor',
                                                   _blockkws.get('facecolor',
                                                                 None))
            self._blocks.create_artist(ax=ax, **_blockkws)
            # at last make sure the datalimits are updated
            self._update_datalim()

    def create_flow_artists(self, ax, **kwargs):
        if self._flows:
            _kwargs = dict(kwargs)
            _kwargs.update(self._flowprops)
            if self.stale:
                self.generte_layout()
                self.create_block_artists(self, ax=ax, **kwargs)
            # _flowkws = normed_kws(_kwargs, self._flows._artistcls)
            self._flows.create_artist(ax=ax, **_kwargs)


@init_defaults((SubDiagram,))
class Alluvial(_Initiator):
    """
    Alluvial diagram.

        Alluvial diagrams are a variant of flow diagram designed to represent
        changes in classifications, in particular changes in network
        structure over time.
        `Wikipedia (23/1/2021) <https://en.wikipedia.org/wiki/Alluvial_diagram>`_
    """
    # @docstring.dedent_interpd
    def __init__(self, x=None, ax=None, blockprops=None, flowprops=None,
                 **kwargs):
        """
        Create a new Alluvial instance.


        Parameters
        ===========
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
        cluster_w_spacing: float, int (default=1)
          Vertical spacing between blocks.
        blockprops : dict, optional
          The properties used to draw the blocks. *blockprops* accepts all
          arguments for :class:`matplotlib.patches.Rectangle`:

          %(Rectangle_kwdoc)s

        flowprops: dict, optional
          The properties used to draw the flows. *flowprops* accepts keyword
          arguments for :class:`matplotlib.patches.PathPatch`:

          %(Patch_kwdoc)s

          TODO: this is old and needs to be redone:
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

          Note that *blockprops* and *flowprops* set the properties of all
          sub-diagrams, unless specific properties are provided when a
          sub-diagram is added (see :meth:`add` for details), or
          :meth:`set_blockprops` is called before adding further sub-diagrams.

        """
        # create axes if not provided
        if ax is None:
            import matplotlib.pyplot as plt
            fig = plt.figure()
            # TODO: not sure if specifying the ticks is necessary
            ax = fig.add_subplot(1, 1, 1, yticks=[])
        self.ax = ax
        if x is not None:
            self._x = _to_valid_arrays(x, 'x')
            self.ax.xaxis.update_units(self._x)
        else:
            self._x = None
        self._diagrams = []
        self._cross_flows = []
        self._tags = defaultdict(_Tag)
        self._extouts = []
        self._diagc = 0
        self._xlim = None
        self._ylim = None
        self.staled_layout = True
        # how many x ticks are maximally shown
        self.max_nbr_xticks = 10
        # now handle the kwargs
        # normalize whatever there is
        _kwargs = normed_kws(kwargs, _ProxyCollection._artistcls)
        _blockprops = normed_kws(blockprops, _ProxyCollection._artistcls)
        _flowprops = normed_kws(flowprops, _ProxyCollection._artistcls)
        # setting up coloring (to be added to styling defaults in get_defaults)
        fc = _kwargs.pop('facecolor', None)
        if fc is None:
            self._color_cycler = itertools.cycle(
                mpl.rcParams['axes.prop_cycle'])
        else:
            # Note passing rgb/rgba arrays is not supported
            self._color_cycler = itertools.cycle(cycler(color=fc))
        if _kwargs:
            # remove specific arguments for `.add` call
            flows = _kwargs.pop('flows', None)
            ext = _kwargs.pop('ext', None)
            extout = _kwargs.pop('extout', None)
            fractionflow = _kwargs.pop('fractionflow', None)
            # separate init parameter for a subdiagram:
            self._subd_init = self.split_inits(_kwargs)
            # whatever remains is used as basis for blockprops and flowprops
            self._set_blockprops(_blockprops, _kwargs)
            self._set_flowprops(_flowprops, _kwargs)
            # what remains are styling defaults and will be passed down when
            # calling `.finish`
            self._defaults = _kwargs
            if flows is not None or ext is not None:
                self.add(flows=flows, ext=ext, extout=extout,
                         fractionflow=fractionflow,
                         blockprops=self._block_init, flowprops=self._flowprops,
                         **self._subd_init)
                self.finish()
        else:
            self._defaults = dict()
            self._subd_init = dict()
            self._set_blockprops(_blockprops)
            self._set_flowprops(_flowprops)

    def get_x(self):
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
        """
        if x is None:
            self._x = None
        else:
            self._x = _to_valid_arrays(x, 'x')

    def get_diagrams(self):
        """Get all sub-diagrams."""
        return self._diagrams

    def get_diagram(self, diag_id):
        """TODO: write docstring."""
        return self._diagrams[diag_id]

    def _to_cols_and_flows(self, cinit, flows, ext, extout, fractionflow):
        """
        Create the columns of an alluvial diagram and convert fractional flows
        to absolute quantities.
        """
        # create the columns
        columns = [cinit]
        if flows is not None:
            flows = np.copy(flows)
            for i in range(len(flows)):
                flow = flows[i]
                e = ext[i + 1]
                if len(flow):
                    if fractionflow:
                        _flow = flow.dot(columns[-1])
                        # create the absolute flow matrix
                        flows[i] = flow * columns[-1]
                    else:
                        _flow = flow.sum(1)
                    _col = _flow + e
                else:
                    _col = e
                columns.append(_col)
        if extout is not None:
            pass  # TODO: check extout format
        return columns, flows

    def _set_blockprops(self, block_kws, other_kws=None):
        """Completing styling properties of blocks with sensible defaults."""
        # TODO: put values to rcParams
        self._blockprops = dict(
            linewidth=1.0,
        )
        if other_kws is not None:
            self._blockprops.update(other_kws)
        self._blockprops.update(block_kws)
        # separate layout specific parameters
        self._block_init = SubDiagram.split_inits(self._blockprops)

    def _set_flowprops(self, flow_kws, other_kws=None):
        """Completing styling properties of flows with sensible defaults."""
        self._flowprops = dict(
            alpha=0.7,
            facecolor='source',  # TODO: avoid since > match_original == True
            linewidth=0.0
        )
        if other_kws is not None:
            self._flowprops.update(other_kws)
        self._flowprops.update(flow_kws)
        # note: flows have no default layout parameters
        self._flow_init = dict()

    def get_defaults(self,):
        """TODO: write docstring."""
        self._defaults['facecolor'] = next(self._color_cycler)['color']
        self._defaults['edgecolor'] = self._defaults['facecolor']
        return self._defaults

    @classmethod
    def from_memberships(cls, memberships, dcs=None, absentval=None,
                         separate_dcs=False, dcsprops=None, **kwargs):
        """
        Add an new subdiagram from a sequence of membership lists.

        Parameters
        ----------
        memberships : sequence of array-like objects or dataframe.
            The length of the sequence determines the number of columns in the
            diagram. Each element in the sequence must be an array-like object
            representing a membership list. For further details see below.
        dcs : sequence of array-like objects or dataframe (optional)
            Sequence to further classify the classes used in the membership
            lists. If provided *dcs* must be of the same length as
            *memberships* and in each array of *memberships* the values
            map to the index in the corresponding array of *dcs*.
        absentval : int or np.nan (default=None)
            Notes for which  this value is encountered in the membership lists
            are considered to not be present.
        separate_dcs : bool (default=False)
            By default (`False`) all blocks in a dynamic community get a tag
            that styles this community.
            If set to `True` each dynamic community will be drawn as a separate
            sub-diagram and no Tags are used.
        dcsprops : dict or sequence of dict, optional
            Defines the styling for each dynamic community.

        Note that all elements in *memberships* must be membership lists of
        identical length. Further the group identifiers present in a membership
        list must be derived from an enumeration of the groups.

        """
        # create an alluvial instance
        x = kwargs.pop('x', None)
        ax = kwargs.pop('ax', None)
        blockprops = kwargs.pop('blockprops', None)
        flowprops = kwargs.pop('flowprops', None)
        alluvial = Alluvial(x=x, ax=ax, blockprops=blockprops,
                            flowprops=flowprops, **kwargs)
        if dcs is not None:
            dcs = _to_valid_arrays(dcs, 'dcs', np.int)
            if separate_dcs:
                # create individual sub-diagrams for each dc
                raise NotImplementedError('Creating for each dcs a separate'
                                          ' subdiagram is not yet implemented')
            else:
                # nbr_dcs = max(map(lambda x: x.max()))
                nbr_dcs = int(max(np.amax(dc, initial=-1) for dc in dcs) + 1)
                dc_tags = []
                if not np.iterable(dcsprops):
                    dcsprops = nbr_dcs * [dcsprops or dict()]
                for i in range(nbr_dcs):
                    _kwargs = dict(alluvial.get_defaults())
                    _kwargs.update(dcsprops[i])
                    dc_tags.append(alluvial.register_tag(label=f'dc{i}',
                                                         **_kwargs))
                # create a sequence of tag arrays that will match with columns
                tags = []
                for col in dcs:
                    tag_col = [dc_tags[i] for i in col]
                    tags.append(tag_col)
                alluvial.add_from_memberships(memberships, absentval, tags=tags)
        else:
            alluvial.add_from_memberships(memberships, absentval)
        alluvial.finish()
        return alluvial

    def add(self, flows, ext=None, extout=None, fractionflow=False, **kwargs):
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
        fractionflow : bool, default: False
            When set to *False* (the default) the values in *flows* are
            considered to be absolute values.

            If set to *True* the values in *flows* are considered to be
            fractions of block sizes, and the actual flow between columns *i*
            and *i+1* is given by the dot product of *flows[i]* and the array
            of block sizes in column *i*.

            If fractions are provided,  you must set *ext* to provide at least
            the block sizes for the initial column of the Alluvial diagram.
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
        # TODO: make sure empty flows are accepted
        if flows is not None:
            nbr_cols = len(flows) + 1
            flows = _to_valid_arrays(flows, attribute='flows', dtype=np.float64)
        else:
            flows = []
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
            # Note: extout from the first column are ignored
            cinit = flows[0].sum(0)
        else:
            ext = _to_valid_arrays(ext, 'ext', np.float64)
            if isinstance(ext[0], np.ndarray):
                cinit = ext[0]
                if nbr_cols is None:  # if no flows were provided
                    nbr_cols = len(ext)
                    flows = [[] for _ in range(nbr_cols - 1)]
            else:
                cinit = ext[:]
                if nbr_cols is None:
                    nbr_cols = 1
                ext = np.zeros(nbr_cols)  # Note: we overwrite ext in this case
        columns, flows = self._to_cols_and_flows(cinit, flows, ext, extout,
                                                 fractionflow)

        # TODO: extout are not processed so far

        self._extouts.append(extout)
        return self._add(columns=columns, flows=flows, **kwargs)

    # TODO: add support for changing axis=0/1 in case of pandas df
    def add_from_memberships(self, memberships, absentval=None, **kwargs):
        """
        Add an new subdiagram from a sequence of membership lists.

        Parameters
        ----------
        memberships : sequence of array-like objects or dataframe
            The length of the sequence determines the number of columns in the
            diagram. Each element in the sequence must be an array-like object
            representing a membership list. For further details see below.
        absentval : int or np.nan (default=None)
            Notes for which  this value is encountered in the membership lists
            are considered to not be present.

        Note that all elements in *memberships* must be membership lists of
        identical length. Further the group identifiers present in a membership
        list must be derived from an enumeration of the groups.

        """
        memberships = _to_valid_arrays(memberships, 'memberships', np.int)
        for i, membership in enumerate(memberships):
            if np.unique(membership).size != np.max(membership) + 1:
                raise ValueError("The provided membership lists must associate"
                                 " nodes to groups that are continuously"
                                 " numbered starting from 0. This is not the"
                                 f" case at index {i}:\n{membership}")
        # TODO: make it work with dataframe
        # if not isinstance(memberships, (list, tuple)):
        #     try:
        #         memberships.index.values
        #     except AttributeError:
        #         memberships.shape
        #         # return np.arange(y.shape[0], dtype=float), y
        #         pass
        columns, flows = [], []
        # process the first membership list
        membership_current = memberships[0]
        nbr_blocks_last, col = memship_to_column(membership_current, absentval)
        columns, flows = [col], []
        for i in range(1, len(memberships)):
            # create the flow matrix
            nbr_blocks, col = memship_to_column(memberships[i])
            dims = (nbr_blocks, nbr_blocks_last)
            flow, ext = _between_memships_flow(dims, membership_current, memberships[i])
            flows.append(flow)
            # add ext to the column and append to the columns
            columns.append(col + ext)
            membership_current = memberships[i]
            nbr_blocks_last = nbr_blocks
        # do not yet handle the x axis
        # x, columns = self._determine_x(x, columns)
        return self._add(columns, flows, **kwargs)

    def _add(self, columns, flows, blockprops=None, flowprops=None, **kwargs):
        """TODO: write docstring."""
        # get the default initiation parameters for a subdiagram
        _subd_init = dict(self._subd_init)
        _kwargs = normed_kws(kwargs, _ProxyCollection._artistcls)
        # use the provided kwargs to update the initiation parameter
        _subd_init.update(self.split_inits(_kwargs))
        _blockprops = dict(self._block_init)  # pass the default init props
        _blockprops.update(_kwargs)  # update blockprops with kwargs
        if blockprops is not None:
            _blockprops.update(normed_kws(blockprops, _ProxyCollection._artistcls))
        _flowprops = dict(self._flow_init)  # pass the default init props
        _flowprops.update(_kwargs)
        if flowprops is not None:
            _flowprops.update(normed_kws(flowprops, _ProxyCollection._artistcls))
        # set a default label for a subdiagram
        label = _kwargs.pop('label', f'diagram-{self._diagc}')
        diagram = SubDiagram(columns=columns, flows=flows, label=label,
                             blockprops=_blockprops, flowprops=_flowprops,
                             **_subd_init)
        # get the styling params
        self._add_diagram(diagram)
        self._diagc += 1
        return diagram

    def _determine_x(self, x, columns):
        """TODO: write docstring."""
        if x is not None:
            x = _to_valid_arrays(x, 'x')
        elif self._x is not None:
            x = self._x
        else:
            x, columns = index_of(columns)
        return x, columns

    def _add_diagram(self, diagram):
        """
        Add a new subdiagram to the Alluvial diagram.
        """
        self._diagrams.append(diagram)

    def _create_collections(self):
        """TODO: write docstring."""
        # handle the x axis
        x_values = None
        if self.ax.xaxis.units is None:  # check if conversion might be needed
            if self._x is not None:
                self.ax.xaxis.update_units(self._x)
                x_values = copy(self._x)
            else:
                for subd in self._diagrams:
                    subd_x = subd.get_x()
                    if subd_x is not None:
                        self.ax.xaxis.update_units(subd_x)
                        x_values = copy(subd_x)
                        break
        for subd in self._diagrams:
            subd.supplement_x(x_values)

        diag_zorder = 4
        subd_defaults = []
        for diagram in self._diagrams:
            # register the defaults to reuse them for the flows
            subd_defaults.append(dict(self.get_defaults()))  # keep for flows
            defaults = dict(subd_defaults[-1])
            defaults.update(self._blockprops)
            # TODO: Probably should not mess with the zorder, but at least
            # make it a property of Alluvial...
            diagram.create_block_artists(ax=self.ax, zorder=diag_zorder,
                                         **defaults)
        diag_zorder -= diag_zorder
        # ###
        # TODO: first draw the inter sub-diagram flows (corss-flows)
        # ###

        # now draw all other flows
        for diagram, _defaults in zip(self._diagrams, subd_defaults):
            defaults = dict(_defaults)
            defaults.update(self._flowprops)
            diagram.create_flow_artists(ax=self.ax, zorder=diag_zorder,
                                        **defaults)

    def determine_viewlim(self):
        lims = [None, None, None, None]
        extr = [min, min, max, max]
        for diagram in self._diagrams:
            _lims = diagram.get_visuallim()
            for i, lim in enumerate(lims):
                lims[i] = _lims[i] if lim is None else extr[i](lim, _lims[i])
        return lims

    def convert_x(self, x=None):
        if x is None:
            return np.array([])
        elif self.ax.xaxis.units is None:
            return np.array(x)
        else:
            return self.ax.xaxis.convert_units(x)

    def x_collected(self):
        """Get the x coordinates of the columns in all sub-diagrams."""
        combined_x = self.convert_x(self._x).tolist()
        for diagram in self._diagrams:
            combined_x.extend(self.convert_x(diagram.get_x()).tolist())
            _ymin, _ymax = diagram.get_ylim()
            self._ylim = (_ymin, _ymax) if self._ylim is None \
                else (min(self._ylim[0], _ymin), max(self._ylim[1], _ymax))
        self._xlim = min(combined_x), max(combined_x)
        return list(sorted(set(combined_x)))

    def finish(self,):
        """Draw the Alluvial diagram."""
        if self.staled_layout:
            # TODO: advise subds to generate layout
            # TODO: draw the flows in extouts
            self.staled_layout = False
        self._create_collections()
        # do some styling of the axes
        # TODO: make this a function of the layout
        self.ax.xaxis.set_ticks_position('bottom')
        x_positions = self.x_collected()
        x0 = cbook.safe_first_element(x_positions)
        if isinstance(x0, datetime):
            # NOTE: maybe it's not ideal to use autolocator here...
            majloc = self.ax.xaxis.set_major_locator(AutoDateLocator())
            self.ax.xaxis.set_major_formatter(AutoDateFormatter(majloc))
        else:
            self.ax.xaxis.set_major_locator(
                mticker.FixedLocator(x_positions, self.max_nbr_xticks - 1)
            )
        xmin, ymin, xmax, ymax = self.determine_viewlim()
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)
        self.ax.set(frame_on=False)
        self.ax.set(yticks=[])

    # TODO uncomment once in mpl
    # @docstring.dedent_interpd
    def register_tag(self, label, **kwargs):
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
        if label in self._tags:
            _log.warning(
                f"The tag '{label}' was already registered. Registering an"
                " existing tag again has no effect. You must use *update_tag*"
                "if you want to change the styling of an existing tag."
            )
            return None
        # self._tags[label] = _Tag(label=label, **kwargs)
        self._tags[label].set(**kwargs)
        return self._tags[label]

    def select_by_label(self, label):
        """
        Returns a selection of blocks based on a provided label.

        Parameters
        ----------
        label : str
            The label of a subdiagram, tag or block. All blocks associated to
            this label will be returned.

        Note that this method always returns a selection of blocks. This is
        also the case if the label of a subdiagram or tag is provided.
        """
        raise NotImplementedError('To be implemented')

    def update_blocks(self, *selector, **kwargs):
        """
        Update the properties of a selection of blocks.
        For details on the *selector* refer to :meth:`.select_blocks`.
        """
        # select the subdiagrams
        # monitor stale of these subds
        # for each pass rest of selector and kwargs
        subdsel, colsel, blocksel = _separate_selector(*selector)
        _subd_stale = False
        for diagram in self._diagrams[subdsel]:
            is_stale = diagram.update_blocks(colsel, blocksel, **kwargs)
            # TODO: this is likely not done...
            _subd_stale = _subd_stale or is_stale

    def select_blocks(self, *selector):
        """
        Returns a selection of blocks across sub-diagrams and columns.

        Call signature::

            select_blocks([subd_slice], [column_slice], block_slice)

        Each selector, *subd_slice*, *column_slice* and *block_slice* can be a
        single index, a :obj:`slice` or *None*. If a selector is set to *None*
        all elements for this selector will be chosen. Some examples are shown
        below.

        >>> alluv = Alluvial()
        >>> alluv = alluv.add(flows=None, ext=[[1, 2, 3], [2, 3, 1]],
                              layout='bottom')
        >>> alluv = alluv.add(flows=None, ext=[[4, 2, 3], [1, 2, 3]],
                              layout='top')
        >>> alluv.select_blocks(0, None, 0)        # from the 1st subdiagram
        >>>                                        # get the first block in all
        >>>                                        # columns.
        >>> alluv.select_blocks(1, 0, slice(0, 2)) # from the 2nd subdiagram
        >>>                                        # get the first two blocks
        >>>                                        # in the 1st columns.

        Note that the ordering of blocks in a column is determined by the
        layout. As a consequence `alluv.select_blocks(None, None, 0)` will
        select the block largest block in all columns of all subdiagrams if the
        layout is `'bottom'` and the smallest block in case of `'top'`.
        Therefore, `select_blocks` might be difficult to use on columns with
        the layouts `'centered'` and `'optimized'`.
        """
        subdsel, colsel, blocksel = _separate_selector(*selector)
        blocks = []
        for diagram in self._diagrams[subdsel]:
            for col in diagram[colsel]:
                blocks.extend(col[blocksel])
        return blocks

    def tag_blocks(self, tag, *args):
        """
        Tagging a selection of blocks.

        Parameters
        ----------
        tag : str or `._Tag`
            Identification of a tag. This can either be a tag label or directly
            a `._Tag` object.
        args :

        TODO: description of selection procedure passing slices to args.
        """
        if isinstance(tag, str):
            tag = self._tags[tag]  # this creates a new tag if it did not exist
        blocks = self.select_blocks(*args)
        for block in blocks:
            block.add_tag(tag)

    def style_tag(self, label, **props):
        """
        Note that when attached to a block, the styling of a tag overwrites the
        styling defined in the sub-diagram the block belongs to. One exception
        is if the sub-diagram has a colormap defined. In this case the
        facecolor or colormap of a tag will be ignored on all tagged blocks of
        this sub-diagram. This is because a PatchCollection re-sets the
        facecolor at drawing time if a colormap is provided.
        """
        self._tags[label].set(**props)


# TODO: legend handler for subdiagram and for alluvial diagram
class AlluvialHandler:
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        width, height = handlebox.width, handlebox.height
        # TODO: construct a simple alluvial diag
        patch = _Block(height=height, xa=x0, ya=y0, width=width, fc='red',
                       transform=handlebox.get_transform())
        # patch = mpatches.Rectangle([x0, y0], width, height, facecolor='red',
        #                            edgecolor='black', hatch='xx', lw=3,
        #                            transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


# set the legend handler for an alluvial diagram
Legend.update_default_handler_map({SubDiagram: AlluvialHandler()})
