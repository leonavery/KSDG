import fenics as fe
import numpy as np
import ufl
import ast
import copy
import collections
import itertools
from .ksdgexception import KSDGException

class Parameter:
    """Syntactic sugar for getter/setter pair
    Usage:

    p = Parameter(getter, setter)

    Set parameter value:
    p(value)
    p.val = value
    p.set(value)

    Retrieve parameter value:
    p()
    p.val
    p.get()
    """
    def __init__(self, getter, setter):
        """Create parameter

        Required positional parameters:
        getter: called with no arguments, retrieves the parameter value.
        setter: called with value, sets the parameter.
        """
        self._get = getter
        self._set = setter
        
    def __call__(self, val=None):
        if val is not None:
            self._set(val)
        return self._get()
    
    def get(self):
        return self._get()
    
    def set(self, val):
        self._set(val)
        
    @property
    def val(self):
        return self._get()
    
    @val.setter
    def val(self, val):
        self._set(val)

def find_duplicates(list):
    slist = list.copy()
    slist.sort()
    dups = [ i for i,j in zip(slist[1:], slist[:-1]) if i == j ]
    return dups

class ParameterList:
    """Wrapper around OrderDict

    Required positional parameter
    parameters

    This must be a list of the form
    [(key, default, help),
     (key, default, help),
     (key, p, default, help),
     ...
    ]

    Each tuple in the list must be of length 3 or length 4. In a
    length 3 tuple each help is a string and default is a value of any
    type. The parameter will be stored in an OrderedDict within the
    ParameterList object and its value initialized to the default.

    The length 4 form is used for a parameter that is stored somwhere
    outside the ParameterList. p is a Parameter object that can be
    used to access it. Its value is left unchanged.

    default and help are stored for later use.
    """
    def __init__(self, parameters=()):
        self.values = collections.OrderedDict()
        self.ps = collections.OrderedDict()
        self.keys = self.ps.keys
        self.defaults = collections.OrderedDict()
        self.helps = collections.OrderedDict()
        self.add(parameters)

    def add(self, parameters):
        for param in parameters:
            if len(param) == 3 or len(param) == 2:
                k, d = param[:2]
                h = param[2] if len(param) == 3 else None
                if k in self:
                    p = self.ps[k]
                else:
                    def getter(vd=self.values, key=k):
                        return vd[key]
                    
                    def setter(val, vd=self.values, key=k):
                        vd[key] = val
                        return None
                    
                    p = Parameter(getter, setter)
                    p(d)
            elif len(param) == 4:
                k, p, d, h = param
            else:
                raise ValueError(
                    ('parameter element has length %d,' +
                     '2, 3 or 4 is required')%(len(param))
                )
            self.ps[k] = p
            self.defaults[k] = d
            self.helps[k] = h

    def update(self, parameters):
        if hasattr(parameters, 'items') and callable(parameters.items):
            parameters = parameters.items()
        for param in parameters:
            k, v = param
            if k not in self:
                self.add([param])
            self[k] = v
            
    def items(self):
        for k,p in self.ps.items():
            yield((k, p()))
        return

    def __getitem__(self, key):
        try:
            p = self.ps[key]
        except KeyError as e:
            raise
        return p()

    def __setitem__(self, key, value):
        if not key in self:
            self.values[key] = value
            def getter(vd=self.values, key=key):
                return vd[key]

            def setter(val, vd=self.values, key=key):
                vd[key] = val
                return None

            p = Parameter(getter, setter)
            self.ps[key] = p
        p = self.ps[key]
        return p(value)

    def __delitem__(self, key):
        try:
            del(self.ps[key])
        except KeyError:
            raise
        try:
            del(self.values[key])
        except KeyError:
            pass
        del(self.defaults[key])
        del(self.helps[key])

    def __contains__(self, key):
        return key in self.ps

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def decode(self, params, allow_new=False):
        """Decode parameter commandline arguments

        Required argument:
        params: a list of strs of the form 'key=value'
        """
        keys = [ arg.split('=', maxsplit=1)[0] for arg in params ]
        dups = find_duplicates(keys)
        if dups:
            raise KSDGException(
                'duplicated parameters: ' + ', '.join(dups)
            )
        for arg in params:
            k,val = arg.split('=', maxsplit=1)
            try:
                v = ast.literal_eval(val)
            except ValueError:            # if eval fails,
                v = val                   # ...just use str
            try:
                p = self.ps[k]
            except KeyError as e:         # k not in list
                if allow_new:             # add it if allowed
                    self[k] = v
                else:
                    raise                 # otherwise rethrow KeyError
            else:                         # k is in the list: 
                t = type(p())             # just set value
                v = t(v)
                p(v)

    def params(self):
        """List the parameters

        Returns a list of the parameters of this ligand. Each
        parameter is returned as a tuple of length four. The first
        element of the tuple is the key of the parameter. The second
        element of the tuple is a Parameter that can be used to access
        the parameter. This is followed by the default value of the
        parameter and a str of help text for the parameter. (If either
        the default or the help is not available, None is used.)
        """
        ps = []
        for k in self.ps.keys():
            p = self.ps[k]
            try:
                d = self.defaults[k]
            except KeyError:
                d = None
            try:
                h = self.helps[k]
            except KeyError:
                h = None
            ps.append((k, p, d, h))
        return ps

    def str(self):
        s = ''
        for k,p,d,h in self.params():
            s += '{key}={val}\n'.format(key=k, val=p())
        return s

    def __str__(self):
        return self.str()

class Ligand(collections.OrderedDict):
    """wrapper around dict to access elements as attributes."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(err)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del(self[name])
        except KeyError as err:
            raise AttributeError(err)

    def params(self):
        """List the parameters of this ligand.

        Returns a list of the parameters of this ligand. Each
        parameter is returned as a tuple of length four. The first
        element of the tuple is a string of the form
        name,<group_num>,<ligand_num> where <group_num> and
        <ligand_num> are replaced by integers, e.g. weight,1,1 for the
        weight parameters of ligand 1 of group 1. The names are the
        first elements of the tuples in
        LigandGroups.default_ligand_parameters. The second element of
        the tuple is a Parameter that can be used to access the
        parameter. This is followed by the default value of the
        parameter and a str of help text for the
        parameter.
        """
        ps = []
        for name,d,h in LigandGroups.default_ligand_parameters:
            key = '%s,%d,%d'%(name, self.groupnum, self.ligandnum)
            def getter(name=name, lig=self):
                return lig[name]

            def setter(value, name=name, lig=self):
                lig[name] = value

            p = Parameter(getter, setter)
            hstr = h.format(group=self.groupnum, ligand=self.ligandnum)
            ps.append((key, p, d, hstr))
        return ps
    
    def fourier_series(self, adjust=True):
        """Approximate a ligand with a Fourier series.
        
        Required parameter:
        lig: the ligand to be replaced.

        Optional parameter:
        adjust=True: If this parameter is True, the s value for all
        series members will be adjusted so that the total
        concentration of ligand at local steady-state equals that for
        the single ligand.

        This function implements a simple way to implement the
        diffusion of a ligand in depth. Suppose the worms are moving
        on the W x H surface of a W x H x Depth block of agar. The worms
        are trapped on the surface, but the ligands can diffuse not
        only in the x and y dimensions, but also the z. The 3D
        distribution of ligand can be represented as a Fourier series:

        [l](x,y,z,t) = sum([l_i](x,y,t) cos(i*pi*z/Depth) 
                           for i in range(0, inf))

        The basis functions cos(i*pi*z/D) satisfy Neumann boundary
        conditions at z=0 and z=D, and all have the value 1 at
        z=0. Each series concentration function [l_i] satifies the
        usual Keller-Segel PDEs in x,y. However, the diffusion term
        contributes a decay rate of -D * pi^2 * i^2 / Depth^2 to
        component i. This is simply added to gamma.

        This function uses the 'D', 'depth', 'gamma', 's', and
        'series' attributes of the ligand passed as its
        argument. self.series is converted to int to give the number of
        Fourier components. self.depth and self.D are used to calculate
        the diffusional decay rate of each component, which is added
        to gamma. Finally, the components are all assumed to be
        produced at equal rates, each self.s/self.series, so they add up
        to the original self.s. This function also adds two attributes
        to self: 'fourier_term' and 'omega'. fourier_term corresponds
        to i in the above expression and runs from 0 to self.series -
        1. self.omega is pi * i / Depth. The terms are returned as a
        list of length self.series in order of fourier_term.

        If self.series == 1, a list of length 1 is returned, and its
        sole member is identical to the ligand provided as input,
        except for the addition of fourier_term = 0 and omega = 0.0
        attributes. If self.series doesn't exist, 1 is used, giving
        this effective no-op result.
        """
        try:
            n = round(self.series)
        except AttributeError:
            n = 1
        
        ligs = []
        for i in range(n):
            ligi = copy.deepcopy(self)
            ligi.fourier_term = i
            try: 
                ligi.s /= n
                ligi.weight /= n
                ligi.omega = np.pi * i / ligi.depth
                ligi.gamma += ligi.D * ligi.omega**2
            except AttributeError:
                # fail gracefully if necessary attributes not present
                pass
            ligs.append(ligi)
        if adjust:
            singlessconc = self.s / self.gamma
            seriesssconc = sum([
                    lig.s / lig.gamma for lig in ligs
            ])
            for lig in ligs:
                lig.s *= singlessconc / seriesssconc
        return ligs

class LigandGroup:
    """A single group of ligands

    Just a convenient holder for all the info that defines a ligand
    group. Generally gets attributes by direct assignment by clients.
    Has some useful methods.
    """
    def __init__(
        self,
        groupnum=1,
        nligands=0
    ):
        """Create an empty LigandGroup"""
        self.groupnum = groupnum
        self.nligands = nligands
        self.ligands = []
        for i in range(1, nligands+1):
            lig = Ligand({
                n: d for n,d,h in LigandGroups.default_ligand_parameters
            })
            lig.groupnum = groupnum
            lig.ligandnum = i
            lig.nligands = nligands
            self.ligands.append(lig)
        for n,d,h in LigandGroups.default_group_parameters:
            setattr(self, n, d)
        self.blank = True

    # Define alpha, beta, and nligands as properties so that a setter can
    # propagate any change to all ligands in the group.
    #
    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha
        for lig in self.ligands:
            lig.alpha = alpha

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, beta):
        self._beta = beta
        for lig in self.ligands:
            lig.beta = beta

    def params(self):
        """List the parameters of ligands in this group

        Returns a list of the parameters of this LigandGroup. Each
        parameter is returned as a tuple of length two. The first
        element of the tuple is a string of the form
        "('name', <group_num>, <ligand_num>)" or "('name',
        <group_num>)" where <group_num> and
        <ligand_num> are replaced by integers. The first form is used
        for ligand parameters (keys of LigandGroups.ligand_defaults)
        and the second for group parameters (keys of
        LigandGroups.ligand_defaults). 
        """
        ps = []
        for name,d,h in LigandGroups.default_group_parameters:
            key = '%s,%d'%(name, self.groupnum)
            def getter(name=name, group=self):
                return getattr(group, name)

            def setter(value, name=name, group=self):
                setattr(group, name, value)

            p = Parameter(getter, setter)
            hstr = h.format(group=self.groupnum)
            ps.append((key, p, d, hstr))
        for lig in self.ligands:
            ps += lig.params()
        return ps

    def collect(
        self,
        gvals,
        name,
        groupnum=None,
    ):
        if groupnum is None:
            groupnum = self.groupnum
        else:
            if (not self.blank) and (groupnum != self.groupnum):
                raise KSDGException(
                    'inconsistent group numbers %d and %d'%(self.groupnum, groupnum)
                )
            self.groupnum = groupnum
        vals = []
        for g, v in gvals:
            if groupnum == int(g):
                vals.append(v)
        if self.blank:
            self.nligands = len(vals)
        # else:
            # if self.nligands != len(vals):
            #     raise KSDGException(
            #         'inconsistent number of %s values in group %d'%(name, groupnum)
            #     )
        if self.blank:
            self.ligands = [None] * self.nligands
            gparams = LigandGroups.group_defaults.copy()
            for p in gparams.keys():
                if hasattr(self, p):
                    gparams[p] = getattr(self, p)
            for i in range(self.nligands):
                self.ligands[i] = Ligand(groupnum=groupnum, **gparams)
        for i, l in enumerate(self.ligands):
            try:
                self.ligands[i][name] = vals[i]
                self.ligands[i].ligandnum = i + 1
            except IndexError:
                pass
        self.blank = False

    def fourier_series(self):
        for i, l in enumerate(copy.deepcopy(self.ligands)):
            sligands = l.fourier_series()
            self.ligands[i:i+1] = sligands
        self.nligands = len(self.ligands)
        for i, l in enumerate(self.ligands):
            l.ligandnum = i + 1
            l.nligands = self.nligands
        
    def V(
        self,
        Us
    ):
        """Calculate potential due to this group
        
        Required argument:
        Us: an iterable producing the concentrations of the ligands in the 
        group. These may be numbers, but in the most important applcation
        they are UFL expressions.
        """
        if len(Us) != self.nligands:
            raise KSDGException(
                'wrong number of ligands %d, should be %d'%(len(Us), self.nligands)
            )
        if self.nligands == 0:
            return(0.0)
        sU = sum(l.weight * U for l,U in zip(self.ligands, Us))
        return(
            - self.beta * ufl.ln(self.alpha + sU)
        )

class LigandGroups:
    """Groups of chemotactic ligands
    The multiple ligand options work as follows. First, the ligands
    are divided up into groups, and each group may have multiple
    ligands. The contribution of group i to the potential is

    V[i] = beta[i] log(alpha[i] + sum(w[i, j] U[i, j]))

    The V[i] are then summed over i to get the total potential. This
    is a bit overly complicated. Most of the function of multiple
    ligans could be handled simply by the weighted sum inside the
    log. However, I want to be able to have both attractive and
    repellent ligands. A negative weight would lead to the possibility
    of a negative argument to the log, which would be bad. By
    separating the ligands into groups, I can make inhibitory ligands
    by setting beta[i] negative.

    The command line arguments are --alpha <alpha> and --beta <beta>
    for defining groups, and --weight <g> <weight>, --s <g> <s>,
    --gamma <g> <gamma>, and --D <g> <D> for defining ligands in group
    g. (Group numbers start with 1.) The number of --alpha and --beta
    arguments must be the same, or ksdgsolver3 will throw an
    exception. Likewise, --weight, --s, --gamma, and --D arguments
    must come in matching sets. 
    """

    group_defaults = collections.OrderedDict({
        'alpha': [1.0],
        'beta': [1.0],
    })

    default_group_parameters = [
        ('alpha', 1.0, 'V = -beta*log(w.U + alpha) for group {group}'),
        ('beta', 1.0, 'V = -beta*log(w.U + alpha) for group {group}'),
        ('nligands', 1, 'number of ligands in group {group}'),
    ]

    ligand_defaults = collections.OrderedDict({
        'weight': [[1.0, 1.0]],
        's': [[1.0, 10.0]],
        'gamma': [[1.0, 1.0]],
        'D': [[1.0, 0.01]],
        'series': [[1.0, 1.0]],
        'depth': [[1.0, 0.4]]
    })

    default_ligand_parameters = [
        ('weight', 1.0, 'weight of ligand {ligand} in group {group}'),
        ('s', 1.0, 'secretion rate of ligand {ligand} in group {group}'),
        ('gamma', 1.0, 'decay rate of ligand {ligand} in group {group}'),
        ('D', 1.0, 'diffusion of ligand {ligand} in group {group}'),
        ('series', 1, 'Fourier series component of ligand {ligand} in group {group}'),
        ('depth', 0.4, 'depth for ligand {ligand} in group {group}'),
    ]

    def __init__(
        self,
        command_line_arguments=None,
        **kwargs
    ):
        """Creates a LigandGroups
        Optional arguments:
        command_line_arguments: This should be the return from
        parse_commandline, a NameSpace with attributes alpha, beta,
        weight, s, gamma, and D.
        alpha, beta, weight, s, gamma, D: The defaults for these
        arguments are found in the class variables group_defaults and
        ligand_defaults. alpha should be a list of numbers, the value
        of alpha for each group, e.g. [1.0]. beta is also a list of
        numbers. Each of weight, s, gamma, and D is a nested list,
        each of whose elements is a list of two numbers, e.g. [1.0,
        10.0]. The first of these numbers, converted to int, is the
        number of the group to which this ligand belongs. The second
        is the value of the parameter for that ligand.
        alpha, beta, weight, s, gamma, and D are ignored if
        command_line_arguments is provided. 
        """
        ligand_params = self.ligand_defaults.keys()
        if (command_line_arguments is None) and (not kwargs):
            self.groups = []
            return(None)
        if (command_line_arguments is not None):
            if kwargs:
                raise KSDGException(
                    'command_line_arguments and parameters are mutually exclusive'
                )
            kwargs = dict(command_line_arguments._get_kwargs())
        if ('ngroups' in kwargs and kwargs['ngroups']):
            self.groups = []
            nldict = dict(kwargs['nligands']) if 'nligands' in kwargs else {}
            for g in range(1, kwargs['ngroups']+1):
                nligands = nldict[g] if g in nldict else 1
                self.groups.append(LigandGroup(groupnum=g, nligands=nligands))
            return None
        #
        # try to find groups in params argument
        #
        if 'params' in kwargs:
            params = ParameterList()
            params.decode(kwargs['params'], allow_new=True)
            ngroups = int(params.get('ngroups', 1))
            self.groups = []
            for g in range(1, ngroups+1):
                key = 'nligands,' + str(g)
                nligands = int(params.get(key, 1))
                self.groups.append(LigandGroup(groupnum=g, nligands=nligands))
            return None
        for name,val in self.group_defaults.items():
            if not getattr(command_line_arguments, name):
                kwargs[name] = val
        for name,val in self.ligand_defaults.items():
            if not getattr(command_line_arguments, name):
                kwargs[name] = val
        alpha = kwargs['alpha']
        beta = kwargs['beta']
        ng = len(alpha)
        if len(beta) != ng:
            raise KSDGException('alphas and betas must be equal in number')
        for param in ligand_params:
            gvs = kwargs[param]
            gnums = [int(g) for g,v in gvs]
            gmin = min(gnums)
            gmax = max(gnums)
            if gmin < 1:
                raise KSDGException(
                    'invalid group number %d in parameter %s'%(gmin, param)
                )
            if gmax > ng:
                raise KSDGException(
                    'invalid group number %d in parameter %s'%(gmax, param)
                )
        self.groups = []
        for gnum in range(1, ng+1):
            group = LigandGroup()
            group.alpha = alpha[gnum-1]
            group.beta = beta[gnum-1]
            for param in ligand_params:
                group.collect(kwargs[param], param, gnum)
            # group.fourier_series()
            self.groups.append(group)
            
    def nligands(
        self
    ):
        """Total ligands in all groups"""
        return sum(
            [group.nligands for group in self.groups]
        )

    def ligands(
        self
    ):
        return itertools.chain(
            *[group.ligands for group in self.groups]
        )

    def params(self):
        """List the parameters of ligands in these groups

        Returns a list of the parameters of the LigandGroups. Each
        parameter is returned as a tuple of length two. The first
        element of the tuple is a string of the form
        "('name', <group_num>, <ligand_num>)" or "('name',
        <group_num>)" where <group_num> and
        <ligand_num> are replaced by integers. The first form is used
        for ligand parameters (keys of LigandGroups.ligand_defaults)
        and the second for group parameters (keys of
        LigandGroups.ligand_defaults). 
        """
        ps = []
        for group in self.groups:
            ps += group.params()
        return ps

    def fourier_series(self):
        for group in self.groups:
            group.fourier_series()
        
    def V(
        self,
        Us
    ):
        """Calculate potential due to all groups
        
        Required argument:
        Us: an iterable producing the concentrations of all ligands in the 
        groups, in order. These may be numbers, but in the most important applcation
        they are UFL expressions.
        """
        if len(Us) != self.nligands():
            raise KSDGException(
                'provided %d ligands, need %d'%(len(Us), self.nligands())
            )
        Ufirst = 0
        if self.nligands() == 0:
            return(0.0)
        Ulast = self.groups[0].nligands
        sV = self.groups[0].V(Us[Ufirst:Ulast])
        for group in self.groups[1:]:
            Ufirst = Ulast
            Ulast = Ufirst + group.nligands
            sV = sV + group.V(Us[Ufirst:Ulast])
        return(sV)

