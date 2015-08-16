class LazyProperty(object):
    def __init__(self, fget=None, fset=None, doc=None, name=None):
        self.fget = fget
        self.fset = fset
        if doc is None and fget is not None:
            doc = fget.__doc__
        if name is None:
            if hasattr(fget, '__name__'):
                self.name = fget.__name__
            elif hasattr(fset, '__name__'):
                self.name = fset.__name__
            else:
                raise AttributeError("can't find name")
        self.__doc__ = doc

    @property
    def cache_name(self):
        return '_cached_' + self.name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if hasattr(obj, self.cache_name):
            return getattr(obj, self.cache_name)
        elif self.fget is None:
            raise AttributeError("unreadable attribute")
        result = self.fget(obj)
        setattr(obj, self.cache_name, result)
        return result

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)

    def __delete__(self, obj):
        if hasattr(obj, self.cache_name):
            delattr(obj, self.cache_name)

    def getter(self, fget):
        return type(self)(fget, self.fset, self.fdel, self.__doc__)

    def setter(self, fset):
        return type(self)(self.fget, fset, self.fdel, self.__doc__)

lazy_property = LazyProperty

def declare(name, value=None, changes=None, prefix='_variable_', doc=None):
    var_name = prefix + name

    def getter(self):
        return getattr(self, var_name, value)

    def deleter(self):
        return delattr(self, var_name)

    if changes is None:
        def setter(self, val):
            return delattr(self, var_name)
    else:
        str_changes = []
        for prop in changes:
            if isinstance(prop, str):
                str_changes.append(prop)
            elif hasattr(prop, '__name__'):
                str_changes.append(prop.__name__)
            else:
                str_changes.append(str(prop))

        def setter(self, val):
            for prop in str_changes:
                delattr(self, prop)
            setattr(self, var_name, val)

    return property(getter, setter, deleter)