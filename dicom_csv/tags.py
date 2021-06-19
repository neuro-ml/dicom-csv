import inspect
from itertools import groupby
from .utils import Series, Instance, collect
from .exceptions import *

__all__ = ['get_tag', 'get_common_tag', 'drop_duplicated_instances']


def get_tag(instance: Instance, tag, default=inspect.Parameter.empty):
    try:
        return getattr(instance, tag)
    except AttributeError as e:
        if default == inspect.Parameter.empty:
            raise TagMissingError(tag) from e
        else:
            return default


def get_common_tag(series: Series, tag, default=inspect.Parameter.empty):
    try:
        try:
            unique_values = {get_tag(i, tag) for i in series}
        except TypeError:
            raise TagTypeError('Unhashable tags are not supported.')

        if len(unique_values) > 1:
            raise ConsistencyError(f'{tag} varies across instances.')

        value, = unique_values
        return value

    except (TagMissingError, TagTypeError, ConsistencyError):
        if default == inspect.Parameter.empty:
            raise
        else:
            return default
        
        
def _get_sop_uid(instance):
    return str(get_tag(instance, 'SOPInstanceUID'))


@collect
def drop_duplicated_instances(series: Series) -> Series:
    """ Arbitarly changes the order of instances. """
    
    series = sorted(series, key=_get_sop_uid)
    for _, duplicated in groupby(series, key=_get_sop_uid):
        yield list(duplicated)[0]
