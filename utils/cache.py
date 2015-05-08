import pickle
from functools import wraps

def cache_result(filename):
    return cache_result_ex(lambda *args, **kwargs: filename)

def cache_result_ex(filename_f):
    def cache_result_decorator(f):
        @wraps(f)
        def cached_f(*args, reload=False, **kwargs):
            if reload:
                result = f(*args, **kwargs)
                with open(filename_f(*args, **kwargs), 'wb') as file:
                    pickle.dump(result, file)
                return result
            else:
                try:
                    with open(filename_f(*args, **kwargs), 'rb') as file:
                        return pickle.load(file)
                except FileNotFoundError:
                    return cached_f(*args, reload=True, **kwargs)
        return cached_f
    return cache_result_decorator
