# an infinitely looping iterator
class infiniter():
    def __init__(self, iterable):
        self.iterable=iterable
        self.it = iter(iterable)
    def __iter__(self):
        return self
    def __next__(self):
        try:
            return next(self.it)
        except StopIteration:
            self.it = iter(self.iterable)
            return next(self.it)
    def next(self):
        return self.__next__()