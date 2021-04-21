class TagMissingError(AttributeError):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name


class TagTypeError(TypeError):
    pass


class ConsistencyError(ValueError):
    pass
