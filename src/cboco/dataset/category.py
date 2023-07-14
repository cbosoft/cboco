class Category:

    def __init__(self, id: int, name: int, **extra):
        self.id = id
        self.name = name
        self.extra = extra
    
    def to_dict(self) -> dict:
        return dict(
            id=self.id,
            name=self.name,
            **self.extra
        )