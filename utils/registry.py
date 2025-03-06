class ClassOrFunctionRegistry:
    def __init__(self) -> None:
        self._entries = {}

    def add_entry(self, c: type, force: bool = False) -> None:
        if force and c.__name__ in self._entries:
            del self._entries[c.__name__]
        self._entries[c.__name__] = c

    def del_entry(self, c):
        if c in self.get_entry_names():
            del self._entries[c]
        else:
            raise KeyError(f"{c} is not in entry names")

    def del_all_entries(self):
        if self._entries is not {}:
            for name in self.get_entry_names():
                self.del_entry(name)
        else:
            pass

    def get_entry_names(self) -> list[str]:
        return sorted(self._entries.keys())

    def register(self, c: type):
        self.add_entry(c=c)
        return c

    def force_register(self, c: type):
        self.add_entry(c=c, force=True)

    def __getitem__(self, key: str) -> type:
        return self._entries[key]

    def __contains__(self, key: str) -> bool:
        return key in self._entries.keys()

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__}: [" + ", ".join(self.get_entry_names()) + "]>"
        )
