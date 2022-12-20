import json

import maelstrom


class Timer:
    """Class for adding and writing logging information"""

    def __init__(self, filename):
        self.filename = filename
        self.results = dict()
        self.count = dict()

    def add(self, name, value):
        if name not in self.results:
            self.results[name] = 0
            self.count[name] = 0

        self.results[name] += value
        self.count[name] += 1

    def __str__(self):
        output = ""
        output += "average Key count\n"
        for k,v in self.results.items():
            c = self.count[k]
            mean = v / c
            sum = v
            output += f"{sum:.2f} {k} {c:g}\n"
        return output

    def write(self):
        maelstrom.util.create_directory(self.filename)
        with open(self.filename, "w") as file:
            output = self.__str__()
            file.write(output)
