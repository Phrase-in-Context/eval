""" Abstract classes """


class AbsExtractor(object):
    def __init__(self):
        pass

    def extract(self, texts):
        assert False, "implement extract()"


class AbsSearch(object):
    def __init__(self):
        pass

    def search(self, query, top_n):
        assert False, "implement search"
