class Pipeline():

    def __init__(self, *args):
        self._pipeline = list(args)

    def __getitem__(self, arg):
        return self._pipeline[arg]

    def __call__(self, arg):
        return self.forward(arg)

    def forward(self, X):
        for module in self._pipeline:
            X = module(X)
        
        return X

    def train(self, X, y=None):
        if y == None:
            for module in self._pipeline:
                X = module(X)
        else:
            for module in self._pipeline:
                try:
                    temp = module.train(X, y)
                    X = temp
                except:
                    X = module(X)
        
        return X
            