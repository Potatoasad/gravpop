class SampledFromAnalytic(SampledPopulationModel):
    def __init__(self, analytic_model):
        self.analytic_model = analytic_model
        self.__dict__.update(analytic_model.__dict__)
        
    def __call__(self, data, params):
        return self.analytic_model.evaluate(data, params)
        
    def evaluate(self, data, params):
        return self.analytic_model.evaluate(data, params)