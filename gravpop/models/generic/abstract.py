class AbstractPopulationModel:
	pass


class SampledPopulationModel(AbstractPopulationModel):
	def evaluate(self, data, params):
		return self.__call__(data, params)

class AnalyticPopulationModel(AbstractPopulationModel):
	pass



class SpinPopulationModel(AbstractPopulationModel):
	pass

class SpinMagnitudePopulationModel(SpinPopulationModel):
	pass

class SpinOrientationPopulationModel(SpinPopulationModel):
	pass

class RedshiftPopulationModel(AbstractPopulationModel):
	pass

class MassPopulationModel(AbstractPopulationModel):
	pass