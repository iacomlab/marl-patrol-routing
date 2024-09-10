from mongo_manager import RepositoryBase

from mongo.entity.resultado_train import ResultadoTrain


class RepositoryMongoResultadoTrain(RepositoryBase[ResultadoTrain]):

    def __init__(self) -> None:
        super().__init__('resultados_train_revista', ResultadoTrain)
