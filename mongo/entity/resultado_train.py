import json

from mongo_manager import ObjetoMongoAbstract


class ResultadoTrain(ObjetoMongoAbstract):
    def __init__(self, algorithm, env, area_algoritmo, size_square_obs, name, path, params, date,
                 name_gridfs=None, hash_params=None, _id=None,
                 **kwargs):
        super().__init__(_id, **kwargs)
        self.algorithm = algorithm
        self.area_algoritmo = area_algoritmo
        self.size_square_obs = size_square_obs
        self.name = name
        self.params = params
        self.hash_params = hash(json.dumps(self.params)) if hash_params is None else hash_params
        self.env = env
        self.path = path
        self.date = date
        self.name_gridfs = f'{self.name}_{self.hash_params}' if hash_params is None else name_gridfs


