import json
import os

import pandas as pd

from mongo.entity.resultado_train import ResultadoTrain
from mongo.repository.resultados_trains_repository import RepositoryMongoResultadoTrain
import gridfs
from mongo import db

if __name__ == '__main__':
    repo = RepositoryMongoResultadoTrain()
    grid = gridfs.GridFS(db.bd, 'resultados_train_gridfs_revista')
    path = 'results_revista/neww/'
    for filename in os.listdir(path):
        # if filename.startswith('mappo'):
        #     continue
        # if filename.startswith('vdppo'):
        #     continue
        # if filename.startswith('matrpo'):
        #     continue
        # if filename.startswith('vda'):
        #     continue
        # names = filename.split('_')
        # d = {'algorithm': names[0], 'type_network': names[1], 'env': '_'.join(names[2:])}
        print(filename)
        path_dir = f'{path}{filename}'
        # for filename2 in os.listdir(path_dir):
        if os.path.isdir(path_dir) and len(filename) > 5:
            print('-')
            d2 = {'algorithm': 'vdppo', 'env': 'hotspots'}
            # path_dir_example = f'{path_dir}/{filename}'
            path_dir_example = path_dir
            names = filename.split('--')
            d2['name'] = filename
            d2['date'] = names[1]
            d2['path'] = f'{filename}'
            results = None
            print(path_dir_example)
            for filename3 in os.listdir(path_dir_example):
                if not os.path.isdir(os.path.join(path_dir_example, filename3)):
                    if filename3 == 'params.json':
                        d2['params'] = json.loads(open(os.path.join(path_dir_example, filename3), 'r').read())
                        d2['area_algoritmo'] = d2['params']['model']['custom_model_config']['env_args']['area_algortimo']
                        d2['size_square_obs'] = d2['params']['model']['custom_model_config']['env_args']['size_square_obs']
                        d2['patrols'] = d2['params']['model']['custom_model_config']['env_args']['patrols']
                        d2['reward_exploration'] = d2['params']['model']['custom_model_config']['env_args'].get('reward_exploration', '5')
                        d2['out_of_zone'] = d2['params']['model']['custom_model_config']['env_args'].get('out_of_zone', '-25')
                        d2['initial_position'] = d2['params']['model']['custom_model_config']['env_args'].get('initial_position', 'random')
                    elif filename3 == 'progress.csv':
                        df_results = pd.read_csv(os.path.join(path_dir_example, filename3))
                        results = df_results.to_dict('records')
            r = ResultadoTrain(**d2)
            repo.insert_one(r)
            grid.put(json.dumps(results), filename=r.name_gridfs, encoding='utf8')
            # grid.get()


