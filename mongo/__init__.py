import os

from dotenv import load_dotenv
from mongo_manager import MongoManager

from .repository.cuadrado_octate_repository import RepositoryMongoCuadradoOctate

load_dotenv()
db = MongoManager(username=os.getenv('MONGO_USERNAME'),
                  password=os.getenv('MONGO_PASSWORD'),
                  auth_source=os.getenv('MONGO_AUTH_SOURCE'),
                  db=os.getenv('MONGO_DB'),
                  port_local=int(os.getenv('MONGO_PORT')))
