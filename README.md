# Cooperative Patrol Routing: Optimizing Crime Surveillance through Multi-Agent Reinforcement Learning

## Instalation

The code was implemented in Python 3.9. And all the dependencies are specified in the requirements.txt. The code was executed via cpu in windows and via gpu in ubuntu.

## Data

All the data is stored in MongoDB while the configurations and the parameters of the algorithms are in yaml format. Unfortunately, we cannot make the mongodata used publicly available due to confidentiality concerns, as they involve police-related information.

The only information we can publish is the format of the data.

### Data format
The data was stored in a collection named cuadrado_octates_resumido, the format is the following.

  v5: object (internal_position inside the grid, corner left top is 0,0; it is used as an identifier for the cell)
    lat: int 
    lon: int
  zona_algoritmo: int or str (identifier of the zone)
  transitable: bool (is the cell walkable or not)
  cuadricula: object (dimension of the grid)
    - numero_celdas_anchura: int (number of cell for the base of the grid (witdh))
    - numero_celdas_altura: int (number of cell for the side of the grid (height))
    - lat_min: float (real value of minimal latitude inside the grid)
    - lat_max: float (real value of maximum latitude inside the grid)
    - lon_min: float (real value of minimal longitude inside the grid)
    - lon_max: float (real value of maximum longitude inside the grid)
  vecinos: array[] (Identifiers of the neightbours cells (v5))
  vecinos_vias: array[] (Identifiers of the neightbours cells which has a common street (v5))
  
  
