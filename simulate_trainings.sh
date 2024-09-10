#!/bin/bash

echo $$

echo "Inicializando entorno"
source ./venv/bin/activate

min=1
max=5
for a in 'vdppo'
do
for i in `seq $min $max`
do
    echo "Inicio del entrenamiento con el algoritmo $a y con el identificador $i"
    CUDA_VISIBLE_DEVICES=1, python render_no_render_2.py -ie $i -a $a
    echo "Finalizando el entrenamiento"
done
done

deactivate
echo "Desactivando entorno"
