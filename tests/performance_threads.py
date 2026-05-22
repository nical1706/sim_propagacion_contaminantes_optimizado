import numpy as np
import time
import gc
import os

from utils import init_data_omp

import simulator_core_omp

####################################
##    PARÁMETROS DE SIMULACIÓN    ##
####################################

car_types = [0, 1, 2] # 0 = EV, 1 = Petrol, 2 = Diesel
pollutants = ['CO2', 'NOx']#['CO2', 'NOx', 'VOC', 'PMexhaust', 'PMexhaustprueba', 'PMnonexhaust25', 'PMnonexhaust10']
gamma = 0.01
delta = 0.1
corner_factor = 1
WN = -0.2
WE = 0.4
width, height, times = 1000, 1000, 200
num_car_types, num_pollutants = len(car_types), len(pollutants)
num_tensors = num_car_types * num_pollutants

sim_data = simulator_core_omp.SimulationData(
    delta=delta, 
    corner_factor=corner_factor, 
    gamma=gamma, 
    num_cartypes=num_car_types, 
    num_pollutants=num_pollutants, 
    WN=WN, 
    WE=WE
)

_, map_data, _, G_numpy_para_C = init_data_omp(width, height, times, car_types, pollutants)



####################################
##    PRUEBA RENDIMIENTO HILOS    ##
####################################

n = 10
ls = []

for j in range(n):
    #Desactivación de Garbage Collector para evitar que interfiera en mediciones
    gc.disable()
    t_inicio_simulacion = time.perf_counter()
    P = simulator_core_omp.get_P_whole_p(sim_data, map_data, car_types, pollutants, G_numpy_para_C)
    t_fin_simulacion = time.perf_counter()
    ls.append(t_fin_simulacion - t_inicio_simulacion)
    #Activación de Garbage Collector
    gc.enable()
    #Liberado memoria antes de la siguiente iteración
    P = None
    #Forzado de Garbage Collector para limpiar residuos en memoria
    gc.collect()
    print(f"OMP it: {j}")
print(f"Tiempo medio de C con OMP con {os.environ['OMP_NUM_THREADS']} hilos: {sum(ls)/n}")
