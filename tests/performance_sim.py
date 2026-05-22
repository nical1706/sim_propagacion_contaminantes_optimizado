import numpy as np
import time
import gc

from utils import *

import contaminationSeparated
import simulator_core_no_omp
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


sim_data_n = simulator_core_no_omp.SimulationData(
    delta=delta, 
    corner_factor=corner_factor, 
    gamma=gamma, 
    num_cartypes=num_car_types, 
    num_pollutants=num_pollutants, 
    WN=WN, 
    WE=WE
)
sim_data = simulator_core_omp.SimulationData(
    delta=delta, 
    corner_factor=corner_factor, 
    gamma=gamma, 
    num_cartypes=num_car_types, 
    num_pollutants=num_pollutants, 
    WN=WN, 
    WE=WE
)



##################################################
##    PRUEBA RENDIMIENTO VERSIONES SIMULADOR    ##
##################################################

#Parámetro de número muestras
n = 10

#Rendimiento respecto a espacio
for i in range(100, 1001, 100):
    #Parámetros
    ls_py, ls_no_omp, ls_omp = [], [], []
    width, height = i, i

    acc, G = init_data_python(width, height, times, car_types, pollutants)

    contaminationSeparated.width = width
    contaminationSeparated.height = height
    contaminationSeparated.times = times
    contaminationSeparated.carTypes = car_types
    contaminationSeparated.pollutants = pollutants
    contaminationSeparated.WN = WN
    contaminationSeparated.WE = WE
    contaminationSeparated.gamma = gamma
    contaminationSeparated.delta = delta
    contaminationSeparated.corner_factor = corner_factor
    contaminationSeparated.acc = acc
    contaminationSeparated.G = G

    #Versión Python
    for j in range(n):
        #Desactivación de Garbage Collector para evitar que interfiera en mediciones
        gc.disable()
        t_inicio_simulacion = time.perf_counter()
        P = contaminationSeparated.get_P_whole(acc)
        t_fin_simulacion = time.perf_counter()
        ls_py.append(t_fin_simulacion - t_inicio_simulacion)
        #Activación de Garbage Collector
        gc.enable()
        #Liberado memoria antes de la siguiente iteración
        P = None
        #Forzado de Garbage Collector para limpiar residuos en memoria
        gc.collect()
        print(f"Python it: {j}")
    print(f"Tiempo medio de Python para matriz {i}*{i}: {sum(ls_py)/n}")

    _, map_data, _, G_numpy_para_C = init_data_no_omp(width, height, times, car_types, pollutants)

    #Versión C no OMP
    for j in range(n):
        #Desactivación de Garbage Collector para evitar que interfiera en mediciones
        gc.disable()
        t_inicio_simulacion = time.perf_counter()
        P = simulator_core_no_omp.get_P_whole_p(sim_data_n, map_data, car_types, pollutants, G_numpy_para_C)
        t_fin_simulacion = time.perf_counter()
        ls_no_omp.append(t_fin_simulacion - t_inicio_simulacion)
        #Activación de Garbage Collector
        gc.enable()
        #Liberado memoria antes de la siguiente iteración
        P = None
        #Forzado de Garbage Collector para limpiar residuos en memoria
        gc.collect()
        print(f"No OMP it: {j}")
    print(f"Tiempo medio de C sin OMP para matriz {i}*{i}: {sum(ls_no_omp)/n}")

    _, map_data, _, G_numpy_para_C = init_data_omp(width, height, times, car_types, pollutants)

    #Versión C OMP
    for j in range(n):
        #Desactivación de Garbage Collector para evitar que interfiera en mediciones
        gc.disable()
        t_inicio_simulacion = time.perf_counter()
        P = simulator_core_omp.get_P_whole_p(sim_data, map_data, car_types, pollutants, G_numpy_para_C)
        t_fin_simulacion = time.perf_counter()
        ls_omp.append(t_fin_simulacion - t_inicio_simulacion)
        #Activación de Garbage Collector
        gc.enable()
        #Liberado memoria antes de la siguiente iteración
        P = None
        #Forzado de Garbage Collector para limpiar residuos en memoria
        gc.collect()
        print(f"OMP it: {j}")
    print(f"Tiempo medio de C con OMP para matriz {i}*{i}: {sum(ls_omp)/n}")


#Rendimiento respecto a tiempo
for i in range(20, 201, 20):
    #Parámetros
    times = i
    #Parámetros
    ls_py, ls_no_omp, ls_omp = [], [], []
    
    acc, G = init_data_python(width, height, times, car_types, pollutants)

    contaminationSeparated.width = width
    contaminationSeparated.height = height
    contaminationSeparated.times = times
    contaminationSeparated.carTypes = car_types
    contaminationSeparated.pollutants = pollutants
    contaminationSeparated.WN = WN
    contaminationSeparated.WE = WE
    contaminationSeparated.gamma = gamma
    contaminationSeparated.delta = delta
    contaminationSeparated.corner_factor = corner_factor
    contaminationSeparated.acc = acc
    contaminationSeparated.G = G

    #Versión Python
    for j in range(n):
        #Desactivación de Garbage Collector para evitar que interfiera en mediciones
        gc.disable()
        t_inicio_simulacion = time.perf_counter()
        P = contaminationSeparated.get_P_whole(acc)
        t_fin_simulacion = time.perf_counter()
        ls_py.append(t_fin_simulacion - t_inicio_simulacion)
        #Activación de Garbage Collector
        gc.enable()
        #Liberado memoria antes de la siguiente iteración
        P = None
        #Forzado de Garbage Collector para limpiar residuos en memoria
        gc.collect()
        print(f"Python it: {j}")
    print(f"Tiempo medio de Python para tiempo {i}: {sum(ls_py)/n}")

    _, map_data, _, G_numpy_para_C = init_data_no_omp(width, height, times, car_types, pollutants)

    #Versión C no OMP
    for j in range(n):
        #Desactivación de Garbage Collector para evitar que interfiera en mediciones
        gc.disable()
        t_inicio_simulacion = time.perf_counter()
        P = simulator_core_no_omp.get_P_whole_p(sim_data_n, map_data, car_types, pollutants, G_numpy_para_C)
        t_fin_simulacion = time.perf_counter()
        ls_no_omp.append(t_fin_simulacion - t_inicio_simulacion)
        #Activación de Garbage Collector
        gc.enable()
        #Liberado memoria antes de la siguiente iteración
        P = None
        #Forzado de Garbage Collector para limpiar residuos en memoria
        gc.collect()
        print(f"No OMP it: {j}")
    print(f"Tiempo medio de C sin OMP para tiempo {i}: {sum(ls_no_omp)/n}")

    _, map_data, _, G_numpy_para_C = init_data_omp(width, height, times, car_types, pollutants)
    
    #Versión C OMP
    for j in range(n):
        #Desactivación de Garbage Collector para evitar que interfiera en mediciones
        gc.disable()
        t_inicio_simulacion = time.perf_counter()
        P = simulator_core_omp.get_P_whole_p(sim_data, map_data, car_types, pollutants, G_numpy_para_C)
        t_fin_simulacion = time.perf_counter()
        ls_omp.append(t_fin_simulacion - t_inicio_simulacion)
        #Activación de Garbage Collector
        gc.enable()
        #Liberado memoria antes de la siguiente iteración
        P = None
        #Forzado de Garbage Collector para limpiar residuos en memoria
        gc.collect()
        print(f"OMP it: {j}")
    print(f"Tiempo medio de C con OMP para tiempo {i}: {sum(ls_omp)/n}")