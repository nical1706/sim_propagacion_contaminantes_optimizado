import numpy as np
import time
import gc

from utils import *

import contaminationSeparated
import simulator_core_no_omp
import simulator_core_omp

car_types = [0, 1, 2] # 0 = EV, 1 = Petrol, 2 = Diesel
pollutants = ['CO2']#['CO2', 'NOx', 'VOC', 'PMexhaust', 'PMexhaustprueba', 'PMnonexhaust25', 'PMnonexhaust10']
#No necesario (ni posible) pasar valores dimensionales y temporales a float32 ya que se usan únicamente en indexación
width, height, times = 100, 100, 1000
num_car_types, num_pollutants = len(car_types), len(pollutants)
num_tensors = num_car_types * num_pollutants
#Para no cometer el error de comparar precisiones diferentes, se establece float32 como la precisión general de todos los datos de tipo decimal (ya que 
#es la que se usa en todo el código numpy)
gamma = np.float32(0.01)
delta = np.float32(0.1)
corner_factor = np.float32(1.0)
WN = np.float32(-0.2)
WE = np.float32(0.4)

#Inicializado datos sim_data implementación no OMP
sim_data_n = simulator_core_no_omp.SimulationData(
    delta=delta, 
    corner_factor=corner_factor, 
    gamma=gamma, 
    num_cartypes=num_car_types, 
    num_pollutants=num_pollutants, 
    WN=WN, 
    WE=WE
)
#Inicializado datos sim_data implementación OMP
sim_data = simulator_core_omp.SimulationData(
    delta=delta, 
    corner_factor=corner_factor, 
    gamma=gamma, 
    num_cartypes=num_car_types, 
    num_pollutants=num_pollutants, 
    WN=WN, 
    WE=WE
)



##############################################
##         CÓDIGO PYTHON MODIFICADO         ##
##############################################

#Para evitar cambios de tipos por parte de Python de int a float64, se han fijado las siguientes variables
CERO = np.float32(0.0)
UNO = np.float32(1.0)
CUATRO = np.float32(4.0)

def get_wind_prec(acc):
    #Inicializado de matrices en 32 bits
    displ_N = np.zeros((width+2, height+2), dtype=np.float32)
    displ_S = np.zeros_like(displ_N)
    displ_E = np.zeros_like(displ_N)
    displ_W = np.zeros_like(displ_N)
    displ_NW = np.zeros_like(displ_N)
    displ_NE = np.zeros_like(displ_N)
    displ_SW = np.zeros_like(displ_N)
    displ_SE = np.zeros_like(displ_N)
    
    sign_WN = int(np.sign(WN))
    sign_WE = int(np.sign(WE))
    
    for p in range(1, width+1):
        for q in range(1, height+1):
            #Casting explícito de las funciones máximo y absoluto
            max_WN = np.maximum(WN, CERO, dtype=np.float32)
            max_m_WN = np.maximum(-WN, CERO, dtype=np.float32)
            max_WE = np.maximum(WE, CERO, dtype=np.float32)
            max_m_WE = np.maximum(-WE, CERO, dtype=np.float32)
            
            abs_WE = np.abs(WE, dtype=np.float32)
            abs_WN = np.abs(WN, dtype=np.float32)

            #Ecuaciones separadas para evitar upcast de Python
            term_N = UNO - np.maximum(acc[p + sign_WE, q - 1], acc[p + sign_WE, q], dtype=np.float32) * abs_WE
            displ_N[p,q] = (acc[p,q] * max_WN * term_N * acc[p, q - 1]).astype(np.float32)
            
            term_S = UNO - np.maximum(acc[p + sign_WE, q + 1], acc[p + sign_WE, q], dtype=np.float32) * abs_WE
            displ_S[p,q] = (acc[p,q] * max_m_WN * term_S * acc[p, q + 1]).astype(np.float32)
            
            term_E = UNO - np.maximum(acc[p + 1, q - sign_WN], acc[p, q - sign_WN], dtype=np.float32) * abs_WN
            displ_E[p,q] = (acc[p,q] * max_WE * term_E * acc[p + 1, q]).astype(np.float32)
            
            term_W = UNO - np.maximum(acc[p - 1, q - sign_WN], acc[p, q - sign_WN], dtype=np.float32) * abs_WN
            displ_W[p,q] = (acc[p,q] * max_m_WE * term_W * acc[p - 1, q]).astype(np.float32)
            
            displ_NE[p,q] = (acc[p,q] * max_WN * max_WE * acc[p + 1, q - 1]).astype(np.float32)
            displ_NW[p,q] = (acc[p,q] * max_WN * max_m_WE * acc[p - 1, q - 1]).astype(np.float32)
            displ_SE[p,q] = (acc[p,q] * max_m_WN * max_WE * acc[p + 1, q + 1]).astype(np.float32)
            displ_SW[p,q] = (acc[p,q] * max_m_WN * max_m_WE * acc[p - 1, q + 1]).astype(np.float32)
            
    #Suma estricta en 32 bits
    sum_wind = (displ_N + displ_S + displ_E + displ_W + displ_NE + displ_NW + displ_SE + displ_SW).astype(np.float32)
    stays = (UNO - sum_wind).astype(np.float32)
    
    wind = (displ_N[1:-1, 2:], displ_S[1:-1, :-2], displ_E[:-2, 1:-1], displ_W[2:, 1:-1], 
            displ_NE[:-2, 2:], displ_NW[2:, 2:], displ_SE[:-2, :-2], displ_SW[2:, :-2], stays[1:-1, 1:-1])
    return wind

def get_difMatrix_prec(acc):
    acc_neig_edge = (acc[0:-2, 1:-1] + acc[2:, 1:-1] + acc[1:-1, 0:-2] + acc[1:-1, 2:]).astype(np.float32)
    acc_neig_corner = (acc[0:-2, 0:-2] + acc[2:, 2:] + acc[2:, 0:-2] + acc[0:-2, 2:]).astype(np.float32)
    
    #Extracción de cálculo respecto a original para forzar fácilmente tipado
    f_factor = (delta / (CUATRO + CUATRO * corner_factor)).astype(np.float32)
    dif_matrix = (UNO - (acc_neig_edge + acc_neig_corner * corner_factor).astype(np.float32) * f_factor).astype(np.float32)
    return dif_matrix

def get_P_whole_prec(acc):
    P = {f"{cartype}_{pollutant}": np.zeros((width+2, height+2, times+1), dtype=np.float32)
				for cartype in car_types
				for pollutant in pollutants}
    wind = get_wind_prec(acc)
    dif_matrix = get_difMatrix_prec(acc)

    #Cambio de variable respecto a código original para evitar sobreescribir acc de entrada
    acc_m = acc[1:-1,1:-1]
    for i in range(times+1):
        for key in P:
            diffusion_edge = (P[key][0:-2, 1:-1, i] + P[key][2:, 1:-1, i] + P[key][1:-1, 0:-2, i] + P[key][1:-1, 2:, i]).astype(np.float32)
            diffusion_corner = (P[key][0:-2, 0:-2, i] + P[key][2:, 2:, i] + P[key][2:, 0:-2, i] + P[key][0:-2, 2:, i]).astype(np.float32)

            f_factor = (delta / (CUATRO + CUATRO * corner_factor)).astype(np.float32)
            dif_P = (dif_matrix * P[key][1:-1, 1:-1, i]).astype(np.float32)
            P[key][1:-1,1:-1, i] = (acc_m * (dif_P + (diffusion_edge + diffusion_corner * corner_factor).astype(np.float32) * f_factor).astype(np.float32)).astype(np.float32)
			
            if i < times:
                m_gamma = (UNO - gamma).astype(np.float32)
                # Wind, sources and loss to atmosphere
                P[key][1:-1, 1:-1, i+1] = (m_gamma * acc_m * (wind[0]*P[key][1:-1, 2:, i] + wind[1]*P[key][1:-1, :-2, i] + wind[2]*P[key][:-2, 1:-1, i] + 
                                            wind[3]*P[key][2:, 1:-1, i] + wind[4]*P[key][:-2, 2:, i] + wind[5]*P[key][2:, 2:, i] + wind[6]*P[key][:-2, :-2, i] + 
                                            wind[7]*P[key][2:, :-2, i] + wind[8]*P[key][1:-1, 1:-1, i] + G[key][1:-1, 1:-1, i]).astype(np.float32)).astype(np.float32)

    return P



##############################################
##         COMPARACIÓN DE PRECISIÓN         ##
##############################################

print("Calculando versión Python modificada")
#Fijado de semilla para evitar datos de prueba distintos
np.random.seed(42)
acc, G = init_data_python(width, height, times, car_types, pollutants)
#Cambio de tipo para no usar int
acc = acc.astype(np.float32)
P_python = get_P_whole_prec(acc)


print("Calculando versión Python original")
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

P_python_o = contaminationSeparated.get_P_whole(acc)


print("Calculando versión no OMP")
#Fijado de semilla para evitar datos de prueba distintos
np.random.seed(42)
_, map_data, _, G_numpy_para_C = init_data_no_omp(width, height, times, car_types, pollutants)
P_n_omp = simulator_core_no_omp.get_P_whole_p(sim_data_n, map_data, car_types, pollutants, G_numpy_para_C)


print("Calculando versión OMP")
#Fijado de semilla para evitar datos de prueba distintos
np.random.seed(42)
_, map_data, _, G_numpy_para_C = init_data_omp(width, height, times, car_types, pollutants)
P_omp = simulator_core_omp.get_P_whole_p(sim_data, map_data, car_types, pollutants, G_numpy_para_C)



atol = 1e-6

print("\nComprobación de diferencia entre código original y modificado")
for key in P_python:
    tensor_py = P_python[key]
    tensor_o = P_python_o[key]
    
    diferencia_absoluta = np.abs(tensor_py - tensor_o)
    max_diff = np.max(diferencia_absoluta)
    
    print(f"Tensor {key}:")
    print(f"  Diferencia máxima absoluta: {max_diff:.8e}")
    
    if np.allclose(tensor_py, tensor_o, atol=atol, rtol=0.0):
        print(f"  Los resultados superan la precisión: {atol}")
    else:
        print(f"  Los resultados difieren con la precisión: {atol}")
        indices_error = np.unravel_index(np.argmax(diferencia_absoluta), diferencia_absoluta.shape)
        print(f"     Mayor discrepancia en coordenada {indices_error}")
        print(f"     Valor Python modificado: {tensor_py[indices_error]}")
        print(f"     Valor Python original:      {tensor_o[indices_error]}")


print("\nComprobación de diferencia entre código Python y C sin OMP")
for key in P_python:
    tensor_py = P_python[key]
    tensor_c = P_n_omp[key]
    
    #Transposición de tensor de Python de (X, Y, Tiempo) a (Tiempo, X, Y) para realizar comparación
    tensor_py_transpuesto = np.transpose(tensor_py, (2, 0, 1))
    
    diferencia_absoluta = np.abs(tensor_py_transpuesto - tensor_c)
    max_diff = np.max(diferencia_absoluta)
    
    print(f"Tensor {key}:")
    print(f"  Diferencia máxima absoluta: {max_diff:.8e}")
    
    if np.allclose(tensor_py_transpuesto, tensor_c, atol=atol, rtol=0.0):
        print(f"  Los resultados superan la precisión: {atol}")
    else:
        print(f"  Los resultados difieren con la precisión: {atol}")
        indices_error = np.unravel_index(np.argmax(diferencia_absoluta), diferencia_absoluta.shape)
        print(f"     Mayor discrepancia en coordenada {indices_error}")
        print(f"     Valor Python modificado: {tensor_py_transpuesto[indices_error]}")
        print(f"     Valor C sin OMP:      {tensor_c[indices_error]}")

print("\nComprobación de diferencia entre código Python y C con OMP")
for key in P_python:
    tensor_py = P_python[key]
    tensor_c = P_omp[key]
    
    #Transposición de tensor de Python de (X, Y, Tiempo) a (Tiempo, X, Y) para realizar comparación
    tensor_py_transpuesto = np.transpose(tensor_py, (2, 0, 1))
    
    diferencia_absoluta = np.abs(tensor_py_transpuesto - tensor_c)
    max_diff = np.max(diferencia_absoluta)
    
    print(f"Tensor {key}:")
    print(f"  Diferencia máxima absoluta: {max_diff:.8e}")
    
    if np.allclose(tensor_py_transpuesto, tensor_c, atol=atol, rtol=0.0):
        print(f"  Los resultados superan la precisión: {atol}")
    else:
        print(f"  Los resultados difieren con la precisión: {atol}")
        indices_error = np.unravel_index(np.argmax(diferencia_absoluta), diferencia_absoluta.shape)
        print(f"     Mayor discrepancia en coordenada {indices_error}")
        print(f"     Valor Python modificado: {tensor_py_transpuesto[indices_error]}")
        print(f"     Valor C con OMP:      {tensor_c[indices_error]}")