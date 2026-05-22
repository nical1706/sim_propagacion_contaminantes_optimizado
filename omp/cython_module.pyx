#Flags relacionadas con el tratamiento de python: compilado a python3, desactivación de comprobación de límites de array de
#python, desactivación de indices negativos de python, activación de división de C

# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True


import numpy as np
import time
cimport numpy as cnp
from libc.stdlib cimport malloc, free
from cpython.ref cimport Py_INCREF

import cython

#Inicializado de API de numpy
cnp.import_array()


###############################
##    STRUCTS Y FUNCIONES    ##
###############################


#Mapeo de structs y funciones de código C a cython
cdef extern from "contaminationSeparated.h":
    ctypedef struct SimulationData_c "SimulationData":
        int num_cartypes
        int num_pollutants
        float gamma
        float delta
        float corner_factor
        float WN
        float WE

    ctypedef struct MapData_c "MapData":
        int rows
        int cols
        int times
        float* acc

    ctypedef struct Tensor_c "Tensor":
        char* name
        float* data

    void free_T(Tensor_c* T, SimulationData_c* sim_data)
    Tensor_c* get_P_whole(SimulationData_c* sim_data, MapData_c* map_data, Tensor_c* G, int* cartypes, char** pollutants)
    Tensor_c* get_P_last(SimulationData_c* sim_data, MapData_c* map_data, Tensor_c* G, int* cartypes, char** pollutants)


#############################################
##    CLASES AUXILIARES Y DE CONVERSIÓN    ##
#############################################


#Clases puente para creación de structs
cdef class SimulationData:
    cdef SimulationData_c c_struct 

    def __init__(self, float delta, float corner_factor, float gamma, int num_cartypes, int num_pollutants, float WN, float WE):
        self.c_struct.num_cartypes = num_cartypes
        self.c_struct.num_pollutants = num_pollutants
        self.c_struct.gamma = gamma
        self.c_struct.delta = delta
        self.c_struct.corner_factor = corner_factor
        self.c_struct.WN = WN
        self.c_struct.WE = WE

cdef class MapData:
    cdef MapData_c c_struct
    cdef cnp.ndarray acc_ndarray

    def __init__(self, int rows, int cols, int times, cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] acc):
        self.c_struct.rows = rows
        self.c_struct.cols = cols
        self.c_struct.times = times
        #Se toma directamente el puntero de los datos del array numpy acc
        self.c_struct.acc = <float*> acc.data
        #Para evitar que python elimine automáticamente el dato acc, se asigna a un atributo auxiliar de la clase
        self.acc_ndarray = acc

#Dado que el resultado obtenido de la simulación se encuentra gestionado en memoria por C, es necesario indicar 
#a python cómo se debe liberar para no generar memory leaks

#Para ello, se ha creado la siguiente clase, cuya función será la de limpiar la memoria correspondiente a c_P y a sim_data de la manera
#especificada en __dealloc__, la cual se aplicará cuando el garbage collector de python decida eliminar la instancia correspondiente
#de TensorDeallocator
cdef class TensorDeallocator:
    cdef Tensor_c* c_P
    cdef SimulationData_c* sim_data

    def __cinit__(self):
        self.c_P = NULL
        self.sim_data = NULL

    def __dealloc__(self):
        if self.c_P != NULL and self.sim_data != NULL:
            free_T(self.c_P, self.sim_data)


###################################
##    FUNCIONES IMPLEMENTADAS    ##
###################################


def dict_to_contiguous_4d_tensor(dict G_dict, MapData map_data, list cartypes_list, list pollutants_list):
    cdef int num_cartypes = len(cartypes_list)
    cdef int num_pollutants = len(pollutants_list)
    cdef int num_tensors = num_cartypes * num_pollutants

    cdef int rows_l = map_data.c_struct.rows + 2
    cdef int cols_l = map_data.c_struct.cols + 2
    cdef int times_l = map_data.c_struct.times + 1
    
    #np.empty en lugar de np.zeros para no reducir tiempo de ejecución al no tener que inicializar valores a 0
    cdef cnp.ndarray[cnp.float32_t, ndim=4, mode="c"] G_out = np.empty((num_tensors, times_l, rows_l, cols_l), dtype=np.float32)
    
    #Sintaxis exclusiva de Cython, la cual crea un struct con elementos float de 4 dimensiones a la cual se le asigna el puntero de G_out
    #::1 para determinar que la dimensión asignada es continua en memoria
    cdef float[:, :, :, ::1] G_out_view = G_out
    cdef float[:, :, :] G_in_view
    
    cdef int k = 0
    cdef int t, i, j
    cdef str key
    
    for cartype in cartypes_list:
        for pollutant in pollutants_list:
            key = f"{cartype}_{pollutant}"
            
            #pop() extrae el valor y elimina la clave del diccionario
            G_in_view = G_dict.pop(key)
            
            for i in range(rows_l):
                for j in range(cols_l):
                    for t in range(times_l):
                        G_out_view[k, t, i, j] = G_in_view[i, j, t]
            
            k += 1
    
    return G_out




def get_P_last_p(SimulationData sim_data, MapData map_data, list cartypes_list, list pollutants_list, cnp.ndarray[cnp.float32_t, ndim=4, mode="c"] G):

    cdef int num_cartypes = len(cartypes_list)
    cdef int num_pollutants = len(pollutants_list)
    cdef int num_tensors = num_cartypes * num_pollutants


    #Conversión de listas python a arrays C
    cdef int* c_cartypes = <int*> malloc(num_cartypes * sizeof(int))
    cdef int i
    for i in range(num_cartypes):
        c_cartypes[i] = cartypes_list[i]

    #Para evitar que python elimine automáticamente los nombres (al momento de sobreescribirlos), se crea una lista auxiliar la cual va a aumentar automáticamente
    #el número de referencia, además de que se eliminará automáticamente al finalizar la función
    cdef list protected_names = []
    cdef char** c_pollutants = <char**> malloc(num_pollutants * sizeof(char*))
    for i in range(num_pollutants):
        name = pollutants_list[i].encode('utf-8')
        protected_names.append(name)
        c_pollutants[i] = name

    
    #Creación de familia de tensores G
    cdef Tensor_c* c_G = <Tensor_c*> malloc(num_tensors * sizeof(Tensor_c))
    cdef int k = 0
    cdef str key
    for cartype in cartypes_list:
        for pollutant in pollutants_list:
            key = f"{cartype}_{pollutant}"
            name = key.encode('utf-8')
            protected_names.append(name)
            
            c_G[k].name = name
            c_G[k].data = <float*> &G[k, 0, 0, 0]
            k += 1



    #cdef double t_inicio_c = time.perf_counter()

    #Función principal
    cdef Tensor_c* c_P = get_P_last(&sim_data.c_struct, &map_data.c_struct, c_G, c_cartypes, c_pollutants)

    #cdef double t_fin_c = time.perf_counter()



    #Inicializado de gestor de memoria
    cdef TensorDeallocator owner = TensorDeallocator()
    owner.c_P = c_P
    owner.sim_data = &sim_data.c_struct

    #Dimensiones de array np
    cdef cnp.npy_intp shape[2]
    shape[0] = map_data.c_struct.rows + 2
    shape[1] = map_data.c_struct.cols + 2

    cdef dict P_dict = {}
    cdef cnp.ndarray P_arr
    k = 0
    
    for cartype in cartypes_list:
        for pollutant in pollutants_list:
            #Creación de array np a partir de la dirección de c_P[k].data
            P_arr = cnp.PyArray_SimpleNewFromData(2, shape, cnp.NPY_FLOAT32, c_P[k].data)

            #Incremento del número de referencias de owner manual
            Py_INCREF(owner)

            #Establecimiento de las propiedades base del objeto (de destrucción en este caso)

            #Dado que lo que hace la función es guardar owner como un atributo de P_arr, no se incrementa el número de referencia del mismo, pudiendo hacer que se
            #elimine antes de tiempo si se empiezan a eliminar los arrays pese a pertenecer a otros arrays que sí lo usen, de ahí que se incremente manualmente
            #el número de referencia de owner
            cnp.PyArray_SetBaseObject(P_arr, owner)
            
            key = f"{cartype}_{pollutant}"

            #Comprobación nombres de tensores respecto a orden esperado
            if key != c_P[k].name.decode('utf-8'):
                raise ValueError(f"Inconsistencia encontrada de tensores en C respecto a orden establecido: "
                        f"se esperaba '{key}', pero se recibió '{c_P[k].name.decode('utf-8')}'")

            P_dict[key] = P_arr
            k += 1


    free(c_G)
    free(c_cartypes)
    free(c_pollutants)

    return P_dict




def get_P_whole_p(SimulationData sim_data, MapData map_data, list cartypes_list, list pollutants_list, cnp.ndarray[cnp.float32_t, ndim=4, mode="c"] G):

    cdef int num_cartypes = len(cartypes_list)
    cdef int num_pollutants = len(pollutants_list)
    cdef int num_tensors = num_cartypes * num_pollutants


    #Conversión de listas python a arrays C
    cdef int* c_cartypes = <int*> malloc(num_cartypes * sizeof(int))
    cdef int i
    for i in range(num_cartypes):
        c_cartypes[i] = cartypes_list[i]

    #Para evitar que python elimine automáticamente los nombres (al momento de sobreescribirlos), se crea una lista auxiliar la cual va a aumentar automáticamente
    #el número de referencia, además de que se eliminará automáticamente al finalizar la función
    cdef list protected_names = []
    cdef char** c_pollutants = <char**> malloc(num_pollutants * sizeof(char*))
    for i in range(num_pollutants):
        name = pollutants_list[i].encode('utf-8')
        protected_names.append(name)
        c_pollutants[i] = name

    
    #Creación de familia de tensores G
    cdef Tensor_c* c_G = <Tensor_c*> malloc(num_tensors * sizeof(Tensor_c))
    cdef int k = 0
    cdef str key
    for cartype in cartypes_list:
        for pollutant in pollutants_list:
            key = f"{cartype}_{pollutant}"
            name = key.encode('utf-8')
            protected_names.append(name)
            
            c_G[k].name = name
            c_G[k].data = <float*> &G[k, 0, 0, 0]
            k += 1



    #cdef double t_inicio_c = time.perf_counter()

    #Función principal
    cdef Tensor_c* c_P = get_P_whole(&sim_data.c_struct, &map_data.c_struct, c_G, c_cartypes, c_pollutants)

    #cdef double t_fin_c = time.perf_counter()
    #print(f"Tiempo de simulación en C: {t_fin_c - t_inicio_c:.5f} segundos")



    #Inicializado de gestor de memoria
    cdef TensorDeallocator owner = TensorDeallocator()
    owner.c_P = c_P
    owner.sim_data = &sim_data.c_struct

    #Dimensiones de array np
    cdef cnp.npy_intp shape[3]
    shape[0] = map_data.c_struct.times + 1
    shape[1] = map_data.c_struct.rows + 2
    shape[2] = map_data.c_struct.cols + 2

    cdef dict P_dict = {}
    cdef cnp.ndarray P_arr
    k = 0
    
    for cartype in cartypes_list:
        for pollutant in pollutants_list:
            #Creación de array np a partir de la dirección de c_P[k].data
            P_arr = cnp.PyArray_SimpleNewFromData(3, shape, cnp.NPY_FLOAT32, c_P[k].data)

            #Incremento del número de referencias de owner manual
            Py_INCREF(owner)

            #Establecimiento de las propiedades base del objeto (de destrucción en este caso)

            #Dado que lo que hace la función es guardar owner como un atributo de P_arr, no se incrementa el número de referencia del mismo, pudiendo hacer que se
            #elimine antes de tiempo si se empiezan a eliminar los arrays pese a pertenecer a otros arrays que sí lo usen, de ahí que se incremente manualmente
            #el número de referencia de owner
            cnp.PyArray_SetBaseObject(P_arr, owner)
            
            key = f"{cartype}_{pollutant}"

            #Comprobación nombres de tensores respecto a orden esperado
            if key != c_P[k].name.decode('utf-8'):
                raise ValueError(f"Inconsistencia encontrada de tensores en C respecto a orden establecido: "
                        f"se esperaba '{key}', pero se recibió '{c_P[k].name.decode('utf-8')}'")

            P_dict[key] = P_arr
            k += 1


    free(c_G)
    free(c_cartypes)
    free(c_pollutants)

    return P_dict