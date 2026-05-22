import numpy as np

import simulator_core_no_omp
import simulator_core_omp

################################
##    FUNCIONES AUXILIARES    ##
################################

def init_data_python(width, height, times, car_types, pollutants):
    #acc contiguo en memoria (con orden C) para acelerar tiempo transmisión
    acc = np.zeros((width+2, height+2), dtype=np.float32, order='C')

    #Plantilla para aplicación de patrones de edificios
    template = np.zeros(100, dtype=np.float32)

    template[0:4] = 1
    template[96:100] = 1
    template[48:52] = 1

    template[25:27] = 1
    template[73:75] = 1

    #Aplicación del patrón
    #np.ceil asegura copias del template, slicing [:width] corta al tamaño exacto
    pattern_w = np.tile(template, int(np.ceil(width / 100)))[:width]
    pattern_h = np.tile(template, int(np.ceil(height / 100)))[:height]
    row_mask = np.zeros(width + 2, dtype=bool)
    row_mask[1:-1] = (pattern_w == 1)
    col_mask = np.zeros(height + 2, dtype=bool)
    col_mask[1:-1] = (pattern_h == 1)

    acc[row_mask == 1, :] = 1
    acc[:, col_mask == 1] = 1


    #Inicializado tensor de emisiones de entrada G
    G = {f"{cartype}_{pollutant}": np.zeros((width+2, height+2, times+1), dtype=np.float32)
                    for cartype in car_types
                    for pollutant in pollutants}

    #Umbral estadístico para generación de datos aleatorios
    u = 0.1 
    for key in G:
        #Peso (arbitrario) de emisiones según el tipo de origen
        if 'electric' in key:
            emission = 0.0
        elif 'diesel' in key:
            emission = 5000.0
        else:  # gasoline
            emission = 3000.0
        if emission > 0:
            for t in range(times + 1):
                rand = np.random.rand(width+2, height+2)
                #Filtrado de ruido según umbral y acc
                f_rand = (rand < u) * acc
                pollution = f_rand * emission
                
                G[key][:, :, t] = pollution
    return (acc, G)

def init_data_no_omp(width, height, times, car_types, pollutants):
    #acc contiguo en memoria (con orden C) para acelerar tiempo transmisión
    acc = np.zeros((width+2, height+2), dtype=np.float32, order='C')

    #Plantilla para aplicación de patrones de edificios
    template = np.zeros(100, dtype=np.float32)

    template[0:4] = 1
    template[96:100] = 1
    template[48:52] = 1

    template[25:27] = 1
    template[73:75] = 1

    #Aplicación del patrón
    #np.ceil asegura copias del template, slicing [:width] corta al tamaño exacto
    pattern_w = np.tile(template, int(np.ceil(width / 100)))[:width]
    pattern_h = np.tile(template, int(np.ceil(height / 100)))[:height]
    row_mask = np.zeros(width + 2, dtype=bool)
    row_mask[1:-1] = (pattern_w == 1)
    col_mask = np.zeros(height + 2, dtype=bool)
    col_mask[1:-1] = (pattern_h == 1)

    acc[row_mask == 1, :] = 1
    acc[:, col_mask == 1] = 1


    map_data = simulator_core_no_omp.MapData(
        rows=width, 
        cols=height, 
        times=times, 
        acc=acc
    )


    #Inicializado tensor de emisiones de entrada G
    G = {f"{cartype}_{pollutant}": np.zeros((width+2, height+2, times+1), dtype=np.float32)
                    for cartype in car_types
                    for pollutant in pollutants}

    #Umbral estadístico para generación de datos aleatorios
    u = 0.1 
    for key in G:
        #Peso (arbitrario) de emisiones según el tipo de origen
        if 'electric' in key:
            emission = 0.0
        elif 'diesel' in key:
            emission = 5000.0
        else:  # gasoline
            emission = 3000.0
        if emission > 0:
            for t in range(times + 1):
                rand = np.random.rand(width+2, height+2)
                #Filtrado de ruido según umbral y acc
                f_rand = (rand < u) * acc
                pollution = f_rand * emission
                
                G[key][:, :, t] = pollution


    G_numpy_para_C = simulator_core_no_omp.dict_to_contiguous_4d_tensor(G, map_data, car_types, pollutants)
    return (acc, map_data, G, G_numpy_para_C)

def init_data_omp(width, height, times, car_types, pollutants):
    #acc contiguo en memoria (con orden C) para acelerar tiempo transmisión
    acc = np.zeros((width+2, height+2), dtype=np.float32, order='C')

    #Plantilla para aplicación de patrones de edificios
    template = np.zeros(100, dtype=np.float32)

    template[0:4] = 1
    template[96:100] = 1
    template[48:52] = 1

    template[25:27] = 1
    template[73:75] = 1

    #Aplicación del patrón
    #np.ceil asegura copias del template, slicing [:width] corta al tamaño exacto
    pattern_w = np.tile(template, int(np.ceil(width / 100)))[:width]
    pattern_h = np.tile(template, int(np.ceil(height / 100)))[:height]
    row_mask = np.zeros(width + 2, dtype=bool)
    row_mask[1:-1] = (pattern_w == 1)
    col_mask = np.zeros(height + 2, dtype=bool)
    col_mask[1:-1] = (pattern_h == 1)

    acc[row_mask == 1, :] = 1
    acc[:, col_mask == 1] = 1


    map_data = simulator_core_omp.MapData(
        rows=width, 
        cols=height, 
        times=times, 
        acc=acc
    )


    #Inicializado tensor de emisiones de entrada G
    G = {f"{cartype}_{pollutant}": np.zeros((width+2, height+2, times+1), dtype=np.float32)
                    for cartype in car_types
                    for pollutant in pollutants}

    #Umbral estadístico para generación de datos aleatorios
    u = 0.1 
    for key in G:
        #Peso (arbitrario) de emisiones según el tipo de origen
        if 'electric' in key:
            emission = 0.0
        elif 'diesel' in key:
            emission = 5000.0
        else:  # gasoline
            emission = 3000.0
        if emission > 0:
            for t in range(times + 1):
                rand = np.random.rand(width+2, height+2)
                #Filtrado de ruido según umbral y acc
                f_rand = (rand < u) * acc
                pollution = f_rand * emission
                
                G[key][:, :, t] = pollution


    G_numpy_para_C = simulator_core_omp.dict_to_contiguous_4d_tensor(G, map_data, car_types, pollutants)
    return (acc, map_data, G, G_numpy_para_C)
