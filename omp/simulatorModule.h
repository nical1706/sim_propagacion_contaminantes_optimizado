#define SIMULATOR_MODULE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


//#############################
//##    MACROS AUXILIARES    ##
//#############################


//Macros auxiliares de aplanamiento de índices para acceder a datos de tensores y matrices
//El objetivo es reducir los fallos de acceso a la memoria haiendo que los datos estén almacenados de forma contigua en memoria

//Macro para acceder a un elemento de una matriz 2D
//Se usa (size_t) para forzar 64 bits, evitando desbordamientos
#define IDX_2D(a, b, cols) ((size_t) a * cols + (size_t) b)

//Macro para acceder a un elemento de un tensor 3D
#define IDX_3D(a, b, c, dim2, dim3) ((size_t) a * dim2 * dim3 + (size_t) b * dim3 + (size_t) c)


//####################################
//##    ESTRUCTURAS DE ELEMENTOS    ##
//####################################


//Estructura para almacenar datos de simulación, como número de tipos de vehículos, número de contaminantes y parámetros físicos
typedef struct {
    int num_cartypes;
    int num_pollutants;
    float gamma;
    float delta;
    float corner_factor;
    float WN;
    float WE;
} SimulationData;

//Estructura que almacena el tamaño temporal y espacial de la simulación y la matriz de accesibilidad
typedef struct {
    int rows;
    int cols;
    int times;
    float* acc;
} MapData;

//Estructura para almacenar las 9 componentes del viento
typedef struct {
    float* N;
    float* S;
    float* E;
    float* W;
    float* NE;
    float* NW;
    float* SE;
    float* SW;
    float* stays;
} WindComponents;

//Estructura para representar un tensor genérico, la cual está compuesta por un identificador y la información correspondiente
typedef struct {
    char* name;
    float* data;
} Tensor;


//################################
//##    FUNCIONES AUXILIARES    ##
//################################


SimulationData* init_simulation_data(int num_cartypes, int num_pollutants, 
    float gamma, float delta, float corner_factor, float WN, float WE);
void free_simulation_data(SimulationData* sim_data);
MapData* init_map_data(int rows, int cols, float* acc);
void free_map_data(MapData* map_data);
void free_wind(WindComponents* wind);
Tensor* init_T_3D(SimulationData* sim_data, MapData* map_data, int* cartypes, char** pollutants);
Tensor* init_T_2D(SimulationData* sim_data, MapData* map_data, int* cartypes, char** pollutants);
void free_T(Tensor* T, SimulationData* sim_data);
float* init_matrix(int rows, int cols);