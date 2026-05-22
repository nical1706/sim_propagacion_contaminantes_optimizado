#define CONTAMINATION_SEPARATED_H

#include "simulatorModule.h"

//###################################
//##    FUNCIONES DEL ALGORITMO    ##
//###################################

WindComponents* get_wind(SimulationData* sim_data, MapData* map_data);
float* get_difMatrix(SimulationData* sim_data, MapData* map_data);
Tensor* get_P_last(SimulationData* sim_data, MapData* map_data, Tensor* G, int* cartypes, char** pollutants);
Tensor* get_P_whole(SimulationData* sim_data, MapData* map_data, Tensor* G, int* cartypes, char** pollutants);