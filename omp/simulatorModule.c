#include "simulatorModule.h"


//################################
//##    FUNCIONES AUXILIARES    ##
//################################


//Función que inicializa estructura de datos de simulación
SimulationData* init_simulation_data(int num_cartypes, int num_pollutants, 
    float gamma, float delta, float corner_factor, float WN, float WE) {
    SimulationData* sim_data = (SimulationData*) malloc(sizeof(SimulationData));

    sim_data -> num_cartypes = num_cartypes;
    sim_data -> num_pollutants = num_pollutants;
    sim_data -> gamma = gamma;
    sim_data -> delta = delta;
    sim_data -> corner_factor = corner_factor;
    sim_data -> WN = WN;
    sim_data -> WE = WE;

    return sim_data;
}

//Función que libera estructura de datos de simulación
void free_simulation_data(SimulationData* sim_data) {
    free(sim_data);
}

//Función que inicializa estructura de datos de mapa
MapData* init_map_data(int rows, int cols, float* acc) {
    MapData* map_data = (MapData*) malloc(sizeof(MapData));

    map_data -> rows = rows;
    map_data -> cols = cols;
    map_data -> acc = acc;

    return map_data;
}

//Función que libera estructura de datos de mapa
void free_map_data(MapData* map_data) {
    free(map_data);
}

//Función que libera estructura de componentes del viento
void free_wind(WindComponents* wind) {
    free(wind -> N);
    free(wind -> S);
    free(wind -> E);
    free(wind -> W);
    free(wind -> NE);
    free(wind -> NW);
    free(wind -> SE);
    free(wind -> SW);
    free(wind -> stays);
    free(wind);
}

//Función que inicializa familia de tensores 3D 
Tensor* init_T_3D(SimulationData* sim_data, MapData* map_data, int* cartypes, char** pollutants) {

    int num_cartypes = sim_data->num_cartypes;
    int num_pollutants = sim_data->num_pollutants;
    int rows = map_data -> rows + 2;
    int cols = map_data -> cols + 2;
    int times = map_data -> times + 1;

    int index = 0;
    Tensor* T = (Tensor*) calloc((num_cartypes * num_pollutants), sizeof(Tensor));
    for (int ct = 0; ct < num_cartypes; ct++) {
        for (int p = 0; p < num_pollutants; p++) {
            //Como cartypes es un array de enteros, se convierte a string para formar el nombre del tensor
            char car_char[30];
            snprintf(car_char, sizeof(car_char), "%d", cartypes[ct]);

            int name_size = strlen(car_char) + strlen(pollutants[p]) + 2;
            T[index].name = malloc(name_size * sizeof(char));
            //Unión de strings para formar el nombre del tensor
            snprintf(T[index].name, name_size, "%s_%s", car_char, pollutants[p]);

            T[index].data = (float*) calloc((size_t) rows * cols * times, sizeof(float));
            index++;
        }
    }
    return T;
}

//Función que inicializa familia de tensores 2D
Tensor* init_T_2D(SimulationData* sim_data, MapData* map_data, int* cartypes, char** pollutants) {
    int num_cartypes = sim_data -> num_cartypes;
    int num_pollutants = sim_data -> num_pollutants;
    int rows = map_data -> rows + 2;
    int cols = map_data -> cols + 2;

    int index = 0;
    Tensor* T = (Tensor*) calloc((num_cartypes * num_pollutants), sizeof(Tensor));
    for (int ct = 0; ct < num_cartypes; ct++) {
        for (int p = 0; p < num_pollutants; p++) {
            //Como cartypes es un array de enteros, se convierte a string para formar el nombre del tensor
            char car_char[30];
            snprintf(car_char, sizeof(car_char), "%d", cartypes[ct]);

            int name_size = strlen(car_char) + strlen(pollutants[p]) + 2;
            T[index].name = malloc(name_size * sizeof(char));
            //Unión de strings para formar el nombre del tensor
            snprintf(T[index].name, name_size, "%s_%s", car_char, pollutants[p]);

            T[index].data = (float*) calloc((size_t) rows * cols, sizeof(float));
            index++;
        }
    }
    return T;
}

//Función que libera familia de tensores de cualquier tipo
void free_T(Tensor* T, SimulationData* sim_data) {
    int num_cartypes = sim_data->num_cartypes;
    int num_pollutants = sim_data->num_pollutants;
    for (int i = 0; i < num_cartypes * num_pollutants; i++) {
        free(T[i].name);
        free(T[i].data); 
    }
    free(T);
}

//Función que inicializa matriz
float* init_matrix(int rows, int cols) {
    float* matrix = (float*) calloc((size_t) rows * cols, sizeof(float));
    return matrix;
}