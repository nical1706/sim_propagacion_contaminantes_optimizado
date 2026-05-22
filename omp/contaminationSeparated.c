#include "contaminationSeparated.h"
#include <math.h>
#include <omp.h>
#include <x86intrin.h>

//Función que obtiene las componentes del viento para cada celda
//Se retorna un puntero por eficiencia, ya que en caso contrario se precisaría retornar la estructura completa
WindComponents* get_wind(SimulationData* sim_data, MapData* map_data) {
    int rows = map_data -> rows;
    int cols = map_data -> cols;
    //Uso de restrict para indicar que no hay aliasing con otras variables, lo que permite optimizaciones por parte del compilador
    float* restrict acc = map_data -> acc;

    float WN = sim_data -> WN;
    float WE = sim_data -> WE;

    WindComponents* wind = (WindComponents*) malloc(sizeof(WindComponents));

    //Variables auxiliares para acelerar cálculos y legibilidad del código
    int rows_l = rows + 2;
    int cols_l = cols + 2;

    wind -> N = init_matrix(rows_l, cols_l);
    wind -> S  = init_matrix(rows_l, cols_l);
    wind -> E  = init_matrix(rows_l, cols_l);
    wind -> W  = init_matrix(rows_l, cols_l);
    wind -> NE = init_matrix(rows_l, cols_l);
    wind -> NW = init_matrix(rows_l, cols_l);
    wind -> SE = init_matrix(rows_l, cols_l);
    wind -> SW = init_matrix(rows_l, cols_l);
    wind -> stays = init_matrix(rows_l, cols_l);

    //Variables locales para evitar aliasing
    float* restrict w_N = wind -> N;
    float* restrict w_S = wind -> S;
    float* restrict w_E = wind -> E;
    float* restrict w_W = wind -> W;
    float* restrict w_NE = wind -> NE;
    float* restrict w_NW = wind -> NW;
    float* restrict w_SE = wind -> SE;
    float* restrict w_SW = wind -> SW;
    float* restrict w_stays = wind -> stays;

    //Signos de los componentes del viento
    int sign_WN = (WN > 0) ? 1 : ((WN < 0) ? -1 : 0);
    int sign_WE = (WE > 0) ? 1 : ((WE < 0) ? -1 : 0);

    float max_WN = (WN > 0.0f) ? WN : 0.0f;
    float max_m_WN = (WN < 0.0f) ? -WN : 0.0f;
    float max_WE = (WE > 0.0f) ? WE : 0.0f;
    float max_m_WE = (WE < 0.0f) ? -WE : 0.0f;

    //Cálculo de desplazamientos para cada dirección
    int jmp_WE = sign_WE * cols_l;
    //Este se deja por claridad de código
    int jmp_WN = sign_WN;

    #pragma omp parallel for simd collapse(2) schedule(static)
    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= cols; j++) {
            size_t id = IDX_2D(i, j, cols_l);
            float acc_ij = acc[id];

            //Solo calcula desplazamientos para celdas con acc > 0
            if (acc_ij != 0) {
                //Se calculan los índices aparte en vez de usar la macro, dado que así se reducen el 
                //número de multiplicaciones realizadas en el bucle al momento de obtener los índices
                float acc1 = acc[id + jmp_WE];
                float acc2 = acc[id - jmp_WN];

                float max_N = (acc[id + jmp_WE - 1] > acc1) ? acc[id + jmp_WE - 1] : acc1;
                float max_S = (acc[id + jmp_WE + 1] > acc1) ? acc[id + jmp_WE + 1] : acc1;
                float max_E = (acc[id + cols_l - jmp_WN] > acc2) ? acc[id + cols_l - jmp_WN] : acc2;
                float max_W = (acc[id - cols_l - jmp_WN] > acc2) ? acc[id - cols_l - jmp_WN] : acc2;

                w_N[id] = acc_ij * max_WN * (1.0f - max_N * fabsf(WE)) * acc[id - 1];
                w_S[id] = acc_ij * max_m_WN * (1.0f - max_S * fabsf(WE)) * acc[id + 1];
                w_E[id] = acc_ij * max_WE * (1.0f - max_E * fabsf(WN)) * acc[id + cols_l];
                w_W[id] = acc_ij * max_m_WE * (1.0f - max_W * fabsf(WN)) * acc[id - cols_l];
                
                w_NE[id] = acc_ij * max_WN * max_WE * acc[id + cols_l - 1];
                w_NW[id] = acc_ij * max_WN * max_m_WE * acc[id - cols_l - 1];
                w_SE[id] = acc_ij * max_m_WN * max_WE * acc[id + cols_l + 1];
                w_SW[id] = acc_ij * max_m_WN * max_m_WE * acc[id - cols_l + 1];

                w_stays[id] = 1.0f - (w_N[id] + w_S[id] + w_E[id] + w_W[id] + 
                    w_NE[id] + w_NW[id] + w_SE[id] + w_SW[id]);
            } else {
                w_stays[id] = 1.0f;
            }
        }
    }

    return wind;
}

//Retorna matriz de difusión
float* get_difMatrix(SimulationData* sim_data, MapData* map_data) {
    int rows = map_data -> rows;
    int cols = map_data -> cols;
    float* restrict acc = map_data -> acc;

    float delta = sim_data -> delta;
    float corner_factor = sim_data -> corner_factor;

    //Variables auxiliares para acelerar cálculos y legibilidad del código
    int rows_l = rows + 2;
    int cols_l = cols + 2;

    float* restrict dif_matrix = init_matrix(rows_l, cols_l);

    //Para reducir las divisiones y multiplicaciones realizadas, se calcula fuera del bucle la parte fija de la función y se añade
    //a posteriori al cálculo de la difusión
    float f_factor = delta / (4.0f + 4.0f * corner_factor);

    #pragma omp parallel for simd collapse(2) schedule(static)
    for (int i = 1; i <= rows; i++) {
        for (int j = 1; j <= cols; j++) {
            size_t id = IDX_2D(i, j, cols_l);
            //Se calculan los índices aparte en vez de usar la macro, dado que así se reducen el 
            //número de multiplicaciones realizadas en el bucle al momento de obtener los índices

            //Conteo vecinos cruz accesibles
            float acc_neig_edge = acc[id - cols_l] + acc[id + cols_l] + acc[id + 1] + acc[id - 1];
            //Conteo vecinos diagonales accesibles
            float acc_neig_corner = acc[id - cols_l - 1] + acc[id - cols_l + 1] + acc[id + cols_l + 1] + acc[id + cols_l - 1];

            dif_matrix[id] = 1.0f - (acc_neig_edge + acc_neig_corner * corner_factor) * f_factor;
        }
    }

    return dif_matrix;
}

//Retorna familia de tensores 2D que representan el último instante de la simulación
//Recibe las estruturas de datos auxiliares, la matriz de acceso, la familia de tensores de datos G, 
//la lista de coches y la lista de contaminantes
Tensor* get_P_last(SimulationData* sim_data, MapData* map_data, Tensor* G, int* cartypes, char** pollutants) {
    int rows = map_data -> rows;
    int cols = map_data -> cols;
    int times = map_data -> times;
    float* restrict acc = map_data -> acc;

    float delta = sim_data -> delta;
    float corner_factor = sim_data -> corner_factor;
    float gamma = sim_data -> gamma;
    int num_cartypes = sim_data -> num_cartypes;
    int num_pollutants = sim_data -> num_pollutants;

    int num_tensors = num_cartypes * num_pollutants;

    //Variables auxiliares para acelerar cálculos y legibilidad del código
    int rows_l = rows + 2;
    int cols_l = cols + 2;
    
    Tensor* P = init_T_2D(sim_data, map_data, cartypes, pollutants);

    WindComponents* wind = get_wind(sim_data, map_data);
    float* restrict dif_matrix = get_difMatrix(sim_data, map_data);

    //Puesta en variables de datos de viento para reducción de tiempos de acceso a memoria
    float* restrict N = wind -> N;
    float* restrict S = wind -> S;
    float* restrict E = wind -> E;
    float* restrict W = wind -> W;
    float* restrict NE = wind -> NE;
    float* restrict NW = wind -> NW;
    float* restrict SE = wind -> SE;
    float* restrict SW = wind -> SW;
    float* restrict stays = wind -> stays;

    //Para reducir las divisiones y multiplicaciones realizadas, se calcula fuera del bucle la parte fija de la función y se añade
    //a posteriori al cálculo de la difusión
    float f_factor = delta / (4.0f + 4.0f * corner_factor);

    float m_gamma = 1.0f - gamma;

    //Matriz intermedia que almacena estado de P tras bucle difusión
    float* restrict P_diff = init_matrix(rows_l, cols_l);
    
    #pragma omp parallel
    {
        for (int t = 0; t <= times; t++) {
            for (int k = 0; k < num_tensors; k++) {
                //Matriz auxiliar para reducir accesos a memoria
                //También sirve para evitar Pointer Aliasing al momento de guardar el resultado
                float* restrict P_slice = P[k].data;

                //Sección de tiempo de G, para ahorrar accesos a memoria
                float* restrict G_slice = &G[k].data[(size_t)t * rows_l * cols_l];

                #pragma omp for simd collapse(2) schedule(static)
                //Bucle difusión
                for (int i = 1; i <= rows; i++) {
                    for (int j = 1; j <= cols; j++) {
                        size_t id = IDX_2D(i, j, cols_l);

                        float diffusion_edge = P_slice[id - cols_l] + P_slice[id + cols_l] + P_slice[id - 1] + P_slice[id + 1];
                        float diffusion_corner = P_slice[id - cols_l - 1] + P_slice[id - cols_l + 1] + P_slice[id + cols_l + 1] + P_slice[id + cols_l - 1];

                        P_diff[id] = acc[id] * (dif_matrix[id] * P_slice[id] + (diffusion_edge + diffusion_corner * corner_factor) * f_factor);
                    }
                }

                #pragma omp for simd collapse(2) schedule(static)
                //Bucle viento
                for (int i = 1; i <= rows; i++) {
                    for (int j = 1; j <= cols; j++) {
                        size_t id = IDX_2D(i, j, cols_l);

                        float transport = N[id + 1] * P_diff[id + 1] + 
                            S[id - 1] * P_diff[id - 1] + 
                            E[id - cols_l] * P_diff[id - cols_l] + 
                            W[id + cols_l] * P_diff[id + cols_l] + 
                            NE[id - cols_l + 1] * P_diff[id - cols_l + 1] + 
                            NW[id + cols_l + 1] * P_diff[id + cols_l + 1] + 
                            SE[id - cols_l - 1] * P_diff[id - cols_l - 1] + 
                            SW[id + cols_l - 1] * P_diff[id + cols_l - 1] + 
                            stays[id] * P_diff[id];

                        P_slice[id] = (m_gamma) * acc[id] * (transport + G_slice[id]);
                    }
                }
            }
        }
    }

    free_wind(wind);
    free(dif_matrix);
    free(P_diff);

    return P;
}

//Retorna la progresión del tensor P durante un tiempo establecido
Tensor* get_P_whole(SimulationData* sim_data, MapData* map_data, Tensor* G, int* cartypes, char** pollutants) {
    int rows = map_data -> rows;
    int cols = map_data -> cols;
    int times = map_data -> times;
    float* restrict acc = map_data -> acc;

    float delta = sim_data -> delta;
    float corner_factor = sim_data -> corner_factor;
    float gamma = sim_data -> gamma;
    int num_cartypes = sim_data -> num_cartypes;
    int num_pollutants = sim_data -> num_pollutants;

    int num_tensors = num_cartypes * num_pollutants;

    //Variables auxiliares para acelerar cálculos y legibilidad del código
    int rows_l = rows + 2;
    int cols_l = cols + 2;
    
    Tensor* P = init_T_3D(sim_data, map_data, cartypes, pollutants);

    WindComponents* wind = get_wind(sim_data, map_data);
    float* restrict dif_matrix = get_difMatrix(sim_data, map_data);

    //Puesta en variables de datos de viento para reducción de tiempos de acceso a memoria
    float* restrict N = wind -> N;
    float* restrict S = wind -> S;
    float* restrict E = wind -> E;
    float* restrict W = wind -> W;
    float* restrict NE = wind -> NE;
    float* restrict NW = wind -> NW;
    float* restrict SE = wind -> SE;
    float* restrict SW = wind -> SW;
    float* restrict stays = wind -> stays;

    //Para reducir las divisiones y multiplicaciones realizadas, se calcula fuera del bucle la parte fija de la función y se añade
    //a posteriori al cálculo de la difusión
    float f_factor = delta / (4.0f + 4.0f * corner_factor);

    float m_gamma = 1.0f - gamma;

    //Matriz intermedia que almacena estado de P tras bucle difusión
    float* restrict P_diff = init_matrix(rows_l, cols_l);

    #pragma omp parallel
    {
        for (int t = 0; t < times; t++) {
            for (int k = 0; k < num_tensors; k++) {
                //Matriz auxiliar para reducir accesos a memoria
                float* restrict P_slice = &P[k].data[(size_t)t * rows_l * cols_l];
                
                //Matriz auxiliar para reducir accesos a memoria
                //Corresponde al siguiente instante de tiempo, guardado en el mismo tensor para mejorar la localidad temporal y evitar accesos a memoria a estructuras distintas
                float* restrict P_next = &P[k].data[(size_t)(t + 1) * rows_l * cols_l];

                //Sección de tiempo de G, para ahorrar accesos a memoria
                float* restrict G_slice = &G[k].data[(size_t)t * rows_l * cols_l];

                #pragma omp for simd collapse(2) schedule(static)
                //Bucle difusión
                for (int i = 1; i <= rows; i++) {
                    for (int j = 1; j <= cols; j++) {
                        size_t id = IDX_2D(i, j, cols_l);

                        float diffusion_edge = P_slice[id - cols_l] + P_slice[id + cols_l] + P_slice[id - 1] + P_slice[id + 1];
                        float diffusion_corner = P_slice[id - cols_l - 1] + P_slice[id - cols_l + 1] + P_slice[id + cols_l + 1] + P_slice[id + cols_l - 1];

                        P_diff[id] = acc[id] * (dif_matrix[id] * P_slice[id] + (diffusion_edge + diffusion_corner * corner_factor) * f_factor);
                    }
                }

                #pragma omp for simd collapse(2) schedule(static)
                //Bucle viento
                for (int i = 1; i <= rows; i++) {
                    for (int j = 1; j <= cols; j++) {
                        size_t id = IDX_2D(i, j, cols_l);

                        float transport = N[id + 1] * P_diff[id + 1] + 
                            S[id - 1] * P_diff[id - 1] + 
                            E[id - cols_l] * P_diff[id - cols_l] + 
                            W[id + cols_l] * P_diff[id + cols_l] + 
                            NE[id - cols_l + 1] * P_diff[id - cols_l + 1] + 
                            NW[id + cols_l + 1] * P_diff[id + cols_l + 1] + 
                            SE[id - cols_l - 1] * P_diff[id - cols_l - 1] + 
                            SW[id + cols_l - 1] * P_diff[id + cols_l - 1] + 
                            stays[id] * P_diff[id];

                        //Actualizado estado t+1
                        P_next[id] = (m_gamma) * acc[id] * (transport + G_slice[id]);

                        //Actualizado estado t
                        P_slice[id] = P_diff[id];
                    }
                }
            }
        }


        //Para evadir el uso de condicionales en el código que impotan la implementación de SIMD se ha añadido la iteración t=times para que
        //corresponda con el simulador original, modificando únicamente el estado t con un bucle de difusión
        for (int k = 0; k < num_tensors; k++) {
            //Matriz auxiliar para reducir accesos a memoria
            float* restrict P_slice = &P[k].data[(size_t)times * rows_l * cols_l];

            #pragma omp for simd collapse(2) schedule(static)
            //Bucle difusión
            for (int i = 1; i <= rows; i++) {
                for (int j = 1; j <= cols; j++) {
                    size_t id = IDX_2D(i, j, cols_l);

                    float diffusion_edge = P_slice[id - cols_l] + P_slice[id + cols_l] + P_slice[id - 1] + P_slice[id + 1];
                    float diffusion_corner = P_slice[id - cols_l - 1] + P_slice[id - cols_l + 1] + P_slice[id + cols_l + 1] + P_slice[id + cols_l - 1];

                    P_diff[id] = acc[id] * (dif_matrix[id] * P_slice[id] + (diffusion_edge + diffusion_corner * corner_factor) * f_factor);
                }
            }

            //Actualizado estado t
            #pragma omp for simd collapse(2) schedule(static)
            for (int i = 1; i <= rows; i++) {
                for (int j = 1; j <= cols; j++) {
                    size_t id = IDX_2D(i, j, cols_l);
                    P_slice[id] = P_diff[id]; 
                }
            }
        }
    }

    free_wind(wind);
    free(dif_matrix);
    free(P_diff);

    return P;
}