# SIMULADOR DE PROPAGACIÓN DE CONTAMINANTES ACELERADO POR C Y OPENMP

Con el objetivo de acelerar el simulador de propagación de contaminantes del proyecto SAINEVRA, se ha realizado en forma de Trabajo de Fin de Grado la siguiente implementación. Esta permite ejecutar una versión optimizada en C desde un entorno Python estándar por medio de librerías compartidas de Linux, las cuales se pueden (y deben) compilar específicamente para la arquitectura de la máquina a usar para optimizar el programa a nivel nativo.


## Información académica (Datos del TFG)

* **Título del TFG:** Aceleración de código de simulación de difusión de contaminantes mediante optimizaciones y paralelización en CPU.
* **Autor:** Nicolás García Niebla.
* **Director:** José Luis Guisado Lizar.
* **Departamento:** Arquitectura y Tecnología de Computadores.
* **Centro:** Escuela Técnica Superior de Ingeniería Informática (ETSII).
* **Universidad:** Universidad de Sevilla.
* **Grado:** Grado en Ingeniería Informática, Tecnología Informática.
* **Convocatoria / Fecha:** Junio de 2026.


## Estructura

El proyecto se segmenta en 3 archivos:

* **no_omp**: contiene el código referente a la implementación del simulador en C pero sin optimizaciones SIMD u OpenMP, en caso de querer ejecutar en mononúcleo.
- **omp**: contiene el código referente a la implementación del simulador en C on optimizaciones SIMD u OpenMP, para acelerar el sistema mediante hardware multinúcleo.
* **tests**: contiene los test presentados en la memoria como prueba de funcionamiento del código. Pueden servir de ejemplo de integración en un código Python.


## Dependencias de código

Además de las consecuentes dependencias de paquetes, el sistema debe ser un Linux (no importa la distribución) en una arquitectura x86.

Respecto a los paquetes, el sistema requiere de un compilador de C (recomendado GCC). En caso de no tener uno, ejecute el siguiente comando:

```bash
sudo apt update && sudo apt install build-essential
```

Además, dado que se usa OpenMP para la versión paralelizada, se requere de las librerías de uso de OpenMP, las cuales suelen venir por defecto en el sistema. En caso de no tenerlas, ejecute:

```bash
sudo apt install libomp-dev
```

Para la compilación del módulo, además de requerir Python 3.x, se requieren los siguientes paquetes del lenguaje:

```bash
pip install numpy cython setuptools
```
## Uso de simulador

### Ejecución de tests
Para ejecutar los tests en la carpeta del proyecto, se dispone de un Makefile que automatiza los procesos de compilación y ejecución. En caso de querer compilar los módulos para usarlos en los test, ejecute el comando:

```bash
make build_cores
```

Para ejecutar los tests, se tienen los comandos:

```bash
make run_<sustituir_por_test_a_ejecutar>
```

Si se tuvieran dudas sobre el uso del Makefile, se puede ejecutar el siguiente comando:

```bash
make help
```

### Compilación
En caso de querer compilar un módulo en específico, se ha de entrar a una de las carpetas con el código del simulador ("no_omp" o "omp") y ejecutar el siguiente comando:

```bash
make build_c
```

Una vez hecho eso, se generará un archivo .so correspondiente al módulo del simulador, el cual deberá ser movido a la carpeta donde se quiera usar.

### Inicializado en Python
Para usar desde Python, primero se ha de importar el módulo del simulador ya compilado (el posible que el editor de código avise de que no se encuentra el módulo, pero es normal). En caso de importa el módulo sin OpenMP:

```python
import simulator_core_no_omp
```

Para el módulo con OpenMP:

```python
import simulator_core_omp
```

Una vez hecho esto, se han de inicializar las estructuras de datos usadas en C. Para ellos, tras inicializar los valores correspondientes al simulador, se deben inicializar los siguientes 2 objetos (recordar usar el nombre del módulo que se esté usando):

```python
sim_data = simulator_core.SimulationData(
    delta=delta, 
    corner_factor=corner_factor, 
    gamma=gamma, 
    num_cartypes=num_car_types, 
    num_pollutants=num_pollutants, 
    WN=WN, 
    WE=WE
)

map_data = simulator_core.MapData(
    rows=width, 
    cols=height, 
    times=times, 
    acc=acc
)
```

Tras esto, si los datos de entrada G siguen el formato definido por el código del simulador original (es decir, tensores con dimensiones (width+2, height+2, times+1)), se debe transformar al formato aceptado por la implementación usando:

```python
G_numpy_para_C = simulator_core_omp.dict_to_contiguous_4d_tensor(G, map_data, car_types, pollutants)
```

Finalmente, para la ejecución del simulador, se tienen 2 funciones. La primera que almacena el estado final de la simulación (útil para reducir el uso de memoria):

```python
P = simulator_core.get_P_last_p(sim_data, map_data, car_types, pollutants, G_numpy_para_C)
```

La segunda  retorna el histórico de toda la simulación:

```python
P = simulator_core.get_P_whole_p(sim_data, map_data, car_types, pollutants, G_numpy_para_C)
```