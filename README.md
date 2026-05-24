# SIMULADOR DE PROPAGACIÓN DE CONTAMINANTES ACELERADO POR C Y OPENMP

Con el objetivo de acelerar el simulador de propagación de contaminantes del proyecto SAINEVRA, se ha realizado en forma de Trabajo de Fin de Grado la siguiente implementación. Esta permite ejecutar una versión optimizada en C desde un entorno Python estándar por medio de librerías compartidas de Linux, las cuales se pueden (y deben) compilar específicamente para la arquitectura de la máquina a usar para optimizar el programa a nivel nativo. Se adjunta la memoria para más detalles del proyecto como el archivo "TFG.pdf".


## Información académica (Datos del TFG)

* **Título del TFG:** Aceleración de código de simulación de difusión de contaminantes mediante optimizaciones y paralelización en CPU.
* **Autor:** Nicolás García Niebla.
* **Director:** José Luis Guisado Lizar.
* **Departamento:** Arquitectura y Tecnología de Computadores.
* **Centro:** Escuela Técnica Superior de Ingeniería Informática (ETSII).
* **Universidad:** Universidad de Sevilla.
* **Grado:** Grado en Ingeniería Informática, Tecnología Informática.
* **Convocatoria / Fecha:** Junio de 2026.
* **Enlace a proyecto SAINEVRA:** https://grupo.us.es/sainevra/index.html


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
Para usar desde Python, primero se ha de importar el módulo del simulador ya compilado (es posible que el editor de código avise de que no se encuentra el módulo, pero es normal). En caso de importar el módulo sin OpenMP:

```python
import simulator_core_no_omp
```

Para el módulo con OpenMP:

```python
import simulator_core_omp
```

A cuntinuación, se deben de inicializar los parámetros y las estructuras de entradas del simulador, para ello, se declaran las siguientes variables en Python para los parámetros numéricos (valores de ejemplo, el usuario puede modificarlos a su gusto):

```python
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
```

En cuanto a las estructuras, primero se tiene la estructura de datos "acc", la cual almacena qué casillas tienen edificios (con valor 0) y viceversa (con valor 1):

```python
#Inicializado acc
acc = np.zeros((width+2, height+2), dtype=np.float32, order='C')

#Datos de ejemplo, modificar con los que se desee
template = np.zeros(100, dtype=np.float32)
template[0:4] = 1
template[96:100] = 1
template[48:52] = 1
template[25:27] = 1
template[73:75] = 1
pattern_w = np.tile(template, int(np.ceil(width / 100)))[:width]
pattern_h = np.tile(template, int(np.ceil(height / 100)))[:height]
row_mask = np.zeros(width + 2, dtype=bool)
row_mask[1:-1] = (pattern_w == 1)
col_mask = np.zeros(height + 2, dtype=bool)
col_mask[1:-1] = (pattern_h == 1)
acc[row_mask == 1, :] = 1
acc[:, col_mask == 1] = 1
```

De segundo se tiene el diccionario de tensores "G", que almacena los puntos iniciales de contaminación de la simulación:

```python
#Inicializado G
G = {f"{cartype}_{pollutant}": np.zeros((width+2, height+2, times+1), dtype=np.float32)
				for cartype in car_types
				for pollutant in pollutants}

#Datos de ejemplo, modificar con los que se desee
for key in G:
	G[key][:int(width/1.2),:int(height/1.2),:int(times/1.2)]=100
```

Una vez hecho esto, se han de inicializar las estructuras de datos en C. Para ello, tras inicializar los valores correspondientes al simulador, se deben inicializar los siguientes 2 objetos (se recuerda usar el nombre del módulo que se haya importado):

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

## Licencias 
Este repositorio contiene código fuente, datos de prueba asociados al código y documentación académica derivados de un Trabajo Fin de Grado realizado en la Universidad de Sevilla en el marco del proyecto de investigación SAINEVRA. 

Salvo indicación expresa en contrario: 
- El código fuente, scripts, y tests asociados al código se distribuyen bajo la **Apache License 2.0**. Véase el archivo `LICENSE` y `LICENSES/Apache-2.0.txt`. 
- La memoria del TFG en PDF se distribuyen bajo la licencia **Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)**. Véase `LICENSES/CC-BY-NC-ND-4.0.txt`. 

La reutilización del código debe citar adecuadamente este repositorio, el TFG y el proyecto de investigación asociado. Véase `CITATION.cff`.