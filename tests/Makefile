#Declaración de targets
.PHONY: build_cores run_sims run_threads run_precision run_all help

#Comportamiento por defecto (por ser el primero en el archivo)
help:
	@echo "Ejecuta una de las siguiente opciones:"
	@echo "  make build_cores   -->   Elimina todos los archivos residuales y módulos, compila los módulos y los mueve a esta carpeta"
	@echo "  make run_sims      -->   Ejecuta test de comparación entre implementaciones existentes del simulador. Almacena resultados en archivo 'resultados_performance_PC.txt'"
	@echo "  make run_threads   -->   Ejecuta test de ejecución mediante hilos. Almacena resultados en archivo 'resultados_threads_PC.txt'"
	@echo "  make run_precision -->   Ejecuta test de precisión. Almacena resultados en archivo 'resultados_precision_PC.txt'"
	@echo "  make run_all       -->   Ejecuta todos los tests. Almacena resultados en sus correspondientes archivos"
	@echo "  make help          -->   Provee información de opciones de ejecución (se está usando ahora mismo)"

build_cores:
	@echo "Eliminando archivos residuales y módulos"
	@rm -f *.so
	@$(MAKE) -C ../no_omp clean
	@$(MAKE) -C ../omp clean
	@echo "Compilando módulos"
	@$(MAKE) -C ../no_omp build_c
	@$(MAKE) -C ../omp build_c
	@echo "Moviendo módulos"
	@mv ../no_omp/*.so .
	@mv ../omp/*.so .
	@echo "Módulos listos"

run_sims:
	@echo "Test de comparación entre implementaciones ejecutándose, datos almacenados en resultados_performance_PC.txt"
	@python performance_sim.py > resultados_performance_PC.txt
	@echo "Test finalizado con éxito"

run_threads:
	@echo "Test de comparación de hilos ejecutándose, datos almacenados en resultados_threads_PC.txt"
	@./performance_threads.sh > resultados_threads_PC.txt
	@echo "Test finalizado con éxito"

run_precision:
	@echo "Test de precisión ejecutándose, datos almacenados en resultados_precision_PC.txt"
	@python precision.py > resultados_precision_PC.txt
	@echo "Test finalizado con éxito"

run_all: run_sims run_threads run_precision
	@echo "Tests ejecutados con éxito"