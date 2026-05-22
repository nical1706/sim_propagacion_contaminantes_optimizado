#Declaración de targets
.PHONY: build build_c clean help

#Comportamiento por defecto (por ser el primero en el archivo)
help:
	@echo "Ejecuta una de las siguiente opciones:"
	@echo "  make build    -->   Crea el archivo binario del simulador"
	@echo "  make build_c  -->   Crea el archivo binario del simulador y elimina archivos temporales"
	@echo "  make clean    -->   Elimina todo archivo referente al binario y este mismo"
	@echo "  make help     -->   Provee información de opciones de ejecución (se está usando ahora mismo)"

build:
	@echo "Compilando módulo de simulador"
	python setup.py build_ext --inplace
	@echo "Módulo compilado con éxito"

build_c:
	@echo "Compilando módulo de simulador"
	python setup.py build_ext --inplace
	@echo "Módulo compilado con éxito"
	@echo "Eliminando archivos temporales"
	@rm -rf build/
	@rm -f cython_module.c
	@echo "Archivos temporales eliminados con éxito"

clean:
	@echo "Eliminando binarios y archivos temporales"
	@rm -rf build/
	@rm -f cython_module.c
	@rm -f *.so
	@echo "Archivos eliminados con éxito"