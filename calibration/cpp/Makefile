extension = global_refract_loss$(shell python3-config --extension-suffix)

.PHONY: build

build: $(extension)

$(extension): bindings.cpp global_refract_loss.cpp global_refract_loss.h
	g++ -O2 -Wall -Wextra -shared -std=c++17 -fPIC $(shell python3 -m pybind11 --includes) bindings.cpp global_refract_loss.cpp -o $(extension) -fopenmp


