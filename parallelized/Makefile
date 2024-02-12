LINALG_TARGET=mixture_models

all: $(LINALG_TARGET)

$(LINALG_TARGET):
	nvcc main.cu EM.cu ../linalg/vector.cpp ../linalg/matrix.cpp ../linalg/errors.cpp ../linalg/util.cpp ../linalg/linsolve.cpp ../linalg/eigen.cpp ../linalg/linreg.cpp ../linalg/rand.cpp utils.cu stats.cu -o $(LINALG_TARGET) -g

clean:
	rm $(LINALG_TARGET)


.PHONY: clean