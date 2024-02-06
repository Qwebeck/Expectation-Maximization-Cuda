LINALG_TARGET=linalg.o

all: $(LINALG_TARGET)

$(LINALG_TARGET):
	nvcc mixture_models.cu linalg/vector.cpp linalg/matrix.cpp linalg/errors.cpp linalg/util.cpp linalg/linsolve.cpp linalg/eigen.cpp linalg/linreg.cpp linalg/rand.cpp utils.cpp -L/linalg


clean:
	rm $(LINALG_TARGET)

.PHONY: clean