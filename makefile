# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -Xptxas -O2 -lm -std=c++20 -lineinfo

# Target
TARGET = d3

# cudart path
CUDART_PATH = /usr/local/cuda-12.6/targets/x86_64-linux/lib/libcudart_static.a

# Source files
SRCS = src/d3.cu
HEADERS = src/constants.h

elf: $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET)

lib: $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) --cudart static -lib -o libd3.a $(SRCS) -D BUILD_LIBRARY
	mkdir -p temp_objs
	cd temp_objs && ar x ../libd3.a
	cd temp_objs && ar x $(CUDART_PATH)
	ar rcs libd3_full.a temp_objs/*.o
	

test: src/test.c
	gcc src/test.c libd3_full.a  -lstdc++ -lm

clean:
	rm -f $(TARGET)
	rm -f libd3.a
	rm -f libd3_full.a
	rm -rf temp_objs