# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -Xptxas -O2 -lm -std=c++20 -lineinfo

# Target
TARGET = d3

# Source files
SRCS = src/d3.cu
HEADERS = src/constants.h

$(TARGET): $(SRCS) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(SRCS) -o $(TARGET)

clean:
	rm -f $(TARGET)