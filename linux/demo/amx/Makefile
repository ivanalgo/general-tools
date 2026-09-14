CC      = gcc
CXX     = g++
CPPFLAGS ?=
WARN    := -Wall -Wextra -Wpedantic
AMX_FLAGS := -mamx-tile -mamx-bf16 -mamx-int8
CFLAGS   ?= -O2
CXXFLAGS ?= -O2

.PHONY: all clean check check-compiler run fp16

all: amx_c amx_cpp

check-compiler:
	@printf 'int main(void) { return 0; }\n' | \
	    $(CC) -x c -std=c11 $(AMX_FLAGS) -fsyntax-only - >/dev/null 2>&1 || \
	    { echo "error: $(CC) does not support Intel AMX compiler options."; \
	      echo "       GCC 11+ is required; GCC 12+ is recommended."; \
	      echo "       Try: make CC=gcc-12 CXX=g++-12"; \
	      exit 1; }
	@printf 'int main() { return 0; }\n' | \
	    $(CXX) -x c++ -std=c++20 $(AMX_FLAGS) -fsyntax-only - >/dev/null 2>&1 || \
	    { echo "error: $(CXX) lacks C++20 and/or Intel AMX support."; \
	      echo "       GCC 11+ is required; GCC 12+ is recommended."; \
	      echo "       Try: make CC=gcc-12 CXX=g++-12"; \
	      exit 1; }

amx_c: amx.c | check-compiler
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -std=c11 $(AMX_FLAGS) $< -o $@

amx_cpp: amx.cpp | check-compiler
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARN) -std=c++20 $(AMX_FLAGS) $< -o $@

fp16: amx_cpp_fp16

amx_cpp_fp16: amx.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARN) -std=c++20 \
	    $(AMX_FLAGS) -mamx-fp16 -DAMX_ENABLE_FP16 $< -o $@

check: check-compiler
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -std=c11 $(AMX_FLAGS) -fsyntax-only amx.c
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARN) -std=c++20 $(AMX_FLAGS) -fsyntax-only amx.cpp

run: all
	./amx_c
	./amx_cpp

clean:
	$(RM) amx_c amx_cpp amx_cpp_fp16
