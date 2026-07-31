CC      = gcc
CXX     = g++
CPPFLAGS ?=
WARN    := -Wall -Wextra -Wpedantic
AMX_FLAGS := -mamx-tile -mamx-bf16 -mamx-int8
CFLAGS   ?= -O2
CXXFLAGS ?= -O2

.PHONY: all clean check run fp16

all: amx_c amx_cpp

amx_c: amx.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -std=c11 $(AMX_FLAGS) $< -o $@

amx_cpp: amx.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARN) -std=c++20 $(AMX_FLAGS) $< -o $@

fp16: amx_cpp_fp16

amx_cpp_fp16: amx.cpp
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARN) -std=c++20 \
	    $(AMX_FLAGS) -mamx-fp16 -DAMX_ENABLE_FP16 $< -o $@

check:
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -std=c11 $(AMX_FLAGS) -fsyntax-only amx.c
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(WARN) -std=c++20 $(AMX_FLAGS) -fsyntax-only amx.cpp

run: all
	./amx_c
	./amx_cpp

clean:
	$(RM) amx_c amx_cpp amx_cpp_fp16
