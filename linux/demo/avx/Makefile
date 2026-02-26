CXX = g++
CXXFLAGS = -Wall -g -O2 -march=native --std=c++17
DEPFLAGS = -MMD -MP

TARGET = avx
SRCS = avx.cpp
OBJS = $(SRCS:.cpp=.o)
DEPS = $(SRCS:.cpp=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) -c $< -o $@

clean:
	rm -rf $(TARGET) $(OBJS) $(DEPS)

-include $(DEPS)
