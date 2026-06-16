CC ?= gcc
CFLAGS ?= -O2 -g -Wall -Wextra -std=gnu11
LDFLAGS ?=

TARGET := memheat_profiler
SRCS := main.c backend.c backend_pebs.c backend_ibs.c pmu_sysfs.c heatmap.c perf_sampler.c
OBJS := $(SRCS:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LDFLAGS) -lm

%.o: %.c profiler.h backend.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS) *.o
