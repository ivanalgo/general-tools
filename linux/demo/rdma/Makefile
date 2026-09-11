CC ?= gcc
CFLAGS ?= -O2 -g -Wall -Wextra -Werror -std=c11
CPPFLAGS ?= -Iinclude
LDLIBS ?= -lrdmacm -libverbs

TARGET := rdma_demo
SRCS := src/main.c src/args.c src/rdma_context.c src/pattern.c src/verbs_selftest.c src/verbs_socket.c
OBJS := $(SRCS:.c=.o)

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(CPPFLAGS) -o $@ $^ $(LDLIBS)

clean:
	rm -f $(TARGET) $(OBJS)
