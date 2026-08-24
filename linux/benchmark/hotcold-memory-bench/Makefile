CC ?= cc
CFLAGS ?= -O2 -g
CFLAGS += -std=gnu11 -Wall -Wextra -Werror

.PHONY: all clean test
all: hotcold-memory-bench
hotcold-memory-bench: hotcold-memory-bench.c
	$(CC) $(CFLAGS) -o $@ $<
test: hotcold-memory-bench
	./tests/smoke.sh ./hotcold-memory-bench
clean:
	$(RM) hotcold-memory-bench
