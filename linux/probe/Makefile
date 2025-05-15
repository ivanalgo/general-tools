BINS = cache tlb

all: $(BINS)

%: %.c
	gcc -Wall -g -O2 -o $@ $<


clean:
	rm -rf $(BINS)
