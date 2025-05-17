#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/mman.h>
#include <string.h>

#define SIZE_MB (1024UL)			// Memory size to allocate (MB)
#define PAGE_SIZE 4096				// Usually system page size
#define TOUCH_EVERY_PAGE 1			// Whether to touch every page to build page tables

long read_page_tables_kb() {
	FILE *fp = fopen("/proc/meminfo", "r");
	if (!fp) {
		perror("fopen /proc/meminfo");
		return -1;
	}

	char line[128];
	long kb = -1;

	while (fgets(line, sizeof(line), fp)) {
		if (strncmp(line, "PageTables:", 11) == 0) {
			sscanf(line + 11, "%ld", &kb);
			break;
		}
	}

	fclose(fp);
	return kb;
}

void report_ptable_size(unsigned long mem_size)
{
	unsigned long pte_items = mem_size / PAGE_SIZE;
	unsigned long pte_size = pte_items * 8;
	unsigned long pmd_items = pte_size / PAGE_SIZE;
	unsigned long pmd_size = pmd_items * 8;
	unsigned long total_size = pte_size + pmd_size;

	printf("\n\n");
	printf("Page table statistics:\n");
	printf("pte_items:    %lu\n", pte_items);
	printf("pte_size:     %lu\n", pte_size);
	printf("pmd_items:    %lu\n", pmd_items);
	printf("pmd_size:     %lu\n", pmd_size);
	printf("total size:   %lu   %lu KB\n", total_size, total_size / 1024);
}

int main() {
	size_t alloc_size = SIZE_MB * 1024 * 1024;
	printf("Allocating %lu of memory...\n", alloc_size);

	// Use mmap to allocate anonymous memory
	void *ptr = mmap(NULL, alloc_size, PROT_READ | PROT_WRITE,
					 MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
	if (ptr == MAP_FAILED) {
		perror("mmap");
		return 1;
	}

	// Explicitly request not to use huge pages
	if (madvise(ptr, alloc_size, MADV_NOHUGEPAGE) != 0) {
		perror("madvise(NO_HUGEPAGE)");
		munmap(ptr, alloc_size);
		return 1;
	}

	long sys_pg_size = read_page_tables_kb();
	printf("Current page table size is %lu KB\n", sys_pg_size);

	// Touch each page to force physical page allocation + page table creation
	// Then release them with MADV_DONTNEED, which may not release page tables
	for (size_t offset = 0; offset < alloc_size; offset += PAGE_SIZE) {
		((char *)ptr)[offset] = 42;	// Touch

		if (madvise(ptr + offset, PAGE_SIZE, MADV_DONTNEED) != 0) {
			perror("madvise");
			munmap(ptr, alloc_size);
			return 1;
		}
	}

	printf("Mmap memory of %lu, then touch and use MADV_DONTNEED to release them\n", alloc_size);
	sys_pg_size = read_page_tables_kb();
	printf("Current page table size is %lu KB\n", sys_pg_size);

	// Cleanup
	munmap(ptr, alloc_size);

	report_ptable_size(alloc_size);
	return 0;
}

