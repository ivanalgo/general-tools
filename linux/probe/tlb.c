#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <x86intrin.h>  // 用于RDTSC指令

#define PAGE_SIZE (1 << 12)        // 4KB页
#define MAX_MAPPINGS (512 * 32 * 16)           // 最大映射数
#define ITERATIONS 1000000
#define CACHE_LINE_SIZE 64         // 缓存行大小

// 内存屏障，确保指令顺序
static inline void mfence() {
    asm volatile("mfence" ::: "memory");
}

// 获取TSC计数器
static inline uint64_t rdtsc() {
    unsigned int lo, hi;
    asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)hi << 32) | lo;
}

// 创建连续VA空间的多个映射指向同一物理页
char* create_continuous_mappings(int mapping_count) {
    int fd = syscall(SYS_memfd_create, "tlb_test", 0);
    if (fd == -1) {
        perror("memfd_create failed");
        exit(EXIT_FAILURE);
    }
    
    if (ftruncate(fd, PAGE_SIZE) == -1) {
        perror("ftruncate failed");
        exit(EXIT_FAILURE);
    }
    
    char* mapping = mmap(NULL, mapping_count * PAGE_SIZE, PROT_NONE, 
                        MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (mapping == MAP_FAILED) {
        perror("初始mmap失败");
        exit(EXIT_FAILURE);
    }
    
    for (int i = 0; i < mapping_count; i++) {
        void* addr = mmap(mapping + i * PAGE_SIZE, PAGE_SIZE, 
                         PROT_READ|PROT_WRITE,
                         MAP_FIXED|MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            perror("页映射失败");
            exit(EXIT_FAILURE);
        }
    }
    
    close(fd);
    return mapping;
}

void free_continuous_mappings(char* mapping, int mapping_count) {
    munmap(mapping, mapping_count * PAGE_SIZE);
}

// 计算TSC频率(Hz)
static uint64_t calibrate_tsc_freq() {
    struct timespec start_ts, end_ts;
    uint64_t start_tsc, end_tsc;
    
    clock_gettime(CLOCK_MONOTONIC, &start_ts);
    start_tsc = rdtsc();
    
    // 等待约100ms
    usleep(100000);
    
    clock_gettime(CLOCK_MONOTONIC, &end_ts);
    end_tsc = rdtsc();
    
    uint64_t ns_elapsed = (end_ts.tv_sec - start_ts.tv_sec) * 1000000000ULL + 
                         (end_ts.tv_nsec - start_ts.tv_nsec);
    uint64_t tsc_elapsed = end_tsc - start_tsc;
    
    return tsc_elapsed * 1000000000ULL / ns_elapsed;
}

// 测试TLB性能(使用TSC)
double test_tlb_access(int mapping_count, uint64_t tsc_freq) {
    char* mapping = create_continuous_mappings(mapping_count);
    
    // 初始化数据并预热缓存
    for (int i = 0; i < mapping_count; i++) {
        mapping[i * PAGE_SIZE] = 0;
    }
    
    // 确保所有访问都在缓存中(只测量TLB开销)
    for (int i = 0; i < mapping_count; i++) {
        __builtin_ia32_clflush(mapping + i * PAGE_SIZE);
    }
    mfence();
    
    uint64_t start = rdtsc();
    
    // 多次访问所有映射
    for (int iter = 0; iter < ITERATIONS;) {
        for (int i = 0; i < mapping_count; i++, iter++) {
            mapping[i * PAGE_SIZE] += 1;
        }
        //mfence(); // 确保每次迭代的顺序
    }
    
    uint64_t end = rdtsc();
    
    free_continuous_mappings(mapping, mapping_count);
    
    // 转换为纳秒
    double cycles_per_access = (double)(end - start) / (ITERATIONS);
    return cycles_per_access * 1000000000.0 / tsc_freq;
}

int main() {
    // 校准TSC频率
    uint64_t tsc_freq = calibrate_tsc_freq();
    printf("TSC频率: %.2f GHz\n", tsc_freq / 1000000000.0);
    
    printf("Linux TLB大小测试(使用TSC)...\n");
    printf("映射数\t每次访问时间(ns)\t每次访问周期数\n");
    
    for (int mappings = 1; mappings <= MAX_MAPPINGS; mappings *= 2) {
        double time_per_access = test_tlb_access(mappings, tsc_freq);
        double cycles_per_access = time_per_access * tsc_freq / 1000000000.0;
        printf("%d\t%.2f\t\t%.1f\n", mappings, time_per_access, cycles_per_access);
        
        if (mappings * 1.5 <= MAX_MAPPINGS * 0) {
            double mid_time = test_tlb_access(mappings * 1.5, tsc_freq);
            double mid_cycles = mid_time * tsc_freq / 1000000000.0;
            printf("%d\t%.2f\t\t%.1f\n", (int)(mappings * 1.5), mid_time, mid_cycles);
        }
    }
    
    return 0;
}
