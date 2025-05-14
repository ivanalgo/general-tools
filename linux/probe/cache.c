#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#define MIN_SIZE (1 * 1024)         // 1KB
#define MAX_SIZE (100 * 1024 * 1024) // 100MB
#define ITERATIONS 10000
#define STRIDE 64                   // 64字节步长，通常是一个缓存行大小

typedef unsigned long long int u64;

// 获取当前时间(毫秒)
u64  get_time_ms() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (u64)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

// 测试特定大小内存块的访问时间
double test_access_time(size_t size) {
    // 分配内存并填充
    char *buffer = malloc(size);
    if (!buffer) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }
    memset(buffer, 0, size);

    // 预热缓存(可选)
    for (size_t i = 0; i < size; i += STRIDE) {
        buffer[i] = 1;
    }

    // 开始计时
    u64 start = get_time_ms();

    // 多次访问内存
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (size_t i = 0; i < size; i += STRIDE) {
            buffer[i] += 1; // 修改数据
        }
    }

    // 结束计时
    u64 end = get_time_ms();

    free(buffer);
    return (end - start) / (size / STRIDE) / ITERATIONS; // 返回每64字节的纳秒时间
}

int main() {
    printf("Testing memory access times to detect cache sizes...\n");
    printf("Size (KB)\tTime per 64B (ns)\n");

    for (size_t size = MIN_SIZE; size <= MAX_SIZE; size *= 2) {
        double time_per_64b = test_access_time(size);
        printf("%zu\t\t%.2f\n", size / 1024, time_per_64b);

        // 在大小翻倍之间添加中间点(可选)
        if (size * 1.5 <= MAX_SIZE) {
            double mid_time = test_access_time(size * 1.5);
            printf("%zu\t\t%.2f\n", (size_t)(size * 1.5) / 1024, mid_time);
        }
    }

    return 0;
}
