#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <sys/time.h>

#define SCHED_PATH "/proc/self/sched"
#define BUFFER_SIZE 1024

double get_runtime_us() {
    char buffer[BUFFER_SIZE];
    int fd = open(SCHED_PATH, O_RDONLY);
    if (fd < 0) {
        perror("Failed to open sched file");
        exit(EXIT_FAILURE);
    }

    read(fd, buffer, BUFFER_SIZE);
    close(fd);

    char *runtime_str = strstr(buffer, "se.sum_exec_runtime");
    if (!runtime_str) {
        fprintf(stderr, "Failed to find runtime info\n");
        exit(EXIT_FAILURE);
    }

    double runtime;
    sscanf(runtime_str, "se.sum_exec_runtime : %lf", &runtime);
    return runtime * 1000;
}

#define BATCH_RUN_IN_US (10000)

void control_cpu_usage(double target_usage) {
    double start_runtime, end_runtime;
    struct timeval start_time, end_time;

    gettimeofday(&start_time, NULL);
    start_runtime = get_runtime_us();

    while (1) {

        # A small mount of running time of 10 ms
        do {
            gettimeofday(&end_time, NULL);
        } while ((end_time.tv_sec - start_time.tv_sec) * 1e6 +
                 (end_time.tv_usec - start_time.tv_usec) < BATCH_RUN_IN_US);

        end_runtime = get_runtime_us();

        double delta_runtime = (end_runtime - start_runtime);
        double delta_time = (end_time.tv_sec - start_time.tv_sec) * 1e6 +
                            (end_time.tv_usec - start_time.tv_usec);

        double sleep_time = (delta_runtime / target_usage) - delta_time;

        # compensation for overrunning
        if (sleep_time > 0) {
            usleep((useconds_t)sleep_time);
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <target CPU usage>\n", argv[0]);
        return EXIT_FAILURE;
    }

    double target_usage = atof(argv[1]);
    if (target_usage <= 0 || target_usage > 1) {
        fprintf(stderr, "Please specify a target CPU usage between 0 and 1.\n");
        return EXIT_FAILURE;
    }

    control_cpu_usage(target_usage);

    return 0;
}
