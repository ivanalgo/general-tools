// perf_timer_mmap_percpu.c
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <sched.h>
#include <pthread.h>
#include <errno.h>
#include <sys/syscall.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/poll.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <time.h>
#include <inttypes.h>

const uint64_t period_ns = 10000000ULL; // 10ms default, 可调整

static inline uint64_t now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
	return (int64_t)ts.tv_sec * 1000000000ll + ts.tv_nsec;
}

static long perf_event_open(struct perf_event_attr *hw_event, pid_t pid,
							int cpu, int group_fd, unsigned long flags)
{
	return syscall(__NR_perf_event_open, hw_event, pid, cpu, group_fd, flags);
}

static __thread char buf[128];

char *time_to_str(int64_t diff)
{

	if (diff > 1000000000L) {
		snprintf(buf, sizeof(buf), "%ld sec", diff / 1000000000L);
	} else if (diff > 1000000L) {
		snprintf(buf, sizeof(buf), "%ld ms", diff / 1000000L);
	} else if (diff > 1100) {
		snprintf(buf, sizeof(buf), "%ld us", diff / 1000L);
	} else {
		snprintf(buf, sizeof(buf), "less");
	}

	return buf;
}

static inline void memcpy_wrap(void *dst, char *data, size_t data_size,
                               size_t *off, size_t sz)
{
    if (*off + sz < data_size) {
        memcpy(dst, data + *off, sz);
	*off += sz;
    } else {
        size_t first = data_size - *off;
        memcpy(dst, data + *off, first);
        memcpy((char *)dst + first, data, sz - first);
	*off = sz - first;
    }
}

__thread uint64_t last_ts = 0;
/* parse ring buffer events */
static void handle_event_page(char *data, size_t data_size, uint64_t *tail,
							  uint64_t head, int cpu)
{

	while (*tail < head) {
		size_t off = *tail & (data_size - 1);
		struct perf_event_header *hdr = (struct perf_event_header *)(data + off);

		size_t space = head - *tail;
		space -= sizeof(*hdr);

		if ((off + sizeof(*hdr)) > data_size) {
			// header wraps — handle by copying header bytes to stack (rare)
			// For simplicity, break (this shouldn't normally happen with power-of-two buffer)
			break;
		}

		if (hdr->size == 0) break; // defensive

		if (hdr->type == PERF_RECORD_SAMPLE 
			|| hdr->type == PERF_RECORD_THROTTLE
			|| hdr->type == PERF_RECORD_UNTHROTTLE) {

			if (space < 16) {
				printf("cpu buffer %d, less than 16 bytes\n", cpu);
				break;
			}

			// sample_type we set only includes PERF_SAMPLE_TIME, so data after header is timestamp (u64)
			uint64_t ts;
			size_t data_off = off + sizeof(*hdr);
			int warpping = 0;
			int cpu_data;

			memcpy_wrap(&ts, data, data_size, &data_off, sizeof(ts));

			uint32_t cpuid = 0, res = 0;
			memcpy_wrap(&cpuid, data, data_size, &data_off, sizeof(cpuid));

			memcpy_wrap(&res, data, data_size, &data_off, sizeof(res));

			if (hdr->type == PERF_RECORD_THROTTLE) {
				// throttle by nohz idle, reset the process
				//printf("cpu %d throttle at %lu\n", cpu, ts);
				ts = 0;
			} else if (hdr->type == PERF_RECORD_UNTHROTTLE ) {
				//printf("cpu %d unthrottle at %lu\n", cpu, ts);
			} else if (last_ts) {
				if (warpping)
					printf("cpu: %d, warpping = %d\n", cpu, warpping);
				int64_t diff = (int64_t)(ts - last_ts - period_ns);
				if (diff > 100000L)
				//if (cpu != 0)
					printf("cpu: ,%d, cpu_data: %u, now: ,%lu, diff: ,%s, ts: ,%lu, last_ts: ,%lu\n",
						cpu, cpuid, now_ns(), time_to_str(diff), ts, last_ts);
			}

			last_ts = ts;
		} else if (hdr->type == PERF_RECORD_LOST) {
			// lost samples: payload is u64 lost count
			uint64_t lost = 0;
			size_t data_off = off + sizeof(*hdr);
			if (data_off + sizeof(lost) <= data_size)
				memcpy(&lost, data + data_off, sizeof(lost));
			printf("  LOST %" PRIu64 "\n", lost);
		} else {
			// other record types can be ignored for this simple test
			printf("cpu %d, missing event %d processing\n", cpu, hdr->type);
		}

		// advance tail
		*tail += hdr->size;
	}
}

/* worker: one thread per CPU */
void *worker(void *arg)
{
	long cpu = (long)arg;
	const int pages = 8;	// ring buffer pages (power-of-two better)
	const int mmap_pages = 1 + pages; // 1 metadata page + pages data

	// set affinity to CPU
	cpu_set_t cs;
	CPU_ZERO(&cs);
	CPU_SET(cpu, &cs);
	if (pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs) != 0) {
		perror("pthread_setaffinity_np");
		return NULL;
	}

	struct perf_event_attr attr;
	memset(&attr, 0, sizeof(attr));
	attr.type = PERF_TYPE_SOFTWARE;
	attr.config = PERF_COUNT_SW_CPU_CLOCK; // software cpu_clock
	attr.size = sizeof(attr);
	attr.sample_period = period_ns;
	attr.freq = 0;
	attr.sample_type = PERF_SAMPLE_TIME | PERF_SAMPLE_CPU;
	attr.wakeup_events = 1; // wake user-space on each event
	attr.disabled = 0;
	attr.exclude_kernel = 0;
	attr.exclude_hv = 0;

	int fd = perf_event_open(&attr, -1, (int)cpu, -1, 0);
	if (fd < 0) {
		fprintf(stderr, "cpu %ld: perf_event_open failed: %s\n", cpu, strerror(errno));
		return NULL;
	}

	size_t page_size = sysconf(_SC_PAGESIZE);
	size_t mmap_len = page_size * mmap_pages;

	void *map = mmap(NULL, mmap_len, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (map == MAP_FAILED) {
		fprintf(stderr, "cpu %ld: mmap failed: %s\n", cpu, strerror(errno));
		close(fd);
		return NULL;
	}

	struct perf_event_mmap_page *meta = (struct perf_event_mmap_page *)map;
	char *data = (char *)map + page_size;
	size_t data_size = page_size * pages;

	// enable the event (ensure counters start)
	if (ioctl(fd, PERF_EVENT_IOC_RESET, 0) == -1)
		perror("ioctl RESET");
	if (ioctl(fd, PERF_EVENT_IOC_ENABLE, 0) == -1)
		perror("ioctl ENABLE");

	struct pollfd pfd = { .fd = fd, .events = POLLIN };

	while (1) {
		int p = poll(&pfd, 1, -1);
		if (p < 0) {
			if (errno == EINTR) continue;
			perror("poll");
			break;
		}

		if (pfd.revents & POLLIN) {
			// read from ring buffer
			// Acquire head
			uint64_t head = meta->data_head;
			__sync_synchronize();
			uint64_t tail = meta->data_tail;

			if (head == tail) {
				// no data, continue
				continue;
			}

			// traverse events from tail -> head
			handle_event_page(data, data_size, &tail, head, cpu);

			// publish new tail
			__sync_synchronize();
			meta->data_tail = tail;
		} else if (pfd.revents & POLLHUP) {
			printf("cpu %ld: POLLHUP\n", cpu);
			break;
		} else {
			// unexpected revents
			printf("cpu %ld: poll revents=0x%x\n", cpu, pfd.revents);
		}
	}

	munmap(map, mmap_len);
	close(fd);
	return NULL;
}

int main(int argc, char **argv)
{
	int ncpu = sysconf(_SC_NPROCESSORS_ONLN);
	printf("Starting perf mmap test on %d CPUs\n", ncpu);

	pthread_t *t = calloc(ncpu, sizeof(pthread_t));
	for (long i = 0; i < ncpu; i++) {
		if (pthread_create(&t[i], NULL, worker, (void *)i) != 0) {
			fprintf(stderr, "pthread_create failed for cpu %ld\n", i);
			return 1;
		}
		usleep(10000); // stagger thread starts a bit
	}

	for (int i = 0; i < ncpu; i++) {
		pthread_join(t[i], NULL);
	}

	free(t);
	return 0;
}

