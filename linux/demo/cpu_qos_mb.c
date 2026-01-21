#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <dirent.h>
#include <errno.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <getopt.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>

#define RESCTRL_PATH "/sys/fs/resctrl"
#define STRIDE	   64
#define MIN_BUF_MB 128

static char group_name[64];
static char group_path[256];
static char group_schemata[256];

static void die(const char *msg)
{
	perror(msg);
	exit(EXIT_FAILURE);
}

static void write_file(const char *path, const char *buf)
{
	int fd = open(path, O_WRONLY);
	if (fd < 0)
		die(path);

	if (write(fd, buf, strlen(buf)) < 0)
		die(path);

	close(fd);
}

static uint64_t read_u64(const char *path)
{
	char buf[64];
	int fd = open(path, O_RDONLY);
	if (fd < 0)
		die(path);

	ssize_t n = read(fd, buf, sizeof(buf) - 1);
	if (n < 0)
		die(path);

	buf[n] = '\0';
	close(fd);
	return strtoull(buf, NULL, 0);
}

static uint64_t read_group_mbm_metric(const char *group, const char *item)
{
	char mon_path[128];
	snprintf(mon_path, sizeof(mon_path),
			 RESCTRL_PATH "/%s/mon_data", group);

	DIR *dir = opendir(mon_path);
	if (!dir)
		die(mon_path);

	struct dirent *de;
	uint64_t sum = 0;

	while ((de = readdir(dir))) {
		if (strncmp(de->d_name, "mon_L3_", 7) == 0) {
			char file[512];
			snprintf(file, sizeof(file),
					 "%s/%s/%s",
					 mon_path, de->d_name, item);
			sum += read_u64(file);
		}
	}

	closedir(dir);
	return sum;
}

/* ---------- resctrl 清理 ---------- */

static void cleanup_group(void)
{
	char path[512];
	char pid_str[32];

	printf("\nCleaning up resctrl group %s\n", group_name);

	/* 1. 把自己移回 root group */
	snprintf(path, sizeof(path),
			 RESCTRL_PATH "/tasks");
	snprintf(pid_str, sizeof(pid_str), "%d", getpid());
	write_file(path, pid_str);

	/* 2. 删除 group */
	if (rmdir(group_path) < 0) {
		perror("rmdir mon_group");
	} else {
		printf("Removed group %s\n", group_name);
	}
}


#define MBM_CFG_PATH "/sys/fs/resctrl/info/L3_MON/mbm_total_bytes_config"
#define MAX_CFG_LEN 4096

#define CCD_ID_MAX 1024
static int ccd_list[CCD_ID_MAX];
static int ccd_list_cnt = 0;

void check_mbm_total_bytes_config(void)
{
	FILE *fp;
	char buf[MAX_CFG_LEN];
	char *token;
	int  bad = 0;

	fp = fopen(MBM_CFG_PATH, "r");
	if (!fp) {
		fprintf(stderr, "ERROR: cannot open %s: %s\n",
				MBM_CFG_PATH, strerror(errno));
		return;
	}

	if (!fgets(buf, sizeof(buf), fp)) {
		fprintf(stderr, "ERROR: failed to read %s\n", MBM_CFG_PATH);
		fclose(fp);
		return;
	}
	fclose(fp);

	/* 去掉末尾换行 */
	buf[strcspn(buf, "\n")] = '\0';

	token = strtok(buf, ";");
	while (token) {
		int id;
		unsigned int val;

		if (sscanf(token, "%d=0x%x", &id, &val) != 2) {
			fprintf(stderr, "WARN: unrecognized token: %s\n", token);
			token = strtok(NULL, ";");
			continue;
		}

		ccd_list[ccd_list_cnt++] = id;
		if (val != 0x3f) {
			printf("  CCD %d: 0x%x (EXPECTED 0x3f)\n", id, val);
			bad = 1;
		}

		token = strtok(NULL, ";");
	}

	if (!bad) {
		return;
	}

	printf("found mbm_total_bytes_config is not configured as 0x3f\n");
	printf("\nSuggested fix command:\n");
	printf("echo \"");

	fp = fopen(MBM_CFG_PATH, "r");
	if (!fp || !fgets(buf, sizeof(buf), fp)) {
		printf("<failed to re-read config>\"\n");
		if (fp)
			fclose(fp);
		return;
	}

	if (fp)
		fclose(fp);

	buf[strcspn(buf, "\n")] = '\0';

	token = strtok(buf, ";");
	while (token) {
		int id;
		unsigned int val;

		if (sscanf(token, "%d=0x%x", &id, &val) == 2) {
			printf("%d=0x3f;", id);
		}
		token = strtok(NULL, ";");
	}

	printf("\" > %s\n", MBM_CFG_PATH);
	return;
}

static void signal_handler(int sig)
{
	(void)sig;
	cleanup_group();
	exit(0);
}

size_t g_buf_size = MIN_BUF_MB * 1024 * 1024;	 /* bytes */
int	g_duration = 1;
bool   g_verbose = false;
unsigned int g_error = 50;
int g_bw = -1;

static void parse_args(int argc, char *argv[])
{
	static const struct option long_options[] = {
		{ "size",	   required_argument, 0, 's' },
		{ "duration",  required_argument, 0, 'd' },
		{ "verbose",   no_argument,	      0, 'v' },
		{ "error",	   required_argument, 0, 'e' },
		{ "bw",        required_argument, 0, 'b' },
		{ 0,		   0,				  0,  0  }
	};

	int opt;
	while ((opt = getopt_long(argc, argv, "s:d:ve:b:", long_options, NULL)) != -1) {
		switch (opt) {
		case 's': {
			long mb = atol(optarg);
			if (mb < MIN_BUF_MB) {
				fprintf(stderr,
					"Invalid --size %ld MB, must be >= %d MB\n",
					mb, MIN_BUF_MB);
				exit(EXIT_FAILURE);
			}
			g_buf_size = (size_t)mb * 1024 * 1024;
			break;
		}
		case 'd': {
			long d = atol(optarg);
			if (d <= 0) {
				fprintf(stderr,
					"Invalid --iteration %ld, must be > 0\n", d);
				exit(EXIT_FAILURE);
			}
			g_duration = (int)d;
			break;
		}
		case 'v': {
			g_verbose = true;
			break;
		}
		case 'e': {
			unsigned int e = atoi(optarg);
			if (e <= 0 || e > 500) {
				fprintf(stderr,
					"Invalid --error %d, must be >0 and < 500\n", e);
				exit(EXIT_FAILURE);
			}
			g_error = e;
			break;
		}
		case 'b': {
			g_bw = atoi(optarg);
			printf("g_bw = %d\n", g_bw);
			break;
		}
		default:
			fprintf(stderr,
				"Usage: %s [ options ]\n"
				"\n"
				"-d --duration  duration between two outputs\n"
				"-s --size      memory size for each iteration to deal with\n"
				"-v --verbose   show verbose inforation\n"
				"-e --error     specify the tolerable errror\n"
				"-b --bw        specify the bw for memory bandwidth allocation group\n", 
				argv[0]);
			exit(EXIT_FAILURE);
		}
	}
}

static inline double now_sec(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec*1e9 + ts.tv_nsec;
}

int main(int argc, char *argv[])
{
	pid_t pid = getpid();

	parse_args(argc, argv);
	check_mbm_total_bytes_config();

	/* group 名字：pid_xxx */
	snprintf(group_name, sizeof(group_name), "pid_%d", pid);
	snprintf(group_path, sizeof(group_path),
			 RESCTRL_PATH "/%s", group_name);
	snprintf(group_schemata, sizeof(group_schemata),
			RESCTRL_PATH "/pid_%d/schemata", pid); 

	/* 如果 group 已存在，直接报错 */
	if (access(group_path, F_OK) == 0) {
		fprintf(stderr,
			"resctrl group %s already exists.\n"
			"Please remove it manually:\n"
			"  rmdir %s\n",
			group_name, group_path);
		return EXIT_FAILURE;
	}

	/* 注册信号处理 */
	signal(SIGINT,  signal_handler);
	signal(SIGTERM, signal_handler);

	/* 创建 group */
	if (mkdir(group_path, 0755) < 0)
		die("mkdir mon_group");

	/* set mbm control */
	if (g_bw > 0) {
		char buf[256];
		for (int i = 0; i < ccd_list_cnt; i++) {
			snprintf(buf, sizeof(buf),	"MB:%d=%d\n", ccd_list[i], g_bw);
			write_file(group_schemata, buf);
		}
	}

	/* 把自己加入 group */
	char tasks_path[512];
	char pid_str[32];
	snprintf(tasks_path, sizeof(tasks_path),
		"%s/tasks", group_path);
	snprintf(pid_str, sizeof(pid_str), "%d", pid);
	write_file(tasks_path, pid_str);

	/* 分配内存 */
	char *buf;
	if (posix_memalign((void **)&buf, 4096, g_buf_size))
		die("posix_memalign");

	memset(buf, 0, g_buf_size);

	uint64_t prev_local =
		read_group_mbm_metric(group_name, "mbm_local_bytes");
	uint64_t prev_total =
		read_group_mbm_metric(group_name, "mbm_total_bytes");

	printf("Running... Press Ctrl-C to stop.\n\n");

	double prev_time = now_sec();
	uint64_t expected_bytes = 0;

	while (1) {
		for (size_t i = 0; i < g_buf_size; i += STRIDE)
			buf[i]++;

		expected_bytes += g_buf_size;
		usleep(1);

		double now_time = now_sec();
		if (now_time - prev_time < 1e9 * g_duration)
			continue;

		uint64_t cur_local =
			read_group_mbm_metric(group_name, "mbm_local_bytes");
		uint64_t cur_total =
			read_group_mbm_metric(group_name, "mbm_total_bytes");

		uint64_t delta_local = cur_local - prev_local;
		uint64_t delta_total = cur_total - prev_total;

		/* Calculate the error in Unit of one-1000th */
		int error = (delta_total * 1000 / expected_bytes) - 1000;

		if (g_verbose) {
			printf("MBM statistiic\n");
			printf("Sum:\n");
			printf("       local : %lu bytes\n", delta_local);
			printf("       total : %lu bytes\n", delta_total);
			printf("Speed :\n"
			       "       local %.2f MB/s\n"
			       "       total %.2f MB/s\n",
				       delta_local * 1e9 / ((now_time - prev_time) * 1024 * 1024),
			               delta_total * 1e9 / ((now_time - prev_time) * 1024 * 1024));

			printf("Program:\n");
			printf("         %lu bytes\n", expected_bytes);
			printf("         %.2f MB/s\n",
					expected_bytes * 1e9 / ((now_time - prev_time) * 1024 * 1024));
			printf("Error:\n");
			printf("ratio(resctrl / expected) : %.3f\n",
		                        (double)delta_total / expected_bytes);
			printf("error: %d‰\n\n", error);
		}

		int abs_error =abs(error);
		if (abs_error > g_error)
			printf("Unexpect Error (resctrl size / buffer size): %d\n", error);

		prev_local = cur_local;
		prev_total = cur_total;
		prev_time = now_time;
		expected_bytes = 0;
	}

	return 0;
}
