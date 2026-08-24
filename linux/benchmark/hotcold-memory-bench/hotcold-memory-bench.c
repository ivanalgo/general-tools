#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <inttypes.h>
#include <limits.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

enum assess_mode { ASSESS_NONE, ASSESS_EVICTION, ASSESS_TIERING };
struct options {
	uint64_t total, hot, duration, interval, assess_at, settle;
	uint64_t memory_max, cold_threshold, seed;
	const char *cgroup, *fast_nodes, *slow_nodes, *backing_file;
	enum assess_mode assess;
	bool csv, counters;
};
struct bench {
	struct options o;
	uint8_t *mem;
	uint32_t *count;
	int backing_fd;
	uint64_t page_size, pages, hot_pages, current, chases, last_chases;
	long last_faults;
	struct timespec start, last;
	bool assessed;
};
static volatile sig_atomic_t stopped;

static void die(const char *s) { fprintf(stderr, "error: %s: %s\n", s, strerror(errno)); exit(1); }
static void signal_handler(int sig) { (void)sig; stopped = 1; }
static double elapsed(const struct timespec *a, const struct timespec *b)
{ return b->tv_sec - a->tv_sec + (b->tv_nsec - a->tv_nsec) / 1e9; }

static void usage(FILE *f, const char *p)
{
	fprintf(f,
"usage: %s --total-size SIZE --hot-size SIZE [options]\n"
"  --duration SEC --interval SEC --seed N --output text|csv\n"
"  --backing-file PATH\n"
"  --no-page-counters\n"
"  --assess eviction --assess-at SEC --cgroup PATH --memory-max SIZE --settle SEC\n"
"  --assess tiering --assess-at SEC --fast-nodes LIST --slow-nodes LIST\n"
"  --cold-threshold N\n", p);
}

static uint64_t number(const char *s, const char *name)
{
	char *end; unsigned long long v;
	errno = 0; v = strtoull(s, &end, 0);
	if (errno || end == s || *end) {
		fprintf(stderr, "error: invalid %s: %s\n", name, s); exit(1);
	}
	return v;
}

static uint64_t size_value(const char *s, const char *name)
{
	char *end; unsigned long long v; uint64_t m = 1;
	errno = 0; v = strtoull(s, &end, 0);
	if (errno || end == s) goto bad;
	if (*end) {
		switch (*end++) {
		case 'k': case 'K': m = 1ULL << 10; break;
		case 'm': case 'M': m = 1ULL << 20; break;
		case 'g': case 'G': m = 1ULL << 30; break;
		case 't': case 'T': m = 1ULL << 40; break;
		default: goto bad;
		}
		if (*end == 'i' || *end == 'I') end++;
		if (*end == 'b' || *end == 'B') end++;
	}
	if (*end || v > UINT64_MAX / m) goto bad;
	return (uint64_t)v * m;
bad:
	fprintf(stderr, "error: invalid %s: %s\n", name, s); exit(1);
}

static void options(int argc, char **argv, struct options *o)
{
	enum { TOTAL=1000,HOT,DURATION,INTERVAL,SEED,OUTPUT,NOCOUNTERS,ASSESS,
		ASSESSAT,CGROUP,MEMMAX,SETTLE,FAST,SLOW,COLD,BACKING };
	static const struct option lo[] = {
		{"total-size",1,0,TOTAL},{"hot-size",1,0,HOT},{"duration",1,0,DURATION},
		{"interval",1,0,INTERVAL},{"seed",1,0,SEED},{"output",1,0,OUTPUT},
		{"no-page-counters",0,0,NOCOUNTERS},{"assess",1,0,ASSESS},
		{"assess-at",1,0,ASSESSAT},{"cgroup",1,0,CGROUP},
		{"memory-max",1,0,MEMMAX},{"settle",1,0,SETTLE},
		{"fast-nodes",1,0,FAST},{"slow-nodes",1,0,SLOW},
		{"cold-threshold",1,0,COLD},{"backing-file",1,0,BACKING},
		{"help",0,0,'h'},{0}
	};
	int c;
	*o = (struct options){.duration=120,.interval=1,.assess_at=60,
		.settle=10,.seed=1,.counters=true};
	while ((c=getopt_long(argc,argv,"h",lo,0)) != -1) {
		switch (c) {
		case TOTAL:o->total=size_value(optarg,"total-size");break;
		case HOT:o->hot=size_value(optarg,"hot-size");break;
		case DURATION:o->duration=number(optarg,"duration");break;
		case INTERVAL:o->interval=number(optarg,"interval");break;
		case SEED:o->seed=number(optarg,"seed");break;
		case OUTPUT:
			if (!strcmp(optarg,"csv")) o->csv=true;
			else if (strcmp(optarg,"text")) { fputs("error: bad output\n",stderr);exit(1); }
			break;
		case NOCOUNTERS:o->counters=false;break;
		case ASSESS:
			if (!strcmp(optarg,"eviction")) o->assess=ASSESS_EVICTION;
			else if (!strcmp(optarg,"tiering")) o->assess=ASSESS_TIERING;
			else if (strcmp(optarg,"none")) { fputs("error: bad assessment\n",stderr);exit(1); }
			break;
		case ASSESSAT:o->assess_at=number(optarg,"assess-at");break;
		case CGROUP:o->cgroup=optarg;break;
		case MEMMAX:o->memory_max=size_value(optarg,"memory-max");break;
		case SETTLE:o->settle=number(optarg,"settle");break;
		case FAST:o->fast_nodes=optarg;break;
		case SLOW:o->slow_nodes=optarg;break;
		case COLD:o->cold_threshold=number(optarg,"cold-threshold");break;
		case BACKING:o->backing_file=optarg;break;
		case 'h':usage(stdout,argv[0]);exit(0);
		default:usage(stderr,argv[0]);exit(1);
		}
	}
	if (!o->total || !o->hot || o->hot>o->total || !o->duration || !o->interval)
		{ usage(stderr,argv[0]);exit(1); }
	if (o->assess && !o->counters) { fputs("error: assessment needs counters\n",stderr);exit(1); }
	if (o->assess && o->assess_at>=o->duration) { fputs("error: assess-at >= duration\n",stderr);exit(1); }
	if (o->assess==ASSESS_EVICTION && (!o->cgroup || !o->memory_max))
		{ fputs("error: eviction needs cgroup and memory-max\n",stderr);exit(1); }
	if (o->assess==ASSESS_TIERING && (!o->fast_nodes || !o->slow_nodes))
		{ fputs("error: tiering needs fast-nodes and slow-nodes\n",stderr);exit(1); }
	if (!o->seed) o->seed=1;
}

static uint64_t random64(uint64_t *s)
{
	uint64_t x=*s; x^=x>>12; x^=x<<25; x^=x>>27; *s=x;
	return x*2685821657736338717ULL;
}

static void initialize(struct bench *b)
{
	uint64_t *order,state=b->o.seed,i;
	int flags=MAP_POPULATE;
	b->page_size=sysconf(_SC_PAGESIZE);
	b->pages=b->o.total/b->page_size; b->hot_pages=b->o.hot/b->page_size;
	if (!b->pages || !b->hot_pages) { fputs("error: size below one page\n",stderr);exit(1); }
	b->o.total=b->pages*b->page_size; b->o.hot=b->hot_pages*b->page_size;
	b->backing_fd=-1;
	if(b->o.backing_file){
		b->backing_fd=open(b->o.backing_file,O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC,0600);
		if(b->backing_fd<0)die("open backing file");
		if(ftruncate(b->backing_fd,b->o.total))die("ftruncate backing file");
		if(unlink(b->o.backing_file))die("unlink backing file");
		flags|=MAP_SHARED;
	}else flags|=MAP_PRIVATE|MAP_ANONYMOUS;
	b->mem=mmap(0,b->o.total,PROT_READ|PROT_WRITE,flags,b->backing_fd,0);
	if (b->mem==MAP_FAILED) die("mmap");
	if (madvise(b->mem,b->o.total,MADV_NOHUGEPAGE)) die("madvise");
	if (b->o.counters && !(b->count=calloc(b->pages,sizeof(*b->count)))) die("counters");
	if (!(order=malloc(b->hot_pages*sizeof(*order)))) die("permutation");
	for(i=0;i<b->pages;i++){uint64_t *p=(uint64_t*)(b->mem+i*b->page_size);p[0]=i;p[1]=i^0x484f54434f4c4400ULL;}
	for(i=0;i<b->hot_pages;i++)order[i]=i;
	for(i=b->hot_pages-1;i;i--){uint64_t j=random64(&state)%(i+1),t=order[i];order[i]=order[j];order[j]=t;}
	for(i=0;i<b->hot_pages;i++)*(uint64_t*)(b->mem+order[i]*b->page_size)=order[(i+1)%b->hot_pages];
	if(b->o.backing_file&&msync(b->mem,b->o.total,MS_SYNC))die("msync backing file");
	b->current=order[0];free(order);
}

static long faults(void){struct rusage r;if(getrusage(RUSAGE_SELF,&r))die("getrusage");return r.ru_majflt;}
static void perf(struct bench *b,const struct timespec *now)
{
	double dt=elapsed(&b->last,now),t=elapsed(&b->start,now);
	uint64_t n=b->chases-b->last_chases;long f=faults(),df=f-b->last_faults;
	double rate=dt?n/dt:0,ns=n?dt*1e9/n:0;
	if(b->o.csv)printf("PERF,%.3f,%.0f,%.2f,%ld\n",t,rate,ns,df);
	else printf("PERF time=%.3fs chases_per_sec=%.0f ns_per_chase=%.2f major_faults=%ld\n",t,rate,ns,df);
	fflush(stdout);b->last=*now;b->last_chases=b->chases;b->last_faults=f;
}

static void change_limit(struct options *o)
{
	char path[PATH_MAX],v[64];int fd,n;
	if(snprintf(path,sizeof(path),"%s/memory.max",o->cgroup)>=(int)sizeof(path)){errno=ENAMETOOLONG;die("cgroup path");}
	if((fd=open(path,O_WRONLY|O_CLOEXEC))<0)die("open memory.max");
	n=snprintf(v,sizeof(v),"%"PRIu64"\n",o->memory_max);
	if (write(fd, v, n) != n)
		die("write memory.max");
	close(fd);
}

static bool is_cold(struct bench *b,uint64_t p){return b->count[p]<=b->o.cold_threshold;}
static void accuracy(struct bench *b,const char *kind,uint64_t selected,uint64_t selected_cold,
	uint64_t hot,uint64_t selected_hot,uint64_t accesses,uint64_t selected_accesses)
{
	double purity=selected?100.0*selected_cold/selected:100.0;
	double error=hot?100.0*selected_hot/hot:0;
	double loss=accesses?100.0*selected_accesses/accesses:0;
	if(b->o.csv)printf("ACCURACY,%s,%"PRIu64",%.6f,%.6f,%.6f\n",kind,selected,purity,error,loss);
	else printf("ACCURACY kind=%s selected_pages=%"PRIu64" cold_placement_purity=%.6f%% hot_misplacement_rate=%.6f%% hotness_loss=%.6f%%\n",kind,selected,purity,error,loss);
}

static void eviction(struct bench *b)
{
	unsigned char *v;uint64_t i,sel=0,selcold=0,hot=0,selhot=0,acc=0,selacc=0;
	printf("EVENT action=set_memory_max bytes=%"PRIu64"\n",b->o.memory_max);fflush(stdout);
	change_limit(&b->o);if(b->o.settle)sleep(b->o.settle);
	if(!(v=malloc(b->pages)))die("mincore vector");
	if(mincore(b->mem,b->o.total,v))die("mincore");
	for(i=0;i<b->pages;i++){bool c=is_cold(b,i),gone=!(v[i]&1);acc+=b->count[i];if(!c)hot++;
		if(gone){sel++;selacc+=b->count[i];if(c)selcold++;else selhot++;}}
	accuracy(b,"evicted",sel,selcold,hot,selhot,acc,selacc);free(v);
}

static void node_list(bool *set,unsigned n,const char *spec)
{
	const char *p=spec;while(*p){char *e;unsigned long a=strtoul(p,&e,10),z=a,x;if(e==p||a>=n)goto bad;
		if(*e=='-'){p=e+1;z=strtoul(p,&e,10);if(e==p||z<a||z>=n)goto bad;}
		for (x = a; x <= z; x++)
			set[x] = true;
		if (*e == ',')
			p = e + 1;
		else if (!*e)
			break;
		else
			goto bad;
	}
	return;
bad:fprintf(stderr,"error: invalid node list: %s\n",spec);exit(1);
}

static void tiering(struct bench *b)
{
	enum{MAXN=4096};bool *fast=calloc(MAXN,1),*slow=calloc(MAXN,1);
	void **pages=malloc(b->pages*sizeof(*pages));int *nodes=malloc(b->pages*sizeof(*nodes));
	uint64_t i,sel=0,selcold=0,hot=0,selhot=0,acc=0,selacc=0,unknown=0;
	if(!fast||!slow||!pages||!nodes)die("NUMA query arrays");
	node_list(fast,MAXN,b->o.fast_nodes);node_list(slow,MAXN,b->o.slow_nodes);
	for(i=0;i<b->pages;i++)pages[i]=b->mem+i*b->page_size;
	if(syscall(SYS_move_pages,0,(unsigned long)b->pages,pages,NULL,nodes,0)<0)die("move_pages query");
	for(i=0;i<b->pages;i++){bool c=is_cold(b,i);int n=nodes[i];acc+=b->count[i];if(!c)hot++;
		if(n<0||n>=MAXN||(!fast[n]&&!slow[n])){unknown++;continue;}if(!slow[n])continue;
		sel++;selacc+=b->count[i];if(c)selcold++;else selhot++;}
	accuracy(b,"slow_node",sel,selcold,hot,selhot,acc,selacc);
	printf("PLACEMENT unknown_pages=%"PRIu64"\n",unknown);free(fast);free(slow);free(pages);free(nodes);
}

static void assess(struct bench *b)
{
	printf("EVENT action=pause_for_assessment total_chases=%"PRIu64"\n",b->chases);fflush(stdout);
	if(b->o.assess==ASSESS_EVICTION)eviction(b);else tiering(b);b->assessed=true;
}

static void run(struct bench *b)
{
	struct timespec now;uint64_t batch=0;
	clock_gettime(CLOCK_MONOTONIC,&b->start);b->last=b->start;b->last_faults=faults();
	if(b->o.csv)puts("type,time_s,chases_per_sec,ns_per_chase,major_faults");
	while(!stopped){uint64_t *node=(uint64_t*)(b->mem+b->current*b->page_size);
		if(b->count&&b->count[b->current]!=UINT32_MAX)b->count[b->current]++;
		b->current=*node;b->chases++;if(++batch<65536)continue;batch=0;
		clock_gettime(CLOCK_MONOTONIC,&now);
		if(elapsed(&b->last,&now)>=b->o.interval)perf(b,&now);
		if(!b->assessed&&b->o.assess&&elapsed(&b->start,&now)>=b->o.assess_at)assess(b);
		if(elapsed(&b->start,&now)>=b->o.duration)break;}
	clock_gettime(CLOCK_MONOTONIC,&now);if(b->chases!=b->last_chases)perf(b,&now);
}

int main(int argc,char **argv)
{
	struct bench b={0};struct sigaction sa={.sa_handler=signal_handler};
	options(argc,argv,&b.o);sigemptyset(&sa.sa_mask);sigaction(SIGINT,&sa,0);sigaction(SIGTERM,&sa,0);
	initialize(&b);
	printf("CONFIG page_size=%"PRIu64" total_bytes=%"PRIu64" hot_bytes=%"PRIu64
		" pages=%"PRIu64" hot_pages=%"PRIu64" seed=%"PRIu64"\n",
		b.page_size,b.o.total,b.o.hot,b.pages,b.hot_pages,b.o.seed);fflush(stdout);
	run(&b);munmap(b.mem,b.o.total);
	if(b.backing_fd>=0)close(b.backing_fd);
	free(b.count);return 0;
}
