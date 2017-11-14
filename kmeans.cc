
#include <cstdlib>

#include <vector>
#include <string>
using namespace std;

#include "vec.h"
#include "helper.h"

const static int DimNum = 3000;


bool isnearer(float dist1, float dist2) {
    return dist1 > dist2;
}

struct ClusterInfo_t {
    float sum[DimNum];
    float center[DimNum];
    int count;
    float farest_distance;

    void initialize() {
        memset(sum, 0, sizeof(sum));
        farest_distance = -1;
        count = 0;
    }

    void update() {
        memset(center, 0, sizeof(center));
        if (count != 0) {
            for (int i=0; i<DimNum; ++i) {
                if (sum[i]!=0) {
                    center[i] = sum[i] / count;
                }
            }
        }
    }

    float dist(const SparseVector_t& vec) {
        float ret = 0.0f;
        for (int i=0; i<vec.v.size(); ++i) {
            ret += vec.v[i].value * center[vec.v[i].index];
        }
        return ret;
    }
};

struct Job_t {
    int id;
    size_t begin;
    size_t end;

    ClusterInfo_t* cluster_info;
};


int iteration_count;
int cluster_count;
int worker_count;

vector<SparseVector_t> all_data;
vector<string> id_list;
int* belongings;
ClusterInfo_t* cluster_info;

void* KMeansWorker(void* data) {
    Job_t& job = *(Job_t*)data;

    //LOG_NOTICE("worker [%d] : %d ~ %d", job.id, job.begin, job.end);
    for (size_t i = job.begin; i < job.end; ++i) {
        SparseVector_t &v = all_data[i];
        // random select initialize cluster if all zeros.(bad started.)
        int b = random() % cluster_count;
        float current_dist = 0.0;

        for (size_t ci=0; ci<cluster_count; ++ci) {
            float dist = cluster_info[ci].dist(v);
            if (isnearer(dist, current_dist)) {
                current_dist = dist;
                b = ci;
            }
        }

        belongings[i] = b;
        //LOG_NOTICE("belong %d- c[%d]", i, b);
        job.cluster_info[b].count += 1;
        if (job.cluster_info[b].farest_distance<0 || isnearer(job.cluster_info[b].farest_distance, current_dist)) {
            job.cluster_info[b].farest_distance = current_dist;
        }
        for (int c=0; c<v.v.size(); ++c) {
            job.cluster_info[b].sum[ v.v[c].index ] += v.v[c].value;
        }
    }
    //LOG_NOTICE("work [%d] completed.", job.id);
}

int main(int argc, const char** argv) {
    srand(time(NULL));

    if (argc!=5 && argc!=6) {
        fprintf(stderr, "Usage: \n\tkmeans <cluster_count> <iteration_count> <thread_num> <input_file> [<output_file>=/dev/stdout]\n");
        exit(-1);
    }

    cluster_count = atoi(argv[1]);
    iteration_count = atoi(argv[2]);
    worker_count = atoi(argv[3]);

    const char* filename = argv[4];
    const char* output_filename = "/dev/stdout";
    if (argc == 6) {
        output_filename = argv[5];
    }

    LOG_NOTICE("cluster_count=%d, iterator_count=%d, thread_num=%d, input=%s, output=%s",
            cluster_count, iteration_count, worker_count, filename, output_filename);

    // load data.
    LOG_NOTICE("Begin to load [%s]", filename);
    FILE* fp = fopen(filename, "r");
    char buffer[2048];
    while (fgets(buffer, sizeof(buffer), fp)) {
        
        char* valstart = strstr(buffer, "\t");
        if (!valstart) {
            continue;
        }

        *valstart = 0;
        char* val = valstart + 1;
        id_list.push_back( string(buffer) );

        vector<string> tokens;
        split(val, ",", tokens);
        SparseVector_t data;
        for (size_t i=0; i<tokens.size(); ++i) {
            char* p = strstr(tokens[i].c_str(), ":");
            if (p==NULL) {
                continue;
            }
            data.push_back(atoi(tokens[i].c_str()), atof(p+1));
        }
        all_data.push_back(data);
    }
    fclose(fp);
    LOG_NOTICE("Data load over [num = %lu]", all_data.size());
    
    belongings = new int[ all_data.size() ];
    memset(belongings, 0, sizeof(int) * all_data.size());
    cluster_info = new ClusterInfo_t[ cluster_count ];

    // random select item as center.
    for (int i=0; i<cluster_count; ++i) {
        memset(cluster_info[i].center, 0, sizeof(cluster_info[i].center));
        SparseVector_t& v = all_data[ random() % all_data.size() ];
        for (int z=0; z<v.v.size(); ++z) {
            cluster_info[i].center[ v.v[z].index ] = v.v[z].value;
        }
    }
    
    // iterations.
    size_t block_count = all_data.size() / worker_count;
    if (all_data.size() % worker_count != 0) {
        block_count += 1;
    }

    Job_t *jobs = new Job_t[worker_count];
    int current_begin = 0;
    for (int i=0; i<worker_count; ++i) {
        jobs[i].id = i;
        jobs[i].begin = current_begin;
        jobs[i].end = current_begin + block_count;
        jobs[i].cluster_info = new ClusterInfo_t[cluster_count];

        if (jobs[i].end > all_data.size()) {
            jobs[i].end = all_data.size();
        }
        current_begin += block_count;
    }

    for (size_t iter=1; iter<=iteration_count; ++iter) {
        LOG_NOTICE("Iteration [%d] begins", iter);
       
        for (int ji=0; ji<worker_count; ++ji) {
            for (int ci=0; ci<cluster_count; ++ci) {
                jobs[ji].cluster_info[ci] = cluster_info[ci];
                jobs[ji].cluster_info[ci].initialize();
            };
        }


        // multi-thread-run.
        Timer timer;
        timer.begin();
        multi_thread_jobs(KMeansWorker, jobs, worker_count, worker_count);
        timer.end();

        // post-process.
        int maximum_cluster = -1;
        int maximum_cluster_count = 0;
        int minimum_cluster = -1;
        int minimum_cluster_count = 0;
        float average_farest_distance = 0.0f;
        for (int ci=0; ci<cluster_count; ++ci) {

            cluster_info[ci].initialize();
            for (int ji=0; ji<worker_count; ++ji) {
                for (int d=0; d<DimNum; ++d) {
                    cluster_info[ci].sum[d] += jobs[ji].cluster_info[ci].sum[d];
                }
                cluster_info[ci].count += jobs[ji].cluster_info[ci].count;

                if (cluster_info[ci].farest_distance<0 || isnearer(cluster_info[ci].farest_distance, jobs[ji].cluster_info[ci].farest_distance)) {
                    cluster_info[ci].farest_distance = jobs[ji].cluster_info[ci].farest_distance;
                }
            }

            average_farest_distance += cluster_info[ci].farest_distance;
            cluster_info[ci].update();

            if (cluster_info[ci].count > maximum_cluster_count || maximum_cluster == -1) {
                maximum_cluster_count = cluster_info[ci].count;
                maximum_cluster = ci;
            }
            if (cluster_info[ci].count < minimum_cluster_count || minimum_cluster == -1) {
                minimum_cluster_count = cluster_info[ci].count;
                minimum_cluster = ci;
            }

        }
        average_farest_distance /= cluster_count;

        LOG_NOTICE("Iterator [%d] over, max=(%d, cid=%d), min=(%d, cid=%d), mthread_timer=%.3fs, average_farest_distance=%.3f", 
                iter, maximum_cluster_count, maximum_cluster,
                minimum_cluster_count, minimum_cluster,
                timer.cost_time(),
                average_farest_distance
                );
    }

    FILE* output_file = fopen(output_filename, "w");
    for (size_t i=0; i<all_data.size(); ++i) {
        fprintf(output_file, "%s\t%d\n", id_list[i].c_str(), belongings[i]);
    }
    fclose(output_file);

    return 0;
}





