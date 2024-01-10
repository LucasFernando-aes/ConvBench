#include <string>
#include <iostream>
#include "convbench.h"

#include "tmp/INCLUDES.tmp"

#include "tmp/DEFINES.tmp"

namespace convbench {

template<typename T>
class Convolution : public ConvBench<T> {

public:

    #include "tmp/CONSTRUCTOR.tmp"

    ~Convolution() {};

private:

    //architecture infos
    dim_t L1_size;
    dim_t L1_latency;
    float L1_alpha;
    dim_t L2_size;
    dim_t L2_latency;
    float L2_beta;
    dim_t L3_size;
    dim_t L3_latency;
    float L3_gamma;
    dim_t CACHE_block_size;
    dim_t MEM_latency;

    #include "tmp/CONVOLUTION.tmp"

}; //Convolution class

} //convbench namespace

int main(int argc, char *argv[]){
    
    if (argc != 4) {
        std::cout << "This program expects three command line arguments:\n" << \
                     "\t1. CSV convolution operation set.\n" << \
                     "\t2. Data generation strategy (random|follow_dist|load).\n" << \
                     "\t3. Execution strategy (correctness|direct|baseline).\n";
        return 1;
    }

    LOG("main", "program begin");
    LOG("main", "Convolution obj instantiation");
    convbench::Convolution<TYPE> bench = convbench::Convolution<TYPE>(std::string(argv[1]));

    convbench::load_t data_strategy;
    if (std::string(argv[2]) == "random"){
        LOG("main", "data generation strategy : RANDOM");
        data_strategy = convbench::RANDOM;
    } else if (std::string(argv[2]) == "follow_dist") {
        LOG("main", "data generation strategy : FOLLOW_DIST");
        data_strategy = convbench::FOLLOW_DIST;
    } else if (std::string(argv[2]) == "LOAD") {
        LOG("main", "data generation strategy : LOAD");
        data_strategy = convbench::LOAD;
    } else {
        LOG("main", "OPTION NOT RECOGNIZED. FALLING BACK TO RANDOM STRATEGY");
        data_strategy = convbench::RANDOM;
    }

    convbench::running_t running_strategy;
    if (std::string(argv[3]) == "correctness"){
        LOG("main", "running strategy : CORRECTNESS");
        running_strategy = convbench::CORRECTNESS;
    } else if (std::string(argv[3]) == "direct") {
        LOG("main", "running strategy : DIRECT");
        running_strategy = convbench::DIRECT;
    } else if (std::string(argv[3]) == "baseline") {
        LOG("main", "running strategy : BASELINE");
        running_strategy = convbench::BASELINE;
    } else {
        LOG("main", "OPTION NOT RECOGNIZED. FALLING BACK TO DIRECT STRATEGY");
        running_strategy = convbench::DIRECT;
    }

    LOG("main", "obj exec() method calling.");
    bench.convset_exec(data_strategy, running_strategy);

    LOG("main", "program end");
    return 0;
}