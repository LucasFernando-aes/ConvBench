#ifndef __CONVBENCH_H__
#define __CONVBENCH_H__

#include <unordered_map>
#include <iostream>
#include <stdint.h>
#include <fstream>
#include <chrono>
#include <random>
#include <vector>
#include <string>
#include <cmath>

#include "fmt/printf.h"
#include "pystring.h"
#include "timing.h"
#include "defines.h"

namespace convbench {

//TYPEDEFS
typedef enum load { RANDOM, FOLLOW_DIST, LOAD} load_t;
typedef enum running {CORRECTNESS=0, BASELINE, DIRECT} running_t;
typedef uint64_t dim_t;
typedef int64_t index_t;
typedef std::vector<dim_t> shape_t;

//UTILS
inline dim_t multiply_shape(const shape_t shape) {
    dim_t acc = 1;
    for (auto const x: shape) 
        acc *= x;
    return acc;
}

template<typename T>
class ConvBench : public Timing {
public:

    //Constructors
    ConvBench() = delete;

    ConvBench(std::string filename) : 
            Timing::Timing(), 
            csv_filename(filename),
            convs(std::vector<std::unordered_map<std::string, dim_t>>())
        { 
            LOG("ConvBench::ConvBench", "constructor begining");
            convset_load(); 
            init_random_generator();
        }

    //Destructor
    virtual ~ConvBench() {};

    //Convolution function to be overrided
    virtual void convolution(std::vector<T> &inputs,   shape_t &input_shape,
                             std::vector<T> &outputs,  shape_t &output_shape,
                             std::vector<T> &kernels,  shape_t &kernel_shape,  shape_t &strides,   shape_t &pads,  shape_t &dilation,  dim_t &groups,
                             std::vector<T> &bias,     shape_t &bias_shape) = 0;

    virtual void convolution_baseline(std::vector<T> &inputs,   shape_t &input_shape,
                                      std::vector<T> &outputs,  shape_t &output_shape,
                                      std::vector<T> &kernels,  shape_t &kernel_shape,  shape_t &strides,   shape_t &pads,  shape_t &dilation,  dim_t &groups,
                                      std::vector<T> &bias,     shape_t &bias_shape) = 0;

    //exec function
    void convset_exec(load_t data_strategy, running_t running_strategy){
        LOG("ConvBench::convset_exec", "Begin execution of evaluation.");

        //iterate over convs
        index_t i = 1;
        for (auto conv: this->convs){
            LOG("ConvBench::convset_exec", "Executing conv " + std::to_string(i));

            //reset distribution
            LOG("ConvBench::convset_exec", "Init methods");
            TIME(reset_timers())
            init_random_generator();

            //input data
            LOG("ConvBench::convset_exec", "Data creation and initialization");
            shape_t input_shape = {conv["Ni"], conv["Ci"], conv["Hi"], conv["Wi"]};
            std::vector<T> input_data = convdata_gen(input_shape, data_strategy);

            // output_data
            shape_t output_shape = {conv["No"], conv["Do"], conv["Ho"], conv["Wo"]};
            std::vector<T> output_data = std::vector<T>(multiply_shape(output_shape), 0.);
            std::vector<T> output_baseline_data;

            // kernel data
            shape_t kernel_shape = {conv["Do"], conv["Ci"], conv["Hk"], conv["Wk"]};
            std::vector<T> kernel_data = convdata_gen(kernel_shape, data_strategy);
            shape_t padding = {conv["Hpt"], conv["Hpb"], conv["Wpl"], conv["Wpr"]};
            shape_t strides = {conv["Hs"], conv["Ws"]};
            shape_t dilation = {conv["Hd"], conv["Wd"]};
            dim_t groups = conv["G"];

            shape_t bias_shape = {conv["Do"]};
            std::vector<T> bias_data = convdata_gen(bias_shape, data_strategy);

            //exec
            T diff = 0.;
            switch (running_strategy){
                case DIRECT:
                    LOG("ConvBench::convset_exec", "exec DIRECT_CONV overrided operation.");
                    TIME(total_operation_start());
                    convolution(input_data, input_shape, 
                                output_data, output_shape, 
                                kernel_data, kernel_shape, strides, padding, dilation, groups,
                                bias_data, bias_shape);
                    TIME(total_operation_update());
                    break;

                case BASELINE:
                    LOG("ConvBench::convset_exec", "exec BASELINE conv operation.");
                    TIME(total_operation_start());
                    convolution_baseline(input_data, input_shape, 
                                         output_data, output_shape, 
                                         kernel_data, kernel_shape, strides, padding, dilation, groups,
                                         bias_data, bias_shape);
                    TIME(total_operation_update());
                    break;

                case CORRECTNESS:
                    LOG("ConvBench::convset_exec", "Verifying correctness.");
                    LOG("ConvBench::convset_exec", "exec DIRECT_CONV overrided operation.");
                    convolution(input_data, input_shape, 
                                output_data, output_shape, 
                                kernel_data, kernel_shape, strides, padding, dilation, groups,
                                bias_data, bias_shape);

                    LOG("ConvBench::convset_exec", "exec BASELINE conv operation.");
                    output_baseline_data = std::vector<T>(multiply_shape(output_shape), 0.);
                    convolution_baseline(input_data, input_shape, 
                                         output_baseline_data, output_shape, 
                                         kernel_data, kernel_shape, strides, padding, dilation, groups,
                                         bias_data, bias_shape);

                    LOG("ConvBench::convset_exec", "Assessing difference.");
                    for (int i = 0; i < output_data.size(); i++){
                        diff += abs(output_data[i] - output_baseline_data[i]);
                    }

                    fmt::printf("for conv DIFF: %f\n", diff);
                    break;

                default:
                    LOG("ConvBench::convset_exec", "FALLING BACK TO DEFAULT CASE, RUNNING_STRATEGY NOT REGOCNIZED.");
                    exit(1);
            }

            //get times
            if (TIMING and running_strategy != CORRECTNESS){
                if (i == 1)
                    fmt::print(save_header());
                fmt::print(save_timings());
            }
            
            i++;
        }

    }

    void convset_list(){
        for (const auto c : convs){
            for (const auto desc : c){
                std::cout << desc.first << ": " << desc.second << "  ";
            }
            std::cout << std::endl;
        }
    }

private:
    load_t data_strategy;
    std::string csv_filename;
    std::vector<std::unordered_map<std::string, dim_t>> convs;
    std::default_random_engine generator;
    std::normal_distribution<T> distribution;

    //Conv loader
    void convset_load(){
        std::ifstream handle = std::ifstream(this->csv_filename);
        std::string line;

        //header load
        handle >> line;
        std::vector<std::string> header = pystring::split(line, ",");

        //content load
        while (handle >> line){
            std::unordered_map<std::string, dim_t> actual_conv = std::unordered_map<std::string, dim_t>();
            std::vector<std::string> values = pystring::split(line, ",");

            for (index_t i=1; i<values.size(); i++)
                actual_conv[header[i]] = static_cast<dim_t>(std::stoi(values[i]));

            this->convs.push_back(actual_conv);
        }

        handle.close();
    }

    void init_random_generator(float avg = 0.5, float std = 0.5){
        this->distribution = std::normal_distribution<T>(avg, std);
    }

    std::vector<T> convdata_gen(shape_t data_shape, load_t data_strategy){
        std::vector<T> new_vector = std::vector<T>(multiply_shape(data_shape));
        
        switch (data_strategy){
            case LOAD:
                LOG("ConvBench::convdata_gen", "LOAD OPTION NOT YET IMPLEMENTED. FALLING BACK TO RANDOM.");
            case FOLLOW_DIST: //assumes that avg and std are already set, then just fall back to RANDOM
            case RANDOM:
                for (auto &x: new_vector){
                    x = this->distribution(this->generator);
                }
                break;
        }

        return new_vector;
    }
};

}

#endif