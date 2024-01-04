#ifndef __TIMING_H__
#define __TIMING_H__

#include <chrono>
#include <string>
#include <fmt/core.h>

namespace convbench {

/* TODOs:
 *  1. direct log to a file;
 *  2. propose some way to extend this class in order to measure other timings
 *      a. MACRO ?
 *      b. author must inherit and extend ?
 * 
 * HOWTO:
 *  1. add class variables at the private step;
 *  2. initialize the count variables in the consturctor;
 *  3. add the functions <name>_start and <name>_update;
 *  4. add them to the restart method;
 *  5. insert a new line at the to_str method;
 */

class Timing {
public:

    Timing() : total_operation_count(0), total_conv_count(0),
               preconv_packing_count(0), conv_tiling_count(0),
               conv_packing_count(0),    conv_microkernel_count(0),
               conv_unpacking_count(0),  posconv_unpacking_count(0) {}

    ~Timing(){};

    inline void total_operation_start(){ total_operation_s = std::chrono::high_resolution_clock::now(); }
    inline void total_operation_update(bool count = true){ total_operation += (std::chrono::high_resolution_clock::now() - total_operation_s); if (count) total_operation_count++; }

    inline void total_conv_start(){ total_conv_s = std::chrono::high_resolution_clock::now(); }
    inline void total_conv_update(bool count = true){ total_conv += (std::chrono::high_resolution_clock::now() - total_conv_s); if (count) total_conv_count++; }
    
    inline void preconv_packing_start(){ preconv_packing_s = std::chrono::high_resolution_clock::now(); }
    inline void preconv_packing_update(bool count = true){ preconv_packing += (std::chrono::high_resolution_clock::now() - preconv_packing_s); if (count) preconv_packing_count++; }
    
    inline void conv_tiling_start(){ conv_tiling_s = std::chrono::high_resolution_clock::now(); }
    inline void conv_tiling_update(bool count = true){ conv_tiling += (std::chrono::high_resolution_clock::now() - conv_tiling_s); if (count) conv_tiling_count++; }
    
    inline void conv_packing_start(){ conv_packing_s = std::chrono::high_resolution_clock::now(); }
    inline void conv_packing_update(bool count = true){ conv_packing += (std::chrono::high_resolution_clock::now() - conv_packing_s); if (count) conv_packing_count++; }
    
    inline void conv_microkernel_start(){ conv_microkernel_s = std::chrono::high_resolution_clock::now(); }
    inline void conv_microkernel_update(bool count = true){ conv_microkernel += (std::chrono::high_resolution_clock::now() - conv_microkernel_s); if (count) conv_microkernel_count++; }
    
    inline void conv_unpacking_start(){ conv_unpacking_s = std::chrono::high_resolution_clock::now(); }
    inline void conv_unpacking_update(bool count = true){ conv_unpacking += (std::chrono::high_resolution_clock::now() - conv_unpacking_s); if (count) conv_unpacking_count++; }
    
    inline void posconv_unpacking_start(){ posconv_unpacking_s = std::chrono::high_resolution_clock::now(); }
    inline void posconv_unpacking_update(bool count = true){ posconv_unpacking += (std::chrono::high_resolution_clock::now() - posconv_unpacking_s); if (count) posconv_unpacking_count++; }

    inline std::string save_header(){
        return  std::string("total_operation(us),total_operation(count),") + \
                std::string("total_conv(us),total_conv(count),") + \
                std::string("preconv_packing(us),preconv_packing(count),") + \
                std::string("conv_tiling(us),conv_tiling(count),") + \
                std::string("conv_packing(us),conv_packing(count),") + \
                std::string("conv_microkernel(us),conv_microkernel(count),") + \
                std::string("conv_unpacking(us),conv_unpacking(count),") + \
                std::string("posconv_unpacking(us),posconv_unpacking(count)\n");
    }

    inline std::string save_timings(){
        return fmt::format("{:.5f},{},", total_operation.count(), total_operation_count) + \
               fmt::format("{:.5f},{},", total_conv.count(), total_conv_count) + \
               fmt::format("{:.5f},{},", preconv_packing.count(), preconv_packing_count) + \
               fmt::format("{:.5f},{},", conv_tiling.count(), conv_tiling_count) + \
               fmt::format("{:.5f},{},", conv_packing.count(), conv_packing_count) + \
               fmt::format("{:.5f},{},", conv_microkernel.count(), conv_microkernel_count) + \
               fmt::format("{:.5f},{},", conv_unpacking.count(), conv_unpacking_count) + \
               fmt::format("{:.5f},{}\n", posconv_unpacking.count(), posconv_unpacking_count);
    }

    void reset_timers(){
        total_operation     = std::chrono::duration<double, std::micro>::zero();
        total_conv          = std::chrono::duration<double, std::micro>::zero();
        preconv_packing     = std::chrono::duration<double, std::micro>::zero();
        conv_tiling         = std::chrono::duration<double, std::micro>::zero();
        conv_packing        = std::chrono::duration<double, std::micro>::zero();
        conv_microkernel    = std::chrono::duration<double, std::micro>::zero();
        conv_unpacking      = std::chrono::duration<double, std::micro>::zero();
        posconv_unpacking   = std::chrono::duration<double, std::micro>::zero();

        total_operation_count   = 0;
        total_conv_count        = 0;
        preconv_packing_count   = 0;
        conv_tiling_count       = 0;
        conv_packing_count      = 0;
        conv_microkernel_count  = 0;
        conv_unpacking_count    = 0;
        posconv_unpacking_count = 0;
    }

private:
    std::chrono::time_point< std::chrono::high_resolution_clock > total_operation_s;
    std::chrono::duration<double, std::micro> total_operation;
    double total_operation_count;

    std::chrono::time_point< std::chrono::high_resolution_clock > total_conv_s;
    std::chrono::duration<double, std::micro> total_conv;
    double total_conv_count;

    std::chrono::time_point< std::chrono::high_resolution_clock > preconv_packing_s;
    std::chrono::duration<double, std::micro> preconv_packing;
    double preconv_packing_count;

    std::chrono::time_point< std::chrono::high_resolution_clock > conv_tiling_s;
    std::chrono::duration<double, std::micro> conv_tiling;
    double conv_tiling_count;

    std::chrono::time_point< std::chrono::high_resolution_clock > conv_packing_s;
    std::chrono::duration<double, std::micro> conv_packing;
    double conv_packing_count;

    std::chrono::time_point< std::chrono::high_resolution_clock > conv_microkernel_s;
    std::chrono::duration<double, std::micro> conv_microkernel;
    double conv_microkernel_count;

    std::chrono::time_point< std::chrono::high_resolution_clock > conv_unpacking_s;
    std::chrono::duration<double, std::micro> conv_unpacking;
    double conv_unpacking_count;

    std::chrono::time_point< std::chrono::high_resolution_clock > posconv_unpacking_s;
    std::chrono::duration<double, std::micro> posconv_unpacking;
    double posconv_unpacking_count;
};

}
#endif