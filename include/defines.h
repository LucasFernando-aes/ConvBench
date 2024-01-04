#ifndef __DEFINES_H__
#define __DEFINES_H__

#ifdef CERR_DEBUG
    #define DEBUG(FUNC, MESSAGE) std::cerr << "[DEBUG " << FUNC << "] " << MESSAGE << std::endl
#else
    #define DEBUG(FUNC, MESSAGE)
#endif

#ifdef CERR_LOG
    #define LOG(FUNC, MESSAGE) std::cerr << "[LOG " << FUNC << "] " << MESSAGE << std::endl
#else
    #define LOG(FUNC, MESSAGE)
    
#endif

#ifdef TIMING
    #if TIMING > 0
        #define TIME(EXP) EXP;
    #else
        #define TIME(EXP)
    #endif
#else
    #define TIME(EXP)
    #define TIMING 0
#endif

#endif