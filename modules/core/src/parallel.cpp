/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "precomp.hpp"

#if !defined HAVE_TBB && !defined HAVE_OPENMP && !defined HAVE_GCD && !defined HAVE_CONCURRENCY && !defined HAVE_CSTRIPES
    #ifdef __APPLE__
        #define HAVE_GCD
    #elif defined _MSC_VER && _MSC_VER >= 1600
        #define HAVE_CONCURRENCY
    #endif
#endif

#ifdef HAVE_CONCURRENCY
    #include <ppl.h>
#elif defined HAVE_OPENMP
    #include <omp.h>
#elif defined HAVE_GCD
    #include <dispatch/dispatch.h>
#elif defined HAVE_TBB
    #include "tbb/tbb_stddef.h"
    #if TBB_VERSION_MAJOR*100 + TBB_VERSION_MINOR >= 202
        #include "tbb/tbb.h"
        #include "tbb/task.h"
        #undef min
        #undef max
    #else
        #undef HAVE_TBB
    #endif // end TBB version
#elif defined HAVE_CSTRIPES
    #include "C=.h"
    #undef shared
#endif

/*
    HAVE_TBB - using TBB
    HAVE_GCD - using GCD
    HAVE_OPENMP - using OpenMP
    HAVE_CONCURRENCY - using visual studio 2010 concurrency
*/

namespace cv
{
    class ParallelLoopBodyWrapper
    {
    public:
        ParallelLoopBodyWrapper(const ParallelLoopBody& _body, const Range& _r, double _nstripes)
        {
            body = &_body;
            wholeRange = _r;
            double len = wholeRange.end - wholeRange.start;
            nstripes = cvRound(_nstripes < 0 ? len : MIN(MAX(_nstripes, 1.), len));
        }
        void operator()(const Range& sr) const
        {
            Range r;
            r.start = (int)(wholeRange.start +
                            ((size_t)sr.start*(wholeRange.end - wholeRange.start) + nstripes/2)/nstripes);
            r.end = sr.end >= nstripes ? wholeRange.end : (int)(wholeRange.start +
                            ((size_t)sr.end*(wholeRange.end - wholeRange.start) + nstripes/2)/nstripes);
            (*body)(r);
        }
        Range stripeRange() const { return Range(0, nstripes); }

    protected:
        const ParallelLoopBody* body;
        Range wholeRange;
        int nstripes;
    };

    ParallelLoopBody::~ParallelLoopBody() {}

#if defined HAVE_TBB
    class ProxyLoopBody : public ParallelLoopBodyWrapper
    {
    public:
        ProxyLoopBody(const ParallelLoopBody& _body, const Range& _r, double _nstripes)
        : ParallelLoopBodyWrapper(_body, _r, _nstripes)
        {}

        void operator ()(const tbb::blocked_range<int>& range) const
        {
            this->ParallelLoopBodyWrapper::operator()(Range(range.begin(), range.end()));
        }
    };
#elif defined HAVE_GCD

    typedef ParallelLoopBodyWrapper ProxyLoopBody;
    static
    void block_function(void* context, size_t index)
    {
        ProxyLoopBody* ptr_body = static_cast<ProxyLoopBody*>(context);
        (*ptr_body)(Range(index, index + 1));
    }
#elif defined HAVE_CONCURRENCY
    class ProxyLoopBody : public ParallelLoopBodyWrapper
    {
    public:
        ProxyLoopBody(const ParallelLoopBody& _body, const Range& _r, double _nstripes)
        : ParallelLoopBodyWrapper(_body, _r, _nstripes)
        {}

        void operator ()(int i) const
        {
            this->ParallelLoopBodyWrapper::operator()(Range(i, i + 1));
        }
    };
#else
    typedef ParallelLoopBodyWrapper ProxyLoopBody;
#endif

    void parallel_for_(const Range& range, const ParallelLoopBody& body, double nstripes)
    {
        ProxyLoopBody pbody(body, range, nstripes);
        Range stripeRange = pbody.stripeRange();

#if defined HAVE_TBB

        tbb::parallel_for(tbb::blocked_range<int>(stripeRange.start, stripeRange.end), pbody);

#elif defined HAVE_CONCURRENCY

        Concurrency::parallel_for(stripeRange.start, stripeRange.end, pbody);

#elif defined HAVE_OPENMP

#pragma omp parallel for schedule(dynamic)
        for (int i = stripeRange.start; i < stripeRange.end; ++i)
            pbody(Range(i, i + 1));

#elif defined HAVE_GCD

        dispatch_queue_t concurrent_queue = dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
        dispatch_apply_f(stripeRange.end - stripeRange.start, concurrent_queue, &pbody, block_function);

#elif defined HAVE_CSTRIPES

        parallel()
        {
            int offset = stripeRange.start;
            int len = stripeRange.end - offset;
            Range r(offset + CPX_RANGE_START(len), offset + CPX_RANGE_END(len));
            pbody(r);
            barrier();
        }

#else

        pbody(stripeRange);

#endif
    }

} // namespace cv


static int numThreads = 0;
static int numProcs   = 0;

int cv::getNumThreads(void)
{
    if( !numProcs )
        setNumThreads(0);
    return numThreads;
}

void cv::setNumThreads( int
#ifdef _OPENMP
                             threads
#endif
                  )
{
    if( !numProcs )
    {
#ifdef _OPENMP
        numProcs = omp_get_num_procs();
#else
        numProcs = 1;
#endif
    }

#ifdef _OPENMP
    if( threads <= 0 )
        threads = numProcs;
    else
        threads = MIN( threads, numProcs );

    numThreads = threads;
#else
    numThreads = 1;
#endif
}


int cv::getThreadNum(void)
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

#ifdef ANDROID
static inline int getNumberOfCPUsImpl()
{
   FILE* cpuPossible = fopen("/sys/devices/system/cpu/possible", "r");
   if(!cpuPossible)
       return 1;

   char buf[2000]; //big enough for 1000 CPUs in worst possible configuration
   char* pbuf = fgets(buf, sizeof(buf), cpuPossible);
   fclose(cpuPossible);
   if(!pbuf)
      return 1;

   //parse string of form "0-1,3,5-7,10,13-15"
   int cpusAvailable = 0;

   while(*pbuf)
   {
      const char* pos = pbuf;
      bool range = false;
      while(*pbuf && *pbuf != ',')
      {
          if(*pbuf == '-') range = true;
          ++pbuf;
      }
      if(*pbuf) *pbuf++ = 0;
      if(!range)
        ++cpusAvailable;
      else
      {
          int rstart = 0, rend = 0;
          sscanf(pos, "%d-%d", &rstart, &rend);
          cpusAvailable += rend - rstart + 1;
      }

   }
   return cpusAvailable ? cpusAvailable : 1;
}
#endif

int cv::getNumberOfCPUs(void)
{
#if defined WIN32 || defined _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo( &sysinfo );

    return (int)sysinfo.dwNumberOfProcessors;
#elif defined ANDROID
    static int ncpus = getNumberOfCPUsImpl();
    return ncpus;
#elif defined __linux__
    return (int)sysconf( _SC_NPROCESSORS_ONLN );
#elif defined __APPLE__
    int numCPU=0;
    int mib[4];
    size_t len = sizeof(numCPU);

    /* set the mib for hw.ncpu */
    mib[0] = CTL_HW;
    mib[1] = HW_AVAILCPU;  // alternatively, try HW_NCPU;

    /* get the number of CPUs from the system */
    sysctl(mib, 2, &numCPU, &len, NULL, 0);

    if( numCPU < 1 )
    {
        mib[1] = HW_NCPU;
        sysctl( mib, 2, &numCPU, &len, NULL, 0 );

        if( numCPU < 1 )
            numCPU = 1;
    }

    return (int)numCPU;
#else
    return 1;
#endif
}

CV_IMPL void cvSetNumThreads(int nt)
{
    cv::setNumThreads(nt);
}

CV_IMPL int cvGetNumThreads()
{
    return cv::getNumThreads();
}

CV_IMPL int cvGetThreadNum()
{
    return cv::getThreadNum();
}