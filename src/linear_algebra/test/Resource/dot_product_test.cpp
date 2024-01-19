#include <iostream>
#include <fstream>
#include <ctime>
#include <sycl/sycl.hpp>

double sycl_dotp_red_vec_vec(sycl::queue defaultQueue, double *vec1_value, double *vec2_value,  size_t size_row, int r[]) 
{
      //size_t witem_size = int(size_row/512); 
      
      sycl::nd_range<1> range{r[0],r[1]};
      sycl::buffer<double> sum{1};
      auto init = sycl::property::reduction::initialize_to_identity{};
      auto chg = [&](sycl::handler& h)
      {
          
          auto reductor = sycl::reduction(sum, h, double{0.0}, std::plus<double>(), init);
          h.parallel_for(range, reductor,[=](sycl::nd_item<1> it, auto& sum) 
            {
              std::size_t idx = it.get_global_id(0);
              std::size_t size = it.get_global_range(0);
              for (std::size_t i = idx; i < size_row; i += size)
                 sum += vec1_value[i] * vec2_value[i];
            });
       }; 
       defaultQueue.submit(chg).wait();
       sycl::host_accessor sum_host{sum};
       return sum_host[0];
}

double sycl_dotp_vec_vec(sycl::queue defaultQueue, double *vec1_value, double *vec2_value,  size_t size_row, int r[]) 
{
    size_t rows = size_row;
    sycl::nd_range<1> range{r[0],r[1]};
    double *prod;
    prod = (double *)malloc(1);
    prod[0] = 0.;
    double *dotp;
    dotp = sycl::malloc_device<double>(1,defaultQueue);
    defaultQueue.memcpy(dotp,prod,(sizeof(double))).wait();
    auto chg = [&](sycl::handler &hc)
    {
        hc.parallel_for(range,[=](sycl::nd_item<1> it) 
        {
            auto product = 0.;
            std::size_t idx = it.get_global_id(0);
            std::size_t size = it.get_global_range(0);
            for (std::size_t i = idx; i < size_row; i += size)
            {    
                product = vec1_value[i] * vec2_value[i];
                auto v = sycl::atomic_ref<
                         double, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(*dotp);
                v.fetch_add(product);
            }
            
        });
    };
    defaultQueue.submit(chg).wait();
    
    defaultQueue.memcpy(prod,dotp,(sizeof(double))).wait();
    //std::cout<<" Dot product value : "<< prod[0] <<std::endl;
    return (prod[0]);
}

int main(int argc, char *argv[]) 
{
    double *vecA;
    double *vecB;
    auto functime = 0.;
    size_t vsize = 100000000;
    
    int range[2];
    if (argv[1] != NULL)
    { 
        char *a = argv[1];
        range[0] = atoi(a);
    }
    else
    {
        range[0] = 1;
    }
    if (argv[2] != NULL)
    {
        char *b = argv[2];
        range[1] = atoi(b);
    }
    else
    {
        range[1] = 1;
    }
    auto defaultQueue = sycl::queue{sycl::default_selector_v};
    
    std::cout << "Running on: "
              << defaultQueue.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    vecA = sycl::malloc_shared<double>(vsize,defaultQueue);
    vecB = sycl::malloc_shared<double>(vsize,defaultQueue);
    for (int i=0;i<vsize;i++)
    {
        vecA[i] = 0.5;
        vecB[i] = 0.5;
    }

    double product;
    auto t1 = std::chrono::high_resolution_clock::now();
    product = sycl_dotp_red_vec_vec(defaultQueue, vecA, vecB, vsize, range);
    auto t2 = std::chrono::high_resolution_clock::now();

    functime = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

    std::cout<<"Dot product = "<< product << std::endl << "Time elapsed = "<< functime << std::endl;
    /*
    t1 = std::chrono::high_resolution_clock::now();
    product = sycl_dotp_vec_vec(defaultQueue, vecA, vecB, vsize);
    t2 = std::chrono::high_resolution_clock::now();

    functime = std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();
    
    std::cout<<"Dot product = "<< product << std::endl << "Time elapsed = "<< functime << std::endl;
    */
}
