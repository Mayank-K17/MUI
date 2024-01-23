/*****************************************************************************
* Multiscale Universal Interface Code Coupling Library                       *
*                                                                            *
* Copyright (C) 2023 W. Liu                                                  *
*                                                                            *
* This software is jointly licensed under the Apache License, Version 2.0    *
* and the GNU General Public License version 3, you may use it according     *
* to either.                                                                 *
*                                                                            *
* ** Apache License, version 2.0 **                                          *
*                                                                            *
* Licensed under the Apache License, Version 2.0 (the "License");            *
* you may not use this file except in compliance with the License.           *
* You may obtain a copy of the License at                                    *
*                                                                            *
* http://www.apache.org/licenses/LICENSE-2.0                                 *
*                                                                            *
* Unless required by applicable law or agreed to in writing, software        *
* distributed under the License is distributed on an "AS IS" BASIS,          *
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
* See the License for the specific language governing permissions and        *
* limitations under the License.                                             *
*                                                                            *
* ** GNU General Public License, version 3 **                                *
*                                                                            *
* This program is free software: you can redistribute it and/or modify       *
* it under the terms of the GNU General Public License as published by       *
* the Free Software Foundation, either version 3 of the License, or          *
* (at your option) any later version.                                        *
*                                                                            *
* This program is distributed in the hope that it will be useful,            *
* but WITHOUT ANY WARRANTY; without even the implied warranty of             *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
* GNU General Public License for more details.                               *
*                                                                            *
* You should have received a copy of the GNU General Public License          *
* along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
*****************************************************************************/

/**
 * @file solver_cg.h
 * @author W. Liu
 * @date 28 January 2023
 * @brief Implementation to solve problem A.x = b using the Conjugate Gradient method.
 *        Based on Baratta, Igor, Chris Richardson, and Garth Wells. "Performance 
 *        analysis of matrix-free conjugate gradient kernels using SYCL." 
 *        In International Workshop on OpenCL, pp. 1-10. 2022.
 */

#ifndef MUI_CONJUGATE_GRADIENT_H_
#define MUI_CONJUGATE_GRADIENT_H_

#include <cmath>
#include <stdio.h>

namespace mui {
namespace linalg {

// Constructor for one-dimensional Conjugate Gradient solver
template<typename ITYPE, typename VTYPE>
conjugate_gradient_1d<ITYPE, VTYPE>::conjugate_gradient_1d(sparse_matrix<ITYPE,VTYPE> A, sparse_matrix<ITYPE,VTYPE> b, VTYPE cg_solve_tol, ITYPE cg_max_iter, preconditioner<ITYPE,VTYPE>* M)
    : A_(A),
      b_(b),
      cg_solve_tol_(cg_solve_tol),
      cg_max_iter_(cg_max_iter),
      M_(M)
    {
        assert(b_.get_cols() == 1 &&
                "MUI Error [solver_cg.h]: Number of column of b matrix must be 1");
        x_.resize(A_.get_rows(),1);
        x_.sycl_resize(A_.get_rows(),1);
        x_.sycl_assign_memory(A_.get_rows());
        x_.sycl_assign_vec_memory(A_.get_rows());
        r_.resize(A_.get_rows(),1);
        r_.sycl_resize(A_.get_rows(),1);
        r_.sycl_assign_memory(A_.get_rows());
        r_.sycl_assign_vec_memory(A_.get_rows());
        z_.resize(A_.get_rows(),1);
        z_.sycl_resize(A_.get_rows(),1);
        z_.sycl_assign_memory(A_.get_rows());
        z_.sycl_assign_vec_memory(A_.get_rows());
        p_.resize(A_.get_rows(),1);
        p_.sycl_resize(A_.get_rows(),1);
        p_.sycl_assign_memory(A_.get_rows());
        p_.sycl_assign_vec_memory(A_.get_rows());
        Ax0.resize(A_.get_rows(),1);
        Ax0.sycl_resize(A_.get_rows(),1);
        Ax0.sycl_assign_memory(A_.get_rows());
        Ax0.sycl_assign_vec_memory(A_.get_rows());
        tempZ.resize(A_.get_rows(),1);
        tempZ.sycl_resize(A_.get_rows(),1);
        tempZ.sycl_assign_memory(A_.get_rows());
        tempZ.sycl_assign_vec_memory(A_.get_rows());
        Ap.resize(A_.get_rows(),1);
        Ap.sycl_resize(A_.get_rows(),1);
        Ap.sycl_assign_memory(A_.get_rows());
        Ap.sycl_assign_vec_memory(A_.get_rows());
    }

// Constructor for multidimensional Conjugate Gradient solver
template<typename ITYPE, typename VTYPE>
conjugate_gradient<ITYPE, VTYPE>::conjugate_gradient(sparse_matrix<ITYPE,VTYPE> A, sparse_matrix<ITYPE,VTYPE> b, VTYPE cg_solve_tol, ITYPE cg_max_iter, preconditioner<ITYPE,VTYPE>* M)
    : A_(A),
      b_(b),
      cg_solve_tol_(cg_solve_tol),
      cg_max_iter_(cg_max_iter),
      M_(M)
      {
        assert(A_.get_rows() == b_.get_rows() &&
                "MUI Error [solver_cg.h]: Number of rows of A matrix must be the same as the number of rows of b matrix");
        b_column_.resize(b_.get_rows(),1);
        b_column_.sycl_resize(b_.get_rows(),1);
        b_column_.sycl_assign_memory(b_.get_rows(),1);
        b_column_.sycl_assign_vec_memory(b_.get_rows());
        x_.resize(b_.get_rows(),b_.get_cols());
        x_.sycl_resize(b_.get_rows(),b_.get_cols());
        x_.sycl_assign_memory(b_.get_rows(),b_.get_cols());
        x_init_column_.resize(b_.get_rows(),1);
        x_init_column_.sycl_resize(b_.get_rows(),1);
        x_init_column_.sycl_assign_memory(b_.get_rows(),1);
        x_init_column_.sycl_assign_vec_memory(b_.get_rows());
        x_column.resize(b_.get_rows(),1);
        x_column.sycl_resize(b_.get_rows(),1);
        x_column.sycl_assign_memory(b_.get_rows(),1);
        x_column.sycl_assign_vec_memory(b_.get_rows());
      }

// Destructor for one-dimensional Conjugate Gradient solver
template<typename ITYPE, typename VTYPE>
conjugate_gradient_1d<ITYPE, VTYPE>::~conjugate_gradient_1d() {
    // Deallocate the memory for matrices
     A_.set_zero();
    A_.sycl_set_zero();
    x_.set_zero();
    x_.sycl_set_zero();
    b_.set_zero();
    b_.sycl_set_zero();
    r_.set_zero();
    r_.sycl_set_zero();
    z_.set_zero();
    z_.sycl_set_zero();
    p_.set_zero();
    p_.sycl_set_zero();
    Ax0.set_zero();
    Ax0.sycl_set_zero();
    tempZ.set_zero();
    tempZ.sycl_set_zero();
    Ap.set_zero();
    Ap.sycl_set_zero();
    // Set properties to null
    cg_solve_tol_ = 0;
    cg_max_iter_ = 0;
    // Deallocate the memory for preconditioner pointer
    if(M_!=nullptr) {
        M_ = nullptr;
        delete[] M_;
    }
}

// Destructor for multidimensional Conjugate Gradient solver
template<typename ITYPE, typename VTYPE>
conjugate_gradient<ITYPE, VTYPE>::~conjugate_gradient() {
    // Deallocate the memory for matrices
    A_.set_zero();
    A_.sycl_set_zero();
    x_.set_zero();
    x_.sycl_set_zero();
    b_.set_zero();
    b_.sycl_set_zero();
    b_column_.set_zero();
    b_column_.sycl_set_zero();
    x_init_column_.set_zero();
    x_init_column_.sycl_set_zero();
    x_column.set_zero();
    x_column.sycl_set_zero();
    // Set properties to null
    cg_solve_tol_ = 0;
    cg_max_iter_ = 0;
    // Deallocate the memory for preconditioner pointer
    if(M_!=nullptr) {
        M_ = nullptr;
        delete[] M_;
    }
}



// Member function for one-dimensional Conjugate Gradient solver to solve
template<typename ITYPE, typename VTYPE>
std::pair<ITYPE, VTYPE> conjugate_gradient_1d<ITYPE, VTYPE>::sycl_solve(sycl::queue defaultQueue, sparse_matrix<ITYPE,VTYPE> x_init) 
{
    auto functime = 0.;
    if (!x_init.empty())
    {
        assert(((x_init.get_rows() == x_.get_rows()) && (x_init.get_cols() == x_.get_cols())) &&
                "MUI Error [solver_cg.h]: Size of x_init matrix mismatch with size of x_ matrix");
        x_.copy(x_init);
        Ax0.sycl_multiply(defaultQueue, A_, x_init);
        r_.copy(b_-Ax0);
    }
    else 
    {
        r_.sycl_1d_vec_copy(defaultQueue,b_);
    }
 
    z_.sycl_1d_vec_copy(defaultQueue,r_);

    if (M_) 
    {
        M_->sycl_apply(defaultQueue,tempZ,z_);
        z_.sycl_1d_vec_copy(defaultQueue,tempZ);
        
    }
    
    p_.sycl_1d_vec_copy(defaultQueue,z_);
    x_.sycl_copy_val_vector(defaultQueue);
    auto t1 = std::chrono::high_resolution_clock::now();
    VTYPE r_norm0 = r_.sycl_dot_product(defaultQueue,z_);
    auto t2 = std::chrono::high_resolution_clock::now();
    //functime = functime + (std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count());
    
    assert(std::abs(r_norm0) >= std::numeric_limits<VTYPE>::min() &&
            "MUI Error [solver_cg.h]: Divide by zero assert for r_norm0");
    VTYPE r_norm = r_norm0;
    VTYPE r_norm_rel = std::sqrt(r_norm/r_norm0);
    ITYPE acturalKIterCount = 0;
    
    ITYPE kIter;
    if(cg_max_iter_ == 0) 
    {
        kIter = std::numeric_limits<ITYPE>::max();
    } 
    else 
    {
        kIter = cg_max_iter_;
    }

    char ch;
    
    for (ITYPE k = 0; k < kIter; ++k) 
    {
        ++acturalKIterCount;
        //t1 = std::chrono::high_resolution_clock::now();
        Ap.sycl_multiply(defaultQueue,A_,p_);
        
        VTYPE p_dot_Ap = p_.sycl_dot_product(defaultQueue,Ap);
       
       
        //t1 = std::chrono::high_resolution_clock::now();
        assert(std::abs(p_dot_Ap) >= std::numeric_limits<VTYPE>::min() &&
                "MUI Error [solver_cg.h]: Divide by zero assert for p_dot_Ap");
        
        VTYPE alpha = r_norm / p_dot_Ap;
       
        x_.sycl_add_scalar(defaultQueue, p_,alpha);
        r_.sycl_subtract_scalar(defaultQueue, Ap,alpha);
        z_.sycl_1d_vec_copy(defaultQueue,r_);
        //t2 = std::chrono::high_resolution_clock::now();
        //functime = functime + (std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count());
        
        if (M_) 
        {
            
            M_->sycl_apply(defaultQueue,tempZ,z_);
            z_.sycl_1d_vec_copy(defaultQueue,tempZ);
        }
       // t1 = std::chrono::high_resolution_clock::now();
        VTYPE updated_r_norm = r_.sycl_dot_product(defaultQueue,z_);
       // t2 = std::chrono::high_resolution_clock::now();
       // functime = functime + (std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count());
        assert(std::abs(r_norm) >= std::numeric_limits<VTYPE>::min() &&
                "MUI Error [solver_cg.h]: Divide by zero assert for r_norm");
        VTYPE beta = updated_r_norm / r_norm;
        r_norm = updated_r_norm;
        p_.set_axpby(defaultQueue, z_,1,beta,A_.get_rows());
        
        r_norm_rel = std::sqrt(r_norm/r_norm0);
        if (r_norm_rel <= cg_solve_tol_) 
        {
        //    std::cout << std::endl <<  " Updated R norm : " << updated_r_norm << "At iteration k : " << k <<std::endl;
            break;
        }
        // t2 = std::chrono::high_resolution_clock::now();
        // functime = functime + (std::chrono::duration_cast<std::chrono::microseconds>(t2-t1).count());
        
    }
    return std::make_pair(acturalKIterCount,r_norm_rel);
}

// Member function for one-dimensional Conjugate Gradient solver to solve
template<typename ITYPE, typename VTYPE>
std::pair<ITYPE, VTYPE> conjugate_gradient_1d<ITYPE, VTYPE>::solve(sparse_matrix<ITYPE,VTYPE> x_init) {
    if (!x_init.empty())
    {
        assert(((x_init.get_rows() == x_.get_rows()) && (x_init.get_cols() == x_.get_cols())) &&
                "MUI Error [solver_cg.h]: Size of x_init matrix mismatch with size of x_ matrix");
        // Initialize x_ with x_init
        x_.copy(x_init);
        // Initialise r_ with b-Ax0
        sparse_matrix<ITYPE,VTYPE> Ax0 = A_* x_init;
        r_.copy(b_-Ax0);
    }
    else 
    {
        // Initialise r_ with b
        r_.copy(b_);
    }

    // Initialise z_ with r_
    z_.copy(r_);

    if (M_) {
        sparse_matrix<ITYPE,VTYPE> tempZ(z_.get_rows(), z_.get_cols());
        tempZ = M_->apply(z_);
        z_.set_zero();
        z_.copy(tempZ);
    }

    // Initialise p_ with z_
    
    p_.copy(z_);
    VTYPE r_norm0 = r_.dot_product(z_);
    assert(std::abs(r_norm0) >= std::numeric_limits<VTYPE>::min() &&
            "MUI Error [solver_cg.h]: Divide by zero assert for r_norm0");
    VTYPE r_norm = r_norm0;
    VTYPE r_norm_rel = std::sqrt(r_norm/r_norm0);

    ITYPE kIter;
    if(cg_max_iter_ == 0) 
    {
        kIter = std::numeric_limits<ITYPE>::max();
    } 
    else 
    {
        kIter = cg_max_iter_;
    }

    ITYPE acturalKIterCount = 0;
    
    for (ITYPE k = 0; k < kIter; ++k) 
    {
        ++acturalKIterCount;
        //std::cout<< " Iteration number : "<< k <<std::endl;
        sparse_matrix<ITYPE,VTYPE> Ap = A_*p_;
        VTYPE p_dot_Ap = p_.dot_product(Ap);
        assert(std::abs(p_dot_Ap) >= std::numeric_limits<VTYPE>::min() &&
                "MUI Error [solver_cg.h]: Divide by zero assert for p_dot_Ap");
        VTYPE alpha = r_norm / p_dot_Ap;
        
        for (ITYPE j = 0; j < A_.get_rows(); ++j) {
            x_.add_scalar(j, 0, (alpha * (p_.get_value(j,0))));
            r_.subtract_scalar(j, 0, (alpha * (Ap.get_value(j,0))));
        }

        z_.set_zero();
        z_.copy(r_);

        if (M_) {
            sparse_matrix<ITYPE,VTYPE> tempZ(z_.get_rows(), z_.get_cols());
            tempZ = M_->apply(z_);
            z_.set_zero();
            z_.copy(tempZ);
        }
        /*
        std::cout<<" Now printing vector R : "<<std::endl;
        r_.print_vectors();
        std::cout<<" Now printing vector Z : "<<std::endl;
        z_.print_vectors();
        */
        VTYPE updated_r_norm = r_.dot_product(z_);
        assert(std::abs(r_norm) >= std::numeric_limits<VTYPE>::min() &&
                "MUI Error [solver_cg.h]: Divide by zero assert for r_norm");
        VTYPE beta = updated_r_norm / r_norm;
        //std::cout<< std::endl << " Beta Value : " << beta << " Updated R norm : " << updated_r_norm <<std::endl;
        r_norm = updated_r_norm;
        for (ITYPE j = 0; j < A_.get_rows(); ++j) 
        {
            p_.set_value(j, 0, (z_.get_value(j,0)+(beta*p_.get_value(j,0))));
        }
        /*
        std::cout<<" Now printing vector P : "<<std::endl;
        p_.print_vectors();
        */
        r_norm_rel = std::sqrt(r_norm/r_norm0);
        if (r_norm_rel <= cg_solve_tol_) {
            std::cout<<"Number of iterations = "<<k<<std::endl;
            break;
        }
    }
    return std::make_pair(acturalKIterCount,r_norm_rel);
}

// Member function for multidimensional Conjugate Gradient solver to solve
template<typename ITYPE, typename VTYPE>
std::pair<ITYPE, VTYPE> conjugate_gradient<ITYPE, VTYPE>::solve(sparse_matrix<ITYPE,VTYPE> x_init) 
{
    if (!x_init.empty()){
        assert(((x_init.get_rows() == b_.get_rows()) && (x_init.get_cols() == b_.get_cols())) &&
                "MUI Error [solver_cg.h]: Size of x_init matrix mismatch with size of b_ matrix");
    }
    //std::cout<< "X init size of cols = " << x_init.get_cols() << std::endl;
    std::pair<ITYPE, VTYPE> cgReturn;
    //
    // for (ITYPE j = 0; j < 1; ++j) //b_.get_cols(); ++j) 
    //for (ITYPE j = 0; j < b_.get_cols(); ++j) 
    for (ITYPE j = 0; j < 500; ++j)
    {
        b_column_.set_zero();
        b_column_.sycl_set_zero();
        b_column_ = b_.segment(0,(b_.get_rows()-1),j,j);
        conjugate_gradient_1d<ITYPE, VTYPE> cg(A_, b_column_, cg_solve_tol_, cg_max_iter_, M_);
        //std::cout<<"Entering CG solver  2 "<<std::endl;
        if (!x_init.empty()) 
        {
            x_init_column_.set_zero();
            x_init_column_.sycl_set_zero();
            x_init_column_ = x_init.segment(0,(x_init.get_rows()-1),j,j);
        //    std::cout<<"Entering CG solver 2 "<<std::endl;
        }
       // std::cout<<"Entering CG solver  3 "<<std::endl;
        
        std::pair<ITYPE, VTYPE> cgReturnTemp = cg.solve(x_init_column_);
        
        if (cgReturn.first < cgReturnTemp.first)
            cgReturn.first = cgReturnTemp.first;
        cgReturn.second += cgReturnTemp.second;
        
        sparse_matrix<ITYPE,VTYPE> x_column(b_.get_rows(),1);
       // std::cout<< j << " out of " << b_.get_cols() - 1 <<"itertions" <<std::endl;
        x_column = cg.getSolution();
        
        for (ITYPE i = 0; i < x_column.get_rows(); ++i) 
        {
            x_.set_value(i, j, x_column.get_value(i,0));
        }
    }
    cgReturn.second /= b_.get_cols();
    return std::make_pair(0,0.);//cgReturn;
}

template<typename ITYPE, typename VTYPE>
std::pair<ITYPE, VTYPE> conjugate_gradient<ITYPE, VTYPE>::sycl_solve(sparse_matrix<ITYPE,VTYPE> x_init) 
{
    if (!x_init.empty())
    {
        assert(((x_init.get_rows() == b_.get_rows()) && (x_init.get_cols() == b_.get_cols())) &&
                "MUI Error [solver_cg.h]: Size of x_init matrix mismatch with size of b_ matrix");
    }
   
    auto defaultQueue = sycl::queue {sycl::default_selector_v};
    auto functime = 0.;
    std::pair<ITYPE, VTYPE> cgReturn;
    
    std::cout << "Running on: "
              << defaultQueue.get_device().get_info<sycl::info::device::name>()
              << std::endl;
    for (ITYPE j = 0; j <b_.get_cols(); ++j) 
    {
       
        b_column_.sycl_segment_row(defaultQueue, b_, j);
   
        conjugate_gradient_1d<ITYPE, VTYPE> cg(A_, b_column_, cg_solve_tol_, cg_max_iter_, M_);
       
        if (!x_init.empty()) 
        {
            x_init_column_.sycl_segment_row(defaultQueue, x_init, j);
        }
        if (j%100 == 0 || (j == (b_.get_cols()-1)))
            std::cout<< j << " out of " << b_.get_cols() - 1 <<"iteartions" <<std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        std::pair<ITYPE, VTYPE> cgReturnTemp = cg.sycl_solve(defaultQueue,x_init_column_);
        auto t2 = std::chrono::high_resolution_clock::now();

        functime = functime + std::chrono::duration_cast<std::chrono::milliseconds>(t2-t1).count();

        if (cgReturn.first < cgReturnTemp.first)
            cgReturn.first = cgReturnTemp.first;
        cgReturn.second += cgReturnTemp.second;
        
        //sparse_matrix<ITYPE,VTYPE> x_column(b_.get_rows(),1);
        //x_column.sycl_assign_vec_memory(b_.get_rows());
        //std::cout<< j << " out of " << b_.get_cols() - 1 <<"iteartions" <<std::endl;
        //x_column = cg.getSolution();
        //cg.copy_vecSolution(defaultQueue, x_column );
        //for (ITYPE i = 0; i < x_column.get_rows(); ++i) 
        //{
        //      x_.set_value(i, j, x_column.get_sycl_vec_value(i));
        //}
    }
    cgReturn.second /= b_.get_cols();
    std::cout << "Total residuals = " << (cgReturn.second) <<std::endl;
    std::cout << "Total CG function time = " << (functime/1000) <<std::endl;
    return cgReturn;
}

// Member function for one-dimensional Conjugate Gradient solver to get the solution
template<typename ITYPE, typename VTYPE>
sparse_matrix<ITYPE,VTYPE> conjugate_gradient_1d<ITYPE, VTYPE>::getSolution() {
    return x_;
}
template<typename ITYPE, typename VTYPE>
void conjugate_gradient_1d<ITYPE, VTYPE>::copy_vecSolution(sycl::queue defaultQueue, sparse_matrix<ITYPE,VTYPE> &CopyMat ) 
{
    CopyMat.sycl_1d_vec_copy(defaultQueue, x_);
}
// Member function for multidimensional Conjugate Gradient solver to get the solution
template<typename ITYPE, typename VTYPE>
sparse_matrix<ITYPE,VTYPE> conjugate_gradient<ITYPE, VTYPE>::getSolution() {
    return x_;
}

} // linalg
} // mui

#endif /* MUI_CONJUGATE_GRADIENT_H_ */
