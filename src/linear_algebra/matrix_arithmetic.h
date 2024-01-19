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
 * @file matrix_arithmetic.h
 * @author W. Liu
 * @date 01 February 2023
 * @brief Implementation of sparse matrix arithmetic operations.
 */

#ifndef MUI_MATRIX_ARITHMETIC_H_
#define MUI_MATRIX_ARITHMETIC_H_

#include <cassert>
#include <math.h>
#include <bits/stdc++.h>
#include <CL/sycl.hpp>
#define MAX 1000000
namespace mui {
namespace linalg {

// **************************************************
// ************ Public member functions *************
// **************************************************

// Overload addition operator to perform sparse matrix addition
template<typename ITYPE, typename VTYPE>
sparse_matrix<ITYPE,VTYPE> sparse_matrix<ITYPE,VTYPE>::operator+(sparse_matrix<ITYPE,VTYPE> &addend)
{

    if (rows_ != addend.rows_ || cols_ != addend.cols_) 
    {
        std::cerr << "MUI Error [matrix_arithmetic.h]: matrix size mismatch during matrix addition" << std::endl;
        std::abort();
    }

    if (addend.matrix_format_ != matrix_format_) 
    {
        addend.format_conversion(this->get_format(), true, true, "overwrite");
    } 
    else 
    {
        if (!addend.is_sorted_unique("matrix_arithmetic.h", "operator+()")){
            if (addend.matrix_format_ == format::COO) {
                addend.sort_coo(true, true, "overwrite");
            } else if (addend.matrix_format_ == format::CSR) {
                addend.sort_csr(true, "overwrite");
            } else if (addend.matrix_format_ == format::CSC) {
                addend.sort_csc(true, "overwrite");
            } else {
                std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised addend matrix format for matrix operator+()" << std::endl;
                std::cerr << "    Please set the addend matrix_format_ as:" << std::endl;
                std::cerr << "    format::COO: COOrdinate format" << std::endl;
                std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
                std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
                std::abort();
            }
        }
    }

    if (!this->is_sorted_unique("matrix_arithmetic.h", "operator+()"))
    {
        if (matrix_format_ == format::COO) 
        {
            this->sort_coo(true, true, "overwrite");
        } 
        else if (matrix_format_ == format::CSR) 
        {
            this->sort_csr(true, "overwrite");
        }
        else if (matrix_format_ == format::CSC) 
        {
            this->sort_csc(true, "overwrite");
        }
        else 
        {
            std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised matrix format for matrix operator+()" << std::endl;
            std::cerr << "    Please set the matrix_format_ as:" << std::endl;
            std::cerr << "    format::COO: COOrdinate format" << std::endl;
            std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
            std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
            std::abort();
        }
    }

    // Create a new sparse matrix object for the result
    sparse_matrix<ITYPE,VTYPE> res(rows_, cols_, this->get_format());

    if (matrix_format_ == format::COO) 
    {

        // Perform element-wise addition of the COO vectors
        res.matrix_coo.values_.reserve(matrix_coo.values_.size() + addend.matrix_coo.values_.size());
        res.matrix_coo.row_indices_.reserve(matrix_coo.row_indices_.size() + addend.matrix_coo.row_indices_.size());
        res.matrix_coo.col_indices_.reserve(matrix_coo.col_indices_.size() + addend.matrix_coo.col_indices_.size());

        // Insert the COO vectors of the initial sparse matrix to the result sparse matrix
        res.matrix_coo.values_ = std::vector<VTYPE>(matrix_coo.values_.begin(), matrix_coo.values_.end());
        res.matrix_coo.row_indices_ = std::vector<ITYPE>(matrix_coo.row_indices_.begin(), matrix_coo.row_indices_.end());
        res.matrix_coo.col_indices_ = std::vector<ITYPE>(matrix_coo.col_indices_.begin(), matrix_coo.col_indices_.end());

        // Append the addend COO vectors to the result sparse matrix
        res.matrix_coo.values_.insert(res.matrix_coo.values_.end(), addend.matrix_coo.values_.begin(), addend.matrix_coo.values_.end());
        res.matrix_coo.row_indices_.insert(res.matrix_coo.row_indices_.end(), addend.matrix_coo.row_indices_.begin(), addend.matrix_coo.row_indices_.end());
        res.matrix_coo.col_indices_.insert(res.matrix_coo.col_indices_.end(), addend.matrix_coo.col_indices_.begin(), addend.matrix_coo.col_indices_.end());

        // Sort and deduplicate the result
        res.sort_coo(true, true, "plus");
        res.nnz_ = res.matrix_coo.values_.size();

    } 
    else if (matrix_format_ == format::CSR) 
    {
        // Perform element-wise addition of the CSR vectors
        res.matrix_csr.values_.reserve(matrix_csr.values_.size() + addend.matrix_csr.values_.size());
        res.matrix_csr.row_ptrs_.resize(rows_ + 1);
        res.matrix_csr.col_indices_.reserve(matrix_csr.col_indices_.size() + addend.matrix_csr.col_indices_.size());  

        auto defaultQueue = sycl::queue {sycl::default_selector_v};;

        double *d_matrix_values;
        double *d_addend_values;
        double *d_res_values;

        int *d_matrix_col;
        int *d_addend_col;
        int *d_res_col;
   
        int *d_matrix_row;
        int *d_addend_row;
        int *d_res_row;
        int *d_row_count;
        
        size_t size_matrix = matrix_csr.values_.size() ;
        size_t size_addend = addend.matrix_csr.values_.size();
        size_t size_res = (matrix_csr.values_.size() + addend.matrix_csr.values_.size());

        size_t size_rows = (rows_ + 1);

        size_t size_matrix_col = matrix_csr.col_indices_.size() ;
        size_t size_addend_col = addend.matrix_csr.col_indices_.size() ;
        size_t size_res_col;// = (matrix_csr.col_indices_.size()+addend.matrix_csr.col_indices_.size());

        d_matrix_values = sycl::malloc_shared<double>(size_matrix,defaultQueue);
        d_addend_values = sycl::malloc_shared<double>(size_addend,defaultQueue);
         
        d_matrix_col    = sycl::malloc_shared<int>(size_matrix_col,defaultQueue);
        d_addend_col    = sycl::malloc_shared<int>(size_addend_col,defaultQueue);
       
        d_matrix_row    = sycl::malloc_shared<int>(size_rows,defaultQueue);
        d_addend_row    = sycl::malloc_shared<int>(size_rows,defaultQueue);
        d_row_count     = sycl::malloc_shared<int>(rows_,defaultQueue);
        d_res_row       = sycl::malloc_shared<int>(size_rows,defaultQueue);

        for (int i=0;i<size_matrix;i++)
        {
            d_matrix_values[i] = matrix_csr.values_.at(i);
            d_matrix_col[i]    = matrix_csr.col_indices_.at(i);
        
        }
        for (int i=0;i<size_addend;i++)
        {
            d_addend_values[i] = addend.matrix_csr.values_.at(i);
            d_addend_col[i] = addend.matrix_csr.col_indices_.at(i);
        }

        for (int i=0;i<=rows_;i++)
        {
            d_matrix_row[i] = matrix_csr.row_ptrs_.at(i);
            d_addend_row[i] = addend.matrix_csr.row_ptrs_.at(i);
           
        }

        //auto defaultQueue = sycl::queue{sycl::default_selector_v};
        /*
        defaultQueue.memcpy(d_matrix_values,h_matrix_values,(size_matrix*sizeof(double))).wait();
        defaultQueue.memcpy(d_addend_values,h_addend_values,(size_addend*sizeof(double))).wait();
        defaultQueue.memcpy(d_matrix_col,h_matrix_col,(size_matrix*sizeof(int))).wait();
        defaultQueue.memcpy(d_addend_col,h_addend_col,(size_addend*sizeof(int))).wait();
        defaultQueue.memcpy(d_matrix_row,h_matrix_row,((rows_+1)*sizeof(int))).wait();
        defaultQueue.memcpy(d_addend_row,h_addend_row,((rows_+1)*sizeof(int))).wait();
        */

        auto cg = [&](sycl::handler &h) 
        {
            
            h.parallel_for(sycl::range(rows_),[=](sycl::id<1> idx) 
            {
                auto startIdx = d_matrix_row[idx];
                auto endIdx = d_matrix_row[idx+1];
                auto startaddIdx = d_addend_row[idx];
                auto endaddIdx = d_addend_row[idx+1];
                auto i = startIdx;
                auto sum = 0.;
                auto j = startaddIdx;
                auto count = 0;
                auto col = 0;
                auto addend_col = 0;
                i = startIdx;
                while (i < endIdx && j < endaddIdx) 
                {
                    col = d_matrix_col[i];
                    addend_col = d_addend_col[j];
                    if (col == addend_col)
                    {
                        sum = d_matrix_values[i] + d_addend_values[j];
                        i++;
                        j++;
                     
                    }
                    else if(col < addend_col)
                    {
                        sum = d_matrix_values[i];
                        i++;
                    
                    }
                    else
                    {
                        sum = d_addend_values[j];
                        j++;
                        
                    }

                    if (std::abs(sum) >= std::numeric_limits<VTYPE>::min())
                    {
                        count++;
                    }
                }
                for (;i<endIdx;i++)
                {
                    sum = d_matrix_values[i];
                    if (std::abs(sum) >=  std::numeric_limits<VTYPE>::min())
                    {
                        count++;
                    }
                }
                for(;j<endaddIdx;j++)
                {
                    sum = d_addend_values[j];
                    if (std::abs(sum) >=  std::numeric_limits<VTYPE>::min())
                    {
                        count++;
                    }
                }
                d_row_count[idx] = count;
            });
        };        
        defaultQueue.submit(cg).wait(); 
        
        auto chg = [&](sycl::handler &gh) 
        {
            gh.parallel_for(sycl::range(size_rows),[=](sycl::id<1> idx) 
            {
                d_res_row[idx] = 0;
                auto count =0;
                if (idx > 0)
                {
                    while (count < idx)
                    {   
                        d_res_row[idx] = d_res_row[idx]+d_row_count[count];
                        count++;
                    }
                } 
            });
        };        
        defaultQueue.submit(chg).wait(); 
        
       // defaultQueue.memcpy(h_res_row,d_res_row,(size_rows*sizeof(int))).wait();
        size_res_col = d_res_row[rows_];

        //h_res_values    = (double *)malloc(size_res_col* sizeof(double));
        //h_res_col       = (int *)malloc(size_res_col*sizeof(int));

        d_res_values    = sycl::malloc_shared<double>(size_res_col,defaultQueue);
        d_res_col       = sycl::malloc_shared<int>(size_res_col,defaultQueue);
        
        auto cag = [&](sycl::handler &ga)
        {
            ga.parallel_for(sycl::range(rows_),[=](sycl::id<1>idx)
            {
                auto startIdx = d_matrix_row[idx];
                auto endIdx = d_matrix_row[idx+1];
                auto startaddIdx = d_addend_row[idx];
                auto endaddIdx = d_addend_row[idx+1];
                auto i = startIdx;
                auto j = startaddIdx;
                auto count = 0;
                auto col = 0;
                auto addend_col = 0;
                auto placeHold = d_res_row[idx];
                while (i < endIdx && j < endaddIdx) 
                {
                    col        = d_matrix_col[i];
                    addend_col = d_addend_col[j];
                    if (col == addend_col)
                    {
                        if (std::abs(d_matrix_values[i] + d_addend_values[j]) >= std::numeric_limits<VTYPE>::min())
                        {                            
                            d_res_values[placeHold + count] = d_matrix_values[i] + d_addend_values[j];
                            d_res_col[placeHold + count] = col;
                            count++;
                        }
                        i++;
                        j++;    
                    }
                    else if (col < addend_col)
                    {
                        if (std::abs(d_matrix_values[i]) >= std::numeric_limits<VTYPE>::min())
                        { 
                            d_res_values[placeHold + count] = d_matrix_values[i];
                            d_res_col[placeHold + count] = col;
                            count++;
                        }
                        i++;    
                    }
                    else
                    {
                        if (std::abs(d_addend_values[j]) >= std::numeric_limits<VTYPE>::min())
                        { 
                            d_res_values[placeHold + count] = d_addend_values[j];
                            d_res_col[placeHold + count] = addend_col;
                            count++;
                        }
                        j++;   
                    }
                }
                for (;i<endIdx;i++)
                {
                    col        = d_matrix_col[i];
                    if (std::abs(d_matrix_values[i]) >= std::numeric_limits<VTYPE>::min())
                    {
                        d_res_values[placeHold + count] = d_matrix_values[i];
                        d_res_col[placeHold + count] = col;
                        count++;
                    }
                }
                for (;j<endaddIdx;j++)
                {
                    addend_col = d_addend_col[j];
                    if (std::abs(d_addend_values[j]) >= std::numeric_limits<VTYPE>::min())
                    {
                        d_res_values[placeHold + count] = d_addend_values[j];
                        d_res_col[placeHold + count] = addend_col;
                        count++;
                    }
                }
            });
        };
        defaultQueue.submit(cag).wait(); 

       //defaultQueue.memcpy(h_res_values,d_res_values,(size_res_col*sizeof(double))).wait();
       //defaultQueue.memcpy(h_res_col,d_res_col,(size_res_col*sizeof(int))).wait();
       res.matrix_sycl.values = sycl::malloc_shared<VTYPE>(size_res_col,defaultQueue);
       res.matrix_sycl.column = sycl::malloc_shared<ITYPE>(size_res_col,defaultQueue);
       res.matrix_sycl.row    = sycl::malloc_shared<ITYPE>((rows_+1),defaultQueue);
       copy_sycl_data(defaultQueue,res.matrix_sycl.values,d_res_values,size_res_col);
       copy_sycl_data(defaultQueue,res.matrix_sycl.column,d_res_col,size_res_col);
       copy_sycl_data(defaultQueue,res.matrix_sycl.row,d_res_row,(rows_+1));
       // defaultQueue.memcpy(h_res_values,d_res_values,(size_res*sizeof(double))).wait();             
       // res.matrix_csr.values_.resize(size_res);
       // res.matrix_csr.values_.resize(size_res);
       sycl::free(d_res_values,defaultQueue);
       sycl::free(d_res_col,defaultQueue);
       sycl::free(d_res_row,defaultQueue);
        for (int i=0;i<size_res_col;i++)
        {
            
            res.matrix_csr.values_.emplace_back(res.matrix_sycl.values[i]);
            res.matrix_csr.col_indices_.emplace_back(res.matrix_sycl.column[i]);
        }
        for (int i=0;i<=rows_;i++)
        {
            res.matrix_csr.row_ptrs_[i] = (res.matrix_sycl.row[i]);
        }
        res.nnz_ = res.matrix_csr.col_indices_.size();
        res.matrix_csr.row_ptrs_[rows_ + 1] = res.nnz_;
    } 


    else if (matrix_format_ == format::CSC) {

        // Perform element-wise addition of the CSC vectors
        res.matrix_csc.values_.reserve(matrix_csc.values_.size() + addend.matrix_csc.values_.size());
        res.matrix_csc.row_indices_.reserve(matrix_csc.row_indices_.size() + addend.matrix_csc.row_indices_.size());
        res.matrix_csc.col_ptrs_.resize(cols_ + 1);   
        
        ITYPE column = 0;
        while (column < cols_) {
            ITYPE start = matrix_csc.col_ptrs_[column];
            ITYPE end = matrix_csc.col_ptrs_[column + 1];

            ITYPE addend_start = addend.matrix_csc.col_ptrs_[column];
            ITYPE addend_end = addend.matrix_csc.col_ptrs_[column + 1];

            res.matrix_csc.col_ptrs_[0] = 0;

            // Merge the values and row indices of the two columns
            ITYPE i = start;
            ITYPE j = addend_start;
            while (i < end && j < addend_end) {
                ITYPE row = matrix_csc.row_indices_[i];
                ITYPE addend_row = addend.matrix_csc.row_indices_[j];

                if (row == addend_row) 
                {
                    // Add the corresponding values if the columns match
                    if (std::abs(matrix_csc.values_[i] + addend.matrix_csc.values_[j]) >= std::numeric_limits<VTYPE>::min()) {
                        res.matrix_csc.values_.emplace_back(matrix_csc.values_[i] + addend.matrix_csc.values_[j]);
                        res.matrix_csc.row_indices_.emplace_back(row);
                    }
                    i++;
                    j++;
                } 
                else if (row < addend_row) {
                    // Add the current value from the initial matrix
                    if (std::abs(matrix_csc.values_[i]) >= std::numeric_limits<VTYPE>::min()) 
                    {
                        res.matrix_csc.values_.emplace_back(matrix_csc.values_[i]);
                        res.matrix_csc.row_indices_.emplace_back(row);
                    }
                    i++;
                } else {
                    // Add the current value from the addend matrix
                    if (std::abs(addend.matrix_csc.values_[j]) >= std::numeric_limits<VTYPE>::min()) {
                        res.matrix_csc.values_.emplace_back(addend.matrix_csc.values_[j]);
                        res.matrix_csc.row_indices_.emplace_back(addend_row);
                    }
                    j++;
                }
            }

            // Add any remaining elements from the initial matrix
            for (; i < end; i++) {
                if (std::abs(matrix_csc.values_[i]) >= std::numeric_limits<VTYPE>::min()){
                    res.matrix_csc.values_.emplace_back(matrix_csc.values_[i]);
                    res.matrix_csc.row_indices_.emplace_back(matrix_csc.row_indices_[i]);
                }
            }

            // Add any remaining elements from the addend matrix
            for (; j < addend_end; j++) {
                if (std::abs(addend.matrix_csc.values_[j]) >= std::numeric_limits<VTYPE>::min()){
                    res.matrix_csc.values_.emplace_back(addend.matrix_csc.values_[j]);
                    res.matrix_csc.row_indices_.emplace_back(addend.matrix_csc.row_indices_[j]);
                }
            }

            // Update the column pointer
            res.nnz_ = res.matrix_csc.row_indices_.size();
            res.matrix_csc.col_ptrs_[column + 1] = res.nnz_;

            column++;
        }

    } else {
        std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised matrix format for matrix operator+()" << std::endl;
        std::cerr << "    Please set the matrix_format_ as:" << std::endl;
        std::cerr << "    format::COO: COOrdinate format" << std::endl;
        std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
        std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
        std::abort();
    }
    
    return res;
}

// Overload subtraction operator to perform sparse matrix subtraction
template<typename ITYPE, typename VTYPE>
sparse_matrix<ITYPE,VTYPE> sparse_matrix<ITYPE,VTYPE>::operator-(sparse_matrix<ITYPE,VTYPE> &subtrahend) {
   if (rows_ != subtrahend.rows_ || cols_ != subtrahend.cols_) {
       std::cerr << "MUI Error [matrix_arithmetic.h]: matrix size mismatch during matrix subtraction" << std::endl;
       std::abort();
   }

   if (subtrahend.matrix_format_ != matrix_format_) {
       subtrahend.format_conversion(this->get_format(), true, true, "overwrite");
   } else {
       if (!subtrahend.is_sorted_unique("matrix_arithmetic.h", "operator-()")){
           if (subtrahend.matrix_format_ == format::COO) {
               subtrahend.sort_coo(true, true, "overwrite");
           } else if (subtrahend.matrix_format_ == format::CSR) {
               subtrahend.sort_csr(true, "overwrite");
           } else if (subtrahend.matrix_format_ == format::CSC) {
               subtrahend.sort_csc(true, "overwrite");
           } else {
               std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised subtrahend matrix format for matrix operator-()" << std::endl;
               std::cerr << "    Please set the subtrahend matrix_format_ as:" << std::endl;
               std::cerr << "    format::COO: COOrdinate format" << std::endl;
               std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
               std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
               std::abort();
           }
       }
   }

    if (!this->is_sorted_unique("matrix_arithmetic.h", "operator-()")){
        if (matrix_format_ == format::COO) {
            this->sort_coo(true, true, "overwrite");
        } else if (matrix_format_ == format::CSR) {
            this->sort_csr(true, "overwrite");
        } else if (matrix_format_ == format::CSC) {
            this->sort_csc(true, "overwrite");
        } else {
            std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised matrix format for matrix operator-()" << std::endl;
            std::cerr << "    Please set the matrix_format_ as:" << std::endl;
            std::cerr << "    format::COO: COOrdinate format" << std::endl;
            std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
            std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
            std::abort();
        }
    }

   // Create a new sparse matrix object for the result
   sparse_matrix<ITYPE,VTYPE> res(rows_, cols_, this->get_format());

   if (matrix_format_ == format::COO) 
   {

       std::vector<VTYPE> subtrahend_value;
       subtrahend_value.reserve(subtrahend.matrix_coo.values_.size());

        for (VTYPE &element : subtrahend.matrix_coo.values_) {
            subtrahend_value.emplace_back(element*(-1));
        }

        // Perform element-wise subtrahend of the COO vectors
        res.matrix_coo.values_.reserve(matrix_coo.values_.size() + subtrahend.matrix_coo.values_.size());
        res.matrix_coo.row_indices_.reserve(matrix_coo.row_indices_.size() + subtrahend.matrix_coo.row_indices_.size());
        res.matrix_coo.col_indices_.reserve(matrix_coo.col_indices_.size() + subtrahend.matrix_coo.col_indices_.size());

       // Insert the COO vectors of the initial sparse matrix to the result sparse matrix
       res.matrix_coo.values_ = std::vector<VTYPE>(matrix_coo.values_.begin(), matrix_coo.values_.end());
       res.matrix_coo.row_indices_ = std::vector<ITYPE>(matrix_coo.row_indices_.begin(), matrix_coo.row_indices_.end());
       res.matrix_coo.col_indices_ = std::vector<ITYPE>(matrix_coo.col_indices_.begin(), matrix_coo.col_indices_.end());

       // Append the subtrahend COO vectors to the result sparse matrix
       res.matrix_coo.values_.insert(res.matrix_coo.values_.end(), subtrahend_value.begin(), subtrahend_value.end());
       res.matrix_coo.row_indices_.insert(res.matrix_coo.row_indices_.end(), subtrahend.matrix_coo.row_indices_.begin(), subtrahend.matrix_coo.row_indices_.end());
       res.matrix_coo.col_indices_.insert(res.matrix_coo.col_indices_.end(), subtrahend.matrix_coo.col_indices_.begin(), subtrahend.matrix_coo.col_indices_.end());

       // Sort and deduplicate the result
       res.sort_coo(true, true, "plus");

    }
     
    else if (matrix_format_ == format::CSR) 
    {

        // Perform element-wise subtraction of the CSR vectors
        res.matrix_csr.values_.reserve(matrix_csr.values_.size() + subtrahend.matrix_csr.values_.size());
        res.matrix_csr.row_ptrs_.resize(rows_ + 1);
        res.matrix_csr.col_indices_.reserve(matrix_csr.col_indices_.size() + subtrahend.matrix_csr.col_indices_.size());
       
        auto defaultQueue = sycl::queue{sycl::default_selector_v};
            
        

        double *d_matrix_values;
        double *d_subtrahend_values;
        double *d_res_values;


        int *d_matrix_col;
        int *d_subtrahend_col;
        int *d_res_col;
   
        int *d_matrix_row;
        int *d_subtrahend_row;
        int *d_res_row;
        int *d_row_count;
        
        size_t size_matrix = matrix_csr.values_.size() ;
        size_t size_subtrahend = subtrahend.matrix_csr.values_.size();
        size_t size_res = (matrix_csr.values_.size() + subtrahend.matrix_csr.values_.size());

        size_t size_rows = (rows_ + 1);

        size_t size_matrix_col = matrix_csr.col_indices_.size() ;
        size_t size_subtrahend_col = subtrahend.matrix_csr.col_indices_.size() ;
        size_t size_res_col;// = (matrix_csr.col_indices_.size()+addend.matrix_csr.col_indices_.size());

        d_matrix_values     = sycl::malloc_shared<double>(size_matrix,defaultQueue);
        d_subtrahend_values = sycl::malloc_shared<double>(size_subtrahend,defaultQueue);
         
        d_matrix_col        = sycl::malloc_shared<int>(size_matrix_col,defaultQueue);
        d_subtrahend_col    = sycl::malloc_shared<int>(size_subtrahend_col,defaultQueue);
       
        d_matrix_row        = sycl::malloc_shared<int>(size_rows,defaultQueue);
        d_subtrahend_row    = sycl::malloc_shared<int>(size_rows,defaultQueue);
        d_row_count         = sycl::malloc_shared<int>(rows_,defaultQueue);
        d_res_row           = sycl::malloc_shared<int>(size_rows,defaultQueue);

        for (int i=0;i<size_matrix;i++)
        {
            d_matrix_values[i] = matrix_csr.values_.at(i);
            d_matrix_col[i]    = matrix_csr.col_indices_.at(i);
        
        }
        for (int i=0;i<size_subtrahend;i++)
        {
            d_subtrahend_values[i] = subtrahend.matrix_csr.values_.at(i);
            d_subtrahend_col[i] = subtrahend.matrix_csr.col_indices_.at(i);
        }

        for (int i=0;i<=rows_;i++)
        {
            d_matrix_row[i] = matrix_csr.row_ptrs_.at(i);
            d_subtrahend_row[i] = subtrahend.matrix_csr.row_ptrs_.at(i);
           
        }

        
        /*
        defaultQueue.memcpy(d_matrix_values,h_matrix_values,(size_matrix*sizeof(double))).wait();
        defaultQueue.memcpy(d_subtrahend_values,h_subtrahend_values,(size_subtrahend*sizeof(double))).wait();
        defaultQueue.memcpy(d_matrix_col,h_matrix_col,(size_matrix*sizeof(int))).wait();
        defaultQueue.memcpy(d_subtrahend_col,h_subtrahend_col,(size_subtrahend*sizeof(int))).wait();
        defaultQueue.memcpy(d_matrix_row,h_matrix_row,((rows_+1)*sizeof(int))).wait();
        defaultQueue.memcpy(d_subtrahend_row,h_subtrahend_row,((rows_+1)*sizeof(int))).wait();
        */
        
        

        auto cg = [&](sycl::handler &h) 
        {
            
            h.parallel_for(sycl::range(rows_),[=](sycl::id<1> idx) 
            {
                auto startIdx = d_matrix_row[idx];
                auto endIdx = d_matrix_row[idx+1];
                auto startaddIdx = d_subtrahend_row[idx];
                auto endaddIdx = d_subtrahend_row[idx+1];
                auto i = startIdx;
                auto j = startaddIdx;
                auto count = 0;
                auto col = 0;
                auto subtrahend_col = 0;
                i = startIdx;
                auto result = 0.;
                while (i < endIdx && j < endaddIdx) 
                {
                    col = d_matrix_col[i];
                    subtrahend_col = d_subtrahend_col[j];
                    if (col == subtrahend_col)
                    {
                        result = d_matrix_values[i] - d_subtrahend_values[j];
                        i++;
                        j++;
                        
                    }
                    else if(col < subtrahend_col)
                    {
                        result = d_matrix_values[i] ;
                        i++;
                        
                    }
                    else
                    {
                        result = - d_subtrahend_values[j];
                        j++;
                        
                    }

                    if (std::abs(result) >= std::numeric_limits<VTYPE>::min())
                    {
                        count++;
                    }
                }
                for (;i<endIdx;i++)
                {
                    result = d_matrix_values[i] ;
                    if (std::abs(result) >= std::numeric_limits<VTYPE>::min())
                    {
                        count++;
                    }
                }
                for(;j<endaddIdx;j++)
                {
                    result = - d_subtrahend_values[j];
                    if (std::abs(result) >= std::numeric_limits<VTYPE>::min())
                    {
                        count++;
                    }
                }
                
                d_row_count[idx] = count;
                
            });
        };        
        defaultQueue.submit(cg).wait(); 
        
        auto chg = [&](sycl::handler &gh) 
        {
            gh.parallel_for(sycl::range(size_rows),[=](sycl::id<1> idx) 
            {
                d_res_row[idx] = 0;
                auto count =0;
                if (idx > 0)
                {
                    while (count < idx)
                    {   
                        d_res_row[idx] = d_res_row[idx]+d_row_count[count];
                        count++;
                    }
                } 
            });
        };        
        defaultQueue.submit(chg).wait(); 
        
       // defaultQueue.memcpy(h_res_row,d_res_row,(size_rows*sizeof(int))).wait();
        size_res_col = d_res_row[rows_];

        //h_res_values    = (double *)malloc(size_res_col* sizeof(double));
        //h_res_col       = (int *)malloc(size_res_col*sizeof(int));

        d_res_values    = sycl::malloc_shared<double>(size_res_col,defaultQueue);
        d_res_col       = sycl::malloc_shared<int>(size_res_col,defaultQueue);
        
        auto cag = [&](sycl::handler &ga)
        {
            ga.parallel_for(sycl::range(rows_),[=](sycl::id<1>idx)
            {
                auto startIdx = d_matrix_row[idx];
                auto endIdx = d_matrix_row[idx+1];
                auto startaddIdx = d_subtrahend_row[idx];
                auto endaddIdx = d_subtrahend_row[idx+1];
                auto i = startIdx;
                auto j = startaddIdx;
                auto count = 0;
                auto col = 0;
                auto subtrahend_col = 0;
                auto placeHold = d_res_row[idx];
                while (i < endIdx && j < endaddIdx) 
                {
                    col        = d_matrix_col[i];
                    subtrahend_col = d_subtrahend_col[j];
                    if (col == subtrahend_col)
                    {
                        if (std::abs(d_matrix_values[i] - d_subtrahend_values[j]) >= std::numeric_limits<VTYPE>::min())
                        {
                            d_res_values[placeHold + count] = d_matrix_values[i] - d_subtrahend_values[j];
                            d_res_col[placeHold + count] = col;                            
                            count++;
                        }
                        i++;
                        j++;
                    }
                    else if (col < subtrahend_col)
                    {
                        if (std::abs(d_matrix_values[i]) >= std::numeric_limits<VTYPE>::min())
                        {
                            d_res_values[placeHold + count] = d_matrix_values[i];
                            d_res_col[placeHold + count] = col;
                            count++;
                        }
                        i++;
                    }
                    else
                    {
                        if (std::abs(d_subtrahend_values[j]) >= std::numeric_limits<VTYPE>::min())
                        {        
                            d_res_values[placeHold + count] = - d_subtrahend_values[j];
                            d_res_col[placeHold + count] = subtrahend_col;
                            count++;
                        }
                        j++;
                        
                    }
                }
                for (;i<endIdx;i++)
                {
                    col = d_matrix_col[i];
                    if (std::abs(d_matrix_values[i]) >= std::numeric_limits<VTYPE>::min())
                    {
                        d_res_values[placeHold + count] = d_matrix_values[i];
                        d_res_col[placeHold + count] = col;
                        count++;
                    }
                }
                for (;j<endaddIdx;j++)
                {
                    subtrahend_col = d_subtrahend_col[j];
                    if (std::abs(d_subtrahend_values[j]) >= std::numeric_limits<VTYPE>::min())
                    {
                        d_res_values[placeHold + count] = - d_subtrahend_values[j];
                        d_res_col[placeHold + count] = subtrahend_col;
                        count++;
                    }
                }
            });
        };
        defaultQueue.submit(cag).wait(); 

       // defaultQueue.memcpy(h_res_values,d_res_values,(size_res_col*sizeof(double))).wait();
       // defaultQueue.memcpy(h_res_col,d_res_col,(size_res_col*sizeof(int))).wait();
        
       res.matrix_sycl.values = sycl::malloc_shared<VTYPE>(size_res_col,defaultQueue);
       res.matrix_sycl.column = sycl::malloc_shared<ITYPE>(size_res_col,defaultQueue);
       res.matrix_sycl.row    = sycl::malloc_shared<ITYPE>((rows_+1),defaultQueue);
       copy_sycl_data(defaultQueue,res.matrix_sycl.values,d_res_values,size_res_col);
       copy_sycl_data(defaultQueue,res.matrix_sycl.column,d_res_col,size_res_col);
       copy_sycl_data(defaultQueue,res.matrix_sycl.row,d_res_row,(rows_+1));
       sycl::free(d_res_values,defaultQueue);
       sycl::free(d_res_col,defaultQueue);
       sycl::free(d_res_row,defaultQueue);
       // defaultQueue.memcpy(h_res_values,d_res_values,(size_res*sizeof(double))).wait();             
       // res.matrix_csr.values_.resize(size_res);
       // res.matrix_csr.values_.resize(size_res);
        for (int i=0;i<size_res_col;i++)
        {
            res.matrix_csr.values_.emplace_back(res.matrix_sycl.values[i]);
            res.matrix_csr.col_indices_.emplace_back(res.matrix_sycl.column[i]);
        }
        for (int i=0;i<=rows_;i++)
        {
            res.matrix_csr.row_ptrs_[i] = (res.matrix_sycl.row[i]);
        }
        res.nnz_ = res.matrix_csr.col_indices_.size();
        res.matrix_csr.row_ptrs_[rows_ + 1] = res.nnz_;
    }

     else if (matrix_format_ == format::CSC) {

        // Perform element-wise subtraction of the CSC vectors
        res.matrix_csc.values_.reserve(matrix_csc.values_.size() + subtrahend.matrix_csc.values_.size());
        res.matrix_csc.row_indices_.reserve(matrix_csc.row_indices_.size() + subtrahend.matrix_csc.row_indices_.size());
        res.matrix_csc.col_ptrs_.resize(cols_ + 1);

        ITYPE column = 0;
        while (column < cols_) {
            ITYPE start = matrix_csc.col_ptrs_[column];
            ITYPE end = matrix_csc.col_ptrs_[column + 1];

            ITYPE subtrahend_start = subtrahend.matrix_csc.col_ptrs_[column];
            ITYPE subtrahend_end = subtrahend.matrix_csc.col_ptrs_[column + 1];

            res.matrix_csc.col_ptrs_[0] = 0;

            // Merge the values and row indices of the two columns
            ITYPE i = start;
            ITYPE j = subtrahend_start;
            while (i < end && j < subtrahend_end) {
                ITYPE row = matrix_csc.row_indices_[i];
                ITYPE subtrahend_row = subtrahend.matrix_csc.row_indices_[j];

                if (row == subtrahend_row) {
                    // Add the corresponding values if the columns match
                    if (std::abs(matrix_csc.values_[i] - subtrahend.matrix_csc.values_[j]) >= std::numeric_limits<VTYPE>::min()) {
                        res.matrix_csc.values_.emplace_back(matrix_csc.values_[i] - subtrahend.matrix_csc.values_[j]);
                        res.matrix_csc.row_indices_.emplace_back(row);
                    }
                    i++;
                    j++;
                } else if (row < subtrahend_row) {
                    // Add the current value from the initial matrix
                    if (std::abs(matrix_csc.values_[i]) >= std::numeric_limits<VTYPE>::min()) {
                        res.matrix_csc.values_.emplace_back(matrix_csc.values_[i]);
                        res.matrix_csc.row_indices_.emplace_back(row);
                    }
                    i++;
                } else {
                    // Add the current value from the subtrahend matrix
                    if (std::abs(-subtrahend.matrix_csc.values_[j]) >= std::numeric_limits<VTYPE>::min()) {
                        res.matrix_csc.values_.emplace_back(-subtrahend.matrix_csc.values_[j]);
                        res.matrix_csc.row_indices_.emplace_back(subtrahend_row);
                    }
                    j++;
                }
            }

            // Add any remaining elements from the initial matrix
            for (; i < end; i++) {
                if (std::abs(matrix_csc.values_[i]) >= std::numeric_limits<VTYPE>::min()){
                    res.matrix_csc.values_.emplace_back(matrix_csc.values_[i]);
                    res.matrix_csc.row_indices_.emplace_back(matrix_csc.row_indices_[i]);
                }
            }

            // Add any remaining elements from the subtrahend matrix
            for (; j < subtrahend_end; j++) {
                if (std::abs(-subtrahend.matrix_csc.values_[j]) >= std::numeric_limits<VTYPE>::min()){
                    res.matrix_csc.values_.emplace_back(-subtrahend.matrix_csc.values_[j]);
                    res.matrix_csc.row_indices_.emplace_back(subtrahend.matrix_csc.row_indices_[j]);
                }
            }

            // Update the column pointer
            res.nnz_ = res.matrix_csc.row_indices_.size();
            res.matrix_csc.col_ptrs_[column + 1] = res.nnz_;

            column++;
        }

    } else {
        std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised matrix format for matrix operator-()" << std::endl;
        std::cerr << "    Please set the matrix_format_ as:" << std::endl;
        std::cerr << "    format::COO: COOrdinate format" << std::endl;
        std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
        std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
        std::abort();
    }

   return res;
}

// Overload multiplication operator to perform sparse matrix multiplication
template<typename ITYPE, typename VTYPE>
void sparse_matrix<ITYPE,VTYPE>::sycl_multiply(sycl::queue defaultQueue, sparse_matrix<ITYPE,VTYPE> &matrix, sparse_matrix<ITYPE,VTYPE> &multi_vec)
{
    ITYPE vec_size = matrix.rows_;
    sycl_multiply_mat_vec(defaultQueue,matrix_sycl.values,matrix_sycl.vector_val,matrix.matrix_sycl.values,multi_vec.matrix_sycl.vector_val,matrix_sycl.column,matrix_sycl.row,matrix.matrix_sycl.column,matrix.matrix_sycl.row, vec_size);
}

template<typename ITYPE, typename VTYPE>
void sparse_matrix<ITYPE,VTYPE>::sycl_multiply_vector(sycl::queue defaultQueue, const sparse_matrix<ITYPE,VTYPE> &matrix, sparse_matrix<ITYPE,VTYPE> &multi_vec)
{
    ITYPE vec_size = matrix.rows_;
    sycl_multiply_vec_vec(defaultQueue,matrix_sycl.vector_val,matrix.matrix_sycl.vector_val,multi_vec.matrix_sycl.vector_val, vec_size);
}

template<typename ITYPE, typename VTYPE>
void sparse_matrix<ITYPE,VTYPE>::sycl_multiply_mat_vec(sycl::queue defaultQueue, VTYPE *res_mat, VTYPE *res_vec, VTYPE *mat_value, VTYPE *vec_value, ITYPE *res_col, ITYPE *res_row, ITYPE *mat_column, ITYPE *mat_row,  ITYPE size_row) 
{
    size_t rows = size_row;
    auto cag = [&](sycl::handler &ga)
    {
        ga.parallel_for(sycl::range(rows),[=](sycl::id<1>idx)
        {
            auto startIdx = mat_row[idx];
            auto endIdx = mat_row[idx+1];
            auto col_idx = 0;
            res_row[0] = 0;
            res_vec[idx] = 0.;
            for (int i = startIdx; i < endIdx; i++)
            {
                col_idx = mat_column[i];
                res_vec[idx] += mat_value[i]*vec_value[col_idx];
            }
            res_mat[idx] = res_vec[idx];
            res_col[idx] = 0;
            res_row[idx+1] = idx;
        });
    };
    defaultQueue.submit(cag).wait(); 
    //for (int i=0;i<rows;i++)
    //{
    //    std::cout<< "Matrix values : " << res_mat[i] << " and vec value : " << res_vec[i] <<std::endl;
    //}
}

template<typename ITYPE, typename VTYPE>
void sparse_matrix<ITYPE,VTYPE>::sycl_multiply_vec_vec(sycl::queue defaultQueue, VTYPE *res_vec, VTYPE *mat_value, VTYPE *vec_value,  ITYPE size_row) 
{
    size_t rows = size_row;
    auto cag = [&](sycl::handler &ga)
    {
        ga.parallel_for(sycl::range(rows),[=](sycl::id<1>idx)
        {
            res_vec[idx] = 0.;
            if (abs(vec_value[idx])>= std::numeric_limits<VTYPE>::min())
            {
                res_vec[idx] = mat_value[idx]*vec_value[idx];
            }
        });
    };
    defaultQueue.submit(cag).wait(); 
}
template<typename ITYPE, typename VTYPE>
sparse_matrix<ITYPE,VTYPE> sparse_matrix<ITYPE,VTYPE>::operator*(sparse_matrix<ITYPE,VTYPE> &multiplicand) 
{
    if (cols_ != multiplicand.rows_) {
        std::cerr << "MUI Error [matrix_arithmetic.h]: matrix size mismatch during matrix multiplication" << std::endl;
        std::abort();
    }
    
    if (multiplicand.matrix_format_ != matrix_format_) 
    {
        multiplicand.format_conversion(this->get_format(), true, true, "overwrite");
    } 
    else 
    {
        if (!multiplicand.is_sorted_unique("matrix_arithmetic.h", "operator*()"))
        {
            if (multiplicand.matrix_format_ == format::COO) {
                multiplicand.sort_coo(true, true, "overwrite");
            } else if (multiplicand.matrix_format_ == format::CSR) {
                multiplicand.sort_csr(true, "overwrite");
            } else if (multiplicand.matrix_format_ == format::CSC) {
                multiplicand.sort_csc(true, "overwrite");
            } else {
                std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised multiplicand matrix format for matrix operator*()" << std::endl;
                std::cerr << "    Please set the multiplicand matrix_format_ as:" << std::endl;
                std::cerr << "    format::COO: COOrdinate format" << std::endl;
                std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
                std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
                std::abort();
            }
        }
    }

    if (!this->is_sorted_unique("matrix_arithmetic.h", "operator*()"))
    {
        if (matrix_format_ == format::COO) 
        {
            this->sort_coo(true, true, "overwrite");
        } 
        else if (matrix_format_ == format::CSR) 
        {
            this->sort_csr(true, "overwrite");
        } 
        else if (matrix_format_ == format::CSC) 
        {
            this->sort_csc(true, "overwrite");
        } 
        else 
        {
            std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised matrix format for matrix operator*()" << std::endl;
            std::cerr << "    Please set the matrix_format_ as:" << std::endl;
            std::cerr << "    format::COO: COOrdinate format" << std::endl;
            std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
            std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
            std::abort();
        }
    }

    // Create a new sparse matrix object for the result
    sparse_matrix<ITYPE,VTYPE> res(rows_, multiplicand.cols_, this->get_format());

    if (matrix_format_ == format::COO) 
    {

        // Perform element-wise multiplication of the COO vectors
        res.matrix_coo.values_.reserve((matrix_coo.values_.size() <= multiplicand.matrix_coo.values_.size()) ? multiplicand.matrix_coo.values_.size() : matrix_coo.values_.size());
        res.matrix_coo.row_indices_.reserve((matrix_coo.row_indices_.size() <= multiplicand.matrix_coo.row_indices_.size()) ? multiplicand.matrix_coo.row_indices_.size() : matrix_coo.row_indices_.size());
        res.matrix_coo.col_indices_.reserve((matrix_coo.col_indices_.size() <= multiplicand.matrix_coo.col_indices_.size()) ? multiplicand.matrix_coo.col_indices_.size() : matrix_coo.col_indices_.size());

        for (ITYPE i = 0; i < static_cast<ITYPE>(matrix_coo.row_indices_.size()); ++i) 
        {
            for (ITYPE j = 0; j < static_cast<ITYPE>(multiplicand.matrix_coo.col_indices_.size()); ++j) 
            {
                if (matrix_coo.col_indices_[i] == multiplicand.matrix_coo.row_indices_[j]) 
                {
                    // Multiply the corresponding values if the columns match
                    VTYPE value = matrix_coo.values_[i] * multiplicand.matrix_coo.values_[j];
                    if (std::abs(value) >= std::numeric_limits<VTYPE>::min()) 
                    {
                        res.matrix_coo.values_.emplace_back(value);
                        res.matrix_coo.row_indices_.emplace_back(matrix_coo.row_indices_[i]);
                        res.matrix_coo.col_indices_.emplace_back(multiplicand.matrix_coo.col_indices_[j]);
                    }
                }
            }
        }

        // Sort and deduplicate the result
        res.sort_coo(true, true, "plus");
        res.nnz_ = res.matrix_coo.values_.size();

    } 
    
    else if (matrix_format_ == format::CSR) 
    {

        // Perform element-wise multiplication of the CSR vectors
        res.matrix_csr.values_.reserve((matrix_csr.values_.size() <= multiplicand.matrix_csr.values_.size()) ? multiplicand.matrix_csr.values_.size() : matrix_csr.values_.size());
        res.matrix_csr.row_ptrs_.resize(rows_+1);
        res.matrix_csr.col_indices_.reserve((matrix_csr.col_indices_.size() <= multiplicand.matrix_csr.col_indices_.size()) ? multiplicand.matrix_csr.col_indices_.size() : matrix_csr.col_indices_.size());

        
        auto defaultQueue = sycl::queue {sycl::default_selector_v};
      
        double *d_matrix_values;
        double *d_multiplicand_values;
        double *d_res_values;

        int *d_matrix_col;
        int *d_multiplicand_col;
        int *d_res_col;
        
        int *d_matrix_row;
        int *d_multiplicand_row;
        int *d_res_row;
        int *d_row_count;

        int *d_intermediate;
        int *d_intermediate_col;
        int *d_intermediate_rowid;
        int *d_intermediate_row;
        double *d_intermediate_values;
        int *h_intermediate;
        double *d_value;

        size_t size_matrix = matrix_csr.values_.size() ;
        size_t size_multiplicand = multiplicand.matrix_csr.values_.size();
        size_t size_res;// = (matrix_csr.values_.size() + subtrahend.matrix_csr.values_.size());

        size_t size_rows = (rows_ + 1);
        size_t size_cols = multiplicand.cols_;
        size_t size_matrix_col = matrix_csr.col_indices_.size() ;
        size_t size_multiplicand_col = multiplicand.matrix_csr.col_indices_.size() ;
        size_t size_res_col;// = (matrix_csr.col_indices_.size()+addend.matrix_csr.col_indices_.size());

        h_intermediate = (int *)malloc(size_cols);

        d_matrix_values       = sycl::malloc_shared<double>(size_matrix,defaultQueue);
        d_multiplicand_values = sycl::malloc_shared<double>(size_multiplicand,defaultQueue);
         
        d_matrix_col          = sycl::malloc_shared<int>(size_matrix_col,defaultQueue);
        d_multiplicand_col    = sycl::malloc_shared<int>(size_multiplicand_col,defaultQueue);
       
        d_matrix_row          = sycl::malloc_shared<int>(size_rows,defaultQueue);
        d_multiplicand_row    = sycl::malloc_shared<int>(size_rows,defaultQueue);
        d_row_count           = sycl::malloc_shared<int>(rows_,defaultQueue);
        d_res_row             = sycl::malloc_shared<int>(size_rows,defaultQueue);
        d_intermediate        = sycl::malloc_device<int>(size_cols,defaultQueue);
        d_intermediate_col    = sycl::malloc_shared<int>((size_cols+1),defaultQueue);
        d_intermediate_rowid  = sycl::malloc_shared<int>((size_multiplicand),defaultQueue);
        d_intermediate_row    = sycl::malloc_shared<int>((size_multiplicand),defaultQueue);
        d_intermediate_values = sycl::malloc_shared<double>((size_multiplicand),defaultQueue);
        d_value               = sycl::malloc_shared<double>((rows_*size_cols),defaultQueue);

        for (int i=0;i<size_matrix;i++)
        {
            d_matrix_values[i] = matrix_csr.values_.at(i);
            d_matrix_col[i]    = matrix_csr.col_indices_.at(i);
           
        
        }
        for (int i=0;i<size_multiplicand;i++)
        {
            d_multiplicand_values[i] = multiplicand.matrix_csr.values_.at(i);
            d_multiplicand_col[i] = multiplicand.matrix_csr.col_indices_.at(i);

        }

        for (int i=0;i<=rows_;i++)
        {
            d_matrix_row[i] = matrix_csr.row_ptrs_.at(i);
            d_multiplicand_row[i] = multiplicand.matrix_csr.row_ptrs_.at(i);
        }
        for (int i=0;i<size_cols;i++)
        {
            h_intermediate[i] = 0;
        }
        for (int i=0;i<(rows_ * size_cols);i++)
        {
            d_value[i] = 0;
        }
        defaultQueue.memcpy(d_intermediate,h_intermediate,(size_cols*sizeof(int))).wait();
        
        // Initialize a vector to store the intermediate results
       
        std::vector<VTYPE> intermediate(multiplicand.cols_, 0.0);

        res.matrix_csr.row_ptrs_[0] = 0;

        auto cgc = [&](sycl::handler &hc) 
        {
            hc.parallel_for(sycl::range(size_multiplicand),[=](sycl::id<1> idx) 
            {
                auto column = d_multiplicand_col[idx];
                d_intermediate_row[idx] = -1;
                auto addi = 1;
                //d_intermediate[column]+=1;
                auto v = sycl::atomic_ref<
                         int, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(d_intermediate[column]);
                v.fetch_add(addi);

            });
        };
        defaultQueue.submit(cgc).wait(); 

        auto cgs = [&](sycl::handler &hs) 
        {
            hs.parallel_for(sycl::range((size_cols+1)),[=](sycl::id<1> idx) 
            {
                d_intermediate_col[idx] = 0;
                auto count =0;
                if (idx > 0)
                {
                    while (count < idx)
                    {   
                        d_intermediate_col[idx] = d_intermediate_col[idx]+d_intermediate[count];
                        count++;
                    }
                } 
            });
        };
        defaultQueue.submit(cgs).wait();

        auto cgrs = [&](sycl::handler &rhs) 
        {
            rhs.parallel_for(sycl::range((size_cols)),[=](sycl::id<1> idx) 
            {
                d_intermediate[idx] = d_intermediate_col[idx]; 
            });
        };
        defaultQueue.submit(cgrs).wait();
     
        auto cgr = [&](sycl::handler &hr)
        {
            hr.parallel_for(sycl::range((size_rows - 1)),[=](sycl::id<1> idx)
            {
                auto row_start = d_multiplicand_row[idx];
                auto row_end = d_multiplicand_row[idx+1];
                for (int i=row_start;i<row_end;i++)
                {
                    d_intermediate_rowid[i] = idx;
                }
            });
        };
        defaultQueue.submit(cgr).wait();
       
        auto cgv = [&](sycl::handler &hv)
        {
            hv.parallel_for(sycl::range(size_cols),[=](sycl::id<1> idx)
            {
                auto start = d_intermediate_col[idx];
                auto end = d_intermediate_col[idx+1]; 
                //auto i=start;
                auto count = 0; 
                for (int i=start;i<end;i++)
                {
                    auto loc = d_multiplicand_col[i];
                    auto pos = d_intermediate_col[loc];
                    auto v = sycl::atomic_ref<
                         int, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(d_intermediate[loc]);
                    auto index = v.fetch_add(1);
                    d_intermediate_row[index] = d_intermediate_rowid[i];
                    d_intermediate_values[index] = d_multiplicand_values[i];
                    ++count;
                }
            });
        };
        defaultQueue.submit(cgv).wait();

        auto cgd = [&](sycl::handler &hd) 
        {
            hd.parallel_for(sycl::range(rows_),[=](sycl::id<1> idx) 
            {
                d_intermediate[idx] = 0;
                auto start = d_matrix_row[idx];
                auto end = d_matrix_row[idx+1];

                for (int k=0;k<size_cols;k++)
                { 
                    auto count = 0.;
                    auto multiplication_start = d_intermediate_col[k];
                    auto multiplication_end = d_intermediate_col[k+1];
                    auto i = start;
                    //auto j= multiplication_start;
                    while (i<end )
                    {
                        auto mat_index = d_matrix_col[i];
                        auto mat_value = d_matrix_values[i];
                            
                        for (int j=multiplication_start;j<multiplication_end;j++)
                        {
                            auto multi_index = d_intermediate_row[j];
                            auto multi_value = d_intermediate_values[j];
                            if (mat_index == multi_index)
                            {
                                count += mat_value*multi_value;
                            }
                        }
                        i++;
                    }
                    if (std::abs(count) >= std::numeric_limits<VTYPE>::min() )
                    {
                        d_intermediate[idx] += 1;
                    }
                }
            });
        };
        defaultQueue.submit(cgd).wait();
        
        auto cgf = [&](sycl::handler &hf) 
        {
            hf.parallel_for(sycl::range((size_rows)),[=](sycl::id<1> idx) 
            {
                d_res_row[idx] = 0;
                auto count =0;
                if (idx > 0)
                {
                    while (count < idx)
                    {   
                        d_res_row[idx] = d_res_row[idx]+d_intermediate[count];
                        count++;
                    }
                }

            });
        };
        defaultQueue.submit(cgf).wait();

        size_res_col = d_res_row[rows_];
        //h_res_values    = (double *)malloc(size_res_col* sizeof(double));
        //h_res_col       = (int *)malloc(size_res_col*sizeof(int));

        d_res_values    = sycl::malloc_shared<double>(size_res_col,defaultQueue);
        d_res_col       = sycl::malloc_shared<int>(size_res_col,defaultQueue);

        auto cg = [&](sycl::handler &h) 
        {
            h.parallel_for(sycl::range(rows_),[=](sycl::id<1> idx) 
            {
                d_intermediate[idx] = 0;
                auto start = d_matrix_row[idx];
                auto end = d_matrix_row[idx+1];
                auto placeHold = d_res_row[idx];
                auto count = 0;
                for (int k=0;k<size_cols;k++)
                { 
                         auto product = 0.;
                         auto multiplication_start = d_intermediate_col[k];
                         auto multiplication_end = d_intermediate_col[k+1];
                         auto i = start;
                         //auto j= multiplication_start;
                         while (i<end)
                         {
                            auto mat_index = d_matrix_col[i];
                            auto mat_value = d_matrix_values[i];
                            
                            for (int j=multiplication_start;j<multiplication_end;j++)
                            {
                                auto multi_index = d_intermediate_row[j];
                                auto multi_value = d_intermediate_values[j];
                                if (mat_index == multi_index)
                                {
                                    product += mat_value*multi_value;
                                }
                            }
                            i++;
                        }
                        if (std::abs(product) >= std::numeric_limits<VTYPE>::min())  
                        {   
                            d_res_values[placeHold + count] = product;
                            d_res_col[placeHold + count] = k;
                            count++;
                        }   
                    }
            });
        };
        defaultQueue.submit(cg).wait();
        
        // Iterate over each row of the initial matrix
       res.matrix_sycl.values = sycl::malloc_shared<VTYPE>(size_res_col,defaultQueue);
       res.matrix_sycl.column = sycl::malloc_shared<ITYPE>(size_res_col,defaultQueue);
       res.matrix_sycl.row    = sycl::malloc_shared<ITYPE>((rows_+1),defaultQueue);
       copy_sycl_data(defaultQueue,res.matrix_sycl.values,d_res_values,size_res_col);
       copy_sycl_data(defaultQueue,res.matrix_sycl.column,d_res_col,size_res_col);
       copy_sycl_data(defaultQueue,res.matrix_sycl.row,d_res_row,(rows_+1));

        for (ITYPE j = 0; j < size_res_col; ++j) 
        {
            res.matrix_csr.values_.emplace_back(res.matrix_sycl.values[j]);
            res.matrix_csr.col_indices_.emplace_back(res.matrix_sycl.column[j]);
        }
        for (ITYPE j = 0; j < size_rows; ++j) 
        {
            res.matrix_csr.row_ptrs_[j] = res.matrix_sycl.row[j];
        }
       // res.matrix_csr.row_ptrs_[i+1]=res.matrix_csr.values_.size();
        res.nnz_ = res.matrix_csr.values_.size();
    }

    else if (matrix_format_ == format::CSC) 
    {

        // Perform element-wise multiplication of the CSC vectors
        res.matrix_csc.values_.reserve((matrix_csc.values_.size() <= multiplicand.matrix_csc.values_.size()) ? multiplicand.matrix_csc.values_.size() : matrix_csc.values_.size());
        res.matrix_csc.row_indices_.reserve((matrix_csc.row_indices_.size() <= multiplicand.matrix_csc.row_indices_.size()) ? multiplicand.matrix_csc.row_indices_.size() : matrix_csc.row_indices_.size());
        res.matrix_csc.col_ptrs_.resize(cols_+1);

        // Initialize a vector to store the intermediate results
        std::vector<VTYPE> intermediate(rows_, 0.0);

        res.matrix_csc.col_ptrs_[0] = 0;

        // Iterate over each column of the initial matrix
        for (ITYPE j = 0; j < multiplicand.cols_; ++j) {
            // Clear the intermediate results vector for each row
            std::fill(intermediate.begin(), intermediate.end(), 0.0);

            ITYPE multiplicand_start = multiplicand.matrix_csc.col_ptrs_[j];
            ITYPE multiplicand_end = multiplicand.matrix_csc.col_ptrs_[j + 1];

            // Iterate over the non-zero elements of the cloumn
            for (ITYPE k = multiplicand_start; k < multiplicand_end; ++k) {
                // Get the row index and value of the element
                ITYPE multiplicand_row = multiplicand.matrix_csc.row_indices_[k];
                VTYPE multiplicand_value = multiplicand.matrix_csc.values_[k];

                ITYPE start = matrix_csc.col_ptrs_[multiplicand_row];
                ITYPE end = matrix_csc.col_ptrs_[multiplicand_row + 1];

                // Multiply the element with the corresponding column of the other matrix
                for (ITYPE i = start; i < end; ++i) {
                    ITYPE row = matrix_csc.row_indices_[i];
                    VTYPE value = matrix_csc.values_[i];
                    intermediate[row] += value * multiplicand_value;
                }
            }

            // Add the intermediate results to the result vectors
            for (ITYPE i = 0; i < multiplicand.rows_; ++i) {
                VTYPE result_value = intermediate[i];
                if (std::abs(result_value) >= std::numeric_limits<VTYPE>::min()) {
                    res.matrix_csc.values_.emplace_back(result_value);
                    res.matrix_csc.row_indices_.emplace_back(i);
                }
            }
            res.matrix_csc.col_ptrs_[j+1]=res.matrix_csc.values_.size();
        }

        res.nnz_ = res.matrix_csc.values_.size();

    } 
    else
    {
        std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised matrix format for matrix operator*()" << std::endl;
        std::cerr << "    Please set the matrix_format_ as:" << std::endl;
        std::cerr << "    format::COO: COOrdinate format" << std::endl;
        std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
        std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
        std::abort();
    }

    return res;
}

// Overload multiplication operator to perform scalar multiplication A*x
template <typename ITYPE, typename VTYPE>
template <typename STYPE>
sparse_matrix<ITYPE,VTYPE> sparse_matrix<ITYPE,VTYPE>::operator*(const STYPE &scalar) const{
    static_assert(std::is_convertible<STYPE, VTYPE>::value,
            "MUI Error [matrix_arithmetic.h]: scalar type cannot be converted to matrix element type in scalar multiplication");

    // Create a new sparse matrix object for the result
    sparse_matrix<ITYPE,VTYPE> res(*this);

    if (matrix_format_ == format::COO) 
    {
        
        for (VTYPE &element : res.matrix_coo.values_) 
        {
            if (static_cast<VTYPE>(scalar) >= std::numeric_limits<VTYPE>::min())
                element *= scalar;
        }

    }

    else if (matrix_format_ == format::CSR) 
    {

        size_t size_matrix = matrix_csr.values_.size();
        size_t size_rows = matrix_csr.row_ptrs_.size();
       
        auto defaultQueue = sycl::queue {sycl::default_selector_v};
        
        VTYPE *d_res_values;
        ITYPE *d_res_rows;
        ITYPE *d_res_col;

        d_res_values = sycl::malloc_shared<VTYPE>(size_matrix,defaultQueue);
        d_res_col = sycl::malloc_shared<ITYPE>(size_matrix,defaultQueue);
        d_res_rows = sycl::malloc_shared<ITYPE>(size_rows,defaultQueue);
        for (int i=0;i<size_matrix;i++)
        {
            d_res_values[i] = matrix_csr.values_[i];
            d_res_col[i] = matrix_csr.col_indices_[i];
        }
        for (int i=0;i<size_rows;i++)
        {
            d_res_rows[i] = matrix_csr.row_ptrs_[i];
        }
        
        //defaultQueue.memcpy(d_res_values,h_res_values,(size_matrix*sizeof(double))).wait();
        auto cg = [&](sycl::handler &h) 
        {
            h.parallel_for(sycl::range(size_matrix),[=](sycl::id<1> idx) 
            {
                d_res_values[idx] = scalar*d_res_values[idx];
            });
        };
        defaultQueue.submit(cg).wait(); 

        //defaultQueue.memcpy(h_res_values,d_res_values,(size_matrix*sizeof(double))).wait();
        

        res.matrix_sycl.values = sycl::malloc_shared<VTYPE>(size_matrix,defaultQueue);
        res.matrix_sycl.column = sycl::malloc_shared<ITYPE>(size_matrix,defaultQueue);
        res.matrix_sycl.row = sycl::malloc_shared<ITYPE>(size_rows,defaultQueue);

        copy_sycl_data(defaultQueue,res.matrix_sycl.values,d_res_values,size_matrix);
        copy_sycl_data(defaultQueue,res.matrix_sycl.column,d_res_col,size_matrix);
        copy_sycl_data(defaultQueue,res.matrix_sycl.row,d_res_rows,size_rows);
       
       sycl::free(d_res_values,defaultQueue);
       sycl::free(d_res_col,defaultQueue);
       sycl::free(d_res_rows,defaultQueue);

        for (int i=0;i<size_matrix;i++)
        {
            res.matrix_csr.values_.at(i) =res.matrix_sycl.values[i];
        }
    }

    else if (matrix_format_ == format::CSC) 
    {

        for (VTYPE &element : res.matrix_csc.values_) 
        {
            if (static_cast<VTYPE>(scalar) >= std::numeric_limits<VTYPE>::min())
                element *= scalar;
        }

    } 
    else 
    {
        std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised matrix format for matrix scalar operator*()" << std::endl;
        std::cerr << "    Please set the matrix_format_ as:" << std::endl;
        std::cerr << "    format::COO: COOrdinate format" << std::endl;
        std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
        std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
        std::abort();
    }

   return res;

}


// Overload multiplication operator to perform scalar multiplication x*A
template<typename ITYPE, typename VTYPE, typename STYPE>
sparse_matrix<ITYPE,VTYPE> operator*(const STYPE &scalar, const sparse_matrix<ITYPE,VTYPE> &exist_mat) {
   return exist_mat * scalar;
}


template<typename ITYPE, typename VTYPE>
sparse_matrix<ITYPE,VTYPE> sparse_matrix<ITYPE,VTYPE>::operator^(sparse_matrix<ITYPE,VTYPE> &multiplicand) {

    
    if (cols_ != multiplicand.rows_) {
        std::cerr << "MUI Error [matrix_arithmetic.h]: matrix size mismatch during matrix multiplication" << std::endl;
        std::abort();
    }
    
    if (multiplicand.matrix_format_ != matrix_format_) 
    {
        multiplicand.format_conversion(this->get_format(), true, true, "overwrite");
    } 
    else 
    {
        if (!multiplicand.is_sorted_unique("matrix_arithmetic.h", "operator*()"))
        {
            if (multiplicand.matrix_format_ == format::COO) {
                multiplicand.sort_coo(true, true, "overwrite");
            } else if (multiplicand.matrix_format_ == format::CSR) {
                multiplicand.sort_csr(true, "overwrite");
            } else if (multiplicand.matrix_format_ == format::CSC) {
                multiplicand.sort_csc(true, "overwrite");
            } else {
                std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised multiplicand matrix format for matrix operator*()" << std::endl;
                std::cerr << "    Please set the multiplicand matrix_format_ as:" << std::endl;
                std::cerr << "    format::COO: COOrdinate format" << std::endl;
                std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
                std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
                std::abort();
            }
        }
    }

    if (!this->is_sorted_unique("matrix_arithmetic.h", "operator*()"))
    {
         if (matrix_format_ == format::COO) {
             this->sort_coo(true, true, "overwrite");
         } else if (matrix_format_ == format::CSR) {
             this->sort_csr(true, "overwrite");
         } else if (matrix_format_ == format::CSC) {
             this->sort_csc(true, "overwrite");
         } else {
             std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised matrix format for matrix operator*()" << std::endl;
             std::cerr << "    Please set the matrix_format_ as:" << std::endl;
             std::cerr << "    format::COO: COOrdinate format" << std::endl;
             std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
             std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
             std::abort();
         }
    }

    // Create a new sparse matrix object for the result
    sparse_matrix<ITYPE,VTYPE> res(rows_, multiplicand.cols_, this->get_format());

    if (matrix_format_ == format::COO) 
    {

        // Perform element-wise multiplication of the COO vectors
        res.matrix_coo.values_.reserve((matrix_coo.values_.size() <= multiplicand.matrix_coo.values_.size()) ? multiplicand.matrix_coo.values_.size() : matrix_coo.values_.size());
        res.matrix_coo.row_indices_.reserve((matrix_coo.row_indices_.size() <= multiplicand.matrix_coo.row_indices_.size()) ? multiplicand.matrix_coo.row_indices_.size() : matrix_coo.row_indices_.size());
        res.matrix_coo.col_indices_.reserve((matrix_coo.col_indices_.size() <= multiplicand.matrix_coo.col_indices_.size()) ? multiplicand.matrix_coo.col_indices_.size() : matrix_coo.col_indices_.size());

        for (ITYPE i = 0; i < static_cast<ITYPE>(matrix_coo.row_indices_.size()); ++i) 
        {
            for (ITYPE j = 0; j < static_cast<ITYPE>(multiplicand.matrix_coo.col_indices_.size()); ++j) 
            {
                if (matrix_coo.col_indices_[i] == multiplicand.matrix_coo.row_indices_[j]) 
                {
                    // Multiply the corresponding values if the columns match
                    VTYPE value = matrix_coo.values_[i] * multiplicand.matrix_coo.values_[j];
                    if (std::abs(value) >= std::numeric_limits<VTYPE>::min()) 
                    {
                        res.matrix_coo.values_.emplace_back(value);
                        res.matrix_coo.row_indices_.emplace_back(matrix_coo.row_indices_[i]);
                        res.matrix_coo.col_indices_.emplace_back(multiplicand.matrix_coo.col_indices_[j]);
                    }
                }
            }
        }

        // Sort and deduplicate the result
        res.sort_coo(true, true, "plus");
        res.nnz_ = res.matrix_coo.values_.size();

    } 
    
    else if (matrix_format_ == format::CSR) 
    {

        // Perform element-wise multiplication of the CSR vectors
        res.matrix_csr.values_.reserve((matrix_csr.values_.size() <= multiplicand.matrix_csr.values_.size()) ? multiplicand.matrix_csr.values_.size() : matrix_csr.values_.size());
        res.matrix_csr.row_ptrs_.resize(rows_+1);
        res.matrix_csr.col_indices_.reserve((matrix_csr.col_indices_.size() <= multiplicand.matrix_csr.col_indices_.size()) ? multiplicand.matrix_csr.col_indices_.size() : matrix_csr.col_indices_.size());

       
        auto defaultQueue = sycl::queue {sycl::gpu_selector_v};
      
        double *d_matrix_values;
        double *d_multiplicand_values;
        double *d_res_values;

        int *d_matrix_col;
        int *d_multiplicand_col;
        int *d_res_col;
        
        int *d_matrix_row;
        int *d_multiplicand_row;
        int *d_res_row;
        int *d_row_count;

        int *d_intermediate;
        int *d_intermediate_col;
        int *d_intermediate_rowid;
        int *d_intermediate_row;
        double *d_intermediate_values;
        int *h_intermediate;
        double *d_value;

        size_t size_matrix = matrix_csr.values_.size() ;
        size_t size_multiplicand = multiplicand.matrix_csr.values_.size();
        size_t size_res;// = (matrix_csr.values_.size() + subtrahend.matrix_csr.values_.size());

        size_t size_rows = (rows_ + 1);
        size_t size_mat_cols = cols_;
        size_t size_cols = multiplicand.cols_;
        size_t multiplicand_rows = multiplicand.rows_;
        size_t size_matrix_col = matrix_csr.col_indices_.size() ;
        size_t size_multiplicand_col = multiplicand.matrix_csr.col_indices_.size() ;
        size_t size_res_col;// = (matrix_csr.col_indices_.size()+addend.matrix_csr.col_indices_.size());

        h_intermediate = (int *)malloc(size_cols);

        d_matrix_values       = sycl::malloc_shared<double>(size_matrix,defaultQueue);
        d_multiplicand_values = sycl::malloc_shared<double>(size_multiplicand,defaultQueue);
         
        d_matrix_col          = sycl::malloc_shared<int>(size_matrix_col,defaultQueue);
        d_multiplicand_col    = sycl::malloc_shared<int>(size_multiplicand_col,defaultQueue);
       
        d_matrix_row          = sycl::malloc_shared<int>(size_rows,defaultQueue);
        d_multiplicand_row    = sycl::malloc_shared<int>(size_rows,defaultQueue);
        d_row_count           = sycl::malloc_shared<int>(rows_,defaultQueue);
        d_res_row             = sycl::malloc_shared<int>(size_rows,defaultQueue);
        d_intermediate        = sycl::malloc_device<int>(size_cols,defaultQueue);
        d_intermediate_col    = sycl::malloc_shared<int>((size_cols+1),defaultQueue);
        d_intermediate_rowid  = sycl::malloc_shared<int>((size_multiplicand),defaultQueue);
        d_intermediate_row    = sycl::malloc_shared<int>((size_multiplicand),defaultQueue);
        d_intermediate_values = sycl::malloc_shared<double>((size_multiplicand),defaultQueue);
        d_value               = sycl::malloc_shared<double>((rows_*size_cols),defaultQueue);

        for (int i=0;i<size_matrix;i++)
        {
            d_matrix_values[i] = matrix_csr.values_.at(i);
            d_matrix_col[i]    = matrix_csr.col_indices_.at(i);
        }  
       
        for (int i=0;i<=rows_;i++)
        {
            d_matrix_row[i] = matrix_csr.row_ptrs_.at(i);
        }

        
        for (int i=0;i<size_multiplicand;i++)
        {
            d_multiplicand_values[i] = multiplicand.matrix_csr.values_.at(i); 
        }
        for (int i=0;i<size_multiplicand_col;i++)
        {
            d_multiplicand_col[i] = multiplicand.matrix_csr.col_indices_.at(i);
        }
        for (int i=0;i<multiplicand_rows;i++)
        {
            d_multiplicand_row[i] = multiplicand.matrix_csr.row_ptrs_.at(i);
        }
        
        // Initialize a vector to store the intermediate results
       
        std::vector<VTYPE> intermediate(multiplicand.cols_, 0.0);

        res.matrix_csr.row_ptrs_[0] = 0;
        d_res_values    = sycl::malloc_shared<double>(rows_,defaultQueue);
        d_res_col       = sycl::malloc_shared<int>(rows_,defaultQueue);
        auto cgd = [&](sycl::handler &hd) 
        {
            hd.parallel_for(sycl::range(rows_),[=](sycl::id<1> idx) 
            {
                d_intermediate[idx] = 0;
                auto start = d_matrix_row[idx];
                auto end = d_matrix_row[idx+1];
                auto count = 0.;
                for (int k=0;k<multiplicand_rows;k++)
                { 
                    
                    auto multiplication_start = d_multiplicand_row[k];
                    auto multiplication_end = d_multiplicand_row[k+1];
                    auto i = start;
                    //auto j= multiplication_start;
                    while (i<end )
                    {
                        auto mat_index = d_matrix_col[i];
                        auto mat_value = d_matrix_values[i];
        
                        for (int j=multiplication_start;j<multiplication_end;j++)
                        {
                            auto multi_index = d_multiplicand_col[j];
                            auto multi_value = d_multiplicand_values[j];
                            if ( mat_index == j)
                            {
                                count += mat_value*multi_value;
                            }
                        }
                        i++;
                    }
                    if (std::abs(count) >= std::numeric_limits<VTYPE>::min() )
                    {
                        d_res_values[idx] = count;
                        d_res_col[idx] = idx;
                        d_intermediate[idx] += 1;
                    }
                }
            });
        };
        defaultQueue.submit(cgd).wait();
        
        
        auto cgf = [&](sycl::handler &hf) 
        {
            hf.parallel_for(sycl::range((size_rows)),[=](sycl::id<1> idx) 
            {
                d_res_row[idx] = 0;
                auto count =0;
                if (idx > 0)
                {
                    while (count < idx)
                    {   
                        d_res_row[idx] = d_res_row[idx]+d_intermediate[count];
                        count++;
                    }
                }
            });
        };
        defaultQueue.submit(cgf).wait();

        size_res_col = d_res_row[rows_];
        //h_res_values    = (double *)malloc(size_res_col* sizeof(double));
        //h_res_col       = (int *)malloc(size_res_col*sizeof(int));

        
        /*
        auto cg = [&](sycl::handler &h) 
        {
            h.parallel_for(sycl::range(rows_),[=](sycl::id<1> idx) 
            {
                d_intermediate[idx] = 0;
                auto start = d_matrix_row[idx];
                auto end = d_matrix_row[idx+1];
                auto placeHold = d_res_row[idx];
                auto count = 0;
                for (int k=0;k<size_cols;k++)
                { 
                         auto product = 0.;
                         auto multiplication_start = d_intermediate_col[k];
                         auto multiplication_end = d_intermediate_col[k+1];
                         auto i = start;
                         //auto j= multiplication_start;
                         while (i<end)
                         {
                            auto mat_index = d_matrix_col[i];
                            auto mat_value = d_matrix_values[i];
                            
                            for (int j=multiplication_start;j<multiplication_end;j++)
                            {
                                auto multi_index = d_intermediate_row[j];
                                auto multi_value = d_intermediate_values[j];
                                if (multi_index == i)
                                {
                                    product += mat_value*multi_value;
                                }
                            }
                            i++;
                        }
                        if (std::abs(product) >= std::numeric_limits<VTYPE>::min())  
                        {   
                            d_res_values[placeHold + count] = product;
                            d_res_col[placeHold + count] = k;
                            count++;
                        }   
                    }
            });
        };
        defaultQueue.submit(cg).wait();
        */
        // Iterate over each row of the initial matrix
        
        for (ITYPE j = 0; j < rows_; ++j) 
        {
            res.matrix_csr.values_.emplace_back(d_res_values[j]);
            res.matrix_csr.col_indices_.emplace_back(d_res_col[j]);
        }
        for (ITYPE j = 0; j < size_rows; ++j) 
        {
            res.matrix_csr.row_ptrs_[j] = d_res_row[j];
        }
       // res.matrix_csr.row_ptrs_[i+1]=res.matrix_csr.values_.size();
        res.nnz_ = res.matrix_csr.values_.size();
    }

    else if (matrix_format_ == format::CSC) 
    {

        // Perform element-wise multiplication of the CSC vectors
        res.matrix_csc.values_.reserve((matrix_csc.values_.size() <= multiplicand.matrix_csc.values_.size()) ? multiplicand.matrix_csc.values_.size() : matrix_csc.values_.size());
        res.matrix_csc.row_indices_.reserve((matrix_csc.row_indices_.size() <= multiplicand.matrix_csc.row_indices_.size()) ? multiplicand.matrix_csc.row_indices_.size() : matrix_csc.row_indices_.size());
        res.matrix_csc.col_ptrs_.resize(cols_+1);

        // Initialize a vector to store the intermediate results
        std::vector<VTYPE> intermediate(rows_, 0.0);

        res.matrix_csc.col_ptrs_[0] = 0;

        // Iterate over each column of the initial matrix
        for (ITYPE j = 0; j < multiplicand.cols_; ++j) {
            // Clear the intermediate results vector for each row
            std::fill(intermediate.begin(), intermediate.end(), 0.0);

            ITYPE multiplicand_start = multiplicand.matrix_csc.col_ptrs_[j];
            ITYPE multiplicand_end = multiplicand.matrix_csc.col_ptrs_[j + 1];

            // Iterate over the non-zero elements of the cloumn
            for (ITYPE k = multiplicand_start; k < multiplicand_end; ++k) {
                // Get the row index and value of the element
                ITYPE multiplicand_row = multiplicand.matrix_csc.row_indices_[k];
                VTYPE multiplicand_value = multiplicand.matrix_csc.values_[k];

                ITYPE start = matrix_csc.col_ptrs_[multiplicand_row];
                ITYPE end = matrix_csc.col_ptrs_[multiplicand_row + 1];

                // Multiply the element with the corresponding column of the other matrix
                for (ITYPE i = start; i < end; ++i) {
                    ITYPE row = matrix_csc.row_indices_[i];
                    VTYPE value = matrix_csc.values_[i];
                    intermediate[row] += value * multiplicand_value;
                }
            }

            // Add the intermediate results to the result vectors
            for (ITYPE i = 0; i < multiplicand.rows_; ++i) {
                VTYPE result_value = intermediate[i];
                if (std::abs(result_value) >= std::numeric_limits<VTYPE>::min()) {
                    res.matrix_csc.values_.emplace_back(result_value);
                    res.matrix_csc.row_indices_.emplace_back(i);
                }
            }
            res.matrix_csc.col_ptrs_[j+1]=res.matrix_csc.values_.size();
        }

        res.nnz_ = res.matrix_csc.values_.size();

    } 
    else
    {
        std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised matrix format for matrix operator*()" << std::endl;
        std::cerr << "    Please set the matrix_format_ as:" << std::endl;
        std::cerr << "    format::COO: COOrdinate format" << std::endl;
        std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
        std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
        std::abort();
    }

    return res;
}

// Member function of dot product
template <typename ITYPE, typename VTYPE>
VTYPE sparse_matrix<ITYPE,VTYPE>::dot_product(sparse_matrix<ITYPE,VTYPE> &exist_mat) const 
{
    assert(((cols_ == 1)&&(exist_mat.cols_ == 1)) &&
        "MUI Error [matrix_arithmetic.h]: dot_product function only works for column vectors");

    sparse_matrix<ITYPE,VTYPE> tempThis(*this);
    
    sparse_matrix<ITYPE,VTYPE> thisT(tempThis.transpose());
    
    sparse_matrix<ITYPE,VTYPE> tempMat(thisT ^ exist_mat);
    assert(((tempMat.get_rows() == 1)&&(tempMat.get_cols() == 1)) &&
                    "MUI Error [matrix_arithmetic.h]: result of dot_product function should be a scalar");
    return (tempMat.get_value(0,0));
}


template <typename ITYPE, typename VTYPE>
VTYPE sparse_matrix<ITYPE,VTYPE>::sycl_dot_product(sycl::queue defaultQueue, sparse_matrix<ITYPE,VTYPE> &exist_mat)
{
    assert(((cols_ == 1)&&(exist_mat.cols_ == 1)) &&
        "MUI Error [matrix_arithmetic.h]: dot_product function only works for column vectors");
    VTYPE product;
    
    product = sycl_dotp_vec_vec(defaultQueue, this->matrix_sycl.vector_val,exist_mat.matrix_sycl.vector_val,exist_mat.get_rows());
    
    return (product);
}

template<typename ITYPE, typename VTYPE>
VTYPE sparse_matrix<ITYPE,VTYPE>::sycl_dotp_vec_vec(sycl::queue defaultQueue, VTYPE *vec1_value, VTYPE *vec2_value,  ITYPE size_row) 
{
    size_t rows = size_row;
    VTYPE *prod;
    prod = (VTYPE *)malloc(1);
    prod[0] = 0.;
    VTYPE *dotp;
    dotp = sycl::malloc_device<VTYPE>(1,defaultQueue);
    defaultQueue.memcpy(dotp,prod,(sizeof(VTYPE))).wait();
    auto chg = [&](sycl::handler &hc)
    {
        hc.parallel_for(sycl::range(rows),[=](sycl::id<1> idx) 
        {
            auto product = 0.;
            product = vec1_value[idx] * vec2_value[idx];
            auto v = sycl::atomic_ref<
                         VTYPE, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>(*dotp);
            v.fetch_add(product);
        });
    };
    defaultQueue.submit(chg).wait();
    
    defaultQueue.memcpy(prod,dotp,(sizeof(VTYPE))).wait();
    //std::cout<<" Dot product value : "<< prod[0] <<std::endl;
    return (prod[0]);
}


template<typename ITYPE, typename VTYPE>
VTYPE sparse_matrix<ITYPE,VTYPE>::sycl_dotp_red_vec_vec(sycl::queue defaultQueue, VTYPE *vec1_value, VTYPE *vec2_value,  ITYPE size_row) 
{
      
      sycl::nd_range<1> range{5000,1000};
      sycl::buffer<VTYPE> sum{1};
      auto init = sycl::property::reduction::initialize_to_identity{};
      auto chg = [&](sycl::handler& h)
      {
          
          auto reductor = sycl::reduction(sum, h, VTYPE{0.0}, std::plus<VTYPE>(), init);
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

// Member function of Hadamard product
template <typename ITYPE, typename VTYPE>
sparse_matrix<ITYPE,VTYPE> sparse_matrix<ITYPE,VTYPE>::hadamard_product(sparse_matrix<ITYPE,VTYPE> &exist_mat) 
{
    
    if (rows_ != exist_mat.rows_ || cols_ != exist_mat.cols_) 
    {
        std::cerr << "MUI Error [matrix_arithmetic.h]: matrix size mismatch during matrix Hadamard product" << std::endl;
        std::abort();
    }
    
    if (exist_mat.matrix_format_ != matrix_format_)
    {
        exist_mat.format_conversion(this->get_format(), true, true, "overwrite");
    } 
    else 
    {
        if (!exist_mat.is_sorted_unique("matrix_arithmetic.h", "hadamard_product()")){
            if (exist_mat.matrix_format_ == format::COO) {
                exist_mat.sort_coo(true, true, "overwrite");
            } else if (exist_mat.matrix_format_ == format::CSR) {
                exist_mat.sort_csr(true, "overwrite");
            } else if (exist_mat.matrix_format_ == format::CSC) {
                exist_mat.sort_csc(true, "overwrite");
            } else {
                std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised exist_mat matrix format for matrix hadamard_product()" << std::endl;
                std::cerr << "    Please set the exist_mat matrix_format_ as:" << std::endl;
                std::cerr << "    format::COO: COOrdinate format" << std::endl;
                std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
                std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
                std::abort();
            }
        }
    }

    if (!this->is_sorted_unique("matrix_arithmetic.h", "hadamard_product()"))
    {
        if (matrix_format_ == format::COO) {
            this->sort_coo(true, true, "overwrite");
        } else if (matrix_format_ == format::CSR) {
            this->sort_csr(true, "overwrite");
        } else if (matrix_format_ == format::CSC) {
            this->sort_csc(true, "overwrite");
        } else {
            std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised matrix format for matrix hadamard_product()" << std::endl;
            std::cerr << "    Please set the matrix_format_ as:" << std::endl;
            std::cerr << "    format::COO: COOrdinate format" << std::endl;
            std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
            std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
            std::abort();
        }
    }

    // Create a new sparse matrix object for the result
    sparse_matrix<ITYPE,VTYPE> res(rows_, cols_, this->get_format());

    if (matrix_format_ == format::COO) {

        // Perform hadamard product of the COO vectors
        res.matrix_coo.values_.reserve(matrix_coo.values_.size() + exist_mat.matrix_coo.values_.size());
        res.matrix_coo.row_indices_.reserve(matrix_coo.row_indices_.size() + exist_mat.matrix_coo.row_indices_.size());
        res.matrix_coo.col_indices_.reserve(matrix_coo.col_indices_.size() + exist_mat.matrix_coo.col_indices_.size());

        // Insert the COO vectors of the initial sparse matrix to the result sparse matrix
        res.matrix_coo.values_ = std::vector<VTYPE>(matrix_coo.values_.begin(), matrix_coo.values_.end());
        res.matrix_coo.row_indices_ = std::vector<ITYPE>(matrix_coo.row_indices_.begin(), matrix_coo.row_indices_.end());
        res.matrix_coo.col_indices_ = std::vector<ITYPE>(matrix_coo.col_indices_.begin(), matrix_coo.col_indices_.end());

        // Append the exist_mat COO vectors to the result sparse matrix
        res.matrix_coo.values_.insert(res.matrix_coo.values_.end(), exist_mat.matrix_coo.values_.begin(), exist_mat.matrix_coo.values_.end());
        res.matrix_coo.row_indices_.insert(res.matrix_coo.row_indices_.end(), exist_mat.matrix_coo.row_indices_.begin(), exist_mat.matrix_coo.row_indices_.end());
        res.matrix_coo.col_indices_.insert(res.matrix_coo.col_indices_.end(), exist_mat.matrix_coo.col_indices_.begin(), exist_mat.matrix_coo.col_indices_.end());

        // Sort and deduplicate the result
        res.sort_coo(true, true, "multiply");

    } 
    else if (matrix_format_ == format::CSR) 
    {

        // Perform element-wise hadamard product of the CSR vectors
        res.matrix_csr.values_.reserve(matrix_csr.values_.size() + exist_mat.matrix_csr.values_.size());
        res.matrix_csr.row_ptrs_.resize(rows_ + 1);
        res.matrix_csr.col_indices_.reserve(matrix_csr.col_indices_.size() + exist_mat.matrix_csr.col_indices_.size());

        res.matrix_csr.row_ptrs_[0] = 0;

        auto defaultQueue = sycl::queue {sycl::default_selector_v};
            
        double *h_matrix_values;
        double *h_exist_values;
        double *h_res_values;

        double *d_matrix_values;
        double *d_exist_values;
        double *d_res_values;

        int *h_matrix_col;
        int *h_exist_col;
        int *h_res_col;

        int *d_matrix_col;
        int *d_exist_col;
        int *d_res_col;
   
        int *h_matrix_row;
        int *h_exist_row;
        int *h_res_row;
        int *h_row_count;
        int *d_matrix_row;
        int *d_exist_row;
        int *d_res_row;
        int *d_row_count;
        
        size_t size_matrix = matrix_csr.values_.size() ;
        size_t size_exist = exist_mat.matrix_csr.values_.size();
        size_t size_res;// = (matrix_csr.values_.size() + addend.matrix_csr.values_.size());

        size_t size_rows = (rows_ + 1);

        size_t size_matrix_col = matrix_csr.col_indices_.size() ;
        size_t size_exist_col = exist_mat.matrix_csr.col_indices_.size() ;
        size_t size_res_col;// = (matrix_csr.col_indices_.size()+addend.matrix_csr.col_indices_.size());

        h_matrix_values = (double *)malloc(size_matrix*sizeof(double));
        h_exist_values  = (double *)malloc(size_exist*sizeof(double));
       
        h_matrix_col    = (int *)malloc(size_matrix_col*sizeof(int));
        h_exist_col     = (int *)malloc(size_exist_col*sizeof(int));
        
        h_matrix_row    = (int *)malloc(size_rows*sizeof(int));
        h_exist_row    = (int *)malloc(size_rows*sizeof(int));
        h_res_row       = (int *)malloc(size_rows*sizeof(int));
        
        h_row_count     = (int *)malloc(rows_*sizeof(int));

        d_matrix_values = sycl::malloc_device<double>(size_matrix,defaultQueue);
        d_exist_values  = sycl::malloc_device<double>(size_exist,defaultQueue);
         
        d_matrix_col    = sycl::malloc_device<int>(size_matrix_col,defaultQueue);
        d_exist_col     = sycl::malloc_device<int>(size_exist_col,defaultQueue);
       
        d_matrix_row    = sycl::malloc_device<int>(size_rows,defaultQueue);
        d_exist_row     = sycl::malloc_device<int>(size_rows,defaultQueue);
        d_res_row       = sycl::malloc_device<int>(size_rows,defaultQueue);
        d_row_count     = sycl::malloc_device<int>(rows_,defaultQueue);

        for (int i=0;i<size_matrix;i++)
        {
            h_matrix_values[i] = matrix_csr.values_.at(i);
            h_matrix_col[i]    = matrix_csr.col_indices_.at(i);
        }
        for (int i=0;i<size_exist;i++)
        {
            h_exist_values[i] = exist_mat.matrix_csr.values_.at(i);
            h_exist_col[i]    = exist_mat.matrix_csr.col_indices_.at(i);
        }

        for (int i=0;i<=rows_;i++)
        {
            h_matrix_row[i] = matrix_csr.row_ptrs_.at(i);
            h_exist_row[i]  = exist_mat.matrix_csr.row_ptrs_.at(i);   
        }

        //auto defaultQueue = sycl::queue{sycl::default_selector_v};
        defaultQueue.memcpy(d_matrix_values,h_matrix_values,(size_matrix*sizeof(double))).wait();
        defaultQueue.memcpy(d_exist_values,h_exist_values,(size_exist*sizeof(double))).wait();
        defaultQueue.memcpy(d_matrix_col,h_matrix_col,(size_matrix*sizeof(int))).wait();
        defaultQueue.memcpy(d_exist_col,h_exist_col,(size_exist*sizeof(int))).wait();
        defaultQueue.memcpy(d_matrix_row,h_matrix_row,((rows_+1)*sizeof(int))).wait();
        defaultQueue.memcpy(d_exist_row,h_exist_row,((rows_+1)*sizeof(int))).wait();

        

        auto cg = [&](sycl::handler &h) 
        {
            
            h.parallel_for(sycl::range(rows_),[=](sycl::id<1> idx) 
            {
                auto startIdx = d_matrix_row[idx];
                auto endIdx = d_matrix_row[idx+1];
                auto startaddIdx = d_exist_row[idx];
                auto endaddIdx = d_exist_row[idx+1];
                auto i = startIdx;
                auto j = startaddIdx;
                auto count = 0;
                auto col = 0;
                auto addend_col = 0;
                i = startIdx;
                while (i < endIdx && j < endaddIdx) 
                {
                    col = d_matrix_col[i];
                    addend_col = d_exist_col[j];
                    if (col == addend_col)
                    {
                        i++;
                        j++;
                        count++;
                    }
                    else if(col < addend_col)
                    {
                        i++;
                    }
                    else
                    {
                        j++;
                    }
                }
                d_row_count[idx] = count;
            });
        };        
        defaultQueue.submit(cg).wait(); 

        auto chg = [&](sycl::handler &gh) 
        {
            gh.parallel_for(sycl::range(size_rows),[=](sycl::id<1> idx) 
            {
                d_res_row[idx] = 0;
                auto count =0;
                if (idx > 0)
                {
                    while (count < idx)
                    {   
                        d_res_row[idx] = d_res_row[idx]+d_row_count[count];
                        count++;
                    }
                } 
            });
        };        
        defaultQueue.submit(chg).wait(); 
        defaultQueue.memcpy(h_res_row,d_res_row,(size_rows*sizeof(int))).wait();
        size_res_col = h_res_row[rows_];

        h_res_values    = (double *)malloc(size_res_col* sizeof(double));
        h_res_col       = (int *)malloc(size_res_col*sizeof(int));

        d_res_values    = sycl::malloc_device<double>(size_res_col,defaultQueue);
        d_res_col       = sycl::malloc_device<int>(size_res_col,defaultQueue);
        
        auto cag = [&](sycl::handler &ga)
        {
            ga.parallel_for(sycl::range(rows_),[=](sycl::id<1>idx)
            {
                auto startIdx = d_matrix_row[idx];
                auto endIdx = d_matrix_row[idx+1];
                auto startaddIdx = d_exist_row[idx];
                auto endaddIdx = d_exist_row[idx+1];
                auto i = startIdx;
                auto j = startaddIdx;
                auto count = 0;
                auto col = 0;
                auto subtrahend_col = 0;
                auto placeHold = d_res_row[idx];
                while (i < endIdx && j < endaddIdx) 
                {
                    col        = d_matrix_col[i];
                    subtrahend_col = d_exist_col[j];
                    if (col == subtrahend_col)
                    {
                        d_res_values[placeHold + count] = d_matrix_values[i] * d_exist_values[j];
                        d_res_col[placeHold + count] = col;
                        i++;
                        j++;
                        count++;
                    }
                    else if (col < subtrahend_col)
                    {
                        i++;
                    }
                    else
                    {
                        j++;
                    }
                }
            });
        };
        defaultQueue.submit(cag).wait(); 

        defaultQueue.memcpy(h_res_values,d_res_values,(size_res_col*sizeof(double))).wait();
        defaultQueue.memcpy(h_res_col,d_res_col,(size_res_col*sizeof(int))).wait();
        
       
       // defaultQueue.memcpy(h_res_values,d_res_values,(size_res*sizeof(double))).wait();             
       // res.matrix_csr.values_.resize(size_res);
       // res.matrix_csr.values_.resize(size_res);
        for (int i=0;i<size_res_col;i++)
        {
            res.matrix_csr.values_.emplace_back(h_res_values[i]);
            res.matrix_csr.col_indices_.emplace_back(h_res_col[i]);
        }
        for (int i=0;i<=rows_;i++)
        {
            res.matrix_csr.row_ptrs_[i] = (h_res_row[i]);
        }
        res.nnz_ = res.matrix_csr.col_indices_.size();
        res.matrix_csr.row_ptrs_[rows_ + 1] = res.nnz_;
    } 
    else if (matrix_format_ == format::CSC) 
    {

        // Perform element-wise hadamard product of the CSC vectors
        res.matrix_csc.values_.reserve(matrix_csc.values_.size() + exist_mat.matrix_csc.values_.size());
        res.matrix_csc.row_indices_.reserve(matrix_csc.row_indices_.size() + exist_mat.matrix_csc.row_indices_.size());
        res.matrix_csc.col_ptrs_.resize(cols_ + 1);

        res.matrix_csc.col_ptrs_[0] = 0;

        ITYPE column = 0;
        while (column < cols_) {
            ITYPE start = matrix_csc.col_ptrs_[column];
            ITYPE end = matrix_csc.col_ptrs_[column + 1];

            ITYPE exist_mat_start = exist_mat.matrix_csc.col_ptrs_[column];
            ITYPE exist_mat_end = exist_mat.matrix_csc.col_ptrs_[column + 1];

            // Merge the values and row indices of the two columns
            ITYPE i = start;
            ITYPE j = exist_mat_start;
            while (i < end && j < exist_mat_end) 
            {
                ITYPE row = matrix_csc.row_indices_[i];
                ITYPE exist_mat_row = exist_mat.matrix_csc.row_indices_[j];

                if ((row == exist_mat_row) && std::abs(matrix_csc.values_[i] * exist_mat.matrix_csc.values_[j]) >= std::numeric_limits<VTYPE>::min()) {
                    // Add the corresponding values if the columns match
                    res.matrix_csc.values_.emplace_back(matrix_csc.values_[i] * exist_mat.matrix_csc.values_[j]);
                    res.matrix_csc.row_indices_.emplace_back(row);
                    i++;
                    j++;
                } else if (row < exist_mat_row) {
                    i++;
                } else {
                    j++;
                }
            }

            // Update the column pointer
            res.nnz_ = res.matrix_csc.row_indices_.size();
            res.matrix_csc.col_ptrs_[column + 1] = res.nnz_;

            column++;
        }

    } else {
        std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised matrix format for matrix hadamard_product()" << std::endl;
        std::cerr << "    Please set the matrix_format_ as:" << std::endl;
        std::cerr << "    format::COO: COOrdinate format" << std::endl;
        std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
        std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
        std::abort();
    }

    return res;
}

// Member function to get transpose of matrix
template <typename ITYPE, typename VTYPE>
sparse_matrix<ITYPE,VTYPE> sparse_matrix<ITYPE,VTYPE>::transpose(bool performSortAndUniqueCheck) const 
{

    sparse_matrix<ITYPE,VTYPE> res(*this);

    if (matrix_format_ == format::COO) 
    {

        if (performSortAndUniqueCheck){
            if (!res.is_sorted_unique("matrix_arithmetic.h", "transpose()")){
                res.sort_coo(true, true, "overwrite");
            }
        }

        res.index_reinterpretation();

    } 
    else if (matrix_format_ == format::CSR) 
    {

        res.format_conversion("CSC", performSortAndUniqueCheck, performSortAndUniqueCheck, "overwrite");

        res.format_reinterpretation();

    } 
    else if (matrix_format_ == format::CSC) 
    {

        res.format_conversion("CSR", performSortAndUniqueCheck, performSortAndUniqueCheck, "overwrite");

        res.format_reinterpretation();

    } 
    else 
    {
        std::cerr << "MUI Error [matrix_arithmetic.h]: Unrecognised matrix format for matrix transpose()" << std::endl;
        std::cerr << "    Please set the matrix_format_ as:" << std::endl;
        std::cerr << "    format::COO: COOrdinate format" << std::endl;
        std::cerr << "    format::CSR (default): Compressed Sparse Row format" << std::endl;
        std::cerr << "    format::CSC: Compressed Sparse Column format" << std::endl;
        std::abort();
    }

    return res;

}

// Member function to perform LU decomposition
template <typename ITYPE, typename VTYPE>
void sparse_matrix<ITYPE,VTYPE>::lu_decomposition(sparse_matrix<ITYPE,VTYPE> &L, sparse_matrix<ITYPE,VTYPE> &U) const 
{
    if (((L.get_rows() != 0) && (L.get_rows() != rows_)) ||
        ((U.get_rows() != 0) && (U.get_rows() != rows_)) ||
        ((L.get_cols() != 0) && (L.get_cols() != cols_)) ||
        ((U.get_cols() != 0) && (U.get_cols() != cols_))) {
        std::cerr << "MUI Error [matrix_arithmetic.h]: L & U Matrices must be null or same size of initial matrix in LU decomposition" << std::endl;
        std::abort();
    }

    if ((!L.empty()) || (!U.empty())) {
        std::cerr << "MUI Error [matrix_arithmetic.h]: L & U Matrices must be empty in LU decomposition" << std::endl;
        std::abort();
    }

    if (rows_ != cols_) {
        std::cerr << "MUI Error [matrix_arithmetic.h]: Only square matrix can perform LU decomposition" << std::endl;
        std::abort();
    }

    if ((L.get_rows() != rows_) || (L.get_cols() != cols_)) {
        // Resize the lower triangular matrix
        L.resize(rows_, cols_);
    }

    if ((U.get_rows() != rows_) || (U.get_cols() != cols_)) {
        // Resize the upper triangular matrix
        U.resize(rows_, cols_);
    }

    ITYPE n = rows_;
    for (ITYPE i = 0; i < rows_; ++i) {
        // Calculate the upper triangular matrix
        for (ITYPE k = i; k < cols_; ++k) {
            VTYPE sum = 0.0;
            for (ITYPE j = 0; j < i; ++j) {
                sum += L.get_value(i, j) * U.get_value(j, k);
            }
            U.set_value(i, k, (this->get_value(i, k) - sum));
        }

        // Calculate the lower triangular matrix
        for (ITYPE k = i; k < rows_; k++) {
            if (i == k) {
                L.set_value(i, i, static_cast<VTYPE>(1.0));
            } else {
                VTYPE sum = 0.0;
                for (ITYPE j = 0; j < i; ++j) {
                    sum += L.get_value(k, j) * U.get_value(j, i);
                }
                assert(std::abs(U.get_value(i, i)) >= std::numeric_limits<VTYPE>::min() &&
                                  "MUI Error [matrix_arithmetic.h]: Divide by zero assert for U.get_value(i, i)");
                L.set_value(k, i, (this->get_value(k, i) - sum) / U.get_value(i, i));
            }
        }
    }
}

// Member function to perform QR decomposition
template <typename ITYPE, typename VTYPE>
void sparse_matrix<ITYPE,VTYPE>::qr_decomposition(sparse_matrix<ITYPE,VTYPE> &Q, sparse_matrix<ITYPE,VTYPE> &R) const 
{
    if (((Q.get_rows() != 0) && (Q.get_rows() != rows_)) ||
        ((R.get_rows() != 0) && (R.get_rows() != rows_)) ||
        ((Q.get_cols() != 0) && (Q.get_cols() != cols_)) ||
        ((R.get_cols() != 0) && (R.get_cols() != cols_))) {
        std::cerr << "MUI Error [matrix_arithmetic.h]: Q & R Matrices must be null in QR decomposition" << std::endl;
        std::abort();
    }
    if ((!Q.empty()) || (!R.empty())) {
        std::cerr << "MUI Error [matrix_arithmetic.h]: Q & R Matrices must be empty in QR decomposition" << std::endl;
        std::abort();
    }
    assert((rows_ >= cols_) &&
          "MUI Error [matrix_arithmetic.h]: number of rows of matrix should larger or equals to number of columns in QR decomposition");

    if ((Q.get_rows() != rows_) || (Q.get_cols() != cols_)) {
        // Resize the orthogonal matrix
        Q.resize(rows_, cols_);
    }
    if ((R.get_rows() != rows_) || (R.get_cols() != cols_)) {
        // Resize the upper triangular matrix
        R.resize(rows_, cols_);
    }

    // Get a copy of the matrix
    sparse_matrix<ITYPE,VTYPE> mat_copy (*this);
    // Diagonal elements
    std::vector<VTYPE> r_diag (cols_);

    // Calculate the diagonal element values
    for (ITYPE c = 0; c <cols_; ++c)  
    {
        VTYPE  nrm (0.0);

       // Compute 2-norm of k-th column without under/overflow.
        for (ITYPE r = c; r < rows_; ++r)
            nrm = std::sqrt((nrm * nrm) + (mat_copy.get_value(r, c) * mat_copy.get_value(r, c)));

        if (nrm != static_cast<VTYPE>(0.0))  {

           // Form k-th Householder vector.
            if (mat_copy.get_value(c, c) < static_cast<VTYPE>(0.0))
                nrm = -nrm;

            for (ITYPE r = c; r < rows_; ++r)
                mat_copy.set_value(r, c, (mat_copy.get_value(r, c)/nrm));

            mat_copy.set_value(c, c, (mat_copy.get_value(c, c) + static_cast<VTYPE>(1.0)));

           // Apply transformation to remaining columns.
            for (ITYPE j = c + 1; j < cols_; ++j)  {
                VTYPE  s = 0.0;

                for (ITYPE r = c; r < rows_; ++r)
                    s += mat_copy.get_value(r, c) * mat_copy.get_value(r, j);

                s /= -mat_copy.get_value(c, c);
                for (ITYPE r = c; r < rows_; ++r)
                    mat_copy.set_value(r, j, (mat_copy.get_value(r, j) + s * mat_copy.get_value(r, c)));
            }
        }
        r_diag[c] = -nrm;
    }

    // Calculate the orthogonal matrix
    for (ITYPE c = cols_ - 1; c >= 0; --c)  
    {
        Q.set_value(c, c, static_cast<VTYPE>(1.0));

        for (ITYPE cc = c; cc < cols_; ++cc)
            if (mat_copy.get_value(c, c) != static_cast<VTYPE>(0.0)) {
                VTYPE s=0.0;

                for (ITYPE r = c; r < rows_; ++r)
                    s += mat_copy.get_value(r, c) * Q.get_value(r, cc);

                s /= -mat_copy.get_value(c, c);
                for (ITYPE r = c; r < rows_; ++r)
                    Q.set_value(r, cc, (Q.get_value(r, cc) + s * mat_copy.get_value(r, c)));
            }
    }

    // Calculate the upper triangular matrix
    for (ITYPE c = 0; c < cols_; ++c)
        for (ITYPE r = 0; r < rows_; ++r)
            if (c < r)
                R.set_value(c, r, mat_copy.get_value(c, r));
            else if (c == r)
                R.set_value(c, r, r_diag[c]);
}

// Member function to get the inverse of matrix by using Gaussian elimination
template <typename ITYPE, typename VTYPE>
sparse_matrix<ITYPE,VTYPE> sparse_matrix<ITYPE,VTYPE>::inverse() const 
{
    if (rows_ != cols_) 
    {
        std::cerr << "MUI Error [matrix_arithmetic.h]: Matrix must be square to find its inverse" << std::endl;
        std::abort();
    }

    sparse_matrix<ITYPE,VTYPE> mat_copy (*this);
    sparse_matrix<ITYPE,VTYPE> inverse_mat (rows_,"identity", this->get_format());

    for (ITYPE r = 0; r < rows_; ++r)  {

        ITYPE max_row = r;
        VTYPE max_value= static_cast<VTYPE>(-1.0);

        // Partial pivoting for Gaussian elimination
        ITYPE ppivot;
        for (ITYPE rb = r; rb < rows_; ++rb)  
        {
            const VTYPE tmp = std::abs(mat_copy.get_value(rb, r));
            if ((tmp > max_value) && (std::abs(tmp) >= std::numeric_limits<VTYPE>::min()))  
            {
                max_value = tmp;
                max_row = rb;
            }
        }

        assert(std::abs(mat_copy.get_value(max_row, r)) >= std::numeric_limits<VTYPE>::min() &&
                          "MUI Error [matrix_arithmetic.h]: Divide by zero assert for mat_copy.get_value(max_row, r). Cannot perform matrix invert due to singular matrix.");

        if (max_row != r)  
        {
            for (ITYPE c = 0; c < cols_; ++c)
                mat_copy.swap_elements(r, c, max_row, c);
            ppivot = max_row;
        } 
        else 
        {
            ppivot = 0;
        }

        const ITYPE indx = ppivot;

        if (indx != 0)
            for (ITYPE c = 0; c < cols_; ++c)
                inverse_mat.swap_elements(r, c, indx, c);

        const VTYPE diag = mat_copy.get_value(r, r);

        for (ITYPE c = 0; c < cols_; ++c)  {
            mat_copy.set_value(r, c, (mat_copy.get_value(r, c) / diag));
            inverse_mat.set_value(r, c, (inverse_mat.get_value(r, c) / diag));
        }

        for (ITYPE rr = 0; rr < rows_; ++rr)
            if (rr != r)  
            {
                const VTYPE off_diag = mat_copy.get_value(rr, r);

                for (ITYPE c = 0; c < cols_; ++c)  
                {
                    mat_copy.set_value(rr, c, (mat_copy.get_value(rr, c) - off_diag * mat_copy.get_value(r, c)));
                    inverse_mat.set_value(rr, c, (inverse_mat.get_value(rr, c) - off_diag * inverse_mat.get_value(r, c)));
                }
            }
    }
    return inverse_mat;
}

// **************************************************
// ********** Protected member functions ************
// **************************************************

// Protected member function to reinterpret the row and column indexes for sparse matrix with COO format - helper function on matrix transpose
template<typename ITYPE, typename VTYPE>
void sparse_matrix<ITYPE,VTYPE>::index_reinterpretation() 
{
    assert((matrix_format_ == format::COO) &&
              "MUI Error [matrix_arithmetic.h]: index_reinterpretation() is for COO format only.");

    std::swap(matrix_coo.row_indices_, matrix_coo.col_indices_);
    ITYPE temp_index = rows_;
    rows_ = cols_;
    cols_ = temp_index;

}

// Protected member function to reinterpret the format of sparse matrix between CSR format and CSC format - helper function on matrix transpose
template<typename ITYPE, typename VTYPE>
void sparse_matrix<ITYPE,VTYPE>::format_reinterpretation() 
{
    assert(((matrix_format_ == format::CSR) || (matrix_format_ == format::CSC)) &&
              "MUI Error [matrix_arithmetic.h]: format_reinterpretation() is for CSR or CSC format.");

    if (matrix_format_ == format::CSR) 
    {

        matrix_csc.col_ptrs_.swap(matrix_csr.row_ptrs_);
        matrix_csc.row_indices_.swap(matrix_csr.col_indices_);
        matrix_csc.values_.swap(matrix_csr.values_);

        ITYPE temp_index = rows_;
        rows_ = cols_;
        cols_ = temp_index;

        matrix_format_ = format::CSC;

        matrix_csr.row_ptrs_.clear();
        matrix_csr.col_indices_.clear();
        matrix_csr.values_.clear();
    } 

    else if (matrix_format_ == format::CSC) 
    {

        matrix_csr.row_ptrs_.swap(matrix_csc.col_ptrs_);
        matrix_csr.col_indices_.swap(matrix_csc.row_indices_);
        matrix_csr.values_.swap(matrix_csc.values_);

        ITYPE temp_index = rows_;
        rows_ = cols_;
        cols_ = temp_index;

        matrix_format_ = format::CSR;

        matrix_csc.col_ptrs_.clear();
        matrix_csc.row_indices_.clear();
        matrix_csc.values_.clear();

    }

}

} // linalg
} // mui

#endif /* MUI_MATRIX_ARITHMETIC_H_ */
