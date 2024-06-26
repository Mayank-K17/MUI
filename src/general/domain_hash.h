/*****************************************************************************
* Multiscale Universal Interface Code Coupling Library                       *
*                                                                            *
* Copyright (C) 2019 Y. H. Tang, S. Kudo, X. Bian, Z. Li, G. E. Karniadakis, *
*                    R. W. Nash                                              *
*                                                                            *
* (* The University of Edinburgh)                                            *
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
 * @file domain_hash.h
 * @author Mayank Kumar <mayank.kumar@stfc.ac.uk>
 * @date 05 Febuary 2024
 * @brief Hash domain for the neighbor search.
*/

#ifndef DOMAIN_HASH_H_
#define DOMAIN_HASH_H_
#include "util.h"
#include "hash_map.h"
namespace mui {


template<typename CONFIG = default_config>
class domain_hash
{
    
    double extend[6];
    int Nx[3];
    double dX;
    public:
        std::vector<hash_map> hash_table;
        domain_hash(std::pair<typename CONFIG::point_type, typename CONFIG::point_type> lbb, double r):
            dX(r)
        {
            extend[0] = (lbb.first[0]-(1.5* dX));
            extend[1] = (lbb.second[0]+(1.5*dX));
            extend[2] = (lbb.first[1]-(1.5*dX));
            extend[3] = (lbb.second[1]+(1.5*dX));
            extend[4] = (lbb.first[2]-(1.5*dX));
            extend[5] = (lbb.second[2]+(1.5*dX));

            
            Nx[0] = (extend[1]-extend[0])/dX;
            Nx[1] = (extend[3]-extend[2])/dX;
            Nx[2] = (extend[5]-extend[4])/dX;
            if (Nx[0] == 0)
            {
                Nx[0] = 1;
            }
            if (Nx[1] == 0)
            {
                Nx[1] = 1;
            }
            if (Nx[2] == 0)
            {
                Nx[2] = 1;
            }
            size_t hash_size = Nx[0] * Nx[1] * Nx[2];
            
            hash_table.reserve(hash_size);
            hash_map hash_value;
            for (int i=0;i<Nx[0];i++)
            {
                for (int j=0;j<Nx[1];j++)
                {
                    for (int k=0;k<Nx[2];k++)
                    {
                        hash_value.map_index(extend,dX,i,j,k,Nx[0],Nx[1],Nx[2]);
                        hash_table.emplace_back(hash_value);
                    }
                }
            }
           // std::cout<<"Size of hash table = "<<hash_table.size()<<std::endl<<"Nx = "<<Nx << "Ny = "<<Ny <<"Nz = "<<Nz<<std::endl<<" and dX = "<<dX<<std::endl;
        }
        ~domain_hash()
        {
            
        }
        int set_point_hash(double,double,double,int,bool);
        void print_hash_size();
        void Voxelize();
};

template <typename CONFIG> inline int domain_hash<CONFIG>::set_point_hash(double X, double Y, double Z, int ind, bool HPoint)
{
    int divX = (X-extend[0])/dX;
    int divY = (Y-extend[2])/dX;
    int divZ = (Z-extend[4])/dX;
    int index;
    index = divZ + Nx[2]*divY + (Nx[2]*Nx[1]*divX);
    if (HPoint)
    {
        hash_table.at(index).data_points.emplace_back(ind);
    }
    else
    {
        hash_table.at(index).ptsExtended.emplace_back(ind);
    }
    return(index);
}
template <typename CONFIG> inline void domain_hash<CONFIG>::print_hash_size()
{
    for (int i = 0;i<hash_table.size();i++)
    {
        if (hash_table.at(i).data_points.size() != 0 && hash_table.at(i).interiorVoxel == false)
        std::cout<<" hash table point size = "<<hash_table.at(i).data_points.size();
        if (hash_table.at(i).ptsExtended.size() != 0 && hash_table.at(i).interiorVoxel == false)
        std::cout<<" hash table pointExt size = "<<hash_table.at(i).ptsExtended.size();
    }
}
template <typename CONFIG> inline void domain_hash<CONFIG>::Voxelize()
{
    int nInd;
    
    for (int i = 0;i<hash_table.size();i++)
    {
        hash_table.at(i).Voxel.push_back(i);
        if (hash_table.at(i).interiorVoxel)
        {
            
            nInd = i-(Nx[1]*Nx[2]);
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+(Nx[1]*Nx[2]);
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i-(Nx[2]);
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+(Nx[2]);
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i-1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i-(Nx[1]*Nx[2])-Nx[2];
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i-(Nx[1]*Nx[2])+Nx[2];
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+(Nx[1]*Nx[2])-Nx[2];
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+(Nx[1]*Nx[2])+Nx[2];
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i-Nx[2]-1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i-Nx[2]+1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+Nx[2]-1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+Nx[2]+1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i-(Nx[1]*Nx[2])-1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i-(Nx[1]*Nx[2])+1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+(Nx[1]*Nx[2])-1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+(Nx[1]*Nx[2])+1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i-(Nx[1]*Nx[2])-Nx[2]-1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i-(Nx[1]*Nx[2])-Nx[2]+1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i-(Nx[1]*Nx[2])+Nx[2]-1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i-(Nx[1]*Nx[2])+Nx[2]+1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+(Nx[1]*Nx[2])-Nx[2]-1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+(Nx[1]*Nx[2])-Nx[2]+1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+(Nx[1]*Nx[2])+Nx[2]-1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
            nInd = i+(Nx[1]*Nx[2])+Nx[2]+1;
            if (nInd >=0 && nInd < hash_table.size())
                hash_table.at(i).Voxel.push_back(nInd);
        }
    }
}
}
#endif