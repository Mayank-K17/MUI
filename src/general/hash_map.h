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
 * @file hash_map.h
 * @author Mayank Kumar <mayank.kumar@stfc.ac.uk>
 * @date 05 Febuary 2024
 * @brief Hash map for the neighbor search.
*/

#ifndef HASH_MAP_H_
#define HASH_MAP_H_
#include "util.h"
class hash_map
{
   public: 
    int index;
    double extends[6];
    bool interiorVoxel;
    std::vector<int> Voxel;
    std::vector<int> data_points;
    std::vector<int> ptsExtended;
    hash_map();
    ~hash_map();
    void map_index(double[6], double,int, int, int, int,int,int );
};
hash_map::hash_map()
{
    index = -1;
    interiorVoxel = false;
}
hash_map::~hash_map()
{

}
void hash_map::map_index( double ext[6], double dX, int i, int j, int k, int Nx,int Ny,int Nz)
{
    index = k + (Nz)*j + ((Nz)*(Ny))*i;
    extends[0] = ext[0] + i*dX;
    extends[1] = ext[0] + (i+1)*dX;
    extends[2] = ext[2] + j*dX;
    extends[3] = ext[2] + (j+1)*dX;
    extends[4] = ext[4] + k*dX;
    extends[5] = ext[4] + (k+1)*dX;
    if (extends[1] > ext[1])
        extends[1] = ext[1];
    if (extends[3] > ext[3])
        extends[3] = ext[3];
    if (extends[5] > ext[5])
        extends[5] = ext[5];
    if (i != 0 || j != 0 || k != 0 || i != (Nx-1) || j != (Ny-1) || k != (Nz-1) )
    {
        interiorVoxel = true;
    }
}
#endif