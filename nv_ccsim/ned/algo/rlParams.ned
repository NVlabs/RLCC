//
// This program is property of its copyright holder. All rights reserved.
// 

package prog_cc;
//////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2019-2020 Mellanox Technologies LTD. All rights reserved.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2
// of the License, or (at your option) any later version.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, see <http://www.gnu.org/licenses/>.
//
//     Redistribution and use in source and binary forms, with or
//     without modification, are permitted provided that the following
//     conditions are met:
//
//      - Redistributions of source code must retain the above
//        copyright notice, this list of conditions and the following
//        disclaimer.
//
//      - Redistributions in binary form must reproduce the above
//        copyright notice, this list of conditions and the following
//        disclaimer in the documentation and/or other materials
//        provided with the distribution.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//////////////////////////////////////////////////////////////////////////////


//
// A CC Algorithm Interface
//
module rlParams like algoParams
{
    parameters:
        @display("i=block/join");
	  // algo parameters are declared in 2 files:
	  // this file
	  // the <algo>_algo_paramaters.h in the FW/Algorithms
	  
	  // The order in which the parameters are declared in the struct
	  // <algo>_algo_paramaters.h
        string orderedParamNames = 
 	     "PORT_NO_TO_PYTHON";

	  //

		int PORT_NO_TO_PYTHON          = default(5555);             // PORT TO PYTHON SERVER WITH THE NETWORK

}
