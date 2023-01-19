//////////////////////////////////////////////////////////////////////////////
//
// Copyright (c) 2018 Mellanox Technologies LTD. All rights reserved.
//
// This program is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
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

package DCTrafficGen;

simple DCTrafficGen
{
    parameters:
      // the time to start generating work
      double startTime @unit(s) = default(0s);
      // the time to stop generating work
      double stopTime @unit(s) = default(10000s);

      // since the generator generates next message in a flow when "done"
      // message is received, it may be required that at the flow beggining
      // we post multiple messages for example to enable sending them in burst
      // the parameter define how many messages to post immediatly at the
      // beggining of a flow
      int numPostedMessages = default(2);
      int numOfFlows = default(3);

      // link BW
      double linkBW_Bps = default(12.5e9); // for 100Gbps
      // Message size to generate
      volatile int pktSize @unit(B) = default(1KiB);
	  
	  
      // app configuration to DCTrafficGen
      string appXmlConfig;

      // statistics

      // statistics collection period
      double statCollPeriod @unit(s) = default(1ms);
      // start statistics collection
      double startColl @unit(s) = default(0s);
      // end statistics collection
      double endColl @unit(s) = default(10000s);
      // signal based
      @statistic[genBWVecMBps](record=max,min,mean,vector);
      @statistic[numMsgs](record=mean,vector);
      @statistic[flowDuration](record=mean,vector);
      @statistic[interArival](record=mean,vector);
      @statistic[msgSize](record=mean,vector);
      @statistic[flowRate](record=max,min,mean,vector);

   gates:
      inout out; // send msgs and receive "done" when msg sent
}
