from agents.adpg import ADPG
from agents.dqn import DQN
from agents.ppo import PPO
from agents.random import RandomAgent
from agents.supervised import Supervised

AGENTS = {'PPO': PPO, 'DQN': DQN, 'SUPERVISED': Supervised, 'ADPG': ADPG, 'RANDOM': RandomAgent}

"""
2_1_qp 4_1_qp 8_1_qp 2_4_a2a_qp 4_8_a2a_qp 8_16_a2a_qp 32_1_qp
PerDest ManyToOne
- Run 0: $0=5555, $1=2^(20), $H=2, $Q=1, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 1: $0=5555, $1=2^(20), $H=2, $Q=2, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 2: $0=5555, $1=2^(20), $H=2, $Q=8, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 3: $0=5555, $1=2^(20), $H=2, $Q=32, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 4: $0=5555, $1=2^(20), $H=2, $Q=64, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 5: $0=5555, $1=2^(20), $H=2, $Q=128, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 6: $0=5555, $1=2^(20), $H=2, $Q=256, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 7: $0=5555, $1=2^(20), $H=2, $Q=512, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 8: $0=5555, $1=2^(20), $H=2, $Q=1024, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
- Run 9: $0=5555, $1=2^(20), $H=4, $Q=1, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 10: $0=5555, $1=2^(20), $H=4, $Q=2, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 11: $0=5555, $1=2^(20), $H=4, $Q=8, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 12: $0=5555, $1=2^(20), $H=4, $Q=16, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 13: $0=5555, $1=2^(20), $H=4, $Q=64, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 14: $0=5555, $1=2^(20), $H=4, $Q=128, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 15: $0=5555, $1=2^(20), $H=4, $Q=256, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 16: $0=5555, $1=2^(20), $H=4, $Q=512, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 17: $0=5555, $1=2^(20), $H=4, $Q=1024, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 18: $0=5555, $1=2^(20), $H=8, $Q=1, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 19: $0=5555, $1=2^(20), $H=8, $Q=2, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 20: $0=5555, $1=2^(20), $H=8, $Q=8, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 21: $0=5555, $1=2^(20), $H=8, $Q=32, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 22: $0=5555, $1=2^(20), $H=8, $Q=64, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 23: $0=5555, $1=2^(20), $H=8, $Q=128, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 24: $0=5555, $1=2^(20), $H=8, $Q=256, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 25: $0=5555, $1=2^(20), $H=8, $Q=512, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 26: $0=5555, $1=2^(20), $H=8, $Q=1024, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 27: $0=5555, $1=2^(20), $H=16, $Q=1, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 28: $0=5555, $1=2^(20), $H=16, $Q=2, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 29: $0=5555, $1=2^(20), $H=16, $Q=8, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 30: $0=5555, $1=2^(20), $H=16, $Q=32, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 31: $0=5555, $1=2^(20), $H=16, $Q=64, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 32: $0=5555, $1=2^(20), $H=16, $Q=128, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 33: $0=5555, $1=2^(20), $H=16, $Q=256, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 34: $0=5555, $1=2^(20), $H=16, $Q=512, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 35: $0=5555, $1=2^(20), $H=32, $Q=1, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 36: $0=5555, $1=2^(20), $H=32, $Q=2, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 37: $0=5555, $1=2^(20), $H=32, $Q=8, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 38: $0=5555, $1=2^(20), $H=32, $Q=32, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 39: $0=5555, $1=2^(20), $H=32, $Q=64, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 40: $0=5555, $1=2^(20), $H=32, $Q=128, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 41: $0=5555, $1=2^(20), $H=32, $Q=256, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 42: $0=5555, $1=2^(20), $H=64, $Q=1, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 43: $0=5555, $1=2^(20), $H=64, $Q=2, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
-Run 44: $0=5555, $1=2^(20), $H=64, $Q=8, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 45: $0=5555, $1=2^(20), $H=64, $Q=32, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 46: $0=5555, $1=2^(20), $H=64, $Q=64, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 47: $0=5555, $1=2^(20), $H=64, $Q=128, $AlgoVerbose=false, $F=16, $B=20, $repetition=0
"""

"""
Config: AllToAll
Number of runs: 27
Run 0: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=4, $QPP=1, $Q=2*(4)*1, $DST=int(index/1), $AlgoVerbose=false, $repetition=0
Run 1: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=4, $QPP=2, $Q=2*(4)*2, $DST=int(index/2), $AlgoVerbose=false, $repetition=0
Run 2: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=4, $QPP=8, $Q=2*(4)*8, $DST=int(index/8), $AlgoVerbose=false, $repetition=0
Run 3: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=4, $QPP=16, $Q=2*(4)*16, $DST=int(index/16), $AlgoVerbose=false, $repetition=0
Run 4: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=4, $QPP=32, $Q=2*(4)*32, $DST=int(index/32), $AlgoVerbose=false, $repetition=0
Run 5: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=4, $QPP=64, $Q=2*(4)*64, $DST=int(index/64), $AlgoVerbose=false, $repetition=0
Run 6: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=4, $QPP=128, $Q=2*(4)*128, $DST=int(index/128), $AlgoVerbose=false, $repetition=0
Run 7: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=4, $QPP=256, $Q=2*(4)*256, $DST=int(index/256), $AlgoVerbose=false, $repetition=0
Run 8: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=4, $QPP=512, $Q=2*(4)*512, $DST=int(index/512), $AlgoVerbose=false, $repetition=0
Run 9: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=8, $QPP=1, $Q=2*(8)*1, $DST=int(index/1), $AlgoVerbose=false, $repetition=0
Run 10: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=8, $QPP=2, $Q=2*(8)*2, $DST=int(index/2), $AlgoVerbose=false, $repetition=0
Run 11: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=8, $QPP=8, $Q=2*(8)*8, $DST=int(index/8), $AlgoVerbose=false, $repetition=0
Run 12: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=8, $QPP=16, $Q=2*(8)*16, $DST=int(index/16), $AlgoVerbose=false, $repetition=0
Run 13: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=8, $QPP=32, $Q=2*(8)*32, $DST=int(index/32), $AlgoVerbose=false, $repetition=0
Run 14: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=8, $QPP=64, $Q=2*(8)*64, $DST=int(index/64), $AlgoVerbose=false, $repetition=0
Run 15: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=8, $QPP=128, $Q=2*(8)*128, $DST=int(index/128), $AlgoVerbose=false, $repetition=0
Run 16: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=8, $QPP=256, $Q=2*(8)*256, $DST=int(index/256), $AlgoVerbose=false, $repetition=0
Run 17: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=8, $QPP=512, $Q=2*(8)*512, $DST=int(index/512), $AlgoVerbose=false, $repetition=0
Run 18: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=16, $QPP=1, $Q=2*(16)*1, $DST=int(index/1), $AlgoVerbose=false, $repetition=0
Run 19: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=16, $QPP=2, $Q=2*(16)*2, $DST=int(index/2), $AlgoVerbose=false, $repetition=0
Run 20: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=16, $QPP=8, $Q=2*(16)*8, $DST=int(index/8), $AlgoVerbose=false, $repetition=0
Run 21: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=16, $QPP=16, $Q=2*(16)*16, $DST=int(index/16), $AlgoVerbose=false, $repetition=0
Run 22: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=16, $QPP=32, $Q=2*(16)*32, $DST=int(index/32), $AlgoVerbose=false, $repetition=0
Run 23: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=16, $QPP=64, $Q=2*(16)*64, $DST=int(index/64), $AlgoVerbose=false, $repetition=0
Run 24: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=16, $QPP=128, $Q=2*(16)*128, $DST=int(index/128), $AlgoVerbose=false, $repetition=0
Run 25: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=16, $QPP=256, $Q=2*(16)*256, $DST=int(index/256), $AlgoVerbose=false, $repetition=0
Run 26: $0=2^(20), $MR=2^(20-14), $1=2^(20-11), $BR=13*1000, $BW=100, $H=16, $QPP=512, $Q=2*(16)*512, $DST=int(index/512), $AlgoVerbose=false, $repetition=0
"""

"""
Config: RL_LongShort_ManyToOne
Number of runs: 15
Run 0: $0=2^(20), $BSL=13, $H=2, $I=120, $QI=1, $Q=1*120, $1=10e-3 + (int(index/1)*15e-3) + uniform(0,5e-5), $2=8e6/((1)*((2)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 1: $0=2^(20), $BSL=13, $H=2, $I=120, $QI=2, $Q=2*120, $1=10e-3 + (int(index/2)*15e-3) + uniform(0,5e-5), $2=8e6/((2)*((2)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 2: $0=2^(20), $BSL=13, $H=2, $I=120, $QI=8, $Q=8*120, $1=10e-3 + (int(index/8)*15e-3) + uniform(0,5e-5), $2=8e6/((8)*((2)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 3: $0=2^(20), $BSL=13, $H=2, $I=120, $QI=16, $Q=16*120, $1=10e-3 + (int(index/16)*15e-3) + uniform(0,5e-5), $2=8e6/((16)*((2)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 4: $0=2^(20), $BSL=13, $H=4, $I=120, $QI=1, $Q=1*120, $1=10e-3 + (int(index/1)*15e-3) + uniform(0,5e-5), $2=8e6/((1)*((4)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 5: $0=2^(20), $BSL=13, $H=4, $I=120, $QI=2, $Q=2*120, $1=10e-3 + (int(index/2)*15e-3) + uniform(0,5e-5), $2=8e6/((2)*((4)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 6: $0=2^(20), $BSL=13, $H=4, $I=120, $QI=8, $Q=8*120, $1=10e-3 + (int(index/8)*15e-3) + uniform(0,5e-5), $2=8e6/((8)*((4)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 7: $0=2^(20), $BSL=13, $H=4, $I=120, $QI=16, $Q=16*120, $1=10e-3 + (int(index/16)*15e-3) + uniform(0,5e-5), $2=8e6/((16)*((4)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 8: $0=2^(20), $BSL=13, $H=8, $I=120, $QI=1, $Q=1*120, $1=10e-3 + (int(index/1)*15e-3) + uniform(0,5e-5), $2=8e6/((1)*((8)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 9: $0=2^(20), $BSL=13, $H=8, $I=120, $QI=2, $Q=2*120, $1=10e-3 + (int(index/2)*15e-3) + uniform(0,5e-5), $2=8e6/((2)*((8)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 10: $0=2^(20), $BSL=13, $H=8, $I=120, $QI=8, $Q=8*120, $1=10e-3 + (int(index/8)*15e-3) + uniform(0,5e-5), $2=8e6/((8)*((8)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 11: $0=2^(20), $BSL=13, $H=16, $I=120, $QI=1, $Q=1*120, $1=10e-3 + (int(index/1)*15e-3) + uniform(0,5e-5), $2=8e6/((1)*((16)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 12: $0=2^(20), $BSL=13, $H=16, $I=120, $QI=2, $Q=2*120, $1=10e-3 + (int(index/2)*15e-3) + uniform(0,5e-5), $2=8e6/((2)*((16)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 13: $0=2^(20), $BSL=13, $H=32, $I=120, $QI=1, $Q=1*120, $1=10e-3 + (int(index/1)*15e-3) + uniform(0,5e-5), $2=8e6/((1)*((32)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
Run 14: $0=2^(20), $BSL=13, $H=32, $I=120, $QI=2, $Q=2*120, $1=10e-3 + (int(index/2)*15e-3) + uniform(0,5e-5), $2=8e6/((2)*((32)-1)), $AlgoVerbose=false, $F=16, $B=20, $repetition=0
"""
# {python name: [config defined name, initial run number, number of subtests in test]}
py_to_c_scenarios = {
    ######################################################
    ################# ManyToOne ##########################
    ######################################################
    '2_1': ['LongSimult_ManyToOne', 0, 48],
    '2_2': ['LongSimult_ManyToOne', 1, 48],
    '2_8': ['LongSimult_ManyToOne', 2, 48],
    '2_32': ['LongSimult_ManyToOne', 3, 48],
    '2_64': ['LongSimult_ManyToOne', 4, 48],
    '2_128': ['LongSimult_ManyToOne', 5, 48],
    '2_256': ['LongSimult_ManyToOne', 6, 48],
    '2_512': ['LongSimult_ManyToOne', 7, 48],
    '2_1024': ['LongSimult_ManyToOne', 8, 48],
    '4_1': ['LongSimult_ManyToOne', 9, 48],
    '4_2': ['LongSimult_ManyToOne', 10, 48],
    '4_8': ['LongSimult_ManyToOne', 11, 48],
    '4_16': ['LongSimult_ManyToOne', 12, 48],
    '4_64': ['LongSimult_ManyToOne', 13, 48],
    '4_128': ['LongSimult_ManyToOne', 14, 48],
    '4_256': ['LongSimult_ManyToOne', 15, 48],
    '4_512': ['LongSimult_ManyToOne', 16, 48],
    '4_1024': ['LongSimult_ManyToOne', 17, 48],
    '8_1': ['LongSimult_ManyToOne', 18, 48],
    '8_2': ['LongSimult_ManyToOne', 19, 48],
    '8_8': ['LongSimult_ManyToOne', 20, 48],
    '8_32': ['LongSimult_ManyToOne', 21, 48],
    '8_64': ['LongSimult_ManyToOne', 22, 48],
    '8_128': ['LongSimult_ManyToOne', 23, 48],
    '8_256': ['LongSimult_ManyToOne', 24, 48],
    '8_512': ['LongSimult_ManyToOne', 25, 48],
    '8_1024': ['LongSimult_ManyToOne', 26, 48],
    '16_1': ['LongSimult_ManyToOne', 27, 48],
    '16_2': ['LongSimult_ManyToOne', 28, 48],
    '16_8': ['LongSimult_ManyToOne', 29, 48],
    '16_32': ['LongSimult_ManyToOne', 30, 48],
    '16_64': ['LongSimult_ManyToOne', 31, 48],
    '16_128': ['LongSimult_ManyToOne', 32, 48],
    '16_256': ['LongSimult_ManyToOne', 33, 48],
    '16_512': ['LongSimult_ManyToOne', 34, 48],
    '32_1': ['LongSimult_ManyToOne', 35, 48],
    '32_2': ['LongSimult_ManyToOne', 36, 48],
    '32_8': ['LongSimult_ManyToOne', 37, 48],
    '32_32': ['LongSimult_ManyToOne', 38, 48],
    '32_64': ['LongSimult_ManyToOne', 39, 48],
    '32_128': ['LongSimult_ManyToOne', 40, 48],
    '32_256': ['LongSimult_ManyToOne', 41, 48],
    '64_1': ['LongSimult_ManyToOne', 42, 48],
    '64_2': ['LongSimult_ManyToOne', 43, 48],
    '64_8': ['LongSimult_ManyToOne', 44, 48],
    '64_32': ['LongSimult_ManyToOne', 45, 48],
    '64_64': ['LongSimult_ManyToOne', 46, 48],
    '64_128': ['LongSimult_ManyToOne', 47, 48],
    ######################################################
    ################## AllToAll ##########################
    ######################################################
    '2_4_a2a': ['LongSimult_AllToAll', 0, 38],
    '2_8_a2a': ['LongSimult_AllToAll', 1, 38],
    '2_16_a2a': ['LongSimult_AllToAll', 2, 38],
    '2_32_a2a': ['LongSimult_AllToAll', 3, 38],
    '2_64_a2a': ['LongSimult_AllToAll', 4, 38],
    '2_128_a2a': ['LongSimult_AllToAll', 5, 38],
    '2_256_a2a': ['LongSimult_AllToAll', 6, 38],
    '2_512_a2a': ['LongSimult_AllToAll', 7, 38],
    '2_1024_a2a': ['LongSimult_AllToAll', 8, 38],
    '4_4_a2a': ['LongSimult_AllToAll', 9, 38],
    '4_8_a2a': ['LongSimult_AllToAll', 10, 38],
    '4_16_a2a': ['LongSimult_AllToAll', 11, 38],
    '4_32_a2a': ['LongSimult_AllToAll', 12, 38],
    '4_64_a2a': ['LongSimult_AllToAll', 13, 38],
    '4_128_a2a': ['LongSimult_AllToAll', 14, 38],
    '4_256_a2a': ['LongSimult_AllToAll', 15, 38],
    '4_512_a2a': ['LongSimult_AllToAll', 16, 38],
    '4_1024_a2a': ['LongSimult_AllToAll', 17, 38],
    '8_8_a2a': ['LongSimult_AllToAll', 18, 38],
    '8_16_a2a': ['LongSimult_AllToAll', 19, 38],
    '8_32_a2a': ['LongSimult_AllToAll', 20, 38],
    '8_64_a2a': ['LongSimult_AllToAll', 21, 38],
    '8_128_a2a': ['LongSimult_AllToAll', 22, 38],
    '8_256_a2a': ['LongSimult_AllToAll', 23, 38],
    '8_512_a2a': ['LongSimult_AllToAll', 24, 38],
    '8_1024_a2a': ['LongSimult_AllToAll', 25, 38],
    '16_16_a2a': ['LongSimult_AllToAll', 26, 38],
    '16_32_a2a': ['LongSimult_AllToAll', 27, 38],
    '16_64_a2a': ['LongSimult_AllToAll', 28, 38],
    '16_128_a2a': ['LongSimult_AllToAll', 29, 38],
    '16_256_a2a': ['LongSimult_AllToAll', 30, 38],
    '16_512_a2a': ['LongSimult_AllToAll', 31, 38],
    '32_32_a2a': ['LongSimult_AllToAll', 32, 38],
    '32_64_a2a': ['LongSimult_AllToAll', 33, 38],
    '32_128_a2a': ['LongSimult_AllToAll', 34, 38],
    '32_256_a2a': ['LongSimult_AllToAll', 35, 38],
    '64_64_a2a': ['LongSimult_AllToAll', 36, 38],
    '64_128_a2a': ['LongSimult_AllToAll', 37, 38],
    ######################################################
    ################# LongShort ##########################
    ######################################################
    '2_1_ls': ['LongShort_ManyToOne', 0, 15],
    '2_2_ls': ['LongShort_ManyToOne', 1, 15],
    '2_8_ls': ['LongShort_ManyToOne', 2, 15],
    '2_16_ls': ['LongShort_ManyToOne', 3, 15],
    '4_1_ls': ['LongShort_ManyToOne', 4, 15],
    '4_2_ls': ['LongShort_ManyToOne', 5, 15],
    '4_8_ls': ['LongShort_ManyToOne', 6, 15],
    '4_16_ls': ['LongShort_ManyToOne', 7, 15],
    '8_1_ls': ['LongShort_ManyToOne', 8, 15],
    '8_2_ls': ['LongShort_ManyToOne', 9, 15],
    '8_8_ls': ['LongShort_ManyToOne', 10, 15],
    '16_1_ls': ['LongShort_ManyToOne', 11, 15],
    '16_2_ls': ['LongShort_ManyToOne', 12, 15],
    '32_1_ls': ['LongShort_ManyToOne', 13, 15],
    '32_2_ls': ['LongShort_ManyToOne', 14, 15],
}

many2one_r_to_h_and_q = {0: {'H': 2, 'Q': 1},
    1: {'H': 2, 'Q': 2},
    2: {'H': 2, 'Q': 8},
    3: {'H': 2, 'Q': 32},
    4: {'H': 2, 'Q': 64},
    5: {'H': 2, 'Q': 128},
    6: {'H': 2, 'Q': 256},
    7: {'H': 2, 'Q': 512},
    8: {'H': 2, 'Q': 1024},
    9: {'H': 4, 'Q': 1},
    10: {'H': 4, 'Q': 2},
    11: {'H': 4, 'Q': 8},
    12: {'H': 4, 'Q': 32},
    13: {'H': 4, 'Q': 64},
    14: {'H': 4, 'Q': 128},
    15: {'H': 4, 'Q': 256},
    16: {'H': 4, 'Q': 512},
    17: {'H': 4, 'Q': 1024},
    18: {'H': 8, 'Q': 1},
    19: {'H': 8, 'Q': 2},
    20: {'H': 8, 'Q': 8},
    21: {'H': 8, 'Q': 32},
    22: {'H': 8, 'Q': 64},
    23: {'H': 8, 'Q': 128},
    24: {'H': 8, 'Q': 256},
    25: {'H': 8, 'Q': 512},
    26: {'H': 8, 'Q': 1024},
    27: {'H': 16, 'Q': 1},
    28: {'H': 16, 'Q': 2},
    29: {'H': 16, 'Q': 8},
    30: {'H': 16, 'Q': 32},
    31: {'H': 16, 'Q': 64},
    32: {'H': 16, 'Q': 128},
    33: {'H': 16, 'Q': 256},
    34: {'H': 16, 'Q': 512},
    35: {'H': 32, 'Q': 1},
    36: {'H': 32, 'Q': 2},
    37: {'H': 32, 'Q': 8},
    38: {'H': 32, 'Q': 32},
    39: {'H': 32, 'Q': 64},
    40: {'H': 32, 'Q': 128},
    41: {'H': 32, 'Q': 256},
    42: {'H': 64, 'Q': 1},
    43: {'H': 64, 'Q': 2},
    44: {'H': 64, 'Q': 8},
    45: {'H': 64, 'Q': 32},
    46: {'H': 64, 'Q': 64},
    47: {'H': 64, 'Q': 128}
}

all2all_r_to_h_and_q = {0: {'H': 2, 'Q': 4},
    1: {'H': 2, 'Q': 8},
    2: {'H': 2, 'Q': 16},
    3: {'H': 2, 'Q': 32},
    4: {'H': 2, 'Q': 64},
    5: {'H': 2, 'Q': 128},
    6: {'H': 2, 'Q': 256},
    7: {'H': 2, 'Q': 512},
    8: {'H': 2, 'Q': 1024},
    9: {'H': 4, 'Q': 4},
    10: {'H': 4, 'Q': 8},
    11: {'H': 4, 'Q': 16},
    12: {'H': 4, 'Q': 32},
    13: {'H': 4, 'Q': 64},
    14: {'H': 4, 'Q': 128},
    15: {'H': 4, 'Q': 256},
    16: {'H': 4, 'Q': 512},
    17: {'H': 4, 'Q': 1024},
    18: {'H': 8, 'Q': 8},
    19: {'H': 8, 'Q': 16},
    20: {'H': 8, 'Q': 32},
    21: {'H': 8, 'Q': 64},
    22: {'H': 8, 'Q': 128},
    23: {'H': 8, 'Q': 256},
    24: {'H': 8, 'Q': 512},
    25: {'H': 8, 'Q': 1024},
    26: {'H': 16, 'Q': 16},
    27: {'H': 16, 'Q': 32},
    28: {'H': 16, 'Q': 64},
    29: {'H': 16, 'Q': 128},
    30: {'H': 16, 'Q': 256},
    31: {'H': 16, 'Q': 512},
    32: {'H': 32, 'Q': 32},
    33: {'H': 32, 'Q': 64},
    34: {'H': 32, 'Q': 128},
    35: {'H': 32, 'Q': 256},
    36: {'H': 64, 'Q': 64},
    37: {'H': 64, 'Q': 128}
}