from agents.adpg import ADPG
from agents.dqn import DQN
from agents.ppo import PPO
from agents.random import RandomAgent
from agents.supervised import Supervised

AGENTS = {'PPO': PPO, 'DQN': DQN, 'SUPERVISED': Supervised, 'ADPG': ADPG, 'RANDOM': RandomAgent}

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
    #################   Tests   ##########################
    '2_1_test': ['Simult_ManyToOne_Vectors', 0, 48],
    '2_2_test': ['Simult_ManyToOne_Vectors', 1, 48],
    '2_8_test': ['Simult_ManyToOne_Vectors', 2, 48],
    '2_32_test': ['Simult_ManyToOne_Vectors', 3, 48],
    '2_64_test': ['Simult_ManyToOne_Vectors', 4, 48],
    '2_128_test': ['Simult_ManyToOne_Vectors', 5, 48],
    '2_256_test': ['Simult_ManyToOne_Vectors', 6, 48],
    '2_512_test': ['Simult_ManyToOne_Vectors', 7, 48],
    '2_1024_test': ['Simult_ManyToOne_Vectors', 8, 48],
    '4_1_test': ['Simult_ManyToOne_Vectors', 9, 48],
    '4_2_test': ['Simult_ManyToOne_Vectors', 10, 48],
    '4_8_test': ['Simult_ManyToOne_Vectors', 11, 48],
    '4_16_test': ['Simult_ManyToOne_Vectors', 12, 48],
    '4_64_test': ['Simult_ManyToOne_Vectors', 13, 48],
    '4_128_test': ['Simult_ManyToOne_Vectors', 14, 48],
    '4_256_test': ['Simult_ManyToOne_Vectors', 15, 48],
    '4_512_test': ['Simult_ManyToOne_Vectors', 16, 48],
    '4_1024_test': ['Simult_ManyToOne_Vectors', 17, 48],
    '8_1_test': ['Simult_ManyToOne_Vectors', 18, 48],
    '8_2_test': ['Simult_ManyToOne_Vectors', 19, 48],
    '8_8_test': ['Simult_ManyToOne_Vectors', 20, 48],
    '8_32_test': ['Simult_ManyToOne_Vectors', 21, 48],
    '8_64_test': ['Simult_ManyToOne_Vectors', 22, 48],
    '8_128_test': ['Simult_ManyToOne_Vectors', 23, 48],
    '8_256_test': ['Simult_ManyToOne_Vectors', 24, 48],
    '8_512_test': ['Simult_ManyToOne_Vectors', 25, 48],
    '8_1024_test': ['Simult_ManyToOne_Vectors', 26, 48],
    '16_1_test': ['Simult_ManyToOne_Vectors', 27, 48],
    '16_2_test': ['Simult_ManyToOne_Vectors', 28, 48],
    '16_8_test': ['Simult_ManyToOne_Vectors', 29, 48],
    '16_32_test': ['Simult_ManyToOne_Vectors', 30, 48],
    '16_64_test': ['Simult_ManyToOne_Vectors', 31, 48],
    '16_128_test': ['Simult_ManyToOne_Vectors', 32, 48],
    '16_256_test': ['Simult_ManyToOne_Vectors', 33, 48],
    '16_512_test': ['Simult_ManyToOne_Vectors', 34, 48],
    '32_1_test': ['Simult_ManyToOne_Vectors', 35, 48],
    '32_2_test': ['Simult_ManyToOne_Vectors', 36, 48],
    '32_8_test': ['Simult_ManyToOne_Vectors', 37, 48],
    '32_32_test': ['Simult_ManyToOne_Vectors', 38, 48],
    '32_64_test': ['Simult_ManyToOne_Vectors', 39, 48],
    '32_128_test': ['Simult_ManyToOne_Vectors', 40, 48],
    '32_256_test': ['Simult_ManyToOne_Vectors', 41, 48],
    '64_1_test': ['Simult_ManyToOne_Vectors', 42, 48],
    '64_2_test': ['Simult_ManyToOne_Vectors', 43, 48],
    '64_8_test': ['Simult_ManyToOne_Vectors', 44, 48],
    '64_32_test': ['Simult_ManyToOne_Vectors', 45, 48],
    '64_64_test': ['Simult_ManyToOne_Vectors', 46, 48],
    '64_128_test': ['Simult_ManyToOne_Vectors', 47, 48],
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
    #################   Tests   ##########################
    '2_4_a2a_test': ['Simult_AllToAll_Vectors', 0, 38],
    '2_8_a2a_test': ['Simult_AllToAll_Vectors', 1, 38],
    '2_16_a2a_test': ['Simult_AllToAll_Vectors', 2, 38],
    '2_32_a2a_test': ['Simult_AllToAll_Vectors', 3, 38],
    '2_64_a2a_test': ['Simult_AllToAll_Vectors', 4, 38],
    '2_128_a2a_test': ['Simult_AllToAll_Vectors', 5, 38],
    '2_256_a2a_test': ['Simult_AllToAll_Vectors', 6, 38],
    '2_512_a2a_test': ['Simult_AllToAll_Vectors', 7, 38],
    '2_1024_a2a_test': ['Simult_AllToAll_Vectors', 8, 38],
    '4_4_a2a_test': ['Simult_AllToAll_Vectors', 9, 38],
    '4_8_a2a_test': ['Simult_AllToAll_Vectors', 10, 38],
    '4_16_a2a_test': ['Simult_AllToAll_Vectors', 11, 38],
    '4_32_a2a_test': ['Simult_AllToAll_Vectors', 12, 38],
    '4_64_a2a_test': ['Simult_AllToAll_Vectors', 13, 38],
    '4_128_a2a_test': ['Simult_AllToAll_Vectors', 14, 38],
    '4_256_a2a_test': ['Simult_AllToAll_Vectors', 15, 38],
    '4_512_a2a_test': ['Simult_AllToAll_Vectors', 16, 38],
    '4_1024_a2a_test': ['Simult_AllToAll_Vectors', 17, 38],
    '8_8_a2a_test': ['Simult_AllToAll_Vectors', 18, 38],
    '8_16_a2a_test': ['Simult_AllToAll_Vectors', 19, 38],
    '8_32_a2a_test': ['Simult_AllToAll_Vectors', 20, 38],
    '8_64_a2a_test': ['Simult_AllToAll_Vectors', 21, 38],
    '8_128_a2a_test': ['Simult_AllToAll_Vectors', 22, 38],
    '8_256_a2a_test': ['Simult_AllToAll_Vectors', 23, 38],
    '8_512_a2a_test': ['Simult_AllToAll_Vectors', 24, 38],
    '8_1024_a2a_test': ['Simult_AllToAll_Vectors', 25, 38],
    '16_16_a2a_test': ['Simult_AllToAll_Vectors', 26, 38],
    '16_32_a2a_test': ['Simult_AllToAll_Vectors', 27, 38],
    '16_64_a2a_test': ['Simult_AllToAll_Vectors', 28, 38],
    '16_128_a2a_test': ['Simult_AllToAll_Vectors', 29, 38],
    '16_256_a2a_test': ['Simult_AllToAll_Vectors', 30, 38],
    '16_512_a2a_test': ['Simult_AllToAll_Vectors', 31, 38],
    '32_32_a2a_test': ['Simult_AllToAll_Vectors', 32, 38],
    '32_64_a2a_test': ['Simult_AllToAll_Vectors', 33, 38],
    '32_128_a2a_test': ['Simult_AllToAll_Vectors', 34, 38],
    '32_256_a2a_test': ['Simult_AllToAll_Vectors', 35, 38],
    '64_64_a2a_test': ['Simult_AllToAll_Vectors', 36, 38],
    '64_128_a2a_test': ['Simult_AllToAll_Vectors', 37, 38],
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
    #################   Tests   ##########################
    '2_1_ls_test': ['LongShort_ManyToOne_Vectors', 0, 15],
    '2_2_ls_test': ['LongShort_ManyToOne_Vectors', 1, 15],
    '2_8_ls_test': ['LongShort_ManyToOne_Vectors', 2, 15],
    '2_16_ls_test': ['LongShort_ManyToOne_Vectors', 3, 15],
    '4_1_ls_test': ['LongShort_ManyToOne_Vectors', 4, 15],
    '4_2_ls_test': ['LongShort_ManyToOne_Vectors', 5, 15],
    '4_8_ls_test': ['LongShort_ManyToOne_Vectors', 6, 15],
    '4_16_ls_test': ['LongShort_ManyToOne_Vectors', 7, 15],
    '8_1_ls_test': ['LongShort_ManyToOne_Vectors', 8, 15],
    '8_2_ls_test': ['LongShort_ManyToOne_Vectors', 9, 15],
    '8_8_ls_test': ['LongShort_ManyToOne_Vectors', 10, 15],
    '16_1_ls_test': ['LongShort_ManyToOne_Vectors', 11, 15],
    '16_2_ls_test': ['LongShort_ManyToOne_Vectors', 12, 15],
    '32_1_ls_test': ['LongShort_ManyToOne_Vectors', 13, 15],
    '32_2_ls_test': ['LongShort_ManyToOne_Vectors', 14, 15],
}