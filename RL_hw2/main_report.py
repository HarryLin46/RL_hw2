import random
import numpy as np
import json
import wandb

from algorithms_report import (
    MonteCarloPrediction,
    TDPrediction,
    NstepTDPrediction,
    MonteCarloPolicyIteration,
    SARSA,
    Q_Learning,
)
from gridworld import GridWorld

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 2-1
STEP_REWARD     = -0.1
GOAL_REWARD     = 1.0
TRAP_REWARD     = -1.0
INIT_POS        = [0]
DISCOUNT_FACTOR = 0.9
POLICY          = None
MAX_EPISODE     = 300
LEARNING_RATE   = 0.01
NUM_STEP        = 3
# 2-2
EPSILON           = 0.2
BUFFER_SIZE       = 10000
UPDATE_FREQUENCY  = 200
SAMPLE_BATCH_SIZE = 500

def bold(s):
    return "\033[1m" + str(s) + "\033[0m"


def underline(s):
    return "\033[4m" + str(s) + "\033[0m"


def green(s):
    return "\033[92m" + str(s) + "\033[0m"


def red(s):
    return "\033[91m" + str(s) + "\033[0m"


def init_grid_world(maze_file: str = "maze.txt", init_pos: list = None):
    print(bold(underline("Grid World")))
    grid_world = GridWorld(
        maze_file,
        step_reward=STEP_REWARD,
        goal_reward=GOAL_REWARD,
        trap_reward=TRAP_REWARD,
        init_pos=init_pos,
    )
    grid_world.print_maze()
    grid_world.visualize(title="Maze", filename="maze.png", show=False)
    print()
    return grid_world


def run_MC_prediction(grid_world: GridWorld,seed):
    print(f"Run MC prediction. Seed:{seed}")
    prediction = MonteCarloPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed = seed
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"Monte Carlo Prediction",
        show=False,
        filename=f"my_result/MC_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()


def run_TD_prediction(grid_world: GridWorld, seed):
    print(f"Run TD(0) prediction. Seed:{seed}")
    prediction = TDPrediction(
        grid_world,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        learning_rate=LEARNING_RATE,
        seed = seed
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"TD(0) Prediction",
        show=False,
        filename=f"my_result/TD0_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()


def run_NstepTD_prediction(grid_world: GridWorld,seed):
    print(f"Run N-step TD prediction. Seed:{seed}")
    prediction = NstepTDPrediction(
        grid_world,
        learning_rate=LEARNING_RATE,
        num_step=NUM_STEP,
        discount_factor=DISCOUNT_FACTOR,
        policy = POLICY,
        max_episode= MAX_EPISODE,
        seed=seed,
    )
    prediction.run()
    grid_world.visualize(
        prediction.get_all_state_values(),
        title=f"N-step TD Prediction",
        show=False,
        filename=f"my_result/NstepTD_prediction.png",
    )
    grid_world.reset()
    grid_world.reset_step_count()
    print()
    return prediction.get_all_state_values()

def run_MC_policy_iteration(grid_world: GridWorld, iter_num: int):
    print(bold(underline("MC Policy Iteration")))
    policy_iteration = MonteCarloPolicyIteration(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"MC Policy Iteration",
        show=False,
        filename=f"my_result/MC_policy_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

def run_SARSA(grid_world: GridWorld, iter_num: int):
    print(bold(underline("SARSA Policy Iteration")))
    policy_iteration = SARSA(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"SARSA",
        show=False,
        filename=f"my_result/SARSA_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()


def run_Q_Learning(grid_world: GridWorld, iter_num: int):
    print(bold(underline("Q_Learning Policy Iteration")))
    policy_iteration = Q_Learning(
            grid_world, 
            discount_factor=DISCOUNT_FACTOR,
            learning_rate=LEARNING_RATE,
            epsilon= EPSILON,
            buffer_size=BUFFER_SIZE,
            update_frequency=UPDATE_FREQUENCY,
            sample_batch_size=SAMPLE_BATCH_SIZE,
            )
    policy_iteration.run(max_episode=iter_num)
    grid_world.visualize(
        policy_iteration.get_max_state_values(),
        policy_iteration.get_policy_index(),
        title=f"Q_Learning",
        show=False,
        filename=f"my_result/Q_Learning_iteration_{iter_num}.png",
    )
    history = grid_world.run_policy(policy_iteration.get_policy_index())
    print(f"Solved in {bold(green(len(history)))} steps")
    print(history)
    print(
        f"Start state: {bold(green(history[0][0]))}, End state: {bold(red(history[-1][0]))}"
    )
    grid_world.reset()
    print()

if __name__ == "__main__":
    seed = 1
    # grid_world = init_grid_world("maze.txt",INIT_POS)
    # # 2-1
    # pred_Vs = []  # 用來儲存多個樣本結果
    
    # for seed in np.random.choice(np.arange(1, 201, dtype=int), size=50, replace=False):
    #     # 使用不同的 seed 來生成預測樣本
    #     # pred_V = run_MC_prediction(grid_world, seed)
    #     pred_V = run_TD_prediction(grid_world,seed)
    #     pred_Vs.append(pred_V)  # 將每個樣本結果加入列表中
    
    # # 將樣本列表轉換成 NumPy 陣列
    # pred_Vs_np = np.array(pred_Vs)
    
    # # 將樣本數據保存為 .npy 檔案
    # print("Saving MC.npy, whose shape",pred_Vs_np.shape)
    # np.save('my_result/TD.npy', pred_Vs)
    # run_TD_prediction(grid_world,seed)
    # run_NstepTD_prediction(grid_world,seed)

    # 2-2
    grid_world = init_grid_world("maze.txt")
    epsilon_values = [0.1, 0.2, 0.3, 0.4]

    for epsilon_value in epsilon_values:
        # 每次實驗都必須初始化 W&B
        wandb.init(project="RL_hw2_derive_all", name=f"Q_learning_epsilon_{epsilon_value}", reinit=True)
        
        # 設置超參數
        # wandb.config.max_episode = 1000  # 最大回合數
        # wandb.config.gamma = 0.9  # 折扣因子
        wandb.config.epsilon = epsilon_value  # 設定 epsilon 為當前實驗值
        EPSILON = epsilon_value

        # 執行策略迭代
        # iter_num = wandb.config.max_episode
        # run_MC_policy_iteration(grid_world, 512000)
        # run_SARSA(grid_world, 512000)
        run_Q_Learning(grid_world, 50000)

        # 記得在每次實驗結束時結束 W&B 實驗
        wandb.finish()
    # run_MC_policy_iteration(grid_world, 512000)
    # run_SARSA(grid_world, 512000)
    # run_Q_Learning(grid_world, 50000)