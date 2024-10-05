def calculate_expected_reward(distance_to_terminal: dict, penalty_per_step=0.1, reward_at_terminal=1):
    """
    計算根據距離終點的狀態數量和對應的回報，獲得總的期望回報。
    
    Args:
        distance_to_terminal (dict): 鍵為距離終點的步數，值為該距離有多少個 state。
        penalty_per_step (float): 每一步的懲罰 (默認為 0.1)。
        reward_at_terminal (float): 到達終點的回報 (默認為 +1)。
        
    Returns:
        float: 所有狀態的期望回報
    """
    total_states = sum(distance_to_terminal.values())  # 總的 state 數量
    print("total state:",total_states)
    total_reward = 0  # 用來累積總回報

    # 依據每個距離和狀態數量計算回報
    for distance, num_states in distance_to_terminal.items():
        # 計算每個距離的回報，除以距離+1
        reward = (reward_at_terminal - penalty_per_step * distance) / (distance + 1)
        print(reward,num_states)
        total_reward += reward * num_states  # 加上該距離的總回報

    # 計算期望回報（所有狀態的平均回報）
    print("total_reward:",total_reward)
    expected_reward = total_reward / total_states
    return expected_reward

# 設定每個距離的狀態數量
distance_to_terminal = {
    0: 1,  # 距離終點 0 的狀態有 1 個
    1: 2,  # 距離終點 1 的狀態有 2 個
    2: 2,  # 距離終點 2 的狀態有 2 個
    3: 2,   # 距離終點 3 的狀態有 2 個
    4: 3,  # 距離終點 1 的狀態有 2 個
    5: 1,  # 距離終點 2 的狀態有 2 個
    6: 2,   # 距離終點 3 的狀態有 2 個
    7: 2,  # 距離終點 0 的狀態有 1 個
    8: 3,  # 距離終點 1 的狀態有 2 個
    9: 2,  # 距離終點 2 的狀態有 2 個
    10: 1  # 距離終點 2 的狀態有 2 個
}

# 計算期望回報
expected_reward = calculate_expected_reward(distance_to_terminal)
print(f"期望回報: {expected_reward:.4f}")
