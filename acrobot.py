import gymnasium as gym
import numpy as np

N = 11
def get_state(observation):
    theta1 = int(np.arccos(observation[0]) * N / np.pi)  # 0~N
    theta2 = int(np.arccos(observation[2]) * N / np.pi)  # 0~N
    v_theta1 = int((observation[4] * 180 / np.pi + 720) * N / 1440)  # 0~N
    v_theta2 = int((observation[4] * 180 / np.pi + 1620) * N / 3240)  # 0~N
    return theta1, theta2, v_theta1, v_theta2


def update_q_table(_q_table, _action,  _observation, _next_observation, _reward, _episode):

    alpha = 0.2  # 学習率
    gamma = 0.99  # 時間割引き率

    # 行動後の状態で得られる最大行動価値 Q(s',a')
    next_theta1, next_theta2, next_v_theta1, next_v_theta2 = get_state(_next_observation)
    next_max_q_value = max(_q_table[next_theta1][next_theta2][next_v_theta1][next_v_theta2])

    # 行動前の状態の行動価値 Q(s,a)
    theta1, theta2, v_theta1, v_theta2 = get_state(_observation)
    q_value = _q_table[theta1][theta2][v_theta1][v_theta2][_action]

    # 行動価値関数の更新
    _q_table[theta1][theta2][v_theta1][v_theta2][_action] = q_value + \
        alpha * (_reward + gamma * next_max_q_value - q_value)

    return _q_table


def get_action(_env, _q_table, _observation, _episode):
    epsilon = 0.002
    if np.random.uniform(0, 1) > epsilon:
        theta1, theta2, v_theta1, v_theta2 = get_state(_observation)
        _action = np.argmax(_q_table[theta1][theta2][v_theta1][v_theta2])
    else:
        _action = np.random.choice([0, 1, 2])
    return _action


if __name__ == "__main__":
    # アクロボットの環境を作成
    env = gym.make('Acrobot-v1')

    # Qテーブルを初期化
    q_table = np.zeros((N + 1, N + 1, N + 1, N + 1, 3))

    # 環境を初期化
    observation, info = env.reset()
    rewards = []

    episode_num = 100000

    # ステップをepsode_num回プレイ
    for episode in range(episode_num + 1):
        tortal_reward = 0
        observation, info = env.reset()

        for _ in range(500):
            # ε-グリーディ法で行動を選択
            action = get_action(env, q_table, observation, episode)
            # print("action:", action)

            # 行動を実行すると、環境の状態が更新される
            next_observation, reward, terminated, truncated, info = env.step(action)

            # Qテーブルを更新
            q_table = update_q_table(q_table, action, observation, next_observation, reward, episode)
            tortal_reward += reward

            observation = next_observation

            # ゲームが終了したら、環境を初期化して再開
            if terminated or truncated:
                if episode % 100 == 0:
                    print("episode: {}, total_reward: {}".format(episode, tortal_reward))
                rewards.append(tortal_reward)
                break
                # observation, info = env.reset()

    env.close()

    env = gym.make('Acrobot-v1', render_mode="human")
    observation, info = env.reset()
    for _ in range(500):
        action = get_action(env, q_table, observation, episode)
        next_observation, reward, terminated, truncated, info = env.step(action)
        observation = next_observation

        if terminated or truncated:
            break
    env.close()
