import gymnasium as gym

# 月着陸(Lunar Lander)ゲームの環境を作成
env = gym.make("LunarLander-v2", render_mode="human")

# ゲーム環境を初期化
observation, info = env.reset()

# ゲームのステップを1000回プレイ
for _ in range(1000):
    # 環境からランダムな行動を取得
    # これがエージェントの行動になるので、本来はAIが行動を決定するべきところ
    action = env.action_space.sample()
    print("action:", action)

    # 行動を実行すると、環境の状態が更新される
    observation, reward, terminated, truncated, info = env.step(action)

    # ゲームが終了したら、環境を初期化して再開
    if terminated or truncated:
        observation, info = env.reset()

env.close()
