import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./logs/monitor.csv', names=['r','l','t'])
df = df.drop(range(2)) # Delete line 1&2

# Reward graph
x = range(len(df['r']))
y = df['r'].astype(float)
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('reward')
plt.savefig('monitor_reward.png')

# Episode length graph
x = range(len(df['l']))
y = df['l'].astype(float)
plt.plot(x, y)
plt.xlabel('episode')
plt.ylabel('episode len')
plt.savefig('monitor_epilen.png')
