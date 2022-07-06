import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import main


df = pd.DataFrame(main.rewards_array)


# Line plot of total reward against episode number
rcParams['figure.dpi'] = 120
plt.title("Noise added from normal distribution std: 0.05")
fig_1 = sns.lineplot(x=df.episode_number, y=df.total_reward)

# Create new column in dataframe of the average reward for every 50 episodes
df['average_reward'] = df['total_reward'].rolling(50).mean()


plt.title("Noise added from normal distribution std: 0.05, reward averaged over 50 episodes")
sns.lineplot(x=df.episode_number, y=df.average_reward)


plt.show()
