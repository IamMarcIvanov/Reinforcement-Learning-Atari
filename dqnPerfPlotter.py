# %%
import matplotlib.pyplot as plt

frames = []
rewards = []
epsilons = []

# %%
def find_nth(s, x, n):
    i = -1
    for _ in range(n):
        i = s.find(x, i + len(x))
        if i == -1:
            break
    return i

# %%
with open('BestYetDQNoutput.txt', 'r') as f:
    for line in f:
        if line.startswith('Best'):
            continue
        frame_end = line.index(':')
        frames.append(int(line[: frame_end]))
        
        reward_start = find_nth(line, ' ', 6) + 1
        reward_end = find_nth(line, ',', 2)
        rewards.append(float(line[reward_start: reward_end]))
        
        epsilon_start = find_nth(line, ' ', 8) + 1
        epsilon_end = find_nth(line, ',', 3)
        epsilons.append(float(line[epsilon_start: epsilon_end]))

# %%
plt.plot(frames, rewards, label='reward')
plt.legend()
plt.title('Best Yet DQN Performance: Frames vs Rewards')
plt.savefig('frame-reward.jpg')
plt.show()
plt.plot(frames, epsilons, label='epsilon')
plt.legend()
plt.title('Best Yet DQN Performance: Frames vs Epsilon')
plt.savefig('frame-epsilon.jpg')
plt.show()
