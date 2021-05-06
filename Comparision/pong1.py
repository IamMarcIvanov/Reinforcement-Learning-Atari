env = gym.make("Pong-v0")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        
        def flat(size, kernel_size, stride):
            return (size - kernel_size // stride) + 1
        
        def convBlock(in_channels, filters, *args, **kwargs):
            return nn.Sequential(
                nn.Conv2d(in_channels, filters, *args, **kwargs),
                nn.BatchNorm2d(filters),
                nn.LeakyReLU()
            )
        
        def linBlock(inDim, outDim):
            return nn.Sequential(
                nn.Linear(inDim, outDim),
                nn.BatchNorm2d(outDim),
                nn.LeakyReLU()
        )

        self.conv1 = convBlock(1, 32, kernel_size=8, stride=4)
        self.conv2 = convBlock(32, 64, kernel_size=4, stride=2)
        self.conv3 = convBlock(64, 64, kernel_size=3, stride=1)

        convw = flat(flat(flat(w, 8, 4), 4, 2), 3, 1)
        convh = flat(flat(flat(h, 8, 4), 4, 2), 3, 1)
        
        self.fc1 = linBlock(convw * convh * 64, 512)
        self.fc2 = nn.Linear(512, outputs)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

resize = T.Compose([T.ToPILImage(), 
                    T.Grayscale(), 
                    T.Resize([84, 84], interpolation=InterpolationMode.NEAREST), 
                    T.ToTensor()])
screen = resize(env.render(mode='rgb_array'))

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10

n_actions = env.action_space.n
screen_height = screen.shape[1]
screen_width = screen.shape[2]

policy_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net = DQN(screen_height, screen_width, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], 
                            device=device, 
                            dtype=torch.long)
        

episode_durations = []


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device,
                                  dtype=torch.bool,)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(1)[0].detach())

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 300
for i_episode in range(num_episodes):
    env.reset()
    last_screen = resize(env.render(mode='rgb_array'))
    current_screen = resize(env.render(mode='rgb_array'))
    state = current_screen - last_screen
    for t in count():
        action = select_action(state)
        _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)

        last_screen = current_screen
        current_screen = resize(env.render(mode='rgb_array'))
        if not done:
            next_state = current_screen - last_screen
        else:
            next_state = None

        memory.push(state, action, next_state, reward)

        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            break
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

env.render()
env.close()
