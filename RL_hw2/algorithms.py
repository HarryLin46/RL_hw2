import numpy as np
import json
from collections import deque
import math

from gridworld import GridWorld

# =========================== 2.1 model free prediction ===========================
class ModelFreePrediction:
    """
    Base class for ModelFreePrediction algorithms
    """
       

    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space], i.e. [22,4]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episode for data collection. Defaults to 10000.
            seed (int): seed for sampling action from the policy
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.max_episode = max_episode
        self.episode_counter = 0  
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.values       = np.zeros(self.state_space)
        self.rng = np.random.default_rng(seed)      # only call this in collect_data()
        if policy:
            self.policy = policy
        else:
            self.policy = np.ones((self.state_space, self.action_space)) / self.action_space  # random policy

    def get_all_state_values(self) -> np.array:
        return self.values

    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state
        
        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  

        next_state, reward, done = self.grid_world.step(action)  
        if done:
            self.episode_counter +=1
        return next_state, reward, done
        

class MonteCarloPrediction(ModelFreePrediction):
    def __init__(self, grid_world: GridWorld, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """
        Constructor for MonteCarloPrediction
        
        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episode (int, optional): Maximum episdoe for data collection. Defaults to 10000.
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with first-visit Monte-Carlo method
        current_state = self.grid_world.reset()
        
        self.Returns = [[] for _ in range(self.state_space)] #remember the collected G in each state

        while self.episode_counter <self.max_episode: #in MC, after collecting self.max_episode, then it's done 
            #collect history
            States = [self.grid_world.get_current_state()]
            # Actions = []
            Rewards = [-999]
            while True:
                next_state, reward, done = self.collect_data()
                if done:
                    Rewards.append(reward) #append RT
                    break
                else:
                    States.append(next_state) #append St
                    # Actions.append()
                    Rewards.append(reward) #append Rt
            #then States, Rewards are the history of a whole episode, using them to update

            #find first visit
            first_visit_table = np.ones(self.state_space,dtype=int)*-1
            for idx,state in enumerate(States):
                if first_visit_table[state]==-1:
                    first_visit_table[state] = idx
            # print("first_visit_table",first_visit_table)
            # print("States",len(States))
            # print("Rewards",len(Rewards))

            Gain=0
            for t in range(len(States)-1,-1,-1): #t=T-1 - 0
                # print("t",t)
                Gain = self.discount_factor*Gain + Rewards[t+1]

                St = States[t]
                # print(first_visit_table[St],t)
                if  first_visit_table[St] == t: #this is first visit
                    self.Returns[St].append(Gain)
                    # print(self.Returns[St])
                    self.values[St] = sum(self.Returns[St]) / len(self.Returns[St])

        



class TDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld,learning_rate: float, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate

    def run(self) -> None:
        """Run the algorithm until max episode"""
        # TODO: Update self.values with TD(0) Algorithm
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            #initialize S, no needed here
            # initial_state = self.grid_world.get_current_state()

            while True: #running an episode
                current_state = self.grid_world.get_current_state()
                next_state, reward, done = self.collect_data()
                self.values[current_state] = self.values[current_state] + self.lr*(reward + self.discount_factor*self.values[next_state]*(1 - done) - self.values[current_state])
                if done:
                    break


class NstepTDPrediction(ModelFreePrediction):
    def __init__(
            self, grid_world: GridWorld, learning_rate: float, num_step: int, policy: np.ndarray = None, discount_factor: float = 1.0, max_episode: int = 300, seed: int = 1):
        """Constructor for Temporal Difference(0) Prediction

        Args:
            grid_world (GridWorld): GridWorld object
            policy (np.ndarray): Stochastic policy representing action probabilities [state_space, action_space]
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
            max_episdoe (int, optional): Maximum episdoe for data collection. Defaults to 10000.
            learning_rate (float): learning rate for updating state value
            num_step (int): n_step look ahead for TD
        """
        super().__init__(grid_world,policy, discount_factor, max_episode, seed)
        self.lr     = learning_rate
        self.n      = num_step

    def run(self) -> None:
        """Run the algorithm until max_episode"""
        # TODO: Update self.values with N-step TD Algorithm
        current_state = self.grid_world.reset()
        while self.episode_counter < self.max_episode:
            # Remember the trace
            States = [self.grid_world.get_current_state()]
            # Actions = []
            Rewards = [-999]

            Ts =  math.inf #T is not allowed to use
            t=0
            while True:
                if t<Ts:
                    #take an action
                    next_state, reward, done = self.collect_data()
                    if done: #t=T-1
                        Ts = t+1
                        Rewards.append(reward) #append RT
                    else:
                        States.append(next_state) #append St
                        # Actions.append()
                        Rewards.append(reward) #append Rt

                tau = t - self.n + 1
                if tau >= 0:#then update
                    Gain = sum(self.discount_factor**(i-tau-1) * Rewards[i] for i in range(tau+1,min(tau+self.n,Ts)+1                                                                                                                                                                                                                                                                                                                               )) 
                    if tau+self.n<Ts:
                        Gain = Gain + self.discount_factor**self.n * self.values[States[tau+self.n]]
                    self.values[States[tau]] = self.values[States[tau]] + self.lr*(Gain - self.values[States[tau]])
                    
                if tau==Ts-1:
                    break
                else:
                    t+=1


# =========================== 2.2 model free control ===========================
class ModelFreeControl:
    """
    Base class for model free control algorithms 
    """

    def __init__(self, grid_world: GridWorld, discount_factor: float = 1.0):
        """
        Args:
            grid_world (GridWorld): GridWorld object
            discount_factor (float, optional): Discount factor gamma. Defaults to 1.0.
        """
        self.grid_world = grid_world
        self.discount_factor = discount_factor
        self.action_space = grid_world.get_action_space()
        self.state_space  = grid_world.get_state_space()
        self.q_values     = np.zeros((self.state_space, self.action_space))  
        self.policy       = np.ones((self.state_space, self.action_space)) / self.action_space # stocastic policy
        self.policy_index = np.zeros(self.state_space, dtype=int)                              # deterministic policy
        self.rng = np.random.default_rng(46466)      # only call this in collect_data()

    def get_policy_index(self) -> np.ndarray:
        """Return the policy

        Returns:
            np.ndarray: policy_index
        """
        for s_i in range(self.state_space):
            self.policy_index[s_i] = self.q_values[s_i].argmax()
        return self.policy_index
    
    def get_max_state_values(self) -> np.ndarray:
        max_values = np.zeros(self.state_space)
        for i in range(self.state_space):
            max_values[i] = self.q_values[i].max()
        return max_values
    
    def collect_data(self) -> tuple:
        """
        Use the stochastic policy to interact with the environment and collect one step of data.
        Samples an action based on the action probability distribution for the current state.
        """

        current_state = self.grid_world.get_current_state()  # Get the current state
        
        # Sample an action based on the stochastic policy's probabilities for the current state
        action_probs = self.policy[current_state]  
        action = self.rng.choice(self.action_space, p=action_probs)  

        next_state, reward, done = self.grid_world.step(action)  
        # if done:
        #     self.episode_counter +=1
        return action,next_state, reward, done



class MonteCarloPolicyIteration(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for MonteCarloPolicyIteration

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_evaluation(self, States, Actions, Rewards) -> None:
        """Evaluate the policy and update the values after one episode"""
        # TODO: Evaluate state value for each Q(s,a)

        # here we run one episode
        while True: #t=1,2,...
            At, next_state, reward, done = self.collect_data()
            Actions.append(At)
            if done: #t=T-1
                Rewards.append(reward) #append RT
                break
            else:
                States.append(next_state) #append St
                # Actions.append()
                Rewards.append(reward) #append Rt
        # print("Policy finding finished!")
        #then update Q, based on history. This time we implement every-visit
        #every states are always be updated
        Gain=0
        for t in range(len(States)-1,-1,-1): #t=T-1 - 0
            # print("t",t)
            Gain = self.discount_factor*Gain + Rewards[t+1]

            St = States[t]
            At = Actions[t]

            self.q_values[St,At] = self.q_values[St,At] + self.lr*(Gain - self.q_values[St,At])
        # print("update Q finished!")


        

    def policy_improvement(self) -> None:
        """Improve policy based on Q(s,a) after one episode"""
        # TODO: Improve the policy based on current Q
        for s in range(self.state_space):
            a_star = np.argmax(self.q_values[s])
            for a in range(4):#a=0,1,2,3
                if a==a_star:
                    self.policy[s,a] = self.epsilon/4 + 1 - self.epsilon
                else:
                    self.policy[s,a] = self.epsilon/4
        # print("update pi finished!")


    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Monte Carlo policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            
            state_trace   = [current_state]
            action_trace  = []
            reward_trace  = [-999] #for index convenient
            self.policy_evaluation(state_trace,action_trace,reward_trace)

            self.policy_improvement()

            #end one episode, DON'T change episilon
            # self.epsilon = 1/(iter_episode+1)
            
            iter_episode += 1
            # print(iter_episode)

        self.policy_index = self.get_policy_index()


class SARSA(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float):
        """Constructor for SARSA

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr      = learning_rate
        self.epsilon = epsilon

    def policy_eval_improve(self, s, a, r, s2, a2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        # TODO: Evaluate Q value after one step and improve the policy
        
        raise NotImplementedError

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the TD policy evaluation with epsilon-greedy
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here
            
            raise NotImplementedError

class Q_Learning(ModelFreeControl):
    def __init__(
            self, grid_world: GridWorld, discount_factor: float, learning_rate: float, epsilon: float, buffer_size: int, update_frequency: int, sample_batch_size: int):
        """Constructor for Q_Learning

        Args:
            grid_world (GridWorld): GridWorld object
            discount (float): discount factor gamma
            learning_rate (float): learning rate for updating state value
            epsilon (float): epsilon-greedy threshold
        """
        super().__init__(grid_world, discount_factor)
        self.lr                = learning_rate
        self.epsilon           = epsilon
        self.buffer            = deque(maxlen=buffer_size)
        self.update_frequency  = update_frequency
        self.sample_batch_size = sample_batch_size

    def add_buffer(self, s, a, r, s2, d) -> None:
        # TODO: add new transition to buffer
        raise NotImplementedError

    def sample_batch(self) -> np.ndarray:
        # TODO: sample a batch of index of transitions from the buffer
        raise NotImplementedError

    def policy_eval_improve(self, s, a, r, s2, is_done) -> None:
        """Evaluate the policy and update the values after one step"""
        #TODO: Evaluate Q value after one step and improve the policy
        raise NotImplementedError

    def run(self, max_episode=1000) -> None:
        """Run the algorithm until convergence."""
        # TODO: Implement the Q_Learning algorithm
        iter_episode = 0
        current_state = self.grid_world.reset()
        prev_s = None
        prev_a = None
        prev_r = None
        is_done = False
        transition_count = 0
        while iter_episode < max_episode:
            # TODO: write your code here
            # hint: self.grid_world.reset() is NOT needed here

            raise NotImplementedError
            