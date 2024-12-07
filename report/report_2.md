## Individual experiments for reward 3

### Diminishing reward
A simple reward system was implemented, where two agents were penalized upon interaction if only one of them was infected (since that could spread infection, both infected interactiong was not penalized), otherwise, the agents were rewarded. The reward system kept track of recent interactions between the same angents, and the reward was 

$$
\mathrm{reward}(a, b) = \begin{cases}
    -\lambda & \text{if } infected(a) != infected(b) \\ 
    +\sigma * \gamma (1 - recent(a,b)) & \text{otherwise.}
\end{cases}
$$

where recent(a,b) is set to 1 upon interacting, and has a diminishing factor of 0.9 every step. The purpose of the diminishing reward is to avoid rewarding consecutive interactions between the same agents, as those do not spread any new information. 

This simple system has showed to be able to lead to well performing agents with the interaction network showing similarities to the real world examples.

Keeping the rewards simple makes the model applciable to a wider range of species, not just ants, while ant behaviour can be used as a baseline to compare results against.

### Optional reward components
To enhance the effectiveness of the agents' behavior further, additional reward components inspired by the ones in \{predator_prey_citation} were implemented and can be used in addition to the diminishing reward to address specific aspects of agent dynamics:     
* Wall collision penalty: In the case of non-periodic environments a simple wall collision penalty can be applied to discourage agents from colliding with walls:

$$
\mathrm{reward}(a) = \begin{cases}
    -\lambda & \text{if a collides with any wall}\\ 
    0 & \text{otherwise.}
\end{cases}
$$

* Control Penalty: A decorative reward that mimics energy consumption due to movement applied proportional to the magnitude of the agent's control inputs $$aF$$ and $$aR$$, which causes agents to exhibit laziness.

$$
\text{reward}(aF, aR) = -(\alpha |aF| + \beta |aR|)
$$

### Interaction network
In order to be able to analyze the performance and statistics of our agents, we needed to create a network of interactions. In every step of the evaluation execution, we keep track of interactions between agents and save it to an $n$ x $n$ matrix. Two agents are considered to be interacting when the distance between them is 2 x [agent width]. Upon completion of the evaluation run, we construct a network from those interactions. First, the interaction matrix is normalized. A node is created for every agent, and an edge is created between each node if their interaction value is > 0.01. We also save the information about the agent's infection status. The actual value from the interaction matrix is saved as the weight of the edge. This network is then used to calculate appropriate network measures such as clustering, modularity (between infected and non-infected), network efficiency and more. This allows us to get a better understanding of how agents are interacting with one another.

### Alternative information exchange between agents

The current interaction between agents is fully depending on collisions. While it works for our goal (mapping out interactions into a social network and seeing a reduction in interaction between healthy and infected agents aka social distancing), it doesn't produce a very clear visualization. The visual distance between a collision and a near miss (which doesn't count as an interaction) is not noteceable enough.

On that note we attempted to introduce an alternative way of agents communicating information between each other and since our main point of reference are ants, we went with a simplified pheromone system. Each agent leaves a mark (+1) in the surrounding area (whatever bigger grid its position belongs to to simplify calculation) accumulating into a concentration heatmap, which decays through time steps. The released pheromones can be positive (indicating "safe" from healthy agents) or negative (indicating "danger" from infected agents). If the concentration is high enough at agent position, it gets percieved and added to observation. This way agents have an indicator of danger/safety ahead, which would hopefully increase the distance between healthy and infected agents, making the visualization clearer.

This sadly didn't work out as planned. We currently assume it doesn't work because there isn't a clear correlation between observation (policy neural network input) and actions (output), which leads to higher rewards. With collisions it's straightforward, the agent percieves the 6 nearest agents in its field of view and corrects its heading and force to hit or avoid them depending if they're healthy or infected, which leads to a higher cumulative reward. With pheromones on the other hand, an agent detects it's in a certain pheromone (positive or negative), but it can't determine what action will lead to better rewards. Though in theory the actions should be stop (force to 0) for positive and flee for negative.

### Results
To test the performance of our trained agents, we ran an episode with 5000 steps with a random untrained network and a trained network. For each, we constructed a network of all the interactions thorughout the episode. Both networks are displayed bellow, where healthy agents are colored blue, and infected agents are colored orange.

![alt text](https://github.com/JuiceVodka/SV_2024-25_groupF/blob/master/report/figures/randomNet.png "random network") 
![alt text](https://github.com/JuiceVodka/SV_2024-25_groupF/blob/master/report/figures/learnedNet.png "trained network")

We can see that in the random network, no structure is apparent. On the learned network, we can see that a separation is starting to occur, where infected agents are seen to interact less with the healthy, but still interact with one another. This behaviour was expected. The structure becomes even more apparent if we only keep the edges with weight > 0.2 to only keep the edges with significant interactions. The network can be seen in the following figure.

![alt text](https://github.com/JuiceVodka/SV_2024-25_groupF/blob/master/report/figures/filteredNet.png)

To get a better understanding of the networks, and to see the network properties that are harder to observe from an image, we also calucalted some network metrics which are relevant to pathogen transmission. 
| Metric              | Random | Trained |
| :---------------- | -----: | -----: |
| Clustering        |   0.041   | 0.064 |
| Modularity           |   -0.016   | 0.050 |
| Density    |  0.728   | 0.779 |
| Efficiency |  0.864   | 0.889 |

These demonstrate that some simple social distancing behaviour has emerged, but to get a better insight, we still need to perform more tests. 

#### Future experiments
To follow the experiment described in (ref Social network plasticity decreases disease transmission in a eusocial insect), we will perform multiple runs of each random, learned without infected and learned with infected individuals. We will use the random network to normalize the metrics of our learned networks, and then compare the changes in network structures before and after a pathogen is introduced. We will keep track of the same ants in pairs of pre- and post-introduction executions.
