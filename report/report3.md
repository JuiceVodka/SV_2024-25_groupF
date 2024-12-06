## Individual experiments for reward 3

### Diminishing reward
A simple reward system was implemented, where two agents were penalized upon interaction if only one of them was infected (since that could spread infection, both infected interactiong was not penalized), otherwise, the agents were rewarded. The reward system kept track of recent interactions between the same angents, and the reward was 

$$
\mathrm{reward}(a, b) = \begin{cases}
    -1 & \text{if } infected(a) != infected(b) \\ 
    +1 * (1 - recent(a,b)) & \text{otherwise.}
\end{cases}
$$

where recent(a,b) is set to 1 upon interacting, and has a diminishing factor of 0.9 every step. The purpose of the diminishing reward is to avoid rewarding consecutive interactions between the same agents, as those do not spread any new information. 

This simple system has showed to be able to lead to well performing agents with the interaction network showing similarities to the real world examples.

### Interaction network
In order to be able to analyze the performance and statistics of our agents, we needed to create a network of interactions. In every step of the evaluation execution, we keep track of interactions between agents and save it to an $n$ x $n$ matrix. Two agents are considered to be interacting when the distance between them is 2 x [agent width]. Upon completion of the evaluation run, we construct a network from those interactions. First, the interaction matrix is normalized. A node is created for every agent, and an edge is created between each node if their interaction value is > 0.01. We also save the information about the agent's infection status. The actual value from the interaction matrix is saved as the weight of the edge. This network is then used to calculate appropriate network measures such as clustering, modularity (between infected and non-infected), network efficiency and more. This allows us to get a better understanding f how agents are interacting with one another.
