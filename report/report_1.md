
## Abstracts

## Introduction

## Related work

## Methods

### Problem definition
We aim to model the spread of infectious diseases in a population of agents. The agents can move freely in a two-dimensional environment and interact with each other. The goal is to minimize the spread of the disease by limiting interactions between agents. The agents can exchange information about their health status. The agents should learn to avoid each other based on the information they receive.

### Disease spread
The study of *Lasius niger* ants *\cite{Stroeymeyt2018}* reveals a fascinating strategy for mitigating disease spread. When exposed to the fungal pathogen *Metarhizium brunneum*, these ants alter their social network structure in a way that directly inhibits disease transmission. This is not simply a matter of avoiding contact with infected individuals. Rather, the entire colony demonstrates a shift in social interactions.

Individual ants, both infected and uninfected, adjust their behavior. Infected ants, for instance, spend more time outside the nest, reducing their exposure to nestmates. Meanwhile, uninfected ants also increase their spatial distance from other ants, particularly those who have been in contact with the pathogen. This change in behavior reinforces the already existing modularity of the network, effectively creating compartments that restrict the spread of the disease.

These complex behavioral adjustments will be incorporated into a reinforcement learning model focused on disease spread. The agents should be rewarded for exchanging information about their health status and penalized for coming into contact with infected individuals. The goal is to minimize the spread of the disease by encouraging social distancing.

The paper also highlihts that low-level exposure can be beneficial. Later improvements to the model could include a more nuanced approach to the rewards and penalties associated with different levels of exposure, and the possibility for an ant to develop immunity to the pathogen through active immunization. This would allow for a more detailed exploration of the trade-offs between information exchange and pathogen transmission.

### Information exchange
At the start, information exchange will be a simple reward, where agents will be simply rewarded for interacting with other agents. Potential improvements could include a more nuanced approach to information exchange, where agents would also be exchanging resources or other benefits. This could be modeled as a form of cooperation, where agents work together to achieve a common goal.

### Reinforcement learning

### Model performance measures
To see if any social distancing patterns emerge, we will transform the agent interactions into a network and analyze the network structure, as shown in *\cite{Stroeymeyt2018}*. We will look at network statistics that affect disease transmission, such as modularity (-), clustering (-), network efficiency (+) and degree centrality (+). We will also measure the average distance between agents, which would indicate that the agents are avoiding each other. First, these will be measured in a network before a pathogen is introducet to see the passive social distancing. Then we will introduce the pathogen and see how the network changes.

## Results

## Discussion

## References