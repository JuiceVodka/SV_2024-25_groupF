
## Abstracts

## Introduction

## Related work

In the article \textbf{Predator–prey survival pressure is sufficient to evolve swarming behaviors}\cite{li2023predator} the authors employ a simple reinforcement learning (RL) approach to model predator and prey behaviours. The prey agents receive rewards based on their role; predator agents receive rewards if they successfully catch prey, while the prey receives rewards for staying alive. This differs drastically from other behaviour modelling approaches, as the authors refrain from handcrafting any drives to make the model exhibit the expected behaviour, which many traditional models do. A problem with traditional approaches is that agents based on static handcrafted rules often fail to capture the dynamic nature and strategies of the biological world. Reinforcement learning addresses this issue nicely, since it only presents rewards that encourage or discourage certain behaviours. The authors, using a predator-prey coevolution framework based on cooperative–competitive multiagent RL. They found, that such an approach is sufficient for a rich diversity of emergent behaviours to evolve. They noticed flocking and swarming behaviours developing in prey agents, while predators started employing dispersion tactics, confusion and marginal predation phenomena. The results of the study offer useful insights into how different group behaviours can be modelled using a simple RL approach. We aim to use said approach to model disease spread, by modifying the agents and the rewards they will receive.

The intricacies of disease spread and the social distancing behaviours that emerge from it in nature are explained in the article \textbf{Infectious diseases and social distancing in nature}\cite{stockmaier2021infectious}

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

### Simulation  

#### Environment
Similar to \cite{li2023predator}, we established a simple physics-based simulation environment where the agents can interact. The environment is a two-dimensional continuous space with periodic boundary conditions, meaning that when an agent crosses one edge of the square environment, it reappears on the opposite side with the same velocity.

#### Agent dynamics
Agents are depicted as circles with a short line segment indicating their heading direction. The forces acting on an agent include both active (controllable) and passive (inherent) forces:

Active Forces (Agent's Actions):
- A forward movement force aligned with the agent's current heading: $$a_F$$
- A rotational force to adjust the agent's heading direction: $$a_R$$

Passive Forces:
- A dragging force, acting in the opposite direction of the agent's velocity, which simulates friction or resistance: $$F_d$$
- A repulsive force between agents in contact: $$F_a$$

Each timestep, the simulation updates the agents' positions and velocities based on the sum of forces acting on them. These can be summed up as:

$$
\dot{x} = v
$$  

$$
\dot{v} = \frac{ha_F + F_d + F_a}{m}
$$  

$$
\dot{\theta} = a_R
$$  

Where:
- $$x \in \mathbb{R}^2$$ is the agent's position,
- $$v \in \mathbb{R}^2$$ is the agent's velocity,
- $$\theta \in [-\pi, \pi]$$ is the agent's heading angle,
- $$h \in \mathbb{R}^2$$ is the unit vector representing the agent's heading direction, calculated as $$h = [\cos(\theta), \sin(\theta)]^T$$ (with $$||h|| = 1$$),
- $$m \in \mathbb{R}$$ is the agent's mass.

Various parameters such as agent masses, drag coefficient, stiffness coefficient, maximum forward acceleration, rotational acceleration, timestep duration, and others can be freely adjusted to customize the agents' behaviors.


### Model performance measures
To see if any social distancing patterns emerge, we will transform the agent interactions into a network and analyze the network structure, as shown in *\cite{Stroeymeyt2018}*. We will look at network statistics that affect disease transmission, such as modularity (-), clustering (-), network efficiency (+) and degree centrality (+). We will also measure the average distance between agents, which would indicate that the agents are avoiding each other. First, these will be measured in a network before a pathogen is introducet to see the passive social distancing. Then we will introduce the pathogen and see how the network changes.

## Results

## Discussion

## References
