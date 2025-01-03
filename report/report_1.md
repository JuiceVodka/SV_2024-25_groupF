
## Abstracts

## Introduction

### Related work

In the article \textbf{Predator–prey survival pressure is sufficient to evolve swarming behaviors}\cite{li2023predator} the authors employ a simple reinforcement learning (RL) approach to model predator and prey behaviours. The prey agents receive rewards based on their role; predator agents receive rewards if they successfully catch prey, while the prey receives rewards for staying alive. This differs drastically from other behaviour modelling approaches, as the authors refrain from handcrafting any drives to make the model exhibit the expected behaviour, which many traditional models do. A problem with traditional approaches is that agents based on static handcrafted rules often fail to capture the dynamic nature and strategies of the biological world. Reinforcement learning addresses this issue nicely, since it only presents rewards that encourage or discourage certain behaviours. The authors, using a predator-prey coevolution framework based on cooperative–competitive multiagent RL. They found, that such an approach is sufficient for a rich diversity of emergent behaviours to evolve. They noticed flocking and swarming behaviours developing in prey agents, while predators started employing dispersion tactics, confusion and marginal predation phenomena. The results of the study offer useful insights into how different group behaviours can be modelled using a simple RL approach. We aim to use said approach to model disease spread, by modifying the agents and the rewards they will receive.

The intricacies of disease spread and the social distancing behaviours that emerge from it in nature are explained in the article \textbf{Infectious diseases and social distancing in nature}\cite{stockmaier2021infectious}. The authors view social distancing as a natural consequence of disease across animals, both human and non-human. Subjects typically exhibit social distancing-like behaviour either as precautionary measures or as physiological consequences of infection in sick individuals. The dive deep into the underlying mechanisms driving the behaviour of both infected and non-infected subjects when a pathogen is present within a population.

Valéria Romano et al. explain, that social distancing in itself is not a sufficient strategy to limit disease spread, as members of a population have a innate need for information transfer, which brings its own benefits to subjects engaging in information exchange. The article \textbf{tradeoff between information and pathogen transmission in animal societies}\cite{romano2022tradeoff} describes the mechanisms underlying behaviour and maintenance of individual relationships when the threat of disease spread is present. They outline the evolutionary mechanism of social transmission and present evidence that network plasticity is a result of individuals navigating between costs and benefits of social relationships. The tradeoffs described in the article can provide useful insights for modelling disease spread in populations.

These studies offer valuable frameworks and insights into behavior modeling, demonstrating how agents adaptively respond to environmental pressures. Leveraging these foundational approaches, our work will employ a reinforcement learning model that integrates social distancing behaviors to understand the complexities of disease spread and agent interaction.

## Methods

### Problem definition
We aim to model the spread of infectious diseases in a population of agents. The agents can move freely in a two-dimensional environment and interact with each other. The goal is to minimize the spread of the disease by limiting interactions between agents. The agents can exchange information about their health status. The agents should learn to avoid each other based on the information they receive.

### Disease spread
The study of *Lasius niger* ants *\cite{Stroeymeyt2018}* reveals a fascinating strategy for mitigating disease spread. When exposed to the fungal pathogen *Metarhizium brunneum*, these ants alter their social network structure in a way that directly inhibits disease transmission. This is not simply a matter of avoiding contact with infected individuals. Rather, the entire colony demonstrates a shift in social interactions.

Individual ants, both infected and uninfected, adjust their behavior. Infected ants, for instance, spend more time outside the nest, reducing their exposure to nestmates. Meanwhile, uninfected ants also increase their spatial distance from other ants, particularly those who have been in contact with the pathogen. This change in behavior reinforces the already existing modularity of the network, effectively creating compartments that restrict the spread of the disease.

These complex behavioral adjustments will be incorporated into a reinforcement learning model focused on disease spread. The agents should be rewarded for exchanging information about their health status and penalized for coming into contact with infected individuals. The goal is to minimize the spread of the disease by encouraging social distancing.

The paper also highlights that low-level exposure can be beneficial. Later improvements to the model could include a more nuanced approach to the rewards and penalties associated with different levels of exposure, and the possibility for an ant to develop immunity to the pathogen through active immunization. This would allow for a more detailed exploration of the trade-offs between information exchange and pathogen transmission.

### Simulation Environment
The simulation framework used in this study is based on the multi-agent RL environment developed in Li et al., 2023 \cite{li2023predator}. The environment is a two-dimensional continuous space with periodic boundary conditions, meaning that when an agent crosses one edge of the square environment, it reappears on the opposite side with the same velocity.

We plan to adapt the framework for our own needs, which will mainly include changing how agents perceive the environment as we need to include the health status of agents, what happens when agents are close by (possibility of pathogen transmission) and the reward policy used to train the model. In this way we don't have to waste time building our own framework and can invest more time experimenting with different reward policies.
### Agent Dynamics
Agents are depicted as circles with a short line segment indicating their heading direction. The forces acting on an agent include both active (controllable) and passive (inherent) forces:

Active Forces (Agent's Actions):
- A forward movement force aligned with the agent's current heading: $a_F$
- A rotational force to adjust the agent's heading direction: $a_R$

Passive Forces:
- A dragging force, acting in the opposite direction of the agent's velocity, which simulates friction or resistance: $F_d$
- A repulsive force between agents in contact: $F_a$

Active forces             |  Passive forces
:-------------------------:|:-------------------------:
<img src="agent_active.png" width="50%"> | <img src="agent_passive.png" width="50%">

Each timestep, the simulation updates the agents' positions and velocities based on the sum of forces acting on them.  
These can be summed up as:  

$$ \dot{x} = v $$  
$$ \dot{v} = \frac{ha_F + F_d + F_a}{m} $$  
$$ \dot{\theta} = a_R $$  

Where:
- $x \in \mathbb{R}^2$ is the agent's position,
- $v \in \mathbb{R}^2$ is the agent's velocity,
- $\theta \in [-\pi, \pi]$ is the agent's heading angle,
- $h \in \mathbb{R}^2$ is the unit vector representing the agent's heading direction, calculated as $h = [\cos(\theta), \sin(\theta)]^T$,
- $m \in \mathbb{R}$ is the agent's mass.

To align the simulation with ant-like movement rather than the smooth, bird-like flight patterns of the original framework, several parameters will require significant adjustments. In their current settings—such as drag coefficient, stiffness coefficient, maximum forward acceleration, and rotational acceleration—the parameters are optimized for smooth, continuous paths with limited turning sharpness and no halting, resembling bird flight. However, to better emulate the more abrupt, flexible movement characteristic of ants in a 2D bird's-eye view, we will modify these parameters. Specifically, we’ll increase rotational flexibility, reduce constraints on movement continuity, and adjust stopping behaviors to allow agents more freedom in directional shifts and pauses.

### Basic Reward Policy
Our current plan for a reward policy that will produce social distancing patterns in agents behavior is fully based on agent interaction. We plan to promote social behavior by rewarding them based on the number of near-by agents, which should result in agents grouping up. Social distancing will be promoted by penalizing healthy agents for being near infected ones and vice versa. Specific reward and penalty values will be determined as we start experimenting.

We plan to start with this simple reward policy, before scaling the simulation complexity with a more nuanced agent interactions e.g., where agents would also be exchanging resources or other benefits. This could be modeled as a form of cooperation, where agents work together to achieve a common goal.

### Model performance measures
To see if any social distancing patterns emerge, we will transform the agent interactions into a network and analyze the network structure, as shown in *\cite{Stroeymeyt2018}*. We will look at network statistics that affect disease transmission, such as modularity (-), clustering (-), network efficiency (+) and degree centrality (+). We will also measure the average distance between agents, which would indicate that the agents are avoiding each other. First, these will be measured in a network before a pathogen is introducet to see the passive social distancing. Then we will introduce the pathogen and see how the network changes.

Other than that, emergent social distancing behavior should also be quite obvious in the simulation visualization. We expect healthy agents to stay away from infected ones and vice versa. This can be clearly seen if we increase the density of agents in the environment, we should see infected agents get isolated, forming empty circles around them, while healthy agents fill the rest of the simulation space.

## Results
No experiments have been run so far, so there are no results to showcase yet.

## Discussion and Future Work
As we have not yet obtained results, this section will outline our immediate plans and objectives leading up to the second report deadline.

Our primary goal is to adapt the existing reinforcement learning (RL) framework to align with our modeling objectives. This will involve the following steps:

1. Agent Movement Adaptation: We will modify agent movement parameters to better simulate ant-like behavior. Current parameters are tuned for smoother, bird-like paths, whereas ants exhibit more abrupt, flexible movements. Adjustments will include increasing rotational flexibility, reducing constraints on movement continuity, and introducing options for sudden stops and directional changes.

2. Policy Network Adjustment: We plan to reconfigure the policy network, which is the agent’s decision-making neural network. This network will be updated to process relevant environmental information and output appropriate actions for each agent based on its health status, position, and nearby agents. This customization will enable the agents to better respond to their surroundings, a necessary step for simulating disease-avoidance behaviors.

3. Reward Policy Development: Our initial reward policy will be simple, rewarding agents for actions that minimize close interactions with infected agents and penalizing actions that lead to unnecessary contact. This will serve as a foundational incentive structure, helping us to examine how agents balance exploration with avoidance behaviors in a disease-prone environment.

These steps will provide the basis for preliminary testing and fine-tuning. Moving forward, we plan to iterate on these adaptations and expand our focus to include more complex reward structures and interaction rules based on early observations from the simulations.

This organization clarifies each objective while positioning them as concrete steps. Additionally, the phrasing of each component emphasizes the purpose behind each modification, making the progression toward expected outcomes more cohesive and actionable.

## References
