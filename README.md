# Modeling Social Distancing with Reinforcement Learning

Group project repository for the course Collective Behavior at UL FRI.

## Group members (Group F)
- Igor Nikolaj Sok - [JuiceVodka](https://github.com/JuiceVodka)
- Leon Todorov - [LeonTodorov](https://github.com/LeonTodorov)
- Nejc Ločičnik - [Nejc-Locicnik](https://github.com/Nejc-Locicnik)
- Andraž Zrimšek - [AndrazZrimsek](https://github.com/AndrazZrimsek)

## Project Description

Our chosen kick-off article was [*Predator–Prey Survival Pressure is Sufficient to Evolve Swarming Behaviors*](https://iopscience.iop.org/article/10.1088/1367-2630/acf33a), which demonstrated that Reinforcement Learning (RL) with a simple reward policy could evolve swarming behaviors in a predator-prey system. This provided the foundation for our exploration of RL's potential to model complex collective behaviors in nature.

We applied RL to simulate the emergence of social distancing behaviors in response to infectious disease spread. Our approach was inspired by studies like [*Infectious Diseases and Social Distancing in Nature*](https://www.science.org/doi/abs/10.1126/science.abc8881) and [*The Tradeoff Between Information and Pathogen Transmission in Animal Societies*](https://nsojournals.onlinelibrary.wiley.com/doi/abs/10.1111/oik.08290), which explored how animals adapt their social interactions to minimize disease transmission. We designed a basic RL policy where agents were rewarded for minimizing contact with infected individuals and penalized for disease transmission. Our goal was to examine whether such a simple reward structure could lead to the emergence of social distancing patterns, similar to those observed in animals like ants ([*Social Network Plasticity Decreases Disease Transmission in a Eusocial Insect*](https://www.science.org/doi/10.1126/science.aat4793)).

The results of our experiments showed that RL could successfully model social distancing behaviors. Through network analysis, we observed increased modularity and reduced interactions between healthy and infected agents, indicating that the agents learned to maintain a greater distance from infected individuals. Additionally, we tested several reward structures and external tasks, ultimately finding that a simple task (wall-touching) significantly improved agent behavior and facilitated social distancing. We also experimented with alternative information exchange methods, such as pheromone-based systems, but found that they were less effective compared to the movement-based tasks. Our work demonstrates the potential of RL in simulating disease dynamics and highlights its applicability in fields such as swarm robotics and public health simulations.


## Simulation setup

### Dependencies
Create the conda environment using environment.yaml.
```
conda env create --name SV --file=environment.yaml
```

### Test simulation environment
Run environment without the policy model with random actions (set env parameters in `src/config/env_params.json`):
```
python pps.py
```

### Training the policy model
Run training (set env parameters in `src/config/train_params.json`) - don't forget to name you're policy model in the file:
```
python train_zoo.py
```

### Evaluating the policy model
Run evaluation (set parameters in `src/config/eval_params.json`) - don't forget to change the model .zip you're testing (in the file):
```
python infer.py
```

## Roadmap
### ~~First Report - 15. 11. 2024~~: 
- ~~Proper review of the use of RL in Collective Behavior~~
- ~~Review the referenced articles to gain a clearer understanding of our methodology~~
- ~~Decide on an ML library that supports RL and build a basic framework for simulations~~
- ~~Code basic "dot" agents (movement + interactions through collisions)~~
- ~~refine report before deadline~~

### ~~Second Report - 6. 12. 2024~~:
- ~~Refine our methodology based on the first report, write second report~~
- ~~Adapt agent movement so it's more aligned with ant movement~~
- ~~Change the policy neural network to include agent health status on the input~~
- ~~Start testing different reward policies aimed at emerging social distancing traits (interactions need to be sufficiently non-random and contribute to reducing pathogen flow)~~
- ~~Potentially increase the complexity of agent interactions (if necessary)~~

### ~~Final report - 10. 1. 2025:~~
- ~~Perform extensive experiments (environment, reward policy, agent movement/interactions, etc.)~~
- ~~Compare emergent behavior with observations in ants from [Social Network Plasticity Decreases Disease Transmission in a Eusocial Insect](https://www.science.org/doi/10.1126/science.aat4793)~~
- ~~Conclude whether emulating social distancing traits was successful~~
- ~~Update the report (text/figures) with the latest results~~
- ~~Make a cool visualization for the presentation~~
