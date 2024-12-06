# Modeling Social Distancing with Reinforcement Learning

Group project repository for the course Collective Behavior at UL FRI.

## Group members (Group F)
- Igor Nikolaj Sok - [JuiceVodka](https://github.com/JuiceVodka)
- Leon Todorov - [LeonTodorov](https://github.com/LeonTodorov)
- Nejc Ločičnik - [Nejc-Locicnik](https://github.com/Nejc-Locicnik)
- Andraž Zrimšek - [AndrazZrimsek](https://github.com/AndrazZrimsek)

## Project Description
Our chosen kick-off article is [Predator–Prey Survival Pressure is Sufficient to Evolve Swarming Behaviors](https://iopscience.iop.org/article/10.1088/1367-2630/acf33a), which has proven that Reinforcement Learning (RL) with a rather simple reward policy is sufficient to evolve swarming behaviours in a predator-prey system.

Modeling with Reinforcement Learning involves an agent learning optimal actions through trial and error based on feedback from the environment, aiming to maximize cumulative rewards over time based on the chosen policy. In contrast, fuzzy logic models (a traditional approach in Collective Behavior) handle uncertainty and approximate reasoning by applying predefined rules to make decisions, rather than learning and adapting based on interaction outcomes.

Since using Reinforcement Learning (RL) to model collective behavior is quite different from traditional methods and still relatively uncommon in this field, we are interested in exploring its potential by modeling other observed patterns in nature.

Our current idea is to model the emergence of social distancing in response to infectious disease spread in nature (referencing: [Infectious Diseases and Social Distancing in Nature](https://www.science.org/doi/abs/10.1126/science.abc8881)). We plan to begin with a simple RL policy that rewards information exchange (grouping/socializing) and penalizes pathogen exchange (disease spread), then build on that. The outline of costs and rewards, along with expected behavior, is described in the article: [The Tradeoff Between Information and Pathogen Transmission in Animal Societies](https://nsojournals.onlinelibrary.wiley.com/doi/abs/10.1111/oik.08290). 

Our goal is to explore whether a basic policy can lead to the evolution of known social distancing patterns (e.g., in ants: [Social Network Plasticity Decreases Disease Transmission in a Eusocial Insect](https://www.science.org/doi/10.1126/science.aat4793)).

## Simulation setup

### Dependencies
Just run the things below and install whatever dependencies you're missing (latest version). The only thing to note is that PyGlet requires an older version (1.5.27 works).

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
### First Report - 15. 11. 2024: 
- ~~Proper review of the use of RL in Collective Behavior~~
- ~~Review the referenced articles to gain a clearer understanding of our methodology~~
- ~~Decide on an ML library that supports RL and build a basic framework for simulations~~
- ~~Code basic "dot" agents (movement + interactions through collisions)~~
- ~~refine report before deadline~~

### Second Report - 6. 12. 2024:
- Refine our methodology based on the first report, write second report
- ~~Adapt agent movement so it's more aligned with ant movement~~
- ~~Change the policy neural network to include agent health status on the input~~
- ~~Start testing different reward policies aimed at emerging social distancing traits (interactions need to be sufficiently non-random and contribute to reducing pathogen flow)~~
- ~~Potentially increase the complexity of agent interactions (if necessary)~~

### Final report - 10. 1. 2025:
- Succeed (or fail) in emulating social distancing traits
- Perform extensive experiments (environment, reward policy, agent movement/interactions, etc.)
- Compare emergent behavior with observations in ants from [Social Network Plasticity Decreases Disease Transmission in a Eusocial Insect](https://www.science.org/doi/10.1126/science.aat4793) 
