# Modeling social distancing with reinforcement learning (title work in progress :))

## Team Members
- Igor Nikolaj Sok (JuiceVodka)
- Leon Todorov (LeonTodorov)
- Nejc Ločičnik (Nejc-Locicnik)
- Andraž Zrimšek (AndrazZrimsek)

## Project
Reinforcement learning using a simple reward sistem has been proven to be able to evolve swarming behaviours in a predator-prey sistem (https://iopscience.iop.org/article/10.1088/1367-2630/acf33a). 
Our idea is to take the RL approach and apply it to a different problem, specifically emergence of social distancing in populations. The system will be constructed so that information exchange is 
rewarded while pathogen exchange is penalized. Our goal is to see if simple rules can lead to the evolution of of known social distancing habits. (example in ants: https://www.science.org/doi/10.1126/science.aat4793).

## Roadmap
1. first report: define a RL model and tune it's parameters so that we can see it is learning.
2. second report: add features and increase model comlexity?, perform experiments to see if social distancing traits are emerging (if interactions are sufficiently non-random and contribute to reduced pathogen flow).
3. final report: perform extensive experiments and compare them to the results of https://www.science.org/doi/10.1126/science.aat4793 to find emerging traits that are observed in their experiments.
