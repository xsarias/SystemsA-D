
# WORKSHOP #3
## 🏀 Kaggle System Simulation

This project implements a simulation system to estimate the probability of victory for all possible matchups in the NCAA 2025 basketball tournament. The simulation uses historical seed data as input and applies a logistic regression model to generate baseline predictions, followed by a perturbation analysis to evaluate the system's robustness under noisy input conditions. The entire process is encapsulated in a Python class, TournamentSimulator, designed according to modular system architecture principles such as data ingestion, preprocessing, prediction engine, and analytics.

## 🧮 Feature Overview

The following table summarizes the main features used (or engineered) during the simulation process. These features are either derived from historical team performance or directly computed from tournament seed data:

| Feature                | Description / Calculation                                                |
|------------------------|---------------------------------------------------------------------------|
| Seed difference        | Difference between the seeds assigned to each team: `SeedB - SeedA`     |
| Win-Loss Ratio         | Total number of wins divided by total number of games played            |
| Average points scored  | Average number of points scored per game                                |
| Average points allowed | Average number of points conceded per game                              |
| Offensive efficiency   | Points scored divided by estimated possessions                          |
| Defensive efficiency   | Points allowed divided by estimated possessions                         |
| Scoring margin         | Average point differential per game (`scored - conceded`)    

To reproduce the simulation, refer to the main Python class in `simulator.py` and run the execution script `run_simulation.py`. Required dependencies are listed in requirements.txt.

📍[Simulation Report](SimulationReport.pdf)