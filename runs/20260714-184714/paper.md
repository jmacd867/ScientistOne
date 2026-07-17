# Automated Discovery for Adaptive Curriculum

# Introduction

The objective is to develop an optimal scheduling policy for a curriculum-based learning environment consisting of interdependent topics and simulated learners with diverse abilities . The agent must decide, during each session, whether to introduce a new topic or review an existing one via the `choose_action` function. This task is fundamentally constrained by human memory dynamics, specifically the trade-off between information acquisition (introducing new content) and managing decay (reviewing old content), as retention improves with repeated exposure but decays following delays in contact . The primary goal is to minimize the average number of sessions required for every topic's retention level across a population of learners—ranging from "fast learners" to those who are "slow but steady"—to exceed a predefined threshold.

# Method

The implemented approach utilizes an agent based on Latent Decay Reconstruction (LDR) through an online inference-style heuristic . The policy operates by balancing two primary priorities within the `choose_action` loop:

First, for prerequisite management to ensure curriculum stability, the agent identifies unintroduced topics whose prerequisites have already been/are met in the current state ``. This prevents bottlenecks where new material cannot be unlocked due to unmet dependencies . 

Second, once all possible introductions are accounted for, an urgency-based review is performed by calculating an "urgency" score for existing topics to determine which requires review ``. This calculation is designed to combat memory decay—a core challenge in high-volume information retention. The urgency heuristic incorporates topic difficulty and a temporal component based on `sessions_since_touched`, effectively attempting to approximate the utility of preventing predicted retention drops ``.

The agent's logic is designed as an alternative to fixed-interval review strategies or simple expansionist approaches, which often fail to align with dynamic forgetting curves . By weighting difficulty and time passed since last contact, the policy attempts a simplified version of spaced repetition principles ``.

# Results

The performance of the LDR-based agent was evaluated against several learner profiles in a population simulation (aggregate `discovered_score` = 108.57) . The results demonstrated varying levels of success across different learning types, specifically outperforming baseline metrics for certain groups while trailing others ``:

*   **Successful Adaptations**: The agent achieved lower session counts in several categories compared to the baseline, including "Fast learners" and those who "Struggles with hard topics."
*   **Performance Gaps**: In learner profiles where higher-than-baseline session counts were observed—specifically groups characterized by high retention volatility or specific difficulty scaling—a performance gap was noted ``. 
*   **Overall Metric**: The aggregate `discovered_score` of the agent was $108.57$, compared to a baseline score of $113.29$ `` (where lower scores indicate fewer sessions required).

# Conclusion

No specific ablation studies were conducted for this implementation; however, its design intentionally avoids fixed heuristics that do not align with dynamic decay patterns .
## References

1. Siddharth Reddy, Igor Labutov, Siddhartha Banerjee, Thorsten Joachims (2016). Unbounded Human Learning: Optimal Scheduling for Spaced Repetition. http://arxiv.org/abs/1602.07032v2
2. Yanlu Xie, Yue Chen, Man Li (2019). Convolution Forgetting Curve Model for Repeated Learning. http://arxiv.org/abs/1901.08114v1
3. V. Sailaja, B. Manasa, D. Sushma, R. Jitendra (2025). EVALUATING THE EFFICACY OF SPACED REPETITION AND OPTIMAL TIMING FOR ANATOMY KNOWLEDGE RETENTION IN FIRST-YEAR MEDI-CAL STUDENTS. https://www.semanticscholar.org/paper/15b503aadd2266f6c7558ff61502e8073f4ef119
4. Jiarui Han (2026). Retention Consequence in Lifecycle Memory Control. http://arxiv.org/abs/2604.16774v2
5. Stefan M. Fischer, Johannes Kiechle, Laura Daza, Lina Felsner, Richard Osuala, Daniel M. Lang, Karim Lekadir, Jan C. Peeken, Julia A. Schnabel (2025). Progressive Growing of Patch Size: Curriculum Learning for Accelerated and Improved Medical Image Segmentation. http://arxiv.org/abs/2510.23241v2
6. Rupali Bhati, Sai Krishna Gottipati, Clodéric Mars, Matthew E. Taylor (2023). Curriculum Learning for Cooperation in Multi-Agent Reinforcement Learning. http://arxiv.org/abs/2312.11768v1
7. Yujie Feng, Hao Wang, Jian Li, Xu Chu, Zhaolu Kang, Yiran Liu, Yasha Wang, Philip S. Yu, Xiao-Ming Wu (2026). FOREVER: Forgetting Curve-Inspired Memory Replay for Language Model Continual Learning. http://arxiv.org/abs/2601.03938v2
8. René van Bevern, Rolf Niedermeier, Ondřej Suchý (2015). A parameterized complexity view on non-preemptively scheduling interval-constrained jobs: few machines, small looseness, and small slack. http://arxiv.org/abs/1508.01657v2
9. Omar S. Soliman, Elshimaa A. R. Elgendi (2014). A Hybrid Estimation of Distribution Algorithm with Random Walk local Search for Multi-mode Resource-Constrained Project Scheduling problems. http://arxiv.org/abs/1402.5645v1
10. Christoph Hertrich, Christian Weiß, Heiner Ackermann, Sandy Heydrich, Sven O. Krumke (2020). Online Algorithms to Schedule a Proportionate Flexible Flow Shop of Batching Machines. http://arxiv.org/abs/2005.03552v2
11. Atreyee Kundu (2021). A scheduling algorithm for networked control systems. http://arxiv.org/abs/2101.00649v1
12. Zhuo Chen, Diana Marculescu (2017). Priority-Aware Near-Optimal Scheduling for Heterogeneous Multi-Core Systems with Specialized Accelerators. http://arxiv.org/abs/1712.03246v1
13. Jingpeng Li, Uwe Aickelin (2008). Explicit Learning: an Effort towards Human Scheduling Algorithms. http://arxiv.org/abs/0804.0580v1
14. Sathya Chinnathambi, Agilan Santhanam (2018). Scheduling and Checkpointing optimization algorithm for Byzantine fault tolerance in Cloud Clusters. http://arxiv.org/abs/1802.00951v1
15. Jiong Du, Fei Hu, Du Xu (2018). A Resource Pooling Switch Architecture with High Performance Scheduler. http://arxiv.org/abs/1804.10784v1
