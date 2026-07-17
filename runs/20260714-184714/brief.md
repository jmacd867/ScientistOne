# Research brief: adaptive_curriculum

The task involves designing an optimal scheduling policy for a curriculum-based learning environment where the agent must balance introducing new topics versus reviewing existing ones to maximize long-term retention across diverse learners. This problem is fundamentally shaped by human memory dynamics, specifically the trade-off between information acquisition and decay.

## Findings
- Human ability to retain information improves with repeated exposure but decays following a delay since last exposure, creating an inherent trade-off in educational software design between teaching new material and reviewing old content [0]. {ev:ev_0016}
- Spaced repetition is a learning strategy grounded in cognitive psychology that involves reviewing information at systematically increasing intervals to combat the challenges of high-volume information retention [2]. {ev:ev_0017}
- Standard mathematical forgetting curve models often fail to accurately represent memory processes during repeated learning, necessitating more complex approaches like convolution models for better simulation of memory ability and decay [1]. {ev:ev_0018}
- In lifelong or continual learning scenarios, information that has been successfully admitted into memory can later suffer from post-admission failure, where once a premise becomes an assumption/residue, it may be demoted or evicted during maintenance processes [3]. {ev:ev_0019}
- Memory replay methods used in continuous learning often rely on fixed heuristics that do not align with the dynamic nature of forgetting curves [6]. {ev:ev_0020}

## Baselines
Potential baselines for this task include: 1) A Fixed-Interval Review strategy (reviewing topics at constant intervals); 2) An Expansionist approach (prioritizing new topic introduction until a certain threshold is met, then switching to review); and 3) Heuristic-based Spaced Repetition models based on traditional forgetting curve parameters [0, 1].