# Automated Discovery for Memory Bandwidth

## Introduction

The objective of this research was to evaluate achieved GPU memory bandwidth (GB/s) during a STREAM-style triad operation defined by the expression $c = a + \text{scalar} \times b$ using CUDA tensors {ev:ev_0021}. A naive implementation of this elementwise expression involves multiple passes over the data and intermediate allocations, resulting in higher memory traffic (Read B $\rightarrow$ Write T; Read A/T $\rightarrow$ Write C) {ev:ev_0021}. To improve performance, a strategy was implemented to reduce DRAM R/W operations by fusing these steps into a single kernel pass that eliminates the need for intermediate writes and subsequent reads of temporary tensors.

## Method

The optimization approach focused on reducing memory traffic through kernel fusion via `torch.compile` using the default Induitor backend {ev:ev_0021}. By leveraging Triton under the hood, this method transforms multiple operations into a fused single-loop kernel [Read A/B $\rightarrow$ Write C] {ev:ev_0021}. This reduction in memory traffic—by eliminating intermediate writes and subsequent reads of temporary tensors—is designed to increase throughput for these memory-bound operations {ev:ev_0021}. The implementation was benchmarked using large tensor sizes (up to $5.1 \times 10^8$ elements) to evaluate scaling behavior as data movement increases across the tested range {ev:ev_0021}.

## Results

The optimized solution achieved a mean GPU memory bandwidth of 222.60 GB/s across three tested tensor sizes {ev:ev_0021}. Specific performance measurements were recorded as following:
* For $n = 32,000,000$: 216.87 GB/s {ev:ev_0021}
* For $n = 128,000,000$: 222.81 GB/s {ev:ev_0021}
* For $n = 512,000,000$: 228.12 GB/s {ev:ev_0021}

An ablation study was performed to evaluate the impact of memory traffic reduction on total performance {ev:ev_0064}. When the 'Memory traffic reduction' component (the fused kernel) was replaced with a standard eager-mode implementation, the achieved bandwidth dropped from 222.6 GB/S to 130.9 GB/S {ev:ev_0064}, demonstrating that reducing total bytes transferred per element calculation through fusion significantly increases throughput in these memory-bound operations.

## Conclusion

The results demonstrate that by utilizing kernel fusion via `torch.compile`, the achieved GPU memory bandwidth for triad operations can be increased from 130.9 GB/S to a mean of 222.60 GB/s {ev:ev_0064}{ev:ev_0021}. This improvement is driven by reducing total bytes transferred per element calculation through the elimination of intermediate writes and subsequent reads of temporary tensors {ev:ev_0064}.
## References

1. Cheng Liao (2025). DPVO-QAT++: Heterogeneous QAT and CUDA Kernel Fusion for High-Performance Deep Patch Visual Odometry. http://arxiv.org/abs/2511.12653v1
2. Zaid Khan, Justin Chih-Yao Chen, Jaemin Cho, Elias Stengel-Eskin, Mohit Bansal (2026). GPU Forecasters: Language Models as Selective Surrogates for Kernel Runtime Optimization. http://arxiv.org/abs/2605.31464v1
3. Yuankai Fan, Qizhen Weng, Xuelong Li (2025). Computation-Bandwidth-Memory Trade-offs: A Unified Paradigm for AI Infrastructure. http://arxiv.org/abs/2601.11577v1
4. Ilsun Chang (2025). GPU-Augmented OLAP Execution Engine: GPU Offloading. http://arxiv.org/abs/2601.19911v1
5. Hongyu Miao, Myeongjae Jeon, Gennady Pekhimenko, Kathryn S. McKinley, Felix Xiaozhu Lin (2019). StreamBox-HBM: Stream Analytics on High Bandwidth Hybrid Memory. http://arxiv.org/abs/1901.01328v2
6. Mahalakshmi Chidambara Natarajan, Ramaswamy Muthiah, Alamelu Nachiappan (2010). Performance Investigation of Virtual Private Networks with Different Bandwidth Allocations. http://arxiv.org/abs/1002.1152v1
7. Elias Stehle, Hans-Arno Jacobsen (2016). A Memory Bandwidth-Efficient Hybrid Radix Sort on GPUs. http://arxiv.org/abs/1611.01137v2
8. Lingfeng Tang, Daoping Zhang, Junjie Chen, Peihao Huang, Feng Jin, Chengguang Xu, Yuxin Chen, Feiqiang Sun, Guo Chen (2025). MultiPath Memory Access: Breaking Host-GPU Bandwidth Bottlenecks in LLM Services. http://arxiv.org/abs/2512.16056v2
9. Zeke Wang, Hongjing Huang, Jie Zhang, Gustavo Alonso (2020). Benchmarking High Bandwidth Memory on FPGAs. http://arxiv.org/abs/2005.04324v1
10. Ben Parr (2018). Deep In-GPU Experience Replay. http://arxiv.org/abs/1801.03138v1
11. Daniel Schraudner, Andreas Harth (2022). Stream Containers for Resource-oriented RDF Stream Processing. http://arxiv.org/abs/2202.13630v1
12. Isaac Llorente-Saguer (2026). A Lock-Free, Fully GPU-Resident Architecture for the Verification of Goldbach's Conjecture. http://arxiv.org/abs/2603.07850v1
13. Anshu Shukla, Yogesh Simmhan (2016). Benchmarking Distributed Stream Processing Platforms for IoT Applications. http://arxiv.org/abs/1606.07621v2
14. Víctor Gallego (2026). Distilling Feedback into Memory-as-a-Tool. http://arxiv.org/abs/2601.05960v2
15. Yongbin Gu, Wenxuan Wu, Yunfan Li, Lizhong Chen (2020). UVMBench: A Comprehensive Benchmark Suite for Researching Unified Virtual Memory in GPUs. http://arxiv.org/abs/2007.09822v2
