Adversarial Attacks on Graph Attention Networks (GATs): Edge-Perturbation Attacks for Evading GAT-Based Financial Anomaly Detection

---

Title Page

Adversarial Attacks on Graph Attention Networks (GATs): Exploring Edge-Perturbation Attacks for Evading GAT-Based Financial Anomaly Detection Systems

Author: Daksh Verma

Institution: National Institute of Technology Delhi

Department: Department of Computer Science and Engineering

Date: March 2025

Keywords: Graph Attention Networks, Adversarial Attacks, Edge Perturbation, Financial Anomaly Detection, Graph Neural Networks, Evasion Attacks, Fraud Detection, Node Embeddings

---

Abstract

Graph Attention Networks (GATs) have emerged as a powerful architecture for financial anomaly detection, leveraging attention mechanisms to learn node embeddings that capture complex transactional relationships. However, the adversarial robustness of GAT-based fraud detection systems remains critically underexplored. This paper investigates how adversaries can evade GAT-based financial anomaly detection systems through edge-perturbation attacks—subtly adding or removing transactional links to manipulate node embeddings and cause misclassification of fraudulent accounts as benign. We conduct a comprehensive analysis of attack methodologies including gradient-based approaches (Nettack, MetaAttack) and explainability-guided edge perturbation strategies, examining their effectiveness against GAT architectures. Through a systematic literature synthesis and formal problem formulation, we demonstrate that GATs exhibit structural vulnerabilities to edge perturbations, particularly when adversaries target inter-class edges. We propose a unified evaluation framework for assessing GAT robustness in financial contexts, and analyze defense mechanisms including adversarial training, graph purification, and attention-based anomaly detection. Our analysis reveals that while GATs' attention mechanism provides some resilience through dynamic neighbor weighting, the architecture remains susceptible to carefully crafted perturbations. We conclude by identifying critical research gaps and proposing directions for developing robust GAT-based financial anomaly detection systems. This work provides a foundational framework for understanding and mitigating adversarial threats in graph-based financial security applications.

Keywords: Graph Attention Networks, Adversarial Attacks, Edge Perturbation, Financial Anomaly Detection, Graph Neural Networks, Evasion Attacks, Fraud Detection, Node Embeddings

---

Table of Contents

1. Introduction
2. Background and Fundamentals
3. Literature Review
4. Problem Formulation
5. Proposed Methodology and Framework
6. Experimental Design
7. Expected Results and Evaluation Framework
8. Discussion
9. Applications and Real-World Implications
10. Limitations
11. Future Work
12. Conclusion
13. References
14. Appendices

---

1. Introduction

1.1 Background and Context

The digital transformation of financial services has generated unprecedented volumes of transactional data, creating both opportunities and challenges for fraud detection. Traditional machine learning approaches, which treat transactions as independent samples, fail to capture the inherently relational nature of financial fraud, where fraudulent activities often manifest as coordinated networks of transactions involving multiple accounts, shared identifiers, and structured money flows. This limitation has driven the adoption of graph-based learning approaches, particularly Graph Neural Networks (GNNs), which model financial entities as nodes and transactions as edges, enabling the exploitation of relational structure for anomaly detection.

Among GNN architectures, Graph Attention Networks (GATs) have demonstrated particular promise for financial fraud detection. GATs employ self-attention mechanisms to dynamically weight the importance of neighboring nodes during message passing, allowing the model to focus on the most relevant transactional relationships for each entity. This attention-based aggregation enables GATs to learn expressive node embeddings that capture both local transaction patterns and broader network context, achieving strong performance in detecting fraudulent accounts and transactions.

1.2 Motivation

Despite their effectiveness, the deployment of GATs in financial security applications introduces a critical vulnerability: adversarial manipulation of the transaction graph. In financial systems, adversaries—including fraudsters seeking to evade detection—can potentially manipulate the graph structure by adding or removing transactional links. Unlike traditional feature-space attacks that require modifying node attributes (which may be subject to validation constraints in financial systems), edge perturbations can be executed through legitimate-looking transactions, making them particularly insidious and difficult to detect.

The implications are profound: an adversary could strategically add transactions to a suspicious account's neighborhood, altering its learned embedding to appear more similar to legitimate accounts, thereby evading detection. Conversely, removing edges could isolate fraudulent nodes or disrupt the propagation of anomaly signals through the graph. The attention mechanism in GATs, while providing expressiveness, may also introduce new attack surfaces, as adversaries can potentially manipulate the attention weights through structural changes.

1.3 Problem Statement

This research addresses the following core problem: How vulnerable are GAT-based financial anomaly detection systems to edge-perturbation attacks, and through what mechanisms do adversarial modifications of transaction graph structure cause evasion?

This problem encompasses several sub-questions:

1. What types of edge-perturbation attacks are effective against GAT architectures in financial contexts?
2. How do GAT-specific characteristics (attention mechanisms, multi-head configurations) influence vulnerability?
3. What is the relationship between perturbation budget and evasion success?
4. Which defense strategies can effectively mitigate edge-perturbation attacks on GAT-based fraud detection?

1.4 Why the Problem Matters

The consequences of adversarial evasion in financial anomaly detection are severe. Fraudulent transactions that evade detection result in direct financial losses, undermine the integrity of financial systems, and erode trust in automated decision-making. Moreover, as financial institutions increasingly deploy GNN-based systems for real-time fraud detection, the attack surface expands, creating opportunities for adversaries to exploit architectural vulnerabilities.

Regulatory frameworks increasingly require financial institutions to demonstrate the robustness and reliability of their automated systems, making it essential to understand and address adversarial vulnerabilities before deployment. The research gap is particularly acute for GATs, as most existing adversarial robustness research focuses on simpler architectures like Graph Convolutional Networks (GCNs) and GraphSAGE.

1.5 Current Landscape and Research Gap

The existing literature on adversarial attacks against GNNs has primarily focused on node classification tasks in citation networks and social networks. Foundational works such as Nettack and MetaAttack have demonstrated that GNNs are vulnerable to small structural perturbations, with accuracy drops exceeding 20% under modest attack budgets. However, these studies predominantly target GCNs and do not address the unique characteristics of attention-based architectures.

For financial fraud detection specifically, several recent works have begun to explore adversarial robustness. The "Feature-Space Illusion" paper demonstrated that GNN-based blockchain fraud detection models exhibit varying robustness, with GraphSAGE showing inherently more stable decision boundaries than attention-based approaches. However, this work focused on feature-space attacks rather than structural perturbations. The multigraph GNN robustness framework introduced by Atasu reformulates message passing over incidence matrices to handle parallel transactions, but does not specifically address GAT architectures.

The gap in the literature can be summarized as follows: no systematic study examines edge-perturbation attacks against GAT-based financial anomaly detection, despite GATs' growing adoption in this domain and their theoretically distinct vulnerability profile due to attention-based aggregation. This paper addresses this gap through a comprehensive analysis of attack methodologies, formal problem formulation, and a proposed evaluation framework.

1.6 Research Questions and Objectives

Primary Research Question: How can adversaries execute edge-perturbation attacks to evade GAT-based financial anomaly detection systems?

Objectives:

1. To characterize the vulnerability of GAT architectures to edge-perturbation attacks in financial transaction graphs
2. To analyze how attention mechanisms influence attack effectiveness and transferability
3. To propose a systematic evaluation framework for assessing GAT robustness in financial contexts
4. To identify and evaluate defense strategies against edge-perturbation attacks
5. To provide actionable recommendations for deploying robust GAT-based fraud detection systems

1.7 Scope

This paper focuses on edge-perturbation attacks (edge insertion and deletion) against GAT-based node classification models in financial anomaly detection contexts. We consider both evasion attacks (manipulation at test time) and poisoning attacks (manipulation during training), with primary emphasis on evasion due to its practical relevance in financial systems where models are retrained periodically. The scope encompasses structural perturbations only; feature-space attacks and node injection attacks are discussed where relevant but do not form the primary focus.

1.8 Contributions

This paper makes the following contributions:

1. Systematic vulnerability analysis: We provide the first comprehensive analysis of edge-perturbation attack effectiveness against GAT-based financial anomaly detection, synthesizing evidence from adversarial graph learning and financial fraud detection literature.
2. Formal problem formulation: We formalize the edge-perturbation attack problem against GATs, including the bi-level optimization framework, perturbation budget constraints, and unnoticeability requirements specific to financial transaction graphs.
3. Evaluation framework: We propose a unified framework for evaluating GAT robustness, incorporating metrics for evasion success, structural plausibility, and computational cost.
4. Defense analysis: We critically evaluate existing defense mechanisms (adversarial training, graph purification, robust aggregation) in the context of GAT-based financial fraud detection.
5. Research agenda: We identify critical gaps and propose specific directions for future research on robust GAT architectures for financial security.

1.9 Organization

The remainder of this paper is organized as follows. Section 2 provides background on GATs, adversarial attacks on graphs, and financial anomaly detection. Section 3 reviews related literature critically. Section 4 presents the formal problem formulation. Section 5 describes the proposed evaluation framework and methodology. Section 6 details the experimental design. Section 7 presents the expected results framework. Section 8 discusses implications and limitations. Section 9 explores applications. Section 10 addresses limitations. Section 11 proposes future work. Section 12 concludes.

---

2. Background and Fundamentals

2.1 Graph Neural Networks

Graph Neural Networks (GNNs) are deep learning architectures designed to operate on graph-structured data through iterative message-passing mechanisms. A graph $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ consists of a set of nodes $\mathcal{V} = \{v_1, ..., v_n\}$ and edges $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$, with node features $\mathbf{X} \in \mathbb{R}^{n \times d}$ and optionally edge features $\mathbf{E} \in \mathbb{R}^{m \times d_e}$. The adjacency matrix $\mathbf{A} \in \{0,1\}^{n \times n}$ encodes the graph structure, where $\mathbf{A}_{ij} = 1$ if $(v_i, v_j) \in \mathcal{E}$.

The general GNN message-passing framework can be expressed as:

\mathbf{h}_v^{(l+1)} = \sigma\left(\mathbf{W}_{\text{self}}^{(l)} \mathbf{h}_v^{(l)} + \mathbf{W}_{\text{neigh}}^{(l)} \bigoplus_{u \in \mathcal{N}(v)} \mathbf{h}_u^{(l)}\right)

where $\mathbf{h}_v^{(l)}$ is the embedding of node $v$ at layer $l$, $\mathcal{N}(v)$ denotes the neighborhood of $v$, $\bigoplus$ is a permutation-invariant aggregation function (sum, mean, max), $\mathbf{W}_{\text{self}}$ and $\mathbf{W}_{\text{neigh}}$ are learnable weight matrices, and $\sigma(\cdot)$ is a non-linear activation function.

2.2 Graph Attention Networks

Graph Attention Networks (GATs), introduced by Veličković et al., extend the GNN framework by replacing fixed aggregation weights with learnable attention coefficients. For each node $v$ and its neighbor $u \in \mathcal{N}(v)$, GAT computes an attention score:

e_{vu} = \text{LeakyReLU}\left(\mathbf{a}^T [\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_u]\right)

where $\mathbf{a} \in \mathbb{R}^{2d'}$ is a learnable attention vector, $\mathbf{W} \in \mathbb{R}^{d' \times d}$ is a shared linear transformation, and $\|$ denotes concatenation. These scores are normalized using softmax:

\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{k \in \mathcal{N}(v)} \exp(e_{vk})}

The node embedding is then computed as a weighted sum of transformed neighbor features:

\mathbf{h}_v' = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} \mathbf{W}\mathbf{h}_u\right)

GATs typically employ multi-head attention, where $K$ independent attention mechanisms compute separate embeddings that are concatenated (for intermediate layers) or averaged (for the final layer):

\mathbf{h}_v' = \big\|_{k=1}^{K} \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu}^k \mathbf{W}^k \mathbf{h}_u\right)

The attention mechanism provides three key properties relevant to adversarial robustness: (1) dynamic weighting allows the model to downweight uninformative or noisy neighbors, (2) interpretability through attention coefficients enables analysis of which relationships drive predictions, and (3) inductive capability allows generalization to unseen nodes, critical for financial systems with continuous account creation.

2.3 Financial Anomaly Detection as Node Classification

In financial anomaly detection, the task is typically formulated as node classification on a transaction graph. Nodes represent financial entities (accounts, transactions, customers), and edges represent transactions or relationships between them. Node features $\mathbf{x}_v$ may include:

· Transaction statistics (frequency, volume, temporal patterns)
· Account attributes (age, type, KYC status)
· Behavioral features (deviation from historical patterns)
· Network features (degree, clustering coefficient, PageRank)

The classification task is binary: $y_v \in \{0, 1\}$ where $y_v = 1$ indicates a fraudulent/anomalous entity. The GAT model $f_\theta: \mathcal{G} \to \mathcal{Y}$ is trained to minimize classification loss $\mathcal{L}_{\text{train}}(f_\theta(\mathcal{G}), \mathbf{Y}_{\text{train}})$ on labeled training nodes.

The challenge is exacerbated by class imbalance (fraud is rare), temporal dynamics (fraud patterns evolve), and the need for real-time detection in high-volume systems.

2.4 Adversarial Attacks on Graphs

Adversarial attacks on graphs can be taxonomized along several dimensions:

By timing:

· Evasion attacks: Perturbations applied at test time, after model training
· Poisoning attacks: Perturbations applied during training, affecting the learned model parameters

By target:

· Targeted attacks: Cause misclassification of specific nodes
· Untargeted attacks: Degrade overall model performance

By perturbation type:

· Feature perturbations: Modify node attributes
· Structural perturbations: Add/remove edges (edge perturbation)
· Node injection: Add new nodes with crafted features/connections

By knowledge:

· White-box: Full access to model architecture and parameters
· Black-box: Only query access to model outputs
· Gray-box: Partial knowledge (e.g., surrogate model)

Edge-perturbation attacks are particularly relevant for financial systems because: (1) transactional links can be created through legitimate-appearing transactions, (2) structural changes propagate through message passing, affecting multiple nodes, and (3) financial graphs are inherently dynamic, making perturbations less conspicuous.

2.5 Nettack and MetaAttack: Foundational Attack Methods

Nettack (Zügner et al., 2018) was the first adversarial attack specifically designed for attributed graphs. It operates in a gray-box setting using a surrogate GCN model and computes perturbations through greedy selection guided by the linearized loss. For a target node $v$, Nettack solves:

\max_{\hat{\mathbf{A}}} \quad \mathcal{L}_{\text{atk}}(f_{\theta^*}(\hat{\mathbf{A}}, \mathbf{X})_v, y_v)


\text{s.t.} \quad \|\hat{\mathbf{A}} - \mathbf{A}\|_0 \leq \Delta, \quad \text{degree distribution preserved}

Nettack ensures unnoticeability by preserving the degree distribution and co-occurrence patterns of the original graph.

MetaAttack (Zügner & Günnemann, 2019) extends this to poisoning attacks via meta-learning. It treats the graph structure as a hyperparameter and computes meta-gradients to solve the bi-level optimization:

\min_{\hat{\mathbf{A}}} \quad \mathcal{L}_{\text{atk}}(f_{\theta^*}(\hat{\mathbf{A}}), \mathbf{Y}_{\text{val}})


\text{s.t.} \quad \theta^* = \arg\min_\theta \mathcal{L}_{\text{train}}(f_\theta(\hat{\mathbf{A}}), \mathbf{Y}_{\text{train}})

MetaAttack preserves the graph's scale-free properties and operates under strict perturbation budget constraints, making perturbations statistically indistinguishable from legitimate structural variations.

2.6 Explainability-Based Attack Strategies

Recent work by Chanda et al. introduced explainability-based edge perturbation, where GNN explainability methods (e.g., GNNExplainer, PGExplainer) identify important nodes and edges for a target prediction, and perturbations are strategically placed to maximize impact. The key insight is that attacking the most influential subgraph is more effective than random perturbation. Specifically, introducing edges between nodes of different classes has a higher impact than removing edges within the same class.

---

3. Literature Review

3.1 Adversarial Attacks on GNNs: Foundations

The study of adversarial attacks on GNNs began with the seminal work of Zügner et al. (2018), who introduced Nettack and demonstrated that GCNs are highly vulnerable to small structural perturbations. Their experiments on citation networks showed accuracy drops of up to 30% with only 5 edge perturbations per target node. A key finding was that attacks transfer across model architectures, suggesting fundamental vulnerabilities in message-passing GNNs.

Dai et al. (2018) extended this to reinforcement learning-based attacks, while Zügner and Günnemann (2019) introduced MetaAttack, the first poisoning attack using meta-gradients. These foundational works established that: (1) GNNs are more vulnerable to structural than feature perturbations, (2) poisoning attacks are more damaging than evasion attacks, and (3) unnoticeability constraints can be satisfied while maintaining attack effectiveness.

Critical assessment: While these works established fundamental vulnerabilities, they focused on homogeneous graphs with relatively simple architectures (2-layer GCNs). Their findings may not directly transfer to GATs, where attention mechanisms provide additional capacity to downweight adversarial neighbors. This re
