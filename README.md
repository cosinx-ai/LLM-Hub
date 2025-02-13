# LLM-Hub

A curated collection of seminal works, tools, models, and datasets in the realm of Large Language Models (LLMs). This hub highlights only the most significant contributions so you can delve into the essentials without sifting through unnecessary details.



---

## 🌟 Overview

LLM-Hub offers a concise roadmap for anyone looking to grasp and stay up-to-date with major breakthroughs in the LLM field. We’ve distilled the vast literature into a streamlined set of classic resources that showcase important milestones, techniques, and applications.

## 🌳 Contents
- [1. Key Papers](#1-key-papers)
  - [1.1 Transformer Foundations](#11-transformer-foundations)
  - [1.2 Reasoning & Prompting](#12-reasoning--prompting)
  - [1.3 Safety & Alignment](#13-safety--alignment)
- [2. Tools & Models](#2-tools--models)
  - [2.1 Tools & Frameworks](#21-tools--frameworks)
  - [2.2 Model Introductions](#22-model-introductions)
- [3. Post-Training & Alignment](#3-post-training--alignment)
  - [3.1 Common Methods](#31-common-methods)
- [4. Datasets](#4-datasets)
- [5. Evaluation](#5-evaluation)
- [6. Deployment & Inference](#6-deployment--inference)
- [7. Contributing](#7-contributing)
- [8. License](#8-license)

---

## 1. Key Papers

### 1.1 Transformer Foundations
1. **Attention Is All You Need**  
   - **Link**: [PDF](https://arxiv.org/pdf/1706.03762.pdf)  
   - **Contribution**: Introduced the Transformer architecture, removing recurrence in favor of self-attention for greater parallelization.

2. **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding**  
   - **Link**: [PDF](https://arxiv.org/pdf/1810.04805.pdf)  
   - **Contribution**: Popularized masked language modeling, sparking a trend of fine-tuning for various NLP tasks.

3. **GPT: Improving Language Understanding by Generative Pre-Training**  
   - **Link**: [PDF](https://papers.nips.cc/paper_files/paper/2018/file/242c8ec9f1fbd41039ba42dcd37f050b-Paper.pdf)  
   - **Contribution**: Demonstrated large-scale pretraining with next-token prediction, revolutionizing text generation and conversation.

### 1.2 Reasoning & Prompting
1. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models**  
   - **Link**: [PDF](https://arxiv.org/pdf/2201.11903.pdf)  
   - **Contribution**: Showed how step-by-step reasoning in prompts yields clearer and more accurate model outputs.

2. **Self-Consistency Improves Chain-of-Thought Reasoning in Language Models**  
   - **Link**: [PDF](https://arxiv.org/pdf/2203.11171.pdf)  
   - **Contribution**: Proposed sampling multiple reasoning paths, then merging them to reduce errors and increase consistency.

3. **Self-Refine: Iterative Refinement with Self-Feedback**  
   - **Link**: [PDF](https://arxiv.org/pdf/2303.17651.pdf)  
   - **Contribution**: Showed how repeated self-analysis and updates can iteratively improve model responses.

### 1.3 Safety & Alignment
1. **Language Models are Few-Shot Learners**  
   - **Link**: [PDF](https://arxiv.org/pdf/2005.14165.pdf)  
   - **Contribution**: Highlighted the need for control/mechanisms as LLMs scale and gain few-shot prowess.

2. **Learning to Summarize from Human Feedback**  
   - **Link**: [PDF](https://arxiv.org/pdf/2009.01325.pdf)  
   - **Contribution**: Demonstrated that reinforcement learning from human evaluations can guide models toward safer, higher-quality outputs.

3. **Constitutional AI: Harmlessness from AI Feedback**  
   - **Link**: [PDF](https://arxiv.org/pdf/2212.08073.pdf)  
   - **Contribution**: Showcased iterative loops guided by defined constitutions to mitigate unsafe or biased behaviors.

---

## 2. Tools & Models

### 2.1 Tools & Frameworks
1. **Hugging Face Transformers**  
   - URLs: [GitHub](https://github.com/huggingface/transformers) | [Docs](https://huggingface.co/docs/transformers/index)  
   - Widely used for training, evaluating, and deploying Transformer-based models.

2. **DeepSpeed**  
   - URLs: [GitHub](https://github.com/microsoft/DeepSpeed) | [Docs](https://www.deepspeed.ai/)  
   - Provides system optimizations for large-scale distributed training.

3. **Accelerate**  
   - URLs: [GitHub](https://github.com/huggingface/accelerate) | [Docs](https://huggingface.co/docs/accelerate)  
   - Boosts multi-GPU training and inference, lowering overhead for large LLM projects.

4. **vLLM**  
   - *Link*: [GitHub](https://github.com/vllm-project/vllm)  
   - Focuses on efficient large-scale inference with minimal performance bottlenecks.

5. **LlamaFactory**  
   - *Link*: [GitHub](https://github.com/username/llamafactory)  
   - Simplifies fine-tuning for LLaMA-based models, offering data processing scripts and training dashboards.

### 2.2 Model Introductions
1. **DeepSeek**  
   - *Highlights*: Emphasizes post-training alignment for high-context search and QA.  
   - *Use Cases*: Enterprise solutions, domain-specific knowledge retrieval.

2. **Qwen**  
   - *Highlights*: Combines fine-grained RLHF with multi-turn dialogues, improving user satisfaction.  
   - *Use Cases*: Tutoring systems, chat agents.

3. **GPT Variants**  
   - *Highlights*: A family ranging from GPT-1 to GPT-4, typically updated via SFT and RLHF.  
   - *Use Cases*: Multi-purpose text generation, summarization, coding assistance.

---

## 3. Post-Training & Alignment

### 3.1 Common Methods
1. **Supervised Fine-Tuning (SFT)**  
   - *Overview*: Curation-based fine-tuning for domain-specific improvement.  
   - *Reference*: [Fine-Tuning Language Models from Human Preferences (PDF)](https://arxiv.org/pdf/2201.08239)

2. **Reinforcement Learning from Human Feedback (RLHF)**  
   - *Overview*: Leverages human preferences to optimize clarity and helpfulness.  
   - *Reference*: [Learning to Summarize from Human Feedback (PDF)](https://arxiv.org/pdf/2009.01325.pdf)

3. **Direct Preference Optimization (DPO)**  
   - *Overview*: Uses preference datasets to iteratively adjust model behavior.  
   - *Reference*: [DPO Paper (PDF)](https://arxiv.org/pdf/2305.10403.pdf)

4. **Proximal Policy Optimization (PPO)**  
   - *Overview*: A stable RL algorithm enabling balanced exploration/exploitation.  
   - *Reference*: [PPO for Dialogue (PDF)](https://arxiv.org/pdf/1707.06347.pdf)

5. **Alignment & Reward Modeling (RM)**  
   - *Overview*: Designs reward signals to ensure outputs remain aligned with ethical norms.  
   - *Reference*: [Constitutional AI (PDF)](https://arxiv.org/pdf/2212.08073.pdf)

---

## 4. Datasets
1. **OpenWebText**  
   - Drawn from open Reddit submissions, commonly used for initial language modeling.  
   - URL: [GitHub](https://github.com/Skylion007/OpenWebText)

2. **The Pile**  
   - A diverse 800GB text corpus.  
   - URL: [GitHub](https://github.com/EleutherAI/the-pile)

3. **C4 (Colossal Clean Crawled Corpus)**  
   - Over 300GB of cleaned web text, used in a variety of LLM training setups.  
   - URL: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/c4)

---

## 5. Evaluation

1. **Automatic Metrics**  
   - *Examples*: BLEU, ROUGE, METEOR, Perplexity.  
   - *Use Cases*: Quick comparisons for text generation or summarization tasks.

2. **Human Evaluation**  
   - *Overview*: Raters judge model outputs on quality, factual accuracy, style, etc.  
   - *Guidance*: Use carefully designed scoring rubrics and multiple reviewers to minimize subjectivity.

3. **Specialized Benchmarks**  
   - *Examples*: GLUE, SQuAD, MMLU, Big-Bench.  
   - *Focus*: Task-specific or broad-coverage challenge sets to quantify model capabilities.

4. **Adversarial & Robustness Testing**  
   - *Overview*: Stress-tests via adversarial examples and real-world data shifts.  
   - *Importance*: Ensures reliability under varying or unexpected inputs.
5. **Large-Language Model as a judge**  
   - *Overview*: Use LLMs to evaluate the quality of generated text.  
   - *Use Cases*: Automated essay scoring, dialogue evaluation, and more.
---

## 6. Deployment & Inference

1. **Containerization & Orchestration**  
   - *Overview*: Use Docker and Kubernetes to handle large-scale model hosting with GPU acceleration and efficient load balancing.

2. **Model Compression**  
   - *Overview*: Techniques like quantization, pruning, and distillation can reduce model size and inference latency.

---

## 7. Contributing
We welcome pointers to influential papers, tools, or datasets aligned with our minimal, high-impact philosophy. Feel free to open a pull request or issue to help improve this hub.

---

## 8. License
Distributed under the [MIT License](LICENSE). Fork and adapt as needed.
