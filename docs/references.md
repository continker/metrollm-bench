# MetroLLM-Bench — Related Work

## Transit + LLM

1. **TransitTalk** — "Leveraging Large Language Models for Enhancing Public Transit Services." Arxiv 2410.14147, Oct 2024. Three LLM transit apps: Tweet Writer, Trip Advisor, Policy Navigator. Conversational only, no structured output, no benchmark. Closest prior work.

2. **DelayPTC-LLM** — "Metro Passenger Travel Choice Prediction under Train Delays with Large Language Models." Arxiv 2410.00052, Sep 2024. LLMs predict passenger behavior under metro delays via zero-shot/few-shot learning on sparse delay event data.

3. **Luo et al.** — "Exploring the potential of large language models in analyzing passengers' perceptions of transit service quality." SAGE Journals, 2026. GPT-3.5/4o on Shenzhen metro Weibo data. Two-stage analysis: customer experience analyst + transport planner. Identified three hallucination types: overthinking, contextual reasoning error, ambiguity error.

4. **LLM-Agent Transportation Framework** — "Toward LLM-Agent-Based Modeling of Transportation Systems: A Conceptual Framework." Arxiv 2412.06681, Dec 2024. Argues LLM agents replace custom code/logic with natural language prompts for transportation simulation. Conceptual, not benchmarked.

5. **Urban LLM Agents** — "Large Language Model Powered Intelligent Urban Agents: Concepts, Capabilities, and Applications." Arxiv 2507.00914, Jul 2025. Survey covering TrafficGPT, OpenTI, TP-GPT, LLMLight, PlanGPT. Transit is one domain among many.

6. **ChatGPT for GTFS** — Devunuri, Qiam & Lehe. "Benchmarking LLMs on GTFS semantics and retrieval." Public Transport 16(2):333–357, 2024. Tests LLM understanding of General Transit Feed Specification data format.

7. **TransGPT** — Wang et al. "Multi-modal generative pre-trained transformer for transportation." CLNLP 2024.

## Agent Benchmarks

8. **TheAgentCompany** — Xu et al. "Benchmarking LLM Agents on Consequential Real World Tasks." NeurIPS 2025 D&B Track. Arxiv 2412.14161. 175 tasks, 12 LLM backbones, checkpoint-based partial credit, LLM-based NPCs for simulated human interaction. Our scoring and interaction design draws from this.

9. **AgentBench** — Liu et al. "Evaluating LLMs as Agents." Arxiv 2308.03688, 2023 (updated 2025). 8 environments for LLM-as-Agent reasoning and decision-making.

10. **WebArena** — Zhou et al. 2023. Self-hosted realistic web environment. 812 tasks across e-commerce, forums, code, content management. Functional correctness evaluation.

11. **BFCL** — Yan et al. "Berkeley Function-Calling Leaderboard." 2024. 2000 question-answer pairs testing function call accuracy, argument structure, API selection, and appropriate abstention.

12. **ToolEmu** — Ruan et al. "Identifying the Risks of LM Agents with an LM-Emulated Sandbox." 2023. 36 high-stakes tools, 144 test cases. Sandbox approach for risk assessment without real tool infrastructure.

13. **MINT** — Multi-turn interactive evaluation. Measures process navigation, feedback learning, and tool effectiveness — not just final answers.

14. **AppWorld** — Trivedi et al. "A controllable world of apps and people for benchmarking interactive coding agents." ACL 2024. Task Goal Completion metric.

15. **KDD Agent Survey** — "Evaluation and Benchmarking of LLM Agents: A Survey." KDD 2025. Arxiv 2507.21504. Taxonomy: evaluation objectives (behavior, capabilities, reliability, safety) × evaluation process (interaction modes, datasets, metrics, tooling, environments).

## Tool Hallucination + Failure Modes

16. **The Reasoning Trap** — Yin et al. "How Enhancing LLM Reasoning Amplifies Tool Hallucination." Arxiv 2510.22977, Oct 2025. ICLR 2026 submission. Causal relationship: RL-enhanced reasoning increases tool hallucination proportionally with task performance. Introduces SimpleToolHalluBench (no-tool-available and distractor-tool-available failure modes). Tested on Qwen2.5-7B-Instruct with ReCall framework.

17. **Tool-Induced Myopia (TIM)** — "From Proof to Program: Characterizing Tool-Induced Reasoning Hallucinations in Large Language Models." Arxiv 2511.10899, Nov 2025. With tool access, models perform empirical checks instead of reasoning. Tool outputs are correct but reasoning depth diminishes. Final-answer accuracy cannot detect this failure. Evaluated on AIME 2024–2025 problems with Code Interpreter.

18. **Graph Reasoning Limits** — Heyman & Zylberberg, 2025. RLLMs remain imperfect even on 4-vertex graph coloring. Error rates increase dramatically at 8-vertex 4-coloring. Relevant because metro routing is graph traversal over 50–146 vertex graphs.

19. **Hallucination Survey** — Anh-Hoang, Tran & Nguyen. "Survey and analysis of hallucinations in large language models." PMC/Frontiers, 2025. Bayesian attribution framework distinguishing prompt-driven vs model-intrinsic hallucination. Covers factual and logical hallucination types.

20. **Reasoning-Driven Hallucination** — EmergentMind survey, 2025. Codifies subtypes: fabrication, factual inconsistency, logical error. Identifies attention/reasoning drift and chain disloyalty as core mechanisms. Tool hallucination as a distinct failure mode of tool-augmented LLMs.

21. **Generalization-Hallucination Trade-off** — Singh et al. "Are you hallucinated? Insights into large language models." ScienceDirect, 2025. Proposes hallucination is inherent to transformer architecture, not a fixable bug. Advocates external verification rather than elimination.

## Structured Output

22. **StructEval** — Yang et al. "Benchmarking LLMs' Capabilities to Generate Structural Outputs." Arxiv 2505.20139, TMLR Jan 2026. 18 formats, 44 task types. GPT-4o: 76.02%, best FOSS (Qwen3-4B): 67.04%. Visual rendering harder than text-only. Critical finding for our UI state generation evaluation.

23. **JSONSchemaBench** — Geng et al. Arxiv 2501.10868, Jan 2025. Rigorous benchmark for JSON schema conformance. Tests constrained decoding engines (Guidance, llama.cpp, Outlines) against unconstrained LM-only generation. Guidance achieves highest coverage. Constrained decoding adds latency overhead.

24. **SoEval** — Liu et al., 2024. Rule-based JSON/XML validation. Fast but flat schemas miss nested hierarchy errors.

25. **FOFO** — Xia et al., 2024. Structured output evaluation in law/finance domains.

26. **LLM-Structured-Output-Benchmarks** — Leo, Zenodo 2024. Open-source benchmark comparing Instructor, Mirascope, Outlines, LlamaIndex on structured extraction tasks.

27. **SoEval (expanded)** — "Are LLMs good at structured outputs?" ScienceDirect, 2024 (Ning et al. 2025 follow-up). Includes user survey showing demand for diverse format outputs but low satisfaction with current LLM structured output capabilities.

## Small Models for Agents

28. **SLMs for Agentic AI** — Belcak & Heinrich. "Small Language Models are the Future of Agentic AI." NVIDIA Research, 2025. Arxiv 2506.02153. SLMs (<10B params) sufficient for most agentic tasks. 40–70% of LLM calls replaceable by fine-tuned SLMs across MetaGPT, Open Operator, and Cradle. Proposes 6-phase LLM-to-SLM conversion algorithm.

29. **Small Agent Collaboration** — "Can Small Agent Collaboration Beat a Single Big LLM?" Arxiv 2601.11327, Jan 2026. Thinking mode helps instruction-tuned small models for planning but degrades accuracy on tool coordination tasks. Full thinking frequently disrupts tool calling — particularly on computation-heavy and long-horizon tasks.

## Local Inference on Consumer Hardware

30. **Private LLM Inference on Blackwell** — Arxiv 2601.09527, Jan 2026. RTX 5060 Ti, 5070 Ti, 5090 benchmarked across 79 configurations. Qwen3-8B, Gemma3-12B, Gemma3-27B. FP8 quantization preserves >99% accuracy on MMLU/HellaSwag/GSM8k. FP8 KV cache enables 32k–64k context on 16GB GPUs.

31. **Private LLM Server Viability** — Arxiv 2512.23029, Dec 2025. Single RTX 5090 with Qwen3-30B-A3B (Q6_K). Competitive quality for single users. Concurrency degrades steeply beyond 2 users. TTFT faster than cloud baselines under light load. Prefill costs and serialization are primary bottlenecks.

32. **RTX 5090 LLM Benchmarks** — hardware-corner.net, Nov 2025. Qwen3 8B: 10,400 tok/s prefill, Qwen3 30B MoE: ~52 tok/s at 147K context on single 5090. 32GB VRAM sustains extreme context windows.

33. **RTX PRO 6000 Benchmarks** — CloudRift, 2025. PRO 6000 beats H100 on single-GPU workloads at 28% lower cost per token. 96GB GDDR7 eliminates VRAM constraints for most FOSS models.

34. **GPU Comparison** — CloudRift, 2025. RTX 4090 vs 5090 vs PRO 6000 across Qwen3-Coder-30B-A3B, Llama-3.3-70B, GLM-4.5-Air. Model selection > hardware selection as performance optimization lever.

## Qwen3.5 Model Family

35. **Qwen3 Technical Report** — Qwen Team. Arxiv 2505.09388, May 2025. Dense (0.6B–32B) + MoE (30B-A3B, 235B-A22B). Thinking/non-thinking modes. Qwen3-30B-A3B outcompetes QwQ-32B with 10× fewer active parameters. Strong tool calling capabilities.

36. **Qwen3.5 Release** — Qwen, 2026. Hybrid MoE + Gated Delta Networks. 35B-A3B, 27B, 122B-A10B, 397B-A17B + Small (0.8B–9B). 256K context, 201 languages. Vision-language multimodal. Multi-token prediction support.

37. **Unsloth Qwen3.5 GGUFs** — Unsloth, Mar 2026. Dynamic quantization achieving SOTA on KL divergence Pareto frontier. UD-Q4_K_XL and UD-Q3_K_XL within 1 accuracy point of original on 750-prompt mixed suite. Tool calling chat template fixes. 35B-A3B fits in 22GB at Q4.

38. **Qwen3.5-397B Benchmarks** — Benjamin Marie (third party), 2026. UD-Q4_K_XL: 80.5% (−0.8 points vs original, +4.3% relative error increase). UD-Q3_K_XL: 80.7%. Sharp memory reduction with minimal practical loss.

## LLM-Powered Kiosks (Practical)

39. **LogRocket LLM Kiosk** — "LLMs are facing a QA crisis." LogRocket Blog, Aug 2025. Documents real-world LLM-powered public kiosk prototype. Evaluating LLM-driven systems in public-facing contexts requires "a careful blend of quantitative rigor and qualitative judgment." Model provider updates can silently break production prompts.

40. **MTA New TVMs** — MTA Press Release, Dec 2025. New Ticket Vending Machines for LIRR and Metro-North. 9 languages + English, barcode scanning, bill change. Represents current state-of-art in traditional (non-LLM) kiosk design.

41. **LLM-Generated UIs** — fka.dev, "Beyond Text-Only AI: On-Demand UI Generation." 2025. Pattern: LLM generates JSON component specs (forms, buttons, tables, cards), frontend renders. System prompt defines available components and when to use them.

42. **Google A2UI** — Agent-to-User Interface, late 2025. Declarative JSON format for agents to describe UI components. Security-first, client-rendered.

## Scaling Laws + LLM Progress

43. **Karpathy 2025 Year in Review** — karpathy.bearblog.dev, Dec 2025. LLM apps as new computing paradigm. RLVR as major new training stage. Benchmark contamination concerns. "LLM apps orchestrate multiple LLM calls under the hood strung into increasingly more complex DAGs."

44. **Raschka State of LLMs 2025** — magazine.sebastianraschka.com, Dec 2025. MCP as standard for tool/data access. Gated DeltaNet and Mamba layers for efficiency. LoRA and DPO remain foundational for adaptation. GRPO as RLVR method.

45. **MIT Scaling Laws** — Choshen, Andreas & Zhang. "How to build AI scaling laws for efficient LLM training." MIT News, Sep 2025. 485 pre-trained models, 1.9M performance metrics. Power-law relationships between parameters, tokens, compute, and loss. Meta-analysis guiding scaling law selection.

## Transit System Data Sources

46. **BART** — Wikipedia (updated weekly), bart.gov fare calculator, Clipper/Next Gen Clipper docs. 50 stations, 6 routes, distance + surcharge fare model.

47. **CTA L** — Wikipedia (updated daily), transitchicago.com, chicago-l.org. 142 stations, 8 lines, flat fare with O'Hare exception, 24h Red/Blue service.

48. **Taipei MRT** — english.metro.taipei, Wikipedia. 107 stations, distance matrix fares NT$20–65, EasyCard cashback tiers, typhoon suspension protocol (official TRTC FAQ).

49. **MARTA** — itsmarta.com, Wikipedia. 38 stations, 4 lines, flat $2.50 fare, Breeze Card.

## Non-Regular Event Sources

50. **Beijing 国庆节 (National Day)** — Beijing local news (bjd.com.cn, bj.bendibao.com), Sep–Oct 2024. Qianmen station closed entire Golden Week. Tian'anmen East/West closed daily until 19:30. 7 lines extended hours. Enhanced security screening.

51. **Amsterdam Dodenherdenking (May 4)** — GVB official (over.gvb.nl), 4en5mei.nl, Amsterdam municipality. All GVB services (bus, tram, metro, ferry) pause at 20:00 for 2 minutes national silence. Dam Square ceremony diversions. Shops close at 19:00 by law.

52. **Taipei Typhoon Protocol** — TRTC official FAQ, Taiwan News coverage of Typhoon Gaemi (Jul 2024) and Kong-rey (Oct 2024). Wenhu Line (elevated) suspends at gusts ≥ Beaufort 10. Underground sections continue at 12–15 min intervals. Maokong Gondola suspends first. YouBike suspended.

53. **Chicago Polar Vortex** — Historical coverage. Extreme cold causes switch heater failures on elevated outdoor track. Subway sections (Red/Blue underground) continue. Mirrors Taipei elevated-vs-underground split.

54. **BART Earthquake** — BART seismic retrofit documentation, 1989 Loma Prieta precedent. Automatic train hold on seismic detection, inspection period (15–45 min), phased resumption starting with underground segments.

## LLM-as-Judge Methodology

55. **Judging LLM-as-a-Judge** — Zheng et al. "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena." NeurIPS 2023 Datasets and Benchmarks Track. Establishes the LLM-as-judge evaluation framework. Shows strong agreement between GPT-4 judge and human preferences (>80%). Position bias and self-enhancement bias documented. Our Haiku judge follows this methodology for tier 2 scoring.

56. **JudgeBench** — Tan et al. "A Benchmark for Evaluating LLM-Based Judges." Arxiv 2410.12784, 2024. Evaluates judge models across GPT-4o, Claude-3.5-Sonnet, Llama. Relevant for validating our choice of Haiku as judge.

## Small Model Tool Calling + PEFT

57. **SLM Tool Calling** — Jhandi et al. "Small Language Models for Efficient Agentic Tool Calling: Outperforming Large Models with Targeted Fine-tuning." Arxiv 2512.15943, 2025. LoRA fine-tuning on small models outperforms larger models on tool calling. Directly supports our PEFT strategy for 9B kiosk deployment.

58. **Agentic RL Survey** — Zhang et al. "The Landscape of Agentic Reinforcement Learning for LLMs: A Survey." Arxiv 2509.02547, 2025. Covers deterministic tool-driven state transitions and reward signal design for agentic LLM fine-tuning. Our tier 1 scoring as PEFT reward aligns with this framework.

## Foundational Agent Architecture

59. **ReAct** — Yao et al. "ReAct: Synergizing Reasoning and Acting in Language Models." ICLR 2023. Arxiv 2210.03629. The thought-action-observation loop that all tool-calling agents implement. Our runner's architecture is a direct instantiation of ReAct.

## RL Training Methods

60. **GRPO / DeepSeekMath** — Shao et al. "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models." Arxiv 2402.03300, 2024. Introduces Group Relative Policy Optimization, the RL algorithm used in Qwen3's training and applicable to our PEFT roadmap with tier 1 reward signal.

61. **DeepSeek-R1** — DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." Arxiv 2501.12948, Jan 2025. RL-for-reasoning paradigm that Qwen3's thinking mode builds on. Sets context for our thinking/non-thinking ablation.

## Agent Benchmarks (additional)

62. **τ-bench** — Yao et al. "τ-bench: A Benchmark for Tool-Agent-User Interaction in Real-World Domains." Arxiv 2406.12045, 2024. Closest existing benchmark to MetroLLM-Bench — tests tool-calling agents in service domains (airline, retail). We differentiate by transit domain, structured output, multi-system coverage, and tier 1/2 scoring split.

63. **GAIA** — Mialon et al. "GAIA: A Benchmark for General AI Assistants." Arxiv 2311.12983, 2023. 466 multi-step tool-use questions. Small Agent Collaboration [29] tested on GAIA showing tool access > model scale.

## LLM-as-Judge Methodology (additional)

64. **FairEval** — Wang et al. "Large Language Models are not Fair Evaluators." Arxiv 2305.17926, 2023. Documents position bias in LLM judges — response ordering affects outcomes. Relevant caveat for our Haiku judge design.

65. **τ²-bench** — Barres et al. "Evaluating Conversational Agents in a Dual-Control Environment." Arxiv 2506.07982, 2025. Extension of τ-bench [62] adding telecom domain and dual-control evaluation. Direct evolution of our closest competitor benchmark.

## Adversarial / Dynamic Evaluation

66. **AgentDojo** — Debenedetti et al. "AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents." NeurIPS 2024. Arxiv 2406.13352. Extensible framework with 97 realistic tasks and 629 security test cases. Finds LLMs fail many tasks even without attacks. Informs MetroLLM-Bench's Cat H adversarial and Cat J tool-hallucination categories.

67. **AgentHarm** — Andriushchenko et al. "AgentHarm: A Benchmark for Measuring Harmfulness of LLM Agents." ICLR 2025. Arxiv 2410.09024. 110 malicious agent tasks (440 augmented), 11 harm categories. Leading LLMs are "surprisingly compliant with malicious agent requests without jailbreaking." Informs Cat H (adversarial) and scope_adherence rubric design.

68. **RedAgent** — Xu et al. "RedAgent: Red Teaming Large Language Models with Context-Aware Autonomous Language Agent." Arxiv 2407.16667, 2024. Multi-agent LLM red-teamer with memory buffer for context-aware jailbreak generation. Achieves successful probes in ~5 queries; discovered 60 vulnerabilities in real GPT applications. Informs the Cat H adversarial case design.

69. **MAGIC** — Wen et al. "MAGIC: A Co-Evolving Attacker-Defender Adversarial Game for Robust LLM Safety." Arxiv 2602.01539, 2026. RL-based co-evolution of attacker and defender agents. Attacker generates "previously unseen combinatorial strategies." Maps onto MetroLLM-Bench PEFT loop: harder cases as models improve, new failure modes expand the case set.

70. **BenchSelf-Evolving** — Wang et al. "Benchmark Self-Evolving: A Multi-Agent Framework for Dynamic LLM Evaluation." COLING 2025. Arxiv 2402.11443. Six reframing operations (context manipulation, noise injection, sub-ability probing) to dynamically extend benchmarks. Models decline on evolved instances. Motivates AI-agent-assisted case augmentation beyond the static 910-case set.

71. **Live API-Bench** — Elder et al. "Live API-Bench: 2500+ Live APIs for Testing Multi-Step Tool Calling." Arxiv 2506.11266, 2025. 11 databases, 2,500+ invocable tools. LLM completion rates 7–47% across 10 models, ~50% in interactive agent setting. Confirms multi-step tool calling as differentiating capability; benchmarks depth vs. breadth trade-off.

72. **RRTL** — Liu et al. "RRTL: Red Teaming Reasoning Large Language Models in Tool Learning." Arxiv 2505.17106, 2025. Red teaming RLLMs (DeepSeek-R1 etc.) in tool use. RLLMs fail to disclose tool usage/risks; CoT prompting reveals multi-lingual vulnerabilities. Directly relevant to thinking-mode ablation and Cat E (cultural/multilingual) evaluation.