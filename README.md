# Stat 360 -- Introduction to Generative AI Winter 2025
Stat 360 course materials

# Office Hours

| Project title                  | Time | Location              
|--------------------------------|---------------|-------------------------|
| Ryan Chen (TA) | Thursdays 3-4pm    | Zoom: https://northwestern.zoom.us/j/3082623966  | 
|  Professor Stadie        |  Posted on Canvas  |  Zoom: https://northwestern.zoom.us/j/3799597115 |

# Course Lectures 

Lecture notes can be found on the course Canvas website. 


| Lecture                  |  Date | Material | Readings                
|--------------------------|-------|----------|----------------------------|
| Week 1, Tuesday        | January 7 |   Introduction  | [Perplexity](https://thegradient.pub/understanding-evaluation-metrics-for-language-models/), [Perplexity 2](https://web.stanford.edu/~jurafsky/slp3/3.pdf), [Linear Models](https://see.stanford.edu/materials/aimlcs229/cs229-notes1.pdf)  |
| Week 1, Thursday         | January 9  | Attention |  [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf), [Attention Mechanisms](https://lilianweng.github.io/posts/2018-06-24-attention/), [Attention with code](https://sebastianraschka.com/blog/2023/self-attention-from-scratch.html), [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) |
| Week 2. Tuesday       | January 14 | Transformers |  [Intro to Transformers](https://arxiv.org/pdf/2304.10557.pdf), [Discussion](https://www.columbia.edu/~jsl2239/transformers.html), [Blog post](https://peterbloem.nl/blog/transformers), [Skip connections](https://theaisummer.com/skip-connections/), [Layer normalization](https://www.kaggle.com/code/halflingwizard/how-does-layer-normalization-work), [Byte-Pair Encoding](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt) |
| Week 2. Thursday       | January 16 | Coding Transformers |   |
| Week 3, Tuesday       | January 21| BERT, GPT, LLAMA | [Annotated GPT 2](https://jalammar.github.io/illustrated-gpt2/),  [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf), [Adversarial attacks on GPT-2](https://arxiv.org/abs/2012.07805) [LLAMA Paper](https://scontent-ord5-1.xx.fbcdn.net/v/t39.8562-6/333078981_693988129081760_4712707815225756708_n.pdf?_nc_cat=108&ccb=1-7&_nc_sid=e280be&_nc_ohc=it_GnOgZ1hMAX_qDhzS&_nc_ht=scontent-ord5-1.xx&oh=00_AfCZyg0NnnD2SfBipL7DBQ467rntvBHugEZo7maieJZNTQ&oe=65ACEFE2), [BERT](https://arxiv.org/pdf/1810.04805.pdf), [Understanding BERT](https://jalammar.github.io/illustrated-bert/)|
| Week 3, Thursday         | January 23| Prompt Tuning, chain of thought, hindsight chain of thought, backwards chain of thought, Graph of Thought, Tree of Thought, Training Chain-of-Thought via Latent-Variable Inference, prompt engineering | [Language Models are Few Shot Learners](https://arxiv.org/abs/2005.14165), [Zero shot chain of thought](https://arxiv.org/abs/2205.11916), [LLMs are human-level prompt engineers](https://arxiv.org/abs/2211.01910), [Tree of Thought](https://arxiv.org/abs/2305.10601), [Chain of verification](https://arxiv.org/abs/2309.11495), [Promptbreeder](https://arxiv.org/abs/2309.16797), [Prompt Engineering Guide](https://github.com/dair-ai/Prompt-Engineering-Guide?tab=readme-ov-file), [Open-AI Prompting Guide](https://platform.openai.com/docs/guides/prompt-engineering/six-strategies-for-getting-better-results)  |
| Week 4, Tuesday       | January 28| Fine tuning, tool use, parameter-efficient fine tuning, LORA, Instruction Tuning (SFT), Neftune, quantization, Hugging face, fine tuning LLAMA  | [LORA](https://arxiv.org/abs/2106.09685), [PEFT](https://huggingface.co/blog/peft), [Quantization](https://arxiv.org/abs/2305.14314), [Quantization Blog](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/),  [Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html), [ToolEMU](https://toolemu.com/), [NEFTune](https://arxiv.org/abs/2310.05914), [Tool Use code](https://python.langchain.com/docs/modules/agents/how_to/intermediate_steps), [Model Calibration](https://arxiv.org/abs/2012.15723) [Mistral Fine Tuning](https://github.com/brevdev/notebooks/blob/main/mistral-finetune-own-data.ipynb), [LLAMA fine tuning](https://github.com/facebookresearch/llama-recipes/blob/main/examples/quickstart.ipynb)|
| Week 4, Thursday    | January 30| ChatGPT and RLHF, rejection sampling, DPO, Gopher Cite   |[Stack LLAMA](https://huggingface.co/blog/stackllama), [Instruct GPT](https://arxiv.org/pdf/2203.02155.pdf), [PPO](https://arxiv.org/abs/1707.06347), [DPO](https://arxiv.org/abs/2305.18290), [RLHF References](https://github.com/opendilab/awesome-RLHF), [TRL](https://github.com/huggingface/trl), [Gopher Cite](https://arxiv.org/abs/2203.11147), [Chain of hindsight](https://arxiv.org/abs/2302.02676) |
| Week 5, Tuesday         | February 4| No class | |
| Week 5, Thursday     | February 6|  RAG, when to use RAG vs SFT, lexacagraphical vs semantic search, sentence transformers, Retrieval transformers and long term memory in transformers, RAG Code. | [RAG](https://python.langchain.com/docs/concepts/rag/), [RAG code](https://github.com/neuml/txtai), [Sentence Transformers](https://arxiv.org/abs/1908.10084), [Rag Evaluation](https://github.com/stanford-futuredata/ARES), [Advanced RAG](https://github.com/NisaarAgharia/Advanced_RAG) |
| Week 6, Tuesday         | February 11|  Multi-Agent LLMs  | [Wonderful Team](https://wonderful-team-robotics.github.io/), [ReWoo](https://arxiv.org/abs/2305.18323), [Reasoning with Language Model](https://arxiv.org/abs/2305.14992), [LLM+P](https://arxiv.org/abs/2304.11477), [ReACT](https://arxiv.org/abs/2210.03629), [AgentVerse](https://arxiv.org/abs/2308.10848)  |
| Week 6, Thursday    | February 13| Prompt optimization, reflection, steering   | |
| Week 7, Tuesday   | February 18| GPTo1, LLMS + Tabular ML | |
| Week 7, Thursday  | February 20 | Vision Models, Asking questions about images. Conditional layer norm, FILM, CLIP, BLIP, LAVA | [BLIP](https://arxiv.org/abs/2301.12597), [CLIP](https://openai.com/research/clip), [Llava](https://llava-vl.github.io/), [FiLM](https://arxiv.org/pdf/1709.07871.pdf) |
| Week 8 Tuesday|  February 25| Stable Diffusion, tuning stable diffusion, Diffusion models, DDPM, classifier-free guidance  | [Stable Diffusion](https://en.wikipedia.org/wiki/Stable_Diffusion), [Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/), [Understanding Stable Diffusion](https://scholar.harvard.edu/binxuw/classes/machine-learning-scratch/materials/stable-diffusion-scratch) [Diffusion](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/), [DDPM](https://arxiv.org/abs/2006.11239), [Diffusion as SDEs](https://arxiv.org/abs/2011.13456), [Classifier Free Guidance](https://sander.ai/2022/05/26/guidance.html), [Diffusion code](https://github.com/huggingface/diffusion-models-class/blob/main/unit1/01_introduction_to_diffusers.ipynb), [More low level code](https://github.com/acids-ircam/diffusion_models/blob/main/diffusion_03_waveform.ipynb) |
| Week 8 Thursday|  February 27| Frontiers, using LLMs to help diffusion models by planning out images. Instance recognition and inserting new objects %s tricks. Consistency models, SD Edit,  Diffusion in robotics.           | | 
| Week 9, Tuesday  |  March 4| Life is Worse with LLMs |  |
| Week 9, Thursday   |  March 6| No class. Complete take-home final exam   |  |


# Homeworks and Due Dates


| Project title                  | Date released | Due date                
|--------------------------------|---------------|-------------------------|
|   Assignment 1: Transformers      | Jan 9   | Jan 21  |
|     Assignment 2: Prompt Tuning      |  Jan 22   |Jan 30  |
|     Assignment 3: SFT      |  Jan 31   |Feb 11  |
|     Assignment 4: RLHF      |  Feb 12   |Feb 21  |
|     Assignment 5: RAG      |  Feb 21   |March 4  |
|  Take-home final exam      |    March 5   | Due Friday, March 7th  |
