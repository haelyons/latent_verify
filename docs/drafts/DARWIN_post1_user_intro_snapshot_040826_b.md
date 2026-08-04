> todo
> - rationalise variant names to -chat and -base, including in figures
> - any captions in figures should be MECE with the text - if its duplicated, prefer deleting the caption from the figure

# Characterizing base vs chat model behaviours under pushback in Gemma 2

Language models sometimes change answer and adopt the user’s when challenged. The model begins correct, the user suggests something false, and the model "folds". I tested this and the opposite, where a model starts incorrect and "listens" to a correction, in -base and -chat model variants of the Gemma 2 open-weights model from Google DeepMind. The -chat variant is "tuned" using various techniques to make it more able to act like a helpful assistant and provide good answers, which changes its behaviour. 

> **TL;DR** Gemma 2 -chat answers directly under user pushback whilst -base abstains and hedges, ex. « I don’t know ». -chat corrects itself more when pushed toward truth, and also more consistently is led astray by falsehood than -base. -chat always answers, never hedging or abstaining. Both variants rate the wrong answer more highly after the pushed wrong answer, but -base expresses it less. My hypothesis is that -chat tuning forces answering, which both improves listening and worsens folding. Mechanistic follow-up didn’t find a single circuit carrying that. There’s a correlational, predictive circuit for folding/listening, but it is non-causal. A causal circuit was found for « Yes » and « No » readout tokens.

These results are derived across -base and -chat Gemma 2 at 2, 9, and 27 billion parameters using 82 correct/plausibly incorrect fact pairs. Each model variant/size has one of the pair items already in its own turn, as though it had said it. It is then pushed with the opposite one and forced to provide a final answer. At -base this a raw Q:/A: template, for -chat this is a User:/Assistant: template - so the format co-varies with the model -variant. 

See in the below sankey (Figure 1) the fold and listen runs, across the -base and -chat variants at every scale, where colours correspond to the answer tokens present in the model turn, either planted or freely replied. Green is a correct fact, red is its plausibly incorrect counterpart, and grey means neither.

![[figB_synthesis_strict_ext2.png]]
*Figure 1:* *Listen/fold answer flows across Gemma 2*. This sankey starts fold from the correct fact $C$ and listen from the plausibly incorrect $W*$, each pushed with its counterpart. 

This plot lets us observe that:
1. -base most often "abstains", and when pushed; “I don’t know,” “I’m not sure,” answers, and otherwise names neither answer, even when explicitly asked at the final turn 
2. -chat almost always gives one of the pair answers ($C$ or $W*$ in its response)
3. -chat almost always takes a correct push, correcting itself
4. -chat folds MORE to plausible falsehood than -base 

Fanous et al. report in SycEval that -chat models ChatGPT-4o, Claude-Sonnet and Gemini-1.5-Pro revise toward truth about three times as often as falsehood over their combined math and medical set, which is similar to my results. What I notice by _also_ comparing -base and -chat is that "folding" - revising toward falsehood - increases with -chat tuning, along with "listening" - revising towards truth. 

I think this can partially be explained by -chat tuning forcing expressions of answers that already exist in -base, but resolved to « I don’t know » responses under user pressure. 

![[Pasted image 20260804105134.png]]
*Figure 2: Answer preferences in Gemma 2 output distributions*: [insert a clean definition of margin here. also note that this isn't a good figure - could this be a "margin sankey" or something? and what are the units? there are no units for these numbers, or a clean definition of how to compute margin, given some values. these should be given, ideally as a clean equation, with a step by step worked example directly from one of our cases.]

The push moves every cell toward $W*$. At 9b-base and 27b-base, and only there, $C$ is still ahead afterwards - so the -base advantage is not only refusal to commit, part of it is a preference the push does not overturn. [The margins] sit at the reply to the challenge, not at the final answer Figure 1 scores - only the 9b -chat "fold" arm has both.

For example, -base models don't name the answer (our planted or pushed strings for $C$ and $W*$) unless pushed, which -chat models do at every turn. Looking at the underlying probability distributions we can also see that -base models more consistently prefer the correct answer $C$ after being pushed. I think a good overall description of this behaviour is that « alignment » (-chat) tuning amplifies (installs?) revisability under user pressure. I’d suggest that this also comes at the expense (in Gemma 2) of expressing uncertainty, and is orthogonal to improved truthfulness. 

To decompose how strongly the model already prefers truth with how far user pressure is moving that preference, I followed recommendations from [De Marez et al.](https://arxiv.org/abs/2606.06306) by reading the difference between $C$ and $W*$ log probabilities. Gemma 2 _usually_ [puts $C$ ahead at every cell before the push. [how could this be expressed in prose like the rest of this para?]]. Under the push that margin moves toward the pushed answer whilst $C$ stays ahead on more than half the pairs at 9b and 27b -base, the only two cells where it does [same here]. This is shown in [figure 2]. [The margins] sit at the reply to the challenge, not at the final answer the sankey scores - only the 9b -chat "fold" arm has both.

Alignment tuning amplifies revisability under user pressure, while base models look more resistant - a pattern that [SYCON](https://arxiv.org/abs/2505.23840) and [Gupta et al.](https://arxiv.org/abs/2607.18114) report from the outside. Here, much of that “resistance” is refusal to commit. A flip-rate eval that treats “I don’t know” as robustness will score -base as steadier than -chat. Chat training deletes the grey band from the elicited column; in the reply column it survives at every cell, in replies that name both answers. De Marez et al. see no such reversal - both their channels favour the tuned model, and their 17 of 23 is a worst-case flip rate over their manipulations, not a margin - because their readout has no "abstain" outcome. Gemma is SYCON's own named exception, the narrowest gap they report. That sits awkwardly against the Gemma 2 report’s claim that post-training encouraged hedging to reduce hallucinations ([Gemma Team, 2024](https://arxiv.org/abs/2408.00118)), and more comfortably next to [Zhou et al., 2024](https://arxiv.org/abs/2401.06730), who find “In base models, we see a preference for weakeners but the trend reverses among RLHF models”. [what’s a weakener? what preference?]

The full lab notes go into further detail. This investigation started by trying to paraphrase prompts, freeze attention to make attribution graphs, and adversarially perturb those graphs (like the prompts) to find common circuitry/mechanisms. "Folding" was one of the mechanisms looked at, and I found that at -base, fold and listen share four of their five most influential attention heads [at the answer string?] - not run at -27b - whilst at -chat fold and listen share all five. Ablating these doesn’t move the behaviour, no single lever moves the behaviour: no write handle beats its matched random floor at any scale (at 9b, write-ablating the top heads flips 0 of 37) [what is a write ablation in plain English]. This roughly fits our behavioural evals in the sankey, where -base often holds the planted answer (or withholds) and -chat revises freely in both directions, more so toward truth. **Chat training does not appear to install a dedicated truth circuit.** [the base and -chat head rankings come from unmatched instruments, so the contrast is qualitative] It makes Gemma 2 less "willing" to say it does not know, and more to revise.

[Full lab notes pending write-up - Characterizing base vs chat behaviours under pushback in Gemma 2]

*Compute kindly provided by Apart Research via Lambda.ai. I'm running out though, so if you want to send me more compute or talk to me about my slowly perplexifying CV from all of this AI safety work please reach out, helioslyons.com*
