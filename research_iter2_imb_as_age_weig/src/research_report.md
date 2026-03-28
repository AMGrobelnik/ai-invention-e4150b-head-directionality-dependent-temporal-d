# IMB as Age-Weighted DLT Generalization + SUD vs UD

## Summary

This research survey maps the lineage from Gibson's DLT storage cost to IMB and documents how SUD vs UD treebank format affects intervener complexity. Key findings: (1) Gibson 1998 SPLT defined storage cost as count of predicted syntactic heads WITH locality, but Gibson 2000 DLT revised this to be INDEPENDENT of time. IMB reintroduces the age-weighting Gibson removed. (2) Chen et al. 2005 validated that binary counting of pending predictions increases reading times but found storage duration had no significant effect — the gap IMB fills via ACT-R-grounded age-weighting. (3) Kobele/Gerth 2013 introduced stack tenure; Graf et al. 2017 formalized tenure/payload/size as separate dimensions. IMB uniquely combines tenure and payload into a single age-weighted sum computed from dependency trees without requiring Minimalist Grammar parsers. (4) SUD reverses headedness for aux, cop, case, mark relative to UD, making function words heads. Yadav et al. 2022 used SUD for 54 languages; Phase 3 uses UD, so IC values will systematically differ, requiring a pilot UD-SUD comparison. (5) Formally: DLT_storage(j) = |{k open at j}| while IMB(j) = sum of ages for k open at j. Three novelty claims: age-weighting reintroduction via ACT-R decay, max(IMB) as temporal overlap signal invisible to DD moments, and parametric convexity framework. A comparison table across six metrics (Gibson 1998/2000, Chen 2005, Kobele tenure/payload, Graf, IMB) and SUD-UD relation reversal details are provided.

## Research Findings

This survey establishes how IMB (Incremental Memory Burden) relates to Gibson's DLT storage cost, Chen et al. (2005) per-position validation, Kobele/Gerth/Graf stack-based metrics, and how SUD vs UD headedness differences affect intervener complexity computation.

## TOPIC 1: Gibson 1998 SPLT and Gibson 2000 DLT — Storage Cost Definition

Gibson (1998) proposed the Syntactic Prediction Locality Theory (SPLT), where memory cost is quantified as "the number of syntactic categories that are necessary to complete the current input string as a grammatical sentence" [1]. Each predicted syntactic head counts as one Memory Unit (MU). Critically, the 1998 SPLT version included locality in BOTH storage and integration cost: "the longer a predicted category must be kept in memory before the prediction is satisfied, the greater is the cost for maintaining that prediction" [1]. This means SPLT storage cost was age-weighted — older predictions cost more.

Gibson (2000) revised this in the Dependency Locality Theory (DLT). The key change: storage cost was made INDEPENDENT of time/distance. As confirmed by secondary sources: "Storage cost is argued to depend on the number of syntactic heads required to complete the current input as a grammatical sentence (Gibson, 2000) and seems to be independent of the amount of time that an incomplete dependency is held in memory" [2, 3]. Gibson moved all locality effects into integration cost, which is computed as the number of new discourse referents intervening between a dependent and its head [4, 5].

This pivot is critical for positioning IMB: Gibson 1998 SPLT had age-weighting in storage (like IMB), but Gibson 2000 DLT explicitly REMOVED it. IMB reintroduces age-weighting with a formal framework.

## TOPIC 2: Chen, Gibson & Wolf (2005) Experimental Validation

Chen, Gibson & Wolf (2005) conducted three self-paced word-by-word reading experiments testing online syntactic storage costs in English [6].

Experiment 1 manipulated the number of predicted verbs (0, 1, or 2 pending) at a critical reading region. Result: "people read the condition in which zero verbs were predicted fastest, followed by the conditions in which one verb was predicted, with the condition in which two verbs were predicted slowest" [6].

Experiments 2 and 3 tested whether incomplete filler-gap dependencies (wh-fillers pending resolution) incur storage cost. Result: "people read the critical region in which a wh-filler is pending slower than if no such wh-filler is pending" [6].

Critically for IMB positioning: "the amount of time that the dependencies were stored did not significantly affect the measured reading times" [6]. This means Chen et al. validated BINARY COUNTING of pending predictions but found NO effect of age/duration. This is the key gap IMB fills — IMB proposes that age-weighting DOES matter, supported by ACT-R activation decay theory rather than the simple counting Chen et al. tested.

## TOPIC 3: Kobele, Gerth & Hale (2013) and Graf et al. (2017) Stack-Based Metrics

Kobele, Gerth & Hale (2013) examined "transient stack states of a top-down parser for Minimalist Grammars" and found that "the number of time steps that a derivation tree node persists on the parser's stack derives the observed contrasts" in center-embedding across English, Dutch, and German [7]. They called this measure "stack tenure."

Graf, Monette & Zhang (2017) formalized a three-way distinction of memory usage metrics [8, 9]:

- TENURE: "how long a node is kept in memory" — defined as the difference between a node's outdex and index (outdex − index) [8].
- PAYLOAD: "how many nodes must be kept in memory" — the count of nodes with tenure strictly greater than 2 [8].
- SIZE: "how many bits a node consumes in memory" — the number of movement arrows passing through a node [8].

Graf et al. generated 1,600 complexity metrics parameterized along these dimensions and found that only combinations of tenure and size could predict the full range of relative clause processing contrasts across English, Chinese, Korean, and Japanese [8].

The critical insight for IMB positioning: IMB at position j is essentially the SUM OF TENURES of all nodes currently active at position j. It combines BOTH tenure (age-weighting) AND payload (concurrent count) into a single measure. Neither Kobele/Gerth nor Graf et al. proposed this combined age-weighted-sum metric — they kept tenure and payload as separate dimensions [7, 8, 9].

## TOPIC 4: SUD vs UD Headedness Differences and IC Impact

SUD (Surface-Syntactic Universal Dependencies) was proposed by Gerdes, Guillaume, Kahane & Perrier (2018) as an annotation scheme "near-isomorphic to UD" but based on syntactic criteria favoring functional heads [10, 11].

The key reversals between SUD and UD [11, 12]:
- AUX: In UD, content verb is head, auxiliary is dependent. In SUD, REVERSED — auxiliary is head (SUD comp:aux).
- COP: In UD, predicate nominal/adjective is head, copula is dependent. In SUD, REVERSED — copula is head (SUD comp:pred with AUX governor).
- CASE: In UD, noun is head, adposition is dependent. In SUD, REVERSED — adposition is head (SUD comp:obj with ADP governor).
- MARK: In UD, clause verb is head, subordinator is dependent. In SUD, REVERSED — subordinator is head (SUD comp:obj with SCONJ governor).

The conversion SUD→UD involves three main steps: (1) transforming conj analysis, (2) reversing the direction of comp:aux, comp:pred with AUX governor, and comp:obj with ADP/SCONJ/PART governor, and (3) relation relabeling [12].

Yadav, Mittal & Husain (2022) explicitly chose SUD treebanks (version 2.4, 54 languages) because "this research subscribes to sentential representations...where function words are held to be syntactic heads" [13]. Their IC (Intervener Complexity) metric counts intervening syntactic HEADS between a dependent and its head. Their key finding: DLM (dependency length minimization) "does not grow slower in real language trees compared to that in IC-matched RLAs," suggesting DLM may be epiphenomenal to IC constraints [13].

The SUD vs UD choice directly affects IC computation: since IC counts intervening HEADS, and SUD treats function words (adpositions, auxiliaries, copulas, subordinators) as heads while UD does not, IC values will systematically differ between the two frameworks [10, 11, 13].

## TOPIC 5: Synthesis — Positioning IMB in the Literature

### Comparison Table

| Metric | What is counted | Per-position? | Age-weighted? | Concurrent? | Framework |
|---|---|---|---|---|---|
| Gibson 1998 SPLT storage | Predicted syntactic heads | Yes | Yes (locality) | Yes (count) | Theory |
| Gibson 2000 DLT storage | Predicted syntactic heads | Yes | No (independent of time) | Yes (count) | Theory |
| Chen et al. 2005 | Predicted syntactic heads | Yes | No (binary count) | Yes (count) | Experimental |
| Kobele/Gerth tenure | Parser stack persistence | Per-node | Yes (duration) | No (per-node) | MG parser |
| Kobele/Gerth payload | Nodes on stack | Yes | No (count) | Yes (count) | MG parser |
| Graf et al. 2017 | Tenure+payload+size | Yes | Separate metrics | Separate metrics | MG parser |
| IMB | Open dependency ages | Yes | Yes (age sum) | Yes (weighted) | Dependency tree |

### Three Specific Novelty Claims

(a) Age-weighting reintroduced: Gibson 1998 SPLT had locality in storage cost [1], but Gibson 2000 DLT explicitly removed it [2, 4]. Chen et al. 2005 experimentally confirmed binary counting works but did NOT test age-weighting [6]. IMB reintroduces age-weighting via formal age-weighted sum, supported by ACT-R activation decay theory.

(b) max(IMB) as temporal overlap signal: max(IMB) captures a signal about simultaneous dependency overlap that is invisible to dependency distance moments (mean, variance). DD measures individual dependency lengths, while IMB measures concurrent load at specific positions.

(c) Parametric convexity framework: Rather than assuming a fixed linear or binary cost function, IMB treats the shape of the cost function as an empirical question.

### SUD vs UD Issue for Phase 3

Yadav et al. 2022 used SUD (54 languages) — IC values depend on SUD headedness conventions [13]. Phase 3 uses UD treebanks — IC values will differ systematically for aux/cop/case/mark relations [11, 12]. Recommendation: pilot comparison of IC values between UD and SUD for 5 treebanks, reporting correlation and mean difference.

### Reframing Paragraph

IMB is best understood as an age-weighted generalization of Gibson's DLT storage cost [2, 4]. Where DLT storage cost at position j counts the NUMBER of open dependencies (a binary tally yielding the payload), IMB at position j sums the AGES of all open dependencies. Formally: Gibson DLT storage(j) = |{k : dependency k is open at j}|, while IMB(j) = Σ_{k open at j} (j − start_k). Gibson's storage cost is thus IMB with all ages set to 1 — the zero-order approximation. This positions IMB not as an entirely new metric but as a principled generalization that reintroduces the locality sensitivity Gibson 1998 originally proposed [1] but Gibson 2000 removed [4], now grounded in ACT-R activation decay. The combination of tenure and payload into a single summed metric is absent from Kobele/Gerth [7] and Graf et al. [8, 9], who kept these as separate dimensions.

## Sources

[1] [Gibson 1998 - Linguistic complexity: locality of syntactic dependencies (Cognition 68:1-76)](https://pubmed.ncbi.nlm.nih.gov/9775516/) — Defines SPLT storage cost as count of predicted syntactic heads (Memory Units), WITH locality — longer predictions cost more. Integration cost also distance-sensitive.

[2] [Working memory differences in long-distance dependency resolution (Frontiers in Psychology 2015)](https://pmc.ncbi.nlm.nih.gov/articles/PMC4369666/) — Confirms Gibson 2000 DLT storage cost is independent of time and integration cost is based on intervening discourse referents.

[3] [Gibson 1998 - Linguistic complexity: locality of syntactic dependencies (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0010027798000341) — Original paper defining SPLT with locality-sensitive storage and integration costs.

[4] [Gibson 2000 - The dependency locality theory: A distance-based theory of linguistic complexity](https://www2.bcs.rochester.edu/sites/fjaeger/teaching/LabSyntax2006/readings/Gibson_2000.pdf) — Revised theory removing locality from storage cost, defining integration cost as function of intervening discourse referents.

[5] [Gibson 2000 DLT on ResearchGate](https://www.researchgate.net/publication/247829094_The_dependency_locality_theory_A_distance-based_theory_of_linguistic_complexity) — DLT storage cost = number of predicted heads; integration cost = distance in discourse referents.

[6] [Chen, Gibson & Wolf 2005 - Online syntactic storage costs in sentence comprehension (JML 52:144-169)](https://www.sciencedirect.com/science/article/abs/pii/S0749596X04001147) — Three experiments validating binary storage cost counting. Key finding: storage duration did NOT significantly affect reading times. Age-weighting NOT tested.

[7] [Kobele, Gerth & Hale 2013 - Memory Resource Allocation in Top-Down Minimalist Parsing (FG 2012, LNCS 8036)](https://link.springer.com/chapter/10.1007/978-3-642-39998-5_3) — Introduced stack tenure metric — number of time steps a derivation tree node persists on parser's stack. Derives center-embedding contrasts.

[8] [Graf, Monette & Zhang 2017 - Relative clauses as a benchmark for Minimalist parsing (JLM 5:57-106)](https://jlm.ipipan.waw.pl/index.php/JLM/article/view/157) — Formalized three-way distinction: tenure (how long), payload (how many), size (how many bits). Generated 1,600 metrics.

[9] [Graf et al. 2015 - A Refined Notion of Memory Usage for Minimalist Parsing (MoL 2015)](https://aclanthology.org/W15-2301/) — Conference version introducing refined tenure/payload/size distinction for minimalist parsing complexity metrics.

[10] [Surface Syntactic Universal Dependencies (SUD) - Official Website](https://surfacesyntacticud.org/) — SUD annotation scheme based on syntactic criteria favoring functional heads, near-isomorphic to UD.

[11] [SUD General Principles and Guidelines](https://surfacesyntacticud.github.io/guidelines/u/general_principles/) — Documents SUD's distributional criterion for headedness and specific relation mappings to UD.

[12] [Gerdes et al. 2019 - Improving Surface-syntactic Universal Dependencies (SUD) (TLT/DepLing 2019)](https://syntaxfest.github.io/syntaxfest19/proceedings/papers/paper_75.pdf) — Details SUD-to-UD conversion steps including reversal of comp:aux, comp:pred, comp:obj relations.

[13] [Yadav, Mittal & Husain 2022 - A Reappraisal of Dependency Length Minimization (Open Mind 6:147-168)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9692064/) — Used SUD v2.4 (54 languages). IC counts intervening heads. Found DLM disappears after IC control.

## Follow-up Questions

- What is the quantitative difference in IC values when computed from UD vs SUD treebanks for the same sentences — and does this difference correlate with language-specific frequency of adpositions and auxiliaries?
- Does ACT-R activation decay theory predict a specific functional form (exponential, power-law) for how dependency age should be weighted in IMB, and has this been tested against reading time data?
- Could the Chen et al. (2005) null finding for time-dependence of storage cost be a power issue, given that their experiments used relatively short sentences where age differences between pending dependencies were small?

---
*Generated by AI Inventor Pipeline*
