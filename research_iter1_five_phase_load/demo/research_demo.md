# Five-Phase Load-Smoothing Experiment Specifications

## Summary

Research delivers executable specifications for all five phases of the load-smoothing experiment. Track 1: Futrell et al. (2015) projective baseline uses Gildea-Temperley recursive algorithm — randomly partition dependents left/right of head, shuffle, recurse — 100 samples/sentence, uniform over projective orderings. Track 2: Yadav et al. (2022) IC counts syntactic heads between head-dependent across 54 SUD treebanks; DLM vanishes when IC+topology controlled via IC-matched RLAs (beta3=0.01). Six rejection-sampled baselines and mixed-effects models specified. Track 3: ACT-R fan effect yields cost~n^(1+W), implying alpha in [1.5, 2.0]; center-embedding and interference evidence support alpha>=2; grid {1.0,1.2,1.5,2.0,3.0} well-motivated. Track 4: Five viable spoken-written UD pairs identified — Slovenian sl_sst/sl_ssj (best, 98K tokens), French fr_rhapsodie/fr_gsd, Norwegian no_nynorsklia/no_nynorsk, Hebrew he_iahltknesset/he_iahltwiki, Czech cs_pdtc/cs_pdt. Track 5: Clark et al. (2023) UID uses surprisal variance and local variation from per-language Transformer LMs with Hahn et al. counterfactual grammars. LLF and UID are orthogonal: UID smooths information content, LLF smooths structural memory burden. SUD-vs-UD choice impacts IC computation.

## Research Findings

## Track 1: Projective Baseline Linearization Algorithm

The random projective linearization algorithm used by Futrell, Mahowald & Gibson (2015) originates from Gildea & Temperley (2010) [1, 7]. The procedure works recursively: starting at the root node of a dependency tree, collect the head word and its dependents and order them randomly, then repeat for each dependent [1]. More precisely, at each node, dependents are randomly partitioned into a left set and right set, each set is randomly permuted, the left dependents are linearized recursively, then the head is placed, then the right dependents are linearized recursively [6, 7]. This is an UNCONSTRAINED projective baseline — each dependent is equally likely to appear on either side of its head, producing a uniform distribution over all projective orderings of the tree [1, 6].

Futrell et al. used 100 random linearization samples per sentence [1]. Dependency length is defined as the number of words between head and dependent (including the dependent), excluding punctuation and root arcs [1]. Sentences up to length 100 were included in regression analysis, with DLM significance emerging at sentence length >= 12 [1]. A second "fixed word order" baseline assigns each relation type a random weight in [-1, 1] and orders dependents by weight (head = weight 0), creating a grammar-like baseline [1]. Code is available at github.com/Futrell/cliqs [8].

**Caveat:** Liu (2015) criticized the projective-only baseline as understating DLM since projectivity itself is a DLM-related property [1].

## Track 2: Intervener Complexity and Yadav et al. (2022) Methodology

Yadav, Mittal & Husain (2022) define intervener complexity (IC) as the number of syntactic heads that intervene a dependency — for arc from d to h, IC counts words w between d and h that have at least one dependent [2]. This differs from Gibson's (2000) DLT which counts new discourse referents. The study used 54 SUD v2.4 treebanks [2]. SUD differs from UD: function words are dependents of content words [16], affecting IC computation.

Six baselines were generated via rejection sampling [2]: random structures, RLAs, DL-matched and IC-matched versions of both. Sentences restricted to <= 12 words [2]. Models were linear mixed-effects: IC_ij = beta0 + (beta1+u1j)*S_ij + (beta2+u2j)*R_ij + (beta3+u3j)*S_ij*R_ij + epsilon, where beta3 is the key coefficient [2].

**Critical finding:** DLM disappears with IC-matched RLAs: beta3 = 0.01, t = 3.5 [2]. IC minimization survives all controls (beta3 = -0.02 to -0.17) [2]. Dyer (2023) replicated on 35 languages with lesser effects in verb-final languages [21].

## Track 3: Psycholinguistic Evidence for Cost Convexity

**ACT-R Fan Effect:** Lewis & Vasishth (2005) model: A_i = B_i + sum(W_j * S_ji), S_ji = S - ln(fan_j), T = F*exp(-A_i) [5, 9]. With fan=n: T proportional to n^W. Total concurrent load for n dependencies: cost ~ n^(1+W). With W=0.5: alpha~1.5; W=1.0: alpha~2.0 [5, 9].

**Center Embedding:** Catastrophic at depth >= 2 (Miller & Chomsky 1963), maximum 3 levels in writing (Karlsson 2007), implying alpha >= 2 [23]. Lewis & Vasishth explained via non-discriminating retrieval cues [5].

**Interference:** Gordon et al. (2002) showed similarity-modulated interference consistent with superlinear cost [10]. Engelmann et al. (2019) extended with prominence effects [18].

**Alpha Grid {1.0, 1.2, 1.5, 2.0, 3.0}:** Well-motivated. alpha=1.0 is negative control; alpha=1.5 matches ACT-R 2-cue prediction; alpha=2.0 matches 1-cue and center-embedding evidence; alpha=3.0 tests strong superlinearity [5, 9, 23].

## Track 4: Spoken vs. Written UD Treebank Pairs

Five viable pairs identified [4, 11, 12, 13, 14, 15, 22, 24]:

1. **Slovenian** sl_sst (98,393 tok) vs sl_ssj — EXCELLENT: same team, spontaneous speech [11]
2. **French** fr_rhapsodie (43,699 tok) vs fr_gsd — GOOD: size imbalance [12]
3. **Norwegian** no_nynorsklia (55,410 tok) vs no_nynorsk — MODERATE: historical dialects [14]
4. **Hebrew** he_iahltknesset (67,007 tok) vs he_iahltwiki — MODERATE: formal speech [24]
5. **Czech** cs_pdtc/PDTSC vs cs_pdt — CONDITIONAL: requires genre filtering [15]

Spanish es_coser is too small (539 sents) [13]. Naija pcm_nsc has no written pair [25]. Annotation varies for reparandum, discourse:filler across treebanks [4].

## Track 5: Clark et al. (2023) UID Methodology

Clark et al. used UIDv (surprisal variance) and UIDlv (local variation) computed from per-language Transformer LMs trained via fairseq on Wiki40b/CC100 [3, 17]. Counterfactual orders use Hahn et al. (2020) formalism assigning weights to UD relation types [3, 17]. Reverse = negated weights; implausible = unattested typological patterns [3].

**LLF vs UID:** Both are uniformity hypotheses but optimize different quantities [3, 19, 20]. UID smooths information content (surprisal); LLF smooths structural memory burden (IMB). They are orthogonal: uniform surprisal can co-occur with variable IMB. LLF is purely structural (needs only parse), UID requires a language model [3, 17]. This makes them complementary, testable independently.

## Sources

[1] [Futrell, Mahowald & Gibson (2015). Large-scale evidence of dependency length minimization. PNAS.](https://pmc.ncbi.nlm.nih.gov/articles/PMC4547262/) — Primary source for projective linearization algorithm (100 samples, Gildea-Temperley procedure), dependency length definition, and baselines.

[2] [Yadav, Mittal & Husain (2022). A Reappraisal of Dependency Length Minimization. Open Mind.](https://pmc.ncbi.nlm.nih.gov/articles/PMC9692064/) — IC definition, six rejection-sampled baselines, mixed-effects models across 54 SUD treebanks, DLM disappearance result.

[3] [Clark et al. (2023). A Cross-Linguistic Pressure for Uniform Information Density in Word Order. TACL.](https://aclanthology.org/2023.tacl-1.59/) — UID metrics (surprisal variance, local variation), counterfactual grammar formalism, 10 languages, Transformer LMs.

[4] [Dobrovoljc (2022). Spoken Language Treebanks in Universal Dependencies. LREC.](https://aclanthology.org/2022.lrec-1.191/) — Comprehensive survey of spoken UD treebanks documenting annotation differences for speech phenomena.

[5] [Lewis & Vasishth (2005). Activation-Based Model of Sentence Processing. Cognitive Science.](https://pubmed.ncbi.nlm.nih.gov/21702779/) — ACT-R activation equations, fan effect, retrieval time formula, center-embedding explanation.

[6] [Temperley & Gildea (2018). Minimizing Syntactic Dependency Lengths.](https://www.cs.rochester.edu/u/gildea/pubs/temperley-gildea-ar18.pdf) — Review chapter with detailed random projective linearization algorithm description.

[7] [Gildea & Temperley (2007). Optimizing Grammars for Minimum Dependency Length. ACL.](https://aclanthology.org/P07-1024/) — Original source for grammar optimization and random baseline generation procedures.

[8] [Futrell/cliqs GitHub repository](https://github.com/Futrell/cliqs) — Python implementation of dependency length analysis with random and optimal linearization baselines.

[9] [ACT-R Tutorial Unit 5: Activation and Probability of Recall](http://act-r.psy.cmu.edu/wordpress/wp-content/themes/ACT-R/tutorials/unit5.htm) — Official ACT-R activation equations, fan effect formula S_ji=S-ln(fan), retrieval time, default parameters.

[10] [Gordon, Hendrick & Levine (2002). Memory-Load Interference. Psychological Science.](https://journals.sagepub.com/doi/10.1111/1467-9280.00475) — Evidence for similarity-based interference modulating processing difficulty in syntactic processing.

[11] [UD Slovenian SST treebank](https://universaldependencies.org/treebanks/sl_sst/index.html) — Spoken Slovenian: 6121 utterances, 98393 tokens, spontaneous speech with disfluency relations.

[12] [UD French Rhapsodie treebank](https://universaldependencies.org/treebanks/fr_rhapsodie/index.html) — Spoken French: 3209 sentences, 43699 tokens, with reparandum and parataxis:insert.

[13] [UD Spanish COSER treebank](https://universaldependencies.org/treebanks/es_coser/index.html) — Spoken rural Spanish: 539 sentences, 7987 tokens. Too small for reliable analysis.

[14] [UD Norwegian NynorskLIA treebank](https://github.com/UniversalDependencies/UD_Norwegian-NynorskLIA) — Spoken Norwegian dialects: 5250 sentences, 55410 tokens from 1950-1990 recordings.

[15] [UD Czech PDTC treebank](https://universaldependencies.org/treebanks/cs_pdtc/index.html) — Contains PDTSC spoken portion within 213897-sentence consolidated treebank.

[16] [Surface Syntactic Universal Dependencies (SUD)](https://surfacesyntacticud.org/) — SUD uses distributional criteria for headedness; near-isomorphic to UD with bidirectional conversion.

[17] [Clark et al. word-order-uid code repository](https://github.com/thomashikaru/word-order-uid) — Code using fairseq Transformer LMs on Wiki40b/CC100 with Hahn et al. counterfactual grammars.

[18] [Engelmann, Jager & Vasishth (2019). Prominence and Cue Association. Cognitive Science.](https://onlinelibrary.wiley.com/doi/10.1111/cogs.12800) — Extended ACT-R retrieval model evaluating predictions against meta-analysis of interference data.

[19] [Futrell, Gibson & Levy (2020). Lossy-Context Surprisal. Cognitive Science.](https://onlinelibrary.wiley.com/doi/10.1111/cogs.12814) — Unified framework bridging surprisal and DLT via lossy memory representations.

[20] [Futrell, Levy & Gibson (2020). Dependency locality as explanatory principle. Language.](http://tedlab.mit.edu/tedlab_website/researchpapers/Futrell_Levy_Gibson_2020.pdf) — Large-scale corpus evidence that dependency locality predicts word order in grammar and usage.

[21] [Dyer (2023). Revisiting dependency length and intervener complexity. SIGTYP.](https://aclanthology.org/2023.sigtyp-1.11/) — Replication on 35 languages confirming DLM and ICM with lesser effects in verb-final languages.

[22] [Universal Dependencies v2.15 release (November 2024)](https://www.mail-archive.com/corpora@list.elra.info/msg03918.html) — Documents new Hebrew IAHLTknesset (67K tokens) and Slovenian SST updates.

[23] [Center embedding - Wikipedia](https://en.wikipedia.org/wiki/Center_embedding) — Miller & Chomsky evidence, Karlsson depth-3 limit, interference-based explanations.

[24] [UD Hebrew IAHLTknesset treebank](https://github.com/UniversalDependencies/UD_Hebrew-IAHLTknesset/tree/dev) — Spoken Hebrew parliamentary proceedings: ~2800 sentences, 67K tokens, UD v2.15.

[25] [UD Naija NSC treebank](https://universaldependencies.org/treebanks/pcm_nsc/index.html) — Spoken Naija: 9242 sentences, 140729 tokens. No written counterpart in UD.

## Follow-up Questions

- What is the exact list of 10 languages in Clark et al. (2023), and what per-language Transformer LM architectures and training hyperparameters were used? The full PDF was not accessible for detailed extraction.
- Can the PDTSC spoken portion of the Czech cs_pdtc treebank be reliably isolated by genre metadata fields, and what is its exact size in sentences and tokens for the spoken-only subset?
- Has anyone attempted to estimate the cost convexity exponent alpha empirically from reading time data, e.g., by fitting IMB^alpha to self-paced reading times across different dependency load configurations and comparing model fit across alpha values?

---
*Generated by AI Inventor Pipeline*
