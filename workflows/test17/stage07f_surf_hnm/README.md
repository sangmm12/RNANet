# RNANet Stage07F — post-hoc CY-C+SURF+HNM on Test17

Purpose: test whether the already-frozen Stage07B HNM rule adds value on top of the already-computed Stage07E CY-C+SURF representation.

Scientific status:
- exploratory/post-hoc for Test17;
- Test17 had already been observed before this arm was defined;
- no Test17 tuning is performed within this arm;
- exact frozen Stage07B Train60 train/validation chain splits are reused;
- exact frozen HNM rule is reused: top 20 negatives per training chain, weight 5;
- Stage-1 hard-negative scores come from the exact saved Stage07E CY-C+SURF model for each seed;
- feature space remains 103D: CY-C42 + nt one-hot4 + SURF57; no B-factor.

This experiment can inform representation/training interaction, but it is not an untouched confirmatory external test.
