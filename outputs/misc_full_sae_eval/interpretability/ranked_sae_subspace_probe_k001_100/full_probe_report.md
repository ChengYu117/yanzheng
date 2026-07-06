# Full Representation Probe

This experiment evaluates whether MISC labels are linearly decodable from complete representation vectors.

It differs from Step 6 `Mean Label AUC`, which measures strongest single-feature association, and from minimal sufficient subspace `Full AUC`, which uses a per-label candidate pool.

## Configuration

- Split policy: `stratified-group-kfold`
- Folds: `5`
- Group column: `file_id`
- PCA components: `full`
- Logistic C: `1.0`
- Solver: `liblinear`
- Standardize: `True`
- Include SAE ranked subspaces: `True`
- SAE subspace rankings: `('cohens_d', 'directional_auc')`
- SAE subspace top-n grid: `(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100)`
- SAE subspace n_jobs: `8`
- Filter SAE subspace candidates: `True`

## Macro Summary

| representation | n_labels | mean_n_features | macro_auc | macro_average_precision | macro_f1 | macro_balanced_accuracy | macro_accuracy | subspace_ranking | top_n | source_representation | candidate_pool_size | candidate_filter_enabled |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| full_sae_latents | 9 | 32768.000 | 0.890 | 0.600 | 0.572 | 0.815 | 0.873 | nan | nan | nan | nan | nan |
| pca_raw_hidden | 9 | 4096.000 | 0.900 | 0.679 | 0.642 | 0.811 | 0.912 | nan | nan | nan | nan | nan |
| raw_hidden | 9 | 4096.000 | 0.900 | 0.679 | 0.642 | 0.811 | 0.912 | nan | nan | nan | nan | nan |
| sae_top_cohens_d_n000 | 9 | 0.000 | 0.500 | 0.142 | 0.000 | 0.500 | 0.858 | cohens_d | 0.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n001 | 9 | 1.000 | 0.696 | 0.357 | 0.407 | 0.684 | 0.857 | cohens_d | 1.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n002 | 9 | 2.000 | 0.746 | 0.414 | 0.440 | 0.714 | 0.837 | cohens_d | 2.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n003 | 9 | 3.000 | 0.762 | 0.440 | 0.459 | 0.726 | 0.849 | cohens_d | 3.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n004 | 9 | 4.000 | 0.774 | 0.461 | 0.472 | 0.734 | 0.849 | cohens_d | 4.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n005 | 9 | 5.000 | 0.784 | 0.473 | 0.479 | 0.740 | 0.847 | cohens_d | 5.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n006 | 9 | 6.000 | 0.797 | 0.489 | 0.492 | 0.750 | 0.850 | cohens_d | 6.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n007 | 9 | 7.000 | 0.806 | 0.500 | 0.494 | 0.756 | 0.846 | cohens_d | 7.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n008 | 9 | 8.000 | 0.814 | 0.510 | 0.504 | 0.764 | 0.848 | cohens_d | 8.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n009 | 9 | 9.000 | 0.818 | 0.518 | 0.507 | 0.768 | 0.848 | cohens_d | 9.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n010 | 9 | 10.000 | 0.822 | 0.524 | 0.511 | 0.772 | 0.847 | cohens_d | 10.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n011 | 9 | 11.000 | 0.828 | 0.532 | 0.510 | 0.771 | 0.844 | cohens_d | 11.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n012 | 9 | 12.000 | 0.837 | 0.540 | 0.514 | 0.779 | 0.844 | cohens_d | 12.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n013 | 9 | 13.000 | 0.840 | 0.544 | 0.518 | 0.781 | 0.846 | cohens_d | 13.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n014 | 9 | 14.000 | 0.841 | 0.549 | 0.519 | 0.783 | 0.846 | cohens_d | 14.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n015 | 9 | 15.000 | 0.843 | 0.552 | 0.522 | 0.787 | 0.845 | cohens_d | 15.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n016 | 9 | 16.000 | 0.845 | 0.555 | 0.524 | 0.789 | 0.845 | cohens_d | 16.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n017 | 9 | 17.000 | 0.845 | 0.557 | 0.524 | 0.789 | 0.845 | cohens_d | 17.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n018 | 9 | 18.000 | 0.846 | 0.559 | 0.526 | 0.790 | 0.846 | cohens_d | 18.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n019 | 9 | 19.000 | 0.848 | 0.559 | 0.527 | 0.791 | 0.846 | cohens_d | 19.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n020 | 9 | 20.000 | 0.849 | 0.560 | 0.528 | 0.791 | 0.847 | cohens_d | 20.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n021 | 9 | 21.000 | 0.850 | 0.560 | 0.528 | 0.792 | 0.847 | cohens_d | 21.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n022 | 9 | 22.000 | 0.851 | 0.562 | 0.527 | 0.791 | 0.846 | cohens_d | 22.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n023 | 9 | 23.000 | 0.852 | 0.563 | 0.529 | 0.793 | 0.846 | cohens_d | 23.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n024 | 9 | 24.000 | 0.853 | 0.568 | 0.532 | 0.794 | 0.848 | cohens_d | 24.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n025 | 9 | 25.000 | 0.853 | 0.568 | 0.532 | 0.794 | 0.848 | cohens_d | 25.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n026 | 9 | 26.000 | 0.855 | 0.570 | 0.533 | 0.795 | 0.849 | cohens_d | 26.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n027 | 9 | 27.000 | 0.854 | 0.571 | 0.534 | 0.795 | 0.849 | cohens_d | 27.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n028 | 9 | 28.000 | 0.854 | 0.572 | 0.532 | 0.795 | 0.848 | cohens_d | 28.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n029 | 9 | 29.000 | 0.855 | 0.573 | 0.533 | 0.795 | 0.849 | cohens_d | 29.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n030 | 9 | 30.000 | 0.858 | 0.576 | 0.533 | 0.795 | 0.849 | cohens_d | 30.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n031 | 9 | 31.000 | 0.860 | 0.578 | 0.534 | 0.796 | 0.849 | cohens_d | 31.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n032 | 9 | 32.000 | 0.861 | 0.580 | 0.535 | 0.797 | 0.850 | cohens_d | 32.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n033 | 9 | 33.000 | 0.863 | 0.583 | 0.537 | 0.799 | 0.850 | cohens_d | 33.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n034 | 9 | 34.000 | 0.865 | 0.584 | 0.540 | 0.801 | 0.851 | cohens_d | 34.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n035 | 9 | 35.000 | 0.865 | 0.585 | 0.540 | 0.801 | 0.851 | cohens_d | 35.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n036 | 9 | 36.000 | 0.867 | 0.585 | 0.539 | 0.801 | 0.851 | cohens_d | 36.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n037 | 9 | 37.000 | 0.867 | 0.585 | 0.541 | 0.802 | 0.851 | cohens_d | 37.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n038 | 9 | 38.000 | 0.866 | 0.587 | 0.541 | 0.803 | 0.852 | cohens_d | 38.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n039 | 9 | 39.000 | 0.868 | 0.588 | 0.542 | 0.804 | 0.851 | cohens_d | 39.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n040 | 9 | 40.000 | 0.868 | 0.589 | 0.543 | 0.804 | 0.852 | cohens_d | 40.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n041 | 9 | 41.000 | 0.870 | 0.590 | 0.544 | 0.806 | 0.852 | cohens_d | 41.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n042 | 9 | 42.000 | 0.870 | 0.589 | 0.545 | 0.807 | 0.852 | cohens_d | 42.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n043 | 9 | 43.000 | 0.872 | 0.589 | 0.546 | 0.808 | 0.853 | cohens_d | 43.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n044 | 9 | 44.000 | 0.872 | 0.589 | 0.545 | 0.806 | 0.852 | cohens_d | 44.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n045 | 9 | 45.000 | 0.872 | 0.589 | 0.544 | 0.806 | 0.852 | cohens_d | 45.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n046 | 9 | 46.000 | 0.872 | 0.590 | 0.544 | 0.806 | 0.852 | cohens_d | 46.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n047 | 9 | 47.000 | 0.872 | 0.590 | 0.544 | 0.806 | 0.852 | cohens_d | 47.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n048 | 9 | 48.000 | 0.873 | 0.590 | 0.545 | 0.808 | 0.852 | cohens_d | 48.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n049 | 9 | 49.000 | 0.874 | 0.591 | 0.546 | 0.808 | 0.853 | cohens_d | 49.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n050 | 9 | 50.000 | 0.874 | 0.591 | 0.544 | 0.806 | 0.852 | cohens_d | 50.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n051 | 9 | 51.000 | 0.874 | 0.592 | 0.545 | 0.807 | 0.852 | cohens_d | 51.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n052 | 9 | 52.000 | 0.875 | 0.593 | 0.545 | 0.807 | 0.852 | cohens_d | 52.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n053 | 9 | 53.000 | 0.875 | 0.593 | 0.546 | 0.808 | 0.853 | cohens_d | 53.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n054 | 9 | 54.000 | 0.875 | 0.595 | 0.546 | 0.808 | 0.853 | cohens_d | 54.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n055 | 9 | 55.000 | 0.875 | 0.595 | 0.547 | 0.808 | 0.854 | cohens_d | 55.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n056 | 9 | 56.000 | 0.876 | 0.596 | 0.548 | 0.809 | 0.854 | cohens_d | 56.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n057 | 9 | 57.000 | 0.876 | 0.596 | 0.549 | 0.810 | 0.854 | cohens_d | 57.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n058 | 9 | 58.000 | 0.876 | 0.596 | 0.550 | 0.811 | 0.855 | cohens_d | 58.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n059 | 9 | 59.000 | 0.876 | 0.597 | 0.550 | 0.811 | 0.855 | cohens_d | 59.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n060 | 9 | 60.000 | 0.877 | 0.598 | 0.550 | 0.811 | 0.856 | cohens_d | 60.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n061 | 9 | 61.000 | 0.877 | 0.598 | 0.550 | 0.810 | 0.856 | cohens_d | 61.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n062 | 9 | 62.000 | 0.877 | 0.598 | 0.550 | 0.811 | 0.855 | cohens_d | 62.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n063 | 9 | 63.000 | 0.878 | 0.598 | 0.550 | 0.811 | 0.855 | cohens_d | 63.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n064 | 9 | 64.000 | 0.878 | 0.598 | 0.551 | 0.812 | 0.856 | cohens_d | 64.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n065 | 9 | 65.000 | 0.878 | 0.599 | 0.551 | 0.812 | 0.856 | cohens_d | 65.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n066 | 9 | 66.000 | 0.878 | 0.600 | 0.552 | 0.812 | 0.856 | cohens_d | 66.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n067 | 9 | 67.000 | 0.878 | 0.601 | 0.552 | 0.812 | 0.856 | cohens_d | 67.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n068 | 9 | 68.000 | 0.878 | 0.599 | 0.552 | 0.811 | 0.856 | cohens_d | 68.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n069 | 9 | 69.000 | 0.879 | 0.600 | 0.553 | 0.812 | 0.857 | cohens_d | 69.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n070 | 9 | 70.000 | 0.880 | 0.600 | 0.553 | 0.813 | 0.857 | cohens_d | 70.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n071 | 9 | 71.000 | 0.881 | 0.600 | 0.553 | 0.812 | 0.857 | cohens_d | 71.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n072 | 9 | 72.000 | 0.881 | 0.600 | 0.554 | 0.812 | 0.857 | cohens_d | 72.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n073 | 9 | 73.000 | 0.881 | 0.600 | 0.553 | 0.812 | 0.857 | cohens_d | 73.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n074 | 9 | 74.000 | 0.882 | 0.600 | 0.554 | 0.813 | 0.857 | cohens_d | 74.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n075 | 9 | 75.000 | 0.882 | 0.601 | 0.554 | 0.813 | 0.857 | cohens_d | 75.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n076 | 9 | 76.000 | 0.882 | 0.601 | 0.555 | 0.814 | 0.857 | cohens_d | 76.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n077 | 9 | 77.000 | 0.882 | 0.601 | 0.555 | 0.814 | 0.858 | cohens_d | 77.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n078 | 9 | 78.000 | 0.882 | 0.601 | 0.556 | 0.814 | 0.858 | cohens_d | 78.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n079 | 9 | 79.000 | 0.882 | 0.601 | 0.556 | 0.815 | 0.858 | cohens_d | 79.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n080 | 9 | 80.000 | 0.882 | 0.601 | 0.556 | 0.814 | 0.858 | cohens_d | 80.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n081 | 9 | 81.000 | 0.882 | 0.603 | 0.556 | 0.814 | 0.859 | cohens_d | 81.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n082 | 9 | 82.000 | 0.882 | 0.602 | 0.556 | 0.815 | 0.859 | cohens_d | 82.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n083 | 9 | 83.000 | 0.883 | 0.603 | 0.557 | 0.814 | 0.859 | cohens_d | 83.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n084 | 9 | 84.000 | 0.883 | 0.603 | 0.557 | 0.814 | 0.859 | cohens_d | 84.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n085 | 9 | 85.000 | 0.884 | 0.603 | 0.557 | 0.814 | 0.859 | cohens_d | 85.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n086 | 9 | 86.000 | 0.883 | 0.603 | 0.557 | 0.815 | 0.859 | cohens_d | 86.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n087 | 9 | 87.000 | 0.883 | 0.601 | 0.557 | 0.815 | 0.859 | cohens_d | 87.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n088 | 9 | 88.000 | 0.883 | 0.601 | 0.558 | 0.815 | 0.859 | cohens_d | 88.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n089 | 9 | 89.000 | 0.884 | 0.601 | 0.558 | 0.815 | 0.859 | cohens_d | 89.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n090 | 9 | 90.000 | 0.884 | 0.602 | 0.558 | 0.815 | 0.860 | cohens_d | 90.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n091 | 9 | 91.000 | 0.884 | 0.604 | 0.558 | 0.815 | 0.860 | cohens_d | 91.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n092 | 9 | 92.000 | 0.884 | 0.603 | 0.557 | 0.813 | 0.859 | cohens_d | 92.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n093 | 9 | 93.000 | 0.884 | 0.603 | 0.557 | 0.814 | 0.859 | cohens_d | 93.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n094 | 9 | 94.000 | 0.884 | 0.604 | 0.558 | 0.814 | 0.860 | cohens_d | 94.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n095 | 9 | 95.000 | 0.884 | 0.604 | 0.557 | 0.814 | 0.860 | cohens_d | 95.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n096 | 9 | 96.000 | 0.884 | 0.603 | 0.557 | 0.814 | 0.860 | cohens_d | 96.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n097 | 9 | 97.000 | 0.883 | 0.603 | 0.558 | 0.815 | 0.860 | cohens_d | 97.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n098 | 9 | 98.000 | 0.884 | 0.602 | 0.559 | 0.815 | 0.860 | cohens_d | 98.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n099 | 9 | 99.000 | 0.883 | 0.602 | 0.558 | 0.814 | 0.860 | cohens_d | 99.000 | full_sae_latents | 11921.000 | True |
| sae_top_cohens_d_n100 | 9 | 100.000 | 0.883 | 0.602 | 0.558 | 0.815 | 0.860 | cohens_d | 100.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n000 | 9 | 0.000 | 0.500 | 0.142 | 0.000 | 0.500 | 0.858 | directional_auc | 0.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n001 | 9 | 1.000 | 0.752 | 0.349 | 0.419 | 0.725 | 0.710 | directional_auc | 1.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n002 | 9 | 2.000 | 0.798 | 0.411 | 0.441 | 0.747 | 0.714 | directional_auc | 2.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n003 | 9 | 3.000 | 0.810 | 0.425 | 0.453 | 0.755 | 0.748 | directional_auc | 3.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n004 | 9 | 4.000 | 0.815 | 0.437 | 0.451 | 0.759 | 0.747 | directional_auc | 4.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n005 | 9 | 5.000 | 0.822 | 0.449 | 0.454 | 0.762 | 0.745 | directional_auc | 5.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n006 | 9 | 6.000 | 0.833 | 0.466 | 0.461 | 0.768 | 0.753 | directional_auc | 6.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n007 | 9 | 7.000 | 0.839 | 0.474 | 0.466 | 0.772 | 0.759 | directional_auc | 7.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n008 | 9 | 8.000 | 0.844 | 0.480 | 0.470 | 0.776 | 0.764 | directional_auc | 8.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n009 | 9 | 9.000 | 0.849 | 0.490 | 0.476 | 0.779 | 0.773 | directional_auc | 9.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n010 | 9 | 10.000 | 0.853 | 0.498 | 0.481 | 0.782 | 0.780 | directional_auc | 10.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n011 | 9 | 11.000 | 0.857 | 0.505 | 0.490 | 0.789 | 0.790 | directional_auc | 11.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n012 | 9 | 12.000 | 0.862 | 0.515 | 0.495 | 0.793 | 0.793 | directional_auc | 12.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n013 | 9 | 13.000 | 0.864 | 0.520 | 0.498 | 0.794 | 0.796 | directional_auc | 13.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n014 | 9 | 14.000 | 0.867 | 0.521 | 0.503 | 0.799 | 0.803 | directional_auc | 14.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n015 | 9 | 15.000 | 0.870 | 0.529 | 0.508 | 0.802 | 0.807 | directional_auc | 15.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n016 | 9 | 16.000 | 0.871 | 0.531 | 0.508 | 0.801 | 0.807 | directional_auc | 16.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n017 | 9 | 17.000 | 0.872 | 0.533 | 0.511 | 0.803 | 0.809 | directional_auc | 17.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n018 | 9 | 18.000 | 0.873 | 0.536 | 0.512 | 0.804 | 0.810 | directional_auc | 18.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n019 | 9 | 19.000 | 0.875 | 0.540 | 0.514 | 0.805 | 0.811 | directional_auc | 19.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n020 | 9 | 20.000 | 0.876 | 0.542 | 0.517 | 0.808 | 0.813 | directional_auc | 20.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n021 | 9 | 21.000 | 0.876 | 0.542 | 0.515 | 0.805 | 0.812 | directional_auc | 21.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n022 | 9 | 22.000 | 0.878 | 0.547 | 0.518 | 0.806 | 0.815 | directional_auc | 22.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n023 | 9 | 23.000 | 0.881 | 0.552 | 0.521 | 0.809 | 0.816 | directional_auc | 23.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n024 | 9 | 24.000 | 0.883 | 0.556 | 0.523 | 0.810 | 0.818 | directional_auc | 24.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n025 | 9 | 25.000 | 0.884 | 0.559 | 0.524 | 0.810 | 0.819 | directional_auc | 25.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n026 | 9 | 26.000 | 0.885 | 0.562 | 0.525 | 0.811 | 0.820 | directional_auc | 26.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n027 | 9 | 27.000 | 0.886 | 0.563 | 0.525 | 0.811 | 0.820 | directional_auc | 27.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n028 | 9 | 28.000 | 0.886 | 0.563 | 0.526 | 0.812 | 0.821 | directional_auc | 28.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n029 | 9 | 29.000 | 0.886 | 0.564 | 0.528 | 0.813 | 0.822 | directional_auc | 29.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n030 | 9 | 30.000 | 0.887 | 0.566 | 0.529 | 0.814 | 0.823 | directional_auc | 30.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n031 | 9 | 31.000 | 0.888 | 0.567 | 0.529 | 0.814 | 0.823 | directional_auc | 31.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n032 | 9 | 32.000 | 0.888 | 0.569 | 0.529 | 0.814 | 0.824 | directional_auc | 32.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n033 | 9 | 33.000 | 0.889 | 0.570 | 0.529 | 0.814 | 0.824 | directional_auc | 33.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n034 | 9 | 34.000 | 0.889 | 0.572 | 0.531 | 0.815 | 0.824 | directional_auc | 34.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n035 | 9 | 35.000 | 0.891 | 0.574 | 0.532 | 0.817 | 0.826 | directional_auc | 35.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n036 | 9 | 36.000 | 0.891 | 0.574 | 0.533 | 0.818 | 0.827 | directional_auc | 36.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n037 | 9 | 37.000 | 0.892 | 0.576 | 0.534 | 0.818 | 0.827 | directional_auc | 37.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n038 | 9 | 38.000 | 0.892 | 0.575 | 0.535 | 0.818 | 0.829 | directional_auc | 38.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n039 | 9 | 39.000 | 0.892 | 0.576 | 0.535 | 0.817 | 0.829 | directional_auc | 39.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n040 | 9 | 40.000 | 0.892 | 0.576 | 0.535 | 0.818 | 0.829 | directional_auc | 40.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n041 | 9 | 41.000 | 0.893 | 0.578 | 0.537 | 0.819 | 0.830 | directional_auc | 41.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n042 | 9 | 42.000 | 0.893 | 0.579 | 0.537 | 0.818 | 0.831 | directional_auc | 42.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n043 | 9 | 43.000 | 0.893 | 0.579 | 0.537 | 0.818 | 0.831 | directional_auc | 43.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n044 | 9 | 44.000 | 0.893 | 0.580 | 0.538 | 0.818 | 0.832 | directional_auc | 44.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n045 | 9 | 45.000 | 0.894 | 0.581 | 0.540 | 0.820 | 0.833 | directional_auc | 45.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n046 | 9 | 46.000 | 0.894 | 0.582 | 0.539 | 0.819 | 0.833 | directional_auc | 46.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n047 | 9 | 47.000 | 0.894 | 0.583 | 0.541 | 0.821 | 0.835 | directional_auc | 47.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n048 | 9 | 48.000 | 0.894 | 0.584 | 0.542 | 0.821 | 0.835 | directional_auc | 48.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n049 | 9 | 49.000 | 0.895 | 0.585 | 0.542 | 0.820 | 0.835 | directional_auc | 49.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n050 | 9 | 50.000 | 0.895 | 0.586 | 0.543 | 0.822 | 0.836 | directional_auc | 50.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n051 | 9 | 51.000 | 0.896 | 0.587 | 0.544 | 0.823 | 0.837 | directional_auc | 51.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n052 | 9 | 52.000 | 0.897 | 0.588 | 0.544 | 0.822 | 0.837 | directional_auc | 52.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n053 | 9 | 53.000 | 0.897 | 0.589 | 0.544 | 0.823 | 0.837 | directional_auc | 53.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n054 | 9 | 54.000 | 0.897 | 0.589 | 0.545 | 0.823 | 0.838 | directional_auc | 54.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n055 | 9 | 55.000 | 0.897 | 0.589 | 0.544 | 0.822 | 0.838 | directional_auc | 55.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n056 | 9 | 56.000 | 0.897 | 0.590 | 0.544 | 0.822 | 0.838 | directional_auc | 56.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n057 | 9 | 57.000 | 0.898 | 0.593 | 0.544 | 0.823 | 0.838 | directional_auc | 57.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n058 | 9 | 58.000 | 0.898 | 0.594 | 0.544 | 0.822 | 0.838 | directional_auc | 58.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n059 | 9 | 59.000 | 0.899 | 0.595 | 0.547 | 0.824 | 0.840 | directional_auc | 59.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n060 | 9 | 60.000 | 0.899 | 0.595 | 0.546 | 0.824 | 0.840 | directional_auc | 60.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n061 | 9 | 61.000 | 0.900 | 0.597 | 0.547 | 0.823 | 0.841 | directional_auc | 61.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n062 | 9 | 62.000 | 0.900 | 0.597 | 0.548 | 0.825 | 0.842 | directional_auc | 62.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n063 | 9 | 63.000 | 0.900 | 0.597 | 0.550 | 0.826 | 0.842 | directional_auc | 63.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n064 | 9 | 64.000 | 0.901 | 0.598 | 0.550 | 0.826 | 0.843 | directional_auc | 64.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n065 | 9 | 65.000 | 0.901 | 0.600 | 0.551 | 0.826 | 0.843 | directional_auc | 65.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n066 | 9 | 66.000 | 0.901 | 0.601 | 0.552 | 0.826 | 0.844 | directional_auc | 66.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n067 | 9 | 67.000 | 0.902 | 0.602 | 0.554 | 0.828 | 0.844 | directional_auc | 67.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n068 | 9 | 68.000 | 0.902 | 0.602 | 0.554 | 0.828 | 0.845 | directional_auc | 68.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n069 | 9 | 69.000 | 0.902 | 0.603 | 0.555 | 0.828 | 0.846 | directional_auc | 69.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n070 | 9 | 70.000 | 0.903 | 0.603 | 0.556 | 0.829 | 0.846 | directional_auc | 70.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n071 | 9 | 71.000 | 0.903 | 0.604 | 0.556 | 0.829 | 0.846 | directional_auc | 71.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n072 | 9 | 72.000 | 0.903 | 0.605 | 0.556 | 0.830 | 0.846 | directional_auc | 72.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n073 | 9 | 73.000 | 0.903 | 0.606 | 0.557 | 0.831 | 0.847 | directional_auc | 73.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n074 | 9 | 74.000 | 0.904 | 0.606 | 0.558 | 0.831 | 0.848 | directional_auc | 74.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n075 | 9 | 75.000 | 0.904 | 0.608 | 0.558 | 0.831 | 0.847 | directional_auc | 75.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n076 | 9 | 76.000 | 0.904 | 0.610 | 0.558 | 0.831 | 0.848 | directional_auc | 76.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n077 | 9 | 77.000 | 0.904 | 0.610 | 0.558 | 0.831 | 0.848 | directional_auc | 77.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n078 | 9 | 78.000 | 0.905 | 0.611 | 0.559 | 0.831 | 0.848 | directional_auc | 78.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n079 | 9 | 79.000 | 0.905 | 0.613 | 0.560 | 0.833 | 0.849 | directional_auc | 79.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n080 | 9 | 80.000 | 0.905 | 0.613 | 0.560 | 0.833 | 0.849 | directional_auc | 80.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n081 | 9 | 81.000 | 0.905 | 0.614 | 0.560 | 0.833 | 0.849 | directional_auc | 81.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n082 | 9 | 82.000 | 0.905 | 0.615 | 0.559 | 0.832 | 0.849 | directional_auc | 82.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n083 | 9 | 83.000 | 0.906 | 0.615 | 0.560 | 0.832 | 0.849 | directional_auc | 83.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n084 | 9 | 84.000 | 0.906 | 0.617 | 0.560 | 0.832 | 0.850 | directional_auc | 84.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n085 | 9 | 85.000 | 0.906 | 0.617 | 0.561 | 0.833 | 0.850 | directional_auc | 85.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n086 | 9 | 86.000 | 0.906 | 0.617 | 0.561 | 0.832 | 0.850 | directional_auc | 86.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n087 | 9 | 87.000 | 0.906 | 0.617 | 0.562 | 0.832 | 0.851 | directional_auc | 87.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n088 | 9 | 88.000 | 0.906 | 0.616 | 0.561 | 0.832 | 0.851 | directional_auc | 88.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n089 | 9 | 89.000 | 0.906 | 0.616 | 0.561 | 0.832 | 0.851 | directional_auc | 89.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n090 | 9 | 90.000 | 0.907 | 0.617 | 0.563 | 0.833 | 0.852 | directional_auc | 90.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n091 | 9 | 91.000 | 0.907 | 0.617 | 0.563 | 0.833 | 0.852 | directional_auc | 91.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n092 | 9 | 92.000 | 0.907 | 0.618 | 0.563 | 0.833 | 0.852 | directional_auc | 92.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n093 | 9 | 93.000 | 0.907 | 0.619 | 0.564 | 0.833 | 0.852 | directional_auc | 93.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n094 | 9 | 94.000 | 0.907 | 0.619 | 0.565 | 0.834 | 0.853 | directional_auc | 94.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n095 | 9 | 95.000 | 0.907 | 0.620 | 0.566 | 0.834 | 0.853 | directional_auc | 95.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n096 | 9 | 96.000 | 0.907 | 0.620 | 0.566 | 0.835 | 0.854 | directional_auc | 96.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n097 | 9 | 97.000 | 0.908 | 0.622 | 0.568 | 0.836 | 0.854 | directional_auc | 97.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n098 | 9 | 98.000 | 0.908 | 0.622 | 0.567 | 0.835 | 0.854 | directional_auc | 98.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n099 | 9 | 99.000 | 0.908 | 0.622 | 0.567 | 0.834 | 0.855 | directional_auc | 99.000 | full_sae_latents | 11921.000 | True |
| sae_top_directional_auc_n100 | 9 | 100.000 | 0.908 | 0.622 | 0.568 | 0.835 | 0.855 | directional_auc | 100.000 | full_sae_latents | 11921.000 | True |

## Per-label Summary

| representation | label | probe_auc_mean | probe_average_precision_mean | probe_f1_mean | probe_balanced_accuracy_mean | n_features_mean |
| --- | --- | --- | --- | --- | --- | --- |
| full_sae_latents | AF | 0.927 | 0.618 | 0.509 | 0.860 | 32768.000 |
| full_sae_latents | GI | 0.823 | 0.452 | 0.447 | 0.743 | 32768.000 |
| full_sae_latents | QU | 0.970 | 0.943 | 0.881 | 0.919 | 32768.000 |
| full_sae_latents | QUC | 0.889 | 0.568 | 0.581 | 0.824 | 32768.000 |
| full_sae_latents | QUO | 0.946 | 0.848 | 0.744 | 0.883 | 32768.000 |
| full_sae_latents | RE | 0.888 | 0.692 | 0.668 | 0.816 | 32768.000 |
| full_sae_latents | REC | 0.902 | 0.642 | 0.597 | 0.832 | 32768.000 |
| full_sae_latents | RES | 0.794 | 0.274 | 0.340 | 0.677 | 32768.000 |
| full_sae_latents | SU | 0.871 | 0.365 | 0.385 | 0.780 | 32768.000 |
| pca_raw_hidden | AF | 0.950 | 0.787 | 0.697 | 0.866 | 4096.000 |
| pca_raw_hidden | GI | 0.823 | 0.470 | 0.460 | 0.711 | 4096.000 |
| pca_raw_hidden | QU | 0.976 | 0.963 | 0.913 | 0.936 | 4096.000 |
| pca_raw_hidden | QUC | 0.903 | 0.710 | 0.653 | 0.820 | 4096.000 |
| pca_raw_hidden | QUO | 0.952 | 0.884 | 0.812 | 0.897 | 4096.000 |
| pca_raw_hidden | RE | 0.898 | 0.751 | 0.695 | 0.817 | 4096.000 |
| pca_raw_hidden | REC | 0.903 | 0.681 | 0.641 | 0.817 | 4096.000 |
| pca_raw_hidden | RES | 0.803 | 0.352 | 0.408 | 0.695 | 4096.000 |
| pca_raw_hidden | SU | 0.891 | 0.513 | 0.495 | 0.740 | 4096.000 |
| raw_hidden | AF | 0.950 | 0.787 | 0.697 | 0.866 | 4096.000 |
| raw_hidden | GI | 0.823 | 0.470 | 0.460 | 0.711 | 4096.000 |
| raw_hidden | QU | 0.976 | 0.963 | 0.913 | 0.936 | 4096.000 |
| raw_hidden | QUC | 0.903 | 0.710 | 0.653 | 0.820 | 4096.000 |
| raw_hidden | QUO | 0.952 | 0.884 | 0.812 | 0.897 | 4096.000 |
| raw_hidden | RE | 0.898 | 0.751 | 0.695 | 0.817 | 4096.000 |
| raw_hidden | REC | 0.903 | 0.681 | 0.641 | 0.817 | 4096.000 |
| raw_hidden | RES | 0.803 | 0.352 | 0.408 | 0.695 | 4096.000 |
| raw_hidden | SU | 0.891 | 0.513 | 0.495 | 0.740 | 4096.000 |
| sae_top_cohens_d_n000 | AF | 0.500 | 0.056 | 0.000 | 0.500 | 0.000 |
| sae_top_cohens_d_n000 | GI | 0.500 | 0.110 | 0.000 | 0.500 | 0.000 |
| sae_top_cohens_d_n000 | QU | 0.500 | 0.319 | 0.000 | 0.500 | 0.000 |
| sae_top_cohens_d_n000 | QUC | 0.500 | 0.124 | 0.000 | 0.500 | 0.000 |
| sae_top_cohens_d_n000 | QUO | 0.500 | 0.195 | 0.000 | 0.500 | 0.000 |
| sae_top_cohens_d_n000 | RE | 0.500 | 0.219 | 0.000 | 0.500 | 0.000 |
| sae_top_cohens_d_n000 | REC | 0.500 | 0.136 | 0.000 | 0.500 | 0.000 |
| sae_top_cohens_d_n000 | RES | 0.500 | 0.083 | 0.000 | 0.500 | 0.000 |
| sae_top_cohens_d_n000 | SU | 0.500 | 0.036 | 0.000 | 0.500 | 0.000 |
| sae_top_cohens_d_n001 | AF | 0.800 | 0.390 | 0.369 | 0.780 | 1.000 |
| sae_top_cohens_d_n001 | GI | 0.544 | 0.167 | 0.167 | 0.544 | 1.000 |
| sae_top_cohens_d_n001 | QU | 0.925 | 0.861 | 0.824 | 0.875 | 1.000 |
| sae_top_cohens_d_n001 | QUC | 0.740 | 0.397 | 0.475 | 0.726 | 1.000 |
| sae_top_cohens_d_n001 | QUO | 0.805 | 0.544 | 0.674 | 0.799 | 1.000 |
| sae_top_cohens_d_n001 | RE | 0.675 | 0.390 | 0.479 | 0.668 | 1.000 |
| sae_top_cohens_d_n001 | REC | 0.621 | 0.247 | 0.338 | 0.617 | 1.000 |
| sae_top_cohens_d_n001 | RES | 0.626 | 0.156 | 0.255 | 0.618 | 1.000 |
| sae_top_cohens_d_n001 | SU | 0.526 | 0.060 | 0.082 | 0.526 | 1.000 |
| sae_top_cohens_d_n002 | AF | 0.831 | 0.453 | 0.412 | 0.802 | 2.000 |
| sae_top_cohens_d_n002 | GI | 0.562 | 0.177 | 0.216 | 0.561 | 2.000 |
| sae_top_cohens_d_n002 | QU | 0.944 | 0.904 | 0.860 | 0.900 | 2.000 |
| sae_top_cohens_d_n002 | QUC | 0.848 | 0.488 | 0.483 | 0.784 | 2.000 |
| sae_top_cohens_d_n002 | QUO | 0.896 | 0.641 | 0.694 | 0.836 | 2.000 |
| sae_top_cohens_d_n002 | RE | 0.739 | 0.483 | 0.515 | 0.697 | 2.000 |
| sae_top_cohens_d_n002 | REC | 0.690 | 0.341 | 0.419 | 0.673 | 2.000 |
| sae_top_cohens_d_n002 | RES | 0.647 | 0.163 | 0.234 | 0.611 | 2.000 |
| sae_top_cohens_d_n002 | SU | 0.558 | 0.072 | 0.132 | 0.556 | 2.000 |
| sae_top_cohens_d_n003 | AF | 0.866 | 0.521 | 0.461 | 0.828 | 3.000 |
| sae_top_cohens_d_n003 | GI | 0.571 | 0.190 | 0.239 | 0.570 | 3.000 |
| sae_top_cohens_d_n003 | QU | 0.947 | 0.914 | 0.864 | 0.903 | 3.000 |
| sae_top_cohens_d_n003 | QUC | 0.852 | 0.518 | 0.489 | 0.787 | 3.000 |
| sae_top_cohens_d_n003 | QUO | 0.920 | 0.668 | 0.714 | 0.860 | 3.000 |
| sae_top_cohens_d_n003 | RE | 0.746 | 0.491 | 0.512 | 0.696 | 3.000 |
| sae_top_cohens_d_n003 | REC | 0.728 | 0.391 | 0.447 | 0.701 | 3.000 |
| sae_top_cohens_d_n003 | RES | 0.657 | 0.178 | 0.262 | 0.628 | 3.000 |
| sae_top_cohens_d_n003 | SU | 0.567 | 0.090 | 0.144 | 0.567 | 3.000 |
| sae_top_cohens_d_n004 | AF | 0.876 | 0.541 | 0.495 | 0.832 | 4.000 |
| sae_top_cohens_d_n004 | GI | 0.588 | 0.203 | 0.267 | 0.586 | 4.000 |
| sae_top_cohens_d_n004 | QU | 0.948 | 0.915 | 0.866 | 0.903 | 4.000 |
| sae_top_cohens_d_n004 | QUC | 0.855 | 0.531 | 0.489 | 0.785 | 4.000 |
| sae_top_cohens_d_n004 | QUO | 0.926 | 0.709 | 0.726 | 0.868 | 4.000 |
| sae_top_cohens_d_n004 | RE | 0.749 | 0.490 | 0.513 | 0.697 | 4.000 |
| sae_top_cohens_d_n004 | REC | 0.764 | 0.450 | 0.465 | 0.717 | 4.000 |
| sae_top_cohens_d_n004 | RES | 0.681 | 0.196 | 0.271 | 0.637 | 4.000 |
| sae_top_cohens_d_n004 | SU | 0.581 | 0.111 | 0.154 | 0.580 | 4.000 |
| sae_top_cohens_d_n005 | AF | 0.877 | 0.567 | 0.506 | 0.837 | 5.000 |
| sae_top_cohens_d_n005 | GI | 0.597 | 0.215 | 0.279 | 0.593 | 5.000 |
| sae_top_cohens_d_n005 | QU | 0.952 | 0.922 | 0.868 | 0.905 | 5.000 |
| sae_top_cohens_d_n005 | QUC | 0.855 | 0.533 | 0.487 | 0.783 | 5.000 |
| sae_top_cohens_d_n005 | QUO | 0.927 | 0.717 | 0.726 | 0.867 | 5.000 |
| sae_top_cohens_d_n005 | RE | 0.754 | 0.499 | 0.519 | 0.701 | 5.000 |
| sae_top_cohens_d_n005 | REC | 0.781 | 0.471 | 0.480 | 0.730 | 5.000 |
| sae_top_cohens_d_n005 | RES | 0.698 | 0.197 | 0.275 | 0.649 | 5.000 |
| sae_top_cohens_d_n005 | SU | 0.613 | 0.134 | 0.171 | 0.599 | 5.000 |
| sae_top_cohens_d_n006 | AF | 0.881 | 0.578 | 0.506 | 0.835 | 6.000 |
| sae_top_cohens_d_n006 | GI | 0.606 | 0.229 | 0.291 | 0.602 | 6.000 |
| sae_top_cohens_d_n006 | QU | 0.954 | 0.925 | 0.871 | 0.907 | 6.000 |
| sae_top_cohens_d_n006 | QUC | 0.884 | 0.544 | 0.521 | 0.798 | 6.000 |
| sae_top_cohens_d_n006 | QUO | 0.933 | 0.739 | 0.737 | 0.873 | 6.000 |
| sae_top_cohens_d_n006 | RE | 0.765 | 0.519 | 0.531 | 0.709 | 6.000 |
| sae_top_cohens_d_n006 | REC | 0.821 | 0.516 | 0.504 | 0.758 | 6.000 |
| sae_top_cohens_d_n006 | RES | 0.698 | 0.198 | 0.273 | 0.649 | 6.000 |
| sae_top_cohens_d_n006 | SU | 0.633 | 0.150 | 0.189 | 0.622 | 6.000 |
| sae_top_cohens_d_n007 | AF | 0.882 | 0.580 | 0.502 | 0.838 | 7.000 |
| sae_top_cohens_d_n007 | GI | 0.617 | 0.241 | 0.293 | 0.604 | 7.000 |
| sae_top_cohens_d_n007 | QU | 0.954 | 0.926 | 0.872 | 0.907 | 7.000 |
| sae_top_cohens_d_n007 | QUC | 0.893 | 0.562 | 0.535 | 0.810 | 7.000 |
| sae_top_cohens_d_n007 | QUO | 0.935 | 0.749 | 0.736 | 0.872 | 7.000 |
| sae_top_cohens_d_n007 | RE | 0.774 | 0.527 | 0.535 | 0.712 | 7.000 |
| sae_top_cohens_d_n007 | REC | 0.844 | 0.538 | 0.510 | 0.766 | 7.000 |
| sae_top_cohens_d_n007 | RES | 0.711 | 0.210 | 0.269 | 0.661 | 7.000 |
| sae_top_cohens_d_n007 | SU | 0.643 | 0.166 | 0.193 | 0.631 | 7.000 |
| sae_top_cohens_d_n008 | AF | 0.888 | 0.587 | 0.510 | 0.845 | 8.000 |
| sae_top_cohens_d_n008 | GI | 0.627 | 0.257 | 0.307 | 0.613 | 8.000 |
| sae_top_cohens_d_n008 | QU | 0.955 | 0.927 | 0.873 | 0.908 | 8.000 |
| sae_top_cohens_d_n008 | QUC | 0.893 | 0.560 | 0.535 | 0.810 | 8.000 |
| sae_top_cohens_d_n008 | QUO | 0.938 | 0.761 | 0.743 | 0.877 | 8.000 |
| sae_top_cohens_d_n008 | RE | 0.778 | 0.535 | 0.548 | 0.722 | 8.000 |
| sae_top_cohens_d_n008 | REC | 0.858 | 0.565 | 0.523 | 0.778 | 8.000 |
| sae_top_cohens_d_n008 | RES | 0.727 | 0.220 | 0.284 | 0.677 | 8.000 |
| sae_top_cohens_d_n008 | SU | 0.666 | 0.179 | 0.209 | 0.648 | 8.000 |
| sae_top_cohens_d_n009 | AF | 0.889 | 0.589 | 0.514 | 0.846 | 9.000 |
| sae_top_cohens_d_n009 | GI | 0.629 | 0.266 | 0.311 | 0.616 | 9.000 |
| sae_top_cohens_d_n009 | QU | 0.956 | 0.930 | 0.879 | 0.912 | 9.000 |
| sae_top_cohens_d_n009 | QUC | 0.893 | 0.559 | 0.535 | 0.810 | 9.000 |
| sae_top_cohens_d_n009 | QUO | 0.938 | 0.767 | 0.738 | 0.873 | 9.000 |
| sae_top_cohens_d_n009 | RE | 0.780 | 0.541 | 0.556 | 0.726 | 9.000 |
| sae_top_cohens_d_n009 | REC | 0.866 | 0.581 | 0.536 | 0.789 | 9.000 |
| sae_top_cohens_d_n009 | RES | 0.732 | 0.232 | 0.284 | 0.681 | 9.000 |
| sae_top_cohens_d_n009 | SU | 0.681 | 0.198 | 0.213 | 0.657 | 9.000 |
| sae_top_cohens_d_n010 | AF | 0.892 | 0.592 | 0.518 | 0.847 | 10.000 |
| sae_top_cohens_d_n010 | GI | 0.644 | 0.278 | 0.322 | 0.626 | 10.000 |
| sae_top_cohens_d_n010 | QU | 0.957 | 0.931 | 0.879 | 0.913 | 10.000 |
| sae_top_cohens_d_n010 | QUC | 0.894 | 0.562 | 0.543 | 0.815 | 10.000 |
| sae_top_cohens_d_n010 | QUO | 0.938 | 0.766 | 0.739 | 0.872 | 10.000 |
| sae_top_cohens_d_n010 | RE | 0.789 | 0.554 | 0.563 | 0.731 | 10.000 |
| sae_top_cohens_d_n010 | REC | 0.868 | 0.589 | 0.541 | 0.794 | 10.000 |
| sae_top_cohens_d_n010 | RES | 0.731 | 0.232 | 0.285 | 0.682 | 10.000 |
| sae_top_cohens_d_n010 | SU | 0.687 | 0.213 | 0.209 | 0.665 | 10.000 |
| sae_top_cohens_d_n011 | AF | 0.895 | 0.594 | 0.515 | 0.843 | 11.000 |
| sae_top_cohens_d_n011 | GI | 0.649 | 0.292 | 0.328 | 0.630 | 11.000 |
| sae_top_cohens_d_n011 | QU | 0.957 | 0.932 | 0.884 | 0.916 | 11.000 |
| sae_top_cohens_d_n011 | QUC | 0.894 | 0.561 | 0.542 | 0.814 | 11.000 |
| sae_top_cohens_d_n011 | QUO | 0.939 | 0.775 | 0.744 | 0.877 | 11.000 |
| sae_top_cohens_d_n011 | RE | 0.796 | 0.566 | 0.566 | 0.734 | 11.000 |
| sae_top_cohens_d_n011 | REC | 0.872 | 0.590 | 0.536 | 0.790 | 11.000 |
| sae_top_cohens_d_n011 | RES | 0.735 | 0.235 | 0.270 | 0.671 | 11.000 |
| sae_top_cohens_d_n011 | SU | 0.716 | 0.242 | 0.203 | 0.667 | 11.000 |
| sae_top_cohens_d_n012 | AF | 0.895 | 0.598 | 0.518 | 0.847 | 12.000 |
| sae_top_cohens_d_n012 | GI | 0.652 | 0.298 | 0.324 | 0.630 | 12.000 |
| sae_top_cohens_d_n012 | QU | 0.958 | 0.931 | 0.884 | 0.916 | 12.000 |
| sae_top_cohens_d_n012 | QUC | 0.895 | 0.568 | 0.543 | 0.814 | 12.000 |
| sae_top_cohens_d_n012 | QUO | 0.939 | 0.780 | 0.741 | 0.875 | 12.000 |
| sae_top_cohens_d_n012 | RE | 0.802 | 0.584 | 0.576 | 0.743 | 12.000 |
| sae_top_cohens_d_n012 | REC | 0.875 | 0.593 | 0.535 | 0.791 | 12.000 |
| sae_top_cohens_d_n012 | RES | 0.757 | 0.256 | 0.290 | 0.696 | 12.000 |
| sae_top_cohens_d_n012 | SU | 0.756 | 0.254 | 0.215 | 0.697 | 12.000 |
| sae_top_cohens_d_n013 | AF | 0.897 | 0.597 | 0.516 | 0.844 | 13.000 |
| sae_top_cohens_d_n013 | GI | 0.656 | 0.303 | 0.325 | 0.633 | 13.000 |
| sae_top_cohens_d_n013 | QU | 0.959 | 0.931 | 0.885 | 0.916 | 13.000 |
| sae_top_cohens_d_n013 | QUC | 0.896 | 0.571 | 0.548 | 0.818 | 13.000 |
| sae_top_cohens_d_n013 | QUO | 0.940 | 0.781 | 0.742 | 0.875 | 13.000 |
| sae_top_cohens_d_n013 | RE | 0.807 | 0.587 | 0.579 | 0.745 | 13.000 |
| sae_top_cohens_d_n013 | REC | 0.876 | 0.595 | 0.540 | 0.795 | 13.000 |
| sae_top_cohens_d_n013 | RES | 0.763 | 0.261 | 0.296 | 0.704 | 13.000 |
| sae_top_cohens_d_n013 | SU | 0.763 | 0.266 | 0.230 | 0.700 | 13.000 |
| sae_top_cohens_d_n014 | AF | 0.902 | 0.600 | 0.513 | 0.842 | 14.000 |
| sae_top_cohens_d_n014 | GI | 0.666 | 0.321 | 0.337 | 0.643 | 14.000 |
| sae_top_cohens_d_n014 | QU | 0.959 | 0.932 | 0.884 | 0.915 | 14.000 |
| sae_top_cohens_d_n014 | QUC | 0.895 | 0.574 | 0.548 | 0.817 | 14.000 |
| sae_top_cohens_d_n014 | QUO | 0.941 | 0.794 | 0.744 | 0.877 | 14.000 |
| sae_top_cohens_d_n014 | RE | 0.811 | 0.596 | 0.574 | 0.741 | 14.000 |
| sae_top_cohens_d_n014 | REC | 0.876 | 0.599 | 0.544 | 0.799 | 14.000 |
| sae_top_cohens_d_n014 | RES | 0.763 | 0.260 | 0.294 | 0.701 | 14.000 |
| sae_top_cohens_d_n014 | SU | 0.759 | 0.270 | 0.237 | 0.713 | 14.000 |
| sae_top_cohens_d_n015 | AF | 0.901 | 0.599 | 0.519 | 0.846 | 15.000 |
| sae_top_cohens_d_n015 | GI | 0.672 | 0.327 | 0.330 | 0.642 | 15.000 |
| sae_top_cohens_d_n015 | QU | 0.959 | 0.932 | 0.885 | 0.916 | 15.000 |
| sae_top_cohens_d_n015 | QUC | 0.896 | 0.580 | 0.550 | 0.819 | 15.000 |
| sae_top_cohens_d_n015 | QUO | 0.941 | 0.797 | 0.747 | 0.878 | 15.000 |
| sae_top_cohens_d_n015 | RE | 0.813 | 0.601 | 0.580 | 0.746 | 15.000 |
| sae_top_cohens_d_n015 | REC | 0.876 | 0.599 | 0.543 | 0.799 | 15.000 |
| sae_top_cohens_d_n015 | RES | 0.766 | 0.261 | 0.296 | 0.703 | 15.000 |
| sae_top_cohens_d_n015 | SU | 0.762 | 0.276 | 0.251 | 0.731 | 15.000 |
| sae_top_cohens_d_n016 | AF | 0.901 | 0.590 | 0.510 | 0.843 | 16.000 |
| sae_top_cohens_d_n016 | GI | 0.681 | 0.341 | 0.340 | 0.652 | 16.000 |
| sae_top_cohens_d_n016 | QU | 0.959 | 0.932 | 0.887 | 0.917 | 16.000 |
| sae_top_cohens_d_n016 | QUC | 0.897 | 0.578 | 0.552 | 0.821 | 16.000 |
| sae_top_cohens_d_n016 | QUO | 0.941 | 0.798 | 0.746 | 0.878 | 16.000 |
| sae_top_cohens_d_n016 | RE | 0.814 | 0.603 | 0.584 | 0.749 | 16.000 |
| sae_top_cohens_d_n016 | REC | 0.879 | 0.602 | 0.548 | 0.803 | 16.000 |
| sae_top_cohens_d_n016 | RES | 0.768 | 0.263 | 0.301 | 0.708 | 16.000 |
| sae_top_cohens_d_n016 | SU | 0.762 | 0.284 | 0.244 | 0.726 | 16.000 |
| sae_top_cohens_d_n017 | AF | 0.902 | 0.591 | 0.511 | 0.843 | 17.000 |
| sae_top_cohens_d_n017 | GI | 0.689 | 0.345 | 0.338 | 0.651 | 17.000 |
| sae_top_cohens_d_n017 | QU | 0.960 | 0.934 | 0.887 | 0.917 | 17.000 |
| sae_top_cohens_d_n017 | QUC | 0.899 | 0.588 | 0.560 | 0.826 | 17.000 |
| sae_top_cohens_d_n017 | QUO | 0.941 | 0.804 | 0.744 | 0.877 | 17.000 |
| sae_top_cohens_d_n017 | RE | 0.816 | 0.605 | 0.583 | 0.749 | 17.000 |
| sae_top_cohens_d_n017 | REC | 0.880 | 0.605 | 0.545 | 0.802 | 17.000 |
| sae_top_cohens_d_n017 | RES | 0.766 | 0.261 | 0.299 | 0.704 | 17.000 |
| sae_top_cohens_d_n017 | SU | 0.753 | 0.278 | 0.245 | 0.730 | 17.000 |
| sae_top_cohens_d_n018 | AF | 0.901 | 0.590 | 0.514 | 0.845 | 18.000 |
| sae_top_cohens_d_n018 | GI | 0.687 | 0.347 | 0.338 | 0.651 | 18.000 |
| sae_top_cohens_d_n018 | QU | 0.962 | 0.937 | 0.891 | 0.920 | 18.000 |
| sae_top_cohens_d_n018 | QUC | 0.901 | 0.596 | 0.565 | 0.828 | 18.000 |
| sae_top_cohens_d_n018 | QUO | 0.943 | 0.807 | 0.746 | 0.878 | 18.000 |
| sae_top_cohens_d_n018 | RE | 0.818 | 0.607 | 0.586 | 0.751 | 18.000 |
| sae_top_cohens_d_n018 | REC | 0.879 | 0.606 | 0.549 | 0.805 | 18.000 |
| sae_top_cohens_d_n018 | RES | 0.767 | 0.260 | 0.298 | 0.705 | 18.000 |
| sae_top_cohens_d_n018 | SU | 0.760 | 0.280 | 0.243 | 0.729 | 18.000 |
| sae_top_cohens_d_n019 | AF | 0.899 | 0.579 | 0.509 | 0.842 | 19.000 |
| sae_top_cohens_d_n019 | GI | 0.690 | 0.344 | 0.342 | 0.654 | 19.000 |
| sae_top_cohens_d_n019 | QU | 0.966 | 0.941 | 0.895 | 0.924 | 19.000 |
| sae_top_cohens_d_n019 | QUC | 0.901 | 0.598 | 0.566 | 0.827 | 19.000 |
| sae_top_cohens_d_n019 | QUO | 0.943 | 0.807 | 0.746 | 0.878 | 19.000 |
| sae_top_cohens_d_n019 | RE | 0.822 | 0.612 | 0.593 | 0.756 | 19.000 |
| sae_top_cohens_d_n019 | REC | 0.881 | 0.608 | 0.550 | 0.805 | 19.000 |
| sae_top_cohens_d_n019 | RES | 0.769 | 0.264 | 0.299 | 0.705 | 19.000 |
| sae_top_cohens_d_n019 | SU | 0.759 | 0.274 | 0.240 | 0.728 | 19.000 |
| sae_top_cohens_d_n020 | AF | 0.899 | 0.583 | 0.509 | 0.840 | 20.000 |
| sae_top_cohens_d_n020 | GI | 0.693 | 0.346 | 0.347 | 0.657 | 20.000 |
| sae_top_cohens_d_n020 | QU | 0.965 | 0.940 | 0.895 | 0.924 | 20.000 |
| sae_top_cohens_d_n020 | QUC | 0.901 | 0.597 | 0.566 | 0.828 | 20.000 |
| sae_top_cohens_d_n020 | QUO | 0.943 | 0.806 | 0.747 | 0.879 | 20.000 |
| sae_top_cohens_d_n020 | RE | 0.823 | 0.617 | 0.594 | 0.757 | 20.000 |
| sae_top_cohens_d_n020 | REC | 0.881 | 0.608 | 0.553 | 0.807 | 20.000 |
| sae_top_cohens_d_n020 | RES | 0.771 | 0.266 | 0.298 | 0.704 | 20.000 |
| sae_top_cohens_d_n020 | SU | 0.764 | 0.278 | 0.239 | 0.726 | 20.000 |
| sae_top_cohens_d_n021 | AF | 0.902 | 0.583 | 0.511 | 0.842 | 21.000 |
| sae_top_cohens_d_n021 | GI | 0.695 | 0.348 | 0.350 | 0.658 | 21.000 |
| sae_top_cohens_d_n021 | QU | 0.965 | 0.940 | 0.894 | 0.924 | 21.000 |
| sae_top_cohens_d_n021 | QUC | 0.901 | 0.597 | 0.569 | 0.829 | 21.000 |
| sae_top_cohens_d_n021 | QUO | 0.943 | 0.807 | 0.747 | 0.879 | 21.000 |
| sae_top_cohens_d_n021 | RE | 0.824 | 0.620 | 0.596 | 0.759 | 21.000 |
| sae_top_cohens_d_n021 | REC | 0.880 | 0.606 | 0.551 | 0.806 | 21.000 |
| sae_top_cohens_d_n021 | RES | 0.774 | 0.264 | 0.300 | 0.705 | 21.000 |
| sae_top_cohens_d_n021 | SU | 0.765 | 0.276 | 0.238 | 0.728 | 21.000 |
| sae_top_cohens_d_n022 | AF | 0.902 | 0.588 | 0.509 | 0.841 | 22.000 |
| sae_top_cohens_d_n022 | GI | 0.698 | 0.350 | 0.354 | 0.662 | 22.000 |
| sae_top_cohens_d_n022 | QU | 0.965 | 0.940 | 0.893 | 0.923 | 22.000 |
| sae_top_cohens_d_n022 | QUC | 0.905 | 0.606 | 0.571 | 0.830 | 22.000 |
| sae_top_cohens_d_n022 | QUO | 0.942 | 0.808 | 0.744 | 0.876 | 22.000 |
| sae_top_cohens_d_n022 | RE | 0.824 | 0.621 | 0.595 | 0.758 | 22.000 |
| sae_top_cohens_d_n022 | REC | 0.881 | 0.606 | 0.548 | 0.804 | 22.000 |
| sae_top_cohens_d_n022 | RES | 0.775 | 0.263 | 0.299 | 0.702 | 22.000 |
| sae_top_cohens_d_n022 | SU | 0.767 | 0.275 | 0.232 | 0.725 | 22.000 |
| sae_top_cohens_d_n023 | AF | 0.901 | 0.589 | 0.514 | 0.844 | 23.000 |
| sae_top_cohens_d_n023 | GI | 0.704 | 0.354 | 0.358 | 0.667 | 23.000 |
| sae_top_cohens_d_n023 | QU | 0.965 | 0.941 | 0.894 | 0.924 | 23.000 |
| sae_top_cohens_d_n023 | QUC | 0.909 | 0.608 | 0.572 | 0.831 | 23.000 |
| sae_top_cohens_d_n023 | QUO | 0.944 | 0.808 | 0.746 | 0.878 | 23.000 |
| sae_top_cohens_d_n023 | RE | 0.824 | 0.627 | 0.596 | 0.757 | 23.000 |
| sae_top_cohens_d_n023 | REC | 0.882 | 0.607 | 0.551 | 0.806 | 23.000 |
| sae_top_cohens_d_n023 | RES | 0.774 | 0.260 | 0.297 | 0.701 | 23.000 |
| sae_top_cohens_d_n023 | SU | 0.764 | 0.277 | 0.233 | 0.727 | 23.000 |
| sae_top_cohens_d_n024 | AF | 0.901 | 0.589 | 0.519 | 0.844 | 24.000 |
| sae_top_cohens_d_n024 | GI | 0.706 | 0.356 | 0.360 | 0.668 | 24.000 |
| sae_top_cohens_d_n024 | QU | 0.965 | 0.942 | 0.894 | 0.924 | 24.000 |
| sae_top_cohens_d_n024 | QUC | 0.914 | 0.632 | 0.592 | 0.843 | 24.000 |
| sae_top_cohens_d_n024 | QUO | 0.944 | 0.807 | 0.743 | 0.874 | 24.000 |
| sae_top_cohens_d_n024 | RE | 0.826 | 0.629 | 0.594 | 0.756 | 24.000 |
| sae_top_cohens_d_n024 | REC | 0.885 | 0.611 | 0.556 | 0.809 | 24.000 |
| sae_top_cohens_d_n024 | RES | 0.777 | 0.263 | 0.298 | 0.702 | 24.000 |
| sae_top_cohens_d_n024 | SU | 0.760 | 0.282 | 0.237 | 0.728 | 24.000 |
| sae_top_cohens_d_n025 | AF | 0.902 | 0.595 | 0.526 | 0.851 | 25.000 |
| sae_top_cohens_d_n025 | GI | 0.704 | 0.352 | 0.358 | 0.666 | 25.000 |
| sae_top_cohens_d_n025 | QU | 0.965 | 0.942 | 0.896 | 0.926 | 25.000 |
| sae_top_cohens_d_n025 | QUC | 0.916 | 0.638 | 0.587 | 0.838 | 25.000 |
| sae_top_cohens_d_n025 | QUO | 0.944 | 0.808 | 0.744 | 0.875 | 25.000 |
| sae_top_cohens_d_n025 | RE | 0.826 | 0.627 | 0.593 | 0.756 | 25.000 |
| sae_top_cohens_d_n025 | REC | 0.885 | 0.613 | 0.556 | 0.810 | 25.000 |
| sae_top_cohens_d_n025 | RES | 0.776 | 0.261 | 0.299 | 0.701 | 25.000 |
| sae_top_cohens_d_n025 | SU | 0.758 | 0.278 | 0.233 | 0.722 | 25.000 |
| sae_top_cohens_d_n026 | AF | 0.905 | 0.597 | 0.521 | 0.852 | 26.000 |
| sae_top_cohens_d_n026 | GI | 0.704 | 0.353 | 0.359 | 0.667 | 26.000 |
| sae_top_cohens_d_n026 | QU | 0.966 | 0.942 | 0.898 | 0.926 | 26.000 |
| sae_top_cohens_d_n026 | QUC | 0.917 | 0.638 | 0.593 | 0.842 | 26.000 |
| sae_top_cohens_d_n026 | QUO | 0.944 | 0.813 | 0.748 | 0.877 | 26.000 |
| sae_top_cohens_d_n026 | RE | 0.829 | 0.631 | 0.599 | 0.761 | 26.000 |
| sae_top_cohens_d_n026 | REC | 0.888 | 0.616 | 0.558 | 0.811 | 26.000 |
| sae_top_cohens_d_n026 | RES | 0.779 | 0.266 | 0.305 | 0.708 | 26.000 |
| sae_top_cohens_d_n026 | SU | 0.760 | 0.269 | 0.221 | 0.713 | 26.000 |
| sae_top_cohens_d_n027 | AF | 0.903 | 0.597 | 0.523 | 0.853 | 27.000 |
| sae_top_cohens_d_n027 | GI | 0.706 | 0.357 | 0.360 | 0.668 | 27.000 |
| sae_top_cohens_d_n027 | QU | 0.966 | 0.942 | 0.897 | 0.926 | 27.000 |
| sae_top_cohens_d_n027 | QUC | 0.917 | 0.639 | 0.595 | 0.843 | 27.000 |
| sae_top_cohens_d_n027 | QUO | 0.944 | 0.814 | 0.745 | 0.875 | 27.000 |
| sae_top_cohens_d_n027 | RE | 0.831 | 0.632 | 0.594 | 0.758 | 27.000 |
| sae_top_cohens_d_n027 | REC | 0.889 | 0.621 | 0.556 | 0.810 | 27.000 |
| sae_top_cohens_d_n027 | RES | 0.779 | 0.266 | 0.307 | 0.711 | 27.000 |
| sae_top_cohens_d_n027 | SU | 0.750 | 0.270 | 0.224 | 0.716 | 27.000 |
| sae_top_cohens_d_n028 | AF | 0.903 | 0.598 | 0.519 | 0.851 | 28.000 |
| sae_top_cohens_d_n028 | GI | 0.703 | 0.354 | 0.359 | 0.667 | 28.000 |
| sae_top_cohens_d_n028 | QU | 0.966 | 0.943 | 0.896 | 0.925 | 28.000 |
| sae_top_cohens_d_n028 | QUC | 0.918 | 0.643 | 0.590 | 0.841 | 28.000 |
| sae_top_cohens_d_n028 | QUO | 0.946 | 0.816 | 0.748 | 0.878 | 28.000 |
| sae_top_cohens_d_n028 | RE | 0.831 | 0.632 | 0.594 | 0.758 | 28.000 |
| sae_top_cohens_d_n028 | REC | 0.889 | 0.622 | 0.555 | 0.809 | 28.000 |
| sae_top_cohens_d_n028 | RES | 0.779 | 0.271 | 0.305 | 0.709 | 28.000 |
| sae_top_cohens_d_n028 | SU | 0.751 | 0.269 | 0.223 | 0.714 | 28.000 |
| sae_top_cohens_d_n029 | AF | 0.905 | 0.605 | 0.517 | 0.850 | 29.000 |
| sae_top_cohens_d_n029 | GI | 0.702 | 0.353 | 0.355 | 0.665 | 29.000 |
| sae_top_cohens_d_n029 | QU | 0.966 | 0.943 | 0.896 | 0.925 | 29.000 |
| sae_top_cohens_d_n029 | QUC | 0.918 | 0.644 | 0.592 | 0.842 | 29.000 |
| sae_top_cohens_d_n029 | QUO | 0.946 | 0.822 | 0.751 | 0.880 | 29.000 |
| sae_top_cohens_d_n029 | RE | 0.831 | 0.632 | 0.597 | 0.760 | 29.000 |
| sae_top_cohens_d_n029 | REC | 0.889 | 0.621 | 0.554 | 0.807 | 29.000 |
| sae_top_cohens_d_n029 | RES | 0.780 | 0.269 | 0.306 | 0.710 | 29.000 |
| sae_top_cohens_d_n029 | SU | 0.759 | 0.269 | 0.228 | 0.718 | 29.000 |
| sae_top_cohens_d_n030 | AF | 0.908 | 0.607 | 0.513 | 0.848 | 30.000 |
| sae_top_cohens_d_n030 | GI | 0.702 | 0.352 | 0.354 | 0.664 | 30.000 |
| sae_top_cohens_d_n030 | QU | 0.971 | 0.947 | 0.902 | 0.930 | 30.000 |
| sae_top_cohens_d_n030 | QUC | 0.918 | 0.644 | 0.591 | 0.841 | 30.000 |
| sae_top_cohens_d_n030 | QUO | 0.947 | 0.824 | 0.748 | 0.879 | 30.000 |
| sae_top_cohens_d_n030 | RE | 0.835 | 0.632 | 0.603 | 0.764 | 30.000 |
| sae_top_cohens_d_n030 | REC | 0.889 | 0.623 | 0.551 | 0.805 | 30.000 |
| sae_top_cohens_d_n030 | RES | 0.781 | 0.271 | 0.309 | 0.713 | 30.000 |
| sae_top_cohens_d_n030 | SU | 0.768 | 0.286 | 0.223 | 0.711 | 30.000 |
| sae_top_cohens_d_n031 | AF | 0.909 | 0.610 | 0.516 | 0.849 | 31.000 |
| sae_top_cohens_d_n031 | GI | 0.702 | 0.353 | 0.355 | 0.664 | 31.000 |
| sae_top_cohens_d_n031 | QU | 0.971 | 0.948 | 0.902 | 0.930 | 31.000 |
| sae_top_cohens_d_n031 | QUC | 0.918 | 0.644 | 0.595 | 0.843 | 31.000 |
| sae_top_cohens_d_n031 | QUO | 0.947 | 0.826 | 0.751 | 0.880 | 31.000 |
| sae_top_cohens_d_n031 | RE | 0.842 | 0.636 | 0.605 | 0.767 | 31.000 |
| sae_top_cohens_d_n031 | REC | 0.889 | 0.625 | 0.555 | 0.807 | 31.000 |
| sae_top_cohens_d_n031 | RES | 0.780 | 0.269 | 0.309 | 0.712 | 31.000 |
| sae_top_cohens_d_n031 | SU | 0.779 | 0.295 | 0.221 | 0.715 | 31.000 |
| sae_top_cohens_d_n032 | AF | 0.908 | 0.611 | 0.518 | 0.849 | 32.000 |
| sae_top_cohens_d_n032 | GI | 0.708 | 0.358 | 0.363 | 0.670 | 32.000 |
| sae_top_cohens_d_n032 | QU | 0.971 | 0.948 | 0.902 | 0.930 | 32.000 |
| sae_top_cohens_d_n032 | QUC | 0.918 | 0.643 | 0.591 | 0.840 | 32.000 |
| sae_top_cohens_d_n032 | QUO | 0.947 | 0.826 | 0.750 | 0.879 | 32.000 |
| sae_top_cohens_d_n032 | RE | 0.843 | 0.638 | 0.607 | 0.768 | 32.000 |
| sae_top_cohens_d_n032 | REC | 0.891 | 0.628 | 0.558 | 0.810 | 32.000 |
| sae_top_cohens_d_n032 | RES | 0.780 | 0.269 | 0.309 | 0.711 | 32.000 |
| sae_top_cohens_d_n032 | SU | 0.787 | 0.298 | 0.221 | 0.715 | 32.000 |
| sae_top_cohens_d_n033 | AF | 0.916 | 0.614 | 0.520 | 0.851 | 33.000 |
| sae_top_cohens_d_n033 | GI | 0.706 | 0.363 | 0.365 | 0.672 | 33.000 |
| sae_top_cohens_d_n033 | QU | 0.972 | 0.949 | 0.902 | 0.930 | 33.000 |
| sae_top_cohens_d_n033 | QUC | 0.917 | 0.646 | 0.596 | 0.844 | 33.000 |
| sae_top_cohens_d_n033 | QUO | 0.947 | 0.826 | 0.749 | 0.878 | 33.000 |
| sae_top_cohens_d_n033 | RE | 0.843 | 0.640 | 0.611 | 0.771 | 33.000 |
| sae_top_cohens_d_n033 | REC | 0.892 | 0.631 | 0.561 | 0.812 | 33.000 |
| sae_top_cohens_d_n033 | RES | 0.779 | 0.270 | 0.306 | 0.707 | 33.000 |
| sae_top_cohens_d_n033 | SU | 0.793 | 0.305 | 0.225 | 0.723 | 33.000 |
| sae_top_cohens_d_n034 | AF | 0.917 | 0.611 | 0.522 | 0.852 | 34.000 |
| sae_top_cohens_d_n034 | GI | 0.714 | 0.363 | 0.368 | 0.675 | 34.000 |
| sae_top_cohens_d_n034 | QU | 0.972 | 0.949 | 0.905 | 0.933 | 34.000 |
| sae_top_cohens_d_n034 | QUC | 0.917 | 0.644 | 0.601 | 0.847 | 34.000 |
| sae_top_cohens_d_n034 | QUO | 0.948 | 0.831 | 0.753 | 0.881 | 34.000 |
| sae_top_cohens_d_n034 | RE | 0.844 | 0.640 | 0.613 | 0.773 | 34.000 |
| sae_top_cohens_d_n034 | REC | 0.892 | 0.632 | 0.560 | 0.811 | 34.000 |
| sae_top_cohens_d_n034 | RES | 0.779 | 0.271 | 0.308 | 0.710 | 34.000 |
| sae_top_cohens_d_n034 | SU | 0.797 | 0.312 | 0.228 | 0.726 | 34.000 |
| sae_top_cohens_d_n035 | AF | 0.915 | 0.611 | 0.520 | 0.852 | 35.000 |
| sae_top_cohens_d_n035 | GI | 0.714 | 0.368 | 0.371 | 0.676 | 35.000 |
| sae_top_cohens_d_n035 | QU | 0.972 | 0.949 | 0.905 | 0.933 | 35.000 |
| sae_top_cohens_d_n035 | QUC | 0.918 | 0.644 | 0.597 | 0.845 | 35.000 |
| sae_top_cohens_d_n035 | QUO | 0.949 | 0.837 | 0.756 | 0.883 | 35.000 |
| sae_top_cohens_d_n035 | RE | 0.844 | 0.641 | 0.615 | 0.774 | 35.000 |
| sae_top_cohens_d_n035 | REC | 0.892 | 0.633 | 0.559 | 0.810 | 35.000 |
| sae_top_cohens_d_n035 | RES | 0.780 | 0.272 | 0.307 | 0.710 | 35.000 |
| sae_top_cohens_d_n035 | SU | 0.802 | 0.307 | 0.227 | 0.726 | 35.000 |
| sae_top_cohens_d_n036 | AF | 0.915 | 0.609 | 0.519 | 0.852 | 36.000 |
| sae_top_cohens_d_n036 | GI | 0.716 | 0.368 | 0.371 | 0.677 | 36.000 |
| sae_top_cohens_d_n036 | QU | 0.971 | 0.949 | 0.905 | 0.932 | 36.000 |
| sae_top_cohens_d_n036 | QUC | 0.918 | 0.641 | 0.597 | 0.845 | 36.000 |
| sae_top_cohens_d_n036 | QUO | 0.951 | 0.840 | 0.756 | 0.882 | 36.000 |
| sae_top_cohens_d_n036 | RE | 0.844 | 0.642 | 0.612 | 0.772 | 36.000 |
| sae_top_cohens_d_n036 | REC | 0.893 | 0.634 | 0.557 | 0.810 | 36.000 |
| sae_top_cohens_d_n036 | RES | 0.780 | 0.272 | 0.305 | 0.707 | 36.000 |
| sae_top_cohens_d_n036 | SU | 0.812 | 0.313 | 0.230 | 0.730 | 36.000 |
| sae_top_cohens_d_n037 | AF | 0.915 | 0.608 | 0.523 | 0.856 | 37.000 |
| sae_top_cohens_d_n037 | GI | 0.714 | 0.369 | 0.372 | 0.679 | 37.000 |
| sae_top_cohens_d_n037 | QU | 0.971 | 0.948 | 0.904 | 0.932 | 37.000 |
| sae_top_cohens_d_n037 | QUC | 0.917 | 0.643 | 0.596 | 0.842 | 37.000 |
| sae_top_cohens_d_n037 | QUO | 0.950 | 0.840 | 0.758 | 0.882 | 37.000 |
| sae_top_cohens_d_n037 | RE | 0.844 | 0.642 | 0.613 | 0.773 | 37.000 |
| sae_top_cohens_d_n037 | REC | 0.894 | 0.636 | 0.559 | 0.811 | 37.000 |
| sae_top_cohens_d_n037 | RES | 0.780 | 0.272 | 0.307 | 0.709 | 37.000 |
| sae_top_cohens_d_n037 | SU | 0.814 | 0.310 | 0.238 | 0.736 | 37.000 |
| sae_top_cohens_d_n038 | AF | 0.918 | 0.607 | 0.516 | 0.855 | 38.000 |
| sae_top_cohens_d_n038 | GI | 0.710 | 0.368 | 0.372 | 0.677 | 38.000 |
| sae_top_cohens_d_n038 | QU | 0.971 | 0.948 | 0.904 | 0.932 | 38.000 |
| sae_top_cohens_d_n038 | QUC | 0.919 | 0.647 | 0.599 | 0.843 | 38.000 |
| sae_top_cohens_d_n038 | QUO | 0.950 | 0.838 | 0.758 | 0.881 | 38.000 |
| sae_top_cohens_d_n038 | RE | 0.844 | 0.643 | 0.611 | 0.772 | 38.000 |
| sae_top_cohens_d_n038 | REC | 0.894 | 0.636 | 0.558 | 0.810 | 38.000 |
| sae_top_cohens_d_n038 | RES | 0.779 | 0.272 | 0.307 | 0.709 | 38.000 |
| sae_top_cohens_d_n038 | SU | 0.814 | 0.325 | 0.242 | 0.743 | 38.000 |
| sae_top_cohens_d_n039 | AF | 0.919 | 0.602 | 0.514 | 0.854 | 39.000 |
| sae_top_cohens_d_n039 | GI | 0.715 | 0.374 | 0.378 | 0.683 | 39.000 |
| sae_top_cohens_d_n039 | QU | 0.971 | 0.948 | 0.903 | 0.931 | 39.000 |
| sae_top_cohens_d_n039 | QUC | 0.919 | 0.648 | 0.599 | 0.844 | 39.000 |
| sae_top_cohens_d_n039 | QUO | 0.950 | 0.838 | 0.756 | 0.880 | 39.000 |
| sae_top_cohens_d_n039 | RE | 0.844 | 0.642 | 0.608 | 0.769 | 39.000 |
| sae_top_cohens_d_n039 | REC | 0.894 | 0.636 | 0.564 | 0.815 | 39.000 |
| sae_top_cohens_d_n039 | RES | 0.781 | 0.272 | 0.308 | 0.710 | 39.000 |
| sae_top_cohens_d_n039 | SU | 0.816 | 0.329 | 0.244 | 0.748 | 39.000 |
| sae_top_cohens_d_n040 | AF | 0.919 | 0.603 | 0.515 | 0.854 | 40.000 |
| sae_top_cohens_d_n040 | GI | 0.716 | 0.376 | 0.379 | 0.684 | 40.000 |
| sae_top_cohens_d_n040 | QU | 0.971 | 0.948 | 0.905 | 0.932 | 40.000 |
| sae_top_cohens_d_n040 | QUC | 0.918 | 0.647 | 0.600 | 0.842 | 40.000 |
| sae_top_cohens_d_n040 | QUO | 0.950 | 0.840 | 0.758 | 0.882 | 40.000 |
| sae_top_cohens_d_n040 | RE | 0.845 | 0.642 | 0.609 | 0.770 | 40.000 |
| sae_top_cohens_d_n040 | REC | 0.894 | 0.636 | 0.566 | 0.816 | 40.000 |
| sae_top_cohens_d_n040 | RES | 0.782 | 0.273 | 0.312 | 0.714 | 40.000 |
| sae_top_cohens_d_n040 | SU | 0.818 | 0.333 | 0.240 | 0.744 | 40.000 |
| sae_top_cohens_d_n041 | AF | 0.919 | 0.606 | 0.517 | 0.855 | 41.000 |
| sae_top_cohens_d_n041 | GI | 0.714 | 0.376 | 0.380 | 0.686 | 41.000 |
| sae_top_cohens_d_n041 | QU | 0.971 | 0.949 | 0.906 | 0.933 | 41.000 |
| sae_top_cohens_d_n041 | QUC | 0.920 | 0.649 | 0.601 | 0.844 | 41.000 |
| sae_top_cohens_d_n041 | QUO | 0.950 | 0.840 | 0.754 | 0.879 | 41.000 |
| sae_top_cohens_d_n041 | RE | 0.845 | 0.643 | 0.607 | 0.768 | 41.000 |
| sae_top_cohens_d_n041 | REC | 0.894 | 0.637 | 0.569 | 0.817 | 41.000 |
| sae_top_cohens_d_n041 | RES | 0.783 | 0.274 | 0.313 | 0.717 | 41.000 |
| sae_top_cohens_d_n041 | SU | 0.830 | 0.336 | 0.247 | 0.752 | 41.000 |
| sae_top_cohens_d_n042 | AF | 0.919 | 0.603 | 0.517 | 0.855 | 42.000 |
| sae_top_cohens_d_n042 | GI | 0.718 | 0.376 | 0.381 | 0.688 | 42.000 |
| sae_top_cohens_d_n042 | QU | 0.971 | 0.949 | 0.906 | 0.933 | 42.000 |
| sae_top_cohens_d_n042 | QUC | 0.921 | 0.645 | 0.606 | 0.847 | 42.000 |
| sae_top_cohens_d_n042 | QUO | 0.950 | 0.839 | 0.754 | 0.879 | 42.000 |
| sae_top_cohens_d_n042 | RE | 0.846 | 0.643 | 0.608 | 0.769 | 42.000 |
| sae_top_cohens_d_n042 | REC | 0.894 | 0.639 | 0.567 | 0.817 | 42.000 |
| sae_top_cohens_d_n042 | RES | 0.781 | 0.274 | 0.318 | 0.722 | 42.000 |
| sae_top_cohens_d_n042 | SU | 0.828 | 0.335 | 0.246 | 0.750 | 42.000 |
| sae_top_cohens_d_n043 | AF | 0.921 | 0.598 | 0.518 | 0.855 | 43.000 |
| sae_top_cohens_d_n043 | GI | 0.720 | 0.375 | 0.379 | 0.688 | 43.000 |
| sae_top_cohens_d_n043 | QU | 0.972 | 0.949 | 0.906 | 0.933 | 43.000 |
| sae_top_cohens_d_n043 | QUC | 0.920 | 0.642 | 0.610 | 0.849 | 43.000 |
| sae_top_cohens_d_n043 | QUO | 0.950 | 0.840 | 0.754 | 0.878 | 43.000 |
| sae_top_cohens_d_n043 | RE | 0.846 | 0.644 | 0.611 | 0.771 | 43.000 |
| sae_top_cohens_d_n043 | REC | 0.894 | 0.637 | 0.569 | 0.821 | 43.000 |
| sae_top_cohens_d_n043 | RES | 0.782 | 0.275 | 0.320 | 0.724 | 43.000 |
| sae_top_cohens_d_n043 | SU | 0.840 | 0.340 | 0.247 | 0.751 | 43.000 |
| sae_top_cohens_d_n044 | AF | 0.920 | 0.599 | 0.517 | 0.855 | 44.000 |
| sae_top_cohens_d_n044 | GI | 0.722 | 0.377 | 0.381 | 0.689 | 44.000 |
| sae_top_cohens_d_n044 | QU | 0.972 | 0.950 | 0.904 | 0.931 | 44.000 |
| sae_top_cohens_d_n044 | QUC | 0.918 | 0.644 | 0.609 | 0.849 | 44.000 |
| sae_top_cohens_d_n044 | QUO | 0.950 | 0.838 | 0.755 | 0.879 | 44.000 |
| sae_top_cohens_d_n044 | RE | 0.846 | 0.643 | 0.610 | 0.771 | 44.000 |
| sae_top_cohens_d_n044 | REC | 0.894 | 0.636 | 0.570 | 0.820 | 44.000 |
| sae_top_cohens_d_n044 | RES | 0.784 | 0.276 | 0.317 | 0.722 | 44.000 |
| sae_top_cohens_d_n044 | SU | 0.839 | 0.339 | 0.239 | 0.741 | 44.000 |
| sae_top_cohens_d_n045 | AF | 0.919 | 0.599 | 0.511 | 0.853 | 45.000 |
| sae_top_cohens_d_n045 | GI | 0.724 | 0.377 | 0.382 | 0.690 | 45.000 |
| sae_top_cohens_d_n045 | QU | 0.972 | 0.950 | 0.904 | 0.931 | 45.000 |
| sae_top_cohens_d_n045 | QUC | 0.918 | 0.646 | 0.610 | 0.849 | 45.000 |
| sae_top_cohens_d_n045 | QUO | 0.950 | 0.838 | 0.752 | 0.878 | 45.000 |
| sae_top_cohens_d_n045 | RE | 0.846 | 0.645 | 0.612 | 0.773 | 45.000 |
| sae_top_cohens_d_n045 | REC | 0.894 | 0.634 | 0.569 | 0.820 | 45.000 |
| sae_top_cohens_d_n045 | RES | 0.784 | 0.279 | 0.317 | 0.720 | 45.000 |
| sae_top_cohens_d_n045 | SU | 0.839 | 0.334 | 0.243 | 0.744 | 45.000 |
| sae_top_cohens_d_n046 | AF | 0.918 | 0.598 | 0.512 | 0.852 | 46.000 |
| sae_top_cohens_d_n046 | GI | 0.731 | 0.384 | 0.379 | 0.688 | 46.000 |
| sae_top_cohens_d_n046 | QU | 0.972 | 0.949 | 0.905 | 0.932 | 46.000 |
| sae_top_cohens_d_n046 | QUC | 0.918 | 0.648 | 0.611 | 0.850 | 46.000 |
| sae_top_cohens_d_n046 | QUO | 0.950 | 0.838 | 0.750 | 0.876 | 46.000 |
| sae_top_cohens_d_n046 | RE | 0.847 | 0.646 | 0.615 | 0.775 | 46.000 |
| sae_top_cohens_d_n046 | REC | 0.895 | 0.637 | 0.569 | 0.820 | 46.000 |
| sae_top_cohens_d_n046 | RES | 0.784 | 0.279 | 0.316 | 0.719 | 46.000 |
| sae_top_cohens_d_n046 | SU | 0.837 | 0.328 | 0.245 | 0.746 | 46.000 |
| sae_top_cohens_d_n047 | AF | 0.918 | 0.599 | 0.508 | 0.850 | 47.000 |
| sae_top_cohens_d_n047 | GI | 0.731 | 0.383 | 0.378 | 0.687 | 47.000 |
| sae_top_cohens_d_n047 | QU | 0.972 | 0.950 | 0.906 | 0.933 | 47.000 |
| sae_top_cohens_d_n047 | QUC | 0.919 | 0.649 | 0.609 | 0.849 | 47.000 |
| sae_top_cohens_d_n047 | QUO | 0.950 | 0.838 | 0.748 | 0.875 | 47.000 |
| sae_top_cohens_d_n047 | RE | 0.848 | 0.647 | 0.613 | 0.773 | 47.000 |
| sae_top_cohens_d_n047 | REC | 0.895 | 0.637 | 0.567 | 0.819 | 47.000 |
| sae_top_cohens_d_n047 | RES | 0.783 | 0.277 | 0.315 | 0.718 | 47.000 |
| sae_top_cohens_d_n047 | SU | 0.835 | 0.327 | 0.251 | 0.752 | 47.000 |
| sae_top_cohens_d_n048 | AF | 0.921 | 0.600 | 0.514 | 0.855 | 48.000 |
| sae_top_cohens_d_n048 | GI | 0.730 | 0.385 | 0.380 | 0.690 | 48.000 |
| sae_top_cohens_d_n048 | QU | 0.973 | 0.950 | 0.905 | 0.933 | 48.000 |
| sae_top_cohens_d_n048 | QUC | 0.920 | 0.651 | 0.611 | 0.850 | 48.000 |
| sae_top_cohens_d_n048 | QUO | 0.950 | 0.838 | 0.750 | 0.876 | 48.000 |
| sae_top_cohens_d_n048 | RE | 0.847 | 0.647 | 0.611 | 0.771 | 48.000 |
| sae_top_cohens_d_n048 | REC | 0.896 | 0.637 | 0.570 | 0.822 | 48.000 |
| sae_top_cohens_d_n048 | RES | 0.785 | 0.277 | 0.316 | 0.719 | 48.000 |
| sae_top_cohens_d_n048 | SU | 0.836 | 0.320 | 0.251 | 0.752 | 48.000 |
| sae_top_cohens_d_n049 | AF | 0.920 | 0.604 | 0.517 | 0.857 | 49.000 |
| sae_top_cohens_d_n049 | GI | 0.733 | 0.386 | 0.377 | 0.688 | 49.000 |
| sae_top_cohens_d_n049 | QU | 0.973 | 0.951 | 0.906 | 0.933 | 49.000 |
| sae_top_cohens_d_n049 | QUC | 0.919 | 0.651 | 0.613 | 0.851 | 49.000 |
| sae_top_cohens_d_n049 | QUO | 0.951 | 0.839 | 0.752 | 0.877 | 49.000 |
| sae_top_cohens_d_n049 | RE | 0.848 | 0.647 | 0.613 | 0.772 | 49.000 |
| sae_top_cohens_d_n049 | REC | 0.897 | 0.639 | 0.568 | 0.821 | 49.000 |
| sae_top_cohens_d_n049 | RES | 0.785 | 0.281 | 0.317 | 0.719 | 49.000 |
| sae_top_cohens_d_n049 | SU | 0.840 | 0.322 | 0.252 | 0.752 | 49.000 |
| sae_top_cohens_d_n050 | AF | 0.920 | 0.607 | 0.517 | 0.855 | 50.000 |
| sae_top_cohens_d_n050 | GI | 0.732 | 0.386 | 0.374 | 0.686 | 50.000 |
| sae_top_cohens_d_n050 | QU | 0.974 | 0.951 | 0.906 | 0.933 | 50.000 |
| sae_top_cohens_d_n050 | QUC | 0.920 | 0.650 | 0.611 | 0.851 | 50.000 |
| sae_top_cohens_d_n050 | QUO | 0.951 | 0.838 | 0.752 | 0.877 | 50.000 |
| sae_top_cohens_d_n050 | RE | 0.848 | 0.648 | 0.612 | 0.772 | 50.000 |
| sae_top_cohens_d_n050 | REC | 0.899 | 0.642 | 0.568 | 0.821 | 50.000 |
| sae_top_cohens_d_n050 | RES | 0.787 | 0.280 | 0.315 | 0.717 | 50.000 |
| sae_top_cohens_d_n050 | SU | 0.838 | 0.320 | 0.245 | 0.745 | 50.000 |
| sae_top_cohens_d_n051 | AF | 0.920 | 0.608 | 0.517 | 0.856 | 51.000 |
| sae_top_cohens_d_n051 | GI | 0.733 | 0.387 | 0.372 | 0.685 | 51.000 |
| sae_top_cohens_d_n051 | QU | 0.974 | 0.951 | 0.904 | 0.932 | 51.000 |
| sae_top_cohens_d_n051 | QUC | 0.920 | 0.652 | 0.611 | 0.850 | 51.000 |
| sae_top_cohens_d_n051 | QUO | 0.952 | 0.839 | 0.751 | 0.877 | 51.000 |
| sae_top_cohens_d_n051 | RE | 0.848 | 0.647 | 0.613 | 0.773 | 51.000 |
| sae_top_cohens_d_n051 | REC | 0.899 | 0.644 | 0.568 | 0.822 | 51.000 |
| sae_top_cohens_d_n051 | RES | 0.788 | 0.279 | 0.317 | 0.718 | 51.000 |
| sae_top_cohens_d_n051 | SU | 0.835 | 0.321 | 0.251 | 0.752 | 51.000 |
| sae_top_cohens_d_n052 | AF | 0.920 | 0.605 | 0.515 | 0.853 | 52.000 |
| sae_top_cohens_d_n052 | GI | 0.734 | 0.388 | 0.375 | 0.688 | 52.000 |
| sae_top_cohens_d_n052 | QU | 0.974 | 0.951 | 0.904 | 0.932 | 52.000 |
| sae_top_cohens_d_n052 | QUC | 0.920 | 0.656 | 0.613 | 0.851 | 52.000 |
| sae_top_cohens_d_n052 | QUO | 0.952 | 0.840 | 0.753 | 0.878 | 52.000 |
| sae_top_cohens_d_n052 | RE | 0.848 | 0.647 | 0.612 | 0.772 | 52.000 |
| sae_top_cohens_d_n052 | REC | 0.900 | 0.646 | 0.570 | 0.822 | 52.000 |
| sae_top_cohens_d_n052 | RES | 0.788 | 0.282 | 0.314 | 0.715 | 52.000 |
| sae_top_cohens_d_n052 | SU | 0.837 | 0.324 | 0.249 | 0.750 | 52.000 |
| sae_top_cohens_d_n053 | AF | 0.921 | 0.603 | 0.514 | 0.852 | 53.000 |
| sae_top_cohens_d_n053 | GI | 0.734 | 0.389 | 0.376 | 0.690 | 53.000 |
| sae_top_cohens_d_n053 | QU | 0.974 | 0.951 | 0.906 | 0.933 | 53.000 |
| sae_top_cohens_d_n053 | QUC | 0.921 | 0.659 | 0.613 | 0.849 | 53.000 |
| sae_top_cohens_d_n053 | QUO | 0.952 | 0.839 | 0.753 | 0.879 | 53.000 |
| sae_top_cohens_d_n053 | RE | 0.847 | 0.648 | 0.614 | 0.773 | 53.000 |
| sae_top_cohens_d_n053 | REC | 0.900 | 0.646 | 0.571 | 0.822 | 53.000 |
| sae_top_cohens_d_n053 | RES | 0.788 | 0.282 | 0.317 | 0.718 | 53.000 |
| sae_top_cohens_d_n053 | SU | 0.838 | 0.323 | 0.254 | 0.754 | 53.000 |
| sae_top_cohens_d_n054 | AF | 0.920 | 0.601 | 0.512 | 0.849 | 54.000 |
| sae_top_cohens_d_n054 | GI | 0.734 | 0.390 | 0.373 | 0.686 | 54.000 |
| sae_top_cohens_d_n054 | QU | 0.974 | 0.951 | 0.905 | 0.933 | 54.000 |
| sae_top_cohens_d_n054 | QUC | 0.921 | 0.661 | 0.613 | 0.850 | 54.000 |
| sae_top_cohens_d_n054 | QUO | 0.952 | 0.840 | 0.756 | 0.880 | 54.000 |
| sae_top_cohens_d_n054 | RE | 0.847 | 0.647 | 0.612 | 0.772 | 54.000 |
| sae_top_cohens_d_n054 | REC | 0.901 | 0.646 | 0.571 | 0.821 | 54.000 |
| sae_top_cohens_d_n054 | RES | 0.788 | 0.285 | 0.318 | 0.719 | 54.000 |
| sae_top_cohens_d_n054 | SU | 0.838 | 0.336 | 0.256 | 0.759 | 54.000 |
| sae_top_cohens_d_n055 | AF | 0.921 | 0.606 | 0.518 | 0.856 | 55.000 |
| sae_top_cohens_d_n055 | GI | 0.735 | 0.388 | 0.372 | 0.686 | 55.000 |
| sae_top_cohens_d_n055 | QU | 0.974 | 0.951 | 0.906 | 0.933 | 55.000 |
| sae_top_cohens_d_n055 | QUC | 0.921 | 0.660 | 0.617 | 0.851 | 55.000 |
| sae_top_cohens_d_n055 | QUO | 0.953 | 0.841 | 0.758 | 0.881 | 55.000 |
| sae_top_cohens_d_n055 | RE | 0.847 | 0.647 | 0.613 | 0.772 | 55.000 |
| sae_top_cohens_d_n055 | REC | 0.900 | 0.645 | 0.569 | 0.820 | 55.000 |
| sae_top_cohens_d_n055 | RES | 0.789 | 0.284 | 0.319 | 0.718 | 55.000 |
| sae_top_cohens_d_n055 | SU | 0.839 | 0.331 | 0.255 | 0.755 | 55.000 |
| sae_top_cohens_d_n056 | AF | 0.920 | 0.607 | 0.527 | 0.860 | 56.000 |
| sae_top_cohens_d_n056 | GI | 0.735 | 0.389 | 0.370 | 0.684 | 56.000 |
| sae_top_cohens_d_n056 | QU | 0.974 | 0.951 | 0.907 | 0.934 | 56.000 |
| sae_top_cohens_d_n056 | QUC | 0.921 | 0.661 | 0.615 | 0.850 | 56.000 |
| sae_top_cohens_d_n056 | QUO | 0.953 | 0.843 | 0.755 | 0.879 | 56.000 |
| sae_top_cohens_d_n056 | RE | 0.847 | 0.648 | 0.611 | 0.771 | 56.000 |
| sae_top_cohens_d_n056 | REC | 0.902 | 0.646 | 0.568 | 0.820 | 56.000 |
| sae_top_cohens_d_n056 | RES | 0.791 | 0.285 | 0.319 | 0.718 | 56.000 |
| sae_top_cohens_d_n056 | SU | 0.843 | 0.337 | 0.261 | 0.762 | 56.000 |
| sae_top_cohens_d_n057 | AF | 0.918 | 0.605 | 0.518 | 0.856 | 57.000 |
| sae_top_cohens_d_n057 | GI | 0.732 | 0.388 | 0.374 | 0.688 | 57.000 |
| sae_top_cohens_d_n057 | QU | 0.974 | 0.951 | 0.907 | 0.934 | 57.000 |
| sae_top_cohens_d_n057 | QUC | 0.920 | 0.660 | 0.617 | 0.851 | 57.000 |
| sae_top_cohens_d_n057 | QUO | 0.954 | 0.844 | 0.757 | 0.879 | 57.000 |
| sae_top_cohens_d_n057 | RE | 0.847 | 0.647 | 0.612 | 0.772 | 57.000 |
| sae_top_cohens_d_n057 | REC | 0.903 | 0.647 | 0.570 | 0.822 | 57.000 |
| sae_top_cohens_d_n057 | RES | 0.790 | 0.284 | 0.317 | 0.716 | 57.000 |
| sae_top_cohens_d_n057 | SU | 0.846 | 0.340 | 0.269 | 0.770 | 57.000 |
| sae_top_cohens_d_n058 | AF | 0.917 | 0.597 | 0.514 | 0.857 | 58.000 |
| sae_top_cohens_d_n058 | GI | 0.733 | 0.390 | 0.377 | 0.690 | 58.000 |
| sae_top_cohens_d_n058 | QU | 0.974 | 0.951 | 0.907 | 0.934 | 58.000 |
| sae_top_cohens_d_n058 | QUC | 0.920 | 0.662 | 0.617 | 0.850 | 58.000 |
| sae_top_cohens_d_n058 | QUO | 0.954 | 0.845 | 0.758 | 0.880 | 58.000 |
| sae_top_cohens_d_n058 | RE | 0.847 | 0.648 | 0.610 | 0.770 | 58.000 |
| sae_top_cohens_d_n058 | REC | 0.903 | 0.649 | 0.575 | 0.824 | 58.000 |
| sae_top_cohens_d_n058 | RES | 0.792 | 0.282 | 0.320 | 0.720 | 58.000 |
| sae_top_cohens_d_n058 | SU | 0.844 | 0.343 | 0.272 | 0.773 | 58.000 |
| sae_top_cohens_d_n059 | AF | 0.919 | 0.595 | 0.509 | 0.853 | 59.000 |
| sae_top_cohens_d_n059 | GI | 0.733 | 0.392 | 0.379 | 0.690 | 59.000 |
| sae_top_cohens_d_n059 | QU | 0.975 | 0.951 | 0.908 | 0.935 | 59.000 |
| sae_top_cohens_d_n059 | QUC | 0.920 | 0.662 | 0.619 | 0.851 | 59.000 |
| sae_top_cohens_d_n059 | QUO | 0.954 | 0.845 | 0.758 | 0.880 | 59.000 |
| sae_top_cohens_d_n059 | RE | 0.847 | 0.647 | 0.614 | 0.773 | 59.000 |
| sae_top_cohens_d_n059 | REC | 0.903 | 0.649 | 0.569 | 0.820 | 59.000 |
| sae_top_cohens_d_n059 | RES | 0.792 | 0.285 | 0.322 | 0.720 | 59.000 |
| sae_top_cohens_d_n059 | SU | 0.845 | 0.345 | 0.271 | 0.773 | 59.000 |
| sae_top_cohens_d_n060 | AF | 0.918 | 0.593 | 0.507 | 0.852 | 60.000 |
| sae_top_cohens_d_n060 | GI | 0.732 | 0.390 | 0.377 | 0.689 | 60.000 |
| sae_top_cohens_d_n060 | QU | 0.975 | 0.951 | 0.907 | 0.935 | 60.000 |
| sae_top_cohens_d_n060 | QUC | 0.921 | 0.668 | 0.617 | 0.850 | 60.000 |
| sae_top_cohens_d_n060 | QUO | 0.955 | 0.850 | 0.764 | 0.882 | 60.000 |
| sae_top_cohens_d_n060 | RE | 0.847 | 0.647 | 0.612 | 0.772 | 60.000 |
| sae_top_cohens_d_n060 | REC | 0.903 | 0.649 | 0.574 | 0.823 | 60.000 |
| sae_top_cohens_d_n060 | RES | 0.792 | 0.287 | 0.317 | 0.716 | 60.000 |
| sae_top_cohens_d_n060 | SU | 0.852 | 0.344 | 0.277 | 0.780 | 60.000 |
| sae_top_cohens_d_n061 | AF | 0.916 | 0.591 | 0.510 | 0.854 | 61.000 |
| sae_top_cohens_d_n061 | GI | 0.734 | 0.390 | 0.379 | 0.690 | 61.000 |
| sae_top_cohens_d_n061 | QU | 0.975 | 0.951 | 0.907 | 0.935 | 61.000 |
| sae_top_cohens_d_n061 | QUC | 0.920 | 0.669 | 0.616 | 0.849 | 61.000 |
| sae_top_cohens_d_n061 | QUO | 0.955 | 0.851 | 0.763 | 0.883 | 61.000 |
| sae_top_cohens_d_n061 | RE | 0.847 | 0.647 | 0.613 | 0.771 | 61.000 |
| sae_top_cohens_d_n061 | REC | 0.903 | 0.649 | 0.571 | 0.821 | 61.000 |
| sae_top_cohens_d_n061 | RES | 0.792 | 0.286 | 0.320 | 0.719 | 61.000 |
| sae_top_cohens_d_n061 | SU | 0.852 | 0.348 | 0.270 | 0.769 | 61.000 |
| sae_top_cohens_d_n062 | AF | 0.915 | 0.590 | 0.508 | 0.853 | 62.000 |
| sae_top_cohens_d_n062 | GI | 0.733 | 0.390 | 0.375 | 0.688 | 62.000 |
| sae_top_cohens_d_n062 | QU | 0.975 | 0.951 | 0.908 | 0.935 | 62.000 |
| sae_top_cohens_d_n062 | QUC | 0.920 | 0.669 | 0.618 | 0.850 | 62.000 |
| sae_top_cohens_d_n062 | QUO | 0.955 | 0.851 | 0.761 | 0.881 | 62.000 |
| sae_top_cohens_d_n062 | RE | 0.847 | 0.647 | 0.613 | 0.772 | 62.000 |
| sae_top_cohens_d_n062 | REC | 0.905 | 0.650 | 0.575 | 0.825 | 62.000 |
| sae_top_cohens_d_n062 | RES | 0.794 | 0.285 | 0.316 | 0.715 | 62.000 |
| sae_top_cohens_d_n062 | SU | 0.852 | 0.346 | 0.275 | 0.776 | 62.000 |
| sae_top_cohens_d_n063 | AF | 0.915 | 0.588 | 0.505 | 0.853 | 63.000 |
| sae_top_cohens_d_n063 | GI | 0.736 | 0.392 | 0.379 | 0.692 | 63.000 |
| sae_top_cohens_d_n063 | QU | 0.975 | 0.951 | 0.907 | 0.935 | 63.000 |
| sae_top_cohens_d_n063 | QUC | 0.921 | 0.673 | 0.617 | 0.848 | 63.000 |
| sae_top_cohens_d_n063 | QUO | 0.955 | 0.850 | 0.759 | 0.880 | 63.000 |
| sae_top_cohens_d_n063 | RE | 0.847 | 0.646 | 0.614 | 0.773 | 63.000 |
| sae_top_cohens_d_n063 | REC | 0.905 | 0.651 | 0.578 | 0.827 | 63.000 |
| sae_top_cohens_d_n063 | RES | 0.793 | 0.282 | 0.318 | 0.719 | 63.000 |
| sae_top_cohens_d_n063 | SU | 0.851 | 0.350 | 0.274 | 0.774 | 63.000 |
| sae_top_cohens_d_n064 | AF | 0.917 | 0.584 | 0.504 | 0.853 | 64.000 |
| sae_top_cohens_d_n064 | GI | 0.738 | 0.392 | 0.381 | 0.693 | 64.000 |
| sae_top_cohens_d_n064 | QU | 0.974 | 0.951 | 0.908 | 0.935 | 64.000 |
| sae_top_cohens_d_n064 | QUC | 0.921 | 0.672 | 0.619 | 0.851 | 64.000 |
| sae_top_cohens_d_n064 | QUO | 0.955 | 0.850 | 0.761 | 0.881 | 64.000 |
| sae_top_cohens_d_n064 | RE | 0.848 | 0.648 | 0.615 | 0.774 | 64.000 |
| sae_top_cohens_d_n064 | REC | 0.905 | 0.653 | 0.578 | 0.827 | 64.000 |
| sae_top_cohens_d_n064 | RES | 0.792 | 0.281 | 0.321 | 0.722 | 64.000 |
| sae_top_cohens_d_n064 | SU | 0.853 | 0.355 | 0.273 | 0.772 | 64.000 |
| sae_top_cohens_d_n065 | AF | 0.916 | 0.585 | 0.503 | 0.853 | 65.000 |
| sae_top_cohens_d_n065 | GI | 0.738 | 0.394 | 0.381 | 0.693 | 65.000 |
| sae_top_cohens_d_n065 | QU | 0.974 | 0.951 | 0.907 | 0.934 | 65.000 |
| sae_top_cohens_d_n065 | QUC | 0.921 | 0.672 | 0.619 | 0.851 | 65.000 |
| sae_top_cohens_d_n065 | QUO | 0.955 | 0.849 | 0.764 | 0.884 | 65.000 |
| sae_top_cohens_d_n065 | RE | 0.848 | 0.649 | 0.614 | 0.773 | 65.000 |
| sae_top_cohens_d_n065 | REC | 0.905 | 0.654 | 0.580 | 0.827 | 65.000 |
| sae_top_cohens_d_n065 | RES | 0.793 | 0.280 | 0.320 | 0.720 | 65.000 |
| sae_top_cohens_d_n065 | SU | 0.851 | 0.355 | 0.275 | 0.774 | 65.000 |
| sae_top_cohens_d_n066 | AF | 0.915 | 0.587 | 0.506 | 0.852 | 66.000 |
| sae_top_cohens_d_n066 | GI | 0.740 | 0.395 | 0.379 | 0.692 | 66.000 |
| sae_top_cohens_d_n066 | QU | 0.974 | 0.951 | 0.907 | 0.934 | 66.000 |
| sae_top_cohens_d_n066 | QUC | 0.922 | 0.676 | 0.619 | 0.851 | 66.000 |
| sae_top_cohens_d_n066 | QUO | 0.956 | 0.850 | 0.767 | 0.887 | 66.000 |
| sae_top_cohens_d_n066 | RE | 0.849 | 0.650 | 0.615 | 0.773 | 66.000 |
| sae_top_cohens_d_n066 | REC | 0.906 | 0.653 | 0.581 | 0.827 | 66.000 |
| sae_top_cohens_d_n066 | RES | 0.792 | 0.279 | 0.321 | 0.721 | 66.000 |
| sae_top_cohens_d_n066 | SU | 0.850 | 0.358 | 0.271 | 0.770 | 66.000 |
| sae_top_cohens_d_n067 | AF | 0.914 | 0.588 | 0.505 | 0.852 | 67.000 |
| sae_top_cohens_d_n067 | GI | 0.745 | 0.397 | 0.379 | 0.693 | 67.000 |
| sae_top_cohens_d_n067 | QU | 0.974 | 0.951 | 0.907 | 0.934 | 67.000 |
| sae_top_cohens_d_n067 | QUC | 0.921 | 0.675 | 0.615 | 0.847 | 67.000 |
| sae_top_cohens_d_n067 | QUO | 0.956 | 0.851 | 0.766 | 0.886 | 67.000 |
| sae_top_cohens_d_n067 | RE | 0.849 | 0.651 | 0.617 | 0.775 | 67.000 |
| sae_top_cohens_d_n067 | REC | 0.906 | 0.654 | 0.582 | 0.828 | 67.000 |
| sae_top_cohens_d_n067 | RES | 0.792 | 0.280 | 0.321 | 0.721 | 67.000 |
| sae_top_cohens_d_n067 | SU | 0.849 | 0.360 | 0.275 | 0.774 | 67.000 |
| sae_top_cohens_d_n068 | AF | 0.916 | 0.586 | 0.506 | 0.851 | 68.000 |
| sae_top_cohens_d_n068 | GI | 0.745 | 0.392 | 0.377 | 0.692 | 68.000 |
| sae_top_cohens_d_n068 | QU | 0.974 | 0.950 | 0.907 | 0.935 | 68.000 |
| sae_top_cohens_d_n068 | QUC | 0.921 | 0.674 | 0.617 | 0.850 | 68.000 |
| sae_top_cohens_d_n068 | QUO | 0.955 | 0.852 | 0.767 | 0.886 | 68.000 |
| sae_top_cohens_d_n068 | RE | 0.849 | 0.652 | 0.616 | 0.774 | 68.000 |
| sae_top_cohens_d_n068 | REC | 0.906 | 0.653 | 0.582 | 0.826 | 68.000 |
| sae_top_cohens_d_n068 | RES | 0.791 | 0.278 | 0.316 | 0.715 | 68.000 |
| sae_top_cohens_d_n068 | SU | 0.848 | 0.356 | 0.276 | 0.775 | 68.000 |
| sae_top_cohens_d_n069 | AF | 0.916 | 0.586 | 0.505 | 0.850 | 69.000 |
| sae_top_cohens_d_n069 | GI | 0.746 | 0.391 | 0.379 | 0.694 | 69.000 |
| sae_top_cohens_d_n069 | QU | 0.974 | 0.950 | 0.906 | 0.934 | 69.000 |
| sae_top_cohens_d_n069 | QUC | 0.922 | 0.678 | 0.616 | 0.850 | 69.000 |
| sae_top_cohens_d_n069 | QUO | 0.956 | 0.855 | 0.773 | 0.890 | 69.000 |
| sae_top_cohens_d_n069 | RE | 0.851 | 0.654 | 0.619 | 0.778 | 69.000 |
| sae_top_cohens_d_n069 | REC | 0.906 | 0.655 | 0.581 | 0.825 | 69.000 |
| sae_top_cohens_d_n069 | RES | 0.790 | 0.275 | 0.316 | 0.715 | 69.000 |
| sae_top_cohens_d_n069 | SU | 0.848 | 0.355 | 0.279 | 0.776 | 69.000 |
| sae_top_cohens_d_n070 | AF | 0.916 | 0.586 | 0.506 | 0.851 | 70.000 |
| sae_top_cohens_d_n070 | GI | 0.747 | 0.392 | 0.381 | 0.695 | 70.000 |
| sae_top_cohens_d_n070 | QU | 0.974 | 0.950 | 0.906 | 0.934 | 70.000 |
| sae_top_cohens_d_n070 | QUC | 0.922 | 0.676 | 0.616 | 0.850 | 70.000 |
| sae_top_cohens_d_n070 | QUO | 0.956 | 0.857 | 0.773 | 0.889 | 70.000 |
| sae_top_cohens_d_n070 | RE | 0.851 | 0.653 | 0.620 | 0.779 | 70.000 |
| sae_top_cohens_d_n070 | REC | 0.906 | 0.655 | 0.584 | 0.828 | 70.000 |
| sae_top_cohens_d_n070 | RES | 0.790 | 0.274 | 0.316 | 0.714 | 70.000 |
| sae_top_cohens_d_n070 | SU | 0.855 | 0.356 | 0.278 | 0.775 | 70.000 |
| sae_top_cohens_d_n071 | AF | 0.914 | 0.584 | 0.503 | 0.848 | 71.000 |
| sae_top_cohens_d_n071 | GI | 0.754 | 0.392 | 0.383 | 0.697 | 71.000 |
| sae_top_cohens_d_n071 | QU | 0.974 | 0.950 | 0.906 | 0.935 | 71.000 |
| sae_top_cohens_d_n071 | QUC | 0.924 | 0.679 | 0.617 | 0.850 | 71.000 |
| sae_top_cohens_d_n071 | QUO | 0.956 | 0.857 | 0.773 | 0.890 | 71.000 |
| sae_top_cohens_d_n071 | RE | 0.852 | 0.655 | 0.620 | 0.778 | 71.000 |
| sae_top_cohens_d_n071 | REC | 0.907 | 0.654 | 0.581 | 0.826 | 71.000 |
| sae_top_cohens_d_n071 | RES | 0.790 | 0.274 | 0.316 | 0.713 | 71.000 |
| sae_top_cohens_d_n071 | SU | 0.855 | 0.354 | 0.279 | 0.775 | 71.000 |
| sae_top_cohens_d_n072 | AF | 0.917 | 0.583 | 0.503 | 0.847 | 72.000 |
| sae_top_cohens_d_n072 | GI | 0.756 | 0.391 | 0.380 | 0.695 | 72.000 |
| sae_top_cohens_d_n072 | QU | 0.974 | 0.950 | 0.905 | 0.934 | 72.000 |
| sae_top_cohens_d_n072 | QUC | 0.923 | 0.677 | 0.617 | 0.849 | 72.000 |
| sae_top_cohens_d_n072 | QUO | 0.956 | 0.856 | 0.770 | 0.887 | 72.000 |
| sae_top_cohens_d_n072 | RE | 0.853 | 0.656 | 0.621 | 0.778 | 72.000 |
| sae_top_cohens_d_n072 | REC | 0.907 | 0.655 | 0.586 | 0.830 | 72.000 |
| sae_top_cohens_d_n072 | RES | 0.790 | 0.272 | 0.316 | 0.712 | 72.000 |
| sae_top_cohens_d_n072 | SU | 0.857 | 0.355 | 0.284 | 0.777 | 72.000 |
| sae_top_cohens_d_n073 | AF | 0.916 | 0.584 | 0.504 | 0.847 | 73.000 |
| sae_top_cohens_d_n073 | GI | 0.755 | 0.389 | 0.377 | 0.692 | 73.000 |
| sae_top_cohens_d_n073 | QU | 0.974 | 0.951 | 0.908 | 0.935 | 73.000 |
| sae_top_cohens_d_n073 | QUC | 0.923 | 0.675 | 0.614 | 0.847 | 73.000 |
| sae_top_cohens_d_n073 | QUO | 0.956 | 0.856 | 0.770 | 0.887 | 73.000 |
| sae_top_cohens_d_n073 | RE | 0.852 | 0.657 | 0.618 | 0.776 | 73.000 |
| sae_top_cohens_d_n073 | REC | 0.907 | 0.656 | 0.585 | 0.829 | 73.000 |
| sae_top_cohens_d_n073 | RES | 0.790 | 0.273 | 0.317 | 0.715 | 73.000 |
| sae_top_cohens_d_n073 | SU | 0.858 | 0.360 | 0.282 | 0.778 | 73.000 |
| sae_top_cohens_d_n074 | AF | 0.918 | 0.586 | 0.505 | 0.848 | 74.000 |
| sae_top_cohens_d_n074 | GI | 0.755 | 0.390 | 0.377 | 0.692 | 74.000 |
| sae_top_cohens_d_n074 | QU | 0.974 | 0.951 | 0.907 | 0.935 | 74.000 |
| sae_top_cohens_d_n074 | QUC | 0.922 | 0.671 | 0.615 | 0.849 | 74.000 |
| sae_top_cohens_d_n074 | QUO | 0.957 | 0.857 | 0.771 | 0.888 | 74.000 |
| sae_top_cohens_d_n074 | RE | 0.852 | 0.656 | 0.621 | 0.779 | 74.000 |
| sae_top_cohens_d_n074 | REC | 0.908 | 0.657 | 0.582 | 0.827 | 74.000 |
| sae_top_cohens_d_n074 | RES | 0.790 | 0.272 | 0.319 | 0.716 | 74.000 |
| sae_top_cohens_d_n074 | SU | 0.858 | 0.364 | 0.287 | 0.781 | 74.000 |
| sae_top_cohens_d_n075 | AF | 0.921 | 0.593 | 0.508 | 0.850 | 75.000 |
| sae_top_cohens_d_n075 | GI | 0.756 | 0.390 | 0.379 | 0.694 | 75.000 |
| sae_top_cohens_d_n075 | QU | 0.975 | 0.952 | 0.908 | 0.935 | 75.000 |
| sae_top_cohens_d_n075 | QUC | 0.922 | 0.669 | 0.614 | 0.848 | 75.000 |
| sae_top_cohens_d_n075 | QUO | 0.957 | 0.856 | 0.771 | 0.888 | 75.000 |
| sae_top_cohens_d_n075 | RE | 0.852 | 0.656 | 0.621 | 0.779 | 75.000 |
| sae_top_cohens_d_n075 | REC | 0.908 | 0.660 | 0.582 | 0.828 | 75.000 |
| sae_top_cohens_d_n075 | RES | 0.791 | 0.271 | 0.319 | 0.717 | 75.000 |
| sae_top_cohens_d_n075 | SU | 0.860 | 0.361 | 0.284 | 0.779 | 75.000 |
| sae_top_cohens_d_n076 | AF | 0.922 | 0.595 | 0.507 | 0.851 | 76.000 |
| sae_top_cohens_d_n076 | GI | 0.753 | 0.388 | 0.377 | 0.693 | 76.000 |
| sae_top_cohens_d_n076 | QU | 0.974 | 0.951 | 0.908 | 0.936 | 76.000 |
| sae_top_cohens_d_n076 | QUC | 0.922 | 0.668 | 0.613 | 0.847 | 76.000 |
| sae_top_cohens_d_n076 | QUO | 0.956 | 0.856 | 0.771 | 0.888 | 76.000 |
| sae_top_cohens_d_n076 | RE | 0.853 | 0.656 | 0.620 | 0.779 | 76.000 |
| sae_top_cohens_d_n076 | REC | 0.908 | 0.660 | 0.588 | 0.832 | 76.000 |
| sae_top_cohens_d_n076 | RES | 0.791 | 0.273 | 0.320 | 0.717 | 76.000 |
| sae_top_cohens_d_n076 | SU | 0.858 | 0.361 | 0.286 | 0.781 | 76.000 |
| sae_top_cohens_d_n077 | AF | 0.920 | 0.592 | 0.505 | 0.850 | 77.000 |
| sae_top_cohens_d_n077 | GI | 0.754 | 0.388 | 0.382 | 0.696 | 77.000 |
| sae_top_cohens_d_n077 | QU | 0.974 | 0.952 | 0.908 | 0.935 | 77.000 |
| sae_top_cohens_d_n077 | QUC | 0.923 | 0.670 | 0.615 | 0.849 | 77.000 |
| sae_top_cohens_d_n077 | QUO | 0.959 | 0.861 | 0.775 | 0.890 | 77.000 |
| sae_top_cohens_d_n077 | RE | 0.853 | 0.656 | 0.619 | 0.778 | 77.000 |
| sae_top_cohens_d_n077 | REC | 0.908 | 0.657 | 0.585 | 0.829 | 77.000 |
| sae_top_cohens_d_n077 | RES | 0.791 | 0.272 | 0.323 | 0.719 | 77.000 |
| sae_top_cohens_d_n077 | SU | 0.857 | 0.359 | 0.287 | 0.783 | 77.000 |
| sae_top_cohens_d_n078 | AF | 0.920 | 0.593 | 0.510 | 0.855 | 78.000 |
| sae_top_cohens_d_n078 | GI | 0.754 | 0.388 | 0.384 | 0.697 | 78.000 |
| sae_top_cohens_d_n078 | QU | 0.974 | 0.952 | 0.908 | 0.935 | 78.000 |
| sae_top_cohens_d_n078 | QUC | 0.921 | 0.666 | 0.616 | 0.848 | 78.000 |
| sae_top_cohens_d_n078 | QUO | 0.959 | 0.863 | 0.774 | 0.889 | 78.000 |
| sae_top_cohens_d_n078 | RE | 0.853 | 0.659 | 0.621 | 0.779 | 78.000 |
| sae_top_cohens_d_n078 | REC | 0.907 | 0.656 | 0.583 | 0.826 | 78.000 |
| sae_top_cohens_d_n078 | RES | 0.790 | 0.271 | 0.322 | 0.719 | 78.000 |
| sae_top_cohens_d_n078 | SU | 0.858 | 0.360 | 0.286 | 0.781 | 78.000 |
| sae_top_cohens_d_n079 | AF | 0.921 | 0.594 | 0.514 | 0.859 | 79.000 |
| sae_top_cohens_d_n079 | GI | 0.757 | 0.390 | 0.384 | 0.697 | 79.000 |
| sae_top_cohens_d_n079 | QU | 0.974 | 0.952 | 0.907 | 0.934 | 79.000 |
| sae_top_cohens_d_n079 | QUC | 0.922 | 0.667 | 0.617 | 0.850 | 79.000 |
| sae_top_cohens_d_n079 | QUO | 0.958 | 0.862 | 0.774 | 0.890 | 79.000 |
| sae_top_cohens_d_n079 | RE | 0.855 | 0.661 | 0.623 | 0.781 | 79.000 |
| sae_top_cohens_d_n079 | REC | 0.908 | 0.657 | 0.583 | 0.827 | 79.000 |
| sae_top_cohens_d_n079 | RES | 0.790 | 0.274 | 0.320 | 0.716 | 79.000 |
| sae_top_cohens_d_n079 | SU | 0.855 | 0.356 | 0.283 | 0.779 | 79.000 |
| sae_top_cohens_d_n080 | AF | 0.922 | 0.593 | 0.507 | 0.857 | 80.000 |
| sae_top_cohens_d_n080 | GI | 0.756 | 0.388 | 0.384 | 0.698 | 80.000 |
| sae_top_cohens_d_n080 | QU | 0.975 | 0.953 | 0.908 | 0.935 | 80.000 |
| sae_top_cohens_d_n080 | QUC | 0.921 | 0.667 | 0.617 | 0.849 | 80.000 |
| sae_top_cohens_d_n080 | QUO | 0.958 | 0.862 | 0.775 | 0.890 | 80.000 |
| sae_top_cohens_d_n080 | RE | 0.856 | 0.662 | 0.622 | 0.780 | 80.000 |
| sae_top_cohens_d_n080 | REC | 0.908 | 0.658 | 0.586 | 0.829 | 80.000 |
| sae_top_cohens_d_n080 | RES | 0.790 | 0.274 | 0.321 | 0.717 | 80.000 |
| sae_top_cohens_d_n080 | SU | 0.855 | 0.355 | 0.282 | 0.773 | 80.000 |
| sae_top_cohens_d_n081 | AF | 0.922 | 0.601 | 0.512 | 0.857 | 81.000 |
| sae_top_cohens_d_n081 | GI | 0.757 | 0.390 | 0.380 | 0.695 | 81.000 |
| sae_top_cohens_d_n081 | QU | 0.975 | 0.952 | 0.907 | 0.934 | 81.000 |
| sae_top_cohens_d_n081 | QUC | 0.921 | 0.671 | 0.622 | 0.853 | 81.000 |
| sae_top_cohens_d_n081 | QUO | 0.958 | 0.864 | 0.775 | 0.891 | 81.000 |
| sae_top_cohens_d_n081 | RE | 0.856 | 0.663 | 0.625 | 0.782 | 81.000 |
| sae_top_cohens_d_n081 | REC | 0.909 | 0.659 | 0.584 | 0.828 | 81.000 |
| sae_top_cohens_d_n081 | RES | 0.790 | 0.274 | 0.320 | 0.715 | 81.000 |
| sae_top_cohens_d_n081 | SU | 0.853 | 0.353 | 0.280 | 0.771 | 81.000 |
| sae_top_cohens_d_n082 | AF | 0.921 | 0.599 | 0.509 | 0.857 | 82.000 |
| sae_top_cohens_d_n082 | GI | 0.757 | 0.391 | 0.380 | 0.694 | 82.000 |
| sae_top_cohens_d_n082 | QU | 0.975 | 0.952 | 0.906 | 0.933 | 82.000 |
| sae_top_cohens_d_n082 | QUC | 0.922 | 0.669 | 0.623 | 0.853 | 82.000 |
| sae_top_cohens_d_n082 | QUO | 0.958 | 0.863 | 0.776 | 0.891 | 82.000 |
| sae_top_cohens_d_n082 | RE | 0.856 | 0.661 | 0.623 | 0.781 | 82.000 |
| sae_top_cohens_d_n082 | REC | 0.909 | 0.657 | 0.585 | 0.828 | 82.000 |
| sae_top_cohens_d_n082 | RES | 0.789 | 0.270 | 0.323 | 0.718 | 82.000 |
| sae_top_cohens_d_n082 | SU | 0.856 | 0.353 | 0.283 | 0.775 | 82.000 |
| sae_top_cohens_d_n083 | AF | 0.924 | 0.605 | 0.512 | 0.856 | 83.000 |
| sae_top_cohens_d_n083 | GI | 0.760 | 0.392 | 0.385 | 0.699 | 83.000 |
| sae_top_cohens_d_n083 | QU | 0.975 | 0.952 | 0.905 | 0.932 | 83.000 |
| sae_top_cohens_d_n083 | QUC | 0.923 | 0.673 | 0.624 | 0.854 | 83.000 |
| sae_top_cohens_d_n083 | QUO | 0.958 | 0.864 | 0.776 | 0.892 | 83.000 |
| sae_top_cohens_d_n083 | RE | 0.855 | 0.660 | 0.621 | 0.779 | 83.000 |
| sae_top_cohens_d_n083 | REC | 0.908 | 0.656 | 0.584 | 0.826 | 83.000 |
| sae_top_cohens_d_n083 | RES | 0.788 | 0.269 | 0.323 | 0.717 | 83.000 |
| sae_top_cohens_d_n083 | SU | 0.856 | 0.354 | 0.283 | 0.775 | 83.000 |
| sae_top_cohens_d_n084 | AF | 0.925 | 0.608 | 0.515 | 0.859 | 84.000 |
| sae_top_cohens_d_n084 | GI | 0.761 | 0.392 | 0.383 | 0.698 | 84.000 |
| sae_top_cohens_d_n084 | QU | 0.974 | 0.952 | 0.905 | 0.932 | 84.000 |
| sae_top_cohens_d_n084 | QUC | 0.923 | 0.674 | 0.623 | 0.852 | 84.000 |
| sae_top_cohens_d_n084 | QUO | 0.958 | 0.863 | 0.775 | 0.890 | 84.000 |
| sae_top_cohens_d_n084 | RE | 0.856 | 0.661 | 0.621 | 0.779 | 84.000 |
| sae_top_cohens_d_n084 | REC | 0.908 | 0.657 | 0.586 | 0.829 | 84.000 |
| sae_top_cohens_d_n084 | RES | 0.788 | 0.267 | 0.324 | 0.719 | 84.000 |
| sae_top_cohens_d_n084 | SU | 0.855 | 0.352 | 0.277 | 0.768 | 84.000 |
| sae_top_cohens_d_n085 | AF | 0.925 | 0.609 | 0.517 | 0.859 | 85.000 |
| sae_top_cohens_d_n085 | GI | 0.762 | 0.393 | 0.385 | 0.700 | 85.000 |
| sae_top_cohens_d_n085 | QU | 0.974 | 0.952 | 0.905 | 0.934 | 85.000 |
| sae_top_cohens_d_n085 | QUC | 0.923 | 0.674 | 0.622 | 0.851 | 85.000 |
| sae_top_cohens_d_n085 | QUO | 0.958 | 0.863 | 0.776 | 0.890 | 85.000 |
| sae_top_cohens_d_n085 | RE | 0.856 | 0.663 | 0.618 | 0.777 | 85.000 |
| sae_top_cohens_d_n085 | REC | 0.908 | 0.657 | 0.588 | 0.830 | 85.000 |
| sae_top_cohens_d_n085 | RES | 0.788 | 0.265 | 0.321 | 0.716 | 85.000 |
| sae_top_cohens_d_n085 | SU | 0.860 | 0.352 | 0.279 | 0.770 | 85.000 |
| sae_top_cohens_d_n086 | AF | 0.923 | 0.605 | 0.508 | 0.852 | 86.000 |
| sae_top_cohens_d_n086 | GI | 0.762 | 0.395 | 0.384 | 0.699 | 86.000 |
| sae_top_cohens_d_n086 | QU | 0.974 | 0.952 | 0.905 | 0.934 | 86.000 |
| sae_top_cohens_d_n086 | QUC | 0.922 | 0.676 | 0.625 | 0.853 | 86.000 |
| sae_top_cohens_d_n086 | QUO | 0.958 | 0.864 | 0.775 | 0.890 | 86.000 |
| sae_top_cohens_d_n086 | RE | 0.856 | 0.662 | 0.618 | 0.778 | 86.000 |
| sae_top_cohens_d_n086 | REC | 0.909 | 0.657 | 0.585 | 0.829 | 86.000 |
| sae_top_cohens_d_n086 | RES | 0.789 | 0.267 | 0.324 | 0.718 | 86.000 |
| sae_top_cohens_d_n086 | SU | 0.857 | 0.350 | 0.291 | 0.783 | 86.000 |
| sae_top_cohens_d_n087 | AF | 0.922 | 0.600 | 0.506 | 0.852 | 87.000 |
| sae_top_cohens_d_n087 | GI | 0.762 | 0.394 | 0.384 | 0.699 | 87.000 |
| sae_top_cohens_d_n087 | QU | 0.974 | 0.951 | 0.905 | 0.933 | 87.000 |
| sae_top_cohens_d_n087 | QUC | 0.923 | 0.674 | 0.625 | 0.853 | 87.000 |
| sae_top_cohens_d_n087 | QUO | 0.959 | 0.864 | 0.777 | 0.891 | 87.000 |
| sae_top_cohens_d_n087 | RE | 0.856 | 0.662 | 0.619 | 0.778 | 87.000 |
| sae_top_cohens_d_n087 | REC | 0.909 | 0.657 | 0.583 | 0.827 | 87.000 |
| sae_top_cohens_d_n087 | RES | 0.789 | 0.266 | 0.323 | 0.717 | 87.000 |
| sae_top_cohens_d_n087 | SU | 0.856 | 0.343 | 0.290 | 0.781 | 87.000 |
| sae_top_cohens_d_n088 | AF | 0.922 | 0.598 | 0.508 | 0.853 | 88.000 |
| sae_top_cohens_d_n088 | GI | 0.763 | 0.394 | 0.382 | 0.698 | 88.000 |
| sae_top_cohens_d_n088 | QU | 0.974 | 0.951 | 0.906 | 0.934 | 88.000 |
| sae_top_cohens_d_n088 | QUC | 0.923 | 0.676 | 0.627 | 0.854 | 88.000 |
| sae_top_cohens_d_n088 | QUO | 0.959 | 0.866 | 0.780 | 0.892 | 88.000 |
| sae_top_cohens_d_n088 | RE | 0.856 | 0.662 | 0.621 | 0.780 | 88.000 |
| sae_top_cohens_d_n088 | REC | 0.909 | 0.657 | 0.584 | 0.829 | 88.000 |
| sae_top_cohens_d_n088 | RES | 0.788 | 0.265 | 0.321 | 0.715 | 88.000 |
| sae_top_cohens_d_n088 | SU | 0.856 | 0.342 | 0.290 | 0.781 | 88.000 |
| sae_top_cohens_d_n089 | AF | 0.923 | 0.596 | 0.507 | 0.851 | 89.000 |
| sae_top_cohens_d_n089 | GI | 0.764 | 0.394 | 0.383 | 0.698 | 89.000 |
| sae_top_cohens_d_n089 | QU | 0.974 | 0.952 | 0.905 | 0.933 | 89.000 |
| sae_top_cohens_d_n089 | QUC | 0.925 | 0.677 | 0.627 | 0.855 | 89.000 |
| sae_top_cohens_d_n089 | QUO | 0.959 | 0.866 | 0.783 | 0.894 | 89.000 |
| sae_top_cohens_d_n089 | RE | 0.856 | 0.662 | 0.617 | 0.777 | 89.000 |
| sae_top_cohens_d_n089 | REC | 0.909 | 0.658 | 0.584 | 0.829 | 89.000 |
| sae_top_cohens_d_n089 | RES | 0.788 | 0.266 | 0.320 | 0.714 | 89.000 |
| sae_top_cohens_d_n089 | SU | 0.856 | 0.340 | 0.295 | 0.786 | 89.000 |
| sae_top_cohens_d_n090 | AF | 0.923 | 0.599 | 0.506 | 0.850 | 90.000 |
| sae_top_cohens_d_n090 | GI | 0.764 | 0.395 | 0.384 | 0.699 | 90.000 |
| sae_top_cohens_d_n090 | QU | 0.975 | 0.952 | 0.905 | 0.933 | 90.000 |
| sae_top_cohens_d_n090 | QUC | 0.925 | 0.678 | 0.628 | 0.856 | 90.000 |
| sae_top_cohens_d_n090 | QUO | 0.959 | 0.865 | 0.780 | 0.892 | 90.000 |
| sae_top_cohens_d_n090 | RE | 0.856 | 0.663 | 0.617 | 0.777 | 90.000 |
| sae_top_cohens_d_n090 | REC | 0.909 | 0.659 | 0.584 | 0.828 | 90.000 |
| sae_top_cohens_d_n090 | RES | 0.787 | 0.266 | 0.323 | 0.716 | 90.000 |
| sae_top_cohens_d_n090 | SU | 0.856 | 0.345 | 0.295 | 0.784 | 90.000 |
| sae_top_cohens_d_n091 | AF | 0.924 | 0.604 | 0.509 | 0.852 | 91.000 |
| sae_top_cohens_d_n091 | GI | 0.764 | 0.397 | 0.384 | 0.699 | 91.000 |
| sae_top_cohens_d_n091 | QU | 0.975 | 0.952 | 0.905 | 0.933 | 91.000 |
| sae_top_cohens_d_n091 | QUC | 0.925 | 0.678 | 0.627 | 0.854 | 91.000 |
| sae_top_cohens_d_n091 | QUO | 0.959 | 0.865 | 0.780 | 0.892 | 91.000 |
| sae_top_cohens_d_n091 | RE | 0.856 | 0.663 | 0.618 | 0.778 | 91.000 |
| sae_top_cohens_d_n091 | REC | 0.910 | 0.661 | 0.583 | 0.827 | 91.000 |
| sae_top_cohens_d_n091 | RES | 0.786 | 0.266 | 0.322 | 0.715 | 91.000 |
| sae_top_cohens_d_n091 | SU | 0.854 | 0.349 | 0.296 | 0.783 | 91.000 |
| sae_top_cohens_d_n092 | AF | 0.923 | 0.598 | 0.509 | 0.853 | 92.000 |
| sae_top_cohens_d_n092 | GI | 0.764 | 0.397 | 0.382 | 0.697 | 92.000 |
| sae_top_cohens_d_n092 | QU | 0.975 | 0.953 | 0.904 | 0.932 | 92.000 |
| sae_top_cohens_d_n092 | QUC | 0.925 | 0.678 | 0.630 | 0.855 | 92.000 |
| sae_top_cohens_d_n092 | QUO | 0.959 | 0.864 | 0.778 | 0.891 | 92.000 |
| sae_top_cohens_d_n092 | RE | 0.857 | 0.663 | 0.618 | 0.778 | 92.000 |
| sae_top_cohens_d_n092 | REC | 0.910 | 0.663 | 0.587 | 0.829 | 92.000 |
| sae_top_cohens_d_n092 | RES | 0.786 | 0.266 | 0.314 | 0.707 | 92.000 |
| sae_top_cohens_d_n092 | SU | 0.855 | 0.346 | 0.291 | 0.778 | 92.000 |
| sae_top_cohens_d_n093 | AF | 0.924 | 0.599 | 0.508 | 0.853 | 93.000 |
| sae_top_cohens_d_n093 | GI | 0.764 | 0.397 | 0.384 | 0.698 | 93.000 |
| sae_top_cohens_d_n093 | QU | 0.975 | 0.954 | 0.905 | 0.933 | 93.000 |
| sae_top_cohens_d_n093 | QUC | 0.925 | 0.677 | 0.627 | 0.854 | 93.000 |
| sae_top_cohens_d_n093 | QUO | 0.959 | 0.864 | 0.778 | 0.890 | 93.000 |
| sae_top_cohens_d_n093 | RE | 0.857 | 0.662 | 0.619 | 0.779 | 93.000 |
| sae_top_cohens_d_n093 | REC | 0.910 | 0.662 | 0.586 | 0.829 | 93.000 |
| sae_top_cohens_d_n093 | RES | 0.786 | 0.266 | 0.318 | 0.711 | 93.000 |
| sae_top_cohens_d_n093 | SU | 0.854 | 0.345 | 0.288 | 0.775 | 93.000 |
| sae_top_cohens_d_n094 | AF | 0.923 | 0.596 | 0.509 | 0.855 | 94.000 |
| sae_top_cohens_d_n094 | GI | 0.764 | 0.396 | 0.384 | 0.698 | 94.000 |
| sae_top_cohens_d_n094 | QU | 0.975 | 0.954 | 0.905 | 0.933 | 94.000 |
| sae_top_cohens_d_n094 | QUC | 0.926 | 0.680 | 0.628 | 0.854 | 94.000 |
| sae_top_cohens_d_n094 | QUO | 0.959 | 0.864 | 0.777 | 0.890 | 94.000 |
| sae_top_cohens_d_n094 | RE | 0.856 | 0.662 | 0.622 | 0.781 | 94.000 |
| sae_top_cohens_d_n094 | REC | 0.910 | 0.664 | 0.586 | 0.830 | 94.000 |
| sae_top_cohens_d_n094 | RES | 0.787 | 0.267 | 0.319 | 0.712 | 94.000 |
| sae_top_cohens_d_n094 | SU | 0.855 | 0.351 | 0.288 | 0.777 | 94.000 |
| sae_top_cohens_d_n095 | AF | 0.922 | 0.597 | 0.505 | 0.853 | 95.000 |
| sae_top_cohens_d_n095 | GI | 0.766 | 0.393 | 0.385 | 0.699 | 95.000 |
| sae_top_cohens_d_n095 | QU | 0.975 | 0.954 | 0.906 | 0.934 | 95.000 |
| sae_top_cohens_d_n095 | QUC | 0.926 | 0.677 | 0.629 | 0.856 | 95.000 |
| sae_top_cohens_d_n095 | QUO | 0.959 | 0.864 | 0.777 | 0.889 | 95.000 |
| sae_top_cohens_d_n095 | RE | 0.856 | 0.663 | 0.620 | 0.780 | 95.000 |
| sae_top_cohens_d_n095 | REC | 0.911 | 0.667 | 0.583 | 0.827 | 95.000 |
| sae_top_cohens_d_n095 | RES | 0.788 | 0.268 | 0.321 | 0.712 | 95.000 |
| sae_top_cohens_d_n095 | SU | 0.855 | 0.350 | 0.291 | 0.778 | 95.000 |
| sae_top_cohens_d_n096 | AF | 0.921 | 0.593 | 0.499 | 0.849 | 96.000 |
| sae_top_cohens_d_n096 | GI | 0.763 | 0.392 | 0.383 | 0.697 | 96.000 |
| sae_top_cohens_d_n096 | QU | 0.976 | 0.955 | 0.905 | 0.933 | 96.000 |
| sae_top_cohens_d_n096 | QUC | 0.926 | 0.677 | 0.631 | 0.857 | 96.000 |
| sae_top_cohens_d_n096 | QUO | 0.959 | 0.862 | 0.776 | 0.889 | 96.000 |
| sae_top_cohens_d_n096 | RE | 0.856 | 0.663 | 0.620 | 0.780 | 96.000 |
| sae_top_cohens_d_n096 | REC | 0.911 | 0.667 | 0.583 | 0.827 | 96.000 |
| sae_top_cohens_d_n096 | RES | 0.788 | 0.269 | 0.320 | 0.712 | 96.000 |
| sae_top_cohens_d_n096 | SU | 0.856 | 0.351 | 0.299 | 0.787 | 96.000 |
| sae_top_cohens_d_n097 | AF | 0.921 | 0.592 | 0.499 | 0.848 | 97.000 |
| sae_top_cohens_d_n097 | GI | 0.764 | 0.392 | 0.385 | 0.698 | 97.000 |
| sae_top_cohens_d_n097 | QU | 0.975 | 0.955 | 0.904 | 0.932 | 97.000 |
| sae_top_cohens_d_n097 | QUC | 0.926 | 0.676 | 0.628 | 0.854 | 97.000 |
| sae_top_cohens_d_n097 | QUO | 0.959 | 0.863 | 0.778 | 0.890 | 97.000 |
| sae_top_cohens_d_n097 | RE | 0.857 | 0.663 | 0.620 | 0.779 | 97.000 |
| sae_top_cohens_d_n097 | REC | 0.910 | 0.665 | 0.580 | 0.825 | 97.000 |
| sae_top_cohens_d_n097 | RES | 0.790 | 0.267 | 0.317 | 0.710 | 97.000 |
| sae_top_cohens_d_n097 | SU | 0.849 | 0.354 | 0.309 | 0.795 | 97.000 |
| sae_top_cohens_d_n098 | AF | 0.920 | 0.584 | 0.493 | 0.845 | 98.000 |
| sae_top_cohens_d_n098 | GI | 0.764 | 0.393 | 0.386 | 0.698 | 98.000 |
| sae_top_cohens_d_n098 | QU | 0.976 | 0.955 | 0.904 | 0.933 | 98.000 |
| sae_top_cohens_d_n098 | QUC | 0.926 | 0.676 | 0.626 | 0.852 | 98.000 |
| sae_top_cohens_d_n098 | QUO | 0.959 | 0.865 | 0.782 | 0.893 | 98.000 |
| sae_top_cohens_d_n098 | RE | 0.858 | 0.663 | 0.623 | 0.781 | 98.000 |
| sae_top_cohens_d_n098 | REC | 0.910 | 0.665 | 0.581 | 0.824 | 98.000 |
| sae_top_cohens_d_n098 | RES | 0.791 | 0.267 | 0.318 | 0.709 | 98.000 |
| sae_top_cohens_d_n098 | SU | 0.851 | 0.350 | 0.318 | 0.801 | 98.000 |
| sae_top_cohens_d_n099 | AF | 0.916 | 0.581 | 0.493 | 0.846 | 99.000 |
| sae_top_cohens_d_n099 | GI | 0.764 | 0.393 | 0.385 | 0.698 | 99.000 |
| sae_top_cohens_d_n099 | QU | 0.975 | 0.955 | 0.905 | 0.933 | 99.000 |
| sae_top_cohens_d_n099 | QUC | 0.925 | 0.677 | 0.626 | 0.852 | 99.000 |
| sae_top_cohens_d_n099 | QUO | 0.959 | 0.864 | 0.782 | 0.892 | 99.000 |
| sae_top_cohens_d_n099 | RE | 0.858 | 0.663 | 0.624 | 0.782 | 99.000 |
| sae_top_cohens_d_n099 | REC | 0.910 | 0.663 | 0.579 | 0.823 | 99.000 |
| sae_top_cohens_d_n099 | RES | 0.790 | 0.268 | 0.318 | 0.710 | 99.000 |
| sae_top_cohens_d_n099 | SU | 0.850 | 0.352 | 0.310 | 0.794 | 99.000 |
| sae_top_cohens_d_n100 | AF | 0.915 | 0.582 | 0.493 | 0.846 | 100.000 |
| sae_top_cohens_d_n100 | GI | 0.765 | 0.395 | 0.382 | 0.695 | 100.000 |
| sae_top_cohens_d_n100 | QU | 0.975 | 0.955 | 0.905 | 0.933 | 100.000 |
| sae_top_cohens_d_n100 | QUC | 0.925 | 0.677 | 0.626 | 0.852 | 100.000 |
| sae_top_cohens_d_n100 | QUO | 0.959 | 0.864 | 0.781 | 0.892 | 100.000 |
| sae_top_cohens_d_n100 | RE | 0.858 | 0.663 | 0.626 | 0.784 | 100.000 |
| sae_top_cohens_d_n100 | REC | 0.910 | 0.662 | 0.578 | 0.823 | 100.000 |
| sae_top_cohens_d_n100 | RES | 0.790 | 0.268 | 0.319 | 0.712 | 100.000 |
| sae_top_cohens_d_n100 | SU | 0.850 | 0.351 | 0.314 | 0.798 | 100.000 |
| sae_top_directional_auc_n000 | AF | 0.500 | 0.056 | 0.000 | 0.500 | 0.000 |
| sae_top_directional_auc_n000 | GI | 0.500 | 0.110 | 0.000 | 0.500 | 0.000 |
| sae_top_directional_auc_n000 | QU | 0.500 | 0.319 | 0.000 | 0.500 | 0.000 |
| sae_top_directional_auc_n000 | QUC | 0.500 | 0.124 | 0.000 | 0.500 | 0.000 |
| sae_top_directional_auc_n000 | QUO | 0.500 | 0.195 | 0.000 | 0.500 | 0.000 |
| sae_top_directional_auc_n000 | RE | 0.500 | 0.219 | 0.000 | 0.500 | 0.000 |
| sae_top_directional_auc_n000 | REC | 0.500 | 0.136 | 0.000 | 0.500 | 0.000 |
| sae_top_directional_auc_n000 | RES | 0.500 | 0.083 | 0.000 | 0.500 | 0.000 |
| sae_top_directional_auc_n000 | SU | 0.500 | 0.036 | 0.000 | 0.500 | 0.000 |
| sae_top_directional_auc_n001 | AF | 0.800 | 0.390 | 0.369 | 0.780 | 1.000 |
| sae_top_directional_auc_n001 | GI | 0.682 | 0.165 | 0.276 | 0.655 | 1.000 |
| sae_top_directional_auc_n001 | QU | 0.925 | 0.861 | 0.824 | 0.875 | 1.000 |
| sae_top_directional_auc_n001 | QUC | 0.822 | 0.368 | 0.468 | 0.783 | 1.000 |
| sae_top_directional_auc_n001 | QUO | 0.864 | 0.513 | 0.648 | 0.827 | 1.000 |
| sae_top_directional_auc_n001 | RE | 0.693 | 0.400 | 0.510 | 0.687 | 1.000 |
| sae_top_directional_auc_n001 | REC | 0.709 | 0.263 | 0.365 | 0.679 | 1.000 |
| sae_top_directional_auc_n001 | RES | 0.614 | 0.123 | 0.208 | 0.604 | 1.000 |
| sae_top_directional_auc_n001 | SU | 0.661 | 0.061 | 0.100 | 0.635 | 1.000 |
| sae_top_directional_auc_n002 | AF | 0.836 | 0.452 | 0.416 | 0.803 | 2.000 |
| sae_top_directional_auc_n002 | GI | 0.697 | 0.172 | 0.286 | 0.672 | 2.000 |
| sae_top_directional_auc_n002 | QU | 0.944 | 0.904 | 0.860 | 0.900 | 2.000 |
| sae_top_directional_auc_n002 | QUC | 0.848 | 0.488 | 0.483 | 0.784 | 2.000 |
| sae_top_directional_auc_n002 | QUO | 0.908 | 0.625 | 0.691 | 0.852 | 2.000 |
| sae_top_directional_auc_n002 | RE | 0.745 | 0.479 | 0.536 | 0.711 | 2.000 |
| sae_top_directional_auc_n002 | REC | 0.755 | 0.316 | 0.373 | 0.688 | 2.000 |
| sae_top_directional_auc_n002 | RES | 0.706 | 0.174 | 0.199 | 0.629 | 2.000 |
| sae_top_directional_auc_n002 | SU | 0.746 | 0.086 | 0.126 | 0.681 | 2.000 |
| sae_top_directional_auc_n003 | AF | 0.880 | 0.481 | 0.422 | 0.814 | 3.000 |
| sae_top_directional_auc_n003 | GI | 0.693 | 0.169 | 0.287 | 0.674 | 3.000 |
| sae_top_directional_auc_n003 | QU | 0.950 | 0.913 | 0.864 | 0.903 | 3.000 |
| sae_top_directional_auc_n003 | QUC | 0.851 | 0.496 | 0.484 | 0.784 | 3.000 |
| sae_top_directional_auc_n003 | QUO | 0.920 | 0.668 | 0.714 | 0.860 | 3.000 |
| sae_top_directional_auc_n003 | RE | 0.744 | 0.482 | 0.545 | 0.719 | 3.000 |
| sae_top_directional_auc_n003 | REC | 0.765 | 0.335 | 0.382 | 0.695 | 3.000 |
| sae_top_directional_auc_n003 | RES | 0.726 | 0.179 | 0.244 | 0.650 | 3.000 |
| sae_top_directional_auc_n003 | SU | 0.762 | 0.106 | 0.134 | 0.698 | 3.000 |
| sae_top_directional_auc_n004 | AF | 0.886 | 0.497 | 0.402 | 0.822 | 4.000 |
| sae_top_directional_auc_n004 | GI | 0.699 | 0.185 | 0.288 | 0.677 | 4.000 |
| sae_top_directional_auc_n004 | QU | 0.952 | 0.920 | 0.866 | 0.904 | 4.000 |
| sae_top_directional_auc_n004 | QUC | 0.852 | 0.509 | 0.483 | 0.782 | 4.000 |
| sae_top_directional_auc_n004 | QUO | 0.927 | 0.696 | 0.729 | 0.871 | 4.000 |
| sae_top_directional_auc_n004 | RE | 0.750 | 0.475 | 0.518 | 0.701 | 4.000 |
| sae_top_directional_auc_n004 | REC | 0.774 | 0.354 | 0.395 | 0.706 | 4.000 |
| sae_top_directional_auc_n004 | RES | 0.722 | 0.185 | 0.233 | 0.641 | 4.000 |
| sae_top_directional_auc_n004 | SU | 0.778 | 0.111 | 0.146 | 0.724 | 4.000 |
| sae_top_directional_auc_n005 | AF | 0.891 | 0.519 | 0.414 | 0.824 | 5.000 |
| sae_top_directional_auc_n005 | GI | 0.706 | 0.194 | 0.289 | 0.680 | 5.000 |
| sae_top_directional_auc_n005 | QU | 0.952 | 0.922 | 0.868 | 0.905 | 5.000 |
| sae_top_directional_auc_n005 | QUC | 0.852 | 0.512 | 0.482 | 0.781 | 5.000 |
| sae_top_directional_auc_n005 | QUO | 0.927 | 0.695 | 0.731 | 0.872 | 5.000 |
| sae_top_directional_auc_n005 | RE | 0.756 | 0.476 | 0.516 | 0.700 | 5.000 |
| sae_top_directional_auc_n005 | REC | 0.797 | 0.387 | 0.415 | 0.721 | 5.000 |
| sae_top_directional_auc_n005 | RES | 0.728 | 0.193 | 0.222 | 0.646 | 5.000 |
| sae_top_directional_auc_n005 | SU | 0.785 | 0.141 | 0.150 | 0.728 | 5.000 |
| sae_top_directional_auc_n006 | AF | 0.898 | 0.533 | 0.421 | 0.824 | 6.000 |
| sae_top_directional_auc_n006 | GI | 0.729 | 0.224 | 0.288 | 0.678 | 6.000 |
| sae_top_directional_auc_n006 | QU | 0.954 | 0.925 | 0.871 | 0.907 | 6.000 |
| sae_top_directional_auc_n006 | QUC | 0.855 | 0.533 | 0.488 | 0.784 | 6.000 |
| sae_top_directional_auc_n006 | QUO | 0.927 | 0.695 | 0.730 | 0.871 | 6.000 |
| sae_top_directional_auc_n006 | RE | 0.763 | 0.487 | 0.513 | 0.698 | 6.000 |
| sae_top_directional_auc_n006 | REC | 0.820 | 0.417 | 0.439 | 0.742 | 6.000 |
| sae_top_directional_auc_n006 | RES | 0.745 | 0.207 | 0.234 | 0.662 | 6.000 |
| sae_top_directional_auc_n006 | SU | 0.811 | 0.169 | 0.163 | 0.748 | 6.000 |
| sae_top_directional_auc_n007 | AF | 0.905 | 0.541 | 0.436 | 0.832 | 7.000 |
| sae_top_directional_auc_n007 | GI | 0.733 | 0.228 | 0.292 | 0.676 | 7.000 |
| sae_top_directional_auc_n007 | QU | 0.954 | 0.927 | 0.870 | 0.906 | 7.000 |
| sae_top_directional_auc_n007 | QUC | 0.852 | 0.531 | 0.489 | 0.784 | 7.000 |
| sae_top_directional_auc_n007 | QUO | 0.928 | 0.701 | 0.728 | 0.871 | 7.000 |
| sae_top_directional_auc_n007 | RE | 0.770 | 0.509 | 0.525 | 0.709 | 7.000 |
| sae_top_directional_auc_n007 | REC | 0.830 | 0.429 | 0.443 | 0.747 | 7.000 |
| sae_top_directional_auc_n007 | RES | 0.749 | 0.208 | 0.238 | 0.666 | 7.000 |
| sae_top_directional_auc_n007 | SU | 0.825 | 0.190 | 0.172 | 0.758 | 7.000 |
| sae_top_directional_auc_n008 | AF | 0.908 | 0.560 | 0.437 | 0.833 | 8.000 |
| sae_top_directional_auc_n008 | GI | 0.735 | 0.224 | 0.291 | 0.671 | 8.000 |
| sae_top_directional_auc_n008 | QU | 0.955 | 0.927 | 0.873 | 0.908 | 8.000 |
| sae_top_directional_auc_n008 | QUC | 0.854 | 0.528 | 0.491 | 0.786 | 8.000 |
| sae_top_directional_auc_n008 | QUO | 0.929 | 0.705 | 0.733 | 0.873 | 8.000 |
| sae_top_directional_auc_n008 | RE | 0.781 | 0.515 | 0.531 | 0.715 | 8.000 |
| sae_top_directional_auc_n008 | REC | 0.837 | 0.432 | 0.451 | 0.757 | 8.000 |
| sae_top_directional_auc_n008 | RES | 0.764 | 0.224 | 0.251 | 0.685 | 8.000 |
| sae_top_directional_auc_n008 | SU | 0.834 | 0.209 | 0.173 | 0.754 | 8.000 |
| sae_top_directional_auc_n009 | AF | 0.908 | 0.562 | 0.436 | 0.831 | 9.000 |
| sae_top_directional_auc_n009 | GI | 0.746 | 0.236 | 0.298 | 0.676 | 9.000 |
| sae_top_directional_auc_n009 | QU | 0.956 | 0.930 | 0.879 | 0.912 | 9.000 |
| sae_top_directional_auc_n009 | QUC | 0.856 | 0.534 | 0.502 | 0.792 | 9.000 |
| sae_top_directional_auc_n009 | QUO | 0.932 | 0.732 | 0.732 | 0.870 | 9.000 |
| sae_top_directional_auc_n009 | RE | 0.784 | 0.523 | 0.530 | 0.714 | 9.000 |
| sae_top_directional_auc_n009 | REC | 0.839 | 0.438 | 0.456 | 0.761 | 9.000 |
| sae_top_directional_auc_n009 | RES | 0.776 | 0.234 | 0.262 | 0.691 | 9.000 |
| sae_top_directional_auc_n009 | SU | 0.844 | 0.223 | 0.185 | 0.764 | 9.000 |
| sae_top_directional_auc_n010 | AF | 0.910 | 0.565 | 0.446 | 0.832 | 10.000 |
| sae_top_directional_auc_n010 | GI | 0.753 | 0.254 | 0.308 | 0.684 | 10.000 |
| sae_top_directional_auc_n010 | QU | 0.956 | 0.932 | 0.879 | 0.912 | 10.000 |
| sae_top_directional_auc_n010 | QUC | 0.870 | 0.543 | 0.522 | 0.800 | 10.000 |
| sae_top_directional_auc_n010 | QUO | 0.934 | 0.754 | 0.731 | 0.869 | 10.000 |
| sae_top_directional_auc_n010 | RE | 0.787 | 0.531 | 0.535 | 0.718 | 10.000 |
| sae_top_directional_auc_n010 | REC | 0.843 | 0.445 | 0.453 | 0.758 | 10.000 |
| sae_top_directional_auc_n010 | RES | 0.777 | 0.235 | 0.264 | 0.695 | 10.000 |
| sae_top_directional_auc_n010 | SU | 0.845 | 0.225 | 0.189 | 0.769 | 10.000 |
| sae_top_directional_auc_n011 | AF | 0.913 | 0.566 | 0.460 | 0.843 | 11.000 |
| sae_top_directional_auc_n011 | GI | 0.761 | 0.256 | 0.319 | 0.693 | 11.000 |
| sae_top_directional_auc_n011 | QU | 0.957 | 0.933 | 0.883 | 0.915 | 11.000 |
| sae_top_directional_auc_n011 | QUC | 0.876 | 0.545 | 0.534 | 0.806 | 11.000 |
| sae_top_directional_auc_n011 | QUO | 0.937 | 0.766 | 0.743 | 0.877 | 11.000 |
| sae_top_directional_auc_n011 | RE | 0.794 | 0.550 | 0.541 | 0.723 | 11.000 |
| sae_top_directional_auc_n011 | REC | 0.847 | 0.451 | 0.462 | 0.766 | 11.000 |
| sae_top_directional_auc_n011 | RES | 0.780 | 0.237 | 0.275 | 0.699 | 11.000 |
| sae_top_directional_auc_n011 | SU | 0.848 | 0.243 | 0.191 | 0.773 | 11.000 |
| sae_top_directional_auc_n012 | AF | 0.919 | 0.576 | 0.471 | 0.848 | 12.000 |
| sae_top_directional_auc_n012 | GI | 0.769 | 0.263 | 0.333 | 0.705 | 12.000 |
| sae_top_directional_auc_n012 | QU | 0.957 | 0.934 | 0.885 | 0.917 | 12.000 |
| sae_top_directional_auc_n012 | QUC | 0.884 | 0.555 | 0.543 | 0.813 | 12.000 |
| sae_top_directional_auc_n012 | QUO | 0.938 | 0.774 | 0.743 | 0.878 | 12.000 |
| sae_top_directional_auc_n012 | RE | 0.796 | 0.551 | 0.547 | 0.728 | 12.000 |
| sae_top_directional_auc_n012 | REC | 0.853 | 0.472 | 0.476 | 0.776 | 12.000 |
| sae_top_directional_auc_n012 | RES | 0.790 | 0.250 | 0.273 | 0.707 | 12.000 |
| sae_top_directional_auc_n012 | SU | 0.849 | 0.263 | 0.185 | 0.766 | 12.000 |
| sae_top_directional_auc_n013 | AF | 0.918 | 0.580 | 0.473 | 0.848 | 13.000 |
| sae_top_directional_auc_n013 | GI | 0.780 | 0.281 | 0.338 | 0.708 | 13.000 |
| sae_top_directional_auc_n013 | QU | 0.957 | 0.934 | 0.883 | 0.916 | 13.000 |
| sae_top_directional_auc_n013 | QUC | 0.886 | 0.558 | 0.548 | 0.815 | 13.000 |
| sae_top_directional_auc_n013 | QUO | 0.939 | 0.775 | 0.745 | 0.879 | 13.000 |
| sae_top_directional_auc_n013 | RE | 0.799 | 0.553 | 0.543 | 0.724 | 13.000 |
| sae_top_directional_auc_n013 | REC | 0.855 | 0.478 | 0.480 | 0.778 | 13.000 |
| sae_top_directional_auc_n013 | RES | 0.799 | 0.264 | 0.278 | 0.714 | 13.000 |
| sae_top_directional_auc_n013 | SU | 0.846 | 0.258 | 0.192 | 0.768 | 13.000 |
| sae_top_directional_auc_n014 | AF | 0.918 | 0.579 | 0.474 | 0.845 | 14.000 |
| sae_top_directional_auc_n014 | GI | 0.784 | 0.281 | 0.344 | 0.714 | 14.000 |
| sae_top_directional_auc_n014 | QU | 0.958 | 0.934 | 0.884 | 0.916 | 14.000 |
| sae_top_directional_auc_n014 | QUC | 0.891 | 0.564 | 0.550 | 0.816 | 14.000 |
| sae_top_directional_auc_n014 | QUO | 0.939 | 0.777 | 0.746 | 0.880 | 14.000 |
| sae_top_directional_auc_n014 | RE | 0.801 | 0.557 | 0.549 | 0.730 | 14.000 |
| sae_top_directional_auc_n014 | REC | 0.857 | 0.478 | 0.484 | 0.780 | 14.000 |
| sae_top_directional_auc_n014 | RES | 0.806 | 0.267 | 0.297 | 0.735 | 14.000 |
| sae_top_directional_auc_n014 | SU | 0.849 | 0.248 | 0.203 | 0.776 | 14.000 |
| sae_top_directional_auc_n015 | AF | 0.920 | 0.589 | 0.483 | 0.849 | 15.000 |
| sae_top_directional_auc_n015 | GI | 0.790 | 0.294 | 0.350 | 0.721 | 15.000 |
| sae_top_directional_auc_n015 | QU | 0.958 | 0.934 | 0.884 | 0.915 | 15.000 |
| sae_top_directional_auc_n015 | QUC | 0.896 | 0.573 | 0.555 | 0.819 | 15.000 |
| sae_top_directional_auc_n015 | QUO | 0.939 | 0.780 | 0.745 | 0.879 | 15.000 |
| sae_top_directional_auc_n015 | RE | 0.806 | 0.563 | 0.557 | 0.736 | 15.000 |
| sae_top_directional_auc_n015 | REC | 0.861 | 0.494 | 0.488 | 0.782 | 15.000 |
| sae_top_directional_auc_n015 | RES | 0.814 | 0.288 | 0.303 | 0.740 | 15.000 |
| sae_top_directional_auc_n015 | SU | 0.847 | 0.249 | 0.204 | 0.774 | 15.000 |
| sae_top_directional_auc_n016 | AF | 0.920 | 0.590 | 0.488 | 0.849 | 16.000 |
| sae_top_directional_auc_n016 | GI | 0.793 | 0.298 | 0.352 | 0.724 | 16.000 |
| sae_top_directional_auc_n016 | QU | 0.958 | 0.934 | 0.884 | 0.915 | 16.000 |
| sae_top_directional_auc_n016 | QUC | 0.897 | 0.577 | 0.557 | 0.820 | 16.000 |
| sae_top_directional_auc_n016 | QUO | 0.940 | 0.782 | 0.744 | 0.878 | 16.000 |
| sae_top_directional_auc_n016 | RE | 0.806 | 0.564 | 0.559 | 0.738 | 16.000 |
| sae_top_directional_auc_n016 | REC | 0.862 | 0.498 | 0.490 | 0.783 | 16.000 |
| sae_top_directional_auc_n016 | RES | 0.812 | 0.287 | 0.300 | 0.735 | 16.000 |
| sae_top_directional_auc_n016 | SU | 0.848 | 0.252 | 0.203 | 0.771 | 16.000 |
| sae_top_directional_auc_n017 | AF | 0.921 | 0.592 | 0.490 | 0.851 | 17.000 |
| sae_top_directional_auc_n017 | GI | 0.795 | 0.301 | 0.353 | 0.723 | 17.000 |
| sae_top_directional_auc_n017 | QU | 0.959 | 0.934 | 0.884 | 0.915 | 17.000 |
| sae_top_directional_auc_n017 | QUC | 0.898 | 0.580 | 0.562 | 0.822 | 17.000 |
| sae_top_directional_auc_n017 | QUO | 0.938 | 0.783 | 0.742 | 0.877 | 17.000 |
| sae_top_directional_auc_n017 | RE | 0.808 | 0.567 | 0.564 | 0.741 | 17.000 |
| sae_top_directional_auc_n017 | REC | 0.864 | 0.502 | 0.493 | 0.785 | 17.000 |
| sae_top_directional_auc_n017 | RES | 0.813 | 0.285 | 0.304 | 0.740 | 17.000 |
| sae_top_directional_auc_n017 | SU | 0.853 | 0.255 | 0.205 | 0.772 | 17.000 |
| sae_top_directional_auc_n018 | AF | 0.920 | 0.595 | 0.489 | 0.849 | 18.000 |
| sae_top_directional_auc_n018 | GI | 0.797 | 0.308 | 0.357 | 0.725 | 18.000 |
| sae_top_directional_auc_n018 | QU | 0.961 | 0.937 | 0.887 | 0.918 | 18.000 |
| sae_top_directional_auc_n018 | QUC | 0.898 | 0.580 | 0.553 | 0.816 | 18.000 |
| sae_top_directional_auc_n018 | QUO | 0.938 | 0.784 | 0.745 | 0.879 | 18.000 |
| sae_top_directional_auc_n018 | RE | 0.813 | 0.571 | 0.568 | 0.746 | 18.000 |
| sae_top_directional_auc_n018 | REC | 0.868 | 0.510 | 0.497 | 0.788 | 18.000 |
| sae_top_directional_auc_n018 | RES | 0.812 | 0.285 | 0.298 | 0.731 | 18.000 |
| sae_top_directional_auc_n018 | SU | 0.853 | 0.257 | 0.214 | 0.786 | 18.000 |
| sae_top_directional_auc_n019 | AF | 0.921 | 0.609 | 0.507 | 0.858 | 19.000 |
| sae_top_directional_auc_n019 | GI | 0.795 | 0.308 | 0.351 | 0.719 | 19.000 |
| sae_top_directional_auc_n019 | QU | 0.960 | 0.937 | 0.887 | 0.918 | 19.000 |
| sae_top_directional_auc_n019 | QUC | 0.898 | 0.580 | 0.552 | 0.816 | 19.000 |
| sae_top_directional_auc_n019 | QUO | 0.940 | 0.788 | 0.743 | 0.878 | 19.000 |
| sae_top_directional_auc_n019 | RE | 0.820 | 0.571 | 0.575 | 0.751 | 19.000 |
| sae_top_directional_auc_n019 | REC | 0.871 | 0.518 | 0.503 | 0.792 | 19.000 |
| sae_top_directional_auc_n019 | RES | 0.812 | 0.285 | 0.301 | 0.735 | 19.000 |
| sae_top_directional_auc_n019 | SU | 0.855 | 0.260 | 0.210 | 0.781 | 19.000 |
| sae_top_directional_auc_n020 | AF | 0.921 | 0.607 | 0.511 | 0.861 | 20.000 |
| sae_top_directional_auc_n020 | GI | 0.798 | 0.310 | 0.360 | 0.726 | 20.000 |
| sae_top_directional_auc_n020 | QU | 0.962 | 0.938 | 0.889 | 0.919 | 20.000 |
| sae_top_directional_auc_n020 | QUC | 0.901 | 0.585 | 0.555 | 0.819 | 20.000 |
| sae_top_directional_auc_n020 | QUO | 0.940 | 0.796 | 0.744 | 0.878 | 20.000 |
| sae_top_directional_auc_n020 | RE | 0.823 | 0.574 | 0.576 | 0.752 | 20.000 |
| sae_top_directional_auc_n020 | REC | 0.872 | 0.518 | 0.504 | 0.793 | 20.000 |
| sae_top_directional_auc_n020 | RES | 0.813 | 0.286 | 0.303 | 0.737 | 20.000 |
| sae_top_directional_auc_n020 | SU | 0.853 | 0.262 | 0.212 | 0.784 | 20.000 |
| sae_top_directional_auc_n021 | AF | 0.920 | 0.606 | 0.502 | 0.855 | 21.000 |
| sae_top_directional_auc_n021 | GI | 0.799 | 0.308 | 0.356 | 0.721 | 21.000 |
| sae_top_directional_auc_n021 | QU | 0.962 | 0.938 | 0.888 | 0.919 | 21.000 |
| sae_top_directional_auc_n021 | QUC | 0.900 | 0.583 | 0.553 | 0.816 | 21.000 |
| sae_top_directional_auc_n021 | QUO | 0.942 | 0.798 | 0.747 | 0.878 | 21.000 |
| sae_top_directional_auc_n021 | RE | 0.826 | 0.575 | 0.577 | 0.753 | 21.000 |
| sae_top_directional_auc_n021 | REC | 0.873 | 0.517 | 0.505 | 0.794 | 21.000 |
| sae_top_directional_auc_n021 | RES | 0.814 | 0.286 | 0.302 | 0.734 | 21.000 |
| sae_top_directional_auc_n021 | SU | 0.853 | 0.266 | 0.205 | 0.774 | 21.000 |
| sae_top_directional_auc_n022 | AF | 0.924 | 0.619 | 0.507 | 0.857 | 22.000 |
| sae_top_directional_auc_n022 | GI | 0.802 | 0.314 | 0.361 | 0.727 | 22.000 |
| sae_top_directional_auc_n022 | QU | 0.964 | 0.940 | 0.892 | 0.922 | 22.000 |
| sae_top_directional_auc_n022 | QUC | 0.901 | 0.583 | 0.553 | 0.817 | 22.000 |
| sae_top_directional_auc_n022 | QUO | 0.943 | 0.804 | 0.747 | 0.878 | 22.000 |
| sae_top_directional_auc_n022 | RE | 0.832 | 0.587 | 0.580 | 0.757 | 22.000 |
| sae_top_directional_auc_n022 | REC | 0.874 | 0.530 | 0.509 | 0.794 | 22.000 |
| sae_top_directional_auc_n022 | RES | 0.813 | 0.288 | 0.304 | 0.733 | 22.000 |
| sae_top_directional_auc_n022 | SU | 0.852 | 0.259 | 0.204 | 0.770 | 22.000 |
| sae_top_directional_auc_n023 | AF | 0.925 | 0.617 | 0.510 | 0.861 | 23.000 |
| sae_top_directional_auc_n023 | GI | 0.801 | 0.314 | 0.361 | 0.726 | 23.000 |
| sae_top_directional_auc_n023 | QU | 0.964 | 0.940 | 0.894 | 0.923 | 23.000 |
| sae_top_directional_auc_n023 | QUC | 0.908 | 0.595 | 0.569 | 0.829 | 23.000 |
| sae_top_directional_auc_n023 | QUO | 0.943 | 0.805 | 0.747 | 0.878 | 23.000 |
| sae_top_directional_auc_n023 | RE | 0.850 | 0.620 | 0.589 | 0.765 | 23.000 |
| sae_top_directional_auc_n023 | REC | 0.875 | 0.531 | 0.509 | 0.794 | 23.000 |
| sae_top_directional_auc_n023 | RES | 0.813 | 0.288 | 0.306 | 0.736 | 23.000 |
| sae_top_directional_auc_n023 | SU | 0.853 | 0.259 | 0.203 | 0.768 | 23.000 |
| sae_top_directional_auc_n024 | AF | 0.925 | 0.618 | 0.511 | 0.861 | 24.000 |
| sae_top_directional_auc_n024 | GI | 0.804 | 0.317 | 0.367 | 0.731 | 24.000 |
| sae_top_directional_auc_n024 | QU | 0.964 | 0.941 | 0.894 | 0.923 | 24.000 |
| sae_top_directional_auc_n024 | QUC | 0.909 | 0.595 | 0.571 | 0.831 | 24.000 |
| sae_top_directional_auc_n024 | QUO | 0.943 | 0.809 | 0.747 | 0.878 | 24.000 |
| sae_top_directional_auc_n024 | RE | 0.853 | 0.625 | 0.595 | 0.769 | 24.000 |
| sae_top_directional_auc_n024 | REC | 0.877 | 0.541 | 0.510 | 0.795 | 24.000 |
| sae_top_directional_auc_n024 | RES | 0.814 | 0.289 | 0.309 | 0.741 | 24.000 |
| sae_top_directional_auc_n024 | SU | 0.858 | 0.272 | 0.205 | 0.765 | 24.000 |
| sae_top_directional_auc_n025 | AF | 0.925 | 0.619 | 0.516 | 0.864 | 25.000 |
| sae_top_directional_auc_n025 | GI | 0.805 | 0.324 | 0.362 | 0.725 | 25.000 |
| sae_top_directional_auc_n025 | QU | 0.966 | 0.942 | 0.894 | 0.923 | 25.000 |
| sae_top_directional_auc_n025 | QUC | 0.909 | 0.595 | 0.573 | 0.832 | 25.000 |
| sae_top_directional_auc_n025 | QUO | 0.944 | 0.811 | 0.745 | 0.875 | 25.000 |
| sae_top_directional_auc_n025 | RE | 0.860 | 0.635 | 0.601 | 0.774 | 25.000 |
| sae_top_directional_auc_n025 | REC | 0.878 | 0.546 | 0.514 | 0.798 | 25.000 |
| sae_top_directional_auc_n025 | RES | 0.812 | 0.287 | 0.309 | 0.737 | 25.000 |
| sae_top_directional_auc_n025 | SU | 0.858 | 0.270 | 0.203 | 0.762 | 25.000 |
| sae_top_directional_auc_n026 | AF | 0.926 | 0.619 | 0.517 | 0.864 | 26.000 |
| sae_top_directional_auc_n026 | GI | 0.808 | 0.329 | 0.364 | 0.727 | 26.000 |
| sae_top_directional_auc_n026 | QU | 0.967 | 0.943 | 0.895 | 0.923 | 26.000 |
| sae_top_directional_auc_n026 | QUC | 0.910 | 0.599 | 0.574 | 0.833 | 26.000 |
| sae_top_directional_auc_n026 | QUO | 0.944 | 0.813 | 0.744 | 0.874 | 26.000 |
| sae_top_directional_auc_n026 | RE | 0.860 | 0.635 | 0.602 | 0.774 | 26.000 |
| sae_top_directional_auc_n026 | REC | 0.881 | 0.559 | 0.514 | 0.796 | 26.000 |
| sae_top_directional_auc_n026 | RES | 0.813 | 0.288 | 0.311 | 0.739 | 26.000 |
| sae_top_directional_auc_n026 | SU | 0.859 | 0.270 | 0.204 | 0.764 | 26.000 |
| sae_top_directional_auc_n027 | AF | 0.927 | 0.621 | 0.509 | 0.864 | 27.000 |
| sae_top_directional_auc_n027 | GI | 0.808 | 0.332 | 0.364 | 0.727 | 27.000 |
| sae_top_directional_auc_n027 | QU | 0.967 | 0.943 | 0.895 | 0.924 | 27.000 |
| sae_top_directional_auc_n027 | QUC | 0.911 | 0.598 | 0.576 | 0.834 | 27.000 |
| sae_top_directional_auc_n027 | QUO | 0.944 | 0.813 | 0.744 | 0.874 | 27.000 |
| sae_top_directional_auc_n027 | RE | 0.864 | 0.644 | 0.605 | 0.776 | 27.000 |
| sae_top_directional_auc_n027 | REC | 0.881 | 0.558 | 0.517 | 0.799 | 27.000 |
| sae_top_directional_auc_n027 | RES | 0.813 | 0.286 | 0.311 | 0.739 | 27.000 |
| sae_top_directional_auc_n027 | SU | 0.858 | 0.270 | 0.202 | 0.762 | 27.000 |
| sae_top_directional_auc_n028 | AF | 0.926 | 0.618 | 0.511 | 0.865 | 28.000 |
| sae_top_directional_auc_n028 | GI | 0.808 | 0.336 | 0.370 | 0.732 | 28.000 |
| sae_top_directional_auc_n028 | QU | 0.968 | 0.943 | 0.894 | 0.923 | 28.000 |
| sae_top_directional_auc_n028 | QUC | 0.911 | 0.597 | 0.576 | 0.834 | 28.000 |
| sae_top_directional_auc_n028 | QUO | 0.945 | 0.816 | 0.745 | 0.875 | 28.000 |
| sae_top_directional_auc_n028 | RE | 0.866 | 0.643 | 0.612 | 0.781 | 28.000 |
| sae_top_directional_auc_n028 | REC | 0.881 | 0.560 | 0.516 | 0.799 | 28.000 |
| sae_top_directional_auc_n028 | RES | 0.813 | 0.285 | 0.311 | 0.738 | 28.000 |
| sae_top_directional_auc_n028 | SU | 0.859 | 0.271 | 0.201 | 0.759 | 28.000 |
| sae_top_directional_auc_n029 | AF | 0.925 | 0.617 | 0.514 | 0.866 | 29.000 |
| sae_top_directional_auc_n029 | GI | 0.809 | 0.338 | 0.374 | 0.736 | 29.000 |
| sae_top_directional_auc_n029 | QU | 0.968 | 0.943 | 0.894 | 0.923 | 29.000 |
| sae_top_directional_auc_n029 | QUC | 0.911 | 0.595 | 0.579 | 0.838 | 29.000 |
| sae_top_directional_auc_n029 | QUO | 0.945 | 0.815 | 0.746 | 0.875 | 29.000 |
| sae_top_directional_auc_n029 | RE | 0.866 | 0.643 | 0.610 | 0.779 | 29.000 |
| sae_top_directional_auc_n029 | REC | 0.882 | 0.561 | 0.520 | 0.801 | 29.000 |
| sae_top_directional_auc_n029 | RES | 0.813 | 0.289 | 0.309 | 0.736 | 29.000 |
| sae_top_directional_auc_n029 | SU | 0.859 | 0.272 | 0.203 | 0.761 | 29.000 |
| sae_top_directional_auc_n030 | AF | 0.927 | 0.615 | 0.511 | 0.863 | 30.000 |
| sae_top_directional_auc_n030 | GI | 0.809 | 0.341 | 0.376 | 0.739 | 30.000 |
| sae_top_directional_auc_n030 | QU | 0.968 | 0.944 | 0.894 | 0.923 | 30.000 |
| sae_top_directional_auc_n030 | QUC | 0.911 | 0.596 | 0.579 | 0.839 | 30.000 |
| sae_top_directional_auc_n030 | QUO | 0.945 | 0.815 | 0.745 | 0.875 | 30.000 |
| sae_top_directional_auc_n030 | RE | 0.871 | 0.655 | 0.616 | 0.784 | 30.000 |
| sae_top_directional_auc_n030 | REC | 0.883 | 0.565 | 0.521 | 0.800 | 30.000 |
| sae_top_directional_auc_n030 | RES | 0.813 | 0.290 | 0.311 | 0.738 | 30.000 |
| sae_top_directional_auc_n030 | SU | 0.857 | 0.271 | 0.206 | 0.761 | 30.000 |
| sae_top_directional_auc_n031 | AF | 0.924 | 0.612 | 0.505 | 0.860 | 31.000 |
| sae_top_directional_auc_n031 | GI | 0.810 | 0.342 | 0.372 | 0.734 | 31.000 |
| sae_top_directional_auc_n031 | QU | 0.968 | 0.944 | 0.894 | 0.923 | 31.000 |
| sae_top_directional_auc_n031 | QUC | 0.913 | 0.599 | 0.576 | 0.837 | 31.000 |
| sae_top_directional_auc_n031 | QUO | 0.944 | 0.815 | 0.745 | 0.874 | 31.000 |
| sae_top_directional_auc_n031 | RE | 0.878 | 0.672 | 0.624 | 0.792 | 31.000 |
| sae_top_directional_auc_n031 | REC | 0.884 | 0.568 | 0.526 | 0.804 | 31.000 |
| sae_top_directional_auc_n031 | RES | 0.812 | 0.283 | 0.313 | 0.740 | 31.000 |
| sae_top_directional_auc_n031 | SU | 0.859 | 0.266 | 0.206 | 0.766 | 31.000 |
| sae_top_directional_auc_n032 | AF | 0.923 | 0.612 | 0.503 | 0.859 | 32.000 |
| sae_top_directional_auc_n032 | GI | 0.814 | 0.355 | 0.376 | 0.738 | 32.000 |
| sae_top_directional_auc_n032 | QU | 0.968 | 0.943 | 0.894 | 0.923 | 32.000 |
| sae_top_directional_auc_n032 | QUC | 0.915 | 0.605 | 0.580 | 0.838 | 32.000 |
| sae_top_directional_auc_n032 | QUO | 0.944 | 0.814 | 0.745 | 0.874 | 32.000 |
| sae_top_directional_auc_n032 | RE | 0.878 | 0.675 | 0.625 | 0.793 | 32.000 |
| sae_top_directional_auc_n032 | REC | 0.884 | 0.569 | 0.522 | 0.802 | 32.000 |
| sae_top_directional_auc_n032 | RES | 0.811 | 0.281 | 0.310 | 0.737 | 32.000 |
| sae_top_directional_auc_n032 | SU | 0.857 | 0.264 | 0.205 | 0.762 | 32.000 |
| sae_top_directional_auc_n033 | AF | 0.923 | 0.612 | 0.507 | 0.862 | 33.000 |
| sae_top_directional_auc_n033 | GI | 0.813 | 0.358 | 0.378 | 0.740 | 33.000 |
| sae_top_directional_auc_n033 | QU | 0.968 | 0.943 | 0.895 | 0.924 | 33.000 |
| sae_top_directional_auc_n033 | QUC | 0.915 | 0.604 | 0.579 | 0.838 | 33.000 |
| sae_top_directional_auc_n033 | QUO | 0.944 | 0.813 | 0.743 | 0.873 | 33.000 |
| sae_top_directional_auc_n033 | RE | 0.881 | 0.680 | 0.628 | 0.795 | 33.000 |
| sae_top_directional_auc_n033 | REC | 0.885 | 0.569 | 0.524 | 0.802 | 33.000 |
| sae_top_directional_auc_n033 | RES | 0.813 | 0.285 | 0.307 | 0.733 | 33.000 |
| sae_top_directional_auc_n033 | SU | 0.859 | 0.264 | 0.203 | 0.758 | 33.000 |
| sae_top_directional_auc_n034 | AF | 0.923 | 0.608 | 0.508 | 0.863 | 34.000 |
| sae_top_directional_auc_n034 | GI | 0.813 | 0.360 | 0.378 | 0.739 | 34.000 |
| sae_top_directional_auc_n034 | QU | 0.969 | 0.945 | 0.896 | 0.924 | 34.000 |
| sae_top_directional_auc_n034 | QUC | 0.915 | 0.603 | 0.578 | 0.836 | 34.000 |
| sae_top_directional_auc_n034 | QUO | 0.943 | 0.813 | 0.743 | 0.874 | 34.000 |
| sae_top_directional_auc_n034 | RE | 0.883 | 0.682 | 0.631 | 0.798 | 34.000 |
| sae_top_directional_auc_n034 | REC | 0.886 | 0.570 | 0.527 | 0.806 | 34.000 |
| sae_top_directional_auc_n034 | RES | 0.813 | 0.291 | 0.311 | 0.737 | 34.000 |
| sae_top_directional_auc_n034 | SU | 0.862 | 0.273 | 0.204 | 0.761 | 34.000 |
| sae_top_directional_auc_n035 | AF | 0.925 | 0.611 | 0.502 | 0.862 | 35.000 |
| sae_top_directional_auc_n035 | GI | 0.812 | 0.359 | 0.376 | 0.735 | 35.000 |
| sae_top_directional_auc_n035 | QU | 0.969 | 0.945 | 0.895 | 0.924 | 35.000 |
| sae_top_directional_auc_n035 | QUC | 0.915 | 0.603 | 0.579 | 0.837 | 35.000 |
| sae_top_directional_auc_n035 | QUO | 0.943 | 0.813 | 0.743 | 0.874 | 35.000 |
| sae_top_directional_auc_n035 | RE | 0.883 | 0.681 | 0.632 | 0.798 | 35.000 |
| sae_top_directional_auc_n035 | REC | 0.889 | 0.580 | 0.527 | 0.805 | 35.000 |
| sae_top_directional_auc_n035 | RES | 0.815 | 0.290 | 0.314 | 0.741 | 35.000 |
| sae_top_directional_auc_n035 | SU | 0.865 | 0.282 | 0.217 | 0.778 | 35.000 |
| sae_top_directional_auc_n036 | AF | 0.925 | 0.609 | 0.503 | 0.861 | 36.000 |
| sae_top_directional_auc_n036 | GI | 0.813 | 0.360 | 0.376 | 0.734 | 36.000 |
| sae_top_directional_auc_n036 | QU | 0.969 | 0.945 | 0.895 | 0.924 | 36.000 |
| sae_top_directional_auc_n036 | QUC | 0.915 | 0.605 | 0.586 | 0.842 | 36.000 |
| sae_top_directional_auc_n036 | QUO | 0.945 | 0.817 | 0.746 | 0.875 | 36.000 |
| sae_top_directional_auc_n036 | RE | 0.882 | 0.678 | 0.633 | 0.799 | 36.000 |
| sae_top_directional_auc_n036 | REC | 0.890 | 0.583 | 0.529 | 0.807 | 36.000 |
| sae_top_directional_auc_n036 | RES | 0.817 | 0.295 | 0.316 | 0.743 | 36.000 |
| sae_top_directional_auc_n036 | SU | 0.865 | 0.275 | 0.217 | 0.778 | 36.000 |
| sae_top_directional_auc_n037 | AF | 0.925 | 0.609 | 0.499 | 0.858 | 37.000 |
| sae_top_directional_auc_n037 | GI | 0.815 | 0.362 | 0.376 | 0.734 | 37.000 |
| sae_top_directional_auc_n037 | QU | 0.969 | 0.946 | 0.895 | 0.923 | 37.000 |
| sae_top_directional_auc_n037 | QUC | 0.916 | 0.605 | 0.585 | 0.841 | 37.000 |
| sae_top_directional_auc_n037 | QUO | 0.945 | 0.818 | 0.747 | 0.876 | 37.000 |
| sae_top_directional_auc_n037 | RE | 0.883 | 0.679 | 0.633 | 0.797 | 37.000 |
| sae_top_directional_auc_n037 | REC | 0.891 | 0.587 | 0.534 | 0.811 | 37.000 |
| sae_top_directional_auc_n037 | RES | 0.818 | 0.294 | 0.316 | 0.743 | 37.000 |
| sae_top_directional_auc_n037 | SU | 0.867 | 0.281 | 0.220 | 0.781 | 37.000 |
| sae_top_directional_auc_n038 | AF | 0.925 | 0.608 | 0.498 | 0.858 | 38.000 |
| sae_top_directional_auc_n038 | GI | 0.814 | 0.360 | 0.377 | 0.734 | 38.000 |
| sae_top_directional_auc_n038 | QU | 0.970 | 0.946 | 0.895 | 0.924 | 38.000 |
| sae_top_directional_auc_n038 | QUC | 0.915 | 0.606 | 0.586 | 0.841 | 38.000 |
| sae_top_directional_auc_n038 | QUO | 0.948 | 0.820 | 0.751 | 0.881 | 38.000 |
| sae_top_directional_auc_n038 | RE | 0.884 | 0.681 | 0.634 | 0.798 | 38.000 |
| sae_top_directional_auc_n038 | REC | 0.891 | 0.586 | 0.532 | 0.809 | 38.000 |
| sae_top_directional_auc_n038 | RES | 0.820 | 0.299 | 0.320 | 0.746 | 38.000 |
| sae_top_directional_auc_n038 | SU | 0.861 | 0.271 | 0.218 | 0.774 | 38.000 |
| sae_top_directional_auc_n039 | AF | 0.923 | 0.608 | 0.499 | 0.857 | 39.000 |
| sae_top_directional_auc_n039 | GI | 0.816 | 0.365 | 0.380 | 0.736 | 39.000 |
| sae_top_directional_auc_n039 | QU | 0.970 | 0.946 | 0.895 | 0.924 | 39.000 |
| sae_top_directional_auc_n039 | QUC | 0.916 | 0.607 | 0.589 | 0.842 | 39.000 |
| sae_top_directional_auc_n039 | QUO | 0.948 | 0.824 | 0.750 | 0.880 | 39.000 |
| sae_top_directional_auc_n039 | RE | 0.884 | 0.682 | 0.637 | 0.800 | 39.000 |
| sae_top_directional_auc_n039 | REC | 0.890 | 0.583 | 0.534 | 0.811 | 39.000 |
| sae_top_directional_auc_n039 | RES | 0.820 | 0.296 | 0.318 | 0.744 | 39.000 |
| sae_top_directional_auc_n039 | SU | 0.861 | 0.269 | 0.211 | 0.764 | 39.000 |
| sae_top_directional_auc_n040 | AF | 0.923 | 0.607 | 0.496 | 0.856 | 40.000 |
| sae_top_directional_auc_n040 | GI | 0.818 | 0.367 | 0.380 | 0.736 | 40.000 |
| sae_top_directional_auc_n040 | QU | 0.970 | 0.947 | 0.896 | 0.925 | 40.000 |
| sae_top_directional_auc_n040 | QUC | 0.917 | 0.607 | 0.588 | 0.840 | 40.000 |
| sae_top_directional_auc_n040 | QUO | 0.948 | 0.824 | 0.750 | 0.879 | 40.000 |
| sae_top_directional_auc_n040 | RE | 0.885 | 0.684 | 0.634 | 0.797 | 40.000 |
| sae_top_directional_auc_n040 | REC | 0.890 | 0.582 | 0.535 | 0.812 | 40.000 |
| sae_top_directional_auc_n040 | RES | 0.819 | 0.296 | 0.319 | 0.745 | 40.000 |
| sae_top_directional_auc_n040 | SU | 0.862 | 0.271 | 0.216 | 0.771 | 40.000 |
| sae_top_directional_auc_n041 | AF | 0.923 | 0.606 | 0.501 | 0.857 | 41.000 |
| sae_top_directional_auc_n041 | GI | 0.818 | 0.372 | 0.380 | 0.737 | 41.000 |
| sae_top_directional_auc_n041 | QU | 0.970 | 0.947 | 0.896 | 0.924 | 41.000 |
| sae_top_directional_auc_n041 | QUC | 0.916 | 0.606 | 0.590 | 0.842 | 41.000 |
| sae_top_directional_auc_n041 | QUO | 0.949 | 0.828 | 0.758 | 0.885 | 41.000 |
| sae_top_directional_auc_n041 | RE | 0.886 | 0.690 | 0.636 | 0.799 | 41.000 |
| sae_top_directional_auc_n041 | REC | 0.891 | 0.583 | 0.535 | 0.810 | 41.000 |
| sae_top_directional_auc_n041 | RES | 0.819 | 0.295 | 0.321 | 0.747 | 41.000 |
| sae_top_directional_auc_n041 | SU | 0.862 | 0.271 | 0.215 | 0.768 | 41.000 |
| sae_top_directional_auc_n042 | AF | 0.924 | 0.605 | 0.507 | 0.857 | 42.000 |
| sae_top_directional_auc_n042 | GI | 0.819 | 0.377 | 0.381 | 0.738 | 42.000 |
| sae_top_directional_auc_n042 | QU | 0.970 | 0.948 | 0.897 | 0.925 | 42.000 |
| sae_top_directional_auc_n042 | QUC | 0.916 | 0.606 | 0.588 | 0.840 | 42.000 |
| sae_top_directional_auc_n042 | QUO | 0.949 | 0.830 | 0.758 | 0.885 | 42.000 |
| sae_top_directional_auc_n042 | RE | 0.887 | 0.693 | 0.639 | 0.801 | 42.000 |
| sae_top_directional_auc_n042 | REC | 0.891 | 0.583 | 0.535 | 0.810 | 42.000 |
| sae_top_directional_auc_n042 | RES | 0.820 | 0.291 | 0.320 | 0.745 | 42.000 |
| sae_top_directional_auc_n042 | SU | 0.860 | 0.274 | 0.212 | 0.761 | 42.000 |
| sae_top_directional_auc_n043 | AF | 0.923 | 0.604 | 0.508 | 0.858 | 43.000 |
| sae_top_directional_auc_n043 | GI | 0.819 | 0.377 | 0.383 | 0.740 | 43.000 |
| sae_top_directional_auc_n043 | QU | 0.970 | 0.948 | 0.897 | 0.925 | 43.000 |
| sae_top_directional_auc_n043 | QUC | 0.916 | 0.610 | 0.584 | 0.837 | 43.000 |
| sae_top_directional_auc_n043 | QUO | 0.950 | 0.833 | 0.754 | 0.882 | 43.000 |
| sae_top_directional_auc_n043 | RE | 0.887 | 0.694 | 0.641 | 0.803 | 43.000 |
| sae_top_directional_auc_n043 | REC | 0.891 | 0.585 | 0.535 | 0.809 | 43.000 |
| sae_top_directional_auc_n043 | RES | 0.821 | 0.294 | 0.319 | 0.743 | 43.000 |
| sae_top_directional_auc_n043 | SU | 0.859 | 0.267 | 0.217 | 0.766 | 43.000 |
| sae_top_directional_auc_n044 | AF | 0.924 | 0.605 | 0.508 | 0.858 | 44.000 |
| sae_top_directional_auc_n044 | GI | 0.820 | 0.381 | 0.383 | 0.740 | 44.000 |
| sae_top_directional_auc_n044 | QU | 0.971 | 0.949 | 0.896 | 0.924 | 44.000 |
| sae_top_directional_auc_n044 | QUC | 0.917 | 0.612 | 0.586 | 0.838 | 44.000 |
| sae_top_directional_auc_n044 | QUO | 0.950 | 0.833 | 0.754 | 0.882 | 44.000 |
| sae_top_directional_auc_n044 | RE | 0.887 | 0.694 | 0.640 | 0.802 | 44.000 |
| sae_top_directional_auc_n044 | REC | 0.891 | 0.583 | 0.534 | 0.808 | 44.000 |
| sae_top_directional_auc_n044 | RES | 0.820 | 0.297 | 0.321 | 0.743 | 44.000 |
| sae_top_directional_auc_n044 | SU | 0.860 | 0.270 | 0.220 | 0.769 | 44.000 |
| sae_top_directional_auc_n045 | AF | 0.925 | 0.606 | 0.510 | 0.862 | 45.000 |
| sae_top_directional_auc_n045 | GI | 0.821 | 0.381 | 0.385 | 0.742 | 45.000 |
| sae_top_directional_auc_n045 | QU | 0.971 | 0.949 | 0.896 | 0.924 | 45.000 |
| sae_top_directional_auc_n045 | QUC | 0.917 | 0.612 | 0.588 | 0.840 | 45.000 |
| sae_top_directional_auc_n045 | QUO | 0.950 | 0.833 | 0.754 | 0.882 | 45.000 |
| sae_top_directional_auc_n045 | RE | 0.887 | 0.695 | 0.640 | 0.802 | 45.000 |
| sae_top_directional_auc_n045 | REC | 0.891 | 0.583 | 0.536 | 0.808 | 45.000 |
| sae_top_directional_auc_n045 | RES | 0.822 | 0.298 | 0.321 | 0.743 | 45.000 |
| sae_top_directional_auc_n045 | SU | 0.860 | 0.271 | 0.226 | 0.775 | 45.000 |
| sae_top_directional_auc_n046 | AF | 0.925 | 0.608 | 0.505 | 0.861 | 46.000 |
| sae_top_directional_auc_n046 | GI | 0.822 | 0.383 | 0.387 | 0.742 | 46.000 |
| sae_top_directional_auc_n046 | QU | 0.971 | 0.949 | 0.898 | 0.925 | 46.000 |
| sae_top_directional_auc_n046 | QUC | 0.917 | 0.612 | 0.585 | 0.837 | 46.000 |
| sae_top_directional_auc_n046 | QUO | 0.950 | 0.834 | 0.755 | 0.883 | 46.000 |
| sae_top_directional_auc_n046 | RE | 0.888 | 0.698 | 0.640 | 0.802 | 46.000 |
| sae_top_directional_auc_n046 | REC | 0.891 | 0.585 | 0.536 | 0.807 | 46.000 |
| sae_top_directional_auc_n046 | RES | 0.823 | 0.299 | 0.324 | 0.746 | 46.000 |
| sae_top_directional_auc_n046 | SU | 0.859 | 0.271 | 0.220 | 0.766 | 46.000 |
| sae_top_directional_auc_n047 | AF | 0.925 | 0.609 | 0.506 | 0.863 | 47.000 |
| sae_top_directional_auc_n047 | GI | 0.823 | 0.382 | 0.394 | 0.749 | 47.000 |
| sae_top_directional_auc_n047 | QU | 0.970 | 0.949 | 0.896 | 0.925 | 47.000 |
| sae_top_directional_auc_n047 | QUC | 0.917 | 0.613 | 0.595 | 0.844 | 47.000 |
| sae_top_directional_auc_n047 | QUO | 0.951 | 0.834 | 0.759 | 0.885 | 47.000 |
| sae_top_directional_auc_n047 | RE | 0.889 | 0.700 | 0.643 | 0.804 | 47.000 |
| sae_top_directional_auc_n047 | REC | 0.891 | 0.586 | 0.538 | 0.810 | 47.000 |
| sae_top_directional_auc_n047 | RES | 0.824 | 0.298 | 0.322 | 0.744 | 47.000 |
| sae_top_directional_auc_n047 | SU | 0.859 | 0.271 | 0.220 | 0.764 | 47.000 |
| sae_top_directional_auc_n048 | AF | 0.926 | 0.607 | 0.504 | 0.860 | 48.000 |
| sae_top_directional_auc_n048 | GI | 0.823 | 0.382 | 0.393 | 0.748 | 48.000 |
| sae_top_directional_auc_n048 | QU | 0.970 | 0.948 | 0.898 | 0.925 | 48.000 |
| sae_top_directional_auc_n048 | QUC | 0.917 | 0.616 | 0.594 | 0.843 | 48.000 |
| sae_top_directional_auc_n048 | QUO | 0.951 | 0.835 | 0.757 | 0.883 | 48.000 |
| sae_top_directional_auc_n048 | RE | 0.888 | 0.699 | 0.641 | 0.803 | 48.000 |
| sae_top_directional_auc_n048 | REC | 0.892 | 0.592 | 0.542 | 0.812 | 48.000 |
| sae_top_directional_auc_n048 | RES | 0.824 | 0.299 | 0.321 | 0.744 | 48.000 |
| sae_top_directional_auc_n048 | SU | 0.859 | 0.277 | 0.224 | 0.768 | 48.000 |
| sae_top_directional_auc_n049 | AF | 0.925 | 0.608 | 0.503 | 0.861 | 49.000 |
| sae_top_directional_auc_n049 | GI | 0.823 | 0.384 | 0.394 | 0.748 | 49.000 |
| sae_top_directional_auc_n049 | QU | 0.970 | 0.949 | 0.899 | 0.926 | 49.000 |
| sae_top_directional_auc_n049 | QUC | 0.917 | 0.613 | 0.595 | 0.843 | 49.000 |
| sae_top_directional_auc_n049 | QUO | 0.952 | 0.837 | 0.759 | 0.884 | 49.000 |
| sae_top_directional_auc_n049 | RE | 0.888 | 0.700 | 0.641 | 0.803 | 49.000 |
| sae_top_directional_auc_n049 | REC | 0.893 | 0.597 | 0.544 | 0.813 | 49.000 |
| sae_top_directional_auc_n049 | RES | 0.823 | 0.297 | 0.317 | 0.738 | 49.000 |
| sae_top_directional_auc_n049 | SU | 0.859 | 0.282 | 0.223 | 0.767 | 49.000 |
| sae_top_directional_auc_n050 | AF | 0.927 | 0.612 | 0.502 | 0.861 | 50.000 |
| sae_top_directional_auc_n050 | GI | 0.823 | 0.382 | 0.396 | 0.750 | 50.000 |
| sae_top_directional_auc_n050 | QU | 0.970 | 0.949 | 0.900 | 0.927 | 50.000 |
| sae_top_directional_auc_n050 | QUC | 0.917 | 0.615 | 0.596 | 0.842 | 50.000 |
| sae_top_directional_auc_n050 | QUO | 0.952 | 0.838 | 0.758 | 0.882 | 50.000 |
| sae_top_directional_auc_n050 | RE | 0.889 | 0.700 | 0.643 | 0.805 | 50.000 |
| sae_top_directional_auc_n050 | REC | 0.893 | 0.597 | 0.543 | 0.812 | 50.000 |
| sae_top_directional_auc_n050 | RES | 0.823 | 0.295 | 0.321 | 0.744 | 50.000 |
| sae_top_directional_auc_n050 | SU | 0.860 | 0.285 | 0.226 | 0.772 | 50.000 |
| sae_top_directional_auc_n051 | AF | 0.929 | 0.613 | 0.503 | 0.865 | 51.000 |
| sae_top_directional_auc_n051 | GI | 0.826 | 0.386 | 0.396 | 0.749 | 51.000 |
| sae_top_directional_auc_n051 | QU | 0.971 | 0.949 | 0.901 | 0.927 | 51.000 |
| sae_top_directional_auc_n051 | QUC | 0.919 | 0.624 | 0.599 | 0.844 | 51.000 |
| sae_top_directional_auc_n051 | QUO | 0.952 | 0.840 | 0.759 | 0.883 | 51.000 |
| sae_top_directional_auc_n051 | RE | 0.888 | 0.700 | 0.642 | 0.803 | 51.000 |
| sae_top_directional_auc_n051 | REC | 0.893 | 0.599 | 0.546 | 0.815 | 51.000 |
| sae_top_directional_auc_n051 | RES | 0.822 | 0.295 | 0.317 | 0.739 | 51.000 |
| sae_top_directional_auc_n051 | SU | 0.860 | 0.276 | 0.232 | 0.778 | 51.000 |
| sae_top_directional_auc_n052 | AF | 0.931 | 0.614 | 0.502 | 0.862 | 52.000 |
| sae_top_directional_auc_n052 | GI | 0.829 | 0.393 | 0.400 | 0.752 | 52.000 |
| sae_top_directional_auc_n052 | QU | 0.972 | 0.950 | 0.902 | 0.928 | 52.000 |
| sae_top_directional_auc_n052 | QUC | 0.919 | 0.624 | 0.598 | 0.843 | 52.000 |
| sae_top_directional_auc_n052 | QUO | 0.953 | 0.841 | 0.755 | 0.881 | 52.000 |
| sae_top_directional_auc_n052 | RE | 0.889 | 0.702 | 0.642 | 0.804 | 52.000 |
| sae_top_directional_auc_n052 | REC | 0.893 | 0.598 | 0.544 | 0.812 | 52.000 |
| sae_top_directional_auc_n052 | RES | 0.824 | 0.297 | 0.314 | 0.736 | 52.000 |
| sae_top_directional_auc_n052 | SU | 0.860 | 0.273 | 0.234 | 0.781 | 52.000 |
| sae_top_directional_auc_n053 | AF | 0.931 | 0.613 | 0.501 | 0.861 | 53.000 |
| sae_top_directional_auc_n053 | GI | 0.830 | 0.395 | 0.403 | 0.754 | 53.000 |
| sae_top_directional_auc_n053 | QU | 0.972 | 0.950 | 0.901 | 0.927 | 53.000 |
| sae_top_directional_auc_n053 | QUC | 0.920 | 0.626 | 0.599 | 0.845 | 53.000 |
| sae_top_directional_auc_n053 | QUO | 0.953 | 0.841 | 0.758 | 0.882 | 53.000 |
| sae_top_directional_auc_n053 | RE | 0.889 | 0.700 | 0.641 | 0.803 | 53.000 |
| sae_top_directional_auc_n053 | REC | 0.893 | 0.597 | 0.542 | 0.811 | 53.000 |
| sae_top_directional_auc_n053 | RES | 0.825 | 0.301 | 0.317 | 0.739 | 53.000 |
| sae_top_directional_auc_n053 | SU | 0.860 | 0.276 | 0.235 | 0.781 | 53.000 |
| sae_top_directional_auc_n054 | AF | 0.931 | 0.615 | 0.498 | 0.859 | 54.000 |
| sae_top_directional_auc_n054 | GI | 0.831 | 0.399 | 0.403 | 0.755 | 54.000 |
| sae_top_directional_auc_n054 | QU | 0.972 | 0.950 | 0.901 | 0.928 | 54.000 |
| sae_top_directional_auc_n054 | QUC | 0.920 | 0.628 | 0.603 | 0.846 | 54.000 |
| sae_top_directional_auc_n054 | QUO | 0.953 | 0.840 | 0.758 | 0.882 | 54.000 |
| sae_top_directional_auc_n054 | RE | 0.890 | 0.702 | 0.641 | 0.803 | 54.000 |
| sae_top_directional_auc_n054 | REC | 0.893 | 0.598 | 0.542 | 0.811 | 54.000 |
| sae_top_directional_auc_n054 | RES | 0.825 | 0.301 | 0.319 | 0.741 | 54.000 |
| sae_top_directional_auc_n054 | SU | 0.859 | 0.272 | 0.237 | 0.784 | 54.000 |
| sae_top_directional_auc_n055 | AF | 0.932 | 0.614 | 0.498 | 0.860 | 55.000 |
| sae_top_directional_auc_n055 | GI | 0.832 | 0.397 | 0.403 | 0.753 | 55.000 |
| sae_top_directional_auc_n055 | QU | 0.972 | 0.951 | 0.901 | 0.928 | 55.000 |
| sae_top_directional_auc_n055 | QUC | 0.921 | 0.631 | 0.603 | 0.847 | 55.000 |
| sae_top_directional_auc_n055 | QUO | 0.953 | 0.840 | 0.756 | 0.881 | 55.000 |
| sae_top_directional_auc_n055 | RE | 0.890 | 0.702 | 0.640 | 0.802 | 55.000 |
| sae_top_directional_auc_n055 | REC | 0.894 | 0.598 | 0.542 | 0.810 | 55.000 |
| sae_top_directional_auc_n055 | RES | 0.825 | 0.302 | 0.320 | 0.741 | 55.000 |
| sae_top_directional_auc_n055 | SU | 0.859 | 0.269 | 0.234 | 0.777 | 55.000 |
| sae_top_directional_auc_n056 | AF | 0.930 | 0.612 | 0.495 | 0.859 | 56.000 |
| sae_top_directional_auc_n056 | GI | 0.833 | 0.397 | 0.405 | 0.755 | 56.000 |
| sae_top_directional_auc_n056 | QU | 0.972 | 0.951 | 0.902 | 0.928 | 56.000 |
| sae_top_directional_auc_n056 | QUC | 0.921 | 0.635 | 0.603 | 0.847 | 56.000 |
| sae_top_directional_auc_n056 | QUO | 0.952 | 0.839 | 0.755 | 0.880 | 56.000 |
| sae_top_directional_auc_n056 | RE | 0.891 | 0.704 | 0.641 | 0.803 | 56.000 |
| sae_top_directional_auc_n056 | REC | 0.894 | 0.597 | 0.541 | 0.809 | 56.000 |
| sae_top_directional_auc_n056 | RES | 0.826 | 0.303 | 0.323 | 0.744 | 56.000 |
| sae_top_directional_auc_n056 | SU | 0.858 | 0.272 | 0.231 | 0.773 | 56.000 |
| sae_top_directional_auc_n057 | AF | 0.932 | 0.619 | 0.494 | 0.861 | 57.000 |
| sae_top_directional_auc_n057 | GI | 0.834 | 0.398 | 0.403 | 0.753 | 57.000 |
| sae_top_directional_auc_n057 | QU | 0.972 | 0.951 | 0.902 | 0.928 | 57.000 |
| sae_top_directional_auc_n057 | QUC | 0.923 | 0.641 | 0.602 | 0.846 | 57.000 |
| sae_top_directional_auc_n057 | QUO | 0.953 | 0.841 | 0.755 | 0.880 | 57.000 |
| sae_top_directional_auc_n057 | RE | 0.891 | 0.704 | 0.642 | 0.804 | 57.000 |
| sae_top_directional_auc_n057 | REC | 0.894 | 0.601 | 0.545 | 0.813 | 57.000 |
| sae_top_directional_auc_n057 | RES | 0.825 | 0.304 | 0.322 | 0.744 | 57.000 |
| sae_top_directional_auc_n057 | SU | 0.858 | 0.274 | 0.232 | 0.777 | 57.000 |
| sae_top_directional_auc_n058 | AF | 0.933 | 0.619 | 0.488 | 0.858 | 58.000 |
| sae_top_directional_auc_n058 | GI | 0.835 | 0.399 | 0.404 | 0.753 | 58.000 |
| sae_top_directional_auc_n058 | QU | 0.972 | 0.951 | 0.900 | 0.927 | 58.000 |
| sae_top_directional_auc_n058 | QUC | 0.922 | 0.645 | 0.600 | 0.844 | 58.000 |
| sae_top_directional_auc_n058 | QUO | 0.953 | 0.842 | 0.758 | 0.883 | 58.000 |
| sae_top_directional_auc_n058 | RE | 0.891 | 0.704 | 0.646 | 0.807 | 58.000 |
| sae_top_directional_auc_n058 | REC | 0.895 | 0.601 | 0.544 | 0.812 | 58.000 |
| sae_top_directional_auc_n058 | RES | 0.825 | 0.305 | 0.321 | 0.741 | 58.000 |
| sae_top_directional_auc_n058 | SU | 0.859 | 0.280 | 0.232 | 0.774 | 58.000 |
| sae_top_directional_auc_n059 | AF | 0.933 | 0.619 | 0.492 | 0.860 | 59.000 |
| sae_top_directional_auc_n059 | GI | 0.836 | 0.404 | 0.410 | 0.756 | 59.000 |
| sae_top_directional_auc_n059 | QU | 0.973 | 0.951 | 0.901 | 0.928 | 59.000 |
| sae_top_directional_auc_n059 | QUC | 0.922 | 0.645 | 0.606 | 0.849 | 59.000 |
| sae_top_directional_auc_n059 | QUO | 0.954 | 0.842 | 0.761 | 0.885 | 59.000 |
| sae_top_directional_auc_n059 | RE | 0.891 | 0.706 | 0.647 | 0.807 | 59.000 |
| sae_top_directional_auc_n059 | REC | 0.896 | 0.601 | 0.553 | 0.818 | 59.000 |
| sae_top_directional_auc_n059 | RES | 0.826 | 0.305 | 0.321 | 0.743 | 59.000 |
| sae_top_directional_auc_n059 | SU | 0.859 | 0.279 | 0.230 | 0.775 | 59.000 |
| sae_top_directional_auc_n060 | AF | 0.933 | 0.618 | 0.492 | 0.859 | 60.000 |
| sae_top_directional_auc_n060 | GI | 0.837 | 0.403 | 0.406 | 0.753 | 60.000 |
| sae_top_directional_auc_n060 | QU | 0.973 | 0.951 | 0.903 | 0.928 | 60.000 |
| sae_top_directional_auc_n060 | QUC | 0.923 | 0.646 | 0.602 | 0.847 | 60.000 |
| sae_top_directional_auc_n060 | QUO | 0.953 | 0.844 | 0.761 | 0.885 | 60.000 |
| sae_top_directional_auc_n060 | RE | 0.892 | 0.706 | 0.645 | 0.805 | 60.000 |
| sae_top_directional_auc_n060 | REC | 0.897 | 0.602 | 0.549 | 0.815 | 60.000 |
| sae_top_directional_auc_n060 | RES | 0.826 | 0.304 | 0.320 | 0.742 | 60.000 |
| sae_top_directional_auc_n060 | SU | 0.861 | 0.283 | 0.235 | 0.779 | 60.000 |
| sae_top_directional_auc_n061 | AF | 0.933 | 0.616 | 0.490 | 0.858 | 61.000 |
| sae_top_directional_auc_n061 | GI | 0.837 | 0.405 | 0.407 | 0.753 | 61.000 |
| sae_top_directional_auc_n061 | QU | 0.973 | 0.951 | 0.904 | 0.929 | 61.000 |
| sae_top_directional_auc_n061 | QUC | 0.923 | 0.645 | 0.603 | 0.844 | 61.000 |
| sae_top_directional_auc_n061 | QUO | 0.953 | 0.844 | 0.763 | 0.886 | 61.000 |
| sae_top_directional_auc_n061 | RE | 0.892 | 0.707 | 0.643 | 0.804 | 61.000 |
| sae_top_directional_auc_n061 | REC | 0.898 | 0.603 | 0.555 | 0.817 | 61.000 |
| sae_top_directional_auc_n061 | RES | 0.828 | 0.313 | 0.322 | 0.742 | 61.000 |
| sae_top_directional_auc_n061 | SU | 0.862 | 0.286 | 0.235 | 0.777 | 61.000 |
| sae_top_directional_auc_n062 | AF | 0.934 | 0.614 | 0.489 | 0.858 | 62.000 |
| sae_top_directional_auc_n062 | GI | 0.838 | 0.403 | 0.407 | 0.754 | 62.000 |
| sae_top_directional_auc_n062 | QU | 0.973 | 0.951 | 0.903 | 0.929 | 62.000 |
| sae_top_directional_auc_n062 | QUC | 0.924 | 0.654 | 0.607 | 0.847 | 62.000 |
| sae_top_directional_auc_n062 | QUO | 0.953 | 0.843 | 0.764 | 0.886 | 62.000 |
| sae_top_directional_auc_n062 | RE | 0.892 | 0.706 | 0.643 | 0.804 | 62.000 |
| sae_top_directional_auc_n062 | REC | 0.898 | 0.602 | 0.559 | 0.820 | 62.000 |
| sae_top_directional_auc_n062 | RES | 0.828 | 0.314 | 0.322 | 0.740 | 62.000 |
| sae_top_directional_auc_n062 | SU | 0.864 | 0.284 | 0.240 | 0.782 | 62.000 |
| sae_top_directional_auc_n063 | AF | 0.933 | 0.605 | 0.491 | 0.860 | 63.000 |
| sae_top_directional_auc_n063 | GI | 0.839 | 0.402 | 0.410 | 0.756 | 63.000 |
| sae_top_directional_auc_n063 | QU | 0.973 | 0.951 | 0.903 | 0.928 | 63.000 |
| sae_top_directional_auc_n063 | QUC | 0.924 | 0.658 | 0.611 | 0.849 | 63.000 |
| sae_top_directional_auc_n063 | QUO | 0.953 | 0.843 | 0.762 | 0.885 | 63.000 |
| sae_top_directional_auc_n063 | RE | 0.892 | 0.706 | 0.645 | 0.805 | 63.000 |
| sae_top_directional_auc_n063 | REC | 0.900 | 0.608 | 0.562 | 0.822 | 63.000 |
| sae_top_directional_auc_n063 | RES | 0.828 | 0.315 | 0.324 | 0.743 | 63.000 |
| sae_top_directional_auc_n063 | SU | 0.863 | 0.285 | 0.241 | 0.783 | 63.000 |
| sae_top_directional_auc_n064 | AF | 0.933 | 0.605 | 0.495 | 0.861 | 64.000 |
| sae_top_directional_auc_n064 | GI | 0.839 | 0.403 | 0.410 | 0.755 | 64.000 |
| sae_top_directional_auc_n064 | QU | 0.973 | 0.951 | 0.902 | 0.928 | 64.000 |
| sae_top_directional_auc_n064 | QUC | 0.924 | 0.660 | 0.610 | 0.847 | 64.000 |
| sae_top_directional_auc_n064 | QUO | 0.954 | 0.846 | 0.767 | 0.888 | 64.000 |
| sae_top_directional_auc_n064 | RE | 0.893 | 0.708 | 0.645 | 0.805 | 64.000 |
| sae_top_directional_auc_n064 | REC | 0.900 | 0.610 | 0.561 | 0.821 | 64.000 |
| sae_top_directional_auc_n064 | RES | 0.828 | 0.317 | 0.323 | 0.743 | 64.000 |
| sae_top_directional_auc_n064 | SU | 0.863 | 0.283 | 0.240 | 0.783 | 64.000 |
| sae_top_directional_auc_n065 | AF | 0.931 | 0.605 | 0.496 | 0.860 | 65.000 |
| sae_top_directional_auc_n065 | GI | 0.839 | 0.403 | 0.408 | 0.754 | 65.000 |
| sae_top_directional_auc_n065 | QU | 0.973 | 0.951 | 0.901 | 0.927 | 65.000 |
| sae_top_directional_auc_n065 | QUC | 0.924 | 0.660 | 0.613 | 0.849 | 65.000 |
| sae_top_directional_auc_n065 | QUO | 0.954 | 0.847 | 0.767 | 0.887 | 65.000 |
| sae_top_directional_auc_n065 | RE | 0.894 | 0.712 | 0.649 | 0.808 | 65.000 |
| sae_top_directional_auc_n065 | REC | 0.902 | 0.618 | 0.563 | 0.821 | 65.000 |
| sae_top_directional_auc_n065 | RES | 0.829 | 0.321 | 0.327 | 0.748 | 65.000 |
| sae_top_directional_auc_n065 | SU | 0.862 | 0.283 | 0.237 | 0.781 | 65.000 |
| sae_top_directional_auc_n066 | AF | 0.932 | 0.602 | 0.499 | 0.860 | 66.000 |
| sae_top_directional_auc_n066 | GI | 0.838 | 0.402 | 0.407 | 0.752 | 66.000 |
| sae_top_directional_auc_n066 | QU | 0.973 | 0.951 | 0.901 | 0.927 | 66.000 |
| sae_top_directional_auc_n066 | QUC | 0.925 | 0.663 | 0.616 | 0.850 | 66.000 |
| sae_top_directional_auc_n066 | QUO | 0.954 | 0.847 | 0.768 | 0.888 | 66.000 |
| sae_top_directional_auc_n066 | RE | 0.894 | 0.711 | 0.653 | 0.811 | 66.000 |
| sae_top_directional_auc_n066 | REC | 0.903 | 0.622 | 0.566 | 0.823 | 66.000 |
| sae_top_directional_auc_n066 | RES | 0.830 | 0.327 | 0.326 | 0.746 | 66.000 |
| sae_top_directional_auc_n066 | SU | 0.863 | 0.285 | 0.237 | 0.781 | 66.000 |
| sae_top_directional_auc_n067 | AF | 0.931 | 0.602 | 0.500 | 0.860 | 67.000 |
| sae_top_directional_auc_n067 | GI | 0.838 | 0.403 | 0.407 | 0.753 | 67.000 |
| sae_top_directional_auc_n067 | QU | 0.973 | 0.951 | 0.901 | 0.927 | 67.000 |
| sae_top_directional_auc_n067 | QUC | 0.925 | 0.666 | 0.618 | 0.852 | 67.000 |
| sae_top_directional_auc_n067 | QUO | 0.954 | 0.848 | 0.768 | 0.887 | 67.000 |
| sae_top_directional_auc_n067 | RE | 0.895 | 0.709 | 0.652 | 0.810 | 67.000 |
| sae_top_directional_auc_n067 | REC | 0.905 | 0.626 | 0.567 | 0.824 | 67.000 |
| sae_top_directional_auc_n067 | RES | 0.831 | 0.326 | 0.334 | 0.755 | 67.000 |
| sae_top_directional_auc_n067 | SU | 0.862 | 0.284 | 0.239 | 0.785 | 67.000 |
| sae_top_directional_auc_n068 | AF | 0.930 | 0.601 | 0.497 | 0.858 | 68.000 |
| sae_top_directional_auc_n068 | GI | 0.839 | 0.405 | 0.405 | 0.752 | 68.000 |
| sae_top_directional_auc_n068 | QU | 0.973 | 0.952 | 0.900 | 0.927 | 68.000 |
| sae_top_directional_auc_n068 | QUC | 0.926 | 0.667 | 0.617 | 0.849 | 68.000 |
| sae_top_directional_auc_n068 | QUO | 0.954 | 0.849 | 0.768 | 0.887 | 68.000 |
| sae_top_directional_auc_n068 | RE | 0.894 | 0.708 | 0.653 | 0.811 | 68.000 |
| sae_top_directional_auc_n068 | REC | 0.906 | 0.628 | 0.570 | 0.826 | 68.000 |
| sae_top_directional_auc_n068 | RES | 0.831 | 0.324 | 0.333 | 0.752 | 68.000 |
| sae_top_directional_auc_n068 | SU | 0.862 | 0.283 | 0.241 | 0.788 | 68.000 |
| sae_top_directional_auc_n069 | AF | 0.931 | 0.603 | 0.497 | 0.855 | 69.000 |
| sae_top_directional_auc_n069 | GI | 0.839 | 0.405 | 0.405 | 0.751 | 69.000 |
| sae_top_directional_auc_n069 | QU | 0.973 | 0.952 | 0.900 | 0.927 | 69.000 |
| sae_top_directional_auc_n069 | QUC | 0.926 | 0.669 | 0.619 | 0.851 | 69.000 |
| sae_top_directional_auc_n069 | QUO | 0.954 | 0.849 | 0.771 | 0.890 | 69.000 |
| sae_top_directional_auc_n069 | RE | 0.895 | 0.710 | 0.660 | 0.816 | 69.000 |
| sae_top_directional_auc_n069 | REC | 0.907 | 0.630 | 0.572 | 0.827 | 69.000 |
| sae_top_directional_auc_n069 | RES | 0.832 | 0.329 | 0.336 | 0.756 | 69.000 |
| sae_top_directional_auc_n069 | SU | 0.864 | 0.277 | 0.238 | 0.782 | 69.000 |
| sae_top_directional_auc_n070 | AF | 0.931 | 0.600 | 0.498 | 0.858 | 70.000 |
| sae_top_directional_auc_n070 | GI | 0.839 | 0.404 | 0.406 | 0.753 | 70.000 |
| sae_top_directional_auc_n070 | QU | 0.973 | 0.952 | 0.901 | 0.928 | 70.000 |
| sae_top_directional_auc_n070 | QUC | 0.926 | 0.669 | 0.617 | 0.850 | 70.000 |
| sae_top_directional_auc_n070 | QUO | 0.954 | 0.849 | 0.771 | 0.890 | 70.000 |
| sae_top_directional_auc_n070 | RE | 0.895 | 0.708 | 0.657 | 0.813 | 70.000 |
| sae_top_directional_auc_n070 | REC | 0.906 | 0.632 | 0.576 | 0.829 | 70.000 |
| sae_top_directional_auc_n070 | RES | 0.832 | 0.330 | 0.336 | 0.755 | 70.000 |
| sae_top_directional_auc_n070 | SU | 0.867 | 0.285 | 0.243 | 0.787 | 70.000 |
| sae_top_directional_auc_n071 | AF | 0.930 | 0.597 | 0.495 | 0.856 | 71.000 |
| sae_top_directional_auc_n071 | GI | 0.839 | 0.404 | 0.411 | 0.756 | 71.000 |
| sae_top_directional_auc_n071 | QU | 0.973 | 0.953 | 0.902 | 0.928 | 71.000 |
| sae_top_directional_auc_n071 | QUC | 0.926 | 0.669 | 0.618 | 0.852 | 71.000 |
| sae_top_directional_auc_n071 | QUO | 0.955 | 0.851 | 0.768 | 0.888 | 71.000 |
| sae_top_directional_auc_n071 | RE | 0.895 | 0.707 | 0.658 | 0.814 | 71.000 |
| sae_top_directional_auc_n071 | REC | 0.906 | 0.634 | 0.574 | 0.827 | 71.000 |
| sae_top_directional_auc_n071 | RES | 0.831 | 0.327 | 0.337 | 0.757 | 71.000 |
| sae_top_directional_auc_n071 | SU | 0.869 | 0.296 | 0.245 | 0.788 | 71.000 |
| sae_top_directional_auc_n072 | AF | 0.930 | 0.597 | 0.492 | 0.856 | 72.000 |
| sae_top_directional_auc_n072 | GI | 0.841 | 0.406 | 0.414 | 0.759 | 72.000 |
| sae_top_directional_auc_n072 | QU | 0.973 | 0.952 | 0.901 | 0.927 | 72.000 |
| sae_top_directional_auc_n072 | QUC | 0.926 | 0.668 | 0.617 | 0.851 | 72.000 |
| sae_top_directional_auc_n072 | QUO | 0.955 | 0.852 | 0.769 | 0.889 | 72.000 |
| sae_top_directional_auc_n072 | RE | 0.894 | 0.706 | 0.654 | 0.812 | 72.000 |
| sae_top_directional_auc_n072 | REC | 0.908 | 0.636 | 0.576 | 0.828 | 72.000 |
| sae_top_directional_auc_n072 | RES | 0.831 | 0.329 | 0.334 | 0.752 | 72.000 |
| sae_top_directional_auc_n072 | SU | 0.870 | 0.294 | 0.249 | 0.795 | 72.000 |
| sae_top_directional_auc_n073 | AF | 0.930 | 0.597 | 0.490 | 0.856 | 73.000 |
| sae_top_directional_auc_n073 | GI | 0.840 | 0.406 | 0.415 | 0.759 | 73.000 |
| sae_top_directional_auc_n073 | QU | 0.973 | 0.952 | 0.901 | 0.927 | 73.000 |
| sae_top_directional_auc_n073 | QUC | 0.927 | 0.671 | 0.622 | 0.855 | 73.000 |
| sae_top_directional_auc_n073 | QUO | 0.955 | 0.853 | 0.770 | 0.890 | 73.000 |
| sae_top_directional_auc_n073 | RE | 0.895 | 0.708 | 0.658 | 0.815 | 73.000 |
| sae_top_directional_auc_n073 | REC | 0.908 | 0.637 | 0.578 | 0.830 | 73.000 |
| sae_top_directional_auc_n073 | RES | 0.831 | 0.330 | 0.333 | 0.752 | 73.000 |
| sae_top_directional_auc_n073 | SU | 0.870 | 0.296 | 0.249 | 0.795 | 73.000 |
| sae_top_directional_auc_n074 | AF | 0.931 | 0.599 | 0.492 | 0.855 | 74.000 |
| sae_top_directional_auc_n074 | GI | 0.840 | 0.406 | 0.413 | 0.757 | 74.000 |
| sae_top_directional_auc_n074 | QU | 0.973 | 0.952 | 0.903 | 0.928 | 74.000 |
| sae_top_directional_auc_n074 | QUC | 0.927 | 0.671 | 0.622 | 0.855 | 74.000 |
| sae_top_directional_auc_n074 | QUO | 0.955 | 0.853 | 0.771 | 0.891 | 74.000 |
| sae_top_directional_auc_n074 | RE | 0.896 | 0.709 | 0.659 | 0.816 | 74.000 |
| sae_top_directional_auc_n074 | REC | 0.909 | 0.638 | 0.576 | 0.828 | 74.000 |
| sae_top_directional_auc_n074 | RES | 0.833 | 0.330 | 0.337 | 0.756 | 74.000 |
| sae_top_directional_auc_n074 | SU | 0.871 | 0.297 | 0.249 | 0.795 | 74.000 |
| sae_top_directional_auc_n075 | AF | 0.931 | 0.599 | 0.492 | 0.856 | 75.000 |
| sae_top_directional_auc_n075 | GI | 0.840 | 0.406 | 0.413 | 0.756 | 75.000 |
| sae_top_directional_auc_n075 | QU | 0.972 | 0.952 | 0.903 | 0.929 | 75.000 |
| sae_top_directional_auc_n075 | QUC | 0.927 | 0.673 | 0.622 | 0.855 | 75.000 |
| sae_top_directional_auc_n075 | QUO | 0.955 | 0.853 | 0.773 | 0.892 | 75.000 |
| sae_top_directional_auc_n075 | RE | 0.896 | 0.711 | 0.660 | 0.815 | 75.000 |
| sae_top_directional_auc_n075 | REC | 0.909 | 0.638 | 0.575 | 0.828 | 75.000 |
| sae_top_directional_auc_n075 | RES | 0.834 | 0.334 | 0.337 | 0.756 | 75.000 |
| sae_top_directional_auc_n075 | SU | 0.871 | 0.308 | 0.249 | 0.793 | 75.000 |
| sae_top_directional_auc_n076 | AF | 0.930 | 0.599 | 0.484 | 0.851 | 76.000 |
| sae_top_directional_auc_n076 | GI | 0.841 | 0.412 | 0.413 | 0.756 | 76.000 |
| sae_top_directional_auc_n076 | QU | 0.972 | 0.951 | 0.904 | 0.930 | 76.000 |
| sae_top_directional_auc_n076 | QUC | 0.926 | 0.672 | 0.623 | 0.855 | 76.000 |
| sae_top_directional_auc_n076 | QUO | 0.955 | 0.855 | 0.773 | 0.891 | 76.000 |
| sae_top_directional_auc_n076 | RE | 0.896 | 0.710 | 0.660 | 0.815 | 76.000 |
| sae_top_directional_auc_n076 | REC | 0.909 | 0.639 | 0.577 | 0.831 | 76.000 |
| sae_top_directional_auc_n076 | RES | 0.834 | 0.334 | 0.337 | 0.755 | 76.000 |
| sae_top_directional_auc_n076 | SU | 0.873 | 0.316 | 0.251 | 0.794 | 76.000 |
| sae_top_directional_auc_n077 | AF | 0.929 | 0.596 | 0.482 | 0.852 | 77.000 |
| sae_top_directional_auc_n077 | GI | 0.842 | 0.412 | 0.415 | 0.759 | 77.000 |
| sae_top_directional_auc_n077 | QU | 0.972 | 0.952 | 0.902 | 0.928 | 77.000 |
| sae_top_directional_auc_n077 | QUC | 0.926 | 0.672 | 0.625 | 0.857 | 77.000 |
| sae_top_directional_auc_n077 | QUO | 0.955 | 0.854 | 0.774 | 0.892 | 77.000 |
| sae_top_directional_auc_n077 | RE | 0.896 | 0.712 | 0.660 | 0.816 | 77.000 |
| sae_top_directional_auc_n077 | REC | 0.909 | 0.640 | 0.574 | 0.828 | 77.000 |
| sae_top_directional_auc_n077 | RES | 0.835 | 0.333 | 0.336 | 0.754 | 77.000 |
| sae_top_directional_auc_n077 | SU | 0.873 | 0.317 | 0.256 | 0.796 | 77.000 |
| sae_top_directional_auc_n078 | AF | 0.929 | 0.598 | 0.487 | 0.854 | 78.000 |
| sae_top_directional_auc_n078 | GI | 0.842 | 0.415 | 0.411 | 0.755 | 78.000 |
| sae_top_directional_auc_n078 | QU | 0.972 | 0.951 | 0.903 | 0.929 | 78.000 |
| sae_top_directional_auc_n078 | QUC | 0.926 | 0.673 | 0.625 | 0.856 | 78.000 |
| sae_top_directional_auc_n078 | QUO | 0.956 | 0.855 | 0.772 | 0.890 | 78.000 |
| sae_top_directional_auc_n078 | RE | 0.897 | 0.712 | 0.663 | 0.817 | 78.000 |
| sae_top_directional_auc_n078 | REC | 0.910 | 0.640 | 0.575 | 0.828 | 78.000 |
| sae_top_directional_auc_n078 | RES | 0.835 | 0.332 | 0.337 | 0.755 | 78.000 |
| sae_top_directional_auc_n078 | SU | 0.875 | 0.321 | 0.255 | 0.796 | 78.000 |
| sae_top_directional_auc_n079 | AF | 0.929 | 0.597 | 0.483 | 0.855 | 79.000 |
| sae_top_directional_auc_n079 | GI | 0.845 | 0.426 | 0.421 | 0.763 | 79.000 |
| sae_top_directional_auc_n079 | QU | 0.972 | 0.951 | 0.903 | 0.929 | 79.000 |
| sae_top_directional_auc_n079 | QUC | 0.925 | 0.672 | 0.624 | 0.855 | 79.000 |
| sae_top_directional_auc_n079 | QUO | 0.956 | 0.856 | 0.770 | 0.889 | 79.000 |
| sae_top_directional_auc_n079 | RE | 0.897 | 0.712 | 0.661 | 0.815 | 79.000 |
| sae_top_directional_auc_n079 | REC | 0.909 | 0.641 | 0.574 | 0.827 | 79.000 |
| sae_top_directional_auc_n079 | RES | 0.835 | 0.331 | 0.338 | 0.756 | 79.000 |
| sae_top_directional_auc_n079 | SU | 0.877 | 0.329 | 0.267 | 0.809 | 79.000 |
| sae_top_directional_auc_n080 | AF | 0.930 | 0.598 | 0.483 | 0.856 | 80.000 |
| sae_top_directional_auc_n080 | GI | 0.845 | 0.427 | 0.416 | 0.759 | 80.000 |
| sae_top_directional_auc_n080 | QU | 0.972 | 0.951 | 0.904 | 0.930 | 80.000 |
| sae_top_directional_auc_n080 | QUC | 0.925 | 0.672 | 0.625 | 0.855 | 80.000 |
| sae_top_directional_auc_n080 | QUO | 0.956 | 0.856 | 0.768 | 0.888 | 80.000 |
| sae_top_directional_auc_n080 | RE | 0.898 | 0.715 | 0.661 | 0.815 | 80.000 |
| sae_top_directional_auc_n080 | REC | 0.910 | 0.643 | 0.576 | 0.829 | 80.000 |
| sae_top_directional_auc_n080 | RES | 0.836 | 0.329 | 0.340 | 0.758 | 80.000 |
| sae_top_directional_auc_n080 | SU | 0.877 | 0.329 | 0.268 | 0.810 | 80.000 |
| sae_top_directional_auc_n081 | AF | 0.930 | 0.603 | 0.484 | 0.856 | 81.000 |
| sae_top_directional_auc_n081 | GI | 0.845 | 0.425 | 0.418 | 0.760 | 81.000 |
| sae_top_directional_auc_n081 | QU | 0.972 | 0.952 | 0.904 | 0.930 | 81.000 |
| sae_top_directional_auc_n081 | QUC | 0.926 | 0.676 | 0.625 | 0.857 | 81.000 |
| sae_top_directional_auc_n081 | QUO | 0.956 | 0.857 | 0.770 | 0.889 | 81.000 |
| sae_top_directional_auc_n081 | RE | 0.898 | 0.715 | 0.661 | 0.815 | 81.000 |
| sae_top_directional_auc_n081 | REC | 0.909 | 0.640 | 0.572 | 0.826 | 81.000 |
| sae_top_directional_auc_n081 | RES | 0.836 | 0.332 | 0.336 | 0.754 | 81.000 |
| sae_top_directional_auc_n081 | SU | 0.874 | 0.326 | 0.268 | 0.807 | 81.000 |
| sae_top_directional_auc_n082 | AF | 0.931 | 0.605 | 0.481 | 0.853 | 82.000 |
| sae_top_directional_auc_n082 | GI | 0.845 | 0.426 | 0.418 | 0.760 | 82.000 |
| sae_top_directional_auc_n082 | QU | 0.972 | 0.952 | 0.905 | 0.930 | 82.000 |
| sae_top_directional_auc_n082 | QUC | 0.926 | 0.677 | 0.625 | 0.856 | 82.000 |
| sae_top_directional_auc_n082 | QUO | 0.956 | 0.857 | 0.769 | 0.889 | 82.000 |
| sae_top_directional_auc_n082 | RE | 0.898 | 0.715 | 0.657 | 0.811 | 82.000 |
| sae_top_directional_auc_n082 | REC | 0.909 | 0.642 | 0.577 | 0.829 | 82.000 |
| sae_top_directional_auc_n082 | RES | 0.836 | 0.333 | 0.338 | 0.755 | 82.000 |
| sae_top_directional_auc_n082 | SU | 0.876 | 0.324 | 0.265 | 0.803 | 82.000 |
| sae_top_directional_auc_n083 | AF | 0.931 | 0.607 | 0.481 | 0.854 | 83.000 |
| sae_top_directional_auc_n083 | GI | 0.845 | 0.426 | 0.416 | 0.758 | 83.000 |
| sae_top_directional_auc_n083 | QU | 0.972 | 0.951 | 0.905 | 0.931 | 83.000 |
| sae_top_directional_auc_n083 | QUC | 0.927 | 0.682 | 0.624 | 0.855 | 83.000 |
| sae_top_directional_auc_n083 | QUO | 0.956 | 0.857 | 0.768 | 0.888 | 83.000 |
| sae_top_directional_auc_n083 | RE | 0.898 | 0.715 | 0.658 | 0.813 | 83.000 |
| sae_top_directional_auc_n083 | REC | 0.910 | 0.642 | 0.581 | 0.832 | 83.000 |
| sae_top_directional_auc_n083 | RES | 0.835 | 0.333 | 0.338 | 0.755 | 83.000 |
| sae_top_directional_auc_n083 | SU | 0.876 | 0.326 | 0.268 | 0.804 | 83.000 |
| sae_top_directional_auc_n084 | AF | 0.931 | 0.606 | 0.489 | 0.858 | 84.000 |
| sae_top_directional_auc_n084 | GI | 0.845 | 0.426 | 0.414 | 0.754 | 84.000 |
| sae_top_directional_auc_n084 | QU | 0.975 | 0.954 | 0.911 | 0.936 | 84.000 |
| sae_top_directional_auc_n084 | QUC | 0.927 | 0.684 | 0.622 | 0.854 | 84.000 |
| sae_top_directional_auc_n084 | QUO | 0.956 | 0.859 | 0.768 | 0.888 | 84.000 |
| sae_top_directional_auc_n084 | RE | 0.898 | 0.713 | 0.658 | 0.813 | 84.000 |
| sae_top_directional_auc_n084 | REC | 0.910 | 0.643 | 0.579 | 0.831 | 84.000 |
| sae_top_directional_auc_n084 | RES | 0.836 | 0.336 | 0.334 | 0.752 | 84.000 |
| sae_top_directional_auc_n084 | SU | 0.877 | 0.329 | 0.267 | 0.802 | 84.000 |
| sae_top_directional_auc_n085 | AF | 0.932 | 0.608 | 0.490 | 0.860 | 85.000 |
| sae_top_directional_auc_n085 | GI | 0.844 | 0.428 | 0.420 | 0.759 | 85.000 |
| sae_top_directional_auc_n085 | QU | 0.975 | 0.954 | 0.910 | 0.936 | 85.000 |
| sae_top_directional_auc_n085 | QUC | 0.928 | 0.683 | 0.620 | 0.854 | 85.000 |
| sae_top_directional_auc_n085 | QUO | 0.956 | 0.860 | 0.770 | 0.889 | 85.000 |
| sae_top_directional_auc_n085 | RE | 0.898 | 0.714 | 0.659 | 0.814 | 85.000 |
| sae_top_directional_auc_n085 | REC | 0.910 | 0.643 | 0.581 | 0.832 | 85.000 |
| sae_top_directional_auc_n085 | RES | 0.835 | 0.336 | 0.335 | 0.752 | 85.000 |
| sae_top_directional_auc_n085 | SU | 0.874 | 0.326 | 0.266 | 0.802 | 85.000 |
| sae_top_directional_auc_n086 | AF | 0.933 | 0.608 | 0.487 | 0.853 | 86.000 |
| sae_top_directional_auc_n086 | GI | 0.845 | 0.427 | 0.417 | 0.756 | 86.000 |
| sae_top_directional_auc_n086 | QU | 0.975 | 0.954 | 0.910 | 0.936 | 86.000 |
| sae_top_directional_auc_n086 | QUC | 0.928 | 0.684 | 0.623 | 0.856 | 86.000 |
| sae_top_directional_auc_n086 | QUO | 0.956 | 0.859 | 0.770 | 0.889 | 86.000 |
| sae_top_directional_auc_n086 | RE | 0.898 | 0.715 | 0.661 | 0.816 | 86.000 |
| sae_top_directional_auc_n086 | REC | 0.910 | 0.645 | 0.582 | 0.834 | 86.000 |
| sae_top_directional_auc_n086 | RES | 0.835 | 0.336 | 0.336 | 0.754 | 86.000 |
| sae_top_directional_auc_n086 | SU | 0.873 | 0.322 | 0.262 | 0.797 | 86.000 |
| sae_top_directional_auc_n087 | AF | 0.931 | 0.603 | 0.483 | 0.850 | 87.000 |
| sae_top_directional_auc_n087 | GI | 0.844 | 0.426 | 0.419 | 0.757 | 87.000 |
| sae_top_directional_auc_n087 | QU | 0.976 | 0.955 | 0.911 | 0.937 | 87.000 |
| sae_top_directional_auc_n087 | QUC | 0.928 | 0.683 | 0.623 | 0.855 | 87.000 |
| sae_top_directional_auc_n087 | QUO | 0.956 | 0.860 | 0.772 | 0.890 | 87.000 |
| sae_top_directional_auc_n087 | RE | 0.898 | 0.716 | 0.660 | 0.814 | 87.000 |
| sae_top_directional_auc_n087 | REC | 0.910 | 0.645 | 0.583 | 0.835 | 87.000 |
| sae_top_directional_auc_n087 | RES | 0.835 | 0.335 | 0.336 | 0.753 | 87.000 |
| sae_top_directional_auc_n087 | SU | 0.876 | 0.326 | 0.267 | 0.801 | 87.000 |
| sae_top_directional_auc_n088 | AF | 0.931 | 0.608 | 0.481 | 0.848 | 88.000 |
| sae_top_directional_auc_n088 | GI | 0.843 | 0.423 | 0.418 | 0.756 | 88.000 |
| sae_top_directional_auc_n088 | QU | 0.976 | 0.955 | 0.911 | 0.936 | 88.000 |
| sae_top_directional_auc_n088 | QUC | 0.928 | 0.683 | 0.622 | 0.854 | 88.000 |
| sae_top_directional_auc_n088 | QUO | 0.956 | 0.860 | 0.771 | 0.889 | 88.000 |
| sae_top_directional_auc_n088 | RE | 0.899 | 0.716 | 0.661 | 0.815 | 88.000 |
| sae_top_directional_auc_n088 | REC | 0.911 | 0.646 | 0.581 | 0.834 | 88.000 |
| sae_top_directional_auc_n088 | RES | 0.835 | 0.334 | 0.335 | 0.753 | 88.000 |
| sae_top_directional_auc_n088 | SU | 0.876 | 0.321 | 0.266 | 0.799 | 88.000 |
| sae_top_directional_auc_n089 | AF | 0.931 | 0.609 | 0.480 | 0.847 | 89.000 |
| sae_top_directional_auc_n089 | GI | 0.844 | 0.425 | 0.419 | 0.757 | 89.000 |
| sae_top_directional_auc_n089 | QU | 0.976 | 0.955 | 0.911 | 0.936 | 89.000 |
| sae_top_directional_auc_n089 | QUC | 0.928 | 0.682 | 0.624 | 0.855 | 89.000 |
| sae_top_directional_auc_n089 | QUO | 0.956 | 0.860 | 0.769 | 0.887 | 89.000 |
| sae_top_directional_auc_n089 | RE | 0.899 | 0.717 | 0.662 | 0.816 | 89.000 |
| sae_top_directional_auc_n089 | REC | 0.910 | 0.644 | 0.580 | 0.833 | 89.000 |
| sae_top_directional_auc_n089 | RES | 0.837 | 0.333 | 0.337 | 0.756 | 89.000 |
| sae_top_directional_auc_n089 | SU | 0.878 | 0.318 | 0.270 | 0.802 | 89.000 |
| sae_top_directional_auc_n090 | AF | 0.932 | 0.618 | 0.491 | 0.855 | 90.000 |
| sae_top_directional_auc_n090 | GI | 0.844 | 0.426 | 0.419 | 0.756 | 90.000 |
| sae_top_directional_auc_n090 | QU | 0.976 | 0.956 | 0.911 | 0.937 | 90.000 |
| sae_top_directional_auc_n090 | QUC | 0.928 | 0.682 | 0.623 | 0.856 | 90.000 |
| sae_top_directional_auc_n090 | QUO | 0.956 | 0.859 | 0.770 | 0.888 | 90.000 |
| sae_top_directional_auc_n090 | RE | 0.899 | 0.717 | 0.664 | 0.817 | 90.000 |
| sae_top_directional_auc_n090 | REC | 0.910 | 0.643 | 0.580 | 0.832 | 90.000 |
| sae_top_directional_auc_n090 | RES | 0.837 | 0.330 | 0.338 | 0.756 | 90.000 |
| sae_top_directional_auc_n090 | SU | 0.877 | 0.322 | 0.268 | 0.800 | 90.000 |
| sae_top_directional_auc_n091 | AF | 0.932 | 0.619 | 0.492 | 0.855 | 91.000 |
| sae_top_directional_auc_n091 | GI | 0.844 | 0.425 | 0.420 | 0.757 | 91.000 |
| sae_top_directional_auc_n091 | QU | 0.976 | 0.956 | 0.911 | 0.936 | 91.000 |
| sae_top_directional_auc_n091 | QUC | 0.928 | 0.681 | 0.624 | 0.856 | 91.000 |
| sae_top_directional_auc_n091 | QUO | 0.956 | 0.859 | 0.771 | 0.889 | 91.000 |
| sae_top_directional_auc_n091 | RE | 0.899 | 0.717 | 0.665 | 0.818 | 91.000 |
| sae_top_directional_auc_n091 | REC | 0.910 | 0.645 | 0.582 | 0.832 | 91.000 |
| sae_top_directional_auc_n091 | RES | 0.836 | 0.329 | 0.339 | 0.758 | 91.000 |
| sae_top_directional_auc_n091 | SU | 0.880 | 0.326 | 0.266 | 0.795 | 91.000 |
| sae_top_directional_auc_n092 | AF | 0.933 | 0.620 | 0.495 | 0.857 | 92.000 |
| sae_top_directional_auc_n092 | GI | 0.844 | 0.427 | 0.421 | 0.759 | 92.000 |
| sae_top_directional_auc_n092 | QU | 0.976 | 0.956 | 0.910 | 0.936 | 92.000 |
| sae_top_directional_auc_n092 | QUC | 0.928 | 0.682 | 0.625 | 0.857 | 92.000 |
| sae_top_directional_auc_n092 | QUO | 0.956 | 0.859 | 0.771 | 0.889 | 92.000 |
| sae_top_directional_auc_n092 | RE | 0.900 | 0.717 | 0.663 | 0.817 | 92.000 |
| sae_top_directional_auc_n092 | REC | 0.911 | 0.645 | 0.584 | 0.835 | 92.000 |
| sae_top_directional_auc_n092 | RES | 0.836 | 0.329 | 0.337 | 0.755 | 92.000 |
| sae_top_directional_auc_n092 | SU | 0.879 | 0.328 | 0.264 | 0.793 | 92.000 |
| sae_top_directional_auc_n093 | AF | 0.934 | 0.624 | 0.495 | 0.858 | 93.000 |
| sae_top_directional_auc_n093 | GI | 0.845 | 0.428 | 0.421 | 0.760 | 93.000 |
| sae_top_directional_auc_n093 | QU | 0.976 | 0.956 | 0.911 | 0.936 | 93.000 |
| sae_top_directional_auc_n093 | QUC | 0.927 | 0.682 | 0.624 | 0.855 | 93.000 |
| sae_top_directional_auc_n093 | QUO | 0.956 | 0.860 | 0.770 | 0.888 | 93.000 |
| sae_top_directional_auc_n093 | RE | 0.900 | 0.718 | 0.663 | 0.817 | 93.000 |
| sae_top_directional_auc_n093 | REC | 0.910 | 0.643 | 0.583 | 0.834 | 93.000 |
| sae_top_directional_auc_n093 | RES | 0.836 | 0.330 | 0.338 | 0.756 | 93.000 |
| sae_top_directional_auc_n093 | SU | 0.879 | 0.329 | 0.267 | 0.795 | 93.000 |
| sae_top_directional_auc_n094 | AF | 0.934 | 0.623 | 0.497 | 0.858 | 94.000 |
| sae_top_directional_auc_n094 | GI | 0.845 | 0.426 | 0.425 | 0.762 | 94.000 |
| sae_top_directional_auc_n094 | QU | 0.976 | 0.956 | 0.910 | 0.936 | 94.000 |
| sae_top_directional_auc_n094 | QUC | 0.928 | 0.683 | 0.628 | 0.858 | 94.000 |
| sae_top_directional_auc_n094 | QUO | 0.956 | 0.861 | 0.772 | 0.889 | 94.000 |
| sae_top_directional_auc_n094 | RE | 0.901 | 0.724 | 0.663 | 0.816 | 94.000 |
| sae_top_directional_auc_n094 | REC | 0.910 | 0.645 | 0.583 | 0.833 | 94.000 |
| sae_top_directional_auc_n094 | RES | 0.836 | 0.329 | 0.341 | 0.759 | 94.000 |
| sae_top_directional_auc_n094 | SU | 0.879 | 0.328 | 0.263 | 0.791 | 94.000 |
| sae_top_directional_auc_n095 | AF | 0.933 | 0.624 | 0.496 | 0.858 | 95.000 |
| sae_top_directional_auc_n095 | GI | 0.845 | 0.427 | 0.424 | 0.761 | 95.000 |
| sae_top_directional_auc_n095 | QU | 0.976 | 0.956 | 0.910 | 0.935 | 95.000 |
| sae_top_directional_auc_n095 | QUC | 0.929 | 0.687 | 0.634 | 0.859 | 95.000 |
| sae_top_directional_auc_n095 | QUO | 0.955 | 0.860 | 0.774 | 0.890 | 95.000 |
| sae_top_directional_auc_n095 | RE | 0.902 | 0.727 | 0.665 | 0.817 | 95.000 |
| sae_top_directional_auc_n095 | REC | 0.910 | 0.645 | 0.581 | 0.832 | 95.000 |
| sae_top_directional_auc_n095 | RES | 0.836 | 0.329 | 0.341 | 0.759 | 95.000 |
| sae_top_directional_auc_n095 | SU | 0.878 | 0.325 | 0.268 | 0.798 | 95.000 |
| sae_top_directional_auc_n096 | AF | 0.934 | 0.626 | 0.495 | 0.858 | 96.000 |
| sae_top_directional_auc_n096 | GI | 0.845 | 0.427 | 0.427 | 0.764 | 96.000 |
| sae_top_directional_auc_n096 | QU | 0.976 | 0.956 | 0.910 | 0.936 | 96.000 |
| sae_top_directional_auc_n096 | QUC | 0.929 | 0.687 | 0.633 | 0.859 | 96.000 |
| sae_top_directional_auc_n096 | QUO | 0.955 | 0.860 | 0.772 | 0.889 | 96.000 |
| sae_top_directional_auc_n096 | RE | 0.902 | 0.727 | 0.665 | 0.817 | 96.000 |
| sae_top_directional_auc_n096 | REC | 0.910 | 0.644 | 0.581 | 0.831 | 96.000 |
| sae_top_directional_auc_n096 | RES | 0.837 | 0.331 | 0.344 | 0.762 | 96.000 |
| sae_top_directional_auc_n096 | SU | 0.878 | 0.323 | 0.268 | 0.796 | 96.000 |
| sae_top_directional_auc_n097 | AF | 0.936 | 0.622 | 0.507 | 0.863 | 97.000 |
| sae_top_directional_auc_n097 | GI | 0.845 | 0.424 | 0.422 | 0.760 | 97.000 |
| sae_top_directional_auc_n097 | QU | 0.977 | 0.956 | 0.909 | 0.935 | 97.000 |
| sae_top_directional_auc_n097 | QUC | 0.930 | 0.693 | 0.636 | 0.862 | 97.000 |
| sae_top_directional_auc_n097 | QUO | 0.955 | 0.860 | 0.772 | 0.889 | 97.000 |
| sae_top_directional_auc_n097 | RE | 0.902 | 0.728 | 0.666 | 0.818 | 97.000 |
| sae_top_directional_auc_n097 | REC | 0.910 | 0.645 | 0.582 | 0.832 | 97.000 |
| sae_top_directional_auc_n097 | RES | 0.838 | 0.330 | 0.345 | 0.763 | 97.000 |
| sae_top_directional_auc_n097 | SU | 0.879 | 0.339 | 0.276 | 0.806 | 97.000 |
| sae_top_directional_auc_n098 | AF | 0.936 | 0.625 | 0.509 | 0.864 | 98.000 |
| sae_top_directional_auc_n098 | GI | 0.845 | 0.425 | 0.424 | 0.762 | 98.000 |
| sae_top_directional_auc_n098 | QU | 0.977 | 0.956 | 0.910 | 0.936 | 98.000 |
| sae_top_directional_auc_n098 | QUC | 0.930 | 0.694 | 0.631 | 0.858 | 98.000 |
| sae_top_directional_auc_n098 | QUO | 0.954 | 0.857 | 0.770 | 0.888 | 98.000 |
| sae_top_directional_auc_n098 | RE | 0.903 | 0.728 | 0.666 | 0.818 | 98.000 |
| sae_top_directional_auc_n098 | REC | 0.910 | 0.646 | 0.582 | 0.833 | 98.000 |
| sae_top_directional_auc_n098 | RES | 0.839 | 0.331 | 0.346 | 0.764 | 98.000 |
| sae_top_directional_auc_n098 | SU | 0.879 | 0.336 | 0.266 | 0.793 | 98.000 |
| sae_top_directional_auc_n099 | AF | 0.936 | 0.627 | 0.507 | 0.864 | 99.000 |
| sae_top_directional_auc_n099 | GI | 0.845 | 0.431 | 0.422 | 0.759 | 99.000 |
| sae_top_directional_auc_n099 | QU | 0.976 | 0.955 | 0.911 | 0.937 | 99.000 |
| sae_top_directional_auc_n099 | QUC | 0.929 | 0.695 | 0.631 | 0.858 | 99.000 |
| sae_top_directional_auc_n099 | QUO | 0.954 | 0.858 | 0.768 | 0.888 | 99.000 |
| sae_top_directional_auc_n099 | RE | 0.903 | 0.728 | 0.667 | 0.819 | 99.000 |
| sae_top_directional_auc_n099 | REC | 0.910 | 0.645 | 0.584 | 0.833 | 99.000 |
| sae_top_directional_auc_n099 | RES | 0.839 | 0.332 | 0.347 | 0.763 | 99.000 |
| sae_top_directional_auc_n099 | SU | 0.876 | 0.329 | 0.267 | 0.789 | 99.000 |
| sae_top_directional_auc_n100 | AF | 0.935 | 0.623 | 0.506 | 0.867 | 100.000 |
| sae_top_directional_auc_n100 | GI | 0.846 | 0.430 | 0.424 | 0.761 | 100.000 |
| sae_top_directional_auc_n100 | QU | 0.976 | 0.956 | 0.911 | 0.937 | 100.000 |
| sae_top_directional_auc_n100 | QUC | 0.930 | 0.694 | 0.631 | 0.858 | 100.000 |
| sae_top_directional_auc_n100 | QUO | 0.954 | 0.858 | 0.770 | 0.889 | 100.000 |
| sae_top_directional_auc_n100 | RE | 0.903 | 0.730 | 0.672 | 0.822 | 100.000 |
| sae_top_directional_auc_n100 | REC | 0.910 | 0.645 | 0.584 | 0.834 | 100.000 |
| sae_top_directional_auc_n100 | RES | 0.840 | 0.334 | 0.346 | 0.761 | 100.000 |
| sae_top_directional_auc_n100 | SU | 0.875 | 0.325 | 0.265 | 0.785 | 100.000 |

## Reading Guide

- `full_sae_latents` tests the complete SAE feature vector.
- `raw_hidden` tests the original layer activation vector.
- `pca_raw_hidden` tests a dense PCA basis fit on the training fold only.
- High AUC means label information is linearly decodable; it does not prove causal mechanism alignment.
