import pandas as pd

class ABAnalyser:

  def __init__(self,A_group,B_group):
    self.A_group = A_group
    self.B_group = B_group

  def getData(self):
    return self.A_group, self.B_group

  def setData(self,new_A, new_B):
    self.A_group, self.B_group = new_A, new_B
  
  def checkAdversarialLabel(self,
                    covariates,
                    sample_frac = 1.0,
                    thr_max = 0.51,
                    thr_min = 0.49,
                    verbose = False
                    ):
    """
    Check if the group variants were stratified correctly using adverserial strategy

    Parameters:
    * covariates: List of covariates
    * sample_frac: Sample of fraction % for faster validation (default 1)
    * thr_max: Max threshold for ROC (default 0.51)
    * thr_min: Min threshold for ROC (default 0.49)
    * verbose: if True then print all step status

    Returns: True / False
    """ 
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score

    if verbose == True:
      print("Adverserial validation started...")
    if len(covariates) < 1:
      return False
    
    A_deepcopy = self.A_group.copy(deep = True)
    B_deepcopy = self.B_group.copy(deep = True)

    # if sample fraction is specified, extract the subsets randomly
    if (sample_frac < 1.0) and (sample_frac > 0.0):
      if verbose == True:
        print("Sampling {} of both datasets".format(sample_frac))

      A_deepcopy = A_deepcopy.sample(frac = sample_frac)
      B_deepcopy = B_deepcopy.sample(frac = sample_frac)
      
    # Prepare a dataset by combining A & B and put labels of the groups - 2 classes
    if verbose == True:
      print("Combine A & B groups and create group label for each")
    X = A_deepcopy.append(B_deepcopy) 
    y = [0]*len(A_deepcopy) + [1]*len(B_deepcopy) # put a pseudolabel

    # convert all categorical variables to binary 
    X = pd.get_dummies(X,columns = covariates)

    if verbose == True:
      print("Run a classifier to distinguish between the 2 datasets")
    model = RandomForestClassifier() # use RandomForest here but could be any classifier
    
    # do cross-val and output prediction of pseudo-label
    cv_preds = cross_val_predict(model, 
                                X, 
                                y, 
                                cv=2, 
                                n_jobs = None,
                                method = "predict_proba",
                                verbose = verbose)
    
    roc_score = roc_auc_score(y_true = y, y_score = cv_preds[:,1])
    if verbose == True:
      print ("ROC Score = {}".format(roc_score))
      print ("Adverserial validation finished.")

    return thr_min <= roc_score <= thr_max

  def checkCovBalance(self,
               covariates,
               alpha = 0.05,
               verbose = False
              ):
    """
    Check if the group variants are balanced based on chi square tests

    Parameters:
    * covariates: List of covariates
    * alpha: Critical value (default 5%)
    * verbose: if True then print all step status

    Returns: List of unbalanced covariates (empty if not)
    """ 
  
    from scipy.stats import chi2_contingency

    if verbose == True:
      print("Covariate balancing check using Chisqr test...")

    if len(covariates) < 1:
      return False

    A_deepcopy = self.A_group.copy(deep = True)
    B_deepcopy = self.B_group.copy(deep = True)

    # label group
    A_deepcopy["group_"] = "A"
    B_deepcopy["group_"] = "B"

    # Prepare a dataset by combining A & B 
    if verbose == True:
      print("Combine A & B groups and create group label for each")
    AB_df = A_deepcopy.append(B_deepcopy) 

    output_list = []

    for cov in covariates:
        pct = pd.crosstab(AB_df[cov],AB_df["group_"],normalize = "columns")
        _, p, _, _ = chi2_contingency(pct) 

        if verbose == True:
          print("Covariate {}:".format(cov))
          print(pct)
          print("p_val = {}".format(p))
          print()
        
        if p <= alpha:
          output_list.append(cov)
    
    return output_list

  def detectTailedOutlier(self,
                          targets,
                          remove_outlier = False,
                          tail_quantile_level = 0.99,
                          verbose = False):    
    """
    Detect outliers based on tailed quantile with option of outlier removal

    Parameters:
    * target: Target variables
    * remove_outlier: Option to remove outlier (default False)
    * tail_quantile_level: The quantile beyond which the outlier is determined
    * verbose: if True then print all step status

    Returns: 
    * List of outlier level corresponding to each target (dict type)
    * A_group outliers removed if remove_outlier = True
    * B_group outliers removed if remove_outlier = True
    """ 

    if verbose == True:
      print("Detect outliers by quantiles...")
    
    if len(targets) < 1:
      return None, self.A_group, self.B_group

    A_deepcopy = self.A_group.copy(deep = True)
    B_deepcopy = self.B_group.copy(deep = True)

    # label group
    A_deepcopy["group_"] = "A"
    B_deepcopy["group_"] = "B"

    AB_df = A_deepcopy.append(B_deepcopy) 

    sum_stats = AB_df[targets].describe([0.01, 0.05, 0.20, 0.50, 0.80, 0.95, 0.99])

    if verbose == True:
      print("Pre-removal of outliers:")
      print(sum_stats.T)
      print()

    outliers = {}

    for tg in targets:
      outliers[tg] = AB_df[tg].quantile(tail_quantile_level)
      if remove_outlier:
        AB_df = AB_df[AB_df[tg] <= outliers[tg]]
        A_deepcopy = A_deepcopy[A_deepcopy[tg] <= outliers[tg]]
        B_deepcopy = B_deepcopy[B_deepcopy[tg] <= outliers[tg]]

    if verbose == True:
      sum_stats = AB_df[targets].describe([0.01, 0.05, 0.20, 0.50, 0.80, 0.95, 0.99])
      print("Post_removal of outliers:")
      print(sum_stats.T)

    
    return outliers, A_deepcopy, B_deepcopy

  def checkGroupNormality(self,
                    group,
                    targets,
                    alpha = 0.05,
                    plot_dist = False,
                    verbose = False):    
    """
    Check normality of the targett variables. If sample size is larger than 5000,
    use Shapiro-Wilk test (parametric). Otherwise, use Kolmogorov-Smirnoff test 
    (non-parametric)

    Parameters:
    * group_label: Group Label (either A or B)
    * targets: Taget variables (list if multiple)
    * alpha: Critical value (default 0.05)
    * plot_dist: Plot histogram & density function if True (default False)
    * verbose:  print all step statuses if True (default False)

    Returns: 
    * Targets that are not normally distributed
    """ 
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns

    non_normal_targets = []
    N_sample = len(group) 
    if plot_dist:
      _,axes = plt.subplots(len(targets) // 3,
                            (len(targets)-1) % 3 + 1,
                            figsize = (15,4))

    for idx,tg in enumerate(targets):
      
      # if plot_dist set to True, plot the histogram & density function
      if plot_dist:
        sns.histplot(group[tg] , 
                    color="skyblue", 
                    kde = True, 
                    stat = "density",
                    label=tg,
                    ax = axes[idx])

      # if sample size is less than 5000, use Shapiro-Wilk test
      if N_sample < 5000:
        _, pval  = stats.shapiro(group[tg])

        if verbose == True:
          print("[Shapiro-Wilk] Test normality: Target {} P-val = {}".format(tg,pval))

      # else use Kolmogorv-Smirnoff test against norm
      else:
        _, pval = stats.kstest(group[tg],stats.norm.cdf)

        if verbose == True:
          print("[KS] Test normality: Target {} P-val = {}".format(tg,pval))

      if pval < alpha:
        non_normal_targets.append(tg)

    return non_normal_targets

  def checkABHomogVariance(self,
                  targets,
                  alpha = 0.05,
                  plot_dist = False,
                  verbose = False):  
    """
    Check homogeneity of variance for the 2 variants using Levene test

    Parameters:
    * targets: Taget variables (list if multiple)
    * alpha: Critical value (default 0.05)
    * plot_dist: Plot histogram & density function if True (default False)
    * verbose:  print all step statuses if True (default False)

    Returns: 
    * Targets that are not normally distributed
    """ 
    from scipy import stats
    import matplotlib.pyplot as plt
    import seaborn as sns

    non_homog_targets = []

    if plot_dist:
      _,axes = plt.subplots(len(targets),
                            2,
                            figsize = (10,5))

    for idx,tg in enumerate(targets):
      
      # if plot_dist set to True, plot the histogram & density function
      if plot_dist:
        sns.histplot(self.A_group[tg],color="skyblue",kde = True,stat = "density",label=tg,ax = axes[idx,0])
        sns.histplot(self.B_group[tg],color="red",kde = True,stat = "density",label=tg,ax = axes[idx,1])


      # if sample size is less than 5000, use Shapiro-Wilk test
      _, pval = stats.levene(self.A_group[tg],self.B_group[tg])

      if pval < alpha:
        non_homog_targets.append(tg)

      if verbose:
        print("[Levene] Test homogenity of variance: Target {} P-val = {}".format(tg,pval))

    return non_homog_targets

  def computePVal(self,
                    targets,
                    alpha = 0.05,
                    verbose = False):    
    """
    Compute P-values for testing 2 groups
    - Apply Shapiro-Wilk or Kolmogorov-Smirnoff Test for normality
    - If normality applies, apply Levene Test for homogeneity of variances
        + If Parametric + homogeneity of variances apply T-Test
        + If Parametric - homogeneity of variances apply Welch Test
    - If not, apply non-parametric test (Mann Whitney U)

    Parameters:
    * A_group: Group A
    * B_group: Group B
    * targets: Taget variables (list if multiple)
    * alpha: Critical value (default 0.05)
    * verbose:  print all step statuses if True (default False)

    Returns: 
    * Pvalues for all targets
    """ 
    if verbose == True:
      print("Compute p-values for testing 2 groups...")

    import scipy.stats as stats

    pval = {}

    for tg in targets:
      # If both groups A & B are normally distributed, we can apply parametric tests
      if (len(self.checkGroupNormality(self.A_group, 
                            targets = [tg],
                            verbose = verbose)) == 0) \
        and (len(self.checkGroupNormality(self.B_group,
                              targets = [tg],
                              verbose = verbose)) == 0):
        
        if verbose:
          print("Both A & B groups are normally distributed...")

        _, pval[tg] = stats.ttest_ind(self.A_group,
                                      self.B_group,
                                      equal_var = len(self.checkABHomogVariance([tg])) == 0)

        if verbose:
          print("[T-test] Target {} P-val = {}".format(tg, pval[tg]))
          print()

      # Otherwise, we should use non-parametric tests    
      else:
        if verbose:
          print("Either A or B group is not normally distributed...")

        _, pval[tg] = stats.mannwhitneyu(self.A_group[tg],
                                        self.B_group[tg])
      
        if verbose:
          print("[Mann Whitney U] Target {} P-val = {}".format(tg, pval[tg]))
          print()

    return pval

