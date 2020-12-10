classdef DTreeHyperparametersClass
  %DTreeHyperparameters: Class to represent a set of hyperparameter values
  % Hyperparameter values for the fictree() method are grouped in this
  % class.
  % Convenience configurations are available using getInstance
  % static methods.
  %
  
  properties % default values are set
    % default random stream initial value
    randomSeed = 300;
    % random stream used by default
    defaultRandomStream;
    % random stream used when training naive bayes
    dTreeRandomStream;      
    numberOfFolds = [2 3 5];
    minLeafSizes = [1 3 5];
    minParentSizes = [6 8 10];
    maxNumSplits = [4000, 8000, 10000];
    splitCriteria = ["gdi", "twoing", "deviance"];
    classNames = {};
  end % properties
  
  methods
    
    %
    % Constructor creates the default DTreeHyperparameters configuration
    % classNames: features & names to use in tree
    %
    function obj = DTreeHyperparametersClass()

    end
    
  end % methods
  
  methods(Static)
    %
    % Default set of hyperparameter values used for the full run
    %
    function hyperparameters = getInstance()
      hyperparameters = DTreeHyperparametersClass();
    end
    
    %
    % Small set of hyperparameter values are useful to test that the code
    % runs without errors.
    %
    function hyperparameters = getQuickTestRunInstance()
      hyperparameters = DTreeHyperparametersClass();
      hyperparameters.numberOfFolds = [2, 3];
      hyperparameters.minLeafSizes = [1 3];
      hyperparameters.minParentSizes = [12 14];      
      hyperparameters.maxNumSplits = [100];
      hyperparameters.splitCriteria = ["gdi"];      
    end
    
    %
    % Set of hyperparameter values used for the detailed run on the
    % gdi split criterion.
    %
    function hyperparameters = getDevianceSplitCriteriaInstance()
      hyperparameters = DTreeHyperparametersClass();
      hyperparameters.randomSeed = 112;
      hyperparameters.numberOfFolds = [2 3 5];
      hyperparameters.splitCriteria = ["deviance"];      
    end
    
    %
    % These are the hyperparameters used to create the final model.
    % The numberOfFolds is set to -1, because the is no crossvalidation
    % when creating the final model.
    function hyperparameters = getFinalHyperparameterInstance()
      hyperparameters = DTreeHyperparametersClass();
      hyperparameters.randomSeed = 192;
      hyperparameters.minLeafSizes = [1];
      hyperparameters.minParentSizes = [6];
      hyperparameters.maxNumSplits = [8000];
      hyperparameters.splitCriteria = ["deviance"]; 
      % unused: train and test sets are used on the whole dataset instead:
      hyperparameters.numberOfFolds = [-1];      
    end
    
    
  end % Static
end

