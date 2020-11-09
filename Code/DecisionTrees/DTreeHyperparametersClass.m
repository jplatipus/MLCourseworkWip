classdef DTreeHyperparametersClass
  %DTreeHyperparameters Class to represent a set of hyperparameter values
  % Hyperparameter values for the fictree() method are grouped in this
  % class.
  % Convenience configurations are available using getInstance
  % static methods.
  %
  
  properties
      randomSeed = 110;
      numberOfHoldOutRuns = [1 5 10 15 20];
      trainValidateProportions = [0.8, 0.7];
      maxNumSplits = [25, 100, 400, 800, 1600];
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
      hyperparameters.numberOfHoldOutRuns = [2, 5];
      hyperparameters.trainValidateProportions = [0.2];
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
      hyperparameters.numberOfHoldOutRuns = [1 10 20];
      hyperparameters.trainValidateProportions = [0.8];
      hyperparameters.maxNumSplits = [50, 100, 200, 300, 350, 400, 450, 500, 550, 600, 800, 1000];
      hyperparameters.splitCriteria = ["deviance"];      
    end
    
    function hyperparameters = getFinalHyperparameterInstance()
      hyperparameters = DTreeHyperparametersClass();
      hyperparameters.randomSeed = 250;
      hyperparameters.maxNumSplits = [400 425 450 500 525 550 600 700 800 1000];
      hyperparameters.splitCriteria = ["deviance"]; 
      % unused: train and test sets are used on the whole dataset instead:
      hyperparameters.numberOfHoldOutRuns = [-1];
      % unused: train and test sets are used on the whole dataset instead:
      hyperparameters.trainValidateProportions = [-1];       
    end
    
    
  end % Static
end

