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
      maxNumSplits = [100, 75, 50, 25];
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
    function hyperparameters = getGdiSplitCriteriaInstance()
      hyperparameters = DTreeHyperparametersClass();
      hyperparameters.randomSeed = 112;
      hyperparameters.numberOfHoldOutRuns = [1 5 10 15 20];
      hyperparameters.trainValidateProportions = [0.9 0.8, 0.7, 0.5];
      hyperparameters.maxNumSplits = [75, 50, 25, 10, 5];
      hyperparameters.splitCriteria = ["gdi"];      
    end
    
    function hyperparameters = getFinalHyperparameterInstance()
      hyperparameters = DTreeHyperparametersClass();
      hyperparameters.randomSeed = 112;
      hyperparameters.numberOfHoldOutRuns = [1];
      hyperparameters.trainValidateProportions = [0.7];
      hyperparameters.maxNumSplits = [5];
      hyperparameters.splitCriteria = ["gdi"];        
    end
    
    
  end % Static
end

