classdef NBayesHyperparameterClass < handle
  %NBAYESHYPERPARAMETERCLASS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
      randomSeed = 110;
      numberOfHoldOutRuns = [1 5 10 15 20];
      distributionNames = [];
      kernelWidths = [];
  end % properties
  
  methods
    function obj = NBayesHyperparameterClass()

    end
  end % methods
    
  methods(Static)
    function hyperparameters = getQuickTestHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfHoldOutRuns = [1];
      % rows of distribution names (eg: kernel, kernel, ....)
      hyperparameters.distributionNames = {[nBayes.distNamesDefault]};
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [-1.0];
    end
    
    function hyperparameters = getDefaultAndKernalTestHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfHoldOutRuns = [1 2 5];
      % rows of distribution names (eg: kernel, kernel, ....)
      hyperparameters.distributionNames = {nBayes.distNamesDefault ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...                                        nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel};
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [-1.0, 2.0 1.5 0.10 0.098351 0.0932 0.090 0.80 0.70 0.50];
    end
    
  end % methods(static)
end % class

