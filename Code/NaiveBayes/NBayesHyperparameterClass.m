classdef NBayesHyperparameterClass < handle
  %NBAYESHYPERPARAMETERCLASS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
      randomSeed = 110;
      numberOfHoldOutRuns = [1 5 10 15 20];
      distributionNames = [];
      kernelSmoother = [];
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
    
    function hyperparameters = getKernalBoxSmootherHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfHoldOutRuns = [1 2 5];
      % rows of distribution names (eg: kernel, kernel, ....) one per
      % width, smoother type
      hyperparameters.distributionNames = {nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel};
      hyperparameters.kernelSmoother = {nBayes.smoothBox, nBayes.smoothBox, ...
        nBayes.smoothBox, nBayes.smoothBox, nBayes.smoothBox};                                    ;
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [0.10 0.098351 0.0932 0.090 0.80];
    end
    
    function hyperparameters = getKernalEpanechnikovSmootherHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfHoldOutRuns = [1 2 5];
      % rows of distribution names (eg: kernel, kernel, ....) one per
      % width, smoother type
      hyperparameters.distributionNames = {nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel};
      hyperparameters.kernelSmoother = {nBayes.smoothEpanechnikov, nBayes.smoothEpanechnikov, ...
        nBayes.smoothEpanechnikov, nBayes.smoothEpanechnikov, nBayes.smoothEpanechnikov};                                    ;
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [0.10 0.098351 0.0932 0.090 0.80];
    end
    
    function hyperparameters = getKernalNormalSmootherHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfHoldOutRuns = [1 2 5];
      % rows of distribution names (eg: kernel, kernel, ....) one per
      % width, smoother type
      hyperparameters.distributionNames = { nBayes.distNamesKernel ...                                        nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel};
      hyperparameters.kernelSmoother = {nBayes.smoothNormal, nBayes.smoothNormal, ...
        nBayes.smoothNormal, nBayes.smoothNormal, nBayes.smoothNormal};                                    ;
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [0.10 0.098351 0.0932 0.090 0.80];
    end
    
        function hyperparameters = getKernelTriangleSmootherHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfHoldOutRuns = [1 2 5];
      % rows of distribution names (eg: kernel, kernel, ....) one per
      % width, smoother type
      hyperparameters.distributionNames = {nBayes.distNamesKernel, ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel ...
                                            nBayes.distNamesKernel};
      hyperparameters.kernelSmoother = {nBayes.smoothTriangle, nBayes.smoothTriangle, ...
        nBayes.smoothTriangle, nBayes.smoothTriangle, nBayes.smoothTriangle};                                    ;
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [0.10 0.098351 0.0932 0.090 0.80];
    end
    
    function hyperparameters = getDefaultTestHyperparametersInstance(nBayes)
      hyperparameters = NBayesHyperparameterClass();
      hyperparameters.numberOfHoldOutRuns = [1 2 5 10];
      % rows of distribution names (eg: kernel, kernel, ....)
      hyperparameters.distributionNames = {nBayes.distNamesDefault };
      % rows of widths to go with distribution names rows. So if a
      % distribution row is normal, it should be -1 (no width), a number
      % otherwise
      hyperparameters.kernelWidths = [-1.0];
      hyperparameters.kernelSmoother = {nBayes.smoothUnsused};
    end
    
  end % methods(static)
end % class

