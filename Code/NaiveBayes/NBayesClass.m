classdef NBayesClass < handle
  %NBAYESCLASS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    randomSeed = 300;
    debug = false;
    % letter dataset used in this instance:
    dataset;
    % Matlab Model:
    nBayesModel;
    % Training attributes:
    x;
    % Training results:
    y; 
    % test attributes:
    xt; 
    % test results:
    yt;
    % distribution names used in the default (normal) distribution
    distNamesDefault = {'normal','normal','normal','normal','normal','normal','normal','normal','normal',...
      'normal','normal','normal','normal','normal','normal','normal'};
    % distribution names used in the kernel distribution
    distNamesKernel = {'kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel','kernel',...
      'kernel','kernel','kernel','kernel','kernel','kernel','kernel'};
  end
  
  methods
    
    %% Constructor
    % Several getInstance static convenience methods use this constructor.
    %
    function obj = NBayesClass(letterDataset)
      obj.dataset = letterDataset;
      rng(obj.randomSeed);
      [obj.x, obj.y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
      [obj.xt, obj.yt] = obj.dataset.extractXYFromTable(obj.dataset.testTable);
    end % constructor
    
    % Fit the model
    function fitModel(obj, distributionNames)
        model = fitcnb(obj.x, obj.y, ...
        'ClassNames', categorical(table2array(obj.dataset.validClassValues)), ...
        'DistributionNames', distributionNames);
      obj.nBayesModel = model;
    end % method
    
    % calculate the model's training and test loss
    function [trainingLoss, testLoss] = getModelLoss(obj)
        trainingLoss = loss(obj.nBayesModel, obj.x, obj.y);
        testLoss = loss(obj.nBayesModel, obj.xt, obj.yt);
    end
    
    % Set the Matlab model's Prior distribution to reflect the dataset's
    % training distribution of classes
    function nBayesModel = setPriorDistributionEmpirical(obj)
      prior = obj.getPriorDistributionEmpirical();
      obj.nBayesModel.Prior = prior;
      nBayesModel = obj.nBayesModel;
    end % function
    
    % calculate the dataset's
    % training distribution of classes
    % returns a vector of the distributions for each class
    function prior = getPriorDistributionEmpirical(obj)
      Y = table2array(obj.y);
      freqDist = cell2table(tabulate(Y));
      prior = freqDist{:,3}/100;
    end
    
    % calculate a normal prior ditribution of classes for each class (1/26)
    function prior = getPriorDistributionNormal(obj)
      prior = zeros([26, 1]);
      prior = prior + 1/26;
    end
    
    % Perform matlab's hyperparameter optimization search
    function nBayesModel = fitMatlabHyperparameterOptimization(obj)
      obj.nBayesModel = fitcnb(obj.x, obj.y, ...
        'ClassNames', categorical(table2array(obj.dataset.validClassValues)), ...
        'OptimizeHyperparameters','auto',...
        'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
        'expected-improvement-plus'));
      nBayesModel = obj.nBayesModel;
    end % function

    %
    % Perform a 'manual' hold out cross validation hyperparameter search,
    % using the different hyperparameter values supplied in
    % hyperparameters.
    % writes the results to the csv file (tab delimitted), and returns the
    % results as a results class instance.
    %
    function letterNBayesResults = performHyperparameterSearch(obj, hyperparameters, resultsCsvFilename)
     fprintf("Starting decision tree analysis...\n");
      rng(hyperparameters.randomSeed);
      numExamples = size(obj.dataset.trainTable, 1);  
      % unique set of values that can appear in the predicted results:
      classNames = categorical(table2array(obj.dataset.validClassValues));
      resultsTable = NBayesResults(resultsCsvFilename);
      resultsTable.startGatheringResults();  
      % for gaussian kernel distribution, we can specify a prior
      % distribution of classes. This test uses normal and emprirical
      % (based on the training set distribution):
      priorDistributions = obj.getPriorDistributionEmpirical(); %, ...
                      % obj.getPriorDistributionNormal()];
      for numberOfHoldOutRun = hyperparameters.numberOfHoldOutRuns       
        for priorDistribution = priorDistributions
            cellsIndex = 1;
            for distributionNameCells = hyperparameters.distributionNames             
              % repeat holdout partition creation, build tree, predict this
              % number of times to get average:
              accuracyIndex = 1;
              startTime = cputime; 
              trainAccuracies = zeros(1, numberOfHoldOutRun);
              testAccuracies = zeros(1, numberOfHoldOutRun);
              misclassificationCounts = zeros(1, numberOfHoldOutRun);  
              meanAccuracy = zeros(1, numberOfHoldOutRun);
              meanPrecision = zeros(1, numberOfHoldOutRun);
              meanRecall = zeros(1, numberOfHoldOutRun);
              meanF1 = zeros(1, numberOfHoldOutRun);
              for holdOutTestRunCount = 1:numberOfHoldOutRun
                % get width row matching distributionNames row
                width = hyperparameters.kernelWidths(cellsIndex);
                % calculate train and test subset indeces
                partition = cvpartition(numExamples, 'Holdout', 0.2);
                trainSubsetIdx = training(partition);
                testSubsetIdx = test(partition);
                [x, y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
                % extract train and test subsets
                xTrain = x(trainSubsetIdx, :);
                yTrain = y(trainSubsetIdx, :);
                xTest = x(testSubsetIdx, :);
                yTest = y(testSubsetIdx, :);
                obj.debug = true;
                if obj.debug
                  fprintf("Test run HoldRunCount: %d of %d,", holdOutTestRunCount,numberOfHoldOutRun);
                  fprintf("Width: %0.04f ",width);
                  fprintf("DistName: ");
                  for str = string(distributionNameCells{1,1})
                    fprintf("'%s' ", str);
                  end
                  fprintf("\n\tprior: "); 
                  for aPrior = priorDistribution
                    fprintf("%0.04f ", aPrior);
                  end
                  fprintf('\n');
                end
                obj.debug = false;
                [trainingLoss, testLoss, misclassifiedCount, ...
                    distributionName, model, ...
                    classAccuracy, classPrecision, classRecall, ...
                    classF1] =...
                    obj.buildAndTestNBayes(xTrain, yTrain, xTest, yTest, ...
                                      distributionNameCells, priorDistribution, width, ...
                                      classNames);
                meanAccuracy(accuracyIndex) = mean(classAccuracy);
                meanPrecision(accuracyIndex) = mean(classPrecision);
                meanRecall(accuracyIndex) = mean(classRecall);
                meanF1(accuracyIndex) = mean(classF1);
                trainAccuracies(accuracyIndex) = 1 - trainingLoss;
                testAccuracies(accuracyIndex) = 1 - testLoss;
                misclassificationCounts(accuracyIndex) = misclassifiedCount;             
                accuracyIndex = accuracyIndex + 1;
              end % holdOutTestRunCount
              cellsIndex = cellsIndex + 1;
              endTime = cputime;
              avgTrainAccuracy = mean(trainAccuracies);
              avgTestAccuracy = mean(testAccuracies);
              avgMisclassificationCount = mean(misclassificationCounts);
              avgAccuracy = mean(numberOfHoldOutRun);
              avgPrecision = mean(numberOfHoldOutRun);
              avgRecall = mean(numberOfHoldOutRun);
              avgF1 = mean(numberOfHoldOutRun);
              resultsTable.appendResult(distributionName, width, numberOfHoldOutRun, ...
                                 avgTrainAccuracy, avgTestAccuracy, ...
                                 avgMisclassificationCount, ...
                                 avgAccuracy, avgPrecision, avgRecall, avgF1, ...
                                 size(yTest, 1), ...
                                 endTime - startTime);
            end % distributionName    
        end % priorDistribution
      end % numberOfHoldOutRun        
      resultsTable.endGatheringResults();
      letterNBayesResults = resultsTable;
      fprintf("Completed NBayes analysis\n");      
    end % function
    
    % Build the model using the training set, predict using the test set,
    % using the hyperparameters supplied.
    % claculate and return:
    % trainingLoss, testLoss, misclassifiedCount, ...
    %   distributionName, model, ...
    %    classAccuracy, classPrecision, classRecall, ...
    %    classF1, classLabel
    function [trainingLoss, testLoss, misclassifiedCount, ...
        distributionName, model, ...
        classAccuracy, classPrecision, classRecall, ...
        classF1] = buildAndTestNBayes(obj, xTrain, ...
                                    yTrain, xTest, yTest, ...
                                    distributionNames, priorDistribution, width, classNames)
      % train naive bayes model
      model = [];
      if width >= 0.0
        model = fitcnb(xTrain, yTrain, ...
          'ClassNames', classNames, 'DistributionNames', string(distributionNames{:,:}), ...
          'Prior', priorDistribution, 'width', width);
        distributionName = 'kernel';
      else
        model = fitcnb(xTrain, yTrain, ...
          'ClassNames', classNames, 'DistributionNames', string(distributionNames{:,:}), ...
          'Prior', priorDistribution);
        distributionName = 'normal';
      end
      
      % predict using given xTest
      [predictionResult, score] = predict(model, xTest);
      % errors
      misclassifiedCount = sum(predictionResult ~= categorical(table2array(yTest)));
      trainingLoss = loss(model, xTrain, yTrain);
      testLoss = loss(model, xTest, yTest);
      [cm, order] = confusionmat(categorical(table2array(yTest)), categorical(predictionResult));
      % pre-allocate vectors of calculations and results (one per class)
      TP = zeros(1, 26);
      FN = zeros(1, 26);
      FP = zeros(1, 26);
      TN = zeros(1, 26);
      classAccuracy = zeros(1, 26);
      classPrecision = zeros(1, 26);
      classRecall = zeros(1, 26);
      classF1 = zeros(1, 26);
      
      for classIndex = 1:size(order)
        %Calculate true/false positives/negatives:
        % True positives = value in the diagonal
        TP(classIndex) = cm(classIndex, classIndex);
        % False Negative = values in the class' column - TP
        FN(classIndex) = sum(cm(classIndex,:))-cm(classIndex, classIndex);
        % False Positive = values in the class' row - TP
        FP(classIndex) = sum(cm(:,classIndex))-cm(classIndex,classIndex);
        % True Negative = total number of vales - (TP + FP + FN)
        TN(classIndex) = sum(cm(:))-(TP(classIndex)+FP(classIndex)+FN(classIndex));  
        
        % Calculate prediction measures:
        classAccuracy(classIndex) = (TP(classIndex) + TN(classIndex))/sum(cm(:));
        classPrecision(classIndex) = TP(classIndex)/(TP(classIndex)+FN(classIndex));
        classRecall(classIndex) = TP(classIndex)/(TP(classIndex)/FN(classIndex));
        classF1(classIndex) = 2 * (classRecall(classIndex) * classPrecision(classIndex)) ...
                                / (classRecall(classIndex) + classPrecision(classIndex));      
      end
      
      %[acc, gmean, fscore, precision, recall, specificity] = PerfMetrics(cm);
      if obj.debug
        % display random 5 predictions
        predictionResult(randsample(numel(predictionResult), 5))
        percentMisclassified = (misclassifiedCount / size(yTest, 1)) * 100;
        fprintf("Misclassified %d entries out of %d. %0.04f pct\n", misclassifiedCount, size(yTest, 1), percentMisclassified);
        fprintf("Training loss: %0.02f. Test Loss: %0.02f\n", trainingLoss, testLoss);
      end
      
    end % function
    
    
  end %methods
  
  %
  % Static methods
  %
  methods(Static)

    % Get default, unfitted class instance
    function nBayesClassInstance = getDefaultInstanceNoFit(letterDataset)
      nBayesClassInstance = NBayesClass(letterDataset);
      NBayesClass.alignDistributionNames(nBayesClassInstance);
    end %function 
    
    % Get the default class instance for the given dataset
    function nBayesClassInstance = getDefaultInstance(letterDataset)
      nBayesClassInstance = NBayesClass(letterDataset);
      NBayesClass.alignDistributionNames(nBayesClassInstance);
      nBayesClassInstance.fitModel(nBayesClassInstance.distNamesDefault);
    end %function    
    
    % Get the kernel distribution class instance for the given dataset
    function nBayesClassInstance = getKernelInstance(letterDataset)
      nBayesClassInstance = NBayesClass(letterDataset);
      NBayesClass.alignDistributionNames(nBayesClassInstance);
      nBayesClassInstance.fitModel(nBayesClassInstance.distNamesKernel);
    end %function  
    
    % Ensures the nBayesClassInstance's distNamesDefault and 
    % distNamesKernel have the same number of entries as the letterDataset
    % has attributes
    function alignDistributionNames(nBayesClassInstance)
      letterDataset = nBayesClassInstance.dataset;
      attributeCount = size(letterDataset.trainTable);
      attributeCount = attributeCount(1,2);
      nBayesClassInstance.distNamesDefault = nBayesClassInstance.distNamesDefault(:,1:attributeCount - 1);
      nBayesClassInstance.distNamesKernel = nBayesClassInstance.distNamesKernel(:,1:attributeCount - 1);
    end
  end % static methods
end

