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
    smoothBox = {'box', 'box', 'box', 'box', 'box', 'box', 'box', 'box', 'box', ...
      'box', 'box', 'box', 'box', 'box', 'box', 'box'};
    smoothEpanechnikov = { 'epanechnikov', 'epanechnikov', 'epanechnikov', ...
      'epanechnikov', 'epanechnikov', 'epanechnikov', 'epanechnikov', ...
      'epanechnikov', 'epanechnikov', 'epanechnikov', 'epanechnikov', ...
      'epanechnikov', 'epanechnikov', 'epanechnikov', 'epanechnikov', 'epanechnikov'};
    smootNormal = { 'normal', 'normal', 'normal', 'normal', 'normal', ...
      'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', ...
      'normal', 'normal', 'normal', 'normal'};
    smoothTriangle = {'triangle', 'triangle', 'triangle', 'triangle', 'triangle',...
      'triangle', 'triangle', 'triangle', 'triangle', 'triangle', 'triangle', ...
      'triangle', 'triangle', 'triangle', 'triangle', 'triangle'};
    smoothUnsused = {'N/A', 'N/A', 'N/A', 'N/A', 'N/A',...
      'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', ...
      'N/A', 'N/A', 'N/A', 'N/A', 'N/A'};
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
      cellsIndex = 1;
      for distributionNameCells = hyperparameters.distributionNames 
        for numberOfHoldOutRun = hyperparameters.numberOfHoldOutRuns       
          for priorDistribution = priorDistributions
              % repeat holdout partition creation, build tree, predict this
              startTime = cputime; 
              trainLosses = zeros(1, numberOfHoldOutRun);
              testLosses = zeros(1, numberOfHoldOutRun);
              misclassificationCounts = zeros(1, numberOfHoldOutRun);  
              meanAccuracy = zeros(1, numberOfHoldOutRun);
              meanPrecision = zeros(1, numberOfHoldOutRun);
              meanRecall = zeros(1, numberOfHoldOutRun);
              meanF1 = zeros(1, numberOfHoldOutRun);
              for holdOutTestRunCount = 1:numberOfHoldOutRun
                % get width row matching distributionNames row
                width = hyperparameters.kernelWidths(cellsIndex);
                % get smoother type matching distribution names row
                smootherType = hyperparameters.kernelSmoother(cellsIndex);
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
                  fprintf("\n\tSmoother Type: ");
                  for str = string(smootherType{1,1})
                    fprintf("%s ", str);
                  end
                  fprintf("\n\tDistName: ");
                  for str = string(distributionNameCells{1,1})
                    fprintf("'%s' ", str);
                  end
                  fprintf("\n\tprior: "); 
                  for aPrior = priorDistribution
                    fprintf("%0.04f ", aPrior);
                  end
                  fprintf('\n');
                end
                %obj.debug = false;
                [trainingLoss, testLoss, misclassifiedCount, ...
                    distributionName, smootherTypeName, model, ...
                    accuracy, precision, recall, ...
                    f1] =...
                  obj.buildAndTestNBayes(xTrain, yTrain, xTest, yTest, ...
                    distributionNameCells, smootherType, priorDistribution, width, ...
                    classNames);
                meanAccuracy(holdOutTestRunCount) = accuracy;
                meanPrecision(holdOutTestRunCount) = precision;
                meanRecall(holdOutTestRunCount) = recall;
                meanF1(holdOutTestRunCount) = f1;
                trainLosses(holdOutTestRunCount) = trainingLoss;
                testLosses(holdOutTestRunCount) = testLoss;
                misclassificationCounts(holdOutTestRunCount) = misclassifiedCount;
              end % holdOutTestRunCount
              endTime = cputime;
              avgTrainLoss = mean(trainLosses);
              avgTestLoss = mean(testLosses);
              avgAccuracy = mean(meanAccuracy);
              avgPrecision = mean(meanPrecision);
              avgRecall = mean(meanRecall);
              avgF1 = mean(meanF1);
              resultsTable.appendResult(distributionName, smootherTypeName, width, numberOfHoldOutRun, ...
                                 avgTrainLoss, avgTestLoss, ...
                                 avgAccuracy, avgPrecision, avgRecall, avgF1, ...
                                 size(yTest, 1), ...
                                 endTime - startTime);
             
          end % priorDistribution
        end % numberOfHoldOutRun  
        cellsIndex = cellsIndex + 1;
      end % distributionName   
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
        distributionName, smootherTypeName,model, ...
        accuracy, precision, recall, ...
        f1] = buildAndTestNBayes(obj, xTrain, ...
                                    yTrain, xTest, yTest, ...
                                    distributionNames, smootherTypes, priorDistribution, width, classNames)
      % train naive bayes model
      model = [];
      if width >= 0.0
        model = fitcnb(xTrain, yTrain, ...
          'ClassNames', classNames, 'DistributionNames', string(distributionNames{:,:}), ...
          'Prior', priorDistribution, 'kernel', string(smootherTypes{:,:}), 'width', width);
        distributionName = 'kernel';
        cellConverter = smootherTypes{1,1};
        smootherTypeName = cellConverter{1,1};
      else
        model = fitcnb(xTrain, yTrain, ...
          'ClassNames', classNames, 'DistributionNames', string(distributionNames{:,:}), ...
          'Prior', priorDistribution);
        distributionName = 'normal';
        smootherTypeName = 'unused';
      end
      
      % predict using given xTest
      [predictionResult, ~] = predict(model, xTest);
      % errors
      misclassifiedCount = sum(predictionResult ~= categorical(table2array(yTest)));
      trainingLoss = loss(model, xTrain, yTrain);
      testLoss = loss(model, xTest, yTest);
      [accuracy, precision, recall, ...
        f1] = obj.calculateMeasuresFromExpectPredict(yTest, predictionResult);
      if obj.debug
        % display random 5 predictions, check loss and misclassified
        % percent match
        %predictionResult(randsample(numel(predictionResult), 5))
        percentMisclassified = (misclassifiedCount / size(yTest, 1)) * 100;
        fprintf("Misclassified %d entries out of %d. %0.04f pct\n", misclassifiedCount, size(yTest, 1), percentMisclassified);
        fprintf("\tTraining loss: %0.02f. Test Loss: %0.02f\n", trainingLoss, testLoss);
        fprintf("\tPrecision: %0.04f Recall: %0.04f Accuracy: %0.04f F1: %0.04f\n", ...
                 precision, recall, accuracy, f1);      
      end % debug
    end % function
    
    % Calculate accuracy, precision, recall and F1 from expected (table)
    % and predicted (array)
    % return measurements.
    function [accuracy, precision, recall, f1] = calculateMeasuresFromExpectPredict(obj, expected, predicted)
      % get confusion matrix from expected vs predictions
      [cm, order] = confusionmat(categorical(table2array(expected)), categorical(predicted));
      % pre-allocate vectors of calculations and results (one per class)
      TP = zeros(1, 26);
      classPrecision = zeros(1, 26);
      classRecall = zeros(1, 26);
      % calculate each class' precision and recall
      for classIndex = 1:size(order)
        % TP: number of actual positives predicted as positive
        % True positives = value in the diagonal
        TP(classIndex) = cm(classIndex, classIndex);         
        % precision: proportion of predicted positives that are actual positive 
        % TP/(TP+FP): TP/sum of class' row
        classPrecision(classIndex) = TP(classIndex) / sum(cm(classIndex, :));
        % recall: proportion of actual positives are predicted as positive:
        % TP/(TP+FN): TP/sum of class' column
        classRecall(classIndex) = TP(classIndex) / sum(cm(:, classIndex));
      end
      % get overall mean for precision and recall
      precision = mean(classPrecision);
      recall = mean(classRecall);
      % accuracy: sum of correctly predicted (diagonal) / sum of all
      % entries
      accuracy = sum(sum(eye(26) .* cm)) / sum(cm(:));
      f1 = 2 * (precision * recall) / (precision + recall);
    end
    
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
      nBayesClassInstance.smoothBox = nBayesClassInstance.smoothBox(:,1:attributeCount - 1);
      nBayesClassInstance.smoothEpanechnikov = nBayesClassInstance.smoothEpanechnikov(:,1:attributeCount - 1);
      nBayesClassInstance.smootNormal = nBayesClassInstance.smootNormal(:,1:attributeCount - 1);
      nBayesClassInstance.smoothTriangle = nBayesClassInstance.smoothTriangle(:,1:attributeCount - 1);
    end
  end % static methods
end

