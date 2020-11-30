classdef CalcUtil
  %UNTITLED Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
  end
  
  methods
  end
  
  methods(Static)
    % Calculate accuracy, precision, recall and F1 from expected (array)
    % and predicted (array)
    % return measurements.
    function [accuracy, precision, recall, f1] = calculateMeasuresFromExpectPredict(expected, predicted)
      % get confusion matrix from expected vs predictions
      [cm, order] = confusionmat(categorical(expected), categorical(predicted));
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
  end
end

