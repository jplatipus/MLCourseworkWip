classdef LetterDecisionTreeResults < handle
  %LETTERDECISIONTREERESULTS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    resultsTable = {};
    resultsColumnNames = ["numberOfHoldOutRun" "trainValidateProportion" ...
      "maxNumSplit" "splitCriterion" "avgTrainAccuracy" "avgTestAccuracy" "elapsedTime"];
        % temporary file to store the results:
    outputResultsTempFilename = "dtreeResultsTemp.csv";
    fileHandle = -1;
  end
  
  methods
    
    %
    % Constructor
    %
    function obj = LetterDecisionTreeResults()
      %create empty results table
      obj.resultsTable = table(obj.resultsColumnNames);
    end
    
    %
    % Open temporary file to write results
    % throws exception if error
    function startGatheringResults(obj)
           [h, msg] = fopen(obj.outputResultsTempFilename, 'w');
           if h == -1
               fprintf("Error opening output file %s\n", msg);
               exception = MException("LetterDecisionTreeResults:startGatheringResults", "Error opening output file: %s\n", msg);
               throw(exception);
           end
           % output header
           obj.fileHandle = h;
           fprintf(obj.fileHandle, "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", obj.resultsColumnNames(:));
    end
    
    function appendResult(obj,trainValidateProportion, maxNumSplit, splitCriterion, ...
                  numberOfHoldOutRun, avgTrainAccuracy, avgTestAccuracy, elapsedTime)
      if obj.fileHandle == -1
        exception = MException("LetterDecisionTreeResults:appendResult", "Error output file %s is not open", obj.outputResultsTempFilename);
        throw(exception);
      end
      fprintf(obj.fileHandle, "%d\t%0.02f\t%d\t%s\t%0.04f\t%0.04f\t%0.04f\n", ...
               numberOfHoldOutRun, trainValidateProportion, ...
               maxNumSplit, splitCriterion, avgTrainAccuracy, avgTestAccuracy, elapsedTime);
    end
    
    function endGatheringResults(obj)
      if obj.fileHandle == -1
        exception = MException("LetterDecisionTreeResults:endGatheringResults", "Error output file %s is not open", obj.outputResultsTempFilename);
        throw(exception);
      end
      fclose(obj.fileHandle);
      obj.resultsTable = readtable(obj.outputResultsTempFilename, "Delimiter", "\t");
    end
  end
end

