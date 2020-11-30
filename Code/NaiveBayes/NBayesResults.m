classdef NBayesResults < handle
  %NBAYESRESULTS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    resultsTable = {};
    resultsColumnNames = [
      "distributionName", "smootherType", "Width", "numberOfHoldOutRun", "avgTrainLoss", ...
      "avgTestLoss", ...
      "avgAccuracy", "avgPrecision", "avgRecall", "avgF1", ...
      "entryCount", "elapsedTime"];
        % temporary file to store the results:
    outputResultsFilename = "nbayesResultsFilenameNotSet.csv";
    fileHandle = -1;
  end
  
  methods
    function obj = NBayesResults(csvResultsFilename)
      %create empty results table
      obj.resultsTable = table(obj.resultsColumnNames);
      obj.outputResultsFilename = csvResultsFilename;
    end
    
    %
    % Open temporary file to write results
    % throws exception if error
    function startGatheringResults(obj)    
       [h, msg] = fopen(obj.outputResultsFilename, 'w');
       if h == -1
           fprintf("Error opening output file %s\n", msg);
           exception = MException("NBayesResults:startGatheringResults", "Error opening output file: %s\n", msg);
           throw(exception);
       end
       % output header
       obj.fileHandle = h;
       fprintf(obj.fileHandle, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", obj.resultsColumnNames(:));
    end % function
    
    function appendResult(obj, distributionName, smootherTypeName, width, ...
                  numberOfHoldOutRun, avgTrainLoss, avgTestLoss, ...
                  avgAccuracy, avgPrecision, avgRecall, avgF1, ...
                  entryCount, elapsedTime)
      if obj.fileHandle == -1
        exception = MException("LetterDecisionTreeResults:appendResult", "Error output file %s is not open", obj.outputResultsTempFilename);
        throw(exception);
      end
      fprintf(obj.fileHandle, "%s\t%s\t%0.08f\t%d\t%0.04f\t%0.04f\t%0.04f\t%0.04f\t%0.04f\t%0.04f\t%d\t%0.04f\n", ...
               distributionName, smootherTypeName, width, ... 
               numberOfHoldOutRun, ...
               avgTrainLoss, avgTestLoss, ...
               avgAccuracy, avgPrecision, avgRecall, avgF1, ...
               entryCount, elapsedTime);
    end % funcion
    
    function endGatheringResults(obj)
      if obj.fileHandle == -1
        exception = MException("NBayesResults:endGatheringResults", ...
          "Error output file %s is not open", obj.outputResultsFilename);
        throw(exception);
      end
      fclose(obj.fileHandle);
      obj.fileHandle = -1;
      obj.resultsTable = readtable(obj.outputResultsFilename, "Delimiter", "\t");
    end % function  
  end
end

