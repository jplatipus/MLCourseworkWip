classdef RFResults < handle
  %RFRESULTS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    resultsTable = {};
    resultsColumnNames = ["feature", "tree", "folds", "randomTreebagSeed", ...
      "avgTrainLoss", "avgTestLoss", "avgOobLoss", ...
      "avgAccuracy", "avgPrecision", "avgRecall", "avgF1", ...
      "entryCount", "elapsedTime"]; 
    outputResultsFilename = "nbayesResultsFilenameNotSet.csv";
    fileHandle = -1;
  end
  
  methods
    
    % Constructor: takes csv file name to create for results
    function obj = RFResults(csvResultsFilename)
      %create empty results table
      obj.resultsTable = table(obj.resultsColumnNames);
      obj.outputResultsFilename = csvResultsFilename;
    end
    
    %
    % Open file to write results
    % throws exception if error
    function startGatheringResults(obj)    
       [h, msg] = fopen(obj.outputResultsFilename, 'w');
       if h == -1
           fprintf("Error opening output file %s\n", msg);
           exception = MException("RFResults:startGatheringResults", "Error opening output file: %s\n", msg);
           throw(exception);
       end
       % output header
       obj.fileHandle = h;
       fprintf(obj.fileHandle, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", obj.resultsColumnNames(:));
    end % function
    
    %
    % append a result row to the csv file
    %
    function appendResult(obj, numFolds, numTree, numFeature, randomTreeBagSeed, ...
        errTrain, errValid, errOob,...
        accuracies, precisions, recalls, f1s, ...
              numEntries, elapsedTime)
     if obj.fileHandle == -1
        exception = MException("RFResults:appendResult", "Error output file %s is not open", obj.outputResultsTempFilename);
        throw(exception);
      end
      fprintf(obj.fileHandle, ...
              "%d\t%d\t%d\t%d\t%0.04f\t%0.04f\t%0.04f"+...
              "\t%0.04f\t%0.04f\t%0.04f\t%0.04f"+ ...
              "\t%d\t%0.04f\n", ...
              numFeature, numTree, numFolds, randomTreeBagSeed, errTrain, errValid, errOob, ...
              accuracies, precisions, recalls, f1s, ...
              numEntries, elapsedTime);      
    end

    %
    % close csv file, read in csv file to resultsTable class member
    function endGatheringResults(obj)
      if obj.fileHandle == -1
        exception = MException("RFResults:endGatheringResults", ...
          "Error output file %s is not open", obj.outputResultsFilename);
        throw(exception);
      end
      fclose(obj.fileHandle);
      obj.fileHandle = -1;
      obj.resultsTable = readtable(obj.outputResultsFilename, "Delimiter", "\t");
    end % function 
    
  end % methods
  
  methods(Static)
    function instance = getInstanceFromCsvResults(csvFilename)
      instance = RFResults(csvFilename);
      instance.resultsTable = readtable(csvFilename, "Delimiter", "\t");  
    end  % function  
    
  end % methods(Static)
end

