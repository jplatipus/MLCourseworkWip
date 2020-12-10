classdef NBayesResults < handle  
  % Class used to write results to a csv file, load the results csv into an
  % instance of this class, and functionality to append a row to the
  % results
  
  properties
    resultsTable = {};
    resultsColumnNames = [
      "distributionName", "smootherType", "Width", "numberOfFolds", ...
      "randomNBayesSeed", "avgTrainLoss", ...
      "avgTestLoss", ...
      "avgAccuracy", "avgPrecision", "avgRecall", "avgF1", ...
      "entryCount", "elapsedTime", "avgPredictTime"];
        % temporary file to store the results:
    outputResultsFilename = "nbayesResultsFilenameNotSet.csv";
    fileHandle = -1;
  end
  
  methods
    %%
    % Constructor
    %    
    function obj = NBayesResults(csvResultsFilename)
      %create empty results table
      obj.resultsTable = table(obj.resultsColumnNames);
      obj.outputResultsFilename = csvResultsFilename;
    end
    
    %%
    % Open file to write results
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
       fprintf(obj.fileHandle, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", obj.resultsColumnNames(:));
    end % function
    
    %%
    % output a result row to the file
    function appendResult(obj, distributionName, smootherTypeName, width, ...
                  numberOfFolds, randomSeed, avgTrainLoss, avgTestLoss, ...
                  avgAccuracy, avgPrecision, avgRecall, avgF1, ...
                  entryCount, elapsedTime, avgPredictTime)
      if obj.fileHandle == -1
        exception = MException("NBayesResults:appendResult", "Error output file %s is not open", obj.outputResultsTempFilename);
        throw(exception);
      end
      fprintf(obj.fileHandle, "%s\t%s\t%0.08f"+ ...
          "\t%d\t%d"+ ...
          "\t%0.04f\t%0.04f"+ ...
          "\t%0.04f\t%0.04f\t%0.04f\t%0.04f"+ ...
          "\t%d\t%0.04f\t%0.04f\n", ...
               distributionName, smootherTypeName, width, ... 
               numberOfFolds, randomSeed, ...
               avgTrainLoss, avgTestLoss, ...
               avgAccuracy, avgPrecision, avgRecall, avgF1, ...
               entryCount, elapsedTime, avgPredictTime);
    end % funcion
    
    %%
    % close the csv file, load the contents into the resultsTable member
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
  end % methods
  
  methods(Static)
    %%
    % loads the csv file into an instance of this class
    function instance = getInstanceFromCsvResults(csvFilename)
      instance = NBayesResults(csvFilename);
      instance.resultsTable = readtable(csvFilename, "Delimiter", "\t");  
    end  % function  
    
  end % methods(Static)
end

