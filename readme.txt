INM431 Machine Learning Coursework readme
Jacques Leplat 2020.

_ Developed using MATLAB 2020b, final models have been tested 
  on MATLAB 2020a.
- All folders and subfolders should be added to the MATLAB path; starting 
  from the folder containing this readme.txt file.
- To run the final model comparison, Code/FinalModelComparison.mlx should be
  run: loads the models, and performs "predict" for each model, displays a 
  confusion matrix for each method's final model. Displays the final results.

Folder descriptions:
====================
Dataset: contains the "Letter Image Recognition Data" downloaded from:
        http://archive.ics.uci.edu/ml/datasets/Letter+Recognition.
Code: contains the MATLAB code that runs the project. 
      - FinalModelComparison.mlx is the live script written for this submission.
      - RunAllScripts.mlx is used to test the code by running all the scripts:
        takes a long time to run.
      - Dataset: 
        - LetterDatasetClass.m: contains the code that wraps the dataset.
        - CreateLetterDatasetInstances.m: is used to create 3 workspace classes
          that have been saved manually. 
        - DatasetAnalysis.mlx: contains code to report on the dataset: 
          distribution of classes, correlation, normalization, feature selection.
     - DecisionTrees: all the code for the decision tree machine learning method.
        - DTreeHyperparametersClass.m: used to hold the hyperparameter search
          configurations.
        - LetterDecisionTreeClass.m: code to grow trees, perform hyperparameter 
          search (saves results in csv files), grow the final tree model.
        - LetterDecisionTreeResults.m: loads the results csv file, holds the 
          results in a table, has methods to plot information of interest.
        - TestLetterDecisionTree.mlx: run of the decision tree 
          hyperparameter using several configurations, plots of interest, 
          build a final (best) model.
     - MatlabObjects: the workspace  variables created and saved manually.
        - letterDatasetClass.mat: 3 worskapce variables for the dataset class:
             - letterDatasetNormalised
             - letterDatasetNormalisedReducedFeatures
             - letterDatasetNotNormalised
        - nBayesModel.mat: final Naive Bayes MATLAB model
        - treeBaggerModel.mat: final Random Forest model
        - treedModel.mat: final decision tree model.
     - NaiveBayes: all the code written for the Naive Bayes method.
        - NBayesClass.m: class that wraps all functionality to fit a model, 
            run hyperparameter searches (saves results in csv files), 
            and to build the final model. Several static instance creators 
            are present.
        - NBayesHyperparameterClass.m: this class is used to hold the 
            hyperparameter values to use during a search. Several static 
            instance creators are present.
        - NBayesResults.m: class used to save and load the results to/from 
            a csv file.
        - testNBayes.mlx: live script that performs MATLAB's automatic 
            hyperparameter search.
        - testNBayes2.mlx: live script that uses NBayesClass to perform several
            hyperparameter searches, and display the results, build the final model.
     - RandomForest: all the code written for the Random Forest ensemble method.
        - RandomForestClass.m: class that wraps all functionality to fit a model, 
            run hyperparameter searches (saves results in csv files), 
            and to build the final model. A static method to create the final 
            model is present
        - RFHyperparameters.m: this class is used to hold the 
            hyperparameter values to use during a search. Two static 
            instance creators are present.
        - RFResults.m: class used to save and load the results to/from a csv file.
        - TestRandomForest.mlx: live script that uses RandomForestClass.m to 
            perform several hyperparameter searches, and display the results, build the final model.
thirdPartyCode: code found on the internet that is used in this project.
     - heatmap2.m: Ameya Deoras (2020). Customizable Heat Maps 
       (https://www.mathworks.com/matlabcentral/fileexchange/24253-customizable-heat-maps), 
        MATLAB Central File Exchange. Retrieved November 2, 2020.          
Utilities: classes which contain functionality that is used several times.
     - CalcUtil.m: has a static method to calculate performance petrics: accuracy,
        precision, recall, F1.
     - PlotUtil.m: contains methods that generate plots of results.
Results: csv files of results generated in this project.        

