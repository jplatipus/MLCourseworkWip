classdef NBayesClass < handle
  %NBAYESCLASS Summary of this class goes here
  %   Detailed explanation goes here
  
  properties
    dataset;
    distNames = {'kernel','mvmn','mvmn','mvmn','mvmn','kernel','mvmn','mvmn','mvmn',...
    'kernel','mvmn','kernel','kernel','kernel','kernel','mvmn'};
  end
  
  methods
    function obj = NBayesClass(letterDataset)
      obj.dataset = letterDataset;
    end
    
    function nBayesModel = simpleNaiveBayesClassifier(obj)
      [x, y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
      prior = [0.3 0.7];
      width = 90;
      [xt, yt] = obj.dataset.extractXYFromTable(obj.dataset.testTable);
      nBayesModel = fitcnb(x, y, 'ClassNames', categorical(table2array(obj.dataset.validClassValues)));
      trainingLoss = loss(nBayesModel, x, y);
      testLoss = loss(nBayesModel, xt, yt);
      fprintf("Training loss: %0.02f Test loss: %0.02f\n", trainingLoss, testLoss);
    end
    
    function plotGaussianContrours(nBayesModel)
    [x, y] = obj.dataset.extractXYFromTable(obj.dataset.trainTable);
    figure
    gscatter(X(:,1),X(:,2),Y);
    h = gca;
    cxlim = h.XLim;
    cylim = h.YLim;
    hold on
    Params = cell2mat(nBayesModel.DistributionParameters); 
    Mu = Params(2*(1:3)-1,1:2); % Extract the means
    Sigma = zeros(2,2,3);
    for j = 1:3
        Sigma(:,:,j) = diag(Params(2*j,:)).^2; % Create diagonal covariance matrix
        xlim = Mu(j,1) + 4*[-1 1]*sqrt(Sigma(1,1,j));
        ylim = Mu(j,2) + 4*[-1 1]*sqrt(Sigma(2,2,j));
        f = @(x,y) arrayfun(@(x0,y0) mvnpdf([x0 y0],Mu(j,:),Sigma(:,:,j)),x,y);
        fcontour(f,[xlim ylim]) % Draw contours for the multivariate normal distributions 
    end
    h.XLim = cxlim;
    h.YLim = cylim;
    title('Naive Bayes Classifier -- Fisher''s Iris Data')
    xlabel('Petal Length (cm)')
    ylabel('Petal Width (cm)')
    legend('setosa','versicolor','virginica')
    hold off
    end
  end
end

