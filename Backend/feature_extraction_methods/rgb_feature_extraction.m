% Define paths for datasets
imdsPath = 'E:\dataset\rice';

% Create original image datastore
imds = imageDatastore(imdsPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Split data into training and validation
[trainImds, valImds] = splitEachLabel(imds, 0.8, 'randomized');

% Resize parameters
inputSize = [224 224];

% Augment training and validation images
augmentedTrainImds = augmentedImageDatastore(inputSize, trainImds);
augmentedValImds = augmentedImageDatastore(inputSize, valImds);

% Define CNN layers for RGB model
layers = [imageInputLayer([224 224 3])
          convolution2dLayer(3, 8, 'Padding', 'same')
          batchNormalizationLayer
          reluLayer
          maxPooling2dLayer(2, 'Stride', 2)
          fullyConnectedLayer(numel(categories(imds.Labels)))
          softmaxLayer
          classificationLayer];

% Define training options
options = trainingOptions('sgdm', 'MaxEpochs', 10, 'InitialLearnRate', 0.001, ...
                           'Verbose', false, 'Plots', 'training-progress', 'ValidationData', augmentedValImds);

% Train RGB model
rgbModel = trainNetwork(augmentedTrainImds, layers, options);

% Evaluate RGB model
rgbYPred = classify(rgbModel, augmentedValImds);
rgbYTrue = valImds.Labels;
rgbCM = confusionmat(rgbYTrue, rgbYPred);
figure;
confusionchart(rgbCM);
title('Confusion Matrix for RGB Model');


