% Load Pretrained ResNet-50
net = resnet50;
lgraph = layerGraph(net);

% Modify Fully Connected Layer for New Classes
numClasses = 4; % Change this based on your dataset for wheat it is 4
newFC = fullyConnectedLayer(numClasses, 'Name', 'fc_new');
newSoftmax = softmaxLayer('Name', 'softmax_new');
newClassOutput = classificationLayer('Name', 'output');

% Replace the last layers
lgraph = replaceLayer(lgraph, 'fc1000', newFC);
lgraph = replaceLayer(lgraph, 'fc1000_softmax', newSoftmax);
lgraph = replaceLayer(lgraph, 'ClassificationLayer_fc1000', newClassOutput);

% Load Training & Validation Data
imageSize = [224 224 3];
datastoreTrain = imageDatastore("E:\dataset\wheat\train", 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
datastoreVal = imageDatastore("E:\dataset\wheat\validate", 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Data Augmentation to Improve Accuracy
augmenter = imageDataAugmenter('RandRotation', [-30 30], 'RandXReflection', true, 'RandYReflection', true);
augmentedTrain = augmentedImageDatastore(imageSize, datastoreTrain, 'DataAugmentation', augmenter);
augmentedVal = augmentedImageDatastore(imageSize, datastoreVal);

% Training Options (Use GPU if available)
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...
    'MaxEpochs', 20, ...
    'InitialLearnRate', 1e-4, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', augmentedVal, ...
    'ValidationFrequency', 5, ...
    'ExecutionEnvironment', 'gpu', ... % 'gpu' or 'cpu' based on availability
    'Verbose', true, ...
    'Plots', 'training-progress');

% Train the Network
[trainedNet, trainInfo] = trainNetwork(augmentedTrain, lgraph, options);

% Save the Model
save('wheattrained', 'trainedNet');

% Evaluate Accuracy on Validation Set
predictedLabels = classify(trainedNet, augmentedVal, 'ExecutionEnvironment', 'gpu');

trueLabels = datastoreVal.Labels;
accuracy = sum(predictedLabels == trueLabels) / numel(trueLabels) * 100;
disp("Validation Accuracy: " + accuracy + "%");

% Plot Confusion Matrix
figure;
confusionchart(trueLabels, predictedLabels);
title('Confusion Matrix');