% Specify the dataset path
imdsPath = 'E:\dataset\rice';  % Change this to your dataset path
imds = imageDatastore(imdsPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Resize images to match model input size
inputSize = [224 224 3];  % Grayscale input

% Define Edge Detection Function
function edgeImg = edgeDetector(img, inputSize)
    % Convert to grayscale if needed
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    % Apply edge detection (Canny)
    edgeImg = edge(img, 'Canny');
    % Resize to model input size
    edgeImg = imresize(edgeImg, inputSize(1:2));
end

% Create Edge Detection Datastore
edgeImds = augmentedImageDatastore(inputSize, imds, ...
    'DataAugmentation', imageDataAugmenter('RandXReflection', true));

% Apply edgeDetector to all images
outputFolder = 'Edge_Detection_Images';
if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

for i = 1:numel(imds.Files)
    img = readimage(imds, i);
    edgeImg = edgeDetector(img, inputSize);
    % Write edge-detected images to a folder (Optional for inspection)
    imwrite(edgeImg, fullfile(outputFolder, ['img' num2str(i) '.png']));
end

% Define CNN Layers for Edge Detection
edgeLayers = [
    imageInputLayer(inputSize)  
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(numel(categories(imds.Labels)))
    softmaxLayer
    classificationLayer
];

% Set Training Options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ValidationFrequency', 30);

% Split Data for Training and Validation
[trainImds, valImds] = splitEachLabel(imds, 0.8, 'randomized');
trainEdgeImds = augmentedImageDatastore(inputSize, trainImds, ...
    'DataAugmentation', imageDataAugmenter('RandXReflection', true));
valEdgeImds = augmentedImageDatastore(inputSize, valImds);

% Train the Edge Detection Model
edgeModel = trainNetwork(trainEdgeImds, edgeLayers, options);

% Evaluate the model
YPred = classify(edgeModel, valEdgeImds);
YTrue = valImds.Labels;
accuracy = sum(YPred == YTrue) / numel(YTrue);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

% Plot Confusion Matrix
confusionchart(YTrue, YPred);
