
imdsPath = 'E:\dataset\rice'; 
imds = imageDatastore(imdsPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
inputSize = [224 224 3];  
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

edgeImds = augmentedImageDatastore(inputSize, imds, ...
    'DataAugmentation', imageDataAugmenter('RandXReflection', true));


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

options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 32, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ValidationFrequency', 30);

[trainImds, valImds] = splitEachLabel(imds, 0.8, 'randomized');
trainEdgeImds = augmentedImageDatastore(inputSize, trainImds, ...
    'DataAugmentation', imageDataAugmenter('RandXReflection', true));
valEdgeImds = augmentedImageDatastore(inputSize, valImds);

edgeModel = trainNetwork(trainEdgeImds, edgeLayers, options);

YPred = classify(edgeModel, valEdgeImds);
YTrue = valImds.Labels;
accuracy = sum(YPred == YTrue) / numel(YTrue);
fprintf('Validation Accuracy: %.2f%%\n', accuracy * 100);

