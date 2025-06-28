newImage = imread("E:\dataset\Apple\test\Apple Black Rot\0bb8fb61-d561-43fd-89a3-d24d253adef9___JR_FrgE.S 2849_270deg.JPG");
resizedImage = imresize(newImage, [224 224]);
predictedLabel = classify(trainedNet, resizedImage);
fprintf("Predicted Label: %s\n", char(predictedLabel));

imshow(newImage);
title("Predicted Label: " + string(predictedLabel));

