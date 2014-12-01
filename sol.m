clear all;

load labeled_images

% for i=1:size(tr_images, 3)
%     tr_images(:, :, i) = imgradient(tr_images(:,:,i));
% end

trainSet = tr_images(:,:,1:size(tr_images, 3));
trainLabel = tr_labels(:,1);
% Seperate the training data into different sets corresponding to each
% emotion.

[imSetAnger, imSetDisgust, imSetFear, imSetHappy, imSetSad, imSetSurprise, imSetNeutral] = separate(trainSet, trainLabel);

imSetAnger = double(reshape(imSetAnger, 1024, size(imSetAnger, 3)));
imSetDisgust = double(reshape(imSetDisgust, 1024, size(imSetDisgust, 3)));
imSetFear = double(reshape(imSetFear, 1024, size(imSetFear, 3)));
imSetHappy = double(reshape(imSetHappy, 1024, size(imSetHappy, 3)));
imSetSad = double(reshape(imSetSad, 1024, size(imSetSad, 3)));
imSetSurprise = double(reshape(imSetSurprise, 1024, size(imSetSurprise, 3)));
imSetNeutral = double(reshape(imSetNeutral, 1024, size(imSetNeutral, 3)));

% imSetAnger = imSetAnger(:,1:200);
% imSetDisgust = imSetDisgust(:,1:200);
% imSetFear = imSetFear(:,1:200);
% imSetHappy = imSetHappy(:,1:200);
% imSetSad = imSetSad(:,1:200);
% imSetSurprise = imSetSurprise(:,1:200);
% imSetNeutral = imSetNeutral(:,1:200);



% Num PCA
vecs = 500;

% Use PCA
[baseAnger, meanAnger, imSetAnger] = pcaimg(imSetAnger, vecs);
[baseDisgust, meanDisgust, imSetDisgust] = pcaimg(imSetDisgust, vecs);
[baseFear, meanFear, imSetFear] = pcaimg(imSetFear, vecs);
[baseHappy, meanHappy, imSetHappy] = pcaimg(imSetHappy, vecs);
[baseSad, meanSad, imSetSad] = pcaimg(imSetSad, vecs);
[baseSurprise, meanSurprise, imSetSurprise] = pcaimg(imSetSurprise, vecs);
[baseNeutral, meanNeutral, imSetNeutral] = pcaimg(imSetNeutral, vecs);


% Set Components
components = 3;
iterations = 10;

% Perform MoG on each set
[pAnger, muAnger, varyAnger, logProbAnger] = mogEM(imSetAnger, components, iterations, 0.01, 0);
[pDisgust, muDisgust, varyDisgust, logProbDisgust] = mogEM(imSetDisgust, components, iterations, 0.01, 0);
[pFear, muFear, varyFear, logProbFear] = mogEM(imSetFear, components, iterations, 0.01, 0);
[pHappy, muHappy, varyHappy, logProbHappy] = mogEM(imSetHappy, components, iterations, 0.01, 0);
[pSad, muSad, varySad, logProbSad] = mogEM(imSetSad, components, iterations, 0.01, 0);
[pSurprise, muSurprise, varySurprise, logProbSurprise] = mogEM(imSetSurprise, components, iterations, 0.01, 0);
[pNeutral, muNeutral, varyNeutral, logProbNeutral] = mogEM(imSetNeutral, components, iterations, 0.01, 0);

% Reshape 'validation' set
inputs_valid = double(reshape(tr_images(:,:,size(tr_images, 3)-499:size(tr_images, 3)), 1024, 500));
[baseValid, meanValid, inputs_valid] = pcaimg(inputs_valid, vecs);


predictions = zeros(0, 1);
% Perform Assignments

lProbAnger = mogLogProb(pAnger, muAnger, varyAnger, inputs_valid);
lProbDisgust = mogLogProb(pDisgust, muDisgust, varyDisgust, inputs_valid);
lProbFear = mogLogProb(pFear, muFear, varyFear, inputs_valid);
lProbHappy = mogLogProb(pHappy, muHappy, varyHappy, inputs_valid);
lProbSad = mogLogProb(pSad, muSad, varySad, inputs_valid);
lProbSurprise = mogLogProb(pSurprise, muSurprise, varySurprise, inputs_valid);
lProbNeutral = mogLogProb(pNeutral, muNeutral, varyNeutral, inputs_valid);

target = zeros(500, 1);

for i=1:size(target, 1)
    clear max;
    clear index;
    toCheck = [lProbAnger(1, i), lProbDisgust(1, i), lProbFear(1, i), lProbHappy(1, i), lProbSad(1, i), lProbSurprise(1, i), lProbNeutral(1, i)];
    [max, index] = max(toCheck);
    target(i, 1) = index;
end

realTarget = tr_labels((size(tr_labels, 1)-499:size(tr_labels, 1)), 1)

result = (target == realTarget);
correctlyClassified = sum(result);
fprintf('Correctly classified = %f\n', correctlyClassified/size(target, 1));

