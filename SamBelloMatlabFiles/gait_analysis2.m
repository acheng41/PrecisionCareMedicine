%% Gait Analysis 2
% get spatiotemporal gait parameters using two insoles and classify them
% data from 2 feet
% By: Samuel Bello
% Created: 07/10/24
% Last Updated: 08/06/24

clear % clear the workspace
close all % close all open plots

% "080624\gait_recording_080624_walk.mat";
% "080624\gait_recording_080624_walk2.mat";
% "080624\gait_recording_080624_walk3.mat";
% "080624\gait_recording_080624_walk4.mat";
% "080624\gait_recording_080624_walk5.mat";
% "080624\gait_recording_080624_walk6.mat";
% "080624\gait_recording_080624_walk7.mat";

% legend labels
leg2 = ["Left: 0.6 m/s; Right: 0.6 m/s", ...
    "Left: 0.6 m/s; Right: 1.0 m/s", ...
    "Left: 0.6 m/s; Right: 1.4 m/s", ...
    "Left: 0.6 m/s; Right: 1.8 m/s", ...
    "Left: 1.0 m/s; Right: 0.6 m/s", ...
    "Left: 1.4 m/s; Right: 0.6 m/s", ...
    "Left: 1.8 m/s; Right: 0.6 m/s"];

% place the file names to match above legend
files = ["data\gait_recording_080624_walk.mat";
"data\gait_recording_080624_walk2.mat";
"data\gait_recording_080624_walk4.mat";
"data\gait_recording_080624_walk3.mat";
"080624\gait_recording_080624_walk5.mat";
"080624\gait_recording_080624_walk6.mat"
"080624\gait_recording_080624_walk7.mat"];

numPC = 74; % number of principal components to test
numIter = 5; % number of times to repeat the test
model = "LDA"; % model to use. Options: "LDA", "kNN", "SVM"
numNN = 3; % for the kNN model: how many nearest neighbors to test

%% Get Gait Parameters
for i = 1:length(files) % loop through all the files
    load(files(i)) % load the data from the file
    gaits(i) = get_gait_parameters_insole2_v2(insoleAll_r,insoleAll_l,t_insole_r,t_insole_l); % save the output (a structure of gait parameters) in an array
end

% comment out the line below (add a % at the start of the line) if you want
% to see all the gait parameters ploted
close all
%% Create Feature Matrix (FM):
% FM = [Peak Pressure, Peak Pressure coordinate (X,Y), Center of Pressure coordinate(X,Y), Contact Area, Stance Time, Cycle Duration] for each stance (right, left)
% The Peak Pressure, Peak Pressure coordinate (X,Y), Center of Pressure
% coordinate(X,Y), and Contact Area are only during the stance portion of
% the gait cycle.
t_end = 120; % cutoff time in sec

% find the largest index where the cutoff time occurs
t_end_idx = max([find(gaits(1).t_r <= t_end,1,'last'), find(gaits(1).t_l <= t_end,1,'last')]); % initialize the index to the larger cutoff index between the left and right foot from the first file
for i = 2:length(files) % loop through the rest of the files
    t_end_idx = max([t_end_idx, find(gaits(i).t_r <= t_end,1,'last'), find(gaits(i).t_l <= t_end,1,'last')]); % compare the current cutoff index with the cutoff index of the left and right foot from the current file and keep the largest index
end

% find the longest stance duration before the cutoff time
n_stance = max([max(gaits(1).off_r(gaits(1).off_r < t_end_idx) - gaits(1).strike_r(gaits(1).off_r < t_end_idx)), max(gaits(1).off_l(gaits(1).off_l < t_end_idx) - gaits(1).strike_l(gaits(1).off_l < t_end_idx))]); % initialize the stance duration to the larger stance duration between the left and right foot of the first file
for i = 2:length(files) % loop through the rest of the files
    n_stance = max([n_stance, max(gaits(i).off_r(gaits(i).off_r < t_end_idx) - gaits(i).strike_r(gaits(i).off_r < t_end_idx)), max(gaits(i).off_l(gaits(i).off_l < t_end_idx) - gaits(i).strike_l(gaits(i).off_l < t_end_idx))]); % compare the current stance duration with the stance duration of the left and right foot from the current file and keep the largest stance duration
end

% find the largest number of steps (trials) before the cutoff time
n_steps = max([length(find(gaits(1).off_r < t_end_idx)), length(find(gaits(1).off_l < t_end_idx))]); % initialize the number of steps to the larger number of steps between the left and right foot of the first file
for i = 2:length(files) % loop through the remaining files
    n_steps = max([n_steps, length(find(gaits(i).off_r < t_end_idx)), length(find(gaits(i).off_l < t_end_idx))]); % compare the current number of steps to the number of steps from the left and right foot of the current file and keep the largest number of steps
end

% create feature matrix and labels
featureMatrix = zeros(n_steps * length(files),(n_stance * 6 + 2) * 2); % initialize feature matrix
labels = zeros(n_steps * length(files), 1); % initialize labels
l_start = size(featureMatrix,2) / 2; % index for where to start placing the features from the left foot
for i = 1:length(files) % loop through each file
    file_start = (i - 1) * n_steps; % starting index for the current file in the feature matrix
    for j = 1:n_steps % loop through each step
        labels(file_start + j) = i; % save label for the current step

        try % save as many features from the right insole for the current step
            len = gaits(i).off_r(j) - gaits(i).strike_r(j) + 1; % number of data points in the step
            featureMatrix(file_start + j, 1:len) = gaits(i).pp_r(gaits(i).strike_r(j):gaits(i).off_r(j))'; % peak pressure
            featureMatrix(file_start + j,n_stance+1:n_stance+len) = gaits(i).pp_x_r(gaits(i).strike_r(j):gaits(i).off_r(j))'; % peak pressure x-coordinate
            featureMatrix(file_start + j,2*n_stance+1:2*n_stance+len) = gaits(i).pp_y_r(gaits(i).strike_r(j):gaits(i).off_r(j))'; % peak pressure y-coordinate
            featureMatrix(file_start + j,3*n_stance+1:3*n_stance+len) = gaits(i).cop_x_r(gaits(i).strike_r(j):gaits(i).off_r(j))'; % center of pressure x-coordinate
            featureMatrix(file_start + j,4*n_stance+1:4*n_stance+len) = gaits(i).cop_y_r(gaits(i).strike_r(j):gaits(i).off_r(j))'; % center of pressure y-coordinate
            featureMatrix(file_start + j,5*n_stance+1:5*n_stance+len) = gaits(i).cont_area_r(gaits(i).strike_r(j):gaits(i).off_r(j))'; % contact area
            featureMatrix(file_start + j,6*n_stance+1) = gaits(i).stance_r(j); % stance time
            featureMatrix(file_start + j,6*n_stance+2) = gaits(i).cycle_dur_r(j); % cycle duration
        end

        try % save as many features from the left insole for the current step
            len = gaits(i).off_l(j) - gaits(i).strike_l(j) + 1; % number of data points in the step
            featureMatrix(file_start + j,l_start+1:l_start+len) = gaits(i).pp_l(gaits(i).strike_l(j):gaits(i).off_l(j))'; % peak pressure
            featureMatrix(file_start + j,l_start+n_stance+1:l_start+n_stance+len) = gaits(i).pp_x_l(gaits(i).strike_l(j):gaits(i).off_l(j))'; % peak pressure x-coordinate
            featureMatrix(file_start + j,l_start+2*n_stance+1:l_start+2*n_stance+len) = gaits(i).pp_y_l(gaits(i).strike_l(j):gaits(i).off_l(j))'; % peak pressure y-coordinate
            featureMatrix(file_start + j,l_start+3*n_stance+1:l_start+3*n_stance+len) = gaits(i).cop_x_l(gaits(i).strike_l(j):gaits(i).off_l(j))'; % center of pressure x-coordinate
            featureMatrix(file_start + j,l_start+4*n_stance+1:l_start+4*n_stance+len) = gaits(i).cop_y_l(gaits(i).strike_l(j):gaits(i).off_l(j))'; % center of pressure y-coordinate
            featureMatrix(file_start + j,l_start+5*n_stance+1:l_start+5*n_stance+len) = gaits(i).cont_area_l(gaits(i).strike_l(j):gaits(i).off_l(j))'; % contact area
            featureMatrix(file_start + j,l_start+6*n_stance+1) = gaits(i).stance_l(j); % stance time
            featureMatrix(file_start + j,l_start+6*n_stance+2) = gaits(i).cycle_dur_l(j); % cycle dur
        end
    end
end

%% K-Fold Cross Validation
% k-fold cross validation is a method for testing your data in a uniform way
% way. The data is randomly split up into k fold (groups) and one group is 
% used as the test group. The remaining groups are used to train your model 
% or classifier. Each group takes a turn being the test group.
k = 4; % The number of folds
avgAcc = zeros(1,numPC); % initialize an array to store the average accuracy for each principal component (PC)
stdAcc = zeros(1,numPC); % initialize an array to store the standard deviation of the accuracy for each principal component (PC)
for i = 1:numPC % loop through the number of principal components
    testAcc = zeros(1,numIter); % initialize an array to store the accuracy for each iteration
    for j = 1:numIter % repeat the k-fold cross validation for numIter times
        cvFolds = crossvalind('Kfold',labels,k); % This splits the data into the k random folds
        cp = classperf(labels); % This tracks the classification results
        for l = 1:k % for each fold
            testIdx = (cvFolds == l); % indices of test instances
            trainIdx = ~testIdx; % indices training instances

            testData = featureMatrix(testIdx,:); % separate the test trials
            testLabels = labels(testIdx); % separate the test labels
            trainData = featureMatrix(trainIdx,:); % separate the training trials
            trainLabels = labels(trainIdx); % separate the training labels

            % perform principal component analysis (PCA)
            [coeffTrain, scoreTrain, latentTrain,~,~,mu] = pca(trainData,'Centered',true); % PCA sorts the features based on the amount of variability. This arranges the features such that the features that tell us the most information are used first

            % train and test model over training instances
            scoreTest = (testData - mu)*coeffTrain; % apply the pca transformation to the testing data
            switch model
                case "LDA"
                    % linear discriminant analysis (LDA) separates the data
                    % based on their labels using discriminant functions
                    LDA_mdl = fitcdiscr(scoreTrain(:,1:i),trainLabels,'discrimType','pseudoLinear'); %  train using LDA
                    output = predict(LDA_mdl,scoreTest(:,1:i)); % test using the LDA model

                case "SVM"
                    % support vector machines (SVM) separates the data by
                    % finding the optimal line or plane that maximizes the
                    % distance between each class
                    SVM_mdl = fitcecoc(scoreTrain(:,1:i),trainLabels); % train using SVM
                    output = predict(SVM_mdl,scoreTest(:,1:i)); % test using SVM

                case "kNN"
                    % k-nearest neighbors (kNN) is a classification method
                    % where each data point is assiged a label based on the
                    % label of the k-nearest points
                    kNN_mdl = fitcknn(scoreTrain(:,1:i),trainLabels,'NumNeighbors',numNN); % train using kNN
                    output = predict(kNN_mdl,scoreTest(:,1:i)); % test using kNN
            end

            % evaluate and update performance object
            cp = classperf(cp, output, testIdx);
        end
        % Save the classification accuracy
        testAcc(j) = cp.CorrectRate;
    end
    avgAcc(i) = 100 * mean(testAcc); % save the average accuracy for the current principal component (PC)
    stdAcc(i) = 100 * std(testAcc); % save the standard deviation of the accuracy for the current principal component (PC)
end

% Plot number of PCs vs Accuracy
fig1 = figure();
errorbar(1:numPC,avgAcc,stdAcc)
title(strcat("K-Fold Classification Accuracy with ", model))
xlabel("Number of PCs")
ylabel("Accuracy (%)")
ylim([0 100])

%% Plot in Model Space
% This section plots the data in LDA and PCA space so you can visualize how
% well they are separated or clustered
numGroups = length(unique(labels));
clr = hsv(numGroups);
numDiscriminants = length(unique(labels))-1;

% get discriminative functions for test and training data
SB = LDA_mdl.BetweenSigma;
SW = LDA_mdl.Sigma;

[eigVectors, eigValues] = eig(SB, SW);
eigValues = diag(eigValues);

sortedValues = sort(eigValues,'descend');
[c, ind] = sort(eigValues,'descend'); %store indices
sortedVectors = eigVectors(:,ind); % reorder columns

vectors = sortedVectors;%(:,1:numDiscriminants);
transformedTrainSamples = scoreTrain(:,1:numPC)*vectors;
transformedTestSamples = scoreTest(:,1:numPC)*vectors;

fig2 = figure();
for i = 1:numGroups
    group = find(trainLabels == i);
    plot3(transformedTrainSamples(group,1),transformedTrainSamples(group,2),transformedTrainSamples(group,3),'.','MarkerSize',10);
    % scatter3(transformedTrainSamples(:,1),transformedTrainSamples(:,2),transformedTrainSamples(:,3),15,labels,'filled');
    hold on
    leg(i) = strcat("file",int2str(i));
end
hold off
colormap(clr);
title(strcat("Training Data in ", model, " Space"))
xlabel("DF 1")
ylabel("DF 2")
zlabel("DF 3")
legend(leg2)

fig3 = figure();
for i = 1:numGroups
    group = find(testLabels == i);
    plot3(transformedTestSamples(group,1),transformedTestSamples(group,2),transformedTestSamples(group,3),'.','MarkerSize',10);
    % scatter3(transformedTestSamples(:,1),transformedTestSamples(:,2),transformedTestSamples(:,3),15,testLabels,'filled');
    hold on
    leg(i) = strcat("file",int2str(i));
end
hold off
colormap(clr);
title(strcat("Testing Data in ", model, " Space"))
xlabel("DF 1")
ylabel("DF 2")
zlabel("DF 3")
legend(leg2)

fig4 = figure();
for i = 1:numGroups
    group = find(trainLabels == i);
    plot3(scoreTrain(group,1),scoreTrain(group,2),scoreTrain(group,3),'.','MarkerSize',10);
    % scatter3(scoreTrain(:,1),scoreTrain(:,2),scoreTrain(:,3),15,labels,'filled');
    hold on
    leg(i) = strcat("file",int2str(i));
end
hold off
colormap(clr);
title("Training Data in PCA Space")
xlabel("PC 1")
ylabel("PC 2")
zlabel("PC 3")
legend(leg2)

fig5 = figure();
for i = 1:numGroups
    group = find(testLabels == i);
    plot3(scoreTest(group,1),scoreTest(group,2),scoreTest(group,3),'.','MarkerSize',10);
    % scatter3(scoreTest(:,1),scoreTest(:,2),scoreTest(:,3),15,labels,'filled');
    hold on
    leg(i) = strcat("file",int2str(i));
end
hold off
colormap(clr);
title("Testing Data in PCA Space")
xlabel("PC 1")
ylabel("PC 2")
zlabel("PC 3")
legend(leg2)

% get discriminative functions for all data
[coeffTrain, scoreTrain, latentTrain,~,~,mu] = pca(featureMatrix,'Centered',true);
LDA_mdl = fitcdiscr(scoreTrain(:,1:numPC),labels,'discrimType','pseudoLinear'); %  train

SB = LDA_mdl.BetweenSigma;
SW = LDA_mdl.Sigma;

[eigVectors, eigValues] = eig(SB, SW);
eigValues = diag(eigValues);

sortedValues = sort(eigValues,'descend');
[c, ind] = sort(eigValues,'descend'); %store indices
sortedVectors = eigVectors(:,ind); % reorder columns

vectors = sortedVectors;%(:,1:numDiscriminants);
transformedSamples = scoreTrain(:,1:numPC)*vectors;

fig6 = figure();
for i = 1:numGroups
    group = find(labels == i);
    plot3(transformedSamples(group,1),transformedSamples(group,2),transformedSamples(group,3),'.','MarkerSize',10);
    % scatter3(transformedSamples(:,1),transformedSamples(:,2),transformedSamples(:,3),15,labels,'filled');
    hold on
    leg(i) = strcat("file",int2str(i));
end
hold off
colormap(clr);
title(strcat("All Data in ", model, " Space"))
xlabel("DF 1")
ylabel("DF 2")
zlabel("DF 3")
legend(leg)

fig7 = figure();
for i = 1:numGroups
    group = find(labels == i);
    plot3(scoreTrain(group,1),scoreTrain(group,2),scoreTrain(group,3),'.','MarkerSize',10);
    % scatter3(scoreTrain(:,1),scoreTrain(:,2),scoreTrain(:,3),15,labels,'filled');
    hold on
    leg(i) = strcat("file",int2str(i));
end
hold off
colormap(clr);
title("All Data in PCA Space")
xlabel("PC 1")
ylabel("PC 2")
zlabel("PC 3")
legend(leg2)


%% Confusion Matrix
fig8 = figure();
cm = [cp.CountingMatrix(7,7), cp.CountingMatrix(7,6), cp.CountingMatrix(7,5), cp.CountingMatrix(7,1), cp.CountingMatrix(7,2), cp.CountingMatrix(7,3), cp.CountingMatrix(7,4);
    cp.CountingMatrix(6,7), cp.CountingMatrix(6,6), cp.CountingMatrix(6,5), cp.CountingMatrix(6,1), cp.CountingMatrix(6,2), cp.CountingMatrix(6,3), cp.CountingMatrix(6,4);
    cp.CountingMatrix(5,7), cp.CountingMatrix(5,6), cp.CountingMatrix(5,5), cp.CountingMatrix(5,1), cp.CountingMatrix(5,2), cp.CountingMatrix(5,3), cp.CountingMatrix(5,4);
    cp.CountingMatrix(1,7), cp.CountingMatrix(1,6), cp.CountingMatrix(1,5), cp.CountingMatrix(1,1), cp.CountingMatrix(1,2), cp.CountingMatrix(1,3), cp.CountingMatrix(1,4);
    cp.CountingMatrix(2,7), cp.CountingMatrix(2,6), cp.CountingMatrix(2,5), cp.CountingMatrix(2,1), cp.CountingMatrix(2,2), cp.CountingMatrix(2,3), cp.CountingMatrix(2,4);
    cp.CountingMatrix(3,7), cp.CountingMatrix(3,6), cp.CountingMatrix(3,5), cp.CountingMatrix(3,1), cp.CountingMatrix(3,2), cp.CountingMatrix(3,3), cp.CountingMatrix(3,4);
    cp.CountingMatrix(4,7), cp.CountingMatrix(4,6), cp.CountingMatrix(4,5), cp.CountingMatrix(4,1), cp.CountingMatrix(4,2), cp.CountingMatrix(4,3), cp.CountingMatrix(4,4)];

leg3 = ["Left: 1.8 m/s; Right: 0.6 m/s", ...
    "Left: 1.4 m/s; Right: 0.6 m/s", ...
    "Left: 1.0 m/s; Right: 0.6 m/s", ...
    "Left: 0.6 m/s; Right: 0.6 m/s", ...
    "Left: 0.6 m/s; Right: 1.0 m/s", ...
    "Left: 0.6 m/s; Right: 1.4 m/s", ...
    "Left: 0.6 m/s; Right: 1.8 m/s"];

confusionchart(cp.CountingMatrix(1:end-1,:),leg2,'Normalization','row-normalized')
title("Confusion Matrix")